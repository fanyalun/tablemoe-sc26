import time

import torch
from transformers.generation.streamers import BaseStreamer


class _BaseTimingStreamer(BaseStreamer):
    def __init__(self, profiler=None, synchronize_cuda=True):
        super().__init__()
        self.start_prefill_time = None
        self.start_decode_time = None
        self.end_decode_time = None
        self.num_new_tokens = 0
        self.profiler = profiler
        self.synchronize_cuda = bool(synchronize_cuda)

    def _on_decode_start(self):
        return None

    def _count_tokens(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0
            return int(value.numel())
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 1

    def put(self, value):
        if torch.cuda.is_available() and self.synchronize_cuda:
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            if self.profiler is not None:
                self.profiler.add_duration(
                    "streamer.cuda_synchronize",
                    time.perf_counter() - sync_start,
                )

        now = time.perf_counter()
        if self.profiler is not None:
            self.profiler.increment_counter("streamer.put_calls")
        if self.start_prefill_time is None:
            self.start_prefill_time = now
            return

        if self.start_decode_time is None:
            self.start_decode_time = now
            self._on_decode_start()

        token_count = self._count_tokens(value)
        if token_count <= 0:
            return

        self.end_decode_time = now
        self.num_new_tokens += token_count

    def end(self):
        if self.end_decode_time is None and self.start_decode_time is not None:
            self.end_decode_time = time.perf_counter()

    def get_metrics(self):
        if self.start_prefill_time is None or self.start_decode_time is None or self.num_new_tokens == 0:
            return 0.0, 0.0, 0

        ttft = self.start_decode_time - self.start_prefill_time
        if self.num_new_tokens > 1 and self.end_decode_time is not None:
            tpot = (self.end_decode_time - self.start_decode_time) / (self.num_new_tokens - 1)
        else:
            tpot = 0.0
        return ttft, tpot, self.num_new_tokens


class TimingStreamer(_BaseTimingStreamer):
    pass


class StopWatch(_BaseTimingStreamer):
    def __init__(self, engine, profiler=None, synchronize_cuda=True):
        super().__init__(
            profiler=profiler,
            synchronize_cuda=synchronize_cuda,
        )
        self.engine = engine

    def _on_decode_start(self):
        self.engine.expert_dispatcher.clear_expert_cache_counts()
