import copy
import time
from contextlib import contextmanager

import torch


class PerfProfileRecorder:
    def __init__(self, enabled=False, sample_id=None):
        self.enabled = bool(enabled)
        self.sample_id = None if sample_id is None else str(sample_id)
        self.current = None
        self.results = []

    def should_profile(self, sample_id):
        if not self.enabled:
            return False
        if self.sample_id is None:
            return True
        return str(sample_id) == self.sample_id

    def begin_sample(self, sample_id, metadata=None):
        self.current = None
        if not self.should_profile(sample_id):
            return False

        self.current = {
            "id": str(sample_id),
            "metadata": dict(metadata or {}),
            "timings": {},
            "values": {},
            "counters": {},
        }
        return True

    def is_active(self):
        return self.current is not None

    def add_metadata(self, key, value):
        if not self.current:
            return
        self.current["metadata"][key] = value

    def set_value(self, key, value):
        if not self.current:
            return
        self.current["values"][key] = value

    def increment_counter(self, key, amount=1, layer_id=None):
        if not self.current:
            return

        counters = self.current["counters"]
        counters[key] = counters.get(key, 0) + amount
        if layer_id is None:
            return

        key_with_layer = f"{key}.layer.{layer_id}"
        counters[key_with_layer] = counters.get(key_with_layer, 0) + amount

    def add_duration(self, key, seconds, layer_id=None):
        if not self.current:
            return

        entry = self.current["timings"].setdefault(
            key,
            {"total_ms": 0.0, "count": 0},
        )
        entry["total_ms"] += float(seconds) * 1000.0
        entry["count"] += 1

        if layer_id is None:
            return

        layers = entry.setdefault("layers", {})
        layer_key = str(layer_id)
        layer_entry = layers.setdefault(
            layer_key,
            {"total_ms": 0.0, "count": 0},
        )
        layer_entry["total_ms"] += float(seconds) * 1000.0
        layer_entry["count"] += 1

    @contextmanager
    def measure(self, key, layer_id=None):
        if not self.current:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.add_duration(
                key,
                time.perf_counter() - start_time,
                layer_id=layer_id,
            )

    @contextmanager
    def measure_cuda(self, key, layer_id=None, device=None):
        if not self.current:
            yield
            return

        if device is not None:
            device = torch.device(device)

        use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
        if not use_cuda:
            with self.measure(key, layer_id=layer_id):
                yield
            return

        stream = torch.cuda.current_stream(device=device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        try:
            yield
        finally:
            end_event.record(stream)
            end_event.synchronize()
            self.add_duration(
                key,
                start_event.elapsed_time(end_event) / 1000.0,
                layer_id=layer_id,
            )

    def finish_sample(self, status="ok", error=None):
        if not self.current:
            return None

        result = copy.deepcopy(self.current)
        result["status"] = status
        if error is not None:
            result["error"] = str(error)

        self.results.append(result)
        self.current = None
        return result

    def clear_results(self):
        self.results.clear()
