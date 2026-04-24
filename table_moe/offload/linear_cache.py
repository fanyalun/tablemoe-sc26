from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Iterator, Tuple, List
from collections import deque, defaultdict, OrderedDict
import threading
import queue

import torch
from torch import nn


def _maybe_dynamo_disable(fn):
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable(fn)
    return fn


ExpertUID = Any


@dataclass(frozen=False)
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    offloaded_index: int
    main_index: int = 0
    prefetched: bool = False
    using: bool = False


@dataclass
class EvictionGroupInfo:
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = field(default=0)
    resident_hits: int = field(default=0)
    prefetched_hits: int = field(default=0)
    misses: int = field(default=0)

    def add(self, info: ExpertInfo):
        infos_odict = self.offloaded_infos if info.offloaded else self.main_infos
        assert info.uid not in infos_odict, f"expert {info.uid} already exists"
        infos_odict[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        for uid, info in self.main_infos.items():
            if not info.using:
                return info
        raise ValueError("No evictable experts")

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.uid in self.offloaded_infos and info_to_evict.uid in self.main_infos
        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        info.using = True
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
            self.resident_hits += 1
        elif info.prefetched and info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.hits += 1
            self.prefetched_hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info} not in group")


class LinearCache:
    def __init__(self, make_module: callable, main_size: int, offload_size: int, buffer_size: int):
        self.module_type = self.w1_size = self.w2_size = self.w3_size = self.device = None
        self.active = False
        self._closed = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = dict()

        temp_module = self._check_module(make_module())

        self.main_modules = [temp_module] + [self._check_module(make_module()) for i in range(main_size - 1)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.w1_size is not None

        print(f"[LinearCache] Allocating Pinned Memory Pool for {offload_size} experts...")
        self.w1_pool = torch.empty(offload_size * self.w1_size, dtype=torch.uint8).pin_memory()
        self.w2_pool = torch.empty(offload_size * self.w2_size, dtype=torch.uint8).pin_memory()
        self.w3_pool = torch.empty(offload_size * self.w3_size, dtype=torch.uint8).pin_memory()

        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        self.device_expert_buffers = deque([self._check_module(make_module()) for _ in range(buffer_size)])
        self.info2buffer = {}
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

        self.copy_stream = torch.cuda.Stream()
        self.prefetch_lock = torch.cuda.Event()
        self.load_event = torch.cuda.Event()

        self.buffer_lock = threading.RLock()

        self.prefetch_queue = queue.Queue()
        self.worker_lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._daemon_prefetch_worker, daemon=True)
        self.worker_thread.start()

    def _check_module(self, module: nn.Module):
        assert hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3")
        assert isinstance(module.w1.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.w1_size = len(module.w1.storage)
            self.w2_size = len(module.w2.storage)
            self.w3_size = len(module.w3.storage)
            self.device = module.w1.storage.device
        else:
            assert len(module.w1.storage) == self.w1_size
            assert len(module.w2.storage) == self.w2_size
            assert len(module.w3.storage) == self.w3_size
            assert module.w1.storage.device == self.device
            assert module.w2.storage.device == self.device
            assert module.w3.storage.device == self.device
        return module

    def add_expert(self, uid: ExpertUID, module: nn.Module, eviction_group: int = 0, offload: Optional[bool] = None):
        return self.add_linear_storage(uid, [module.w1.storage, module.w2.storage, module.w3.storage], eviction_group=eviction_group, offload=offload)

    def add_linear_storage(self, uid: ExpertUID, storage: List[torch.UntypedStorage], eviction_group: int = 0, offload: Optional[bool] = None):
        assert uid not in self.registered_experts, f"expert {uid} already registered"
        w1_storage, w2_storage, w3_storage = storage

        for i in range(len(self.offloaded_infos)):
            if self.offloaded_infos[i] is None:
                s1, e1 = i * self.w1_size, (i + 1) * self.w1_size
                s2, e2 = i * self.w2_size, (i + 1) * self.w2_size
                s3, e3 = i * self.w3_size, (i + 1) * self.w3_size

                self.w1_pool[s1:e1].copy_(torch.as_tensor(w1_storage, dtype=torch.uint8))
                self.w2_pool[s2:e2].copy_(torch.as_tensor(w2_storage, dtype=torch.uint8))
                self.w3_pool[s3:e3].copy_(torch.as_tensor(w3_storage, dtype=torch.uint8))

                info = ExpertInfo(uid, eviction_group=eviction_group, offloaded=offload, offloaded_index=i)
                self.registered_experts[uid] = self.offloaded_infos[i] = info
                self.group_infos[eviction_group].add(info)
                break

        if offload is None or not offload:
            for i in range(len(self.main_modules)):
                if self.main_infos[i] is None:
                    self.main_modules[i].w1.storage.copy_(w1_storage)
                    self.main_modules[i].w2.storage.copy_(w2_storage)
                    self.main_modules[i].w3.storage.copy_(w3_storage)
                    self.main_infos[i] = info
                    info.main_index = i
                    break

    def clear_cache_stats(self):
        for group in self.group_infos.values():
            group.hits = 0
            group.resident_hits = 0
            group.prefetched_hits = 0
            group.misses = 0

    def get_cache_stats(self):
        total_hits = 0
        total_resident_hits = 0
        total_prefetched_hits = 0
        total_misses = 0
        group_stats = {}

        for group_id, group in self.group_infos.items():
            hits = int(group.hits)
            resident_hits = int(group.resident_hits)
            prefetched_hits = int(group.prefetched_hits)
            misses = int(group.misses)
            total = hits + misses
            total_hits += hits
            total_resident_hits += resident_hits
            total_prefetched_hits += prefetched_hits
            total_misses += misses
            group_stats[group_id] = {
                "hits": hits,
                "resident_hits": resident_hits,
                "prefetched_hits": prefetched_hits,
                "misses": misses,
                "total": total,
                "hit_rate": float(hits) / float(total) if total > 0 else 0.0,
            }

        total = total_hits + total_misses
        return {
            "hits": total_hits,
            "resident_hits": total_resident_hits,
            "prefetched_hits": total_prefetched_hits,
            "misses": total_misses,
            "total": total,
            "hit_rate": float(total_hits) / float(total) if total > 0 else 0.0,
            "groups": group_stats,
        }

    def check(self, layer_index, selected_experts):
        uids_to_fetch = []
        for expert in selected_experts:
            uid = (layer_index, expert)
            if self.registered_experts[uid].offloaded:
                uids_to_fetch.append(uid)
        return uids_to_fetch

    def release(self, uids):
        with self.buffer_lock:
            for uid in uids:
                info = self.registered_experts[uid]
                if info.prefetched:
                    if info.uid in self.info2buffer:
                        buffer = self.info2buffer.pop(info.uid)
                        try:
                            self.device_expert_buffers.remove(buffer)
                        except ValueError:
                            pass
                        buffer.free = True
                        if hasattr(buffer, "expert_uid"):
                            del buffer.expert_uid
                        self.device_expert_buffers.appendleft(buffer)
                    info.prefetched = False
                    info.offloaded = True

    @_maybe_dynamo_disable
    def _daemon_prefetch_worker(self):
        while True:
            try:
                uid = self.prefetch_queue.get()

                if uid is None:
                    self.prefetch_queue.task_done()
                    if self._closed:
                        break
                    continue

                if uid not in self.registered_experts:
                    self.prefetch_queue.task_done()
                    continue

                with self.worker_lock:
                    info = self.registered_experts[uid]

                    with self.buffer_lock:
                        if info.offloaded and not info.prefetched and uid not in self.info2buffer:
                            truly_free = sum(1 for buf in self.device_expert_buffers if buf.free)
                            if truly_free > 0 and self.device_expert_buffers[0].free:
                                self.prefetch(info)

                self.copy_stream.synchronize()
                with self.buffer_lock:
                    if uid in self.info2buffer:
                        self.info2buffer[uid].load = False
                self.prefetch_queue.task_done()

            except Exception as e:
                print(f"[LinearCache] Prefetch worker error: {e}")

    @_maybe_dynamo_disable
    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop_prefetch()
        try:
            self.prefetch_queue.put_nowait(None)
        except Exception:
            pass
        try:
            self.worker_thread.join(timeout=1.0)
        except Exception:
            pass

    @_maybe_dynamo_disable
    def prefetch_mult_experts(self, uids: List[ExpertUID]):
        if not uids:
            return

        self.stop_prefetch()

        for uid in uids:
            if uid in self.registered_experts:
                info = self.registered_experts[uid]
                if info.offloaded and not info.prefetched:
                    self.prefetch_queue.put(uid)

    @_maybe_dynamo_disable
    def stop_prefetch(self):
        try:
            with self.prefetch_queue.mutex:
                self.prefetch_queue.queue.clear()
        except Exception:
            pass

        with self.worker_lock:
            pass

    @_maybe_dynamo_disable
    def load_experts(
        self,
        *uids: ExpertUID,
        unordered: bool = False,
        uids_to_prefetch: List[ExpertUID] = None,
        values_fetcher: callable = None,
        values_event: torch.cuda.Event = None
    ) -> Iterator[Tuple[ExpertUID, nn.Module]]:
        self.stop_prefetch()

        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"

        if not unordered and len(uids) > 1:
            def _ordered_generator():
                last_idx = len(uids) - 1
                for idx, uid in enumerate(uids):
                    sub_prefetch = uids_to_prefetch if idx == last_idx else None
                    sub_fetcher = values_fetcher if idx == last_idx else None
                    sub_event = values_event if idx == last_idx else None
                    inner_iter = self.load_experts(
                        uid,
                        unordered=False,
                        uids_to_prefetch=sub_prefetch,
                        values_fetcher=sub_fetcher,
                        values_event=sub_event,
                    )
                    for item in inner_iter:
                        yield item

            return _ordered_generator()

        if not uids:
            if values_fetcher is not None:
                with torch.cuda.stream(self.copy_stream):
                    self.load_event.record()
                values_fetcher()
            if uids_to_prefetch:
                if values_event is not None:
                    values_event.wait(self.copy_stream)
                self.prefetch_mult_experts(uids_to_prefetch)
            return iter([])

        infos = [self.registered_experts[uid] for uid in uids]

        if unordered:
            infos.sort(key=lambda info: (info.offloaded, info.prefetched))

        n = len(infos)

        cached_end = 0
        for i, info in enumerate(infos):
            if info.offloaded or info.prefetched:
                cached_end = i
                break
        else:
            cached_end = n

        prefetched_end = cached_end
        for i in range(cached_end, n):
            if infos[i].offloaded:
                prefetched_end = i
                break
        else:
            prefetched_end = n

        needed_uids = set(info.uid for info in infos)
        with self.buffer_lock:
            uids_to_release = [uid for uid in self.info2buffer.keys() if uid not in needed_uids]
        if uids_to_release:
            self.release(uids_to_release)

        assert len(set(info.eviction_group for info in infos)) == 1, "experts must be in the same eviction group"
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        self.active = True

        experts = [None] * n

        for i in range(cached_end):
            experts[i] = self.main_modules[infos[i].main_index]

        eviction_list = []
        for uid, info in eviction_group.main_infos.items():
            if not info.using:
                eviction_list.append(info)
        evict_ptr = 0

        swap_ptr = cached_end
        load_ptr = prefetched_end

        with self.buffer_lock:
            while swap_ptr < prefetched_end and evict_ptr < len(eviction_list):
                experts[swap_ptr] = self._swap(infos[swap_ptr], eviction_list[evict_ptr])
                swap_ptr += 1
                evict_ptr += 1

        with self.buffer_lock:
            available_buffers = len(self.device_expert_buffers) - 1
            max_concurrent_loads = min(available_buffers, len(eviction_list) - evict_ptr, n - load_ptr)
            for _ in range(max_concurrent_loads):
                experts[load_ptr] = self._load(infos[load_ptr], eviction_list[evict_ptr])
                load_ptr += 1
                evict_ptr += 1

        def _generator():
            nonlocal swap_ptr, load_ptr, evict_ptr

            try:
                prefetch_started = False
                values_fetched = False

                for i in range(n):
                    info = infos[i]

                    if i > 0:
                        infos[i - 1].using = False
                        eviction_list.append(infos[i - 1])

                        if swap_ptr < prefetched_end and evict_ptr < len(eviction_list):
                            with self.buffer_lock:
                                experts[swap_ptr] = self._swap(infos[swap_ptr], eviction_list[evict_ptr])
                                swap_ptr += 1
                                evict_ptr += 1
                        elif load_ptr < n and evict_ptr < len(eviction_list):
                            with self.buffer_lock:
                                experts[load_ptr] = self._load(infos[load_ptr], eviction_list[evict_ptr])
                                load_ptr += 1
                                evict_ptr += 1
                        elif not values_fetched:
                            if values_fetcher is not None:
                                with torch.cuda.stream(self.copy_stream):
                                    self.load_event.record()
                                values_fetcher()
                            values_fetched = True
                        elif not prefetch_started:
                            if values_event is not None:
                                values_event.wait(self.copy_stream)
                            if uids_to_prefetch:
                                self.prefetch_mult_experts(uids_to_prefetch)
                            prefetch_started = True

                    yield (info.uid, experts[i])

                infos[-1].using = False
                eviction_list.append(infos[-1])

                if not values_fetched:
                    if values_fetcher is not None:
                        with torch.cuda.stream(self.copy_stream):
                            self.load_event.record()
                        values_fetcher()
                    values_fetched = True

                if not prefetch_started:
                    if values_event is not None:
                        values_event.wait(self.copy_stream)
                    if uids_to_prefetch:
                        self.prefetch_mult_experts(uids_to_prefetch)

            finally:
                self.active = False

        return _generator()

    @_maybe_dynamo_disable
    def prefetch(self, info_to_load: ExpertInfo):
        assert info_to_load.offloaded
        device_expert_buffer = self.device_expert_buffers.popleft()
        assert device_expert_buffer.free, "Prefetch caught a busy buffer"

        device_expert_buffer.compute_event.wait(self.copy_stream)

        device_expert_buffer.free = False
        device_expert_buffer.expert_uid = info_to_load.uid
        device_expert_buffer.load = True

        idx = info_to_load.offloaded_index

        with torch.cuda.stream(self.copy_stream):
            src_w1 = self.w1_pool[idx * self.w1_size : (idx + 1) * self.w1_size]
            src_w2 = self.w2_pool[idx * self.w2_size : (idx + 1) * self.w2_size]
            src_w3 = self.w3_pool[idx * self.w3_size : (idx + 1) * self.w3_size]

            dst_w1 = torch.as_tensor(device_expert_buffer.w1.storage, device=self.device, dtype=torch.uint8).view(self.w1_size)
            dst_w2 = torch.as_tensor(device_expert_buffer.w2.storage, device=self.device, dtype=torch.uint8).view(self.w2_size)
            dst_w3 = torch.as_tensor(device_expert_buffer.w3.storage, device=self.device, dtype=torch.uint8).view(self.w3_size)

            dst_w1.copy_(src_w1, non_blocking=True)
            device_expert_buffer.w1_event.record()
            dst_w3.copy_(src_w3, non_blocking=True)
            device_expert_buffer.w3_event.record()
            dst_w2.copy_(src_w2, non_blocking=True)
            device_expert_buffer.w2_event.record()

            self.prefetch_lock.record()

            info_to_load.prefetched = True
            info_to_load.offloaded = False
            self.info2buffer[info_to_load.uid] = device_expert_buffer

        self.device_expert_buffers.append(device_expert_buffer)

    @_maybe_dynamo_disable
    def _load(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        assert info_to_load.offloaded and not info_to_evict.offloaded
        device_expert_buffer = self.device_expert_buffers.popleft()
        assert device_expert_buffer.free, "Load caught a busy buffer"

        device_expert_buffer.compute_event.wait(self.copy_stream)
        device_expert_buffer.load = True
        idx = info_to_load.offloaded_index

        with torch.cuda.stream(self.copy_stream):
            src_w1 = self.w1_pool[idx * self.w1_size : (idx + 1) * self.w1_size]
            src_w2 = self.w2_pool[idx * self.w2_size : (idx + 1) * self.w2_size]
            src_w3 = self.w3_pool[idx * self.w3_size : (idx + 1) * self.w3_size]

            dst_w1 = torch.as_tensor(device_expert_buffer.w1.storage, device=self.device, dtype=torch.uint8).view(self.w1_size)
            dst_w2 = torch.as_tensor(device_expert_buffer.w2.storage, device=self.device, dtype=torch.uint8).view(self.w2_size)
            dst_w3 = torch.as_tensor(device_expert_buffer.w3.storage, device=self.device, dtype=torch.uint8).view(self.w3_size)

            dst_w1.copy_(src_w1, non_blocking=True)
            device_expert_buffer.w1_event.record()

            dst_w3.copy_(src_w3, non_blocking=True)
            device_expert_buffer.w3_event.record()

            dst_w2.copy_(src_w2, non_blocking=True)
            device_expert_buffer.w2_event.record()
            self.load_event.record()

        self.main_modules[info_to_evict.main_index].free = True
        device_expert_buffer.free = False

        self.device_expert_buffers.appendleft(self.main_modules[info_to_evict.main_index])

        self.main_modules[info_to_evict.main_index] = device_expert_buffer

        self.main_infos[info_to_evict.main_index] = info_to_load
        info_to_evict.offloaded, info_to_load.offloaded = info_to_load.offloaded, info_to_evict.offloaded
        info_to_load.main_index = info_to_evict.main_index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)
        return device_expert_buffer

    @_maybe_dynamo_disable
    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        assert info_to_load.prefetched

        device_expert_buffer = self.info2buffer[info_to_load.uid]

        try:
            self.device_expert_buffers.remove(device_expert_buffer)
        except ValueError:
            pass

        self.main_modules[info_to_evict.main_index].free = True
        device_expert_buffer.free = False

        self.device_expert_buffers.appendleft(self.main_modules[info_to_evict.main_index])

        self.main_modules[info_to_evict.main_index] = device_expert_buffer

        self.main_infos[info_to_evict.main_index] = info_to_load
        info_to_evict.offloaded = True
        info_to_load.main_index = info_to_evict.main_index
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)

        del self.info2buffer[info_to_load.uid]
        info_to_load.prefetched = False
        return device_expert_buffer
