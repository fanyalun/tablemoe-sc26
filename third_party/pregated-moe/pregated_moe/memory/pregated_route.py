from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch


@dataclass
class LayerRoute:
    router_mask: torch.Tensor
    routing_weights_mask: torch.Tensor
    expert_ids: List[int]
    router_logits: Optional[torch.Tensor] = None
    topk_idx: Optional[torch.Tensor] = None
    topk_weight: Optional[torch.Tensor] = None


class PregatedRouteController:
    def __init__(self, archer_engine, expert_tensor_map, sparse_modules):
        self.archer_engine = archer_engine
        self.expert_tensor_map = expert_tensor_map
        self.debug_enabled = os.getenv("PREGATED_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.layer_modules = {
            int(module.layer_id): module for module in sparse_modules
        }
        self.layer_ids = sorted(self.layer_modules)
        self.next_layer_map = {}
        for idx, layer_id in enumerate(self.layer_ids):
            next_layer = (
                self.layer_ids[idx + 1]
                if idx + 1 < len(self.layer_ids)
                else None
            )
            self.next_layer_map[layer_id] = next_layer

        self._next_request_id = 0
        self._active_request_id = None
        self._pending_routes: Dict[int, LayerRoute] = {}
        self._submitted_layers: set[int] = set()
        self._released_layers: set[int] = set()

    def _debug(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[PregatedDebug] {message}", flush=True)

    def _reset_request_state(self) -> None:
        self._active_request_id = None
        self._pending_routes.clear()
        self._submitted_layers.clear()
        self._released_layers.clear()

    def begin_request_if_needed(self, layer_id: int) -> int:
        if self.is_first_layer(layer_id) and self._active_request_id is not None:
            self.archer_engine.end_exact_request(int(self._active_request_id))
            self._reset_request_state()
        if self._active_request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._active_request_id = request_id
            self.archer_engine.begin_exact_request(request_id)
            self._debug(
                f"begin_request request_id={request_id} first_layer={layer_id}"
            )
        return int(self._active_request_id)

    def current_request_id(self) -> Optional[int]:
        return self._active_request_id

    def is_first_layer(self, layer_id: int) -> bool:
        return bool(self.layer_ids) and layer_id == self.layer_ids[0]

    def is_last_layer(self, layer_id: int) -> bool:
        return bool(self.layer_ids) and layer_id == self.layer_ids[-1]

    def next_sparse_layer_id(self, layer_id: int) -> Optional[int]:
        return self.next_layer_map.get(int(layer_id))

    def get_next_module(self, layer_id: int):
        next_layer = self.next_sparse_layer_id(layer_id)
        if next_layer is None:
            return None
        return self.layer_modules[next_layer]

    def set_pending_route(self, layer_id: int, route: LayerRoute) -> None:
        self._pending_routes[int(layer_id)] = route
        self._debug(
            f"set_pending_route layer={layer_id} experts={route.expert_ids[:16]} total={len(route.expert_ids)}"
        )

    def pop_pending_route(self, layer_id: int) -> LayerRoute:
        if layer_id not in self._pending_routes:
            raise RuntimeError(f"Missing pending pregated route for layer {layer_id}")
        route = self._pending_routes.pop(int(layer_id))
        self._debug(
            f"pop_pending_route layer={layer_id} experts={route.expert_ids[:16]} total={len(route.expert_ids)}"
        )
        return route

    def submit_layer(
        self, request_id: int, layer_id: int, expert_ids: Iterable[int]
    ) -> None:
        if layer_id in self._submitted_layers:
            return

        tensor_ids = []
        seen = set()
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id in seen:
                continue
            seen.add(expert_id)
            tensor_id = self.expert_tensor_map.get((int(layer_id), expert_id))
            if tensor_id is None:
                raise RuntimeError(
                    f"Missing expert tensor id for layer={layer_id}, expert={expert_id}"
                )
            tensor_ids.append(int(tensor_id))

        self._debug(
            f"submit_layer request={request_id} layer={layer_id} experts={list(seen)[:16]} total={len(seen)} tensor_ids={tensor_ids[:16]}"
        )
        self.archer_engine.submit_exact_layer(
            int(request_id),
            int(layer_id),
            tensor_ids,
        )
        self._submitted_layers.add(int(layer_id))

    def wait_layer_ready(self, request_id: int, layer_id: int) -> None:
        self._debug(f"wait_layer_ready request={request_id} layer={layer_id}")
        self.archer_engine.wait_layer_ready(int(request_id), int(layer_id))
        self._debug(
            f"wait_layer_ready_done request={request_id} layer={layer_id}"
        )

    def release_layer(self, request_id: int, layer_id: int) -> None:
        if layer_id in self._released_layers:
            return
        self._debug(f"release_layer request={request_id} layer={layer_id}")
        self.archer_engine.release_layer(int(request_id), int(layer_id))
        self._released_layers.add(int(layer_id))

    def finish_request_if_last(self, request_id: int, layer_id: int) -> None:
        if not self.is_last_layer(layer_id):
            return
        self.archer_engine.end_exact_request(int(request_id))
        self._reset_request_state()

    def fetch_gate_tensors(self, request_id: int, module) -> torch.device:
        tensor_ids = list(getattr(module, "gate_tensor_ids", ()) or ())
        if not tensor_ids:
            raise RuntimeError(
                f"Missing gate_tensor_ids for sparse layer {getattr(module, 'layer_id', 'unknown')}"
            )
        default_device_idx = self.archer_engine.get_node_default_device(tensor_ids)
        current_device_idx = self.archer_engine.get_node_device(tensor_ids)
        if current_device_idx == default_device_idx:
            self._debug(
                f"fetch_gate_tensors_skip request={request_id} next_layer={getattr(module, 'layer_id', 'unknown')} device=cuda:{current_device_idx}"
            )
            return torch.device(f"cuda:{current_device_idx}")
        self._debug(
            f"fetch_gate_tensors request={request_id} next_layer={getattr(module, 'layer_id', 'unknown')} tensor_ids={tensor_ids[:16]}"
        )
        self.archer_engine.fetch_tensors(int(request_id), tensor_ids)
        device_idx = self.archer_engine.get_node_default_device(tensor_ids)
        self._debug(
            f"fetch_gate_tensors_done request={request_id} next_layer={getattr(module, 'layer_id', 'unknown')} device=cuda:{device_idx}"
        )
        return torch.device(f"cuda:{device_idx}")
