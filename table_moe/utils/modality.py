import threading

import torch


class ModalityContext:
    """
    用于在推理过程中透传 input_ids 和 attention weights，
    以便在 MoE Layer 内部判断 Vision/Text 模态以及 Token 重要性。
    使用 ThreadLocal 确保并发安全。
    """

    _local = threading.local()

    @classmethod
    def set_input_ids(cls, input_ids: torch.Tensor):
        cls._local.input_ids = input_ids
        cls._clear_cache()

    @classmethod
    def get_input_ids(cls):
        return getattr(cls._local, "input_ids", None)

    @classmethod
    def set_attn_weights(cls, weights: torch.Tensor):
        cls._local.attn_weights = weights

    @classmethod
    def get_attn_weights(cls):
        return getattr(cls._local, "attn_weights", None)

    @classmethod
    def clear(cls):
        if hasattr(cls._local, "input_ids"):
            del cls._local.input_ids
        if hasattr(cls._local, "attn_weights"):
            del cls._local.attn_weights
        cls._clear_cache()

    @classmethod
    def _clear_cache(cls):
        if hasattr(cls._local, "cached_result"):
            del cls._local.cached_result

    @classmethod
    def get_modality_mask(cls, current_device, seq_len=None):
        cached = getattr(cls._local, "cached_result", None)
        if cached is not None:
            c_device, c_seq_len, c_mask = cached
            if c_device == current_device and c_seq_len == seq_len:
                return c_mask

        input_ids = cls.get_input_ids()

        if input_ids is None:
            fallback_len = seq_len if seq_len is not None else 1
            mask = torch.zeros((1, fallback_len), dtype=torch.bool, device=current_device)
        else:
            is_vision = (input_ids == 151655) | (input_ids == 151656)
            if seq_len is not None and is_vision.shape[1] != seq_len:
                if seq_len == 1:
                    mask = is_vision[:, -1:].to(current_device)
                else:
                    mask = is_vision[:, -seq_len:].to(current_device)
            else:
                mask = is_vision.to(current_device)

        cls._local.cached_result = (current_device, seq_len, mask)
        return mask
