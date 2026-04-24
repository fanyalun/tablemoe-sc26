from __future__ import annotations

import logging
import warnings
from pathlib import Path

from transformers import AutoProcessor

from ._tablemoe_common import (
    configure_lmu_data_root,
    configure_tablemoe_cache_paths,
    ensure_repo_root_on_path,
    resolve_torch_dtype,
)
from .qwen3_vl.model import Qwen3VLChat
from .qwen3_vl.prompt import Qwen3VLPromptMixin
import torch

class Qwen3VLTableMoE(Qwen3VLChat):

    INSTALL_REQ = False

    def __init__(
        self,
        model_path: str,
        mode: str = "tablemoe",
        torch_dtype=None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        pca_dir: str | None = None,
        cache_dir: str | None = None,
        lmu_data_root: str | None = None,
        cache_ratio: float | None = None,
        keep_rate: float | None = None,
        **kwargs,
    ) -> None:
        use_vllm = kwargs.pop('use_vllm', False)
        use_lmdeploy = kwargs.pop('use_lmdeploy', False)
        attn_implementation = kwargs.pop('attn_implementation', None)
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video
        self.use_vllm = False
        self.use_lmdeploy = False
        self.limit_mm_per_prompt = 24
        normalized_mode = str(mode).strip().lower().replace("-", "_")
        if normalized_mode == "hybrid":
            normalized_mode = "tablemoe"
        if normalized_mode not in {"tablemoe", "skip", "offline", "online"}:
            raise ValueError(
                f"Unsupported table_moe mode: {mode}, expected one of ['tablemoe', 'skip', 'offline', 'online']"
            )
        self.tablemoe_mode = normalized_mode

        if use_vllm or use_lmdeploy:
            logging.warning('Qwen3VLTableMoE only supports transformers inference. use_vllm/use_lmdeploy will be ignored.')

        Qwen3VLPromptMixin.__init__(self, use_custom_prompt=use_custom_prompt)
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)

        repo_root = ensure_repo_root_on_path()
        configure_lmu_data_root(lmu_data_root)
        cache_path_overrides = configure_tablemoe_cache_paths(pca_dir=pca_dir, cache_dir=cache_dir)

        if pca_dir is not None:
            logging.warning('Qwen3VLTableMoE PCA dir: %s', pca_dir)
        if cache_dir is not None:
            logging.warning('Qwen3VLTableMoE cache dir: %s', cache_dir)
        if cache_ratio is not None:
            logging.warning('Qwen3VLTableMoE cache_ratio override: %s', cache_ratio)
        if keep_rate is not None:
            logging.warning('Qwen3VLTableMoE keep_rate override: %s', keep_rate)
        if lmu_data_root is None:
            logging.warning('Qwen3VLTableMoE LMUData root: %s', repo_root / 'LMUData')
        else:
            logging.warning('Qwen3VLTableMoE LMUData root: %s', Path(lmu_data_root))

        from table_moe import build_model
        from table_moe.models.qwen3_vl_moe import get_offload_config, update_offload_config

        previous_config = get_offload_config()
        build_kwargs = {}
        if attn_implementation is not None:
            build_kwargs['attn_implementation'] = attn_implementation

        try:
            dtype_override = {}
            if resolved_dtype is not None:
                dtype_override["model_dtype"] = resolved_dtype
            cache_ratio_override = {}
            if cache_ratio is not None:
                cache_ratio_override["cache_ratio"] = float(cache_ratio)
            keep_rate_override = {}
            if keep_rate is not None and normalized_mode == "tablemoe":
                keep_rate_override["keep_rate"] = float(keep_rate)
            update_offload_config(
                **dtype_override,
                **cache_path_overrides,
                **cache_ratio_override,
                **keep_rate_override,
            )
            model, lang_cfg, expert_cache = build_model(
                model_family='qwen3_vl_moe',
                mode=normalized_mode,
                model_id=model_path,
                **build_kwargs,
            )
        finally:
            update_offload_config(**previous_config)

        self.model = model.eval()
        self.model_dtype = resolved_dtype if resolved_dtype is not None else getattr(self.model, 'dtype', None)
        self.lang_cfg = lang_cfg
        self.expert_cache = expert_cache
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id

        torch.cuda.empty_cache()
        warnings.warn(
            f'Qwen3VLTableMoE uses offload_config model_dtype by default and lets config torch_dtype override it. mode={normalized_mode}'
        )


class Qwen3VLSkipBaseline(Qwen3VLTableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "skip"
        super().__init__(*args, **kwargs)


class Qwen3VLOfflineTableMoE(Qwen3VLTableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "offline"
        super().__init__(*args, **kwargs)


class Qwen3VLOnlineTableMoE(Qwen3VLTableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "online"
        super().__init__(*args, **kwargs)
