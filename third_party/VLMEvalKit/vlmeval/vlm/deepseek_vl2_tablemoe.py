from __future__ import annotations

import logging
from pathlib import Path

from ._tablemoe_common import (
    configure_lmu_data_root,
    configure_tablemoe_cache_paths,
    ensure_repo_root_on_path,
    resolve_torch_dtype,
)
from .deepseek_vl2 import DeepSeekVL2


class DeepSeekVL2TableMoE(DeepSeekVL2):

    INSTALL_REQ = False

    def __init__(
        self,
        model_path='deepseek-ai/deepseek-vl2',
        mode: str = "tablemoe",
        torch_dtype=None,
        attn_implementation=None,
        pca_dir: str | None = None,
        cache_dir: str | None = None,
        lmu_data_root: str | None = None,
        cache_ratio: float | None = None,
        keep_rate: float | None = None,
        **kwargs,
    ):
        ignored_device_map = kwargs.pop('device_map', None)
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        normalized_mode = str(mode).strip().lower().replace("-", "_")
        if normalized_mode == "hybrid":
            normalized_mode = "tablemoe"
        if normalized_mode not in {"tablemoe", "skip", "offline", "online"}:
            raise ValueError(
                f"Unsupported table_moe mode: {mode}, expected one of ['tablemoe', 'skip', 'offline', 'online']"
            )
        self.tablemoe_mode = normalized_mode
        if attn_implementation is not None:
            logging.warning('DeepSeekVL2TableMoE ignores attn_implementation and always uses the table_moe builder default.')
        if ignored_device_map is not None:
            logging.warning('DeepSeekVL2TableMoE ignores device_map and uses the table_moe builder placement.')

        self._init_common(model_path=model_path, **kwargs)

        repo_root = ensure_repo_root_on_path()
        configure_lmu_data_root(lmu_data_root)
        cache_path_overrides = configure_tablemoe_cache_paths(pca_dir=pca_dir, cache_dir=cache_dir)

        if pca_dir is not None:
            logging.warning('DeepSeekVL2TableMoE PCA dir: %s', pca_dir)
        if cache_dir is not None:
            logging.warning('DeepSeekVL2TableMoE cache dir: %s', cache_dir)
        if cache_ratio is not None:
            logging.warning('DeepSeekVL2TableMoE cache_ratio override: %s', cache_ratio)
        if keep_rate is not None:
            logging.warning('DeepSeekVL2TableMoE keep_rate override: %s', keep_rate)
        if lmu_data_root is None:
            logging.warning('DeepSeekVL2TableMoE LMUData root: %s', repo_root / 'LMUData')
        else:
            logging.warning('DeepSeekVL2TableMoE LMUData root: %s', Path(lmu_data_root))

        from table_moe import build_model
        from table_moe.models.deepseek_vl2 import (
            get_deepseek_offload_config,
            update_deepseek_offload_config,
        )

        previous_config = get_deepseek_offload_config()
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
            update_deepseek_offload_config(
                **dtype_override,
                **cache_path_overrides,
                **cache_ratio_override,
                **keep_rate_override,
            )
            model, lang_cfg, expert_cache = build_model(
                model_family='deepseek_vl2',
                mode=normalized_mode,
                model_id=model_path,
            )
        finally:
            update_deepseek_offload_config(**previous_config)

        self.lang_cfg = lang_cfg
        self.expert_cache = expert_cache
        self._set_model(model, requested_dtype=resolved_dtype)
        logging.warning(
            'DeepSeekVL2TableMoE uses offload_config model_dtype by default and lets config torch_dtype override it. '
            'mode=%s',
            normalized_mode,
        )


class DeepSeekVL2SkipBaseline(DeepSeekVL2TableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "skip"
        super().__init__(*args, **kwargs)


class DeepSeekVL2OfflineTableMoE(DeepSeekVL2TableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "offline"
        super().__init__(*args, **kwargs)


class DeepSeekVL2OnlineTableMoE(DeepSeekVL2TableMoE):
    def __init__(self, *args, **kwargs):
        kwargs["mode"] = "online"
        super().__init__(*args, **kwargs)
