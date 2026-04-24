class DeepSeekV2Adapter:
    model_family = "deepseek_vl2"
    default_attn_implementation = "flash_attention_2"

    def build_baseline(self, model_id: str, **kwargs):
        from .builder import build_offload_model_deepseek_baseline

        return build_offload_model_deepseek_baseline(model_id=model_id, **kwargs)

    def build_skip_offload(self, model_id: str, **kwargs):
        from .builder import build_offload_model_deepseek_skip_offload

        return build_offload_model_deepseek_skip_offload(model_id=model_id, **kwargs)

    def build_hybrid(self, model_id: str, **kwargs):
        from .builder import build_offload_model_deepseek_hybrid

        return build_offload_model_deepseek_hybrid(model_id=model_id, **kwargs)

    def build_offline(self, model_id: str, **kwargs):
        from .builder import build_offload_model_deepseek_offline

        return build_offload_model_deepseek_offline(model_id=model_id, **kwargs)

    def build_online(self, model_id: str, **kwargs):
        from .builder import build_offload_model_deepseek_online

        return build_offload_model_deepseek_online(model_id=model_id, **kwargs)

    def build_skip(self, model_id: str, **kwargs):
        from .builder import build_full_model_deepseek_skip

        return build_full_model_deepseek_skip(model_id=model_id, **kwargs)
