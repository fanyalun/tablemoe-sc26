import warnings
from pathlib import Path
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from deepseek_vl2.models.modeling_deepseek import (
        DeepseekV2FlashAttention2,
        apply_rotary_pos_emb,
        logger,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    vendored_repo = repo_root / "third_party" / "DeepSeek-VL2"
    if vendored_repo.exists():
        vendored_repo_str = str(vendored_repo)
        if vendored_repo_str not in sys.path:
            sys.path.insert(0, vendored_repo_str)
    from deepseek_vl2.models.modeling_deepseek import (
        DeepseekV2FlashAttention2,
        apply_rotary_pos_emb,
        logger,
    )

from ...utils.modality import ModalityContext


class HybridDeepSeekAttention(DeepseekV2FlashAttention2):
    """
    复用 DeepSeek-VL2 官方 FlashAttention 路径。
    仅在 prefill 阶段额外采集“最后一个 token 对当前 chunk token”的注意力分数。
    """

    def __init__(self, original_attn: DeepseekV2FlashAttention2):
        torch.nn.Module.__init__(self)

        attr_names = [
            "config",
            "layer_idx",
            "attention_dropout",
            "hidden_size",
            "num_heads",
            "max_position_embeddings",
            "rope_theta",
            "q_lora_rank",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
            "qk_nope_head_dim",
            "q_head_dim",
            "is_causal",
            "softmax_scale",
            "_flash_attn_uses_top_left_mask",
        ]
        module_names = [
            "q_proj",
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o_proj",
            "rotary_emb",
        ]

        for name in attr_names:
            if hasattr(original_attn, name):
                setattr(self, name, getattr(original_attn, name))

        for name in module_names:
            if hasattr(original_attn, name):
                setattr(self, name, getattr(original_attn, name))

        self.train(original_attn.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                "The input hidden states seems to be silently casted in float32, this might be related to"
                " the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        dropout_rate = self.attention_dropout if self.training else 0.0
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )

        if q_len > 1:
            self._cache_last_token_attention(
                query_states=query_states,
                key_states=key_states,
                attention_mask=attention_mask,
                query_length=q_len,
            )
        else:
            ModalityContext.set_attn_weights(None)

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _cache_last_token_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
    ) -> None:
        try:
            with torch.no_grad():
                kv_seq_len = key_states.shape[1]
                last_query = query_states[:, -1].float()
                key_states_fp32 = key_states.float()

                attn_scores = (
                    torch.einsum("bhd,bkhd->bhk", last_query, key_states_fp32)
                    * float(self.softmax_scale)
                )

                key_padding_mask = self._extract_key_padding_mask(
                    attention_mask=attention_mask,
                    kv_seq_len=kv_seq_len,
                    device=attn_scores.device,
                )

                if key_padding_mask is not None:
                    attn_scores = attn_scores.masked_fill(
                        ~key_padding_mask[:, None, :],
                        torch.finfo(attn_scores.dtype).min,
                    )

                attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32)

                if key_padding_mask is not None:
                    attn_probs = attn_probs.masked_fill(
                        ~key_padding_mask[:, None, :], 0.0
                    )

                # 只保留当前 chunk 的 token，对齐当前层 hidden_states 的 flatten 布局。
                current_chunk_probs = attn_probs[:, :, -query_length:]
                mean_score = current_chunk_probs.mean(dim=1)
                ModalityContext.set_attn_weights(mean_score.reshape(-1))

        except Exception:
            ModalityContext.set_attn_weights(None)

    def _extract_key_padding_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        kv_seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            return attention_mask[:, -kv_seq_len:].to(device=device, dtype=torch.bool)

        if attention_mask.dim() == 4:
            mask_slice = attention_mask[:, 0, -1, -kv_seq_len:]
            return (mask_slice == 0).to(device=device)

        return None
