import torch
from contextlib import nullcontext
from collections.abc import Callable
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextAttention, 
    apply_rotary_pos_emb,
    eager_attention_forward
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...utils.modality import ModalityContext

class HybridQwen3Attention(Qwen3VLMoeTextAttention):
    """
    继承自 Qwen3VLMoeTextAttention。
    在执行 FlashAttention 之前，拦截 Q 和 K，计算最后一个 Token 的全局注意力权重。
    """
    def __init__(self, original_attn: Qwen3VLMoeTextAttention):
        # 1. 初始化父类，复制配置
        super().__init__(original_attn.config, original_attn.layer_idx)
        
        # 2. 权重偷梁换柱 (共享内存，不增加显存)
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.q_norm = original_attn.q_norm
        self.k_norm = original_attn.k_norm
        
        # 复制其他属性
        self.scaling = original_attn.scaling
        self.attention_dropout = original_attn.attention_dropout
        self.is_causal = original_attn.is_causal
        self.profiler = None

    def set_perf_profiler(self, profiler):
        self.profiler = profiler

    def _measure_cuda(self, key, enabled):
        if not enabled or self.profiler is None or not self.profiler.is_active():
            return nullcontext()
        return self.profiler.measure_cuda(
            key,
            layer_id=getattr(self, "layer_idx", None),
            device=self.q_proj.weight.device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # === Part 1: 标准 Qwen3 Attention 前处理 (参照源码) ===
        input_shape = hidden_states.shape[:-1]
        is_prefill = input_shape[1] > 1

        with self._measure_cuda("qwen.hybrid_attention.total", enabled=is_prefill):
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # === 🚀 核心修改: 在此处插入 Importance Calculation ===
            # 仅在 Prefill 阶段 (seq_len > 1) 且非 decode 阶段计算
            if is_prefill:
                with self._measure_cuda("qwen.hybrid_attention.importance_only", enabled=True):
                    self._compute_and_cache_importance(query_states, key_states)
            else:
                ModalityContext.set_attn_weights(None)
            # ========================================================

            # === Part 2: 标准 Attention 计算 (FlashAttn/SDPA) ===
            # 强制不输出权重以节省显存
            kwargs['output_attentions'] = False 
            
            # # 获取 Attention 实现 (通常是 flash_attention_2)
            # attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            #     self.config._attn_implementation, eager_attention_forward
            # )
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            
            with self._measure_cuda("qwen.hybrid_attention.flash_attn2", enabled=is_prefill):
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            
            return attn_output, attn_weights

    def _compute_and_cache_importance(self, query_states, key_states):
        """
        利用现成的 RoPE 后的 Q 和 K 计算 1xN 注意力
        Q: [Batch, Heads, Seq_Q, Dim] -> 取最后一个 Token
        K: [Batch, KV_Heads, Seq_K, Dim]
        """
        try:
            with torch.no_grad():
                # 1. 取出最后一个 Token 的 Query
                # query_states: [Batch, Heads, Seq_Len, Dim]
                last_q = query_states[:, :, -1:, :] # [Batch, Heads, 1, Dim]

                # 2. 处理 GQA (如果 KV heads 少于 Q heads)
                if self.num_key_value_groups > 1:
                    key_states_expanded = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
                else:
                    key_states_expanded = key_states

                # 3. 点积计算 (Batch, Heads, 1, Seq_Len)
                # 显存占用: O(Heads * Seq_Len) -> 极小
                attn_score = torch.matmul(last_q, key_states_expanded.transpose(2, 3)) * self.scaling
                
                # 4. Softmax (Head维度独立，Seq维度归一化)
                attn_score = torch.softmax(attn_score, dim=-1, dtype=torch.float32)

                # 5. 平均多头 (Batch, 1, Seq_Len) -> (Batch, Seq_Len)
                mean_score = attn_score.mean(dim=1).squeeze(1)

                # 6. 存入 Context
                # 必须展平，因为 MoE Layer 接收的是 Flatten 后的 tokens
                ModalityContext.set_attn_weights(mean_score.view(-1))
                
        except Exception as e:
            # 容错：如果出错不应卡死主流程
            # print(f"Hybrid Attn Error: {e}")
            ModalityContext.set_attn_weights(None)
