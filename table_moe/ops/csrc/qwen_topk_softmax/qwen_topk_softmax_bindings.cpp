#include <torch/extension.h>

#include "qwen_topk_softmax_ops.h"

std::tuple<torch::Tensor, torch::Tensor> qwen_topk_softmax_binding(torch::Tensor gating_output,
                                                                   int64_t topk) {
  TORCH_CHECK(gating_output.is_cuda(), "qwen_topk_softmax: gating_output must be CUDA");
  TORCH_CHECK(gating_output.scalar_type() == at::kFloat,
              "qwen_topk_softmax: gating_output must be float32");
  TORCH_CHECK(gating_output.dim() == 2, "qwen_topk_softmax: gating_output must be 2D");
  TORCH_CHECK(topk > 0, "qwen_topk_softmax: topk must be positive");
  TORCH_CHECK(topk <= gating_output.size(1),
              "qwen_topk_softmax: topk cannot exceed num_experts");

  const auto num_tokens = gating_output.size(0);
  auto topk_weights = torch::empty({num_tokens, topk}, gating_output.options().dtype(at::kFloat));
  auto topk_indices = torch::empty({num_tokens, topk}, gating_output.options().dtype(at::kLong));
  auto token_expert_indices =
      torch::empty({num_tokens, topk}, gating_output.options().dtype(at::kInt));

  qwen_topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output);
  return std::make_tuple(topk_weights, topk_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qwen_topk_softmax", &qwen_topk_softmax_binding, "Qwen topk softmax kernel");
}
