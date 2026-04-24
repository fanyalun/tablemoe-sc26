#pragma once

#include <torch/all.h>

void qwen_topk_softmax(torch::Tensor& topk_weights,          // [num_tokens, topk]
                       torch::Tensor& topk_indices,          // [num_tokens, topk]
                       torch::Tensor& token_expert_indices,  // [num_tokens, topk]
                       torch::Tensor& gating_output);        // [num_tokens, num_experts]
