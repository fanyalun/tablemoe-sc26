#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fp16_fused_moe_mlp.h"

namespace {

void fused_moe_ffn_fp16_into_binding(torch::Tensor hidden, torch::Tensor gate_proj,
                                     torch::Tensor up_proj, torch::Tensor down_proj,
                                     torch::Tensor gate_buf, torch::Tensor fused_buf,
                                     torch::Tensor output) {
  TORCH_CHECK(hidden.is_cuda(), "fused_moe_ffn_fp16_into: hidden must be CUDA");
  TORCH_CHECK(gate_proj.is_cuda() && up_proj.is_cuda() && down_proj.is_cuda(),
              "fused_moe_ffn_fp16_into: weights must be CUDA");
  TORCH_CHECK(gate_buf.is_cuda() && fused_buf.is_cuda() && output.is_cuda(),
              "fused_moe_ffn_fp16_into: buffers must be CUDA");
  TORCH_CHECK(hidden.is_contiguous(), "fused_moe_ffn_fp16_into: hidden must be contiguous");
  TORCH_CHECK(gate_proj.is_contiguous() && up_proj.is_contiguous() && down_proj.is_contiguous(),
              "fused_moe_ffn_fp16_into: weights must be contiguous");
  TORCH_CHECK(gate_buf.is_contiguous() && fused_buf.is_contiguous() && output.is_contiguous(),
              "fused_moe_ffn_fp16_into: buffers must be contiguous");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_moe_ffn_fp16_into(hidden, gate_proj, up_proj, down_proj, gate_buf, fused_buf, output,
                          stream);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_moe_ffn_fp16_into", &fused_moe_ffn_fp16_into_binding,
        "FP16 CUTLASS fused MoE FFN");
}
