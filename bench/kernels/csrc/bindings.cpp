// torch 绑定层：只做形状/dtype 检查 + 转成裸指针，计算全在 .cu 里。
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "ops.h"

namespace {

inline void check_fp16_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kHalf, name, " must be fp16");
}

inline void check_fp32_contig(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be fp32");
}

}  // namespace

void silu_and_mul(const torch::Tensor& x, torch::Tensor& out) {
  check_fp16_contig(x, "x");
  check_fp16_contig(out, "out");
  TORCH_CHECK(x.dim() == 2, "x must be [T, 2I]");
  const int rows = static_cast<int>(x.size(0));
  const int cols = static_cast<int>(x.size(1) / 2);
  TORCH_CHECK(out.size(0) == rows && out.size(1) == cols, "out must be [T, I]");
  silu_and_mul_launch(x.data_ptr(), out.data_ptr(), rows, cols,
                      at::cuda::getCurrentCUDAStream());
}

int64_t rms_norm(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, double eps,
              int64_t impl) {
  check_fp16_contig(x, "x");
  check_fp16_contig(w, "w");
  check_fp16_contig(out, "out");
  TORCH_CHECK(x.dim() == 2 && out.sizes() == x.sizes(), "x/out must be [T, H]");
  TORCH_CHECK(w.numel() == x.size(1), "w must be [H]");
  const int vpt = rms_norm_launch(x.data_ptr(), w.data_ptr(), out.data_ptr(), static_cast<int>(x.size(0)),
                                 static_cast<int>(x.size(1)), static_cast<float>(eps),
                                 static_cast<int>(impl), at::cuda::getCurrentCUDAStream());
  return vpt;
}

int64_t fused_add_rms_norm(const torch::Tensor& x, const torch::Tensor& residual,
                        torch::Tensor& residual_out, const torch::Tensor& w, torch::Tensor& out,
                        double eps) {
  check_fp16_contig(x, "x");
  check_fp16_contig(residual, "residual");
  check_fp16_contig(residual_out, "residual_out");
  check_fp16_contig(w, "w");
  check_fp16_contig(out, "out");
  TORCH_CHECK(x.dim() == 2 && residual.sizes() == x.sizes() && out.sizes() == x.sizes(),
              "x/residual/out must be [T, H]");
  TORCH_CHECK(w.numel() == x.size(1), "w must be [H]");
  const int vpt = fused_add_rms_norm_launch(x.data_ptr(), residual.data_ptr(), residual_out.data_ptr(),
                            w.data_ptr(), out.data_ptr(), static_cast<int>(x.size(0)),
                                          static_cast<int>(x.size(1)), static_cast<float>(eps),
                                          at::cuda::getCurrentCUDAStream());
  return vpt;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul, "fused silu(gate)*up (fp16)");
  m.def("rms_norm", &rms_norm, "RMSNorm (fp16)", py::arg("x"), py::arg("w"), py::arg("out"),
        py::arg("eps") = 1e-6, py::arg("impl") = 1);
  m.def("fused_add_rms_norm", &fused_add_rms_norm, "residual add + RMSNorm (fp16)",
        py::arg("x"), py::arg("residual"), py::arg("residual_out"), py::arg("w"), py::arg("out"),
        py::arg("eps") = 1e-6);
}
