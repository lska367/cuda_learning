#include <torch/extension.h>

// 只做声明，实现分散在各自的 .cu 里（这样 nvcc 并行编译多个文件）
void silu_mul_v0(torch::Tensor x, torch::Tensor out);
void silu_mul_v1(torch::Tensor x, torch::Tensor out);

void rms_norm_v0(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps);
void rms_norm_v1(torch::Tensor x, torch::Tensor w, torch::Tensor out, double eps);
void fused_add_rms_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor w,
                        torch::Tensor out, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuda_learning 练习 kernel";
    m.def("silu_mul_v0", &silu_mul_v0, "fused silu_and_mul, 标量版");
    m.def("silu_mul_v1", &silu_mul_v1, "fused silu_and_mul, 向量化版 (16B 访存)");
    m.def("rms_norm_v0", &rms_norm_v0, "RMSNorm, 1 CTA/行 + smem 归约");
    m.def("rms_norm_v1", &rms_norm_v1, "RMSNorm, 1 warp/行 + shuffle 归约");
    m.def("fused_add_rms_norm", &fused_add_rms_norm, "residual add + RMSNorm 融合");
}
