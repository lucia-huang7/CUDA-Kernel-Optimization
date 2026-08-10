#include <torch/extension.h>

at::Tensor add_scalar_cuda(const at::Tensor& input, double value);
at::Tensor rmsnorm_cuda(const at::Tensor& input, const at::Tensor& weight, double eps);

at::Tensor rmsnorm_warp_cuda(const at::Tensor& input, const at::Tensor& weight, double eps);

at::Tensor rmsnorm_cached_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor rmsnorm_rows_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor rmsnorm_vector_cuda(const at::Tensor&, const at::Tensor&, double);

at::Tensor rmsnorm_reload_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor rmsnorm_vector128_cuda(const at::Tensor&, const at::Tensor&, double);

at::Tensor layernorm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);

at::Tensor layernorm_warp_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor layernorm_cached_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor layernorm_vector_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);

at::Tensor gemm_naive_cuda(const at::Tensor&, const at::Tensor&);
at::Tensor gemm_cublas_cuda(const at::Tensor&, const at::Tensor&);

at::Tensor gemm_tiled_cuda(const at::Tensor&, const at::Tensor&);
at::Tensor gemm_register_cuda(const at::Tensor&, const at::Tensor&, int64_t);

at::Tensor gemm_vector_cuda(const at::Tensor&, const at::Tensor&, int64_t);

at::Tensor gemm_splitk_cuda(const at::Tensor&, const at::Tensor&, int64_t);

at::Tensor gemm_tuned_cuda(const at::Tensor&, const at::Tensor&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemm_tuned", &gemm_tuned_cuda, "Shape-regime FP32 GEMM dispatch");
  module.def("gemm_splitk", &gemm_splitk_cuda, "Guarded small-M split-K FP32 GEMM",
             pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("splits") = 8);
  module.def("gemm_vector", &gemm_vector_cuda, "Vectorized register-blocked FP32 GEMM",
             pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("config") = 0);
  module.def("gemm_tiled", &gemm_tiled_cuda, "Shared-memory FP32 GEMM");
  module.def("gemm_register", &gemm_register_cuda, "Register-blocked FP32 GEMM",
             pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("config") = 0);
  module.def("gemm_naive", &gemm_naive_cuda, "FP32 row-major GEMM baseline");
  module.def("gemm_cublas", &gemm_cublas_cuda, "FP32 pedantic cuBLAS GEMM control");
  module.def("layernorm_warp", &layernorm_warp_cuda, "Forward-only LayerNorm warp variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps") = 1e-5);
  module.def("layernorm_cached", &layernorm_cached_cuda, "Forward-only LayerNorm cached variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps") = 1e-5);
  module.def("layernorm_vector", &layernorm_vector_cuda, "Forward-only LayerNorm vector variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps") = 1e-5);
  module.def("layernorm", &layernorm_cuda, "Forward-only affine LayerNorm",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps") = 1e-5);
  module.def("add_scalar", &add_scalar_cuda, "Add a scalar on the current CUDA stream",
             pybind11::arg("input"), pybind11::arg("value") = 1.0);
  module.def("rmsnorm", &rmsnorm_cuda, "Forward-only row-wise RMSNorm",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_warp", &rmsnorm_warp_cuda, "RMSNorm with warp-shuffle reduction",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_cached", &rmsnorm_cached_cuda, "Specialized RMSNorm cached variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_rows", &rmsnorm_rows_cuda, "Specialized RMSNorm rows variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_vector", &rmsnorm_vector_cuda, "Specialized RMSNorm vector variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_reload", &rmsnorm_reload_cuda, "RMSNorm reload variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
  module.def("rmsnorm_vector128", &rmsnorm_vector128_cuda, "RMSNorm vector128 variant",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
}
