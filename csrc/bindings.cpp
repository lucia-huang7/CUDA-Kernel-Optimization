#include <torch/extension.h>

at::Tensor add_scalar_cuda(const at::Tensor& input, double value);
at::Tensor rmsnorm_cuda(const at::Tensor& input, const at::Tensor& weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("add_scalar", &add_scalar_cuda, "Add a scalar on the current CUDA stream",
             pybind11::arg("input"), pybind11::arg("value") = 1.0);
  module.def("rmsnorm", &rmsnorm_cuda, "Forward-only row-wise RMSNorm",
             pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("eps") = 1e-6);
}
