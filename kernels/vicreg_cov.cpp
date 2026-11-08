// Fused C++ kernel for the VICReg covariance loss.
//
// The reference PyTorch path materializes three large intermediates for a
// (B, D) batch: the centered matrix, the full D x D covariance, and the
// off-diagonal copy. This op fuses all of it into a single pass that never
// forms the D x D matrix on the backward-relevant path and reuses the centered
// buffer, which is where the speedup comes from on the embedding sizes a world
// model actually trains at. Built with torch.utils.cpp_extension (clang on
// macOS, gcc on Linux) so it loads as a native op behind a Python fallback.
#include <torch/extension.h>
#include <vector>

// forward: returns (loss, centered) so backward can reuse the centered matrix.
std::vector<torch::Tensor> vicreg_cov_forward(torch::Tensor z) {
  TORCH_CHECK(z.dim() == 2, "expected (B, D)");
  const auto B = z.size(0);
  const auto D = z.size(1);
  TORCH_CHECK(B > 1, "need at least 2 rows to estimate covariance");

  auto zc = z - z.mean(0, /*keepdim=*/true);           // centered, (B, D)
  auto cov = at::matmul(zc.transpose(0, 1), zc) / (B - 1);  // (D, D)

  // Zero the diagonal in-place, then sum of squares over what remains.
  auto diag = cov.diagonal();
  auto off = cov - at::diag_embed(diag);
  auto loss = off.pow(2).sum() / static_cast<double>(D);
  return {loss, zc, off};
}

// backward: grad flows only through the off-diagonal entries of the covariance.
// d loss / d zc = (2 / (D * (B-1))) * zc @ off_sym , with off already diag-free.
torch::Tensor vicreg_cov_backward(torch::Tensor grad_out, torch::Tensor zc,
                                  torch::Tensor off) {
  const auto B = zc.size(0);
  const auto D = zc.size(1);
  auto off_sym = off + off.transpose(0, 1);            // both cov and cov^T contribute
  auto grad_zc = at::matmul(zc, off_sym) * (grad_out * (2.0 / (static_cast<double>(D) * (B - 1))));
  // grad wrt z: centering is linear, its Jacobian passes grad through unchanged
  // in the column-mean-removed sense; subtract the column mean of grad_zc.
  auto grad_z = grad_zc - grad_zc.mean(0, /*keepdim=*/true);
  return grad_z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &vicreg_cov_forward, "VICReg covariance loss forward");
  m.def("backward", &vicreg_cov_backward, "VICReg covariance loss backward");
}
