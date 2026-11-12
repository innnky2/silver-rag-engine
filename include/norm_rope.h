#pragma once
// norm_rope.h — PROD math kernels: RMSNorm, RoPE, SiLU, softmax, matvec.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ours {

void rmsnorm(const float* x, const float* w, size_t n, float eps, float* out);
void rope_inplace(float* x, size_t head_dim, size_t pos, float theta,
                  size_t n_heads, size_t n_kv_heads);
// YaRN (Peng et al., arXiv:2309.00071; llama.cpp rope_yarn): NTK-by-parts.
// corr dims from (head_dim, orig_ctx, theta, beta_fast=32, beta_slow=1);
// linear ramp ramp=1 at low dims (high freq, keep extrapolated/original freq)
// -> 0 at high dims (low freq, full interpolation freq/scale); attention
// magnitude mscale = 1+0.1*ln(scale) multiplies the rotated vectors
// (llama.cpp scales the rope cache: both Q and K, so logits gain mscale^2).
void rope_inplace_scaled(float* x, size_t head_dim, size_t pos, float theta,
                         size_t n_heads, float rope_scale, int32_t orig_ctx,
                         float beta_fast, float beta_slow);
float silu(float x);
void silu_inplace(float* x, size_t n);
void softmax_inplace(float* x, size_t n);
// y[m] = W[m x n] (row-major f32) * x[n]
void matvec_f32(const float* W, const float* x, size_t m, size_t n, float* y);

}  // namespace ours
