// norm_rope.cpp — PROD kernels. No RNG.
#include "norm_rope.h"

#include <cmath>

namespace ours {

void rmsnorm(const float* x, const float* w, size_t n, float eps, float* out) {
  double ss = 0.0;
  for (size_t i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
  float inv = 1.0f / std::sqrt((float)(ss / (double)(n ? n : 1) + eps));
  for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

void rope_inplace(float* x, size_t head_dim, size_t pos, float theta,
                  size_t n_heads, size_t n_kv_heads) {
  (void)n_kv_heads;
  for (size_t h = 0; h < n_heads; ++h) {
    float* hd = x + h * head_dim;
    for (size_t i = 0; i + 1 < head_dim; i += 2) {
      float freq = 1.0f / std::pow(theta, (float)i / (float)head_dim);
      float ang = (float)pos * freq;
      float c = std::cos(ang), s = std::sin(ang);
      float a = hd[i], b = hd[i + 1];
      hd[i] = a * c - b * s;
      hd[i + 1] = a * s + b * c;
    }
  }
}

static float yarn_corr_dim(float n_dims, float n_ctx_orig, float n_rot, float base) {
  // llama.cpp ggml_rope_yarn_corr_dim (exact port).
  const float pi = 3.14159265358979f;
  return n_dims * std::log(n_ctx_orig / (n_rot * 2.0f * pi)) / (2.0f * std::log(base));
}

static float yarn_ramp(float low, float high, int i0) {
  // llama.cpp rope_yarn_ramp (exact port): linear, 1 at low dims -> 0 at high.
  const float y = ((float)i0 / 2.0f - low) / (high - low > 0.001f ? (high - low) : 0.001f);
  float c = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
  return 1.0f - c;
}

void rope_inplace_scaled(float* x, size_t head_dim, size_t pos, float theta,
                         size_t n_heads, float rope_scale, int32_t orig_ctx,
                         float beta_fast, float beta_slow) {
  if (rope_scale <= 1.0f || orig_ctx <= 0 || pos < (size_t)orig_ctx) {
    rope_inplace(x, head_dim, pos, theta, n_heads, n_heads);
    return;
  }
  // YaRN active (llama.cpp rope_yarn, ext_factor=1): low dims (high freq)
  // keep the extrapolated/original frequency, high dims (low freq) take the
  // interpolated frequency theta/scale. mscale folds the YaRN attention
  // temperature into the rotated vectors.
  float low = yarn_corr_dim((float)head_dim, (float)orig_ctx, beta_fast, theta);
  float high = yarn_corr_dim((float)head_dim, (float)orig_ctx, beta_slow, theta);
  if (low < 0.0f) low = 0.0f;
  if (high > (float)head_dim - 1.0f) high = (float)head_dim - 1.0f;
  const float freq_scale = 1.0f / rope_scale;
  const float mscale = 1.0f + 0.1f * std::log(rope_scale);
  for (size_t h = 0; h < n_heads; ++h) {
    float* hd = x + h * head_dim;
    for (size_t i = 0; i + 1 < head_dim; i += 2) {
      // theta_p is the pair FREQUENCY (1/period growth), matching rope_inplace:
      // freq = theta^{-2p/d}. theta_interp scales it by 1/scale; the NTK-by-parts
      // mix keeps the original frequency at low dims (high freq) and takes the
      // interpolated one at high dims (low freq) — llama.cpp rope_yarn exactly.
      const float theta_p = 1.0f / std::pow(theta, (float)i / (float)head_dim);
      const float theta_interp = freq_scale * theta_p;
      const float mix = yarn_ramp(low, high, (int)i);  // ext_factor = 1
      const float freq = theta_interp * (1.0f - mix) + theta_p * mix;
      float ang = (float)pos * freq;
      float c = std::cos(ang), sn = std::sin(ang);
      float a = hd[i], b = hd[i + 1];
      hd[i]     = (a * c - b * sn) * mscale;
      hd[i + 1] = (a * sn + b * c) * mscale;
    }
  }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

void silu_inplace(float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = silu(x[i]);
}

void softmax_inplace(float* x, size_t n) {
  if (n == 0) return;
  float mx = x[0];
  for (size_t i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
  float inv = (float)(1.0 / sum);
  for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

void matvec_f32(const float* W, const float* x, size_t m, size_t n, float* y) {
  for (size_t r = 0; r < m; ++r) {
    const float* row = W + r * n;
    double acc = 0.0;
    for (size_t c = 0; c < n; ++c) acc += (double)row[c] * (double)x[c];
    y[r] = (float)acc;
  }
}

}  // namespace ours
