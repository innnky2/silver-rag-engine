// blocked_attn.cpp — decode attention implementation.
#include "blocked_attn.h"
#include <cmath>

namespace ours {

void attn_decode_step(const KvCacheQ8& kv, int head, const float* q,
                      float* tmp, float* out) {
  const int L = kv.seq_len();
  const int D = kv.head_dim;
  float* scores = tmp;          // [L]
  float* krow = tmp + L;        // caller provides L + 2D scratch
  float* vrow = tmp + L + D;
  // 1) Scores: Q.K_p per position (exact scalar dot; AVX2 path fuses dequant+dot
  // in matvec_weight_q8fast/H1 — attention scores stay scalar because decode is
  // 1xL streaming and the bottleneck is KV bytes, not score FLOPs).
  // Scale = 1/sqrt(D) computed at runtime (was hardcoded 1/sqrt(128)).
  const float inv_sqrt_d = 1.0f / std::sqrt((float)(D > 0 ? D : 1));
  float mx = -1e30f;
  int pos = 0;
  for (auto& b : kv.blocks)
    for (int p = 0; p < b.len; ++p, ++pos) {
      b.load_k_row(head, p, krow);
      double s = 0.0;
      for (int d = 0; d < D; ++d) s += (double)q[d] * (double)krow[d];
      scores[pos] = (float)(s * (double)inv_sqrt_d);
      if (scores[pos] > mx) mx = scores[pos];
    }
  // 2) Stable softmax.
  double denom = 0.0;
  for (int t = 0; t < L; ++t) {
    scores[t] = expf(scores[t] - mx);
    denom += scores[t];
  }
  // 3) out = sum softmax * V.
  for (int d = 0; d < D; ++d) out[d] = 0.0f;
  pos = 0;
  for (auto& b : kv.blocks)
    for (int p = 0; p < b.len; ++p, ++pos) {
      b.load_v_row(head, p, vrow);
      float w = scores[pos] / (float)denom;
      for (int d = 0; d < D; ++d) out[d] += w * vrow[d];
    }
}

void attn_decode_step_fp16(const float* K, const float* V, int seq_len,
                           int n_head, int head_dim, int head, const float* q,
                           float* tmp, float* out) {
  const int L = seq_len;
  const int D = head_dim;
  float* scores = tmp;
  float* krow = tmp + L;
  float* vrow = tmp + L + D;
  const float inv_sqrt_d = 1.0f / std::sqrt((float)(D > 0 ? D : 1));
  float mx = -1e30f;
  for (int p = 0; p < L; ++p) {
    const float* kr = K + ((std::size_t)p * n_head + head) * D;
    double s = 0.0;
    for (int d = 0; d < D; ++d) s += (double)q[d] * (double)kr[d];
    scores[p] = (float)(s * (double)inv_sqrt_d);
    if (scores[p] > mx) mx = scores[p];
  }
  double denom = 0.0;
  for (int t = 0; t < L; ++t) {
    scores[t] = expf(scores[t] - mx);
    denom += scores[t];
  }
  for (int d = 0; d < D; ++d) out[d] = 0.0f;
  for (int p = 0; p < L; ++p) {
    const float* vr = V + ((std::size_t)p * n_head + head) * D;
    float w = scores[p] / (float)denom;
    for (int d = 0; d < D; ++d) out[d] += w * vr[d];
  }
  (void)vrow;
  (void)krow;
}

}  // namespace ours
