// blocked_attn.h — single-query decode attention over Q8 KV (H2 decode path).
//
// THEORY (what is actually implemented):
//   Scaled dot-product attention (Vaswani et al., NeurIPS 2017,
//   arXiv:1706.03762): scores s[p] = Q.K_p / sqrt(D), weights = softmax(s),
//   out = sum weights[p] * V_p. For single-query decode the score matrix is
//   1xL, so no N^2 materialization is needed; this file streams over the KV
//   blocks in O(L*D) time with an O(L) score buffer: pass 1 accumulates
//   Q.K_p with running max, pass 2 applies the stable softmax, pass 3 reloads
//   V rows for the weighted sum (double accumulation for the denominator).
//   This is an EXACT two-pass streaming decode, not a fused single-pass or
//   online-softmax kernel in the FlashAttention sense (Dao et al.,
//   arXiv:2205.14135); FlashAttention is cited only for the IO insight that a
//   1xL decode needs no tiling machinery. The block-iterated KV traversal
//   follows the PagedAttention layout idea (Kwon et al., SOSP 2023,
//   arXiv:2309.06180): fixed 256-token blocks. The score loop stays scalar by
//   design (1xL streaming: bytes dominate, not FLOPs). Numerics stay exact:
//   stable softmax with max subtraction, double accumulator.
#pragma once
#include <cstddef>
#include "kv_q8.h"

namespace ours {

// One decode step for one head. q: [head_dim] float.
// tmp: caller-provided scratch [seq_len + 2*head_dim] float.
// out: [head_dim] float.
void attn_decode_step(const KvCacheQ8& kv, int head, const float* q,
                      float* tmp, float* out);

// FP16 reference path (H2 ground truth for overhead measurement).
// K, V: [seq_len][n_head][head_dim] contiguous float (row = position).
// Same math as attn_decode_step; no quantization. Overhead% = (Q8-FP16)/FP16.
void attn_decode_step_fp16(const float* K, const float* V, int seq_len,
                           int n_head, int head_dim, int head, const float* q,
                           float* tmp, float* out);

}  // namespace ours
