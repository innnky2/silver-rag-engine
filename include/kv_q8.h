// kv_q8.h — H2: per-block Q8 KV cache (2x capacity, ~zero overhead).
//
// THEORY (why per-block Q8):
//   Autoregressive decode is memory-bound: per step the full weight matrix
//   plus the full KV cache is re-read, so tokens/s <= bandwidth / bytes. KIVI
//   (Liu et al., ICML 2024, arXiv:2402.02750) shows the key cache has strong
//   per-channel outliers (quantize per-channel) while the value cache is
//   flatter (per-token works), reaching 2-bit with small quality loss and
//   2.6x less peak memory. FlexGen (Sheng et al., 2023) and vLLM PagedAttention
//   (Kwon et al., SOSP 2023, arXiv:2309.06180) establish the block/paged
//   layout: fixed blocks (vLLM regime: O(1) amortized append with no rewrite;
//   ours differs on range widening — see the O(block) note below).
//   Layout: uniform Q8, one (min,scale) pair per head-dim per 256-token block,
//   K and V quantized independently, running min/max with exact float history.
//   Append is O(1) when the new token lies inside all running ranges; a
//   range-widening append requantizes its block from history (<=256 rows),
//   so worst-case append is O(block) during range growth, amortized O(1)
//   afterwards. Load is O(1) per element. Wire size is ~1/2 FP16 bytes, at
//   the cost of accuracy vs KIVI's asymmetric per-channel/per-token scheme.
//   H2 therefore predicts parity only up to the context where the coarse scale
//   still holds (<=8K) and a crossover beyond it; if H2 fails at <=8K the
//   scale scheme is wrong and must move toward per-channel (KIVI) or
//   per-64 scales. Falsification, not tuning, decides.
//
// Design: sequence split into 256-token blocks. Each block stores int8 plus
// (min, scale) per head-dim plus exact float history (resident); dequant is
// O(1) per element on read.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ours {

struct KvBlockQ8 {
  static constexpr int kBlock = 256;
  int n_head = 0;     // KV heads (GQA)
  int head_dim = 0;   // dim per head
  int len = 0;        // tokens stored in this block (0..kBlock)
  // PROD: K and V have SEPARATE int8 buffers + running min/max scales.
  // Layout per block: [len][n_head][head_dim] int8 + float history for exact
  // requantize when the range widens (bounded 256 rows; no first-token bias).
  std::vector<int8_t> kdata, vdata;
  std::vector<float> k_mn, k_sc, v_mn, v_sc;
  std::vector<float> k_hist, v_hist;  // [len][n_head][head_dim] exact floats

  KvBlockQ8() = default;
  KvBlockQ8(int n_head_, int head_dim_);

  // Write one token (k[v][d], v[v][d] float) into the block. False when full.
  // Running min/max per head-dim; widening requantizes from float history.
  bool append_key(const float* k_row, const float* v_row);

  // Read one head-h row at position p -> float (dequantized). For attention.
  void load_k_row(int h, int p, float* out) const;
  void load_v_row(int h, int p, float* out) const;

  // Bytes of a full block on the wire (K+V int8 + scales) for H5 roofline accounting.
  // NOTE: resident bytes additionally hold exact float history (see bytes_used).
  std::size_t bytes_full() const;
  // Len-aware RESIDENT bytes actually used (full blocks + partial tail + scales
  // + exact float history for lossless requantize). Honest memory accounting.
  std::size_t bytes_used() const;
  // Len-aware WIRE bytes (K+V int8 + scales, no history): the H2 ~1/2 FP16 claim.
  std::size_t bytes_wire_full() const;
  std::size_t bytes_wire_used() const;
};

struct KvCacheQ8 {
  int n_head = 0, head_dim = 0;
  std::vector<KvBlockQ8> blocks;

  KvCacheQ8() = default;
  KvCacheQ8(int n_head_, int head_dim_);
  void append(const float* k_row, const float* v_row);  // k/v: [n_head][head_dim]
  int seq_len() const;
  std::size_t bytes_total() const;  // PROD resident: K+V + scales + history
  std::size_t bytes_wire_total() const;  // PROD wire: len-aware K+V (H2 claim)
  std::size_t bytes_fp16_same_len() const;
};

}  // namespace ours
