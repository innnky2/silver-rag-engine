// kv_q8.h — H2: per-block Q8 KV cache.
//
// Decode is memory-bound (tokens/s <= bandwidth / bytes), so the KV cache is
// quantized to int8 per 256-token block: one (min,scale) pair per head-dim,
// K and V quantized independently with running min/max over exact float
// history. Append is O(1) inside the running range; a range-widening append
// requantizes its block (<=256 rows). Wire size is ~1/2 of FP16.
// References: KIVI (Liu et al., ICML 2024, arXiv:2402.02750), FlexGen
// (Sheng et al., 2023), PagedAttention (Kwon et al., SOSP 2023,
// arXiv:2309.06180).
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
  // K and V have separate int8 buffers + running min/max scales.
  // Layout per block: [len][n_head][head_dim] int8 + float history for exact
  // requantize when the range widens (bounded 256 rows).
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

  // Bytes of a full block on the wire (K+V int8 + scales).
  std::size_t bytes_full() const;
  // Len-aware resident bytes actually used (blocks + tail + scales + history).
  std::size_t bytes_used() const;
  // Len-aware wire bytes (K+V int8 + scales, no history).
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
  std::size_t bytes_total() const;  // resident: K+V + scales + history
  std::size_t bytes_wire_total() const;  // wire: len-aware K+V
  std::size_t bytes_fp16_same_len() const;
};

}  // namespace ours
