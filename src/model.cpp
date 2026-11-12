// model.cpp — PROD llama decode on real GGUF weights.
#include "model.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

#include "blocked_attn.h"
#include "lut_gemm.h"
#include "norm_rope.h"
#include "quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ours {

static int32_t get_i32(const gguf::ModelInfo& mi, const std::string& arch, const std::string& key, int32_t dflt) {
  auto it = mi.i32_meta.find(arch + "." + key);
  if (it != mi.i32_meta.end()) return it->second;
  auto u = mi.u32_meta.find(arch + "." + key);
  if (u != mi.u32_meta.end()) return (int32_t)u->second;
  return dflt;
}
static float get_f32(const gguf::ModelInfo& mi, const std::string& arch, const std::string& key, float dflt) {
  auto it = mi.f32_meta.find(arch + "." + key);
  if (it != mi.f32_meta.end()) return it->second;
  return dflt;
}

bool LlamaModel::load(const std::string& gguf_path, int threads, bool pin, std::string& err) {
  gguf_path_ = gguf_path;
  if (!gguf::open(gguf_path, mi_, err)) return false;
  std::string arch = mi_.arch.empty() ? "llama" : mi_.arch;
  dims_.n_layer = get_i32(mi_, arch, "block_count", 0);
  dims_.n_embd = get_i32(mi_, arch, "embedding_length", 0);
  dims_.n_head = get_i32(mi_, arch, "attention.head_count", 0);
  dims_.n_kv = get_i32(mi_, arch, "attention.head_count_kv", dims_.n_head);
  dims_.n_ff = get_i32(mi_, arch, "feed_forward_length", 0);
  dims_.vocab = (int32_t)mi_.tokens.size();
  if (dims_.vocab == 0) dims_.vocab = get_i32(mi_, arch, "vocab_size", 32000);
  dims_.norm_eps = get_f32(mi_, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
  if (dims_.norm_eps <= 0) dims_.norm_eps = 1e-5f;
  dims_.rope_theta = get_f32(mi_, arch, "rope.freq_base", 10000.0f);
  dims_.rope_scale = get_f32(mi_, arch, "rope.scaling.factor", 1.0f);
  if (dims_.rope_scale <= 0) dims_.rope_scale = 1.0f;
  dims_.rope_orig_ctx = get_i32(mi_, arch, "rope.scaling.original_context_length", 0);
  dims_.rope_beta_fast = get_f32(mi_, arch, "rope.scaling.beta_fast", 32.0f);
  dims_.rope_beta_slow = get_f32(mi_, arch, "rope.scaling.beta_slow", 1.0f);
  if (dims_.n_head > 0 && dims_.n_embd > 0)
    dims_.head_dim = dims_.n_embd / dims_.n_head;
  else
    dims_.head_dim = get_i32(mi_, arch, "rope.dimension_count", 128);
  if (dims_.n_layer <= 0 || dims_.n_embd <= 0 || dims_.n_head <= 0 || dims_.head_dim <= 0) {
    err = "bad arch dims (missing " + arch + ".* KV)";
    return false;
  }
  // Tokenizer from GGUF arrays (pieces + scores + types + BPE merges).
  tok_.pieces = mi_.tokens;
  tok_.scores = mi_.scores;
  tok_.types = mi_.toktypes;
  tok_.merges = mi_.merges;
  auto f = [&](const std::string& k, int32_t d) {
    auto it = mi_.u32_meta.find(std::string("tokenizer.ggml.") + k);
    if (it != mi_.u32_meta.end()) return (int32_t)it->second;
    auto j = mi_.i32_meta.find(std::string("tokenizer.ggml.") + k);
    if (j != mi_.i32_meta.end()) return j->second;
    return d;
  };
  tok_.bos = f("bos_token_id", 1);
  tok_.eos = f("eos_token_id", 2);
  tok_.unk = f("unknown_token_id", 0);
  tok_.pad = f("padding_token_id", -1);
  {
    auto ab = mi_.bool_meta.find("tokenizer.ggml.add_bos_token");
    tok_.add_bos = (ab == mi_.bool_meta.end()) ? true : ab->second;
  }
  // Index + load all weight payloads needed for decode.
  index_.clear();
  weights_.clear();
  for (size_t i = 0; i < mi_.tensors.size(); ++i) index_[mi_.tensors[i].name] = i;
  auto need = [&](const std::string& name, bool optional) -> bool {
    auto it = index_.find(name);
    if (it == index_.end()) return optional;
    const auto& ti = mi_.tensors[it->second];
    WeightBlob wb;
    wb.qtype = ti.type;
    wb.dims = ti.dims;
    if (!gguf::read_tensor_bytes(gguf_path_, mi_, it->second, wb.raw, err)) return false;
    weights_[name] = std::move(wb);
    return true;
  };
  if (!need("token_embd.weight", false)) { if (err.empty()) err = "missing token_embd.weight"; return false; }
  if (!need("output_norm.weight", true)) { /* some archs use blk output norm only */ }
  if (!need("output.weight", true)) {
    if (!need("output.weight", false)) { if (err.empty()) err = "missing output.weight"; return false; }
  }
  for (int l = 0; l < dims_.n_layer; ++l) {
    std::string b = "blk." + std::to_string(l) + ".";
    const char* names[] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                           "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight",
                           "ffn_up.weight", "ffn_down.weight"};
    for (auto n : names) {
      if (!need(b + n, false)) { if (err.empty()) err = std::string("missing ") + b + n; return false; }
    }
  }
  kv_.clear();
  kv_.reserve((size_t)dims_.n_layer);
  for (int l = 0; l < dims_.n_layer; ++l) kv_.emplace_back(dims_.n_kv, dims_.head_dim);
  size_t N = (size_t)dims_.n_embd;
  size_t FF = (size_t)(dims_.n_ff > 0 ? dims_.n_ff : 0);
  size_t HD = (size_t)dims_.head_dim;
  size_t NH = (size_t)dims_.n_head;
  xb_.assign(N, 0); xb2_.assign(N, 0);
  hb_.assign(FF, 0); hb2_.assign(FF, 0);
  q_.assign(NH * HD, 0); k_.assign((size_t)dims_.n_kv * HD, 0); v_.assign((size_t)dims_.n_kv * HD, 0);
  att_.assign(NH * HD, 0);
  logits_f_.assign((size_t)(dims_.vocab > 0 ? dims_.vocab : 0), 0);
  // Scratch sized for the first decode; grown on demand in decode_step
  // (needs seq_len + 2*head_dim floats; init covers typical short prompts).
  constexpr size_t kInitSeq = 4096, kHeadDimMax = 256;
  tmp_attn_.assign(kInitSeq + 2 * kHeadDimMax, 0);  // grown on demand in decode_step
  q8buf_.assign(N > FF ? N : FF, 0);
  pool_ = ThreadPool(threads, pin);
  loaded_ = true;
  return true;
}

void LlamaModel::reset_kv() {
  kv_.clear();
  for (int l = 0; l < dims_.n_layer; ++l) kv_.emplace_back(dims_.n_kv, dims_.head_dim);
}

bool LlamaModel::get_row(const std::string& name, size_t row, std::vector<float>& out, std::string& err) {
  auto it = weights_.find(name);
  if (it == weights_.end()) { err = "weight not loaded: " + name; return false; }
  const WeightBlob& wb = it->second;
  if (wb.dims.size() == 1) {
    if (row != 0) { err = "row out of range (1-D): " + name; return false; }
    return quant::dequant_to_f32(wb.qtype, wb.raw.data(), wb.raw.size(), (size_t)wb.dims[0], out, err);
  }
  if (wb.dims.size() != 2) { err = "weight not 1-D/2-D: " + name; return false; }
  size_t n_cols = (size_t)wb.dims[0], n_rows = (size_t)wb.dims[1];
  return quant::dequant_row_to_f32(wb.qtype, wb.raw.data(), wb.raw.size(), n_cols, n_rows, row, out, err);
}

bool LlamaModel::matvec_weight_q8fast(const WeightBlob& wb, const std::vector<int8_t>& q8buf,
                                      float dx, float* y, std::string& err) {
  // Fast path only for Q4_0 2-D with per-block scales; else fall back to f32.
  if (wb.qtype != gguf::Q4_0 || wb.dims.size() != 2) return false;
  size_t n_cols = (size_t)wb.dims[0], n_rows = (size_t)wb.dims[1];
  size_t row_blk = (n_cols / 32) * 18;
  if (wb.raw.size() < n_rows * row_blk) { err = "Q4_0 payload truncated"; return false; }
  if (q8buf.size() < n_cols) { err = "q8buf smaller than n_cols (caller must quantize x first)"; return false; }
  // Caller quantizes x once; here we consume q8buf+dx with per-block scales.
  // Inner 32-lane dot is integer (qq-8)*q8 via AVX2 maddubs when available.
  pool_.parallel_for(n_rows, [&](size_t r) {
    const uint8_t* rp = wb.raw.data() + r * row_blk;
    // Per-block scales extracted on the fly (18B blocks, no temp alloc).
    // Inner 32-dot is integer (qq-8)*q8 via AVX2 maddubs (x86) / exact scalar
    // (ARM) — same Jacob-affine form as quant::q4_blockscales_dot, one FP
    // scale per block outside the hot loop.
    double acc = 0.0;
    size_t nb = n_cols / 32;
    for (size_t b = 0; b < nb; ++b) {
      uint16_t h; std::memcpy(&h, rp + b * 18, 2);
      float d = quant::fp16_to_fp32(h);
      const uint8_t* qs = rp + b * 18 + 2;
      const int8_t* xb = q8buf.data() + b * 32;
      // RAW GGML Q4_0 block (split-halves): elements 0..15 = low nibbles of
      // qs[0..15], elements 16..31 = high nibbles (dequantize_row_q4_0).
      uint8_t wraw[16];
      for (int i = 0; i < 16; ++i) wraw[i] = (uint8_t)(qs[i] & 0x0F);
      uint8_t wraw2[16];
      for (int i = 0; i < 16; ++i) wraw2[i] = (uint8_t)((qs[i] >> 4) & 0x0F);
      long long sum_wx = 0, sum_x = 0;
#if defined(__AVX2__)
      {
        __m128i w0 = _mm_loadu_si128((const __m128i*)wraw);
        __m128i x0 = _mm_loadu_si128((const __m128i*)xb);
        __m128i w1 = _mm_loadu_si128((const __m128i*)wraw2);
        __m128i x1 = _mm_loadu_si128((const __m128i*)(xb + 16));
        const __m128i ones = _mm_set1_epi16(1);
        __m128i p0 = _mm_maddubs_epi16(w0, x0);
        __m128i p1 = _mm_maddubs_epi16(w1, x1);
        __m128i s0 = _mm_madd_epi16(p0, ones);
        __m128i s1 = _mm_madd_epi16(p1, ones);
        __m128i s = _mm_add_epi32(s0, s1);
        __m128i sh = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        __m128i sl = _mm_add_epi32(sh, _mm_shuffle_epi32(sh, 0xB1));
        sum_wx += _mm_cvtsi128_si32(sl);
      }
#else
      // Scalar fallback (ARM): same split-halves mapping as the AVX2 path
      // above (elements 0..15 = low nibbles, 16..31 = high nibbles).
      for (int i = 0; i < 16; ++i) {
        sum_wx += (long long)(int)(qs[i] & 0x0F) * (long long)xb[i];
        sum_wx += (long long)(int)((qs[i] >> 4) & 0x0F) * (long long)xb[16 + i];
      }
#endif
      for (int i = 0; i < 32; ++i) sum_x += (long long)xb[i];
      acc += (double)(sum_wx - 8LL * sum_x) * (double)d;
    }
    y[r] = (float)(acc * (double)dx);
  });
  return true;
}

bool LlamaModel::matvec_weight(const std::string& name, const float* x, size_t n_in, size_t n_out,
                               float* y, std::string& err) {
  auto it = weights_.find(name);
  if (it == weights_.end()) { err = "weight not loaded: " + name; return false; }
  const WeightBlob& wb = it->second;
  // Try Q4 fast path on real activations (not a probe).
  if (wb.qtype == gguf::Q4_0 && wb.dims.size() == 2) {
    float dx = 0;
    if (q8buf_.size() < n_in) q8buf_.assign(n_in, 0);
    quant::quant_f32_to_q8(x, n_in, q8buf_.data(), dx);
    if (matvec_weight_q8fast(wb, q8buf_, dx, y, err)) return true;
    if (!err.empty()) return false;
  }
  // Generic: dequant rows then float dot, parallel over rows.
  // Error propagation via atomic flag (failures inside pool are not swallowed).
  std::atomic<bool> fail{false};
  std::string derr;
  std::mutex derr_mu;
  pool_.parallel_for(n_out, [&](size_t r) {
    if (fail.load(std::memory_order_relaxed)) return;
    std::vector<float> row;
    std::string e;
    if (!quant::dequant_row_to_f32(wb.qtype, wb.raw.data(), wb.raw.size(), n_in, n_out, r, row, e)) {
      bool expected = false;
      if (!fail.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(derr_mu);
        if (derr.empty()) derr = name + ": " + e;
      }
      fail.store(true, std::memory_order_relaxed);
      return;
    }
    double acc = 0.0;
    for (size_t c = 0; c < n_in; ++c) acc += (double)row[c] * (double)x[c];
    y[r] = (float)acc;
  });
  if (fail.load()) { err = derr.empty() ? (name + ": row dequant failed") : derr; return false; }
  return true;
}

bool LlamaModel::decode_step(int32_t token, int pos, std::vector<float>& logits, std::string& err) {
  if (!loaded_) { err = "model not loaded"; return false; }
  size_t N = (size_t)dims_.n_embd;
  size_t NH = (size_t)dims_.n_head, NKV = (size_t)dims_.n_kv, HD = (size_t)dims_.head_dim;
  size_t FF = (size_t)dims_.n_ff;
  // 1) token embedding row.
  std::vector<float> emb;
  if (!get_row("token_embd.weight", (size_t)(token >= 0 ? token : 0), emb, err)) return false;
  std::vector<float> x = emb;  // [N]
  std::vector<float> xn(N), q(NH * HD), k(NKV * HD), v(NKV * HD), att(NH * HD);
  std::vector<float> hb(FF), hb2(FF);
  // Per-layer.
  for (int l = 0; l < dims_.n_layer; ++l) {
    std::string b = "blk." + std::to_string(l) + ".";
    std::vector<float> w;
    err.clear();
    if (!get_row(b + "attn_norm.weight", 0, w, err)) {
      // attn_norm is [N] vector stored as 1-D tensor; read directly.
      auto it = weights_.find(b + "attn_norm.weight");
      if (it == weights_.end()) { err = "missing " + b + "attn_norm.weight"; return false; }
      if (!quant::dequant_to_f32(it->second.qtype, it->second.raw.data(), it->second.raw.size(),
                                 N, w, err)) return false;
    }
    rmsnorm(x.data(), w.data(), N, dims_.norm_eps, xn.data());
    // Q/K/V projections.
    if (!matvec_weight(b + "attn_q.weight", xn.data(), N, NH * HD, q.data(), err)) return false;
    if (!matvec_weight(b + "attn_k.weight", xn.data(), N, NKV * HD, k.data(), err)) return false;
    if (!matvec_weight(b + "attn_v.weight", xn.data(), N, NKV * HD, v.data(), err)) return false;
    // RoPE on Q (all heads) and K (kv heads), YaRN-aware for long context.
    rope_inplace_scaled(q.data(), HD, (size_t)pos, dims_.rope_theta, NH,
                        dims_.rope_scale, dims_.rope_orig_ctx,
                        dims_.rope_beta_fast, dims_.rope_beta_slow);
    rope_inplace_scaled(k.data(), HD, (size_t)pos, dims_.rope_theta, NKV,
                        dims_.rope_scale, dims_.rope_orig_ctx,
                        dims_.rope_beta_fast, dims_.rope_beta_slow);
    // Append K/V to this layer's Q8 cache (separate K/V buffers).
    kv_[(size_t)l].append(k.data(), v.data());
    // Attention per query head with GQA repeat.
    size_t L = (size_t)kv_[(size_t)l].seq_len();
    if (tmp_attn_.size() < L + 2 * HD) tmp_attn_.assign(L + 2 * HD, 0);
    size_t g = NH / (NKV ? NKV : 1);
    if (g == 0) g = 1;
    for (size_t h = 0; h < NH; ++h) {
      size_t kh = h / g;
      if (kh >= NKV) kh = NKV - 1;
      attn_decode_step(kv_[(size_t)l], (int)kh, q.data() + h * HD, tmp_attn_.data(), att.data() + h * HD);
    }
    // Output proj.
    std::vector<float> attn_out(N, 0.0f);
    if (!matvec_weight(b + "attn_output.weight", att.data(), N, N, attn_out.data(), err)) return false;
    for (size_t i = 0; i < N; ++i) x[i] += attn_out[i];
    // FFN SwiGLU.
    err.clear();
    if (!get_row(b + "ffn_norm.weight", 0, w, err)) {
      auto it = weights_.find(b + "ffn_norm.weight");
      if (it == weights_.end()) { err = "missing " + b + "ffn_norm.weight"; return false; }
      if (!quant::dequant_to_f32(it->second.qtype, it->second.raw.data(), it->second.raw.size(),
                                 N, w, err)) return false;
    }
    rmsnorm(x.data(), w.data(), N, dims_.norm_eps, xn.data());
    if (!matvec_weight(b + "ffn_gate.weight", xn.data(), N, FF, hb.data(), err)) return false;
    if (!matvec_weight(b + "ffn_up.weight", xn.data(), N, FF, hb2.data(), err)) return false;
    for (size_t i = 0; i < FF; ++i) hb[i] = (hb[i] / (1.0f + std::exp(-hb[i]))) * hb2[i];
    std::vector<float> ffn_out(N, 0.0f);
    if (!matvec_weight(b + "ffn_down.weight", hb.data(), FF, N, ffn_out.data(), err)) return false;
    for (size_t i = 0; i < N; ++i) x[i] += ffn_out[i];
  }
  // Final norm + lm_head.
  {
    auto it = weights_.find("output_norm.weight");
    if (it != weights_.end()) {
      std::vector<float> w;
      if (!quant::dequant_to_f32(it->second.qtype, it->second.raw.data(), it->second.raw.size(),
                                 N, w, err)) return false;
      rmsnorm(x.data(), w.data(), N, dims_.norm_eps, xn.data());
    } else {
      xn = x;
    }
  }
  size_t V = (size_t)(dims_.vocab > 0 ? dims_.vocab : 0);
  logits.assign(V, 0.0f);
  if (V == 0) { err = "empty vocab"; return false; }
  if (!matvec_weight("output.weight", xn.data(), N, V, logits.data(), err)) {
    // Tied embeddings fallback.
    if (!matvec_weight("token_embd.weight", xn.data(), N, V, logits.data(), err)) return false;
  }
  return true;
}

bool LlamaModel::generate(const std::vector<int32_t>& prompt, int max_new, const SamplerConfig& scfg,
                          std::vector<int32_t>& out_ids, std::string& err) {
  out_ids.clear();
  if (!loaded_) { err = "model not loaded"; return false; }
  if (prompt.empty()) { err = "empty prompt"; return false; }
  reset_kv();
  uint64_t rng = scfg.seed ? scfg.seed : 0x9E3779B97F4A7C15ULL;
  std::vector<float> logits;
  // Prefill: feed all prompt tokens (KV grows); only last logits used.
  for (size_t i = 0; i < prompt.size(); ++i) {
    if (!decode_step(prompt[i], (int)i, logits, err)) return false;
  }
  int32_t cur = sample_logits(logits.data(), logits.size(), scfg, rng);
  out_ids.push_back(cur);
  for (int t = 1; t < max_new; ++t) {
    if (cur == tok_.eos) break;
    if (!decode_step(cur, (int)(prompt.size() + (size_t)t - 1), logits, err)) return false;
    cur = sample_logits(logits.data(), logits.size(), scfg, rng);
    out_ids.push_back(cur);
  }
  return true;
}

}  // namespace ours
