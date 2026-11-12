// main.cpp — CLI (info|gemm|kv|h4|spec|run|roofline).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "bench.h"
#include "engine.h"
#include "gguf.h"
#include "kv_q8.h"
#include "lut_gemm.h"
#include "model.h"
#include "ngram_spec.h"
#include "quant.h"

namespace ours {

static double now_s() {
  using clock = std::chrono::steady_clock;
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
             .count() *
         1e-9;
}

int run_roofline(double bw_gbs, double gguf_bytes, double tg_meas) {
  // bw_gbs is decimal GB/s (1e9 B/s); plafon = bytes_per_s / bytes_per_token.
  // 40-60% of STREAM is healthy, <30% means a config problem (threads, swap,
  // offload, KV).
  constexpr double kHealthyLo = 0.40, kHealthyHi = 0.60, kMisconfig = 0.30;
  constexpr double kBytesPerGBDecimal = 1e9;
  if (bw_gbs <= 0 || gguf_bytes <= 0 || tg_meas <= 0) {
    std::printf("ROOFLINE: invalid args (need --bw GB/s --bytes GGUF --tg meas).\n");
    return 2;
  }
  double plafon = bw_gbs * kBytesPerGBDecimal / gguf_bytes;
  double rasio = tg_meas / plafon;
  std::printf("ROOFLINE: bw=%.2fGB/s(decimal) bytes=%.0f plafon=%.2f tok/s meas=%.2f ratio=%.2f\n",
              bw_gbs, gguf_bytes, plafon, tg_meas, rasio);
  if (rasio >= kHealthyLo && rasio <= kHealthyHi) {
    std::printf("VERDICT: HEALTHY (40-60%%). Engine near roofline; blame model size.\n");
    return 0;
  }
  if (rasio < kMisconfig) {
    std::printf("VERDICT: MISCONFIGURED (<30%%). Fix threads/swap/-b first; NOT a reason to switch engines.\n");
    return 1;
  }
  std::printf("VERDICT: GRAY. Further triage (partial offload? KV?).\n");
  return 1;
}

int run_info_real(const std::string& model_path) {
  gguf::ModelInfo mi;
  std::string err;
  if (!gguf::open(model_path, mi, err)) {
    std::printf("INFO: %s\n", err.c_str());
    return 2;
  }
  std::printf("INFO: file=%s arch=%s version=%u tensors=%zu data_start=%llu file_size=%llu\n",
              model_path.c_str(), mi.arch.empty() ? "?" : mi.arch.c_str(), mi.version,
              mi.tensors.size(), (unsigned long long)mi.data_start,
              (unsigned long long)mi.file_size);
  std::map<uint32_t, int> counts;
  for (auto& t : mi.tensors) counts[t.type]++;
  for (auto& kv : counts)
    std::printf("INFO: type %s x%d\n", gguf::type_name(kv.first).c_str(), kv.second);
  std::string arch = mi.arch.empty() ? "llama" : mi.arch;
  auto gi = [&](const std::string& k) -> long long {
    auto it = mi.u32_meta.find(arch + "." + k);
    if (it != mi.u32_meta.end()) return it->second;
    auto j = mi.i32_meta.find(arch + "." + k);
    if (j != mi.i32_meta.end()) return j->second;
    return -1;
  };
  std::printf("INFO: arch %s block=%lld embd=%lld heads=%lld kv=%lld ff=%lld vocab_tok=%zu\n",
              arch.c_str(), gi("block_count"), gi("embedding_length"),
              gi("attention.head_count"), gi("attention.head_count_kv"),
              gi("feed_forward_length"), mi.tokens.size());
  for (size_t i = 0; i < mi.tensors.size() && i < 40; ++i) {
    const auto& t = mi.tensors[i];
    std::printf("INFO: [%zu] %s type=%s ndim=%zu n=%llu bytes=%llu\n", i, t.name.c_str(),
                gguf::type_name(t.type).c_str(), t.dims.size(),
                (unsigned long long)t.n_elems,
                (unsigned long long)gguf::tensor_nbytes(t));
  }
  return 0;
}

// Real activation for H1/H4: first token_embd row (real model vector), Q8-quantized.
static bool real_act_from_model(const std::string& model_path, std::vector<int8_t>& q8, float& dx,
                                size_t n_cols, std::string& err) {
  gguf::ModelInfo mi;
  if (!gguf::open(model_path, mi, err)) return false;
  size_t idx = mi.tensors.size();
  if (!gguf::find_tensor(mi, "token_embd.weight", idx)) { err = "missing token_embd.weight"; return false; }
  std::vector<uint8_t> raw;
  if (!gguf::read_tensor_bytes(model_path, mi, idx, raw, err)) return false;
  const auto& ti = mi.tensors[idx];
  if (ti.dims.size() != 2) { err = "token_embd not 2-D"; return false; }
  std::vector<float> row;
  if (!quant::dequant_row_to_f32(ti.type, raw.data(), raw.size(), (size_t)ti.dims[0],
                                 (size_t)ti.dims[1], 0, row, err)) return false;
  if (row.size() < n_cols) { err = "embd narrower than GEMM cols"; return false; }
  q8.assign(n_cols, 0);
  quant::quant_f32_to_q8(row.data(), n_cols, q8.data(), dx);
  return true;
}

int run_gemm_real(const std::string& model_path, int threads, int repeats) {
  gguf::ModelInfo mi;
  std::string err;
  if (!gguf::open(model_path, mi, err)) {
    std::printf("GEMM: gguf open failed: %s\n", err.c_str());
    return 2;
  }
  size_t ti = mi.tensors.size();
  for (size_t i = 0; i < mi.tensors.size(); ++i) {
    const auto& t = mi.tensors[i];
    if (t.type == gguf::Q4_0 && t.dims.size() == 2) { ti = i; break; }
  }
  if (ti >= mi.tensors.size()) {
    std::printf("GEMM: no Q4_0 2-D tensor; full-model path covers K-quants via ours run.\n");
    std::printf("GEMM: use 'ours run --model' for Q4_K_M models.\n");
    return 2;
  }
  Engine eng;
  if (!eng.open_gguf(model_path, err)) {
    std::printf("GEMM: reopen failed: %s\n", err.c_str());
    return 2;
  }
  eng.pool().reset(threads, true);
  const auto& t = mi.tensors[ti];
  size_t cols = (size_t)t.dims[0];
  size_t rows = (size_t)t.dims[1];
  size_t use_rows = rows < 32 ? rows : 32;
  std::vector<std::vector<uint8_t>> trows;
  std::vector<std::vector<float>> srows;
  std::vector<uint8_t> raw_all;
  if (!gguf::read_tensor_bytes(model_path, mi, ti, raw_all, err)) {
    std::printf("GEMM: read tensor failed: %s\n", err.c_str());
    return 2;
  }
  size_t row_bytes = (cols / 32) * 18;
  for (size_t r = 0; r < use_rows; ++r) {
    std::vector<uint8_t> nib;
    std::vector<float> sc;
    if (!gguf::q4_0_to_prod(raw_all.data() + r * row_bytes, row_bytes, nib, sc)) {
      std::printf("GEMM: row conversion failed at row %zu\n", r);
      return 2;
    }
    trows.push_back(std::move(nib));
    srows.push_back(std::move(sc));
  }
  std::vector<int8_t> q8;
  float dx = 0.03f;
  if (!real_act_from_model(model_path, q8, dx, cols, err)) {
    std::printf("GEMM: real act failed: %s\n", err.c_str());
    return 2;
  }
  for (size_t r = 0; r < use_rows; ++r) {
    float ref = quant::q4_blockscales_dot(cols, trows[r].data(), srows[r].data(), srows[r].size(), q8.data(), dx);
    float got = 0;
    {
      std::vector<std::vector<uint8_t>> one{trows[r]};
      std::vector<std::vector<float>> one_s{srows[r]};
      std::vector<float> o;
      eng.gemv_rows(one, one_s, q8.data(), dx, o);
      got = o[0];
    }
    double diff = std::fabs((double)ref - (double)got);
    if (diff > 1e-3 * (1.0 + std::fabs((double)ref))) {
      std::printf("GEMM row %zu CHECKSUM-FAIL ref=%.6f got=%.6f diff=%.6g\n", r, ref, got, diff);
      return 1;
    }
  }
  volatile float sink = 0;
  double t0 = now_s();
  for (int k = 0; k < repeats; ++k)
    for (size_t r = 0; r < use_rows; ++r)
      sink += quant::q4_blockscales_dot(cols, trows[r].data(), srows[r].data(), srows[r].size(), q8.data(), dx);
  double t1 = now_s();
  for (int k = 0; k < repeats; ++k) {
    std::vector<float> o;
    eng.gemv_rows(trows, srows, q8.data(), dx, o);
    for (auto v : o) sink += v;
  }
  double t2 = now_s();
  (void)sink;
  double single = (t1 - t0) / (repeats > 0 ? repeats : 1) / (use_rows ? use_rows : 1);
  double pooled = (t2 - t1) / (repeats > 0 ? repeats : 1) / (use_rows ? use_rows : 1);
  std::printf("GEMM-REAL tensor=%s rows=%zu cols=%zu threads=%d\n", t.name.c_str(), use_rows, cols, threads);
  std::printf("GEMM-REAL single=%.3fus pooled=%.3fus speedup=%.2f CHECKSUM-OK\n",
              single * 1e6, pooled * 1e6, single / (pooled > 0 ? pooled : 1e-9));
  std::printf("NOTE: act = token_embd row 0 slice (real model vector, Q8).\n");
  return 0;
}

static int parse_depths(const std::string& s, std::vector<int>& out) {
  out.clear();
  std::string cur;
  for (size_t i = 0; i <= s.size(); ++i) {
    char c = i < s.size() ? s[i] : ',';
    if (c == ',' || c == ' ' || c == '\0') {
      if (!cur.empty()) {
        int mult = 1, v = 0;
        std::string num = cur;
        if (num.size() > 1 && (num.back() == 'K' || num.back() == 'k')) {
          mult = 1024;
          num.pop_back();
        }
        v = std::atoi(num.c_str());
        if (v > 0) out.push_back(v * mult);
        cur.clear();
      }
    } else cur.push_back(c);
  }
  if (out.empty()) out = {2048, 8192, 16384, 32768};
  return 0;
}

int run_kv_real(const std::string& model_path, const std::string& depths) {
  gguf::ModelInfo mi;
  std::string err;
  if (!gguf::open(model_path, mi, err)) {
    std::printf("KV: gguf open failed: %s\n", err.c_str());
    return 2;
  }
  std::string arch = mi.arch.empty() ? "llama" : mi.arch;
  auto getu = [&](const std::string& k) -> int {
    auto f = mi.u32_meta.find(k);
    if (f != mi.u32_meta.end()) return (int)f->second;
    auto g = mi.i32_meta.find(k);
    if (g != mi.i32_meta.end()) return (int)g->second;
    return -1;
  };
  int n_head = 8, head_dim = 128;
  int hk = getu(arch + ".attention.head_count_kv");
  int emb = getu(arch + ".embedding_length");
  int hq = getu(arch + ".attention.head_count");
  if (hk > 0) n_head = hk;
  if (hq > 0 && emb > 0 && (emb % hq) == 0) head_dim = emb / hq;
  std::vector<int> Ls;
  parse_depths(depths, Ls);
  // K/V vectors cycle rows of the model's own attn_k/attn_v weights
  // (dequantized to f32). Reports wire vs resident bytes plus the
  // quantize-dequantize roundtrip error (max-abs + MSE vs f32).
  size_t ik = mi.tensors.size(), iv = mi.tensors.size();
  bool has_k = gguf::find_tensor(mi, "blk.0.attn_k.weight", ik);
  bool has_v = gguf::find_tensor(mi, "blk.0.attn_v.weight", iv);
  if (!has_k || !has_v) {
    std::printf("KV: REFUSED (needs blk.0.attn_k/v.weight).\n");
    return 2;
  }
  std::vector<uint8_t> rawk, rawv;
  if (!gguf::read_tensor_bytes(model_path, mi, ik, rawk, err) ||
      !gguf::read_tensor_bytes(model_path, mi, iv, rawv, err)) {
    std::printf("KV: read attn_k/v failed: %s\n", err.c_str());
    return 2;
  }
  const auto& tk = mi.tensors[ik];
  const auto& tv = mi.tensors[iv];
  if (tk.dims.size() != 2 || tv.dims.size() != 2) {
    std::printf("KV: REFUSED (attn_k/v not 2-D in this arch).\n");
    return 2;
  }
  size_t kc = (size_t)tk.dims[0], kr = (size_t)tk.dims[1];
  size_t vc = (size_t)tv.dims[0], vr = (size_t)tv.dims[1];
  size_t need = (size_t)n_head * (size_t)head_dim;
  std::printf("KV-REAL arch=%s heads_kv=%d head_dim=%d (K/V from blk.0 weights)\n",
              arch.c_str(), n_head, head_dim);
  for (int L : Ls) {
    KvCacheQ8 kv(n_head, head_dim);
    std::vector<float> krow(need), vrow(need);
    double max_abs = 0.0, se_sum = 0.0;
    long long se_n = 0;
    for (int pp = 0; pp < L; ++pp) {
      // Cycle weight rows deterministically; slice/pad to [n_head*head_dim].
      std::vector<float> wk, wv;
      std::string e2;
      size_t rk = (size_t)pp % (kr ? kr : 1), rv = (size_t)pp % (vr ? vr : 1);
      if (!quant::dequant_row_to_f32(tk.type, rawk.data(), rawk.size(), kc, kr, rk, wk, e2) ||
          !quant::dequant_row_to_f32(tv.type, rawv.data(), rawv.size(), vc, vr, rv, wv, e2)) {
        std::printf("KV: dequant row failed at p=%d: %s\n", pp, e2.c_str());
        return 1;
      }
      for (size_t i = 0; i < need; ++i) {
        krow[i] = wk[i % wk.size()];
        vrow[i] = wv[i % wv.size()];
      }
      kv.append(krow.data(), vrow.data());
      // Roundtrip error on this token: reload and compare to source.
      const auto& b = kv.blocks.back();
      int blen = b.len;
      std::vector<float> rkq((size_t)head_dim), rvq((size_t)head_dim);
      for (int h = 0; h < n_head; ++h) {
        b.load_k_row(h, blen - 1, rkq.data());
        b.load_v_row(h, blen - 1, rvq.data());
        for (int d = 0; d < head_dim; ++d) {
          size_t j = (size_t)h * (size_t)head_dim + (size_t)d;
          double ek = (double)rkq[d] - (double)krow[j];
          double ev = (double)rvq[d] - (double)vrow[j];
          double a = ek < 0 ? -ek : ek, c = ev < 0 ? -ev : ev;
          if (a > max_abs) max_abs = a;
          if (c > max_abs) max_abs = c;
          se_sum += ek * ek + ev * ev;
          se_n += 2;
        }
      }
    }
    std::size_t wire = kv.bytes_wire_total();
    std::size_t res = kv.bytes_total();
    std::size_t fp16 = kv.bytes_fp16_same_len();
    double mse = se_n ? se_sum / (double)se_n : 0.0;
    std::printf("KV-REAL L=%d wire=%zu resident=%zu fp16=%zu wire_ratio=%.2f resident_ratio=%.2f maxabs=%.4f mse=%.6f\n",
                L, wire, res, fp16,
                (double)fp16 / (double)(wire ? wire : 1),
                (double)fp16 / (double)(res ? res : 1), max_abs, mse);
  }
  std::printf("NOTE: H2 needs wire_ratio~=2 with small maxabs/MSE at <=8K plus\n"
              "      same-box decode timing. Bytes alone are not a pass.\n");
  return 0;
}

int run_h4_real(const std::string& model_path, int threads) {
  // H4: tiled-32 + persistent pool vs serial naive, same integer-dot on the
  // same Q4_0 rows + token_embd activation.
  gguf::ModelInfo mi;
  std::string err;
  if (!gguf::open(model_path, mi, err)) {
    std::printf("H4: gguf open failed: %s\n", err.c_str());
    return 2;
  }
  size_t ti = mi.tensors.size();
  for (size_t i = 0; i < mi.tensors.size(); ++i) {
    const auto& t = mi.tensors[i];
    if (t.type == gguf::Q4_0 && t.dims.size() == 2) { ti = i; break; }
  }
  if (ti >= mi.tensors.size()) {
    std::printf("H4: no Q4_0 2-D tensor in %s\n", model_path.c_str());
    return 2;
  }
  Engine eng;
  if (!eng.open_gguf(model_path, err)) {
    std::printf("H4: reopen failed: %s\n", err.c_str());
    return 2;
  }
  eng.pool().reset(threads, true);
  const auto& t = mi.tensors[ti];
  size_t cols = (size_t)t.dims[0];
  size_t rows = (size_t)t.dims[1];
  if (cols % 32 != 0) {
    std::printf("H4: Q4_0 cols not multiple of 32\n");
    return 2;
  }
  size_t m = rows < 128 ? rows : 128;
  std::vector<uint8_t> raw_all;
  if (!gguf::read_tensor_bytes(model_path, mi, ti, raw_all, err)) {
    std::printf("H4: read failed: %s\n", err.c_str());
    return 2;
  }
  size_t row_bytes = (cols / 32) * 18;
  std::size_t nb = (cols + 31) / 32;
  std::vector<std::vector<uint8_t>> trows;
  std::vector<std::vector<float>> srows;
  for (size_t r = 0; r < m; ++r) {
    std::vector<uint8_t> nib;
    std::vector<float> sc;
    if (!gguf::q4_0_to_prod(raw_all.data() + r * row_bytes, row_bytes, nib, sc)) return 2;
    trows.push_back(std::move(nib));
    srows.push_back(std::move(sc));
  }
  std::vector<int8_t> q8;
  float dx = 0.03f;
  if (!real_act_from_model(model_path, q8, dx, cols, err)) {
    std::printf("H4: real act failed: %s\n", err.c_str());
    return 2;
  }
  // Tiled arena: 64B-aligned, best-effort huge-page advice (logged, not required).
  std::vector<const uint8_t*> rowptrs(m);
  for (size_t r = 0; r < m; ++r) rowptrs[r] = trows[r].data();
  std::size_t arena_bytes = q4_tiled_bytes(m, cols);
  uint8_t* arena = q4_tiled_alloc(arena_bytes);
  if (!arena) {
    std::printf("H4: arena alloc failed (%zu bytes)\n", arena_bytes);
    return 2;
  }
  pack_q4_tiled_mat(m, cols, rowptrs.data(), arena);
  std::vector<float> scales_tiled(m * nb);
  for (size_t r = 0; r < m; ++r)
    for (std::size_t b = 0; b < nb; ++b) scales_tiled[r * nb + b] = srows[r][b];
  bool hp = advise_hugepages(arena, arena_bytes);
  std::vector<float> ref(m);
  for (size_t r = 0; r < m; ++r)
    ref[r] = quant::q4_blockscales_dot(cols, trows[r].data(), srows[r].data(), srows[r].size(), q8.data(), dx);
  const int REPS = 10;
  volatile float sink = 0;
  double t0 = now_s();
  for (int k = 0; k < REPS; ++k) {
    for (size_t r = 0; r < m; ++r)
      sink += quant::q4_blockscales_dot(cols, trows[r].data(), srows[r].data(), srows[r].size(), q8.data(), dx);
  }
  double t1 = now_s();
  for (int k = 0; k < REPS; ++k) {
    std::vector<float> tmp(m);
    q4_blockscales_gemv_tiled_pool(m, cols, arena, scales_tiled.data(), q8.data(), dx, tmp.data(), eng.pool());
    for (auto v : tmp) sink += v;
  }
  double t2 = now_s();
  (void)sink;
  std::vector<float> chk(m);
  q4_blockscales_gemv_tiled_pool(m, cols, arena, scales_tiled.data(), q8.data(), dx, chk.data(), eng.pool());
  for (size_t r = 0; r < m; ++r) {
    double tol = 1e-3 * (1.0 + std::fabs((double)ref[r]));
    if (std::fabs((double)chk[r] - (double)ref[r]) > tol) {
      std::printf("H4-REAL CHECKSUM-FAIL row %zu\n", r);
      q4_tiled_free(arena);
      return 1;
    }
  }
  double s_us = (t1 - t0) / REPS * 1e6, p_us = (t2 - t1) / REPS * 1e6;
  std::printf("H4-REAL tensor=%s m=%zu n=%zu threads=%d arena=%zuB hugepages=%s single=%.1fus tiled_pool=%.1fus speedup=%.2f CHECKSUM-OK\n",
              t.name.c_str(), m, cols, threads, arena_bytes, hp ? "advised" : "unsupported/failed",
              s_us, p_us, s_us / (p_us > 0 ? p_us : 1e-9));
  q4_tiled_free(arena);
  return 0;
}

int run_spec_real(const std::string& model_path, const std::string& prompts_path, int K) {
  // H3 on tokenizer ids: loads the GGUF tokenizer via LlamaModel and encodes
  // each prompt line.
  if (model_path.empty()) {
    std::printf("SPEC: REFUSED (needs --model for the GGUF tokenizer).\n");
    return 2;
  }
  LlamaModel lm;
  std::string err;
  if (!lm.load(model_path, 1, false, err)) {
    std::printf("SPEC: model load failed: %s\n", err.c_str());
    return 2;
  }
  FILE* f = std::fopen(prompts_path.c_str(), "r");
  if (!f) {
    std::printf("SPEC: cannot open prompts file %s\n", prompts_path.c_str());
    return 2;
  }
  char buf[8192];
  std::vector<std::string> lines;
  while (std::fgets(buf, sizeof(buf), f)) lines.emplace_back(buf);
  std::fclose(f);
  std::unordered_set<std::string> seen;
  size_t dup = 0;
  for (auto& ln : lines) {
    if (!seen.insert(ln).second) ++dup;
  }
  double ratio = lines.empty() ? 0.0 : (double)dup / (double)lines.size();
  std::printf("SPEC-REAL: model=%s prompts=%s lines=%zu dup_ratio=%.2f\n",
              model_path.c_str(), prompts_path.c_str(), lines.size(), ratio);
  if (ratio > 0.20) {
    std::printf("VERDICT: REJECTED (duplicate>20%%).\n");
    return 1;
  }
  std::vector<std::vector<int32_t>> toks;
  for (auto& ln : lines) {
    // Strip trailing newline only; keep all other whitespace (tokenizer-real).
    std::string t = ln;
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    if (t.empty()) continue;
    std::vector<int32_t> v;
    lm.tok().encode(t, v, false);
    if (!v.empty()) toks.push_back(std::move(v));
  }
  if (toks.empty()) {
    std::printf("SPEC: REFUSED (no encodable prompts; check tokenizer pieces).\n");
    return 2;
  }
  NgramDraft ng;
  for (auto& ln : toks) ng.observe(ln.data(), ln.size());
  long covered = 0, sum_m = 0;
  std::vector<int32_t> db(8);
  for (auto& ln : toks) {
    int m = ng.draft(ln.data(), ln.size(), K > 8 ? 8 : K, db.data());
    covered += (m > 0);
    sum_m += m;
  }
  double nl = toks.empty() ? 1.0 : (double)toks.size();
  std::printf("SPEC-REAL: cover_frac=%.3f avg_draft=%.2f ids=REAL-BPE (coverage only; "
              "accept/speedup need a bound target model)\n",
              (double)covered / nl, (double)sum_m / nl);
  return 0;
}

int run_generate_real(const std::string& model_path, const std::string& prompt,
                      int max_new, float temp, int threads) {
  LlamaModel m;
  std::string err;
  if (!m.load(model_path, threads, true, err)) {
    std::printf("RUN: load failed: %s\n", err.c_str());
    return 2;
  }
  std::vector<int32_t> pin;
  m.tok().encode(prompt, pin, m.tok().add_bos);
  std::printf("RUN: prompt_tokens=%zu vocab=%d layers=%d embd=%d heads=%d kv=%d\n",
              pin.size(), m.dims().vocab, m.dims().n_layer, m.dims().n_embd,
              m.dims().n_head, m.dims().n_kv);
  SamplerConfig sc;
  sc.temp = temp;
  std::vector<int32_t> out;
  double t0 = now_s();
  if (!m.generate(pin, max_new, sc, out, err)) {
    std::printf("RUN: generate failed: %s\n", err.c_str());
    return 1;
  }
  double t1 = now_s();
  std::string text = m.tok().decode(out);
  double dt = t1 - t0;
  std::printf("RUN: new_tokens=%zu time=%.2fs tok/s=%.2f\n", out.size(), dt,
              (double)out.size() / (dt > 0 ? dt : 1e-9));
  std::printf("RUN-OUT-BEGIN\n%s\nRUN-OUT-END\n", text.c_str());
  return 0;
}

}  // namespace ours

static void usage() {
  std::printf("Usage: ours {info|gemm|kv|h4|spec|run|roofline} ...\n"
              "  ours info --model X.gguf\n"
              "  ours gemm --model X.gguf [--threads T] [--repeats R]\n"
              "  ours kv   --model X.gguf [--depths 2K,8K,16K,32K]\n"
              "  ours h4   --model X.gguf [--threads T]\n"
              "  ours spec --model X.gguf --prompts file.txt [--K 8]\n"
              "  ours run  --model X.gguf --prompt \"...\" [--max-new N] [--temp T] [--threads H]\n"
              "  ours roofline --bw GB/s --bytes N --tg T\n");
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  std::string cmd = argv[1];
  auto get = [&](const char* k, const char* dflt) -> std::string {
    for (int i = 2; i < argc; ++i) {
      if (std::strcmp(argv[i], k) == 0 && i + 1 < argc) return argv[i + 1];
    }
    return dflt ? dflt : "";
  };
  if (cmd == "roofline") {
    double bw = std::atof(get("--bw", "0").c_str());
    double by = std::atof(get("--bytes", "0").c_str());
    double tg = std::atof(get("--tg", "0").c_str());
    return ours::run_roofline(bw, by, tg);
  }
  if (cmd == "info") {
    std::string mp = get("--model", "");
    if (mp.empty()) { usage(); return 2; }
    return ours::run_info_real(mp);
  }
  if (cmd == "gemm") {
    std::string mp = get("--model", "");
    if (mp.empty()) { usage(); return 2; }
    int th = std::atoi(get("--threads", "4").c_str());
    int rp = std::atoi(get("--repeats", "5").c_str());
    return ours::run_gemm_real(mp, th, rp);
  }
  if (cmd == "kv") {
    std::string mp = get("--model", "");
    if (mp.empty()) { usage(); return 2; }
    return ours::run_kv_real(mp, get("--depths", "2K,8K,16K,32K"));
  }
  if (cmd == "h4") {
    std::string mp = get("--model", "");
    if (mp.empty()) { usage(); return 2; }
    int th = std::atoi(get("--threads", "4").c_str());
    return ours::run_h4_real(mp, th);
  }
  if (cmd == "spec") {
    std::string mp = get("--model", "");
    std::string p = get("--prompts", "");
    if (p.empty() || mp.empty()) { usage(); return 2; }
    int K = std::atoi(get("--K", "8").c_str());
    return ours::run_spec_real(mp, p, K);
  }
  if (cmd == "run") {
    std::string mp = get("--model", "");
    std::string pr = get("--prompt", "");
    if (mp.empty() || pr.empty()) { usage(); return 2; }
    int mx = std::atoi(get("--max-new", "64").c_str());
    float tp = (float)std::atof(get("--temp", "0").c_str());
    int th = std::atoi(get("--threads", "4").c_str());
    return ours::run_generate_real(mp, pr, mx, tp, th);
  }
  usage();
  return 2;
}
