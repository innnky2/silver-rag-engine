// gguf.cpp — GGUFv3 reader, full KV spec.
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ours {
namespace gguf {

namespace {
bool rd(FILE* f, void* p, size_t n) { return std::fread(p, 1, n, f) == n; }
bool rd_u8(FILE* f, uint8_t& v) { return rd(f, &v, 1); }
bool rd_u32(FILE* f, uint32_t& v) {
  uint8_t b[4];
  if (!rd(f, b, 4)) return false;
  v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
      ((uint32_t)b[3] << 24);
  return true;
}
bool rd_i32(FILE* f, int32_t& v) { uint32_t u; if (!rd_u32(f, u)) return false; v = (int32_t)u; return true; }
bool rd_u64(FILE* f, uint64_t& v) {
  uint8_t b[8];
  if (!rd(f, b, 8)) return false;
  v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)b[i] << (8 * i);
  return true;
}
bool rd_f32(FILE* f, float& v) { uint32_t u; if (!rd_u32(f, u)) return false; std::memcpy(&v, &u, 4); return true; }
bool rd_f64(FILE* f, double& v) {
  uint8_t b[8];
  if (!rd(f, b, 8)) return false;
  uint64_t u = 0;
  for (int i = 0; i < 8; ++i) u |= (uint64_t)b[i] << (8 * i);
  std::memcpy(&v, &u, 8);
  return true;
}
bool rd_str(FILE* f, std::string& s) {
  uint64_t n = 0;
  if (!rd_u64(f, n)) return false;
  if (n > (64ULL << 20)) return false;
  s.resize((size_t)n);
  if (n && !rd(f, &s[0], (size_t)n)) return false;
  return true;
}
uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
float fp16_to_fp32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s << 31;
    else {
      e = 1;
      while (!(m & 0x400)) { m <<= 1; --e; }
      m &= 0x3FF;
      f = (s << 31) | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) {
    f = (s << 31) | (0xFF << 23) | (m << 13);
  } else {
    f = (s << 31) | ((e + 112) << 23) | (m << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}
uint64_t file_size_of(FILE* f) {
#if defined(_WIN32)
  _fseeki64(f, 0, SEEK_END);
  __int64 s = _ftelli64(f);
  _fseeki64(f, 0, SEEK_SET);
  return s < 0 ? 0 : (uint64_t)s;
#else
  fseeko(f, 0, SEEK_END);
  off_t s = ftello(f);
  fseeko(f, 0, SEEK_SET);
  return s < 0 ? 0 : (uint64_t)s;
#endif
}
void file_seek_to(FILE* f, uint64_t off) {
#if defined(_WIN32)
  _fseeki64(f, (long long)off, SEEK_SET);
#else
  fseeko(f, (off_t)off, SEEK_SET);
#endif
}
long file_tell_pos(FILE* f) {
#if defined(_WIN32)
  __int64 p = _ftelli64(f);
  return (long)p;
#else
  return ftell(f);
#endif
}
}  // namespace

std::string type_name(uint32_t t) {
  switch (t) {
    case F32: return "F32";
    case F16: return "F16";
    case Q4_0: return "Q4_0";
    case Q4_1: return "Q4_1";
    case Q5_0: return "Q5_0";
    case Q5_1: return "Q5_1";
    case Q8_0: return "Q8_0";
    case Q8_1: return "Q8_1";
    case Q2_K: return "Q2_K";
    case Q3_K: return "Q3_K";
    case Q4_K: return "Q4_K";
    case Q5_K: return "Q5_K";
    case Q6_K: return "Q6_K";
    case Q8_K: return "Q8_K";
    case IQ2_XXS: return "IQ2_XXS";
    case IQ2_XS: return "IQ2_XS";
    case IQ2_S: return "IQ2_S";
    case IQ3_XXS: return "IQ3_XXS";
    case IQ1_S: return "IQ1_S";
    case IQ4_NL: return "IQ4_NL";
    case IQ3_S: return "IQ3_S";
    case IQ2_M: return "IQ2_M";
    case IQ4_XS: return "IQ4_XS";
    case IQ1_M: return "IQ1_M";
    case BF16: return "BF16";
    default: return "UNKNOWN";
  }
}

bool is_k_quant(uint32_t t) {
  return t == Q2_K || t == Q3_K || t == Q4_K || t == Q5_K || t == Q6_K || t == Q8_K;
}

bool is_iq(uint32_t t) {
  return (t >= IQ2_XXS && t <= IQ2_XS) || (t >= IQ2_S && t <= IQ1_M) || t == BF16;
}

uint64_t tensor_nbytes(const TensorInfo& ti) {
  uint64_t n = 1;
  for (auto d : ti.dims) n *= d;
  switch (ti.type) {
    case F32: return n * 4;
    case F16: return n * 2;
    case BF16: return n * 2;
    case Q4_0: return (n / 32) * (2 + 16);        // d + 16B
    case Q4_1: return (n / 32) * (2 + 2 + 16);      // d + m + 16B (GGML block_q4_1)
    case Q5_0: return (n / 32) * (2 + 4 + 16);      // d + qh + 16B
    case Q5_1: return (n / 32) * (2 + 2 + 4 + 16);  // d + m + qh + 16B (GGML block_q5_1)
    case Q8_0: return (n / 32) * (2 + 32);
    case Q8_1: return (n / 32) * (4 + 32);
    case Q2_K: return (n / 256) * 84;
    case Q3_K: return (n / 256) * 110;
    case Q4_K: return (n / 256) * 144;
    case Q5_K: return (n / 256) * 176;
    case Q6_K: return (n / 256) * 210;
    case Q8_K: return (n / 256) * 292;
    default: return 0;
  }
}

bool find_tensor(const ModelInfo& mi, const std::string& name, size_t& idx) {
  for (size_t i = 0; i < mi.tensors.size(); ++i)
    if (mi.tensors[i].name == name) { idx = i; return true; }
  return false;
}

bool open(const std::string& path, ModelInfo& out, std::string& err) {
  out = ModelInfo{};
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open file: " + path; return false; }
  out.file_size = file_size_of(f);
  if (out.file_size < 32) { err = "file too small for GGUF"; std::fclose(f); return false; }
  char magic[4];
  if (!rd(f, magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    err = "not a GGUF file (bad magic)";
    std::fclose(f);
    return false;
  }
  uint32_t ver = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!rd_u32(f, ver) || !rd_u64(f, n_tensors) || !rd_u64(f, n_kv)) {
    err = "truncated GGUF header";
    std::fclose(f);
    return false;
  }
  out.version = ver;
  if (ver != 3) { err = "unsupported GGUF version (need v3)"; std::fclose(f); return false; }
  if (n_tensors > (1ULL << 20) || n_kv > (1ULL << 20)) {
    err = "implausible header counts";
    std::fclose(f);
    return false;
  }
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    uint32_t tp = 0;
    if (!rd_str(f, key) || !rd_u32(f, tp)) { err = "truncated KV header"; std::fclose(f); return false; }
    switch (tp) {
      case 0: { uint8_t v; if (!rd_u8(f, v)) { err = "truncated KV u8"; std::fclose(f); return false; } break; }
      case 1: { int8_t v; if (!rd(f, &v, 1)) { err = "truncated KV i8"; std::fclose(f); return false; } break; }
      case 2: { uint8_t b[2]; if (!rd(f, b, 2)) { err = "truncated KV u16"; std::fclose(f); return false; } break; }
      case 3: { uint8_t b[2]; if (!rd(f, b, 2)) { err = "truncated KV i16"; std::fclose(f); return false; } break; }
      case 4: { uint32_t v; if (!rd_u32(f, v)) { err = "truncated KV u32"; std::fclose(f); return false; }
                out.u32_meta[key] = v;
                if (key == "general.alignment") out.alignment = v ? v : 32;
                break; }
      case 5: { int32_t v; if (!rd_i32(f, v)) { err = "truncated KV i32"; std::fclose(f); return false; }
                out.i32_meta[key] = v; break; }
      case 6: { float v; if (!rd_f32(f, v)) { err = "truncated KV f32"; std::fclose(f); return false; }
                out.f32_meta[key] = v; break; }
      case 7: { uint8_t b; if (!rd_u8(f, b)) { err = "truncated KV bool"; std::fclose(f); return false; }
                out.bool_meta[key] = (b != 0); break; }
      case 8: { std::string v; if (!rd_str(f, v)) { err = "truncated KV string"; std::fclose(f); return false; }
                out.str_meta[key] = v;
                if (key == "general.architecture") out.arch = v;
                break; }
      case 9: {
        uint32_t at = 0; uint64_t alen = 0;
        if (!rd_u32(f, at) || !rd_u64(f, alen)) { err = "truncated KV array"; std::fclose(f); return false; }
        if (alen > (1ULL << 22)) { err = "KV array too large: " + key; std::fclose(f); return false; }
        bool is_tokens = (key == "tokenizer.ggml.tokens");
        bool is_scores = (key == "tokenizer.ggml.scores");
        bool is_types = (key == "tokenizer.ggml.token_type");
        bool is_merges = (key == "tokenizer.ggml.merges");
        for (uint64_t k = 0; k < alen; ++k) {
          switch (at) {
            case 8: { std::string s; if (!rd_str(f, s)) { err = "truncated str array"; std::fclose(f); return false; }
                      if (is_tokens) out.tokens.push_back(s);
                      else if (is_merges) out.merges.push_back(s);
                      break; }
            case 6: { float v; if (!rd_f32(f, v)) { err = "truncated f32 array"; std::fclose(f); return false; }
                      if (is_scores) out.scores.push_back(v); break; }
            case 4: { uint32_t v; if (!rd_u32(f, v)) { err = "truncated u32 array"; std::fclose(f); return false; } break; }
            case 5: { int32_t v; if (!rd_i32(f, v)) { err = "truncated i32 array"; std::fclose(f); return false; }
                      if (is_types) out.toktypes.push_back(v); break; }
            case 0: case 1: case 7: { uint8_t b; if (!rd_u8(f, b)) { err = "truncated array elem"; std::fclose(f); return false; } break; }
            case 2: case 3: { uint8_t b[2]; if (!rd(f, b, 2)) { err = "truncated array elem"; std::fclose(f); return false; } break; }
            case 10: case 11: { uint8_t b[8]; if (!rd(f, b, 8)) { err = "truncated array elem"; std::fclose(f); return false; } break; }
            case 12: { double d; if (!rd_f64(f, d)) { err = "truncated f64 array"; std::fclose(f); return false; } break; }
            default: { err = "unsupported KV array subtype"; std::fclose(f); return false; }
          }
        }
        break; }
      case 10: { uint64_t v; if (!rd_u64(f, v)) { err = "truncated KV u64"; std::fclose(f); return false; }
                 out.u64_meta[key] = v; break; }
      case 11: { int64_t v; if (!rd(f, &v, 8)) { err = "truncated KV i64"; std::fclose(f); return false; } break; }
      case 12: { double v; if (!rd_f64(f, v)) { err = "truncated KV f64"; std::fclose(f); return false; } break; }
      default: { err = "unsupported KV type"; std::fclose(f); return false; }
    }
  }
  out.tensors.reserve((size_t)n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo ti;
    uint32_t nd = 0;
    if (!rd_str(f, ti.name) || !rd_u32(f, nd)) { err = "truncated tensor info"; std::fclose(f); return false; }
    if (nd > 4) { err = "tensor ndim > 4: " + ti.name; std::fclose(f); return false; }
    ti.n_elems = 1;
    for (uint32_t d = 0; d < nd; ++d) {
      uint64_t v = 0;
      if (!rd_u64(f, v)) { err = "truncated tensor dims"; std::fclose(f); return false; }
      if (v == 0 || v > (1ULL << 32)) { err = "bad tensor dim: " + ti.name; std::fclose(f); return false; }
      ti.dims.push_back(v);
      ti.n_elems *= v;
    }
    if (!rd_u32(f, ti.type) || !rd_u64(f, ti.offset)) { err = "truncated tensor type/off"; std::fclose(f); return false; }
    out.tensors.push_back(ti);
  }
  long hdr = file_tell_pos(f);
  std::fclose(f);
  if (hdr < 0) { err = "ftell failed"; return false; }
  out.data_start = align_up((uint64_t)hdr, out.alignment);
  if (out.data_start > out.file_size) { err = "data_start beyond EOF"; return false; }
  return true;
}

bool read_tensor_bytes(const std::string& path, const ModelInfo& mi,
                       size_t index, std::vector<uint8_t>& out, std::string& err) {
  out.clear();
  if (index >= mi.tensors.size()) { err = "tensor index out of range"; return false; }
  const TensorInfo& ti = mi.tensors[index];
  uint64_t nb = tensor_nbytes(ti);
  if (nb == 0) {
    err = "unsupported tensor type for raw read: " + ti.name + " (" + type_name(ti.type) +
          "); convert model to F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q8_K first";
    return false;
  }
  if (nb > (8ULL << 30)) { err = "tensor too large (>8GiB): " + ti.name; return false; }
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { err = "cannot open file"; return false; }
  uint64_t off = mi.data_start + ti.offset;
  if (off + nb > mi.file_size) { err = "tensor payload out of file bounds: " + ti.name; std::fclose(f); return false; }
  file_seek_to(f, off);
  out.resize((size_t)nb);
  bool ok = rd(f, out.data(), (size_t)nb);
  std::fclose(f);
  if (!ok) { err = "failed reading tensor payload: " + ti.name; out.clear(); return false; }
  return true;
}

bool q4_0_to_prod(const uint8_t* src, size_t src_bytes, std::vector<uint8_t>& q4_nib,
                  std::vector<float>& blk_scales) {
  // GGML split-halves input: byte j holds element j (low) and j+16 (high).
  // Prod layout is sequential even=LOW: element 2k = low nibble of byte k.
  q4_nib.clear();
  blk_scales.clear();
  if (!src || src_bytes == 0 || src_bytes % 18 != 0) return false;
  size_t nblocks = src_bytes / 18;
  size_t n = nblocks * 32;
  q4_nib.resize((n + 1) / 2, 0);
  blk_scales.resize(nblocks);
  auto set_nib = [&](size_t gi, uint8_t nib) {
    if (gi & 1) q4_nib[gi >> 1] |= (uint8_t)((nib & 0x0F) << 4);
    else q4_nib[gi >> 1] = (uint8_t)((q4_nib[gi >> 1] & 0xF0) | (nib & 0x0F));
  };
  for (size_t b = 0; b < nblocks; ++b) {
    const uint8_t* blk = src + b * 18;
    uint16_t h;
    std::memcpy(&h, blk, 2);
    float d = fp16_to_fp32(h);
    blk_scales[b] = d;
    const uint8_t* qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
      set_nib(b * 32 + (size_t)j, (uint8_t)(qs[j] & 0x0F));
      set_nib(b * 32 + (size_t)(16 + j), (uint8_t)((qs[j] >> 4) & 0x0F));
    }
  }
  return true;
}

bool q4_0_to_prod_legacy(const uint8_t* src, size_t src_bytes,
                         std::vector<uint8_t>& q4_nib, float& d_mean) {
  std::vector<float> sc;
  if (!q4_0_to_prod(src, src_bytes, q4_nib, sc)) return false;
  double acc = 0;
  for (auto v : sc) acc += v;
  d_mean = sc.empty() ? 0.0f : (float)(acc / (double)sc.size());
  return true;
}

}  // namespace gguf
}  // namespace ours
