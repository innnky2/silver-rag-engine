// test_gguf.cpp — tests for GGUF reader (needs --model file for file part).
// Build: g++ -std=c++17 -Iinclude tests/test_gguf.cpp src/gguf.cpp -o /tmp/t_gguf
// Run: /tmp/t_gguf --model model.gguf
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "quant.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
  // Pure table checks (no file needed).
  {
    ours::gguf::TensorInfo ti;
    ti.dims = {4096, 32};
    ti.n_elems = 4096ULL * 32ULL;
    ti.type = ours::gguf::Q4_0;
    CHECK(ours::gguf::tensor_nbytes(ti) == (4096ULL * 32ULL / 32ULL) * 18ULL, "Q4_0 bytes");
    ti.type = ours::gguf::Q4_K;
    CHECK(ours::gguf::tensor_nbytes(ti) == (4096ULL * 32ULL / 256ULL) * 144ULL, "Q4_K bytes");
    ti.type = ours::gguf::Q8_K;
    CHECK(ours::gguf::tensor_nbytes(ti) == (4096ULL * 32ULL / 256ULL) * 292ULL, "Q8_K bytes");
    ti.type = ours::gguf::Q4_1;
    CHECK(ours::gguf::tensor_nbytes(ti) == (4096ULL * 32ULL / 32ULL) * 20ULL, "Q4_1 bytes (2+2+16)");
    ti.type = ours::gguf::Q5_1;
    CHECK(ours::gguf::tensor_nbytes(ti) == (4096ULL * 32ULL / 32ULL) * 24ULL, "Q5_1 bytes (2+2+4+16)");
    ti.type = ours::gguf::IQ4_NL;
    CHECK(ours::gguf::tensor_nbytes(ti) == 0, "IQ returns 0");
    CHECK(ours::gguf::is_k_quant(ours::gguf::Q4_K), "is_k_quant");
    CHECK(!ours::gguf::is_k_quant(ours::gguf::Q4_0), "not k_quant");
  }
  {
    // q4_0_to_prod preserves per-block scales (no averaging).
    uint8_t blk[36];
    // block0 d=1.0, block1 d=2.0, nibbles zero.
    blk[0] = 0x00; blk[1] = 0x3C; std::memset(blk + 2, 0x88, 16);
    blk[18] = 0x00; blk[19] = 0x40; std::memset(blk + 20, 0x88, 16);
    std::vector<uint8_t> nib;
    std::vector<float> sc;
    bool ok = ours::gguf::q4_0_to_prod(blk, 36, nib, sc);
    CHECK(ok && sc.size() == 2, "q4_0 prod 2 blocks");
    if (sc.size() == 2) {
      CHECK(std::fabs(sc[0] - 1.0f) < 1e-3 && std::fabs(sc[1] - 2.0f) < 1e-3, "per-block scales kept");
    }
    // Bad size rejected.
    CHECK(!ours::gguf::q4_0_to_prod(blk, 10, nib, sc), "bad size rejected");
    // GGML-exact nibble order, ASYMMETRIC (discriminates even=LOW vs HIGH):
    // only qs[0]=0x21, rest zero -> elements: e0 = 1 (low), e16 = 2 (high),
    // all others 0. Prod sequential even=LOW: byte0 = (e1<<4)|e0 = 0x01,
    // byte8 = (e17<<4)|e16 = 0x02. An even=HIGH swap would give 0x10/0x20.
    {
      uint8_t raw[18];
      raw[0] = 0x00; raw[1] = 0x3C;
      std::memset(raw + 2, 0, 16);
      raw[2] = 0x21;
      std::vector<uint8_t> pn;
      std::vector<float> ps;
      CHECK(ours::gguf::q4_0_to_prod(raw, 18, pn, ps), "prod asym ok");
      if (pn.size() >= 16) {
        CHECK(pn[0] == 0x01, "prod byte0 = 0x01 (even=LOW)");
        CHECK(pn[8] == 0x02, "prod byte8 = 0x02");
      } else {
        CHECK(false, "prod size");
      }
    }
    // End-to-end chain prod->dot, asymmetric: raw qs[0]=0xAB (lo=11,hi=10),
    // rest zero, d=1.0. GGML values: e0 = 11-8 = 3, e16 = 10-8 = 2, all
    // other 30 elements = 0-8 = -8. With q8 = all ones, dx = 1:
    // dot = 3 + 2 + 30*(-8) = -235. Any nibble swap/order bug changes this.
    {
      uint8_t raw[18];
      raw[0] = 0x00; raw[1] = 0x3C;
      std::memset(raw + 2, 0, 16);
      raw[2] = 0xAB;
      std::vector<uint8_t> pn;
      std::vector<float> ps;
      CHECK(ours::gguf::q4_0_to_prod(raw, 18, pn, ps), "chain prod ok");
      if (pn.size() >= 16 && ps.size() == 1) {
        int8_t act[32];
        for (int i = 0; i < 32; ++i) act[i] = 1;
        float got = ours::quant::q4_blockscales_dot(32, pn.data(), ps.data(), 1, act, 1.0f);
        CHECK(std::fabs(got - (-235.0f)) < 1e-3f, "prod->dot chain == -235");
        // Oracle cross-check: GGML dequant in float, then plain dot.
        std::vector<float> dec;
        std::string e2;
        CHECK(ours::quant::dequant_to_f32(2, raw, 18, 32, dec, e2), "chain oracle dequant");
        if (dec.size() == 32) {
          double acc = 0;
          for (int i = 0; i < 32; ++i) acc += (double)dec[i];
          CHECK(std::fabs(acc - (-235.0)) < 1e-4, "oracle dot == -235");
        }
      } else {
        CHECK(false, "chain sizes");
      }
    }
  }
  if (model.empty()) {
    std::printf("SKIP file part (no --model). Table checks %s.\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
  }
  ours::gguf::ModelInfo mi;
  std::string err;
  CHECK(ours::gguf::open(model, mi, err), ("open " + err).c_str());
  if (fails) return 1;
  CHECK(mi.version == 3, "version 3");
  CHECK(!mi.tensors.empty(), "has tensors");
  CHECK(mi.data_start >= 32 && mi.data_start < mi.file_size, "data_start sane");
  size_t idx = 0;
  CHECK(ours::gguf::find_tensor(mi, "token_embd.weight", idx), "has token_embd");
  // Read first tensor payload fully.
  std::vector<uint8_t> raw;
  CHECK(ours::gguf::read_tensor_bytes(model, mi, 0, raw, err), ("read t0 " + err).c_str());
  CHECK(raw.size() == ours::gguf::tensor_nbytes(mi.tensors[0]), "raw size matches");
  if (fails == 0) std::printf("PASS test_gguf model=%s tensors=%zu\n", model.c_str(), mi.tensors.size());
  return fails ? 1 : 0;
}
