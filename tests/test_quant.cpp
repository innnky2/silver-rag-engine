// test_quant.cpp — tests for quant.h.
// Build: g++ -std=c++17 -Iinclude tests/test_quant.cpp src/quant.cpp -o /tmp/t_quant
// Run: /tmp/t_quant  (exit 0 = PASS, non-zero = FAIL)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

static void test_fp16_roundtrip() {
  // 1.0 in fp16 = 0x3C00
  float v = ours::quant::fp16_to_fp32(0x3C00);
  CHECK(std::fabs(v - 1.0f) < 1e-6, "fp16 1.0");
  CHECK(std::fabs(ours::quant::fp16_to_fp32(0xC000) + 2.0f) < 1e-3, "fp16 -2.0");
}

static void test_q4_0_known() {
  // One Q4_0 block: d=1.0 (0x3C00), all nibbles=8 -> all zeros.
  uint8_t blk[18];
  blk[0] = 0x00; blk[1] = 0x3C;
  std::memset(blk + 2, 0x88, 16);
  std::vector<float> out;
  std::string err;
  bool ok = ours::quant::dequant_to_f32(2, blk, 18, 32, out, err);
  CHECK(ok, "q4_0 dequant ok");
  for (auto x : out) CHECK(std::fabs(x) < 1e-6, "q4_0 zero block");
  // All nibbles=9 -> all ones * d.
  std::memset(blk + 2, 0x99, 16);
  ok = ours::quant::dequant_to_f32(2, blk, 18, 32, out, err);
  CHECK(ok, "q4_0 ones ok");
  for (auto x : out) CHECK(std::fabs(x - 1.0f) < 1e-6, "q4_0 ones==d");
}

static void test_q8_act() {
  float x[4] = {0.0f, 127.0f, -127.0f, 63.5f};
  int8_t q[4];
  float dx = 0;
  ours::quant::quant_f32_to_q8(x, 4, q, dx);
  CHECK(std::fabs(dx - 1.0f) < 1e-6, "q8 scale");
  CHECK(q[0] == 0 && q[1] == 127 && q[2] == -127, "q8 values");
}

static void test_q4_blockscales_dot() {
  // n=32, one block scale 2.0, all w nibbles=9 (val 1), all act=1, dx=0.5
  // dot = sum(1*1)*2.0*0.5 = 32*1.0 = 32.
  uint8_t q4[16];
  std::memset(q4, 0x99, 16);
  float sc[1] = {2.0f};
  int8_t act[32];
  for (int i = 0; i < 32; ++i) act[i] = 1;
  float got = ours::quant::q4_blockscales_dot(32, q4, sc, 1, act, 0.5f);
  CHECK(std::fabs(got - 32.0f) < 1e-3, "q4 blockscales dot");
}

static void test_truncated_rejected() {
  std::vector<float> out;
  std::string err;
  uint8_t tiny[4] = {0, 0, 0, 0};
  bool ok = ours::quant::dequant_to_f32(2, tiny, 4, 32, out, err);
  CHECK(!ok && !err.empty(), "truncated Q4_0 rejected");
  ok = ours::quant::dequant_to_f32(99, tiny, 4, 4, out, err);
  CHECK(!ok, "unknown type rejected");
}

static void test_ggml_split_halves() {
  // GGML-exact split-halves (catches sequential hi-first bugs): qs[0]=0x21
  // means element0 = low nibble 1, element16 = high nibble 2.
  std::vector<float> out;
  std::string err;
  uint8_t blk[18];
  blk[0] = 0x00; blk[1] = 0x3C;  // d = 1.0
  std::memset(blk + 2, 0, 16);
  blk[2] = 0x21;
  bool ok = ours::quant::dequant_to_f32(2, blk, 18, 32, out, err);
  CHECK(ok, "q4_0 asym ok");
  if (ok) {
    CHECK(std::fabs(out[0] - (-7.0f)) < 1e-6, "q4_0 elem0 = lo nibble");
    CHECK(std::fabs(out[16] - (-6.0f)) < 1e-6, "q4_0 elem16 = hi nibble");
  }
  // Q4_1: d=1.0, m=0.5 (0x3800), qs[0]=0x21 -> y0=1.5, y16=2.5.
  uint8_t b1[20];
  b1[0] = 0x00; b1[1] = 0x3C; b1[2] = 0x00; b1[3] = 0x38;
  std::memset(b1 + 4, 0, 16);
  b1[4] = 0x21;
  ok = ours::quant::dequant_to_f32(3, b1, 20, 32, out, err);
  CHECK(ok, "q4_1 asym ok");
  if (ok) {
    CHECK(std::fabs(out[0] - 1.5f) < 1e-6, "q4_1 elem0");
    CHECK(std::fabs(out[16] - 2.5f) < 1e-6, "q4_1 elem16");
  }
  // Q5_0: d=1.0, qh=0, qs[0]=0x21 -> y0=-15, y16=-14.
  uint8_t b5[22];
  b5[0] = 0x00; b5[1] = 0x3C;
  std::memset(b5 + 2, 0, 20);
  b5[6] = 0x21;
  ok = ours::quant::dequant_to_f32(6, b5, 22, 32, out, err);
  CHECK(ok, "q5_0 asym ok");
  if (ok) {
    CHECK(std::fabs(out[0] - (-15.0f)) < 1e-6, "q5_0 elem0");
    CHECK(std::fabs(out[16] - (-14.0f)) < 1e-6, "q5_0 elem16");
  }
  // Q5_1: d=1.0, m=0.0, qh=0, qs[0]=0x21 -> y0=1, y16=2.
  uint8_t b51[24];
  b51[0] = 0x00; b51[1] = 0x3C; b51[2] = 0x00; b51[3] = 0x00;
  std::memset(b51 + 4, 0, 20);
  b51[8] = 0x21;
  ok = ours::quant::dequant_to_f32(7, b51, 24, 32, out, err);
  CHECK(ok, "q5_1 asym ok");
  if (ok) {
    CHECK(std::fabs(out[0] - 1.0f) < 1e-6, "q5_1 elem0");
    CHECK(std::fabs(out[16] - 2.0f) < 1e-6, "q5_1 elem16");
  }
}

static void test_k_quants_golden() {
  std::vector<float> out;
  std::string err;
  bool ok;
  // Q2_K: d=1.0, dmin=0.5, scales[0]=0x21 (sc=1,mn=2), qs=0 -> y[0..15] = -1.
  uint8_t k2[84];
  std::memset(k2, 0, 84);
  k2[0] = 0x00; k2[1] = 0x3C; k2[2] = 0x00; k2[3] = 0x38;
  k2[4] = 0x21;
  ok = ours::quant::dequant_to_f32(10, k2, 84, 256, out, err);
  CHECK(ok, "q2_k ok");
  if (ok) CHECK(std::fabs(out[0] - (-1.0f)) < 1e-5, "q2_k y0");
  // Q3_K: all-zero scales -> effective sc=-32 each, qs=hmask=0 -> y=(-32)*(0-4)=128.
  uint8_t k3[110];
  std::memset(k3, 0, 110);
  k3[0] = 0x00; k3[1] = 0x3C;  // d = 1.0
  ok = ours::quant::dequant_to_f32(11, k3, 110, 256, out, err);
  CHECK(ok, "q3_k ok");
  if (ok) CHECK(std::fabs(out[0] - 128.0f) < 1e-3, "q3_k y0");
  // Q4_K: dall=1.0, dmin=0.5, scales[0]=1 (b[4]), scales[8]=2 (b[12]) ->
  // d1=1,m1=1; qs=0 -> y[0..31] = -1; y[32] = 0 (d2=m2=0).
  uint8_t k4[144];
  std::memset(k4, 0, 144);
  k4[0] = 0x00; k4[1] = 0x3C; k4[2] = 0x00; k4[3] = 0x38;
  k4[4] = 1; k4[12] = 2;
  ok = ours::quant::dequant_to_f32(12, k4, 144, 256, out, err);
  CHECK(ok, "q4_k ok");
  if (ok) {
    CHECK(std::fabs(out[0] - (-1.0f)) < 1e-5, "q4_k y0");
    CHECK(std::fabs(out[32] - 0.0f) < 1e-5, "q4_k y32");
  }
  // Q5_K: same scales, ql=qh=0 -> y[0] = -1.
  uint8_t k5[176];
  std::memset(k5, 0, 176);
  k5[0] = 0x00; k5[1] = 0x3C; k5[2] = 0x00; k5[3] = 0x38;
  k5[4] = 1; k5[12] = 2;
  ok = ours::quant::dequant_to_f32(13, k5, 176, 256, out, err);
  CHECK(ok, "q5_k ok");
  if (ok) CHECK(std::fabs(out[0] - (-1.0f)) < 1e-5, "q5_k y0");
  // Q6_K: d=1.0, sc[0]=1, ql[0]=0x11, qh[0]=0x55 -> q1=(1|(1<<4))-32=-15.
  uint8_t k6[210];
  std::memset(k6, 0, 210);
  k6[208] = 0x00; k6[209] = 0x3C;  // d = 1.0
  k6[192] = 1;                      // sc[0]
  k6[0] = 0x11; k6[128] = 0x55;
  ok = ours::quant::dequant_to_f32(14, k6, 210, 256, out, err);
  CHECK(ok, "q6_k ok");
  if (ok) CHECK(std::fabs(out[0] - (-15.0f)) < 1e-4, "q6_k y0");
  // Q8_K: d=2.0 (float LE), qs[0]=3 -> y0=6 (bsums not read).
  uint8_t k8[292];
  std::memset(k8, 0, 292);
  float dd = 2.0f;
  std::memcpy(k8, &dd, 4);
  k8[4] = 3;
  ok = ours::quant::dequant_to_f32(15, k8, 292, 256, out, err);
  CHECK(ok, "q8_k ok");
  if (ok) CHECK(std::fabs(out[0] - 6.0f) < 1e-5, "q8_k y0");
}

int main() {
  test_fp16_roundtrip();
  test_q4_0_known();
  test_q8_act();
  test_q4_blockscales_dot();
  test_truncated_rejected();
  test_ggml_split_halves();
  test_k_quants_golden();
  if (fails == 0) std::printf("PASS test_quant (7 groups)\n");
  return fails ? 1 : 0;
}
