// test_hlq.cpp — PROD tests for HLQ bit-serial path (no model file needed).
// Covers: plane-interleave roundtrip, nibble unpack roundtrip, packed-file
// path == per-group path, interleaved path == per-group path.
#include <cmath>
#include <cstdio>
#include <vector>

#include "hlq_tmac.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main() {
  // 1) Interleave roundtrip (q=2, ngroups=37, odd size).
  {
    const int q = 2;
    const size_t ng = 37;
    std::vector<uint8_t> p0(ng), p1(ng);
    for (size_t g = 0; g < ng; ++g) { p0[g] = (uint8_t)(g & 0xF); p1[g] = (uint8_t)((g * 7) & 0xF); }
    const uint8_t* planes[2] = {p0.data(), p1.data()};
    std::vector<uint8_t> packed(ng * q);
    ours::hlq::pack_planes_interleaved(planes, q, packed.data(), ng);
    CHECK(packed[0 * q + 0] == p0[0] && packed[0 * q + 1] == p1[0], "interleave g0");
    CHECK(packed[(ng - 1) * q + 1] == p1[ng - 1], "interleave tail");
    std::vector<uint8_t> q0(ng), q1(ng);
    uint8_t* back[2] = {q0.data(), q1.data()};
    ours::hlq::unpack_planes_interleaved(packed.data(), back, q, ng);
    bool eq = true;
    for (size_t g = 0; g < ng; ++g) if (q0[g] != p0[g] || q1[g] != p1[g]) eq = false;
    CHECK(eq, "interleave roundtrip");
  }
  // 2) Nibble unpack roundtrip (odd n).
  {
    uint8_t packed[3] = {0x21, 0x43, 0x05};
    uint8_t nib[5] = {0};
    ours::hlq::unpack_nibbles(packed, nib, 5);
    CHECK(nib[0] == 2 && nib[1] == 1 && nib[2] == 4 && nib[3] == 3 && nib[4] == 0, "unpack order");
    uint8_t nib2[5] = {0};
    ours::hlq::unpack_nibbles_avx2(packed, nib2, 5);
    bool eq = true;
    for (int i = 0; i < 5; ++i) if (nib2[i] != nib[i]) eq = false;
    CHECK(eq, "unpack alias identical");
  }
  // 3) Packed/interleaved/pergroup/ref agree on a {0,1}-bit instance.
  // (Packed planes store 1 bit per group, so with nibble values in {0,1}
  // the unpacked groups are identical and every path must match.)
  {
    const int q = 2;
    const size_t ng = 40;
    std::vector<uint8_t> pls[2] = {std::vector<uint8_t>(ng), std::vector<uint8_t>(ng)};
    for (size_t g = 0; g < ng; ++g) {
      pls[0][g] = (uint8_t)((g * 5 + 1) & 1u);
      pls[1][g] = (uint8_t)((g * 3 + 2) & 1u);
    }
    const uint8_t* planes[2] = {pls[0].data(), pls[1].data()};
    std::vector<float> sc((size_t)q * ng), scg(2), act(ng * 4);
    scg[0] = 0.5f;
    scg[1] = 0.25f;
    for (size_t g = 0; g < ng; ++g) {
      sc[g] = scg[0];
      sc[ng + g] = scg[1];
      act[g * 4 + 0] = 0.30f;
      act[g * 4 + 1] = -0.10f;
      act[g * 4 + 2] = 0.20f;
      act[g * 4 + 3] = 0.05f;
    }
    float ref = ours::hlq::bitserial_gemv_ref(planes, q, ng, scg.data(), 0.0f, act.data(), nullptr);
    float pg = ours::hlq::bitserial_gemv_tmac_pergroup(planes, q, ng, sc.data(), 0.0f, act.data());
    CHECK(std::fabs(pg - ref) < 1e-4f * (1.0f + std::fabs(ref)), "pergroup == ref");
    // Packed file image -> packed path (bits == nibbles here, exact match).
    std::size_t pb = ours::hlq::bitplane_packed_bytes(ng);
    CHECK(pb == (ng + 7) / 8, "plane bytes");
    std::vector<uint8_t> file(2 * pb, 0);
    for (int j = 0; j < q; ++j)
      for (size_t g = 0; g < ng; ++g)
        if (planes[j][g] & 1u) file[(size_t)j * pb + (g >> 3)] |= (uint8_t)(1u << (g & 7));
    float pk = ours::hlq::bitserial_gemv_tmac_packed(file.data(), q, ng, sc.data(), 0.0f, act.data());
    CHECK(std::fabs(pk - pg) < 1e-4f * (1.0f + std::fabs(pg)), "packed == pergroup");
    // Interleaved path must equal per-group exactly (same math, one stream).
    std::vector<uint8_t> inter(ng * q);
    ours::hlq::pack_planes_interleaved(planes, q, inter.data(), ng);
    float il = ours::hlq::bitserial_gemv_interleaved(inter.data(), q, ng, sc.data(), 0.0f, act.data());
    CHECK(std::fabs(il - pg) < 1e-5f, "interleaved == pergroup");
  }
  if (fails == 0) std::printf("PASS test_hlq (3 groups)\n");
  return fails ? 1 : 0;
}
