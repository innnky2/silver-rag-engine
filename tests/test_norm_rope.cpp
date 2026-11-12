// test_norm_rope.cpp — tests for RMSNorm/RoPE/softmax/silu.
// Build: g++ -std=c++17 -Iinclude tests/test_norm_rope.cpp src/norm_rope.cpp -o /tmp/t_nr
#include <cmath>
#include <cstdio>
#include <vector>

#include "norm_rope.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main() {
  {
    float x[2] = {3.0f, 4.0f}, w[2] = {1.0f, 1.0f}, y[2];
    ours::rmsnorm(x, w, 2, 0.0f, y);
    // rms = sqrt((9+16)/2) = sqrt(12.5); y = x / rms
    float rms = std::sqrt(12.5f);
    CHECK(std::fabs(y[0] - 3.0f / rms) < 1e-5 && std::fabs(y[1] - 4.0f / rms) < 1e-5, "rmsnorm");
  }
  {
    float x[3] = {1.0f, 2.0f, 3.0f};
    ours::softmax_inplace(x, 3);
    double s = std::exp(-2.0) + std::exp(-1.0) + 1.0;
    CHECK(std::fabs(x[2] - 1.0 / s) < 1e-6, "softmax");
    CHECK(std::fabs(x[0] + x[1] + x[2] - 1.0) < 1e-6, "softmax sum");
  }
  {
    CHECK(std::fabs(ours::silu(0.0f)) < 1e-7, "silu 0");
    CHECK(ours::silu(100.0f) > 99.0f, "silu large");
  }
  {
    // RoPE at pos 0 = identity.
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ours::rope_inplace(x, 4, 0, 10000.0f, 1, 1);
    CHECK(std::fabs(x[0] - 1.0f) < 1e-6 && std::fabs(x[3] - 4.0f) < 1e-6, "rope pos0 identity");
    // RoPE preserves norm per pair.
    float y[2] = {1.0f, 0.0f};
    ours::rope_inplace(y, 2, 5, 10000.0f, 1, 1);
    CHECK(std::fabs(y[0] * y[0] + y[1] * y[1] - 1.0f) < 1e-6, "rope rotation norm");
  }
  {
    // YaRN (llama.cpp rope_yarn): fallback paths identical to plain RoPE.
    float a[4] = {1,2,3,4}, b[4] = {1,2,3,4};
    ours::rope_inplace_scaled(a, 4, 5, 10000.0f, 1, 2.0f, 128, 32.0f, 1.0f);
    ours::rope_inplace(b, 4, 5, 10000.0f, 1, 1);
    CHECK(std::fabs(a[0]-b[0])+std::fabs(a[3]-b[3]) < 1e-6, "yarn fallback pos<orig");
    float e[4] = {1,2,3,4}, f[4] = {1,2,3,4};
    ours::rope_inplace_scaled(e, 4, 300, 10000.0f, 1, 1.0f, 128, 32.0f, 1.0f);
    ours::rope_inplace(f, 4, 300, 10000.0f, 1, 1);
    CHECK(std::fabs(e[0]-f[0])+std::fabs(e[3]-f[3]) < 1e-6, "yarn scale=1 identity");
  }
  {
    // YaRN active: low dims keep original frequency, high dims take
    // interpolated freq/scale; vectors scale by mscale = 1+0.1*ln(s).
    // head_dim=128, theta=10000, scale=2, orig=128, pos=200.
    const float mscale = 1.0f + 0.1f * std::log(2.0f);
    float x[128];
    for (int i = 0; i < 128; ++i) x[i] = 0.0f;
    x[0] = 1.0f;  // pair 0: freq = 1.0, must stay unscaled (angle = 200)
    x[126] = 1.0f;  // pair 63: freq = 10000^(-126/128), must interpolate (/2)
    ours::rope_inplace_scaled(x, 128, 200, 10000.0f, 1, 2.0f, 128, 32.0f, 1.0f);
    float f0 = 1.0f;
    CHECK(std::fabs(x[0] - std::cos(200.0f * f0) * mscale) < 1e-4, "yarn pair0 keeps orig freq");
    CHECK(std::fabs(x[1] - std::sin(200.0f * f0) * mscale) < 1e-4, "yarn pair0 sin");
    float f63 = 1.0f / std::pow(10000.0f, 126.0f / 128.0f);
    CHECK(std::fabs(x[126] - std::cos(200.0f * f63 / 2.0f) * mscale) < 1e-4, "yarn pair63 interpolated");
    CHECK(std::fabs(x[127] - std::sin(200.0f * f63 / 2.0f) * mscale) < 1e-4, "yarn pair63 sin");
  }
  {
    float W[6] = {1, 2, 3, 4, 5, 6};  // 2x3
    float x[3] = {1, 0, -1};
    float y[2];
    ours::matvec_f32(W, x, 2, 3, y);
    CHECK(std::fabs(y[0] - (1.0f - 3.0f)) < 1e-6 && std::fabs(y[1] - (4.0f - 6.0f)) < 1e-6, "matvec");
  }
  if (fails == 0) std::printf("PASS test_norm_rope (5 groups)\n");
  return fails ? 1 : 0;
}
