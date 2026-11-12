#pragma once
// sampler.h — PROD greedy/temp/top-k/top-p sampler. RNG only for sampling.
#include <cstdint>
#include <vector>

namespace ours {

struct SamplerConfig {
  float temp = 0.0f;    // <=0 = greedy
  int32_t top_k = 40;   // <=0 = off
  float top_p = 0.95f;  // >=1 = off
  float min_p = 0.0f;   // <=0 = off
  uint64_t seed = 0;    // 0 = deterministic default
};

int32_t sample_logits(const float* logits, size_t n_vocab, const SamplerConfig& cfg,
                      uint64_t& rng_state);

}  // namespace ours
