// sampler.cpp — PROD sampler (greedy + temp/top-k/top-p/min-p).
#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ours {
namespace {
uint64_t xorshift64(uint64_t& s) {
  if (s == 0) s = 0x9E3779B97F4A7C15ULL;
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return s;
}
double rand01(uint64_t& s) {
  return (double)(xorshift64(s) >> 11) * (1.0 / 9007199254740992.0);
}
}  // namespace

int32_t sample_logits(const float* logits, size_t n_vocab, const SamplerConfig& cfg,
                      uint64_t& rng_state) {
  if (!logits || n_vocab == 0) return 0;
  if (cfg.temp <= 0.0f) {
    int32_t best = 0;
    for (size_t i = 1; i < n_vocab; ++i)
      if (logits[i] > logits[best]) best = (int32_t)i;
    return best;
  }
  // Temperature + top-k prefilter, then top-p/min-p on softmax.
  std::vector<int32_t> idx(n_vocab);
  for (size_t i = 0; i < n_vocab; ++i) idx[i] = (int32_t)i;
  float t = cfg.temp > 0 ? cfg.temp : 1.0f;
  if (cfg.top_k > 0 && (size_t)cfg.top_k < n_vocab) {
    std::nth_element(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                     [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    idx.resize((size_t)cfg.top_k);
  }
  std::vector<double> p(idx.size());
  double mx = -1e300;
  for (size_t i = 0; i < idx.size(); ++i) {
    double v = (double)logits[idx[i]] / (double)t;
    if (v > mx) mx = v;
  }
  double sum = 0;
  for (size_t i = 0; i < idx.size(); ++i) { p[i] = std::exp((double)logits[idx[i]] / (double)t - mx); sum += p[i]; }
  for (auto& v : p) v /= sum;
  // Sort by prob desc for top-p/min-p.
  std::vector<size_t> order(idx.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return p[a] > p[b]; });
  size_t keep = order.size();
  if (cfg.top_p < 1.0f && !order.empty()) {
    double csum = 0;
    keep = order.size();
    for (size_t k = 0; k < order.size(); ++k) {
      csum += p[order[k]];
      if (csum >= cfg.top_p) { keep = k + 1; break; }
    }
  }
  if (cfg.min_p > 0.0f && !order.empty()) {
    double pmax = p[order[0]];
    size_t k2 = 0;
    for (size_t k = 0; k < keep; ++k)
      if (p[order[k]] >= cfg.min_p * pmax) k2 = k + 1;
    keep = k2 > 0 ? k2 : 1;
  }
  double kept_sum = 0;
  for (size_t k = 0; k < keep; ++k) kept_sum += p[order[k]];
  if (kept_sum <= 0) return idx[order[0]];
  double r = rand01(rng_state) * kept_sum;
  double acc = 0;
  for (size_t k = 0; k < keep; ++k) {
    acc += p[order[k]];
    if (r <= acc) return idx[order[k]];
  }
  return idx[order[keep - 1]];
}

}  // namespace ours
