// ngram_spec.cpp — H3 implementation.
#include "ngram_spec.h"

#include <unordered_map>

namespace ours {

uint64_t NgramDraft::hash_gram(const int32_t* toks, std::size_t n) {
  // FNV-1a 64-bit: the production table hash (order-4 key -> continuations).
  // No separate "production rolling hash": FNV over the gram IS the index.
  uint64_t h = 1469598103934665603ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= (uint64_t)(uint32_t)toks[i] + 0x9E3779B97F4A7C15ULL;
    h *= 1099511628211ULL;
  }
  return h;
}

void NgramDraft::observe(const int32_t* toks, std::size_t n) {
  if (n <= (std::size_t)kOrder) return;
  for (std::size_t i = 0; i + kOrder < n; ++i)
    table[hash_gram(toks + i, kOrder)].push_back(toks[i + kOrder]);
}

int NgramDraft::draft(const int32_t* ctx, std::size_t nctx, int K,
                      int32_t* out) {
  if (K > kMaxK) K = kMaxK;
  // Scratch context = ctx + already-drafted tokens (chained).
  int32_t window[kOrder + kMaxK];
  std::size_t base = nctx > (std::size_t)kOrder ? nctx - kOrder : 0;
  std::size_t wlen = nctx - base;
  for (std::size_t i = 0; i < wlen; ++i) window[i] = ctx[base + i];
  int m = 0;
  for (; m < K; ++m) {
    if (wlen < (std::size_t)kOrder) break;
    auto it = table.find(hash_gram(window + wlen - kOrder, kOrder));
    if (it == table.end() || it->second.empty()) break;
    // Most frequent continuation (mode) via single-pass frequency map: O(N).
    const std::vector<int32_t>& c = it->second;
    std::unordered_map<int32_t, int> freq;
    freq.reserve(c.size() * 2);
    int32_t best = c[0];
    int bestn = 0;
    for (auto v : c) {
      int n = ++freq[v];
      if (n > bestn) { bestn = n; best = v; }
    }
    out[m] = best;
    window[wlen++] = best;
  }
  return m;
}

int32_t spec_step(NgramDraft& ng, TargetVerify target, const int32_t* ctx,
                  std::size_t nctx, int K, SpecStats& stats) {
  int32_t d[NgramDraft::kMaxK];
  int m = ng.draft(ctx, nctx, K, d);
  if (m == 0) {
    // Fallback: ask for 1 token with no draft (target called with K=0
    // means "continue greedy 1 step" per the TargetVerify contract).
    int32_t corr = -1;
    target(ctx, nctx, nullptr, 0, &corr);
    return corr;
  }
  int32_t corr = -1;
  int accepted = target(ctx, nctx, d, (std::size_t)m, &corr);
  if (accepted < 0) accepted = 0;
  if (accepted > m) accepted = m;
  stats.drafted += m;
  stats.accepted += accepted;
  // The H3 gate is evaluated by ours spec on real prompts, NOT here:
  // keep the draft iff speedup>=1.2 && accept>=0.6 on real work prompts.
  return (accepted < m) ? corr : d[m - 1];
}

}  // namespace ours
