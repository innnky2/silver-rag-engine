// ngram_spec.h — H3: n-gram draft + thresholded verification.
//
// Speculative decoding (Leviathan et al., ICML 2023, arXiv:2211.17192;
// Chen et al., arXiv:2302.01318): cheap draft + one parallel target pass
// over K tokens. Passes the H3 gate at speedup >= 1.2 with accept >= 0.6
// (K <= 8); the dup>20% guard rejects repeated prompts that inflate alpha.
// Design: order-4 hash table over prompt + emitted tokens.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ours {

// Abstract target model: given context + K draft tokens, return how many
// draft tokens are ACCEPTED (0..K) and fill the correction token when < K.
// Any engine (including llama.cpp via binding) can be plugged in here.
using TargetVerify = std::function<int(const int32_t* ctx, std::size_t nctx,
                                      const int32_t* draft, std::size_t k,
                                      int32_t* correction)>;

struct SpecStats {
  long long drafted = 0;
  long long accepted = 0;
  double accept_rate() const {
    return drafted ? (double)accepted / (double)drafted : 0.0;
  }
};

struct NgramDraft {
  static constexpr int kOrder = 4;
  static constexpr int kMaxK = 8;
  // key: rolling hash of N-gram -> observed continuation tokens.
  std::unordered_map<uint64_t, std::vector<int32_t>> table;

  static uint64_t hash_gram(const int32_t* toks, std::size_t n);
  void observe(const int32_t* toks, std::size_t n);  // train from prompt
  // Propose <=K tokens after context. Returns number proposed.
  int draft(const int32_t* ctx, std::size_t nctx, int K, int32_t* out);
};

// One generate step with the H3 gate. Returns the token used.
// Stats accumulate; the bench harness decides pass/fail on the gate.
int32_t spec_step(NgramDraft& ng, TargetVerify target, const int32_t* ctx,
                  std::size_t nctx, int K, SpecStats& stats);

}  // namespace ours
