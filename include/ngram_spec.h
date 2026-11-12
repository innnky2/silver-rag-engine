// ngram_spec.h — H3: free n-gram draft + thresholded verification.
//
// THEORY (why a parameter-free draft can win):
//   Speculative decoding (Leviathan et al., ICML 2023, arXiv:2211.17192;
//   Chen et al., arXiv:2302.01318) splits generation into draft (cheap) and
//   verify (one parallel target pass over K tokens). The output distribution
//   is preserved by construction; wall-clock wins iff the acceptance rate
//   alpha pays for the draft overhead. Expected tokens per target call is
//   ~1 + alpha*K for greedy acceptance, so the H3 gate (speedup >= 1.2 with
//   accept >= 0.6 at K <= 8) is the break-even check, not a tuning knob.
//   An order-4 n-gram table is the cheapest draft that exploits the only
//   structure it can rely on: local repetition (code templates, JSON keys,
//   refactor spans). This is the same observation behind prompt-lookup
//   decoding (llama.cpp speculative path): on diverse prose the table misses
//   and H3 correctly predicts 0-10%; on templated text it hits and predicts
//   >30%. The dup>20% guard in bench exists because repeated prompts fake
//   alpha (the 4.75x pseudo-trap); real work text is required.
//
// Design: rolling-hash table (order N=4) over prompt + emitted tokens.
// draft() proposes <=K continuations. verify_accept() is the INTEGRATION
// point: the target model is called once for K tokens, longest accepted
// prefix is kept. Acceptance statistics decide the H3 gate, not this file.
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
