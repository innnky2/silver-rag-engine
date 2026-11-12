// test_token_sampler.cpp — tests for tokenizer + sampler (needs --model).
// Build: g++ -std=c++17 -Iinclude tests/test_token_sampler.cpp src/tokenizer.cpp src/sampler.cpp src/gguf.cpp -o /tmp/t_ts
// Run: /tmp/t_ts --model model.gguf
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "sampler.h"
#include "tokenizer.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main(int argc, char** argv) {
  // Sampler unit checks (no file).
  {
    float lg[3] = {1.0f, 5.0f, 2.0f};
    ours::SamplerConfig g;
    g.temp = 0;
    uint64_t rng = 1;
    CHECK(ours::sample_logits(lg, 3, g, rng) == 1, "greedy argmax");
    ours::SamplerConfig t;
    t.temp = 0.8f; t.top_k = 1;
    rng = 42;
    CHECK(ours::sample_logits(lg, 3, t, rng) == 1, "top_k=1 greedy");
  }
  // BPE merge unit check (no file): pieces a,b,ab + merge "a b" -> "ab" one id.
  {
    ours::Tokenizer tk;
    tk.pieces = {"a", "b", "ab"};
    tk.scores = {0, 0, 0};
    tk.types = {1, 1, 1};
    tk.merges = {"a b"};
    tk.bos = -1;
    std::vector<int32_t> ids;
    tk.encode("ab", ids, false);
    CHECK(ids.size() == 1 && ids[0] == 2, "bpe merge rank");
  }
  std::string model;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
  if (model.empty()) {
    std::printf("SKIP file part (no --model). Sampler checks %s.\n", fails ? "FAILED" : "OK");
    return fails ? 1 : 0;
  }
  ours::gguf::ModelInfo mi;
  std::string err;
  CHECK(ours::gguf::open(model, mi, err), ("open " + err).c_str());
  if (fails) return 1;
  CHECK(!mi.tokens.empty(), "tokenizer tokens present");
  ours::Tokenizer tk;
  tk.pieces = mi.tokens; tk.scores = mi.scores; tk.types = mi.toktypes;
  tk.bos = 1; tk.eos = 2; tk.unk = 0;
  std::vector<int32_t> ids;
  tk.encode("Halo dunia", ids, false);
  CHECK(!ids.empty(), "encode non-empty");
  std::string back = tk.decode(ids);
  CHECK(!back.empty(), "decode non-empty");
  // Roundtrip stability: decode(encode(x)) contains input words.
  CHECK(back.find("Halo") != std::string::npos, "roundtrip Halo");
  if (fails == 0) std::printf("PASS test_token_sampler tokens=%zu\n", mi.tokens.size());
  return fails ? 1 : 0;
}
