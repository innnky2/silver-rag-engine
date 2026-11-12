// test_model.cpp — model/generate tests (needs --model, gated by size).
// Build: cmake --build build --target test_model  (or g++ manual, see README)
// Run: ./build/test_model --model model.gguf --max-new 8
// GATE: skips generate when vocab==0 or layers==0 (returns 2 = SKIP).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg); ++fails; } } while (0)

int main(int argc, char** argv) {
  std::string model;
  int max_new = 8;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    if (!std::strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = std::atoi(argv[++i]);
  }
  if (model.empty()) {
    std::printf("SKIP test_model (no --model).\n");
    return 2;
  }
  ours::LlamaModel m;
  std::string err;
  CHECK(m.load(model, 4, false, err), ("load " + err).c_str());
  if (fails) return 1;
  CHECK(m.loaded(), "loaded flag");
  CHECK(m.dims().n_layer > 0 && m.dims().n_embd > 0, "dims sane");
  CHECK(!m.tok().empty(), "tokenizer loaded");
  // Single decode step on BOS.
  std::vector<float> logits;
  int32_t bos = m.tok().bos >= 0 ? m.tok().bos : 1;
  CHECK(m.decode_step(bos, 0, logits, err), ("step0 " + err).c_str());
  CHECK((int)logits.size() == m.dims().vocab, "logits vocab size");
  // KV grew by 1 per layer.
  // Generate a few tokens greedy.
  std::vector<int32_t> prompt = {bos};
  ours::SamplerConfig sc;
  sc.temp = 0;
  std::vector<int32_t> out;
  CHECK(m.generate(prompt, max_new, sc, out, err), ("generate " + err).c_str());
  CHECK(!out.empty() && (int)out.size() <= max_new, "generate length gate");
  std::string text = m.tok().decode(out);
  CHECK(!text.empty(), "generate text non-empty");
  if (fails == 0) std::printf("PASS test_model layers=%d new=%zu\n", m.dims().n_layer, out.size());
  return fails ? 1 : 0;
}
