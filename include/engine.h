#pragma once
// engine.h — PROD engine: GGUF reader + full llama decode + H1/H4 GEMV primitives.
// Princeton: GEMV integer-dot (H1), tiled pool GEMV (H4), Q8 KV (H2),
// n-gram draft (H3), roofline (H5), plus full generate() for real text.
#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "kv_q8.h"
#include "model.h"
#include "threadpool.h"

namespace ours {

struct EngineConfig {
  int threads = 0;
  bool pin_threads = true;
};

class Engine {
 public:
  Engine() = default;
  explicit Engine(int threads, bool pin = true) { pool_.reset(threads, pin); }
  bool open_gguf(const std::string& path, std::string& err);
  bool open_model(const std::string& path, int threads, bool pin, std::string& err);
  // H1/H4 primitive on a real Q4_0 2-D tensor row range.
  bool build_tiled_row(size_t tensor_idx, size_t row, std::vector<uint8_t>& tiled_row,
                       std::vector<float>& blk_scales, size_t& n_cols, std::string& err);
  void gemv_rows(const std::vector<std::vector<uint8_t>>& tiled_rows,
                 const std::vector<std::vector<float>>& row_scales, const int8_t* q8_act,
                 float d_x, std::vector<float>& out);
  // Full generation on real weights + real tokenizer.
  bool generate(const std::string& prompt, int max_new, float temp, std::vector<int32_t>& out_ids,
                std::string& decoded, std::string& err);

  const gguf::ModelInfo& info() const { return mi_; }
  LlamaModel& model() { return model_; }
  const std::string& path() const { return path_; }
  ThreadPool& pool() { return pool_; }

 private:
  std::string path_;
  gguf::ModelInfo mi_;
  LlamaModel model_;
  bool model_ready_ = false;
  EngineConfig cfg_;
  ThreadPool pool_{0, true};
};

}  // namespace ours
