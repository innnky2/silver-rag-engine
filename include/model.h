#pragma once
// model.h — llama-arch transformer on GGUF weights.
// Full decode path: embed -> per-layer attn (GQA+RoPE+Q8 KV) -> SwiGLU FFN
// -> output_norm -> lm_head -> sampler.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gguf.h"
#include "kv_q8.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"

namespace ours {

struct LlamaDims {
  int32_t n_layer = 0;
  int32_t n_embd = 0;
  int32_t n_head = 0;
  int32_t n_kv = 0;
  int32_t head_dim = 0;
  int32_t n_ff = 0;
  int32_t vocab = 0;
  float norm_eps = 1e-5f;
  float rope_theta = 10000.0f;
  // YaRN / linear scaling (GGUF rope.scaling.* when present; 1.0 = off).
  float rope_scale = 1.0f;
  float rope_beta_fast = 32.0f;  // llama.cpp YaRN defaults
  float rope_beta_slow = 1.0f;
  int32_t rope_orig_ctx = 0;
};

struct WeightBlob {
  uint32_t qtype = 0;
  std::vector<uint64_t> dims;  // GGUF order: [cols, rows]
  std::vector<uint8_t> raw;    // exact tensor_nbytes()
};

class LlamaModel {
 public:
  LlamaModel() = default;
  bool load(const std::string& gguf_path, int threads, bool pin, std::string& err);
  bool loaded() const { return loaded_; }

  const LlamaDims& dims() const { return dims_; }
  const Tokenizer& tok() const { return tok_; }
  const gguf::ModelInfo& meta() const { return mi_; }

  // Single decode step: given previous token + position, produce logits.
  // logits sized [vocab]. Uses per-layer Q8 KV caches (member kv_).
  bool decode_step(int32_t token, int pos, std::vector<float>& logits, std::string& err);
  void reset_kv();
  // Greedy/sampled generation of up to max_new tokens.
  bool generate(const std::vector<int32_t>& prompt, int max_new, const SamplerConfig& scfg,
                std::vector<int32_t>& out_ids, std::string& err);

 private:
  bool get_row(const std::string& name, size_t row, std::vector<float>& out, std::string& err);
  bool matvec_weight(const std::string& name, const float* x, size_t n_in, size_t n_out,
                     float* y, std::string& err);
  bool matvec_weight_q8fast(const WeightBlob& wb, const std::vector<int8_t>& q8buf,
                            float dx, float* y, std::string& err);

  std::string gguf_path_;
  gguf::ModelInfo mi_;
  LlamaDims dims_;
  Tokenizer tok_;
  std::map<std::string, WeightBlob> weights_;
  std::map<std::string, size_t> index_;
  std::vector<KvCacheQ8> kv_;
  ThreadPool pool_{0, true};
  // Scratch (sized on load).
  std::vector<float> xb_, xb2_, hb_, hb2_, q_, k_, v_, att_, logits_f_;
  std::vector<float> tmp_attn_;
  std::vector<int8_t> q8buf_;
  bool loaded_ = false;
};

}  // namespace ours
