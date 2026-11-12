#pragma once
// bench.h — benchmark entry points (run on GGUF files).
#include <string>

namespace ours {

int run_roofline(double bw_gbs, double gguf_bytes, double tg_meas);
int run_info_real(const std::string& model_path);
int run_gemm_real(const std::string& model_path, int threads, int repeats);
int run_kv_real(const std::string& model_path, const std::string& depths);
int run_h4_real(const std::string& model_path, int threads);
// SPEC on tokenizer ids: requires --model (BPE from GGUF) + --prompts.
int run_spec_real(const std::string& model_path, const std::string& prompts_path, int K);
// Full generation: prompt -> new tokens via LlamaModel decode.
int run_generate_real(const std::string& model_path, const std::string& prompt,
                      int max_new, float temp, int threads);

}  // namespace ours
