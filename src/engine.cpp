// engine.cpp — GGUF rows + full model wrapper.
#include "engine.h"

#include <cstring>

#include "lut_gemm.h"
#include "quant.h"

namespace ours {

bool Engine::open_gguf(const std::string& path, std::string& err) {
  path_ = path;
  model_ready_ = false;
  return gguf::open(path, mi_, err);
}

bool Engine::open_model(const std::string& path, int threads, bool pin, std::string& err) {
  path_ = path;
  cfg_.threads = threads;
  cfg_.pin_threads = pin;
  pool_.reset(threads, pin);
  if (!gguf::open(path, mi_, err)) return false;
  if (!model_.load(path, threads, pin, err)) return false;
  model_ready_ = true;
  return true;
}

bool Engine::build_tiled_row(size_t tensor_idx, size_t row,
                             std::vector<uint8_t>& tiled_row, std::vector<float>& blk_scales,
                             size_t& n_cols, std::string& err) {
  tiled_row.clear();
  blk_scales.clear();
  if (tensor_idx >= mi_.tensors.size()) { err = "tensor index out of range"; return false; }
  const auto& ti = mi_.tensors[tensor_idx];
  if (ti.type != gguf::Q4_0) {
    err = "build_tiled_row expects Q4_0; tensor '" + ti.name + "' is " +
          gguf::type_name(ti.type) + " (use model generate path for K-quants)";
    return false;
  }
  std::vector<uint8_t> raw;
  if (!gguf::read_tensor_bytes(path_, mi_, tensor_idx, raw, err)) return false;
  if (ti.dims.size() != 2) { err = "Q4_0 tensor is not 2-D: " + ti.name; return false; }
  size_t cols = (size_t)ti.dims[0];
  size_t rows = (size_t)ti.dims[1];
  if (row >= rows) { err = "row out of range"; return false; }
  if (cols % 32 != 0) { err = "Q4_0 cols not multiple of 32"; return false; }
  size_t row_bytes = (cols / 32) * 18;
  const uint8_t* rp = raw.data() + row * row_bytes;
  if (!gguf::q4_0_to_prod(rp, row_bytes, tiled_row, blk_scales)) { err = "q4_0 conversion failed"; return false; }
  n_cols = cols;
  return true;
}

void Engine::gemv_rows(const std::vector<std::vector<uint8_t>>& tiled_rows,
                       const std::vector<std::vector<float>>& row_scales, const int8_t* q8_act,
                       float d_x, std::vector<float>& out) {
  size_t m = tiled_rows.size();
  out.assign(m, 0.0f);
  if (m == 0 || !q8_act) return;
  pool_.parallel_for(m, [&](size_t r) {
    const std::vector<uint8_t>& row = tiled_rows[r];
    size_t n = row.size() * 2;
    const float* sc = row_scales[r].data();
    out[r] = quant::q4_blockscales_dot(n, row.data(), sc, row_scales[r].size(), q8_act, d_x);
  });
}

bool Engine::generate(const std::string& prompt, int max_new, float temp,
                      std::vector<int32_t>& out_ids, std::string& decoded, std::string& err) {
  if (!model_ready_) {
    if (!open_model(path_, cfg_.threads, cfg_.pin_threads, err)) return false;
  }
  std::vector<int32_t> pin;
  model_.tok().encode(prompt, pin, model_.tok().add_bos);
  SamplerConfig sc;
  sc.temp = temp;
  if (!model_.generate(pin, max_new, sc, out_ids, err)) return false;
  decoded = model_.tok().decode(out_ids);
  return true;
}

}  // namespace ours
