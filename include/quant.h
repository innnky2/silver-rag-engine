#pragma once
// quant.h — dequant/quant for GGUF types.
// Reference: GGML dequant semantics (llama.cpp ggml.c), Jacob affine.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ours {
namespace quant {

float fp16_to_fp32(uint16_t h);
// Dequant one full tensor payload (any supported type) to f32 vector.
// Returns false on IQ/BF16/unknown with err set.
bool dequant_to_f32(uint32_t qtype, const uint8_t* src, size_t src_bytes,
                    size_t n_elems, std::vector<float>& out, std::string& err);
// Dequant one matrix row (n_cols elems) from a 2-D tensor payload.
bool dequant_row_to_f32(uint32_t qtype, const uint8_t* payload, size_t payload_bytes,
                        size_t n_cols, size_t n_rows, size_t row,
                        std::vector<float>& out, std::string& err);
// float activations -> int8 Q8 with scale (per-tensor). Returns scale via d_x.
void quant_f32_to_q8(const float* x, size_t n, int8_t* q8, float& d_x);
// int dot used by H1 on any Q4 nibble row + Q8 act (per-block scales version).
float q4_blockscales_dot(size_t n, const uint8_t* q4, const float* blk_scales,
                         size_t blk_n, const int8_t* q8, float d_x);

}  // namespace quant
}  // namespace ours
