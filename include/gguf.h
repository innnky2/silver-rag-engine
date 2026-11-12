#pragma once
// gguf.h — PROD GGUFv3 reader, full KV spec + all common quant types.
//
// Reads real .gguf files: magic, header, full KV metadata (all GGUF value
// types), tensor infos, tokenizer arrays. Payloads streamed on demand.
// K-quants (Q2_K..Q8_K) fully sized; IQ/BF16 reported explicitly.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ours {
namespace gguf {

enum Type : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
  Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
  Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ2_S = 18, IQ3_XXS = 19, IQ1_S = 20,
  IQ4_NL = 21, IQ3_S = 22, IQ2_M = 23, IQ4_XS = 24, IQ1_M = 25,
  BF16 = 30,
};

struct TensorInfo {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims;
  uint64_t offset = 0;
  uint64_t n_elems = 1;
};

struct ModelInfo {
  uint32_t version = 0;
  std::map<std::string, std::string> str_meta;
  std::map<std::string, int32_t> i32_meta;
  std::map<std::string, uint32_t> u32_meta;
  std::map<std::string, float> f32_meta;
  std::map<std::string, bool> bool_meta;
  std::map<std::string, uint64_t> u64_meta;
  std::vector<TensorInfo> tensors;
  uint64_t data_start = 0;
  uint64_t alignment = 32;
  uint64_t file_size = 0;
  std::string arch;
  // Tokenizer (GGUF arrays, may be empty if file has none).
  std::vector<std::string> tokens;
  std::vector<float> scores;
  std::vector<int32_t> toktypes;
  std::vector<std::string> merges;  // "a b" BPE merge list (rank = index).
};

std::string type_name(uint32_t t);
bool is_k_quant(uint32_t t);
bool is_iq(uint32_t t);
// Exact payload bytes; 0 only for IQ/BF16/unknown (caller gets err string).
uint64_t tensor_nbytes(const TensorInfo& ti);
bool open(const std::string& path, ModelInfo& out, std::string& err);
bool read_tensor_bytes(const std::string& path, const ModelInfo& mi,
                       size_t index, std::vector<uint8_t>& out, std::string& err);
bool find_tensor(const ModelInfo& mi, const std::string& name, size_t& idx);
// Q4_0 one-row converter preserving PER-BLOCK scales (no mean averaging).
// out nibbles (low=nibble0 layout), out_scales one float per 32-elem block.
bool q4_0_to_prod(const uint8_t* src, size_t src_bytes, std::vector<uint8_t>& q4_nib,
                  std::vector<float>& blk_scales);
bool q4_0_to_prod_legacy(const uint8_t* src, size_t src_bytes,
                         std::vector<uint8_t>& q4_nib, float& d_mean);

}  // namespace gguf
}  // namespace ours
