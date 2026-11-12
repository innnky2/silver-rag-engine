#pragma once
// tokenizer.h — BPE tokenizer from GGUF arrays (pieces + merges).
#include <cstdint>
#include <string>
#include <vector>

namespace ours {

struct Tokenizer {
  std::vector<std::string> pieces;
  std::vector<float> scores;
  std::vector<int32_t> types;  // 1=normal,2=unknown,3=control,4=user,5=unused,6=byte
  std::vector<std::string> merges;  // "left right" rank = index
  int32_t bos = 1, eos = 2, unk = 0, pad = -1;
  bool add_bos = true;

  bool empty() const { return pieces.empty(); }
  // True BPE: split to chars, then apply merges by rank until fixpoint,
  // then longest-match map to ids. Byte fallback for unknown.
  void encode(const std::string& text, std::vector<int32_t>& out, bool add_bos_flag) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  int32_t find_piece(const std::string& s) const;
};

}  // namespace ours
