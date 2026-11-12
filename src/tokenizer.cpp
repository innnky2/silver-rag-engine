// tokenizer.cpp — PROD BPE with GGUF merge ranks + longest-match map.
#include "tokenizer.h"

#include <cstdio>
#include <unordered_map>

namespace ours {

int32_t Tokenizer::find_piece(const std::string& s) const {
  for (size_t i = 0; i < pieces.size(); ++i)
    if (pieces[i] == s) return (int32_t)i;
  return -1;
}

static std::string sp_map(const std::string& t) {
  std::string o;
  o.reserve(t.size());
  for (size_t i = 0; i < t.size();) {
    if (i + 3 <= t.size() && (unsigned char)t[i] == 0xE2 &&
        (unsigned char)t[i + 1] == 0x96 && (unsigned char)t[i + 2] == 0x81) {
      o.push_back(' ');
      i += 3;
    } else {
      o.push_back(t[i++]);
    }
  }
  return o;
}

void Tokenizer::encode(const std::string& text, std::vector<int32_t>& out, bool add_bos_flag) const {
  out.clear();
  if (pieces.empty()) return;
  if (add_bos_flag && bos >= 0) out.push_back(bos);
  // 1) Start from byte/char symbols mapped through ▁ convention.
  std::vector<std::string> syms;
  syms.reserve(text.size());
  for (size_t i = 0; i < text.size();) {
    unsigned char c = (unsigned char)text[i];
    size_t len = 1;
    if ((c & 0x80) == 0) len = 1;
    else if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    if (i + len > text.size()) len = 1;
    std::string ch = text.substr(i, len);
    if (ch == " ") ch = "\xE2\x96\x81";  // space -> ▁ piece convention
    syms.push_back(ch);
    i += len;
  }
  // 2) BPE merge by rank until fixpoint (only when merges present).
  if (!merges.empty()) {
    std::unordered_map<std::string, int> rank;
    rank.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); ++i) rank[merges[i]] = (int)i;
    while (syms.size() >= 2) {
      int best_rank = -1;
      size_t best_pos = 0;
      for (size_t p = 0; p + 1 < syms.size(); ++p) {
        auto it = rank.find(syms[p] + " " + syms[p + 1]);
        if (it != rank.end() && (best_rank < 0 || it->second < best_rank)) {
          best_rank = it->second;
          best_pos = p;
        }
      }
      if (best_rank < 0) break;
      syms[best_pos] += syms[best_pos + 1];
      syms.erase(syms.begin() + (long)best_pos + 1);
    }
  }
  // 3) Map symbols to ids via exact piece table; greedy split fallback.
  std::unordered_map<std::string, int32_t> mp;
  mp.reserve(pieces.size() * 2);
  size_t max_piece = 1;
  for (size_t i = 0; i < pieces.size(); ++i) {
    mp[pieces[i]] = (int32_t)i;
    if (pieces[i].size() > max_piece) max_piece = pieces[i].size();
  }
  for (auto& s : syms) {
    auto it = mp.find(s);
    if (it != mp.end()) {
      out.push_back(it->second);
      continue;
    }
    // Greedy longest-prefix split of an unmerged symbol.
    // Bound = longest piece in the table (covers CJK/emoji multi-byte pieces).
    size_t pos = 0;
    bool ok_all = true;
    while (pos < s.size()) {
      size_t best_len = 0;
      int32_t best_id = -1;
      size_t lim = s.size() - pos < max_piece ? s.size() - pos : max_piece;
      for (size_t l = 1; l <= lim; ++l) {
        auto jt = mp.find(s.substr(pos, l));
        if (jt != mp.end()) { best_len = l; best_id = jt->second; }
      }
      if (best_id < 0) { ok_all = false; break; }
      out.push_back(best_id);
      pos += best_len;
    }
    if (!ok_all) out.push_back(unk);
  }
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string o;
  for (int32_t id : ids) {
    if (id < 0 || (size_t)id >= pieces.size()) continue;
    if (id == bos || id == eos) continue;
    o += sp_map(pieces[(size_t)id]);
  }
  return o;
}

}  // namespace ours
