#include "tokenizer.h"

static inline bool is_dash(uint8_t c) { return c == (uint8_t)'-'; }
static inline bool is_apos(uint8_t c) { return c == (uint8_t)'\''; }

bool Tokenizer::is_ascii_letter(uint8_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool Tokenizer::is_ascii_digit(uint8_t c) { return c >= '0' && c <= '9'; }

bool Tokenizer::is_cyrillic_pair(uint8_t c1, uint8_t c2) {
  return (c1 == 0xD0 || c1 == 0xD1);
}

void Tokenizer::to_lower_cyrillic_pair(uint8_t& c1, uint8_t& c2) {
  if (c1 == 0xD0 && c2 == 0x81) { c1 = 0xD1; c2 = 0x91; return; }
  if (c1 == 0xD0 && c2 >= 0x90 && c2 <= 0x9F) { c2 = (uint8_t)(c2 + 0x20); return; }
  if (c1 == 0xD0 && c2 >= 0xA0 && c2 <= 0xAF) { c1 = 0xD1; c2 = (uint8_t)(c2 - 0x20); return; }
}

void Tokenizer::normalize_yo_pair(uint8_t& c1, uint8_t& c2) {
  if (c1 == 0xD1 && c2 == 0x91) { c1 = 0xD0; c2 = 0xB5; }
}

int Tokenizer::utf8_token_len_chars(const std::string& s) {
  int n = 0;
  for (size_t i = 0; i < s.size();) {
    uint8_t c = (uint8_t)s[i];
    if (c < 0x80) { i += 1; n += 1; }
    else if (i + 1 < s.size()) { i += 2; n += 1; }
    else { break; }
  }
  return n;
}

void Tokenizer::tokenize(const std::string& text, std::vector<std::string>& out_tokens) const {
  out_tokens.clear();
  std::string cur;

  auto flush = [&]() {
    if (!cur.empty()) {
      if (utf8_token_len_chars(cur) >= cfg_.min_len) {
        if (!cfg_.keep_numbers) {
          bool all_digits = true;
          for (size_t i = 0; i < cur.size(); ++i) {
            uint8_t c = (uint8_t)cur[i];
            if (c >= 0x80) { all_digits = false; break; }
            if (!is_ascii_digit(c)) { all_digits = false; break; }
          }
          if (!all_digits) out_tokens.push_back(cur);
        } else {
          out_tokens.push_back(cur);
        }
      }
      cur.clear();
    }
  };

  for (size_t i = 0; i < text.size();) {
    uint8_t c1 = (uint8_t)text[i];

    if (c1 < 0x80) {
      if (is_ascii_letter(c1)) {
        uint8_t x = c1;
        if (cfg_.lowercase && x >= 'A' && x <= 'Z') x = (uint8_t)(x - 'A' + 'a');
        cur.push_back((char)x);
        i += 1;
        continue;
      }
      if (is_ascii_digit(c1)) {
        cur.push_back((char)c1);
        i += 1;
        continue;
      }
      if ((is_dash(c1) || is_apos(c1)) && !cur.empty()) {
        if (i + 1 < text.size()) {
          uint8_t n1 = (uint8_t)text[i + 1];
          bool next_ok = is_ascii_letter(n1) || is_ascii_digit(n1);
          if (!next_ok && n1 >= 0x80 && i + 2 <= text.size()) next_ok = true;
          if (next_ok) { cur.push_back((char)c1); i += 1; continue; }
        }
      }
      flush();
      i += 1;
      continue;
    }

    if (i + 1 < text.size()) {
      uint8_t c2 = (uint8_t)text[i + 1];
      if (is_cyrillic_pair(c1, c2)) {
        if (cfg_.lowercase) to_lower_cyrillic_pair(c1, c2);
        if (cfg_.normalize_yo) normalize_yo_pair(c1, c2);
        cur.push_back((char)c1);
        cur.push_back((char)c2);
        i += 2;
        continue;
      }
    }

    flush();
    i += 1;
  }

  flush();
}
