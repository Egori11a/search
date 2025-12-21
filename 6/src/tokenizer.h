#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct TokenizerConfig {
  bool lowercase = true;
  bool normalize_yo = true;
  bool keep_numbers = true;
  int min_len = 2;
};

class Tokenizer {
public:
  explicit Tokenizer(const TokenizerConfig& cfg) : cfg_(cfg) {}
  void tokenize(const std::string& text, std::vector<std::string>& out_tokens) const;

private:
  TokenizerConfig cfg_;

  static bool is_ascii_letter(uint8_t c);
  static bool is_ascii_digit(uint8_t c);
  static bool is_cyrillic_pair(uint8_t c1, uint8_t c2);
  static void to_lower_cyrillic_pair(uint8_t& c1, uint8_t& c2);
  static void normalize_yo_pair(uint8_t& c1, uint8_t& c2);

  static int utf8_token_len_chars(const std::string& s);
};
