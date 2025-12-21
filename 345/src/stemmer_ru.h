#pragma once
#include <string>

class RussianStemmer {
public:
  std::string stem(const std::string& token) const;

private:
  std::string stem_word(const std::string& w) const;

  static bool is_cyr_pair(unsigned char c1, unsigned char c2);
  static bool is_vowel_pair(unsigned char c1, unsigned char c2);

  static int utf8_chars(const std::string& s);

  static int rv_start_byte(const std::string& s);
  static int r2_start_byte(const std::string& s);

  static bool ends_with(const std::string& s, const char* suf);
  static bool remove_suffix_any(std::string& s, int region_byte, const char* const* sufs, int n);
  static bool remove_suffix_longest(std::string& s, int region_byte, const char* const* sufs, int n);

  static bool replace_suffix(std::string& s, int region_byte, const char* suf, const char* repl);

  static void trim_i(std::string& s, int region_byte);
  static void trim_soft_sign(std::string& s, int region_byte);
  static void post_process(std::string& s);
};
