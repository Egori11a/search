#include "stemmer_ru.h"
#include <cstring>
#include <vector>

// ---------------- UTF-8 Cyrillic ----------------

// Cyrillic in UTF-8: mainly 0xD0/0xD1 pairs for Russian letters
bool RussianStemmer::is_cyr_pair(unsigned char c1, unsigned char c2) {
  (void)c2;
  return (c1 == 0xD0 || c1 == 0xD1);
}

// Vowels in Russian: а е и о у ы э ю я
// lowercase, ё already normalized to е
bool RussianStemmer::is_vowel_pair(unsigned char c1, unsigned char c2) {
  // а (D0 B0), е (D0 B5), и (D0 B8), о (D0 BE), у (D1 83),
  // ы (D1 8B), э (D1 8D), ю (D1 8E), я (D1 8F)
  if (c1 == 0xD0) {
    return (c2 == 0xB0) || (c2 == 0xB5) || (c2 == 0xB8) || (c2 == 0xBE);
  }
  if (c1 == 0xD1) {
    return (c2 == 0x83) || (c2 == 0x8B) || (c2 == 0x8D) || (c2 == 0x8E) || (c2 == 0x8F);
  }
  return false;
}

int RussianStemmer::utf8_chars(const std::string& s) {
  int n = 0;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) { i += 1; n += 1; }
    else if (i + 1 < s.size()) { i += 2; n += 1; }
    else break;
  }
  return n;
}

static int next_char_byte(const std::string& s, int byte_pos) {
  if (byte_pos >= (int)s.size()) return (int)s.size();
  unsigned char c = (unsigned char)s[(size_t)byte_pos];
  if (c < 0x80) return byte_pos + 1;
  if (byte_pos + 1 < (int)s.size()) return byte_pos + 2;
  return (int)s.size();
}

// RV: region after first vowel
int RussianStemmer::rv_start_byte(const std::string& s) {
  int i = 0;
  while (i + 1 < (int)s.size()) {
    unsigned char c1 = (unsigned char)s[(size_t)i];
    if (c1 < 0x80) {
      i = next_char_byte(s, i);
      continue;
    }
    unsigned char c2 = (unsigned char)s[(size_t)i + 1];
    if (is_vowel_pair(c1, c2)) {
      return next_char_byte(s, i);
    }
    i = next_char_byte(s, i);
  }
  return (int)s.size();
}

// R2: region after the second non-vowel following a vowel, twice (like Porter R2)
// Implementation by bytes scanning over Cyrillic letters only.
int RussianStemmer::r2_start_byte(const std::string& s) {
  // We compute R1 then R2 (classic porter):
  // R1 is region after first non-vowel following a vowel.
  // R2 is same applied to R1.
  auto r1_start = [&](int start_byte) -> int {
    bool seen_vowel = false;
    int i = start_byte;
    while (i + 1 < (int)s.size()) {
      unsigned char c1 = (unsigned char)s[(size_t)i];
      if (c1 < 0x80) { i = next_char_byte(s, i); continue; }
      unsigned char c2 = (unsigned char)s[(size_t)i + 1];
      bool v = is_vowel_pair(c1, c2);
      if (v) seen_vowel = true;
      else if (seen_vowel) return next_char_byte(s, i);
      i = next_char_byte(s, i);
    }
    return (int)s.size();
  };

  int r1 = r1_start(0);
  int r2 = r1_start(r1);
  return r2;
}

// ---------------- Suffix helpers ----------------

bool RussianStemmer::ends_with(const std::string& s, const char* suf) {
  size_t sl = s.size();
  size_t tl = std::strlen(suf);
  if (sl < tl) return false;
  return std::memcmp(s.data() + (sl - tl), suf, tl) == 0;
}

static bool ends_with_bytes(const std::string& s, const char* suf) {
  size_t sl = s.size();
  size_t tl = std::strlen(suf);
  if (sl < tl) return false;
  return std::memcmp(s.data() + (sl - tl), suf, tl) == 0;
}

static bool ends_with_from_region(const std::string& s, int region_byte, const char* suf) {
  size_t tl = std::strlen(suf);
  if ((int)s.size() - (int)tl < region_byte) return false;
  return ends_with_bytes(s, suf);
}

bool RussianStemmer::remove_suffix_any(std::string& s, int region_byte, const char* const* sufs, int n) {
  for (int i = 0; i < n; ++i) {
    if (ends_with_from_region(s, region_byte, sufs[i])) {
      s.resize(s.size() - std::strlen(sufs[i]));
      return true;
    }
  }
  return false;
}

bool RussianStemmer::remove_suffix_longest(std::string& s, int region_byte, const char* const* sufs, int n) {
  int best = -1;
  size_t best_len = 0;
  for (int i = 0; i < n; ++i) {
    const char* suf = sufs[i];
    if (ends_with_from_region(s, region_byte, suf)) {
      size_t len = std::strlen(suf);
      if (len > best_len) { best = i; best_len = len; }
    }
  }
  if (best >= 0) {
    s.resize(s.size() - best_len);
    return true;
  }
  return false;
}

bool RussianStemmer::replace_suffix(std::string& s, int region_byte, const char* suf, const char* repl) {
  if (!ends_with_from_region(s, region_byte, suf)) return false;
  size_t suf_len = std::strlen(suf);
  s.resize(s.size() - suf_len);
  s.append(repl);
  return true;
}

void RussianStemmer::trim_i(std::string& s, int region_byte) {
  // remove ending "и" (D0 B8) in RV
  const char* suf = "и";
  if (ends_with_from_region(s, region_byte, suf)) {
    s.resize(s.size() - std::strlen(suf));
  }
}

void RussianStemmer::trim_soft_sign(std::string& s, int region_byte) {
  // remove ending "ь" (D1 8C) in RV
  const char* suf = "ь";
  if (ends_with_from_region(s, region_byte, suf)) {
    s.resize(s.size() - std::strlen(suf));
  }
}

void RussianStemmer::post_process(std::string& s) {
  // minimal sanity: avoid empty, keep at least 2 chars if possible
  if (s.empty()) return;
  if (utf8_chars(s) < 2) return;
}

// ---------------- Porter Russian rules ----------------
//
// Steps (classic Russian Porter):
// 1) In RV remove perfective gerund, else reflexive, then adjectival/verb/noun
// 2) Remove "и" in RV
// 3) In R2 remove derivational ("ость", "ост")
// 4) In RV remove superlative, then "нн"->"н", then remove soft sign "ь"
//
// Note: token expected lowercase and ё->е.

std::string RussianStemmer::stem(const std::string& token) const {
  if (token.empty()) return token;

  size_t hy = token.find('-');
  if (hy != std::string::npos) {
    std::vector<std::string> parts;
    parts.clear();
    std::string cur;
    for (size_t i = 0; i < token.size(); ++i) {
      if (token[i] == '-') { parts.push_back(cur); cur.clear(); }
      else cur.push_back(token[i]);
    }
    parts.push_back(cur);

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) out.push_back('-');
      out += stem_word(parts[i]);
    }
    return out;
  }

  return stem_word(token);
}

std::string RussianStemmer::stem_word(const std::string& w) const {
  // If not russian-looking or too short — keep.
  bool has_cyr = false;
  for (size_t i = 0; i + 1 < w.size(); ++i) {
    unsigned char c1 = (unsigned char)w[i];
    if (c1 >= 0x80) { has_cyr = true; break; }
  }
  if (!has_cyr) return w;
  if (utf8_chars(w) <= 3) return w;

  std::string s = w;

  int rv = rv_start_byte(s);
  int r2 = r2_start_byte(s);

  // ---------------- step1 ----------------

  // Perfective gerund
  // Russian Porter (UTF-8 strings):
  static const char* const perfective_1[] = {
    "вшись","вши","в"
  };
  static const char* const perfective_2[] = {
    "ившись","ывшись","ивши","ывши","ив","ыв"
  };

  // Heuristic: try longer list first
  bool removed = false;
  removed = remove_suffix_longest(s, rv, perfective_2, (int)(sizeof(perfective_2)/sizeof(perfective_2[0])));
  if (!removed) {
    removed = remove_suffix_longest(s, rv, perfective_1, (int)(sizeof(perfective_1)/sizeof(perfective_1[0])));
  }

  if (!removed) {
    // Reflexive
    static const char* const reflexive[] = {"ся","сь"};
    remove_suffix_any(s, rv, reflexive, (int)(sizeof(reflexive)/sizeof(reflexive[0])));

    // Adjectival = Adjective + Participle
    static const char* const adjective[] = {
      "ими","ыми","его","ого","ему","ому","ее","ие","ые","ое",
      "ей","ий","ый","ой","ем","им","ым","ом",
      "их","ых","ую","юю","ая","яя","ою","ею"
    };

    static const char* const participle_1[] = {"ем","нн","вш","ющ","щ"};
    static const char* const participle_2[] = {"ивш","ывш","ующ"};

    std::string before = s;
    bool adj_removed = remove_suffix_longest(s, rv, adjective, (int)(sizeof(adjective)/sizeof(adjective[0])));
    if (adj_removed) {
      // try participle on the remaining
      bool part_removed = remove_suffix_longest(s, rv, participle_2, (int)(sizeof(participle_2)/sizeof(participle_2[0])));
      if (!part_removed) remove_suffix_longest(s, rv, participle_1, (int)(sizeof(participle_1)/sizeof(participle_1[0])));
      removed = true;
    } else {
      s = before;
    }

    if (!removed) {
      // Verb
      static const char* const verb_1[] = {
        "ила","ыла","ена","ейте","уйте","ите","или","ыли","ей","уй","ил","ыл","им","ым","ен","ило","ыло","ено",
        "ят","ует","уют","ит","ыт","ены","ить","ыть","ишь","ую","ю"
      };
      static const char* const verb_2[] = {
        "ла","на","ете","йте","ли","й","л","ем","н","ло","но","ет","ют","ны","ть","ешь","нно"
      };
      removed = remove_suffix_longest(s, rv, verb_1, (int)(sizeof(verb_1)/sizeof(verb_1[0])));
      if (!removed) removed = remove_suffix_longest(s, rv, verb_2, (int)(sizeof(verb_2)/sizeof(verb_2[0])));
    }

    if (!removed) {
      // Noun
      static const char* const noun[] = {
        "иями","ями","ами","ией","иям","ием","иях","ях","ию","ью","ия","ья","а","ев","ов","ие","ье","е",
        "иями","ями","ами","еи","ии","и","ией","ей","ой","ий","й","иям","ям","ием","ем","ам","ом","о","у","ах","иях","ях","ы","ь","ю"
      };
      remove_suffix_longest(s, rv, noun, (int)(sizeof(noun)/sizeof(noun[0])));
    }
  }

  // ---------------- step2 ----------------
  trim_i(s, rv);

  // ---------------- step3 ----------------
  // Derivational (in R2)
  static const char* const derivational[] = {"ост","ость"};
  remove_suffix_longest(s, r2, derivational, (int)(sizeof(derivational)/sizeof(derivational[0])));

  // ---------------- step4 ----------------
  static const char* const superlative[] = {"ейше","ейш"};
  remove_suffix_longest(s, rv, superlative, (int)(sizeof(superlative)/sizeof(superlative[0])));

  // "нн" -> "н"
  if (ends_with_from_region(s, rv, "нн")) {
    // remove one 'н' (note: UTF-8 'н' is two bytes D0 BD)
    // "нн" = "н" + "н" => remove last two bytes (one letter)
    // But because it's UTF-8, "н" length is 2 bytes.
    // So "нн" is 4 bytes. We reduce to 2 bytes by erasing 2 bytes.
    s.resize(s.size() - 2);
  }

  trim_soft_sign(s, rv);

  post_process(s);
  return s;
}
