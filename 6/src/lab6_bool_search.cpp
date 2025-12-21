#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

#pragma pack(push, 1)
struct FileHeader {
  char magic[4];
  uint32_t version;
  uint32_t flags;
  uint32_t section_count;
};
struct SectionEntry {
  uint32_t id;
  uint32_t reserved;
  uint64_t offset;
  uint64_t size;
  uint64_t aux;
};

struct DocRec {
  uint32_t docId;
  uint32_t url_off;
  uint16_t url_len;
  uint16_t pad1;
  uint32_t title_off;
  uint16_t title_len;
  uint16_t pad2;
  uint32_t text_bytes;
  uint32_t reserved;
};
struct TermRec {
  uint32_t term_off;
  uint16_t term_len;
  uint16_t pad1;
  uint32_t post_off;
  uint32_t df;
  uint32_t reserved;
};
#pragma pack(pop)

static std::string trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

static bool read_all(std::ifstream& in, uint64_t off, uint64_t size, std::vector<uint8_t>& out) {
  out.resize((size_t)size);
  in.seekg((std::streamoff)off, std::ios::beg);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(out.data()), (std::streamsize)size);
  return (bool)in;
}

static uint32_t read_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

static uint32_t read_varint_u32(const std::vector<uint8_t>& buf, size_t& pos) {
  uint32_t v = 0;
  uint32_t shift = 0;
  while (pos < buf.size()) {
    uint8_t b = buf[pos++];
    v |= (uint32_t)(b & 0x7Fu) << shift;
    if ((b & 0x80u) == 0) break;
    shift += 7u;
  }
  return v;
}

static std::string get_str(const std::vector<uint8_t>& strs, uint32_t off, uint16_t len) {
  if (off + len > strs.size()) return "";
  return std::string((const char*)strs.data() + off, (size_t)len);
}

static int term_cmp(const std::vector<uint8_t>& strs, const TermRec& t, const std::string& key) {
  std::string s = get_str(strs, t.term_off, t.term_len);
  if (s == key) return 0;
  return (s < key) ? -1 : 1;
}

static int lex_find(const std::vector<uint8_t>& strs, const std::vector<TermRec>& terms, const std::string& key) {
  int l = 0, r = (int)terms.size() - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    int c = term_cmp(strs, terms[(size_t)m], key);
    if (c == 0) return m;
    if (c < 0) l = m + 1;
    else r = m - 1;
  }
  return -1;
}

static void postings_for_term(const std::vector<uint8_t>& posts_payload,
                              const TermRec& tr,
                              std::vector<uint32_t>& out_docs) {
  out_docs.clear();

  const uint8_t* p = posts_payload.data();
  uint32_t bytes = read_u32(p);
  (void)bytes;

  size_t pos = 4 + (size_t)tr.post_off;
  uint32_t prev = 0;
  for (uint32_t i = 0; i < tr.df; ++i) {
    uint32_t gap = read_varint_u32(posts_payload, pos);
    uint32_t d = (i == 0) ? (gap - 1) : (prev + gap);
    out_docs.push_back(d);
    prev = d;
  }
}

static void op_and(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out) {
  out.clear();
  size_t i=0,j=0;
  while (i<a.size() && j<b.size()) {
    if (a[i]==b[j]) { out.push_back(a[i]); i++; j++; }
    else if (a[i]<b[j]) i++;
    else j++;
  }
}
static void op_or(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out) {
  out.clear();
  size_t i=0,j=0;
  while (i<a.size() || j<b.size()) {
    if (j>=b.size() || (i<a.size() && a[i]<b[j])) out.push_back(a[i++]);
    else if (i>=a.size() || b[j]<a[i]) out.push_back(b[j++]);
    else { out.push_back(a[i]); i++; j++; }
  }
}
static void op_not(const std::vector<uint32_t>& universe, const std::vector<uint32_t>& a, std::vector<uint32_t>& out) {
  out.clear();
  size_t i=0,j=0;
  while (i<universe.size()) {
    if (j>=a.size() || universe[i] < a[j]) out.push_back(universe[i++]);
    else if (universe[i] == a[j]) { i++; j++; }
    else j++;
  }
}

static bool is_ascii_alnum(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static std::string strip_token_edges(const std::string& tok) {
  if (tok.empty()) return tok;
  size_t l = 0;
  size_t r = tok.size();

  auto ok = [](unsigned char c) -> bool {
    return is_ascii_alnum(c) || c >= 0x80;
  };

  while (l < r && !ok((unsigned char)tok[l])) l++;
  while (r > l && !ok((unsigned char)tok[r - 1])) r--;

  return tok.substr(l, r - l);
}

static std::string upper_ascii(const std::string& s) {
  std::string out = s;
  for (size_t i = 0; i < out.size(); ++i) {
    unsigned char c = (unsigned char)out[i];
    if (c >= 'a' && c <= 'z') out[i] = (char)(c - 'a' + 'A');
  }
  return out;
}


static std::vector<std::string> split_ws(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (size_t i=0;i<s.size();++i) {
    char c=s[i];
    if (c==' '||c=='\t'||c=='\r'||c=='\n') {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    } else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

static std::string lower_ascii(const std::string& s) {
  std::string out = s;
  for (size_t i=0;i<out.size();++i) {
    unsigned char c = (unsigned char)out[i];
    if (c>='A' && c<='Z') out[i]=(char)(c - 'A' + 'a');
  }
  return out;
}

static std::string lower_ru_utf8(const std::string& s) {
  std::string out;
  out.reserve(s.size());

  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      if (c >= 'A' && c <= 'Z') out.push_back((char)(c - 'A' + 'a'));
      else out.push_back((char)c);
      i += 1;
      continue;
    }

    if (i + 1 < s.size()) {
      unsigned char c2 = (unsigned char)s[i + 1];

      if (c == 0xD0 && c2 >= 0x90 && c2 <= 0xAF) {
        out.push_back((char)0xD0);
        out.push_back((char)(c2 + 0x20));
        i += 2;
        continue;
      }
      if (c == 0xD0 && c2 == 0x81) { // Ё
        out.push_back((char)0xD0);
        out.push_back((char)0xB5);
        i += 2;
        continue;
      }

      out.push_back((char)c);
      out.push_back((char)c2);
      i += 2;
      continue;
    }

    out.push_back((char)c);
    break;
  }

  return out;
}

static void print_sample(const std::vector<uint32_t>& res,
                         const std::vector<DocRec>& docs,
                         const std::vector<uint8_t>& strs) {
  std::cout << "hits: " << res.size() << "\n";
  if (res.empty()) { std::cout << "\n"; return; }

  if (res.size() <= 10) {
    for (size_t i = 0; i < res.size(); ++i) {
      uint32_t d = res[i];
      const DocRec& dr = docs[(size_t)d];
      std::string title = get_str(strs, dr.title_off, dr.title_len);
      std::string url = get_str(strs, dr.url_off, dr.url_len);
      std::cout << d << "\t" << title << "\t" << url << "\n";
    }
    std::cout << "\n";
    return;
  }

  size_t n = res.size();
  size_t mid = n / 2;

  std::vector<size_t> idx;
  idx.reserve(9);

  idx.push_back(0); idx.push_back(1); idx.push_back(2);

  if (mid > 0) idx.push_back(mid - 1);
  idx.push_back(mid);
  if (mid + 1 < n) idx.push_back(mid + 1);

  idx.push_back(n - 3); idx.push_back(n - 2); idx.push_back(n - 1);

  std::vector<size_t> uniq;
  uniq.reserve(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    bool seen = false;
    for (size_t j = 0; j < uniq.size(); ++j) {
      if (uniq[j] == idx[i]) { seen = true; break; }
    }
    if (!seen) uniq.push_back(idx[i]);
  }

  for (size_t k = 0; k < uniq.size(); ++k) {
    if (k == 3) std::cout << "...\n";
    if (k == 6) std::cout << "...\n";

    uint32_t d = res[uniq[k]];
    const DocRec& dr = docs[(size_t)d];
    std::string title = get_str(strs, dr.title_off, dr.title_len);
    std::string url = get_str(strs, dr.url_off, dr.url_len);
    std::cout << d << "\t" << title << "\t" << url << "\n";
  }
  std::cout << "\n";
}


int main(int argc, char** argv) {
  std::string path = (argc >= 2) ? argv[1] : "index.bidx";

  std::ifstream in(path, std::ios::binary);
  if (!in) { std::cerr << "Cannot open: " << path << "\n"; return 1; }

  FileHeader hdr{};
  in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!in || std::memcmp(hdr.magic, "BIDX", 4) != 0) {
    std::cerr << "Bad index file\n"; return 2;
  }

  std::vector<SectionEntry> secs(hdr.section_count);
  in.read(reinterpret_cast<char*>(secs.data()), (std::streamsize)(secs.size()*sizeof(SectionEntry)));

  auto find_sec = [&](uint32_t id)->SectionEntry {
    for (size_t i=0;i<secs.size();++i) if (secs[i].id==id) return secs[i];
    return SectionEntry{};
  };

  SectionEntry s_strs = find_sec(1);
  SectionEntry s_docs = find_sec(2);
  SectionEntry s_terms = find_sec(3);
  SectionEntry s_posts = find_sec(4);

  std::vector<uint8_t> strs_payload, docs_payload, terms_payload, posts_payload;
  read_all(in, s_strs.offset, s_strs.size, strs_payload);
  read_all(in, s_docs.offset, s_docs.size, docs_payload);
  read_all(in, s_terms.offset, s_terms.size, terms_payload);
  read_all(in, s_posts.offset, s_posts.size, posts_payload);

  uint32_t strs_bytes = read_u32(strs_payload.data());
  std::vector<uint8_t> strs;
  strs.assign(strs_payload.begin() + 4, strs_payload.begin() + 4 + strs_bytes);

  uint32_t doc_count = read_u32(docs_payload.data());
  std::vector<DocRec> docs(doc_count);
  if (doc_count > 0) {
    std::memcpy(docs.data(), docs_payload.data() + 4, (size_t)doc_count * sizeof(DocRec));
  }

  uint32_t term_count = read_u32(terms_payload.data());
  std::vector<TermRec> terms(term_count);
  if (term_count > 0) {
    std::memcpy(terms.data(), terms_payload.data() + 4, (size_t)term_count * sizeof(TermRec));
  }

  std::vector<uint32_t> universe;
  universe.reserve(doc_count);
  for (uint32_t i=0;i<doc_count;++i) universe.push_back(i);

  std::cout << "=== LAB6 BOOL SEARCH ===\n";
  std::cout << "docs: " << doc_count << " terms: " << term_count << "\n";
  std::cout << "Query format: term AND term OR term NOT term (operators uppercase). Default op = AND.\n";
  std::cout << "Example: курица AND помидоры\n\n";

  while (true) {
    std::cout << "> ";
    std::string q;
    if (!std::getline(std::cin, q)) break;
    q = trim(q);
    if (q.empty()) break;

    std::vector<std::string> qt = split_ws(q);

for (size_t i = 0; i < qt.size(); ++i) {
  std::string t = strip_token_edges(qt[i]);
  if (t.empty()) { qt[i].clear(); continue; }

  std::string up = upper_ascii(t);
  if (up == "AND" || up == "OR" || up == "NOT") {
    qt[i] = up;
  } else {
    qt[i] = lower_ru_utf8(t);
  }
}

{
  std::vector<std::string> cleaned;
  cleaned.reserve(qt.size());
  for (size_t i = 0; i < qt.size(); ++i) {
    if (!qt[i].empty()) cleaned.push_back(qt[i]);
  }
  qt.swap(cleaned);
}

    std::vector<std::vector<uint32_t>> parts;
    std::vector<std::string> ops;

    for (size_t i=0;i<qt.size();) {
      bool neg = false;
      if (qt[i] == "NOT") { neg = true; i++; }
      if (i >= qt.size()) break;

      std::string term = qt[i++];
      int idx = lex_find(strs, terms, term);

      std::vector<uint32_t> docs_term;
      if (idx >= 0) postings_for_term(posts_payload, terms[(size_t)idx], docs_term);

      if (neg) {
        std::vector<uint32_t> negv;
        op_not(universe, docs_term, negv);
        parts.push_back(negv);
      } else {
        parts.push_back(docs_term);
      }

      if (i < qt.size()) {
        if (qt[i] == "AND" || qt[i] == "OR") {
          ops.push_back(qt[i]);
          i++;
        } else {
          ops.push_back("AND");
        }
      }
    }

    std::vector<std::vector<uint32_t>> parts2;
    std::vector<std::string> ops2;
    if (!parts.empty()) {
      std::vector<uint32_t> cur = parts[0];
      for (size_t i=0;i<ops.size();++i) {
        if (ops[i] == "AND") {
          std::vector<uint32_t> tmp;
          op_and(cur, parts[i+1], tmp);
          cur.swap(tmp);
        } else {
          parts2.push_back(cur);
          ops2.push_back("OR");
          cur = parts[i+1];
        }
      }
      parts2.push_back(cur);
    }

    std::vector<uint32_t> res;
    if (!parts2.empty()) {
      res = parts2[0];
      for (size_t i=1;i<parts2.size();++i) {
        std::vector<uint32_t> tmp;
        op_or(res, parts2[i], tmp);
        res.swap(tmp);
      }
    }

    print_sample(res, docs, strs);
    std::cout << "\n";
  }

  return 0;
}
