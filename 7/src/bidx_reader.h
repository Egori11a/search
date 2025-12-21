#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

// ====== Index binary structs (same as Lab6) ======
#pragma pack(push, 1)
struct BidxHeader {
  char magic[4];      // "BIDX"
  uint32_t version;   // 1
  uint32_t flags;     // 0
  uint32_t section_count;
};

struct BidxSection {
  uint32_t id;        // 1=STRS 2=DOCS 3=TERMS 4=POSTS
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
  uint32_t post_off; // offset in POSTS payload (after leading u32 bytes)
  uint32_t df;
  uint32_t reserved;
};
#pragma pack(pop)

// ====== Helpers ======
static inline uint32_t read_u32(const uint8_t* p) {
  uint32_t v; std::memcpy(&v, p, 4); return v;
}

static inline bool read_all(std::ifstream& in, uint64_t off, uint64_t size, std::vector<uint8_t>& out) {
  out.resize((size_t)size);
  in.seekg((std::streamoff)off, std::ios::beg);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(out.data()), (std::streamsize)size);
  return (bool)in;
}

static inline std::string trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

static inline bool is_ascii_alnum(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// strip punctuation from edges, keep UTF-8 bytes >=0x80
static inline std::string strip_token_edges(const std::string& tok) {
  if (tok.empty()) return tok;
  size_t l = 0, r = tok.size();
  auto ok = [](unsigned char c) -> bool { return is_ascii_alnum(c) || c >= 0x80; };
  while (l < r && !ok((unsigned char)tok[l])) l++;
  while (r > l && !ok((unsigned char)tok[r-1])) r--;
  return tok.substr(l, r - l);
}

// AND/OR/NOT are ASCII; make upper for tolerance
static inline std::string upper_ascii(const std::string& s) {
  std::string out = s;
  for (size_t i = 0; i < out.size(); ++i) {
    unsigned char c = (unsigned char)out[i];
    if (c >= 'a' && c <= 'z') out[i] = (char)(c - 'a' + 'A');
  }
  return out;
}

// Lowercase Russian UTF-8 + Ё->е normalization (matches your tokenizer logic expectation)
static inline std::string lower_ru_utf8_yo_to_e(const std::string& s) {
  std::string out;
  out.reserve(s.size());

  for (size_t i = 0; i < s.size();) {
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
      // ASCII
      if (c >= 'A' && c <= 'Z') out.push_back((char)(c - 'A' + 'a'));
      else out.push_back((char)c);
      i += 1;
      continue;
    }
    if (i + 1 < s.size()) {
      unsigned char c2 = (unsigned char)s[i + 1];

      // Ё (D0 81) -> е (D0 B5)
      if (c == 0xD0 && c2 == 0x81) {
        out.push_back((char)0xD0);
        out.push_back((char)0xB5);
        i += 2;
        continue;
      }

      // А..Я (D0 90..AF) -> а..п (D0 B0..BF)
      if (c == 0xD0 && c2 >= 0x90 && c2 <= 0xAF) {
        out.push_back((char)0xD0);
        out.push_back((char)(c2 + 0x20));
        i += 2;
        continue;
      }

      // add 2 bytes as-is
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

// ====== varint decode (same as lab6) ======
static inline uint32_t read_varint_u32(const std::vector<uint8_t>& buf, size_t& pos) {
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

// ====== Index container ======
struct BidxIndex {
  std::vector<uint8_t> strs;        // raw string bytes
  std::vector<DocRec> docs;
  std::vector<TermRec> terms;
  std::vector<uint8_t> posts_payload; // [u32 bytes][bytes...]

  uint32_t doc_count = 0;

  std::string get_str(uint32_t off, uint16_t len) const {
    if ((size_t)off + (size_t)len > strs.size()) return "";
    return std::string((const char*)strs.data() + off, (size_t)len);
  }
};

static inline int term_cmp(const BidxIndex& idx, const TermRec& t, const std::string& key) {
  std::string s = idx.get_str(t.term_off, t.term_len);
  if (s == key) return 0;
  return (s < key) ? -1 : 1;
}

static inline int lex_find(const BidxIndex& idx, const std::string& key) {
  int l = 0, r = (int)idx.terms.size() - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    int c = term_cmp(idx, idx.terms[(size_t)m], key);
    if (c == 0) return m;
    if (c < 0) l = m + 1;
    else r = m - 1;
  }
  return -1;
}

// decode postings for term index -> sorted docIds
static inline void postings_for_term(const BidxIndex& idx, const TermRec& tr, std::vector<uint32_t>& out_docs) {
  out_docs.clear();
  if (idx.posts_payload.size() < 4) return;

  size_t pos = 4 + (size_t)tr.post_off;
  uint32_t prev = 0;
  out_docs.reserve(tr.df);

  for (uint32_t i = 0; i < tr.df; ++i) {
    uint32_t gap = read_varint_u32(idx.posts_payload, pos);
    uint32_t d = (i == 0) ? (gap - 1) : (prev + gap);
    out_docs.push_back(d);
    prev = d;
  }
}

// sorted set ops (no maps)
static inline void op_and(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out) {
  out.clear();
  size_t i=0,j=0;
  while (i<a.size() && j<b.size()) {
    if (a[i]==b[j]) { out.push_back(a[i]); i++; j++; }
    else if (a[i]<b[j]) i++;
    else j++;
  }
}
static inline void op_or(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, std::vector<uint32_t>& out) {
  out.clear();
  size_t i=0,j=0;
  while (i<a.size() || j<b.size()) {
    if (j>=b.size() || (i<a.size() && a[i]<b[j])) out.push_back(a[i++]);
    else if (i>=a.size() || b[j]<a[i]) out.push_back(b[j++]);
    else { out.push_back(a[i]); i++; j++; }
  }
}
static inline void op_not_universe(uint32_t doc_count, const std::vector<uint32_t>& a, std::vector<uint32_t>& out) {
  out.clear();
  size_t j = 0;
  for (uint32_t d = 0; d < doc_count; ++d) {
    while (j < a.size() && a[j] < d) j++;
    if (j < a.size() && a[j] == d) continue;
    out.push_back(d);
  }
}

// ====== load index.bidx ======
static inline bool load_bidx(const std::string& path, BidxIndex& idx) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  BidxHeader hdr{};
  in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!in) return false;
  if (std::memcmp(hdr.magic, "BIDX", 4) != 0) return false;

  std::vector<BidxSection> secs(hdr.section_count);
  in.read(reinterpret_cast<char*>(secs.data()), (std::streamsize)(secs.size() * sizeof(BidxSection)));
  if (!in) return false;

  auto find_sec = [&](uint32_t id)->BidxSection {
    for (size_t i=0;i<secs.size();++i) if (secs[i].id==id) return secs[i];
    return BidxSection{};
  };

  BidxSection s_strs = find_sec(1);
  BidxSection s_docs = find_sec(2);
  BidxSection s_terms = find_sec(3);
  BidxSection s_posts = find_sec(4);

  std::vector<uint8_t> strs_payload, docs_payload, terms_payload, posts_payload;
  if (!read_all(in, s_strs.offset, s_strs.size, strs_payload)) return false;
  if (!read_all(in, s_docs.offset, s_docs.size, docs_payload)) return false;
  if (!read_all(in, s_terms.offset, s_terms.size, terms_payload)) return false;
  if (!read_all(in, s_posts.offset, s_posts.size, posts_payload)) return false;

  // STRS: [u32 bytes][bytes...]
  if (strs_payload.size() < 4) return false;
  uint32_t strs_bytes = read_u32(strs_payload.data());
  if ((size_t)4 + strs_bytes > strs_payload.size()) return false;
  idx.strs.assign(strs_payload.begin() + 4, strs_payload.begin() + 4 + strs_bytes);

  // DOCS: [u32 count][DocRec...]
  if (docs_payload.size() < 4) return false;
  uint32_t doc_count = read_u32(docs_payload.data());
  idx.doc_count = doc_count;
  idx.docs.resize(doc_count);
  if (doc_count > 0) {
    size_t need = 4 + (size_t)doc_count * sizeof(DocRec);
    if (need > docs_payload.size()) return false;
    std::memcpy(idx.docs.data(), docs_payload.data() + 4, (size_t)doc_count * sizeof(DocRec));
  }

  // TERMS: [u32 count][TermRec...]
  if (terms_payload.size() < 4) return false;
  uint32_t term_count = read_u32(terms_payload.data());
  idx.terms.resize(term_count);
  if (term_count > 0) {
    size_t need = 4 + (size_t)term_count * sizeof(TermRec);
    if (need > terms_payload.size()) return false;
    std::memcpy(idx.terms.data(), terms_payload.data() + 4, (size_t)term_count * sizeof(TermRec));
  }

  idx.posts_payload.swap(posts_payload);
  return true;
}

// ====== LAB7 query parsing ======
// Syntax:
//  - spaces or "&&" => AND
//  - "||" => OR
//  - "!" => NOT
//  - parentheses ( )
// tolerant:
//  - variable whitespace
//  - accept single '&' or '|' as well
//  - ignore junk punctuation around terms
enum TokType { TT_TERM, TT_AND, TT_OR, TT_NOT, TT_LP, TT_RP };

struct Tok {
  TokType type;
  std::string text; // for TT_TERM
};

static inline void push_implicit_and_if_needed(std::vector<Tok>& out, TokType nextType) {
  if (out.empty()) return;
  TokType prev = out.back().type;
  bool prevIsValue = (prev == TT_TERM || prev == TT_RP);
  bool nextIsValue = (nextType == TT_TERM || nextType == TT_LP || nextType == TT_NOT);
  if (prevIsValue && nextIsValue) out.push_back({TT_AND, ""});
}

static inline std::vector<Tok> lex_query(const std::string& q) {
  std::vector<Tok> out;
  std::string cur;

  auto flush_term = [&]() {
    if (cur.empty()) return;
    std::string t = strip_token_edges(cur);
    cur.clear();
    if (t.empty()) return;

    std::string up = upper_ascii(t);
    if (up == "AND") { push_implicit_and_if_needed(out, TT_AND); out.push_back({TT_AND,""}); return; }
    if (up == "OR")  { push_implicit_and_if_needed(out, TT_OR);  out.push_back({TT_OR ,""}); return; }
    if (up == "NOT") { push_implicit_and_if_needed(out, TT_NOT); out.push_back({TT_NOT,""}); return; }

    // normal term
    t = lower_ru_utf8_yo_to_e(t);
    push_implicit_and_if_needed(out, TT_TERM);
    out.push_back({TT_TERM, t});
  };

  for (size_t i = 0; i < q.size();) {
    unsigned char c = (unsigned char)q[i];

    // whitespace
    if (c==' ' || c=='\t' || c=='\r' || c=='\n') {
      flush_term();
      i++;
      continue;
    }

    // parentheses
    if (c=='(') { flush_term(); push_implicit_and_if_needed(out, TT_LP); out.push_back({TT_LP,""}); i++; continue; }
    if (c==')') { flush_term(); out.push_back({TT_RP,""}); i++; continue; }

    // NOT
    if (c=='!') { flush_term(); push_implicit_and_if_needed(out, TT_NOT); out.push_back({TT_NOT,""}); i++; continue; }

    // && and ||
    if (c=='&') {
      flush_term();
      if (i+1<q.size() && q[i+1]=='&') i+=2; else i+=1; // tolerate single '&'
      out.push_back({TT_AND,""});
      continue;
    }
    if (c=='|') {
      flush_term();
      if (i+1<q.size() && q[i+1]=='|') i+=2; else i+=1; // tolerate single '|'
      out.push_back({TT_OR,""});
      continue;
    }

    // otherwise: part of term token
    cur.push_back((char)c);
    i++;
  }
  flush_term();

  // clean up: avoid AND/OR at start/end, and collapse AND/OR duplicates
  std::vector<Tok> cleaned;
  cleaned.reserve(out.size());
  for (size_t i=0;i<out.size();++i) {
    Tok t = out[i];
    if (cleaned.empty()) {
      if (t.type == TT_AND || t.type == TT_OR) continue;
    }
    if (!cleaned.empty()) {
      TokType p = cleaned.back().type;
      if ((p==TT_AND || p==TT_OR) && (t.type==TT_AND || t.type==TT_OR)) {
        cleaned.back() = t; // keep last operator
        continue;
      }
      if ((p==TT_LP) && (t.type==TT_AND || t.type==TT_OR)) continue;
    }
    cleaned.push_back(t);
  }
  while (!cleaned.empty() && (cleaned.back().type==TT_AND || cleaned.back().type==TT_OR)) cleaned.pop_back();
  return cleaned;
}

static inline int prec(TokType t) {
  if (t == TT_NOT) return 3;
  if (t == TT_AND) return 2;
  if (t == TT_OR)  return 1;
  return 0;
}
static inline bool right_assoc(TokType t) {
  return (t == TT_NOT);
}

// shunting-yard -> RPN
static inline std::vector<Tok> to_rpn(const std::vector<Tok>& toks) {
  std::vector<Tok> out;
  std::vector<Tok> st;
  out.reserve(toks.size());
  st.reserve(toks.size());

  for (size_t i=0;i<toks.size();++i) {
    Tok t = toks[i];
    if (t.type == TT_TERM) {
      out.push_back(t);
      continue;
    }
    if (t.type == TT_LP) {
      st.push_back(t);
      continue;
    }
    if (t.type == TT_RP) {
      while (!st.empty() && st.back().type != TT_LP) {
        out.push_back(st.back());
        st.pop_back();
      }
      if (!st.empty() && st.back().type == TT_LP) st.pop_back(); // pop '('
      continue;
    }
    // operator
    if (t.type == TT_AND || t.type == TT_OR || t.type == TT_NOT) {
      while (!st.empty()) {
        TokType top = st.back().type;
        if (top==TT_LP) break;
        int p1 = prec(t.type);
        int p2 = prec(top);
        if (p2 > p1 || (p2 == p1 && !right_assoc(t.type))) {
          out.push_back(st.back());
          st.pop_back();
        } else break;
      }
      st.push_back(t);
    }
  }
  while (!st.empty()) {
    if (st.back().type != TT_LP) out.push_back(st.back());
    st.pop_back();
  }
  return out;
}

static inline void eval_rpn(const BidxIndex& idx, const std::vector<Tok>& rpn, std::vector<uint32_t>& out_docs) {
  std::vector<std::vector<uint32_t>> st;
  st.reserve(32);

  std::vector<uint32_t> tmp, tmp2;

  for (size_t i=0;i<rpn.size();++i) {
    const Tok& t = rpn[i];
    if (t.type == TT_TERM) {
      int pos = lex_find(idx, t.text);
      std::vector<uint32_t> docs;
      if (pos >= 0) postings_for_term(idx, idx.terms[(size_t)pos], docs);
      st.push_back(docs);
      continue;
    }
    if (t.type == TT_NOT) {
      if (st.empty()) continue;
      std::vector<uint32_t> a = st.back(); st.pop_back();
      op_not_universe(idx.doc_count, a, tmp);
      st.push_back(tmp);
      continue;
    }
    if (t.type == TT_AND || t.type == TT_OR) {
      if (st.size() < 2) continue;
      std::vector<uint32_t> b = st.back(); st.pop_back();
      std::vector<uint32_t> a = st.back(); st.pop_back();
      if (t.type == TT_AND) op_and(a, b, tmp2);
      else op_or(a, b, tmp2);
      st.push_back(tmp2);
      continue;
    }
  }

  if (st.empty()) {
    out_docs.clear();
    return;
  }

  // If multiple leftovers, AND them (tolerance)
  std::vector<uint32_t> res = st[0];
  for (size_t i=1;i<st.size();++i) {
    op_and(res, st[i], tmp);
    res.swap(tmp);
  }
  out_docs.swap(res);
}

// main entry: parse + evaluate
static inline void boolean_search(const BidxIndex& idx, const std::string& query, std::vector<uint32_t>& out_docs) {
  std::vector<Tok> toks = lex_query(query);
  std::vector<Tok> rpn = to_rpn(toks);
  eval_rpn(idx, rpn, out_docs);
}
