#include "tokenizer.h"

#include <mongoc/mongoc.h>
#include <bson/bson.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <cstring>

static std::string trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

static bool starts_with(const std::string& s, const char* pref) {
  size_t n = ::strlen(pref);
  if (s.size() < n) return false;
  return s.compare(0, n, pref) == 0;
}

static bool strip_quotes(std::string& v) {
  v = trim(v);
  if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front()=='\'' && v.back()=='\''))) {
    v = v.substr(1, v.size()-2);
    return true;
  }
  return false;
}

static bool parse_yaml_kv(const std::string& line, std::string& key, std::string& val) {
  std::string t = line;
  size_t hash = t.find('#');
  if (hash != std::string::npos) t = t.substr(0, hash);
  t = trim(t);
  if (t.empty()) return false;

  size_t colon = t.find(':');
  if (colon == std::string::npos) return false;

  key = trim(t.substr(0, colon));
  val = trim(t.substr(colon + 1));
  return !key.empty();
}

static int to_int_safe(const std::string& s, int defv) {
  try { return std::stoi(trim(s)); } catch (...) { return defv; }
}

static bool to_bool_safe(std::string v, bool defv) {
  v = trim(v);
  for (size_t i=0;i<v.size();++i) if (v[i]>='A' && v[i]<='Z') v[i] = char(v[i]-'A'+'a');
  if (v=="true" || v=="yes" || v=="1") return true;
  if (v=="false"|| v=="no"  || v=="0") return false;
  return defv;
}

struct MongoCfg {
  std::string uri = "mongodb://localhost:27017";
  std::string database = "crawler_db";
  std::string docs_collection = "docs";
};

struct RunCfg {
  int max_docs = 2000;
  bool lowercase = true;
  bool normalize_yo = true;
  bool keep_numbers = true;
  int min_len = 2;
};

static bool load_cfg_minimal(const std::string& path, MongoCfg& m, RunCfg& r) {
  std::ifstream in(path);
  if (!in) return false;

  std::string section;
  std::string line;
  while (std::getline(in, line)) {
    std::string t = trim(line);
    if (t.empty() || t[0] == '#') continue;

    if (t.back() == ':' && t.find(' ') == std::string::npos) {
      section = t.substr(0, t.size()-1);
      continue;
    }

    if (t.find(':') == std::string::npos) continue;

    std::string key, val;
    if (!parse_yaml_kv(t, key, val)) continue;

    std::string vv = val;
    strip_quotes(vv);

    if (section == "db") {
      if (key == "uri") m.uri = vv;
      else if (key == "database") m.database = vv;
      else if (key == "docs_collection") m.docs_collection = vv;
    } else if (section == "corpus") {
      if (key == "max_docs") r.max_docs = to_int_safe(vv, r.max_docs);
    } else if (section == "tokenizer") {
      if (key == "lowercase") r.lowercase = to_bool_safe(vv, r.lowercase);
      else if (key == "normalize_yo") r.normalize_yo = to_bool_safe(vv, r.normalize_yo);
      else if (key == "keep_numbers") r.keep_numbers = to_bool_safe(vv, r.keep_numbers);
      else if (key == "min_len") r.min_len = to_int_safe(vv, r.min_len);
    }
  }

  if (r.max_docs <= 0) r.max_docs = 1;
  if (r.min_len < 1) r.min_len = 1;
  return true;
}

static std::string read_title_fallback_from_text(const std::string& text) {
  if (text.empty()) return "";
  size_t p = text.find('\n');
  std::string line = (p == std::string::npos) ? text : text.substr(0, p);
  line = trim(line);
  if (line.size() >= 5) {
    if (line.size() > 120) line.resize(120);
    return line;
  }
  std::string t = trim(text);
  if (t.size() > 120) t.resize(120);
  return t;
}

static void write_varint_u32(std::vector<uint8_t>& out, uint32_t v) {
  while (v >= 0x80u) {
    out.push_back((uint8_t)((v & 0x7Fu) | 0x80u));
    v >>= 7u;
  }
  out.push_back((uint8_t)(v & 0x7Fu));
}

struct PairTD {
  std::string term;
  uint32_t doc;
};

static void merge_sort_strings(std::vector<std::string>& a, std::vector<std::string>& tmp, int l, int r) {
  if (r - l <= 1) return;
  int m = (l + r) / 2;
  merge_sort_strings(a, tmp, l, m);
  merge_sort_strings(a, tmp, m, r);
  int i = l, j = m, k = l;
  while (i < m && j < r) {
    if (a[i] <= a[j]) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];
  for (int p = l; p < r; ++p) a[p] = tmp[p];
}

static void merge_sort_pairtd(std::vector<PairTD>& a, std::vector<PairTD>& tmp, int l, int r) {
  if (r - l <= 1) return;
  int m = (l + r) / 2;
  merge_sort_pairtd(a, tmp, l, m);
  merge_sort_pairtd(a, tmp, m, r);

  int i = l, j = m, k = l;
  while (i < m && j < r) {
    const PairTD& x = a[i];
    const PairTD& y = a[j];
    bool le = (x.term < y.term) || (x.term == y.term && x.doc <= y.doc);
    if (le) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];
  for (int p = l; p < r; ++p) a[p] = tmp[p];
}

struct StrRef {
  uint32_t off;
  uint16_t len;
};

struct StringTable {
  std::vector<uint8_t> data;

  StrRef add(const std::string& s) {
    StrRef r;
    r.off = (uint32_t)data.size();
    r.len = (uint16_t)((s.size() > 65535) ? 65535 : s.size());
    data.insert(data.end(), s.begin(), s.begin() + r.len);
    return r;
  }
};

#pragma pack(push, 1)
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
#pragma pack(pop)

static void write_u32(std::ofstream& out, uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static void write_bytes(std::ofstream& out, const void* p, size_t n) {
  out.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
}


struct MongoDoc {
  std::string url;
  std::string title;
  std::string text;
  std::string source;
};

static bool bson_get_utf8_safe(const bson_t* doc, const char* key, std::string& out) {
  bson_iter_t it;
  if (!bson_iter_init_find(&it, doc, key)) return false;
  if (BSON_ITER_HOLDS_UTF8(&it)) {
    uint32_t len = 0;
    const char* s = bson_iter_utf8(&it, &len);
    out.assign(s, s + len);
    return true;
  }
  return false;
}

static bool load_docs_from_mongo(const MongoCfg& m, int max_docs, std::vector<MongoDoc>& out) {
  out.clear();

  mongoc_init();

  mongoc_client_t* client = mongoc_client_new(m.uri.c_str());
  if (!client) {
    std::cerr << "Mongo: cannot create client for uri=" << m.uri << "\n";
    return false;
  }

  mongoc_collection_t* coll = mongoc_client_get_collection(client, m.database.c_str(), m.docs_collection.c_str());
  if (!coll) {
    std::cerr << "Mongo: cannot open collection " << m.database << "." << m.docs_collection << "\n";
    mongoc_client_destroy(client);
    return false;
  }

  bson_t query;
  bson_init(&query);

  bson_t proj;
  bson_init(&proj);
  BSON_APPEND_INT32(&proj, "url", 1);
  BSON_APPEND_INT32(&proj, "title", 1);
  BSON_APPEND_INT32(&proj, "text", 1);
  BSON_APPEND_INT32(&proj, "source", 1);

  bson_t opts;
  bson_init(&opts);
  BSON_APPEND_DOCUMENT(&opts, "projection", &proj);
  BSON_APPEND_INT64(&opts, "limit", (int64_t)max_docs);

  bson_t sort;
  bson_init(&sort);
  BSON_APPEND_INT32(&sort, "_id", 1);
  BSON_APPEND_DOCUMENT(&opts, "sort", &sort);

  mongoc_cursor_t* cur = mongoc_collection_find_with_opts(coll, &query, &opts, nullptr);

  const bson_t* doc;
  while (mongoc_cursor_next(cur, &doc)) {
    MongoDoc d;
    bson_get_utf8_safe(doc, "url", d.url);
    bson_get_utf8_safe(doc, "title", d.title);
    bson_get_utf8_safe(doc, "text", d.text);
    bson_get_utf8_safe(doc, "source", d.source);

    if (d.text.empty()) continue;

    out.push_back(d);
    if ((int)out.size() >= max_docs) break;
  }

  bson_error_t err;
  if (mongoc_cursor_error(cur, &err)) {
    std::cerr << "Mongo cursor error: " << err.message << "\n";
    mongoc_cursor_destroy(cur);
    bson_destroy(&sort);
    bson_destroy(&opts);
    bson_destroy(&proj);
    bson_destroy(&query);
    mongoc_collection_destroy(coll);
    mongoc_client_destroy(client);
    return false;
  }

  mongoc_cursor_destroy(cur);
  bson_destroy(&sort);
  bson_destroy(&opts);
  bson_destroy(&proj);
  bson_destroy(&query);

  mongoc_collection_destroy(coll);
  mongoc_client_destroy(client);
  return true;
}


int main(int argc, char** argv) {
  std::string cfg_path = (argc >= 2) ? argv[1] : "config.yaml";
  std::string out_path = (argc >= 3) ? argv[2] : "index.bidx";

  MongoCfg mcfg;
  RunCfg rcfg;
  if (!load_cfg_minimal(cfg_path, mcfg, rcfg)) {
    std::cerr << "Failed to load config: " << cfg_path << "\n";
    return 1;
  }

  std::vector<MongoDoc> mdocs;
  if (!load_docs_from_mongo(mcfg, rcfg.max_docs, mdocs)) {
    std::cerr << "Failed to load docs from Mongo\n";
    return 2;
  }
  if (mdocs.empty()) {
    std::cerr << "No docs loaded from Mongo (collection empty?)\n";
    return 3;
  }

  TokenizerConfig tc;
  tc.lowercase = rcfg.lowercase;
  tc.normalize_yo = rcfg.normalize_yo;
  tc.keep_numbers = rcfg.keep_numbers;
  tc.min_len = rcfg.min_len;
  Tokenizer tokenizer(tc);

  uint32_t doc_count = (uint32_t)mdocs.size();

  StringTable strs;
  std::vector<DocRec> docs;
  docs.reserve(doc_count);

  std::vector<PairTD> pairs;
  pairs.reserve(1500000);

  std::vector<std::string> tokens;

  uint64_t total_text_bytes = 0;

  auto t0 = std::chrono::steady_clock::now();

  for (uint32_t docId = 0; docId < doc_count; ++docId) {
    const std::string& text = mdocs[(size_t)docId].text;
    total_text_bytes += text.size();

    std::string url = mdocs[(size_t)docId].url;
    std::string title = mdocs[(size_t)docId].title;
    if (title.empty()) title = read_title_fallback_from_text(text);

    StrRef urlr = strs.add(url);
    StrRef titr = strs.add(title);

    DocRec dr{};
    dr.docId = docId;
    dr.url_off = urlr.off;
    dr.url_len = urlr.len;
    dr.title_off = titr.off;
    dr.title_len = titr.len;
    dr.text_bytes = (uint32_t)text.size();
    docs.push_back(dr);

    tokenizer.tokenize(text, tokens);

    if (!tokens.empty()) {
      std::vector<std::string> tmp(tokens.size());
      merge_sort_strings(tokens, tmp, 0, (int)tokens.size());

      size_t i = 0;
      while (i < tokens.size()) {
        size_t j = i + 1;
        while (j < tokens.size() && tokens[j] == tokens[i]) j++;
        if (!tokens[i].empty()) {
          pairs.push_back({tokens[i], docId});
        }
        i = j;
      }
    }
  }

  std::vector<PairTD> tmpP(pairs.size());
  merge_sort_pairtd(pairs, tmpP, 0, (int)pairs.size());

  std::vector<TermRec> terms;
  terms.reserve(50000);

  std::vector<uint8_t> posts;
  posts.reserve(4000000);

  uint64_t total_term_bytes = 0;
  uint64_t term_count = 0;

  size_t i = 0;
  while (i < pairs.size()) {
    size_t j = i + 1;
    while (j < pairs.size() && pairs[j].term == pairs[i].term) j++;

    std::string term = pairs[i].term;
    StrRef tr = strs.add(term);
    total_term_bytes += term.size();
    term_count++;

    uint32_t post_off = (uint32_t)posts.size();
    uint32_t df = (uint32_t)(j - i);

    uint32_t prev = 0;
    for (size_t k = i; k < j; ++k) {
      uint32_t d = pairs[k].doc;
      uint32_t gap = (k == i) ? (d + 1) : (d - prev);
      write_varint_u32(posts, gap);
      prev = d;
    }

    TermRec te{};
    te.term_off = tr.off;
    te.term_len = tr.len;
    te.post_off = post_off;
    te.df = df;
    terms.push_back(te);

    i = j;
  }

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  double avg_term_len = (term_count ? (double)total_term_bytes / (double)term_count : 0.0);
  double kb = (double)total_text_bytes / 1024.0;
  double speed_kb_s = (sec > 0.0 ? kb / sec : 0.0);
  double speed_docs_s = (sec > 0.0 ? (double)doc_count / sec : 0.0);

  std::cout << "=== LAB6 BUILD BOOLEAN INDEX (Mongo) ===\n";
  std::cout << "docs: " << doc_count << "\n";
  std::cout << "unique_terms: " << terms.size() << "\n";
  std::cout << "avg_term_len_bytes: " << avg_term_len << "\n";
  std::cout << "total_text_kb: " << kb << "\n";
  std::cout << "time_sec: " << sec << "\n";
  std::cout << "speed_kb_per_sec: " << speed_kb_s << "\n";
  std::cout << "speed_docs_per_sec: " << speed_docs_s << "\n";

  FileHeader hdr{};
  hdr.magic[0]='B'; hdr.magic[1]='I'; hdr.magic[2]='D'; hdr.magic[3]='X';
  hdr.version = 1;
  hdr.flags = 0;
  hdr.section_count = 4;

  SectionEntry secStrs{1,0,0,0,(uint64_t)strs.data.size()};
  SectionEntry secDocs{2,0,0,0,(uint64_t)docs.size()};
  SectionEntry secTerms{3,0,0,0,(uint64_t)terms.size()};
  SectionEntry secPosts{4,0,0,0,(uint64_t)posts.size()};

  uint64_t off = sizeof(FileHeader) + sizeof(SectionEntry) * hdr.section_count;

  uint64_t strs_payload = sizeof(uint32_t) + strs.data.size();
  secStrs.offset = off; secStrs.size = strs_payload; off += strs_payload;

  uint64_t docs_payload = sizeof(uint32_t) + (uint64_t)docs.size() * sizeof(DocRec);
  secDocs.offset = off; secDocs.size = docs_payload; off += docs_payload;

  uint64_t terms_payload = sizeof(uint32_t) + (uint64_t)terms.size() * sizeof(TermRec);
  secTerms.offset = off; secTerms.size = terms_payload; off += terms_payload;

  uint64_t posts_payload = sizeof(uint32_t) + posts.size();
  secPosts.offset = off; secPosts.size = posts_payload; off += posts_payload;

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 4;
  }

  write_bytes(out, &hdr, sizeof(hdr));
  write_bytes(out, &secStrs, sizeof(secStrs));
  write_bytes(out, &secDocs, sizeof(secDocs));
  write_bytes(out, &secTerms, sizeof(secTerms));
  write_bytes(out, &secPosts, sizeof(secPosts));

  write_u32(out, (uint32_t)strs.data.size());
  if (!strs.data.empty()) write_bytes(out, strs.data.data(), strs.data.size());

  write_u32(out, (uint32_t)docs.size());
  if (!docs.empty()) write_bytes(out, docs.data(), docs.size() * sizeof(DocRec));

  write_u32(out, (uint32_t)terms.size());
  if (!terms.empty()) write_bytes(out, terms.data(), terms.size() * sizeof(TermRec));

  write_u32(out, (uint32_t)posts.size());
  if (!posts.empty()) write_bytes(out, posts.data(), posts.size());

  out.close();
  std::cout << "saved: " << out_path << "\n";
  return 0;
}