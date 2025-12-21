#include "common.h"
#include "tokenizer.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <cstring>


static bool ends_with(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  return s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

static std::string derive_meta_path_from_text_dir(const std::string& text_dir) {
  if (ends_with(text_dir, "/text")) {
    std::string base = text_dir.substr(0, text_dir.size() - 5);
    return base + "/meta/meta.jsonl";
  }
  return text_dir + "/../meta/meta.jsonl";
}

static std::string file_stem(const std::string& path) {
  size_t slash = path.find_last_of('/');
  std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
  size_t dot = name.find_last_of('.');
  if (dot == std::string::npos) return name;
  return name.substr(0, dot);
}

static int to_int_safe(const std::string& s) {
  try { return std::stoi(s); } catch (...) { return -1; }
}

static std::string read_title_from_text(const std::string& text) {
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

struct IdUrl {
  int id;
  std::string url;
};

static void merge_sort_idurl(std::vector<IdUrl>& a, std::vector<IdUrl>& tmp, int l, int r) {
  if (r - l <= 1) return;
  int m = (l + r) / 2;
  merge_sort_idurl(a, tmp, l, m);
  merge_sort_idurl(a, tmp, m, r);
  int i = l, j = m, k = l;
  while (i < m && j < r) {
    if (a[i].id <= a[j].id) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];
  for (int p = l; p < r; ++p) a[p] = tmp[p];
}

static int binary_find_url(const std::vector<IdUrl>& v, int id) {
  int l = 0, r = (int)v.size() - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (v[(size_t)m].id == id) return m;
    if (v[(size_t)m].id < id) l = m + 1;
    else r = m - 1;
  }
  return -1;
}

static bool parse_meta_line(const std::string& line, int& out_id, std::string& out_url) {
  out_id = -1;
  out_url.clear();

  size_t pid = line.find("\"id\"");
  if (pid == std::string::npos) return false;
  size_t colon = line.find(':', pid);
  if (colon == std::string::npos) return false;

  size_t pnum = line.find_first_of("0123456789", colon);
  if (pnum == std::string::npos) return false;
  size_t pend = pnum;
  while (pend < line.size() && (line[pend] >= '0' && line[pend] <= '9')) pend++;
  out_id = to_int_safe(line.substr(pnum, pend - pnum));
  if (out_id < 0) return false;

  size_t purl = line.find("\"url\"");
  if (purl == std::string::npos) return false;
  size_t colon2 = line.find(':', purl);
  if (colon2 == std::string::npos) return false;
  size_t q1 = line.find('"', colon2);
  if (q1 == std::string::npos) return false;
  size_t q2 = line.find('"', q1 + 1);
  if (q2 == std::string::npos) return false;
  out_url = line.substr(q1 + 1, q2 - (q1 + 1));
  return !out_url.empty();
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


int main(int argc, char** argv) {
  std::string cfg_path = (argc >= 2) ? argv[1] : "config.yaml";
  std::string out_path = (argc >= 3) ? argv[2] : "index.bidx";

  std::vector<std::string> text_dirs;
  int max_docs, min_len, top_k;
  bool lowercase, normalize_yo, keep_numbers, use_stemming;
  double k1, b;

  if (!load_config_simple(cfg_path, text_dirs, max_docs, lowercase, normalize_yo, keep_numbers, min_len, k1, b, top_k, use_stemming)) {
    std::cerr << "Failed to load config: " << cfg_path << "\n";
    return 1;
  }

  TokenizerConfig tc;
  tc.lowercase = lowercase;
  tc.normalize_yo = normalize_yo;
  tc.keep_numbers = keep_numbers;
  tc.min_len = min_len;
  Tokenizer tokenizer(tc);

  std::vector<std::string> files;
  std::vector<std::string> meta_paths;

  for (size_t i = 0; i < text_dirs.size(); ++i) {
    list_txt_files(text_dirs[i], files);
    meta_paths.push_back(derive_meta_path_from_text_dir(text_dirs[i]));
  }

  if ((int)files.size() > max_docs) files.resize((size_t)max_docs);
  uint32_t doc_count = (uint32_t)files.size();

  std::vector<IdUrl> all_meta;
  all_meta.reserve(4000);

  for (size_t i = 0; i < meta_paths.size(); ++i) {
    std::string meta;
    if (!read_file_utf8(meta_paths[i], meta)) {
      std::cerr << "WARN: can't read meta: " << meta_paths[i] << "\n";
      continue;
    }
    size_t pos = 0;
    while (pos < meta.size()) {
      size_t end = meta.find('\n', pos);
      if (end == std::string::npos) end = meta.size();
      std::string line = meta.substr(pos, end - pos);
      pos = end + 1;

      int id;
      std::string url;
      if (parse_meta_line(line, id, url)) {
        all_meta.push_back({id, url});
      }
    }
  }

  if (!all_meta.empty()) {
    std::vector<IdUrl> tmp(all_meta.size());
    merge_sort_idurl(all_meta, tmp, 0, (int)all_meta.size());
  }

  StringTable strs;
  std::vector<DocRec> docs;
  docs.reserve(doc_count);

  std::vector<PairTD> pairs;
  pairs.reserve(1500000);

  std::vector<std::string> tokens;

  uint64_t total_text_bytes = 0;

  auto t0 = std::chrono::steady_clock::now();

  for (uint32_t docId = 0; docId < doc_count; ++docId) {
    const std::string& path = files[(size_t)docId];

    std::string text;
    read_file_utf8(path, text);
    total_text_bytes += text.size();

    int file_id = to_int_safe(file_stem(path));
    std::string url = "";
    int idx = binary_find_url(all_meta, file_id);
    if (idx >= 0) url = all_meta[(size_t)idx].url;

    std::string title = read_title_from_text(text);

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

  std::cout << "=== LAB6 BUILD BOOLEAN INDEX ===\n";
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
    return 2;
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
