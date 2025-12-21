#include "common.h"
#include "tokenizer.h"
#include "stemmer_ru.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

static bool str_ends_with(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  return s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static std::string str_trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

static void split_by_char(const std::string& s, char delim, std::vector<std::string>& out) {
  out.clear();
  std::string cur;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == delim) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(s[i]);
    }
  }
  out.push_back(cur);
}

static std::string file_stem(const std::string& path) {
  size_t slash = path.find_last_of('/');
  std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
  size_t dot = name.find_last_of('.');
  if (dot == std::string::npos) return name;
  return name.substr(0, dot);
}

static std::string file_dir(const std::string& path) {
  size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return "";
  return path.substr(0, slash);
}

static std::string doc_id_from_path(const std::string& full_path) {
  const std::string needle = "/text/";
  size_t pos = full_path.find(needle);
  std::string site = "unknown";
  if (pos != std::string::npos) {
    std::string left = full_path.substr(0, pos);
    size_t slash = left.find_last_of('/');
    if (slash != std::string::npos) site = left.substr(slash + 1);
    else site = left;
  } else {
    site = parent_dir_name(file_dir(full_path));
  }
  return site + "/" + file_stem(full_path);
}

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

struct Triple {
  std::string term;
  int doc;
  int tf;
};

struct Posting {
  int doc;
  int tf;
};

struct LexEntry {
  std::string term;
  int start;
  int df;
};

static void merge_sort_triples(std::vector<Triple>& a, std::vector<Triple>& tmp, int l, int r) {
  if (r - l <= 1) return;
  int m = (l + r) / 2;
  merge_sort_triples(a, tmp, l, m);
  merge_sort_triples(a, tmp, m, r);
  int i = l, j = m, k = l;
  while (i < m && j < r) {
    const Triple& x = a[i];
    const Triple& y = a[j];
    bool le = (x.term < y.term) || (x.term == y.term && x.doc <= y.doc);
    if (le) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];
  for (int p = l; p < r; ++p) a[p] = tmp[p];
}

static int lex_find(const std::vector<LexEntry>& lex, const std::string& term) {
  int l = 0;
  int r = (int)lex.size() - 1;
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (lex[m].term == term) return m;
    if (lex[m].term < term) l = m + 1;
    else r = m - 1;
  }
  return -1;
}

static double bm25_idf(int N, int df) {
  if (df <= 0) return 0.0;
  return std::log(1.0 + (N - df + 0.5) / (df + 0.5));
}

static double bm25_score(double idf, int tf, int dl, double avgdl, double k1, double b) {
  double denom = tf + k1 * (1.0 - b + b * ((double)dl / (avgdl + 1e-9)));
  return idf * (tf * (k1 + 1.0)) / (denom + 1e-9);
}

static double precision_at_k(const std::vector<int>& retrieved, const std::vector<std::string>& doc_ids,
                             const std::vector<std::string>& relevant, int k) {
  if (k <= 0) return 0.0;
  int hit = 0;
  int upto = (int)retrieved.size() < k ? (int)retrieved.size() : k;

  for (int i = 0; i < upto; ++i) {
    const std::string& did = doc_ids[retrieved[i]];
    for (size_t j = 0; j < relevant.size(); ++j) {
      if (did == relevant[j]) { hit++; break; }
    }
  }
  return upto ? (double)hit / (double)upto : 0.0;
}

static double recall_at_k(const std::vector<int>& retrieved, const std::vector<std::string>& doc_ids,
                          const std::vector<std::string>& relevant, int k) {
  if (relevant.empty()) return 0.0;
  int hit = 0;
  int upto = (int)retrieved.size() < k ? (int)retrieved.size() : k;

  for (size_t j = 0; j < relevant.size(); ++j) {
    const std::string& rel = relevant[j];
    for (int i = 0; i < upto; ++i) {
      if (doc_ids[retrieved[i]] == rel) { hit++; break; }
    }
  }
  return (double)hit / (double)relevant.size();
}

int main(int argc, char** argv) {
  std::string cfg_path = (argc >= 2) ? argv[1] : "config.yaml";

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

  RussianStemmer stemmer;

  std::vector<std::string> files;
  for (size_t i = 0; i < text_dirs.size(); ++i) {
    list_txt_files(text_dirs[i], files);
  }
  if ((int)files.size() > max_docs) files.resize((size_t)max_docs);

  const int N = (int)files.size();
  std::vector<std::string> doc_ids;
  doc_ids.reserve(files.size());

  std::vector<int> doc_len;
  doc_len.reserve(files.size());

  std::vector<Triple> triples;
  triples.reserve(300000);

  std::vector<std::string> tokens;

  auto t0 = std::chrono::steady_clock::now();

  for (int di = 0; di < N; ++di) {
    std::string text;
    if (!read_file_utf8(files[di], text)) {
      doc_ids.push_back(doc_id_from_path(files[di]));
      doc_len.push_back(0);
      continue;
    }

    doc_ids.push_back(doc_id_from_path(files[di]));

    tokenizer.tokenize(text, tokens);
    if (use_stemming) {
      for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = stemmer.stem(tokens[i]);
      }
    }

    doc_len.push_back((int)tokens.size());

    if (tokens.empty()) continue;

    std::vector<std::string> tmp(tokens.size());
    merge_sort_strings(tokens, tmp, 0, (int)tokens.size());

    size_t i = 0;
    while (i < tokens.size()) {
      size_t j = i + 1;
      while (j < tokens.size() && tokens[j] == tokens[i]) j++;
      int tf = (int)(j - i);
      if (!tokens[i].empty()) {
        Triple tr;
        tr.term = tokens[i];
        tr.doc = di;
        tr.tf = tf;
        triples.push_back(tr);
      }
      i = j;
    }
  }

  double avgdl = 0.0;
  {
    long long sum = 0;
    for (size_t i = 0; i < doc_len.size(); ++i) sum += doc_len[i];
    avgdl = doc_len.empty() ? 0.0 : (double)sum / (double)doc_len.size();
  }

  std::vector<Triple> tmpT(triples.size());
  merge_sort_triples(triples, tmpT, 0, (int)triples.size());

  std::vector<LexEntry> lex;
  lex.reserve(200000);

  std::vector<Posting> postings;
  postings.reserve(triples.size());

  for (size_t i = 0; i < triples.size(); ++i) {
    if (lex.empty() || triples[i].term != lex.back().term) {
      LexEntry e;
      e.term = triples[i].term;
      e.start = (int)postings.size();
      e.df = 0;
      lex.push_back(e);
    }
    Posting p;
    p.doc = triples[i].doc;
    p.tf = triples[i].tf;
    postings.push_back(p);
    lex.back().df += 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  double build_sec = std::chrono::duration<double>(t1 - t0).count();

  std::cout << "=== LAB5 SEARCH (BM25, no maps) ===\n";
  std::cout << "docs: " << N << "\n";
  std::cout << "triples (unique term-doc pairs): " << triples.size() << "\n";
  std::cout << "unique_terms: " << lex.size() << "\n";
  std::cout << "avgdl: " << avgdl << "\n";
  std::cout << "k1: " << k1 << " b: " << b << " top_k: " << top_k << "\n";
  std::cout << "stemming: " << (use_stemming ? "ON" : "OFF") << "\n";
  std::cout << "build_time_sec: " << build_sec << "\n\n";

  if (argc >= 4 && std::string(argv[2]) == "--eval") {
    std::string qpath = argv[3];
    std::ifstream in(qpath);
    if (!in) {
      std::cerr << "Cannot open queries file: " << qpath << "\n";
      return 2;
    }
    std::string line;
    int qcount = 0;
    double p1=0,p3=0,p5=0,p10=0;
    double r1=0,r3=0,r5=0,r10=0;

    while (std::getline(in, line)) {
      line = str_trim(line);
      if (line.empty()) continue;
      size_t tab = line.find('\t');
      if (tab == std::string::npos) continue;

      std::string query = str_trim(line.substr(0, tab));
      std::string rels = str_trim(line.substr(tab+1));

      std::vector<std::string> relevant;
      if (!rels.empty()) split_by_char(rels, ',', relevant);

      tokenizer.tokenize(query, tokens);
      if (use_stemming) {
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = stemmer.stem(tokens[i]);
      }

      std::vector<double> scores(N, 0.0);
      for (size_t qi = 0; qi < tokens.size(); ++qi) {
        int idx = lex_find(lex, tokens[qi]);
        if (idx < 0) continue;
        int df = lex[(size_t)idx].df;
        double idf = bm25_idf(N, df);

        int start = lex[(size_t)idx].start;
        int end = (idx + 1 < (int)lex.size()) ? lex[(size_t)idx + 1].start : (int)postings.size();

        for (int pi = start; pi < end; ++pi) {
          int d = postings[(size_t)pi].doc;
          int tf = postings[(size_t)pi].tf;
          scores[(size_t)d] += bm25_score(idf, tf, doc_len[(size_t)d], avgdl, k1, b);
        }
      }

      std::vector<int> doc_idx;
      doc_idx.reserve(200);
      for (int d = 0; d < N; ++d) {
        if (scores[(size_t)d] > 0.0) doc_idx.push_back(d);
      }

      std::vector<int> retrieved;
      retrieved.reserve((size_t)top_k);
      for (int kk = 0; kk < top_k; ++kk) {
        int best = -1;
        double bestScore = 0.0;
        for (size_t i2 = 0; i2 < doc_idx.size(); ++i2) {
          int d = doc_idx[i2];
          if (scores[(size_t)d] > bestScore) {
            bestScore = scores[(size_t)d];
            best = d;
          }
        }
        if (best < 0) break;
        retrieved.push_back(best);
        scores[(size_t)best] = 0.0;
      }

      qcount++;
      p1 += precision_at_k(retrieved, doc_ids, relevant, 1);
      p3 += precision_at_k(retrieved, doc_ids, relevant, 3);
      p5 += precision_at_k(retrieved, doc_ids, relevant, 5);
      p10 += precision_at_k(retrieved, doc_ids, relevant, 10);

      r1 += recall_at_k(retrieved, doc_ids, relevant, 1);
      r3 += recall_at_k(retrieved, doc_ids, relevant, 3);
      r5 += recall_at_k(retrieved, doc_ids, relevant, 5);
      r10 += recall_at_k(retrieved, doc_ids, relevant, 10);
    }

    if (qcount == 0) qcount = 1;
    std::cout << "=== EVAL (" << qcount << " queries) ===\n";
    std::cout << "P@1=" << (p1/qcount) << " P@3=" << (p3/qcount) << " P@5=" << (p5/qcount) << " P@10=" << (p10/qcount) << "\n";
    std::cout << "R@1=" << (r1/qcount) << " R@3=" << (r3/qcount) << " R@5=" << (r5/qcount) << " R@10=" << (r10/qcount) << "\n";
    return 0;
  }

  std::cout << "Enter query (empty = exit)\n";
  while (true) {
    std::cout << "> ";
    std::string query;
    if (!std::getline(std::cin, query)) break;
    query = str_trim(query);
    if (query.empty()) break;

    tokenizer.tokenize(query, tokens);
    if (use_stemming) {
      for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = stemmer.stem(tokens[i]);
    }

    std::vector<double> scores(N, 0.0);

    for (size_t qi = 0; qi < tokens.size(); ++qi) {
      int idx = lex_find(lex, tokens[qi]);
      if (idx < 0) continue;

      int df = lex[(size_t)idx].df;
      double idf = bm25_idf(N, df);

      int start = lex[(size_t)idx].start;
      int end = (idx + 1 < (int)lex.size()) ? lex[(size_t)idx + 1].start : (int)postings.size();

      for (int pi = start; pi < end; ++pi) {
        int d = postings[(size_t)pi].doc;
        int tf = postings[(size_t)pi].tf;
        scores[(size_t)d] += bm25_score(idf, tf, doc_len[(size_t)d], avgdl, k1, b);
      }
    }

    for (int kk = 0; kk < top_k; ++kk) {
      int best = -1;
      double bestScore = 0.0;
      for (int d = 0; d < N; ++d) {
        if (scores[(size_t)d] > bestScore) {
          bestScore = scores[(size_t)d];
          best = d;
        }
      }
      if (best < 0) break;
      std::cout << doc_ids[(size_t)best] << "\t" << bestScore << "\n";
      scores[(size_t)best] = 0.0;
    }
    std::cout << "\n";
  }

  return 0;
}
