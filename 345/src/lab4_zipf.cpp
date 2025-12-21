#include "common.h"
#include "tokenizer.h"
#include "mongo_corpus.h"
#include <iostream>
#include <fstream>
#include <chrono>

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

static void merge_sort_u32_desc(std::vector<uint32_t>& a, std::vector<uint32_t>& tmp, int l, int r) {
  if (r - l <= 1) return;
  int m = (l + r) / 2;
  merge_sort_u32_desc(a, tmp, l, m);
  merge_sort_u32_desc(a, tmp, m, r);
  int i = l, j = m, k = l;
  while (i < m && j < r) {
    if (a[i] >= a[j]) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < r) tmp[k++] = a[j++];
  for (int p = l; p < r; ++p) a[p] = tmp[p];
}

int main(int argc, char** argv) {
  std::string cfg_path = (argc >= 2) ? argv[1] : "config.yaml";

  std::vector<std::string> text_dirs;
  int max_docs, min_len, top_k;
  bool lowercase, normalize_yo, keep_numbers, use_stemming;
  double k1, b;

  if (!load_config_simple(cfg_path, text_dirs, max_docs, lowercase, normalize_yo, keep_numbers, min_len, k1, b, top_k, use_stemming)) {
    std::cerr << "Failed to load config\n";
    return 1;
  }

  TokenizerConfig tc{lowercase, normalize_yo, keep_numbers, min_len};
  Tokenizer tok(tc);

  MongoConfig mcfg;
  int mongo_max_docs = max_docs;
  std::string merr;

  if (!LoadMongoConfigFromYaml(cfg_path, mcfg, mongo_max_docs, merr)) {
    std::cerr << "Failed to load mongo config from yaml: " << merr << "\n";
    return 1;
  }

  std::vector<CorpusDoc> docs;
  if (!LoadDocsFromMongo(mcfg, mongo_max_docs, docs, merr)) {
    std::cerr << "Failed to load docs from mongo: " << merr << "\n";
    return 1;
  }


  std::vector<std::string> all_tokens;
  all_tokens.reserve(2000000);

  std::vector<std::string> tokens;

  auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < docs.size(); ++i) {
    tok.tokenize(docs[i].text, tokens);
    for (size_t k = 0; k < tokens.size(); ++k) all_tokens.push_back(tokens[k]);
  }


  std::vector<std::string> tmp(all_tokens.size());
  merge_sort_strings(all_tokens, tmp, 0, (int)all_tokens.size());

  std::vector<uint32_t> freqs;
  freqs.reserve(200000);

  size_t i = 0;
  while (i < all_tokens.size()) {
    size_t j = i + 1;
    while (j < all_tokens.size() && all_tokens[j] == all_tokens[i]) j++;
    freqs.push_back((uint32_t)(j - i));
    i = j;
  }

  std::vector<uint32_t> tmpf(freqs.size());
  merge_sort_u32_desc(freqs, tmpf, 0, (int)freqs.size());

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  std::ofstream out("zipf.csv");
  out << "rank,freq,zipf\n";
  if (!freqs.empty()) {
    double C = (double)freqs[0];
    for (size_t r = 1; r <= freqs.size(); ++r) {
      double f = (double)freqs[r - 1];
      double z = C / (double)r;
      out << r << "," << (uint64_t)f << "," << z << "\n";
    }
  }
  out.close();

  std::cout << "=== LAB4 ZIPF ===\n";
    std::cout << "docs: " << docs.size() << "\n";
  std::cout << "total_tokens: " << all_tokens.size() << "\n";
  std::cout << "unique_terms: " << freqs.size() << "\n";
  std::cout << "time_sec: " << sec << "\n";
  std::cout << "saved: zipf.csv\n";
  return 0;
}
