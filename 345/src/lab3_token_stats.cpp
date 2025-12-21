#include "common.h"
#include "tokenizer.h"
#include "mongo_corpus.h"
#include <iostream>
#include <chrono>

static double bytes_to_kb(size_t bytes) { return (double)bytes / 1024.0; }

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


  size_t total_bytes = 0;
  uint64_t total_tokens = 0;
  uint64_t total_token_chars = 0;

  std::vector<std::string> tokens;

  auto t0 = std::chrono::steady_clock::now();

  for (size_t i = 0; i < docs.size(); ++i) {
    const std::string& text = docs[i].text;

    total_bytes += text.size();

    tok.tokenize(text, tokens);
    total_tokens += (uint64_t)tokens.size();

    for (size_t k = 0; k < tokens.size(); ++k) {
      total_token_chars += (uint64_t)tokens[k].size();
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  double avg_len = total_tokens ? (double)total_token_chars / (double)total_tokens : 0.0;
  double total_kb = bytes_to_kb(total_bytes);
  double speed = (sec > 0.0) ? (total_kb / sec) : 0.0;

  std::cout << "=== LAB3 TOKENIZATION ===\n";
  std::cout << "docs: " << docs.size() << "\n";
  std::cout << "total_text_kb: " << total_kb << "\n";
  std::cout << "total_tokens: " << total_tokens << "\n";
  std::cout << "avg_token_len_bytes: " << avg_len << "\n";
  std::cout << "time_sec: " << sec << "\n";
  std::cout << "speed_kb_per_sec: " << speed << "\n";

  return 0;
}
