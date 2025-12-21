#include "bidx_reader.h"
#include <chrono>

static std::string htmlish_title_fix(const std::string& s) {
  // Просто убираем лишние пробелы, чтобы вывод был читабельнее.
  std::string out; out.reserve(s.size());
  bool ws=false;
  for (size_t i=0;i<s.size();++i) {
    unsigned char c=(unsigned char)s[i];
    if (c==' '||c=='\t'||c=='\r'||c=='\n') {
      if (!ws) out.push_back(' ');
      ws=true;
    } else { out.push_back((char)c); ws=false; }
  }
  return trim(out);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage:\n  ./lab7_cli <index.bidx> <queries.txt>\n";
    return 1;
  }
  std::string index_path = argv[1];
  std::string queries_path = argv[2];

  BidxIndex idx;
  if (!load_bidx(index_path, idx)) {
    std::cerr << "Failed to load index: " << index_path << "\n";
    return 2;
  }

  std::ifstream in(queries_path);
  if (!in) {
    std::cerr << "Failed to open queries: " << queries_path << "\n";
    return 3;
  }

  std::cout << "=== LAB7 CLI BOOLEAN SEARCH ===\n";
  std::cout << "docs: " << idx.doc_count << " terms: " << idx.terms.size() << "\n\n";

  std::string line;
  while (std::getline(in, line)) {
    std::string q = trim(line);
    if (q.empty()) continue;

    std::vector<uint32_t> res;
    auto t0 = std::chrono::steady_clock::now();
    boolean_search(idx, q, res);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "QUERY: " << q << "\n";
    std::cout << "hits=" << res.size() << " time_ms=" << ms << "\n";

    size_t show = res.size() < 50 ? res.size() : 50;
    for (size_t i=0;i<show;++i) {
      uint32_t d = res[i];
      const DocRec& dr = idx.docs[(size_t)d];
      std::string title = htmlish_title_fix(idx.get_str(dr.title_off, dr.title_len));
      std::string url = idx.get_str(dr.url_off, dr.url_len);
      std::cout << (i+1) << ". " << title << " | " << url << "\n";
    }
    std::cout << "\n";
  }

  return 0;
}
