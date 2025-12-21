#include "common.h"
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>

static bool ends_with(const std::string& s, const std::string& suf) {
  if (s.size() < suf.size()) return false;
  return s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool read_file_utf8(const std::string& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

std::string join_path(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

void list_txt_files(const std::string& dir, std::vector<std::string>& out_files) {
  DIR* dp = opendir(dir.c_str());
  if (!dp) return;

  struct dirent* de;
  while ((de = readdir(dp)) != nullptr) {
    std::string name = de->d_name;
    if (name == "." || name == "..") continue;
    if (!ends_with(name, ".txt")) continue;
    out_files.push_back(join_path(dir, name));
  }
  closedir(dp);
}

// .../povarenok/text -> povarenok
std::string parent_dir_name(const std::string& path) {
  // убираем хвостовой /
  std::string p = path;
  while (!p.empty() && p.back() == '/') p.pop_back();

  // find last /
  size_t last = p.find_last_of('/');
  if (last == std::string::npos) return p;
  std::string base = p.substr(0, last);

  size_t last2 = base.find_last_of('/');
  if (last2 == std::string::npos) return base;
  return base.substr(last2 + 1);
}

static bool parse_bool(const std::string& v, bool def) {
  if (v == "true") return true;
  if (v == "false") return false;
  return def;
}

static int parse_int(const std::string& v, int def) {
  try { return std::stoi(v); } catch (...) { return def; }
}

static double parse_double(const std::string& v, double def) {
  try { return std::stod(v); } catch (...) { return def; }
}

static std::string trim(const std::string& s) {
  size_t i = 0, j = s.size();
  while (i < j && (s[i] == ' ' || s[i] == '\t')) i++;
  while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\r' || s[j-1] == '\n')) j--;
  return s.substr(i, j - i);
}

bool load_config_simple(const std::string& path,
                        std::vector<std::string>& text_dirs,
                        int& max_docs,
                        bool& lowercase,
                        bool& normalize_yo,
                        bool& keep_numbers,
                        int& min_len,
                        double& k1,
                        double& b,
                        int& top_k,
                        bool& use_stemming) {
  std::string cfg;
  if (!read_file_utf8(path, cfg)) return false;

  // дефолты
  max_docs = 2000;
  lowercase = true;
  normalize_yo = true;
  keep_numbers = true;
  min_len = 2;
  k1 = 1.5;
  b = 0.75;
  top_k = 10;
  use_stemming = true;

  text_dirs.clear();

  std::istringstream in(cfg);
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') continue;

    // читаем массив text_dirs:
    if (line.rfind("-", 0) == 0) {
      // элемент списка вида: - /path
      std::string item = trim(line.substr(1));
      if (!item.empty() && (item.front() == '"' || item.front() == '\'')) {
        item = item.substr(1, item.size() - 2);
      }
      // добавляем только если похоже на путь
      if (!item.empty() && item.find('/') != std::string::npos) {
        text_dirs.push_back(item);
      }
      continue;
    }

    auto pos = line.find(':');
    if (pos == std::string::npos) continue;
    std::string key = trim(line.substr(0, pos));
    std::string val = trim(line.substr(pos + 1));
    if (!val.empty() && (val.front() == '"' || val.front() == '\'')) {
      val = val.substr(1, val.size() - 2);
    }

    if (key == "max_docs") max_docs = parse_int(val, max_docs);
    if (key == "lowercase") lowercase = parse_bool(val, lowercase);
    if (key == "normalize_yo") normalize_yo = parse_bool(val, normalize_yo);
    if (key == "keep_numbers") keep_numbers = parse_bool(val, keep_numbers);
    if (key == "min_len") min_len = parse_int(val, min_len);
    if (key == "k1") k1 = parse_double(val, k1);
    if (key == "b") b = parse_double(val, b);
    if (key == "top_k") top_k = parse_int(val, top_k);
    if (key == "use_stemming") use_stemming = parse_bool(val, use_stemming);
  }

  return !text_dirs.empty();
}
