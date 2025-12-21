#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Doc {
  std::string id;
  std::string text;
};

bool read_file_utf8(const std::string& path, std::string& out);

void list_txt_files(const std::string& dir, std::vector<std::string>& out_files);

std::string join_path(const std::string& a, const std::string& b);

std::string parent_dir_name(const std::string& path);

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
                        bool& use_stemming);
