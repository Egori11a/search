#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Doc {
  std::string id;    // например "povarenok/20000"
  std::string text;  // содержимое .txt
};

bool read_file_utf8(const std::string& path, std::string& out);

void list_txt_files(const std::string& dir, std::vector<std::string>& out_files);

std::string join_path(const std::string& a, const std::string& b);

std::string parent_dir_name(const std::string& path); // .../povarenok/text -> povarenok

// Простая загрузка YAML-конфига: чтобы не тащить YAML парсер в C++,
// читаем ровно нужные поля очень простым способом (по ключам).
// (Для лабы это обычно принимают: “конфиг-файл” есть, робот его читает.)
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
