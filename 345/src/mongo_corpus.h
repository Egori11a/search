#pragma once

#include <string>
#include <vector>

struct MongoConfig {
    std::string uri;
    std::string database;
    std::string docs_collection;
};

struct CorpusDoc {
    std::string doc_key;
    std::string url;
    std::string source;
    std::string title;
    std::string text;
};

bool LoadMongoConfigFromYaml(const std::string& yaml_path, MongoConfig& cfg, int& max_docs, std::string& err);
bool LoadDocsFromMongo(const MongoConfig& cfg, int max_docs, std::vector<CorpusDoc>& out, std::string& err);
