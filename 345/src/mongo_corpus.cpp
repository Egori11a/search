#include "mongo_corpus.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <cstring>

#include <mongoc/mongoc.h>
#include <bson/bson.h>

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
    return s.substr(b, e - b);
}

static std::string strip_comment(const std::string& s) {
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '#') return s.substr(0, i);
    }
    return s;
}

static bool starts_with(const std::string& s, const char* pref) {
    size_t n = std::strlen(pref);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; i++) if (s[i] != pref[i]) return false;
    return true;
}

static bool parse_key_value(const std::string& line, std::string& key, std::string& val) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return false;
    key = trim(line.substr(0, pos));
    val = trim(line.substr(pos + 1));
    if (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') || (val.front() == '\'' && val.back() == '\''))) {
        val = val.substr(1, val.size() - 2);
    }
    return !key.empty();
}

bool LoadMongoConfigFromYaml(const std::string& yaml_path, MongoConfig& cfg, int& max_docs, std::string& err) {
    cfg = MongoConfig{};
    max_docs = 0;

    std::ifstream in(yaml_path);
    if (!in) {
        err = "cannot open yaml: " + yaml_path;
        return false;
    }

    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        line = strip_comment(line);
        if (trim(line).empty()) continue;

        size_t indent = 0;
        while (indent < line.size() && line[indent] == ' ') indent++;

        std::string t = trim(line);

        if (indent == 0 && t.size() >= 2 && t.back() == ':') {
            section = t.substr(0, t.size() - 1);
            continue;
        }

        if (section != "db" && section != "corpus") continue;

        if (indent < 2) continue;

        std::string key, val;
        if (!parse_key_value(t, key, val)) continue;

        if (section == "db") {
            if (key == "uri") cfg.uri = val;
            else if (key == "database") cfg.database = val;
            else if (key == "docs_collection") cfg.docs_collection = val;
        } else if (section == "corpus") {
            if (key == "max_docs") {
                max_docs = std::atoi(val.c_str());
            }
        }
    }

    if (cfg.uri.empty() || cfg.database.empty()) {
        err = "yaml must contain db.uri and db.database";
        return false;
    }
    if (cfg.docs_collection.empty()) cfg.docs_collection = "docs";
    if (max_docs <= 0) max_docs = 2000;

    return true;
}

static bool bson_get_utf8(const bson_t* doc, const char* key, std::string& out) {
    bson_iter_t it;
    if (bson_iter_init_find(&it, doc, key) && BSON_ITER_HOLDS_UTF8(&it)) {
        uint32_t len = 0;
        const char* v = bson_iter_utf8(&it, &len);
        out.assign(v, v + len);
        return true;
    }
    return false;
}

static bool extract_id_from_url(const std::string& url, const std::string& marker, std::string& id) {
    auto p = url.find(marker);
    if (p == std::string::npos) return false;
    p += marker.size();
    size_t q = p;
    while (q < url.size() && std::isdigit(static_cast<unsigned char>(url[q]))) q++;
    if (q == p) return false;
    id = url.substr(p, q - p);
    return true;
}

static std::string site_key_from_source(const std::string& source) {
    if (source.find("povarenok") != std::string::npos) return "povarenok";
    if (source.find("koolinar") != std::string::npos) return "koolinar";
    return source;
}

static std::string make_doc_key(const std::string& source, const std::string& url, int fallback_i) {
    std::string id;
    if (source.find("povarenok") != std::string::npos) {
        if (!extract_id_from_url(url, "/recipes/show/", id)) {
            id = std::to_string(fallback_i);
        }
    } else if (source.find("koolinar") != std::string::npos) {
        if (!extract_id_from_url(url, "/recipe/view/", id)) {
            id = std::to_string(fallback_i);
        }
    } else {
        id = std::to_string(fallback_i);
    }
    return site_key_from_source(source) + "/" + id;
}

bool LoadDocsFromMongo(const MongoConfig& cfg, int max_docs, std::vector<CorpusDoc>& out, std::string& err) {
    out.clear();
    if (max_docs <= 0) max_docs = 2000;

    mongoc_init();

    mongoc_client_t* client = mongoc_client_new(cfg.uri.c_str());
    if (!client) {
        err = "mongoc_client_new failed (check uri)";
        mongoc_cleanup();
        return false;
    }

    mongoc_collection_t* coll = mongoc_client_get_collection(client, cfg.database.c_str(), cfg.docs_collection.c_str());
    if (!coll) {
        err = "mongoc_client_get_collection failed";
        mongoc_client_destroy(client);
        mongoc_cleanup();
        return false;
    }

    bson_t filter;
    bson_init(&filter);

    bson_t opts;
    bson_init(&opts);

    BSON_APPEND_INT64(&opts, "limit", static_cast<int64_t>(max_docs));

    bson_t sort;
    BSON_APPEND_DOCUMENT_BEGIN(&opts, "sort", &sort);
    BSON_APPEND_INT32(&sort, "source", 1);
    BSON_APPEND_INT32(&sort, "url", 1);
    bson_append_document_end(&opts, &sort);

    bson_t proj;
    BSON_APPEND_DOCUMENT_BEGIN(&opts, "projection", &proj);
    BSON_APPEND_INT32(&proj, "url", 1);
    BSON_APPEND_INT32(&proj, "source", 1);
    BSON_APPEND_INT32(&proj, "title", 1);
    BSON_APPEND_INT32(&proj, "text", 1);
    bson_append_document_end(&opts, &proj);

    mongoc_cursor_t* cursor = mongoc_collection_find_with_opts(coll, &filter, &opts, nullptr);

    const bson_t* doc = nullptr;
    int i = 0;
    while (mongoc_cursor_next(cursor, &doc)) {
        CorpusDoc d;
        bson_get_utf8(doc, "url", d.url);
        bson_get_utf8(doc, "source", d.source);
        bson_get_utf8(doc, "title", d.title);
        bson_get_utf8(doc, "text", d.text);

        if (d.url.empty() || d.source.empty()) continue;

        if (d.text.empty()) continue;

        d.doc_key = make_doc_key(d.source, d.url, i);
        out.push_back(d);

        i++;
        if (i >= max_docs) break;
    }

    if (mongoc_cursor_error(cursor, nullptr)) {
        err = "mongoc cursor error";
        mongoc_cursor_destroy(cursor);
        bson_destroy(&opts);
        bson_destroy(&filter);
        mongoc_collection_destroy(coll);
        mongoc_client_destroy(client);
        mongoc_cleanup();
        return false;
    }

    mongoc_cursor_destroy(cursor);
    bson_destroy(&opts);
    bson_destroy(&filter);
    mongoc_collection_destroy(coll);
    mongoc_client_destroy(client);
    mongoc_cleanup();

    if (out.empty()) {
        err = "no docs loaded from mongo (check collection, fields, max_docs)";
        return false;
    }
    return true;
}
