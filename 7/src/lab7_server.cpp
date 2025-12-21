#include "bidx_reader.h"
#include <chrono>
#include <sstream>
#include <cstdlib>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

static std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i=0;i<s.size();++i) {
    char c = s[i];
    if (c == '+') { out.push_back(' '); continue; }
    if (c == '%' && i+2 < s.size()) {
      auto hex = [](char x)->int{
        if (x>='0'&&x<='9') return x-'0';
        if (x>='a'&&x<='f') return x-'a'+10;
        if (x>='A'&&x<='F') return x-'A'+10;
        return -1;
      };
      int a = hex(s[i+1]), b = hex(s[i+2]);
      if (a>=0 && b>=0) {
        out.push_back((char)((a<<4) | b));
        i += 2;
        continue;
      }
    }
    out.push_back(c);
  }
  return out;
}

static std::string html_escape(const std::string& s) {
  std::string out; out.reserve(s.size());
  for (size_t i=0;i<s.size();++i) {
    char c = s[i];
    if (c=='&') out += "&amp;";
    else if (c=='<') out += "&lt;";
    else if (c=='>') out += "&gt;";
    else if (c=='"') out += "&quot;";
    else out.push_back(c);
  }
  return out;
}

static bool parse_query_param(const std::string& qs, const std::string& key, std::string& val) {
  // qs like "q=...&offset=50"
  size_t pos = 0;
  while (pos < qs.size()) {
    size_t amp = qs.find('&', pos);
    if (amp == std::string::npos) amp = qs.size();
    std::string part = qs.substr(pos, amp - pos);
    size_t eq = part.find('=');
    if (eq != std::string::npos) {
      std::string k = part.substr(0, eq);
      std::string v = part.substr(eq + 1);
      if (k == key) { val = url_decode(v); return true; }
    }
    pos = amp + 1;
  }
  return false;
}

static std::string build_home_page() {
  std::ostringstream o;
  o << "<!doctype html><html><head><meta charset='utf-8'><title>Lab7 Boolean Search</title></head><body>";
  o << "<h2>Boolean Search (Lab7)</h2>";
  o << "<form action='/search' method='get'>";
  o << "<input type='text' name='q' size='80' placeholder='example: (красный || желтый) автомобиль' />";
  o << "<button type='submit'>Search</button>";
  o << "</form>";
  o << "<p>Syntax: space/&& = AND, || = OR, ! = NOT, parentheses allowed.</p>";
  o << "</body></html>";
  return o.str();
}

static std::string build_results_page(const BidxIndex& idx,
                                     const std::string& q,
                                     size_t offset,
                                     const std::vector<uint32_t>& res,
                                     double ms) {
  std::ostringstream o;
  o << "<!doctype html><html><head><meta charset='utf-8'><title>Results</title></head><body>";
  o << "<h2>Search results</h2>";
  o << "<form action='/search' method='get'>";
  o << "<input type='text' name='q' size='80' value='" << html_escape(q) << "' />";
  o << "<button type='submit'>Search</button>";
  o << "</form>";

  o << "<p><b>Query:</b> " << html_escape(q) << "</p>";
  o << "<p><b>Hits:</b> " << res.size() << " | <b>Time:</b> " << ms << " ms</p>";

  size_t end = offset + 50;
  if (end > res.size()) end = res.size();

  o << "<ol start='" << (offset + 1) << "'>";
  for (size_t i = offset; i < end; ++i) {
    uint32_t d = res[i];
    const DocRec& dr = idx.docs[(size_t)d];
    std::string title = html_escape(idx.get_str(dr.title_off, dr.title_len));
    std::string url = html_escape(idx.get_str(dr.url_off, dr.url_len));
    o << "<li><a href='" << url << "' target='_blank'>" << title << "</a><br/>";
    o << "<small>" << url << "</small></li>";
  }
  o << "</ol>";

  if (end < res.size()) {
    o << "<a href='/search?q=" << html_escape(q) << "&offset=" << (offset + 50) << "'>Next 50</a>";
  }

  o << "</body></html>";
  return o.str();
}

static void send_http(int client_fd, const std::string& body, int code=200, const std::string& content_type="text/html; charset=utf-8") {
  std::ostringstream o;
  o << "HTTP/1.1 " << code << " OK\r\n";
  o << "Content-Type: " << content_type << "\r\n";
  o << "Content-Length: " << body.size() << "\r\n";
  o << "Connection: close\r\n\r\n";
  o << body;
  std::string resp = o.str();
  send(client_fd, resp.c_str(), resp.size(), 0);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage:\n  ./lab7_server <index.bidx> [port]\n";
    return 1;
  }
  std::string index_path = argv[1];
  int port = (argc >= 3) ? std::atoi(argv[2]) : 8080;

  BidxIndex idx;
  if (!load_bidx(index_path, idx)) {
    std::cerr << "Failed to load index: " << index_path << "\n";
    return 2;
  }

  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); return 3; }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);

  if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 4; }
  if (listen(server_fd, 16) < 0) { perror("listen"); return 5; }

  std::cout << "=== LAB7 SERVER ===\n";
  std::cout << "Index loaded: docs=" << idx.doc_count << " terms=" << idx.terms.size() << "\n";
  std::cout << "Open: http://localhost:" << port << "/\n";

  while (true) {
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) { perror("accept"); continue; }

    char buf[8192];
    int n = recv(client_fd, buf, sizeof(buf)-1, 0);
    if (n <= 0) { close(client_fd); continue; }
    buf[n] = 0;
    std::string req(buf);

    // parse first line: GET /path?qs HTTP/1.1
    size_t p1 = req.find(' ');
    size_t p2 = (p1==std::string::npos)?std::string::npos:req.find(' ', p1+1);
    std::string target = (p1!=std::string::npos && p2!=std::string::npos) ? req.substr(p1+1, p2-(p1+1)) : "/";

    std::string path = target;
    std::string qs;
    size_t qpos = target.find('?');
    if (qpos != std::string::npos) {
      path = target.substr(0, qpos);
      qs = target.substr(qpos + 1);
    }

    if (path == "/" || path.empty()) {
      send_http(client_fd, build_home_page());
      close(client_fd);
      continue;
    }

    if (path == "/search") {
      std::string q;
      parse_query_param(qs, "q", q);

      std::string off_s;
      size_t offset = 0;
      if (parse_query_param(qs, "offset", off_s)) {
        try { offset = (size_t)std::stoul(off_s); } catch (...) { offset = 0; }
      }

      std::vector<uint32_t> res;
      auto t0 = std::chrono::steady_clock::now();
      boolean_search(idx, q, res);
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      if (offset > res.size()) offset = 0;
      send_http(client_fd, build_results_page(idx, q, offset, res, ms));
      close(client_fd);
      continue;
    }

    send_http(client_fd, "<html><body>404</body></html>", 404);
    close(client_fd);
  }

  close(server_fd);
  return 0;
}
