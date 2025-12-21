"""

Запуск:
  python crawler.py config.yaml

Зависимости:
  pip install pyyaml requests pymongo beautifulsoup4
"""

import sys
import time
import json
import random
import hashlib
import logging
import signal
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Iterable
from urllib.parse import urlparse, urlunparse, urljoin

import yaml
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crawler")

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Получен сигнал остановки. Завершаюсь после текущей итерации...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = (p.scheme or "http").lower()
    netloc = (p.netloc or "").lower()

    if not netloc and p.path.startswith("//"):
        p2 = urlparse(scheme + ":" + p.path)
        scheme = p2.scheme.lower()
        netloc = p2.netloc.lower()
        path = p2.path
        query = p2.query
    else:
        path = p.path or "/"
        query = p.query

    while "//" in path:
        path = path.replace("//", "/")

    fragment = ""

    if path != "/":
        path = path.rstrip("/") or "/"

    return urlunparse((scheme, netloc, path, "", query, fragment))


@dataclass
class Rule:
    source: str
    domains: List[str]
    allow: List[re.Pattern]
    deny: List[re.Pattern]


def compile_rules(cfg: Dict[str, Any]) -> Tuple[List[Rule], List[re.Pattern]]:
    crawl_cfg = cfg.get("crawl", {})
    rules_cfg = crawl_cfg.get("rules", [])
    global_deny_cfg = crawl_cfg.get("global_deny", []) or []
    global_deny = [re.compile(x) for x in global_deny_cfg]

    rules: List[Rule] = []
    for r in rules_cfg:
        source = r.get("source", "")
        domains = [d.lower() for d in (r.get("domains", []) or [])]
        allow = [re.compile(x) for x in (r.get("allow", []) or [])]
        deny = [re.compile(x) for x in (r.get("deny", []) or [])]
        rules.append(Rule(source=source, domains=domains, allow=allow, deny=deny))

    return rules, global_deny


def match_rule(url: str, rules: List[Rule]) -> Optional[Rule]:
    netloc = urlparse(url).netloc.lower()
    for rule in rules:
        if netloc in rule.domains:
            return rule
    return None


def is_allowed(url: str, rule: Optional[Rule], global_deny: List[re.Pattern]) -> bool:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        return False

    for rx in global_deny:
        if rx.search(url):
            return False

    if rule is None:
        return False

    for rx in rule.deny:
        if rx.search(url):
            return False

    if not rule.allow:
        return True

    return any(rx.search(url) for rx in rule.allow)


def get_db(cfg: Dict[str, Any]):
    uri = cfg["db"]["uri"]
    dbname = cfg["db"].get("database", "crawler_db")
    client = MongoClient(uri)
    db = client[dbname]

    db.urls.create_index([("url", ASCENDING)], unique=True)
    db.urls.create_index([("next_crawl_ts", ASCENDING)])

    db.docs.create_index([("url", ASCENDING)])
    db.docs.create_index([("crawl_ts", ASCENDING)])
    return db


def fetch_with_conditional(
    session: requests.Session,
    url: str,
    etag: Optional[str],
    last_modified: Optional[str],
    timeout: int,
    user_agent: str,
) -> Tuple[int, Dict[str, str], str]:
    headers = {"User-Agent": user_agent}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    hdrs = {k.lower(): v for k, v in r.headers.items()}
    return r.status_code, hdrs, r.text


def extract_links(html: str, base_url: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = urljoin(base_url, href)
        yield abs_url


class Crawler:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.db = get_db(cfg)
        self.urls: Collection = self.db.urls
        self.docs: Collection = self.db.docs

        logic = cfg.get("logic", {}) or {}
        delay = logic.get("delay_between_requests", [0.8, 1.8])
        if isinstance(delay, (int, float)):
            self.delay_min = self.delay_max = float(delay)
        else:
            self.delay_min = float(delay[0])
            self.delay_max = float(delay[1])

        self.revisit_interval = int(logic.get("revisit_interval", 86400))
        self.batch_size = int(logic.get("batch_size", 10))
        self.error_retry_interval = int(logic.get("error_retry_interval", 3600))
        self.http_timeout = int(logic.get("http_timeout", 15))
        self.idle_sleep = int(logic.get("idle_sleep_seconds", 5))
        self.user_agent = str(logic.get("user_agent", "SearchCrawler/1.0 (+lab2)"))

        crawl_cfg = cfg.get("crawl", {}) or {}
        self.max_discovered_per_page = int(crawl_cfg.get("max_discovered_per_page", 200))
        self.parse_links_on_200 = bool(crawl_cfg.get("parse_links_on_200", True))
        self.insert_docs_on_first_fetch_only = bool(crawl_cfg.get("insert_docs_on_first_fetch_only", False))

        self.rules, self.global_deny = compile_rules(cfg)

        self.session = requests.Session()

    def upsert_url(self, url: str, source: str, next_ts: int, discovered_from: Optional[str] = None):
        try:
            self.urls.update_one(
                {"url": url},
                {
                    "$setOnInsert": {
                        "url": url,
                        "source": source,
                        "etag": None,
                        "last_modified": None,
                        "hash": None,
                        "last_crawl_ts": None,
                        "status_code": None,
                        "discovered_from": discovered_from,
                        "first_seen_ts": int(time.time()),
                    },
                    "$min": {"next_crawl_ts": next_ts},
                },
                upsert=True,
            )
        except Exception:
            pass

    def seed_from_yaml(self):
        seeds = self.cfg.get("seeds", {}) or {}
        urls = seeds.get("urls", []) or []
        now = int(time.time())
        for s in urls:
            raw = s.get("url")
            if not raw:
                continue
            u = normalize_url(raw)
            rule = match_rule(u, self.rules)
            source = s.get("source") or (rule.source if rule else urlparse(u).netloc)
            if rule and is_allowed(u, rule, self.global_deny):
                self.upsert_url(u, source, now)
            else:
                logger.warning("Seed URL не подходит под правила crawl.rules: %s", u)

    def seed_from_meta_jsonl(self):
        seeds = self.cfg.get("seeds", {}) or {}
        meta_files = seeds.get("meta_jsonl", []) or []
        now = int(time.time())
        total = 0

        for path in meta_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        raw = obj.get("url")
                        if not raw:
                            continue
                        u = normalize_url(raw)
                        rule = match_rule(u, self.rules)
                        if not rule:
                            continue
                        if not is_allowed(u, rule, self.global_deny):
                            continue
                        self.upsert_url(u, rule.source, now)
                        total += 1
            except FileNotFoundError:
                logger.warning("meta_jsonl не найден: %s", path)
            except Exception as e:
                logger.exception("Ошибка чтения meta_jsonl %s: %s", path, e)

        logger.info("Seed из meta.jsonl добавил URL: %d", total)

    def next_batch(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        return list(
            self.urls.find({"next_crawl_ts": {"$lte": now}})
            .sort("next_crawl_ts", ASCENDING)
            .limit(self.batch_size)
        )

    def schedule_next(self, url: str, ts: int, status_code: Optional[int]):
        self.urls.update_one(
            {"url": url},
            {"$set": {"next_crawl_ts": ts, "status_code": status_code}},
        )

    def discover_from_page(self, base_url: str, html: str, source: str):
        if not self.parse_links_on_200:
            return

        discovered = 0
        for link in extract_links(html, base_url):
            if discovered >= self.max_discovered_per_page:
                break
            u = normalize_url(link)
            rule = match_rule(u, self.rules)
            if not is_allowed(u, rule, self.global_deny):
                continue
            src = rule.source if rule else source
            self.upsert_url(u, src, int(time.time()), discovered_from=base_url)
            discovered += 1

        if discovered:
            logger.info("Нашёл новых URL на странице: %d (%s)", discovered, base_url)

    def process_url(self, rec: Dict[str, Any]):
        url = rec["url"]
        source = rec.get("source") or urlparse(url).netloc
        etag = rec.get("etag")
        last_modified = rec.get("last_modified")
        stored_hash = rec.get("hash")
        was_crawled_before = rec.get("last_crawl_ts") is not None

        logger.info("Обход: %s (source=%s)", url, source)

        try:
            status, headers, body = fetch_with_conditional(
                self.session, url, etag, last_modified,
                timeout=self.http_timeout,
                user_agent=self.user_agent,
            )
        except requests.RequestException as e:
            logger.warning("Ошибка запроса %s: %s", url, e)
            self.schedule_next(url, int(time.time()) + self.error_retry_interval, None)
            return

        now_ts = int(time.time())

        if status == 304:
            self.urls.update_one(
                {"url": url},
                {"$set": {"last_crawl_ts": now_ts, "next_crawl_ts": now_ts + self.revisit_interval, "status_code": 304}},
            )
            logger.info("304 Not Modified: %s", url)
            return

        if status == 200 and body:
            raw_html = body

            new_hash = md5_text(raw_html)
            changed = (stored_hash != new_hash)

            should_insert = changed or (not was_crawled_before and not self.insert_docs_on_first_fetch_only)

            if should_insert:
                doc = {"url": url, "raw_html": raw_html, "source": source, "crawl_ts": now_ts}
                try:
                    self.docs.insert_one(doc)
                    logger.info("Документ сохранён (changed=%s): %s", changed, url)
                except Exception as e:
                    logger.exception("Не смог сохранить документ в docs: %s", e)
            else:
                logger.info("Контент не изменился (hash совпал): %s", url)

            self.discover_from_page(url, raw_html, source)

            self.urls.update_one(
                {"url": url},
                {"$set": {
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last-modified"),
                    "hash": new_hash,
                    "last_crawl_ts": now_ts,
                    "next_crawl_ts": now_ts + self.revisit_interval,
                    "status_code": 200,
                }},
            )
            return

        logger.warning("HTTP %s для %s", status, url)
        self.urls.update_one(
            {"url": url},
            {"$set": {"last_crawl_ts": now_ts, "next_crawl_ts": now_ts + 86400, "status_code": status}},
        )

    def run(self):
        logger.info("Старт робота")
        self.seed_from_yaml()
        self.seed_from_meta_jsonl()

        global RUNNING
        while RUNNING:
            batch = self.next_batch()
            if not batch:
                time.sleep(self.idle_sleep)
                continue

            for rec in batch:
                if not RUNNING:
                    break
                self.process_url(rec)
                time.sleep(random.uniform(self.delay_min, self.delay_max))

        logger.info("Робот корректно остановлен")


def main():
    if len(sys.argv) < 2:
        print("Использование: python crawler.py path/to/config.yaml")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    Crawler(cfg).run()


if __name__ == "__main__":
    main()