"""
Запуск:
  python3 crawler.py config.yaml

Зависимости:
  pip install pyyaml requests pymongo beautifulsoup4
"""

import sys
import time
import random
import hashlib
import logging
import signal
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse

import yaml
import requests
from bs4 import BeautifulSoup, Comment
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import OperationFailure


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


def normalize_url(url: str) -> str:
    p = urlparse(url)
    scheme = (p.scheme or "http").lower()
    netloc = (p.netloc or "").lower()

    path = p.path or "/"
    query = p.query or ""
    fragment = ""

    while "//" in path:
        path = path.replace("//", "/")
    if path != "/":
        path = path.rstrip("/") or "/"

    return urlunparse((scheme, netloc, path, "", query, fragment))


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_title_and_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    h1 = soup.find("h1")
    if not title and h1:
        title = h1.get_text(" ", strip=True)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def is_likely_recipe(html: str, min_len: int, must_any: List[str]) -> bool:
    if not html:
        return False
    if len(html) < min_len:
        return False
    low = html.lower()
    if must_any:
        return any(x.lower() in low for x in must_any)
    return True


def _safe_create_index(coll: Collection, keys, **kwargs):
    try:
        coll.create_index(keys, **kwargs)
    except OperationFailure as e:
        logger.warning("create_index skipped for %s: %s", coll.name, str(e))


def get_db(cfg: Dict[str, Any]) -> Tuple[MongoClient, Any, Collection, Collection, Collection]:
    uri = cfg["db"]["uri"]
    dbname = cfg["db"].get("database", "crawler_db")
    client = MongoClient(uri)
    db = client[dbname]

    urls_name = cfg["db"].get("urls_collection", "urls")
    docs_name = cfg["db"].get("docs_collection", "docs")
    state_name = cfg["db"].get("state_collection", "collect_state")

    urls = db[urls_name]
    docs = db[docs_name]
    state = db[state_name]

    _safe_create_index(urls, [("url", ASCENDING)], name="url_1", unique=True)
    _safe_create_index(urls, [("next_crawl_ts", ASCENDING)], name="next_crawl_ts_1")
    _safe_create_index(urls, [("source", ASCENDING)], name="source_1")

    _safe_create_index(docs, [("url", ASCENDING)], name="url_1", unique=True)
    _safe_create_index(docs, [("source", ASCENDING)], name="source_1")
    _safe_create_index(docs, [("crawl_ts", ASCENDING)], name="crawl_ts_1")

    _safe_create_index(state, [("site_key", ASCENDING)], name="site_key_1", unique=True)

    return client, db, urls, docs, state


def fetch(session: requests.Session, url: str, timeout: int, user_agent: str) -> Tuple[int, str]:
    headers = {"User-Agent": user_agent}
    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text


@dataclass
class CollectSite:
    key: str
    source: str
    pattern: str
    start_id: int
    step: int
    target_per_site: int
    min_html_len: int
    must_contain_any: List[str]


class Crawler:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.client, self.db, self.urls, self.docs, self.state = get_db(cfg)

        logic = cfg.get("logic", {}) or {}
        delay = logic.get("delay_between_requests", [0.8, 1.8])
        if isinstance(delay, (int, float)):
            self.delay_min = self.delay_max = float(delay)
        else:
            self.delay_min = float(delay[0])
            self.delay_max = float(delay[1])

        self.http_timeout = int(logic.get("http_timeout", 15))
        self.idle_sleep = int(logic.get("idle_sleep_seconds", 2))
        self.revisit_interval = int(logic.get("revisit_interval", 86400))
        self.error_retry_interval = int(logic.get("error_retry_interval", 3600))
        self.batch_size = int(logic.get("batch_size", 10))

        headers_cfg = self.cfg.get("headers", {}) or {}
        self.user_agent = str(headers_cfg.get("user_agent", "EgorSearchCrawler/1.0 (lab2)"))

        collect_cfg = self.cfg.get("collect", {}) or {}
        self.collect_enabled = bool(collect_cfg.get("enabled", True))
        self.batch_seed_size = int(collect_cfg.get("batch_seed_size", 200))
        self.stop_when_targets_reached = bool(collect_cfg.get("stop_when_targets_reached", True))

        self.sites: List[CollectSite] = []
        for s in (collect_cfg.get("sites", []) or []):
            self.sites.append(
                CollectSite(
                    key=s["key"],
                    source=s["source"],
                    pattern=s["pattern"],
                    start_id=int(s["start_id"]),
                    step=int(s["step"]),
                    target_per_site=int(s["target_per_site"]),
                    min_html_len=int(s.get("min_html_len", 800)),
                    must_contain_any=list(s.get("must_contain_any", []) or []),
                )
            )

        self.session = requests.Session()

    def _get_next_id(self, site: CollectSite) -> int:
        doc = self.state.find_one({"site_key": site.key})
        if not doc:
            self.state.insert_one({"site_key": site.key, "next_id": site.start_id})
            return site.start_id
        return int(doc.get("next_id", site.start_id))

    def _set_next_id(self, site: CollectSite, next_id: int):
        self.state.update_one(
            {"site_key": site.key},
            {"$set": {"next_id": int(next_id)}},
            upsert=True,
        )

    def _docs_count_for_site(self, site: CollectSite) -> int:
        return int(self.docs.count_documents({"source": site.source}))

    def _format_pattern(self, pattern: str, next_id: int) -> str:
        if "{id}" in pattern:
            return pattern.format(id=next_id)
        return pattern.format(next_id)

    def _upsert_url(self, url: str, source: str, next_ts: int):
        self.urls.update_one(
            {"url": url},
            {
                "$setOnInsert": {
                    "url": url,
                    "source": source,
                    "last_crawl_ts": None,
                    "status_code": None,
                    "hash": None,
                    "next_crawl_ts": next_ts,
                }
            },
            upsert=True,
        )

        self.urls.update_one(
            {"url": url},
            {"$min": {"next_crawl_ts": next_ts}},
            upsert=False,
        )


    def seed_generator_urls(self):
        if not self.collect_enabled:
            return

        now = int(time.time())
        for site in self.sites:
            have = self._docs_count_for_site(site)
            if have >= site.target_per_site:
                continue

            next_id = self._get_next_id(site)

            added = 0
            for _ in range(self.batch_seed_size):
                raw = self._format_pattern(site.pattern, next_id)
                u = normalize_url(raw)
                self._upsert_url(u, site.source, now)
                next_id += site.step
                added += 1

            self._set_next_id(site, next_id)
            logger.info("Seed generator: %s добавил %d URL (docs=%d/%d)", site.key, added, have, site.target_per_site)

    def next_batch(self) -> List[Dict[str, Any]]:
        now = int(time.time())
        return list(
            self.urls.find({"next_crawl_ts": {"$lte": now}})
            .sort("next_crawl_ts", ASCENDING)
            .limit(self.batch_size)
        )

    def process_url(self, rec: Dict[str, Any]):
        url = rec["url"]
        source = rec.get("source") or urlparse(url).netloc
        stored_hash = rec.get("hash")

        logger.info("Обход: %s (source=%s)", url, source)

        try:
            status, body = fetch(self.session, url, timeout=self.http_timeout, user_agent=self.user_agent)
        except requests.RequestException as e:
            logger.warning("Ошибка запроса %s: %s", url, e)
            self.urls.update_one(
                {"url": url},
                {"$set": {"next_crawl_ts": int(time.time()) + self.error_retry_interval, "status_code": None}},
            )
            return

        now_ts = int(time.time())

        if status == 200 and body:
            site_cfg = next((s for s in self.sites if s.source == source), None)
            if site_cfg and not is_likely_recipe(body, site_cfg.min_html_len, site_cfg.must_contain_any):
                self.urls.update_one(
                    {"url": url},
                    {"$set": {"last_crawl_ts": now_ts, "next_crawl_ts": now_ts + self.revisit_interval, "status_code": 200}},
                )
                logger.info("Пропуск: не похоже на рецепт по эвристике: %s", url)
                return

            raw_html = body
            new_hash = md5_text(raw_html)
            changed = (stored_hash != new_hash)

            if changed:
                title, text = extract_title_and_text(raw_html)
                doc = {
                    "url": url,
                    "raw_html": raw_html,
                    "source": source,
                    "crawl_ts": now_ts,
                    "title": title,
                    "text": text,
                }
                self.docs.replace_one({"url": url}, doc, upsert=True)
                logger.info("Документ сохранён/обновлён (changed=True): %s", url)
            else:
                logger.info("Контент не изменился (hash совпал): %s", url)

            self.urls.update_one(
                {"url": url},
                {"$set": {"hash": new_hash, "last_crawl_ts": now_ts, "next_crawl_ts": now_ts + self.revisit_interval, "status_code": 200}},
            )
            return

        logger.warning("HTTP %s для %s", status, url)
        self.urls.update_one(
            {"url": url},
            {"$set": {"last_crawl_ts": now_ts, "next_crawl_ts": now_ts + 86400, "status_code": status}},
        )

    def targets_reached(self) -> bool:
        if not self.collect_enabled:
            return False
        for site in self.sites:
            if self._docs_count_for_site(site) < site.target_per_site:
                return False
        return True

    def run(self):
        logger.info("Старт робота")
        global RUNNING

        while RUNNING:
            if self.stop_when_targets_reached and self.targets_reached():
                logger.info("Цели по документам достигнуты — завершаюсь.")
                break

            self.seed_generator_urls()

            batch = self.next_batch()
            if not batch:
                if self.stop_when_targets_reached and self.targets_reached():
                    logger.info("Цели по документам достигнуты — завершаюсь.")
                    break
                time.sleep(self.idle_sleep)
                continue

            for rec in batch:
                if not RUNNING:
                    break

                if self.stop_when_targets_reached and self.targets_reached():
                    logger.info("Цели по документам достигнуты — завершаюсь.")
                    RUNNING = False
                    break

                self.process_url(rec)
                time.sleep(random.uniform(self.delay_min, self.delay_max))

        logger.info("Робот корректно остановлен")


def main():
    if len(sys.argv) < 2:
        print("Использование: python3 crawler.py path/to/config.yaml")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    Crawler(cfg).run()


if __name__ == "__main__":
    main()
