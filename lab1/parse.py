import os
import time
import random
import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm

OUTPUT_DIR = Path("recipes_corpus")
TARGET_PER_SITE = 150
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = (0.8, 1.8)
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5

SITES = {
    "povarenok": {
        "pattern": "https://www.povarenok.ru/recipes/show/{}/",
        "start_id": 20000,
        "step": -1
    },
    "koolinar": {
        "pattern": "https://www.koolinar.ru/recipe/view/{}",
        "start_id": 150000,
        "step": -1
    }
}

HEADERS = {
    "User-Agent": "RecipeCorpusBot/1.0 (+https://your.email.or.site) Python requests"
}

def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for site in SITES.keys():
        (OUTPUT_DIR / site / "raw").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / site / "text").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / site / "meta").mkdir(parents=True, exist_ok=True)

def fetch_url(session: requests.Session, url: str) -> (int, str):
    wait = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            return r.status_code, r.text
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                return None, None
            time.sleep(wait)
            wait *= BACKOFF_FACTOR
    return None, None

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text

def is_likely_recipe(html: str, site_key: str) -> bool:
    if not html:
        return False
    if len(html) < 800:
        return False
    lower = html.lower()
    if site_key == "povarenok":
        if ("ингредиент" in lower) or ("рецепт" in lower) or ("приготовление" in lower):
            return True
    elif site_key == "koolinar":
        if ("ингредиент" in lower) or ("рецепт" in lower) or ("приготовление" in lower):
            return True
    return len(lower) > 2000

def save_file(path: Path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def save_text_file(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def collect_for_site(site_key: str, cfg: dict):
    pattern = cfg["pattern"]
    start_id = cfg["start_id"]
    step = cfg["step"]

    site_dir = OUTPUT_DIR / site_key
    raw_dir = site_dir / "raw"
    text_dir = site_dir / "text"
    meta_file = site_dir / "meta" / "meta.jsonl"

    seen_ids = set()
    metadata = []
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as mf:
            for line in mf:
                obj = json.loads(line)
                seen_ids.add(obj["id"])
                metadata.append(obj)

    session = requests.Session()
    current_id = start_id
    pbar = tqdm(total=TARGET_PER_SITE - len(seen_ids), desc=f"{site_key} collecting", unit="doc")
    attempts = 0
    while len(seen_ids) < TARGET_PER_SITE:
        url = pattern.format(current_id)
        if current_id in seen_ids:
            current_id += step
            continue

        status, body = fetch_url(session, url)
        attempts += 1

        if status == 200 and body:
            if is_likely_recipe(body, site_key):
                raw_path = raw_dir / f"{current_id}.html"
                save_file(raw_path, body.encode("utf-8"))
                text = extract_text_from_html(body)
                text_path = text_dir / f"{current_id}.txt"
                save_text_file(text_path, text)

                raw_size = raw_path.stat().st_size
                text_size = text_path.stat().st_size
                words = len(text.split())
                meta = {
                    "id": current_id,
                    "url": url,
                    "raw_path": str(raw_path),
                    "text_path": str(text_path),
                    "raw_size_bytes": raw_size,
                    "text_size_bytes": text_size,
                    "word_count": words,
                    "status_code": status
                }
                metadata.append(meta)
                seen_ids.add(current_id)
                with open(meta_file, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                pbar.update(1)

        time.sleep(random.uniform(*DELAY_BETWEEN_REQUESTS))
        current_id += step
        if attempts % 1000 == 0 and attempts > 0:
            tqdm.write(f"[{site_key}] пройдены {attempts} попыток, найдено {len(seen_ids)}")
    pbar.close()
    return metadata

def compute_stats(all_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_sizes = [m["raw_size_bytes"] for m in all_meta]
    text_sizes = [m["text_size_bytes"] for m in all_meta]
    words = [m["word_count"] for m in all_meta]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def median(lst):
        if not lst:
            return 0
        s = sorted(lst)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        else:
            return (s[mid-1] + s[mid]) / 2

    stats = {
        "num_documents": len(all_meta),
        "total_raw_bytes": sum(raw_sizes),
        "total_text_bytes": sum(text_sizes),
        "total_words": sum(words),
        "avg_raw_bytes": avg(raw_sizes),
        "avg_text_bytes": avg(text_sizes),
        "avg_words": avg(words),
        "median_raw_bytes": median(raw_sizes),
        "median_text_bytes": median(text_sizes),
        "median_words": median(words)
    }
    return stats

def save_summary(site_key: str, meta: List[Dict[str, Any]], stats: Dict[str, Any]):
    site_dir = OUTPUT_DIR / site_key
    csv_path = site_dir / f"{site_key}_meta.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            "id", "url", "raw_path", "text_path", "raw_size_bytes", "text_size_bytes", "word_count", "status_code"
        ])
        writer.writeheader()
        for m in meta:
            writer.writerow(m)
    with open(site_dir / f"{site_key}_stats.json", "w", encoding="utf-8") as jf:
        json.dump(stats, jf, ensure_ascii=False, indent=2)

def main():
    ensure_dirs()
    overall = {}
    for site_key, cfg in SITES.items():
        print(f"\n=== Начинаем сбор для {site_key} ===")
        meta = collect_for_site(site_key, cfg)
        stats = compute_stats(meta)
        save_summary(site_key, meta, stats)
        print(f"Завершено для {site_key}: {stats['num_documents']} документов")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        overall[site_key] = {"meta_count": len(meta), "stats": stats}

    with open(OUTPUT_DIR / "overall_summary.json", "w", encoding="utf-8") as of:
        json.dump(overall, of, ensure_ascii=False, indent=2)
    print("\nГотово. Результаты в:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
