"""
prep_data.py (v2) — Build the training corpus for the ad-relevancy scorer.

Fix vs v1: the HF `datasets` library (v4+) no longer supports script-based
datasets, which broke loading Amazon-Reviews-2023 through it. We now stream
the raw .jsonl.gz metadata files directly from the McAuley Lab server
(https://amazon-reviews-2023.github.io/) — no `datasets` needed for positives.

Positives : Amazon product metadata (title + features + description), 25 categories.
Negatives : AG News articles (parquet-backed on HF, still loads fine).

Run:  pip install requests pandas pyarrow datasets
      python prep_data.py
Output: data/products.parquet, data/negatives.parquet
Resumable: categories already saved to data/cache/ are skipped on re-run.
"""

import gzip
import io
import json
import os

import pandas as pd
import requests

SAMPLES_PER_CATEGORY = 8000
NEGATIVE_SAMPLES = 100_000
MIN_TEXT_CHARS = 60
OUT_DIR = "data"
CACHE_DIR = "data/cache"

BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
    "/raw/meta_categories/meta_{cat}.jsonl.gz"
)

# Exact config names from https://amazon-reviews-2023.github.io/
CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Electronics", "Grocery_and_Gourmet_Food", "Health_and_Household",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games",
    "Video_Games",
]


def product_to_text(row: dict) -> str:
    """Concatenate the useful text fields of one product."""
    parts = [row.get("title") or ""]
    parts.extend((row.get("features") or [])[:5])
    parts.extend((row.get("description") or [])[:3])
    return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def collect_category(category: str, limit: int) -> pd.DataFrame:
    """Stream one category's metadata file, keep the first `limit` usable rows."""
    cache_file = f"{CACHE_DIR}/{category}.parquet"
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        print(f"[cache] {category}: {len(df)} samples")
        return df

    url = BASE_URL.format(cat=category)
    print(f"[+] streaming {category} ...")
    rows = []
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            gz = gzip.GzipFile(fileobj=r.raw)
            for raw_line in io.TextIOWrapper(gz, encoding="utf-8", errors="ignore"):
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                text = product_to_text(item)
                if len(text) < MIN_TEXT_CHARS:
                    continue
                rows.append({"text": text[:2000], "category": category})
                if len(rows) >= limit:
                    break
    except Exception as e:
        print(f"    !! error on {category}: {e} — keeping {len(rows)} rows collected so far")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_parquet(cache_file, index=False)
    print(f"    -> {len(df)} samples")
    return df


def build_positives() -> pd.DataFrame:
    frames = [collect_category(cat, SAMPLES_PER_CATEGORY) for cat in CATEGORIES]
    frames = [f for f in frames if len(f) > 0]
    df = pd.concat(frames, ignore_index=True)
    print(f"[=] positives total: {len(df)} across {df['category'].nunique()} categories")
    return df


def build_negatives() -> pd.DataFrame:
    from datasets import load_dataset  # ag_news is parquet-backed; works on datasets v4

    print("[+] loading AG News for negatives ...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    df = pd.DataFrame({"text": ds["text"]})
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
    df = df.sample(n=min(NEGATIVE_SAMPLES, len(df)), random_state=42)
    df["category"] = "NON_PRODUCT"
    print(f"[=] negatives total: {len(df)}")
    return df


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    positives = build_positives()
    positives.to_parquet(f"{OUT_DIR}/products.parquet", index=False)

    negatives = build_negatives()
    negatives.to_parquet(f"{OUT_DIR}/negatives.parquet", index=False)

    print("\nDone. Files written to ./data/")
    print(positives["category"].value_counts())
