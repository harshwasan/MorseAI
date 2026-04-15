"""
Build a fixed source-level ARRL benchmark manifest.

Default layout:
  - 5 folds
  - 1 source each from 15wpm, 20wpm, and 30wpm per fold
  - total: 15 held-out source recordings

Output:
  benchmark/source_benchmark.json
"""
import json
import os
from datetime import datetime

import numpy as np


ARRL_ROOT = "data/arrl_labeled"
OUT_PATH = "benchmark/source_benchmark.json"
WPM_BUCKETS = ["15wpm", "20wpm", "30wpm"]


def _source_key(path_or_name: str) -> str:
    stem = os.path.splitext(os.path.basename(path_or_name))[0]
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def _wpm_from_source(source_key: str) -> str:
    parts = source_key.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _pick_evenly(items: list, n: int) -> list:
    """Pick n items spread across a sorted list, deterministically."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)

    chosen = []
    used = set()
    for i in range(n):
        idx = round(i * (len(items) - 1) / (n - 1)) if n > 1 else len(items) // 2
        while idx in used:
            idx = min(len(items) - 1, idx + 1)
            if idx in used:
                idx = max(0, idx - 2)
        used.add(idx)
        chosen.append(items[idx])
    return chosen


def _load_arrl_sources(root: str = ARRL_ROOT):
    sources = {}
    if not os.path.exists(root):
        return sources

    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".json") or fname.startswith("test_"):
            continue
        json_path = os.path.join(root, fname)
        npy_path = json_path.replace(".json", ".npy")
        if not os.path.exists(npy_path):
            continue

        with open(json_path) as f:
            meta = json.load(f)

        source_key = _source_key(fname)
        wpm = _wpm_from_source(source_key)
        if wpm not in WPM_BUCKETS:
            continue

        entry = sources.setdefault(source_key, {
            "source_key": source_key,
            "wpm": wpm,
            "items": [],
        })
        entry["items"].append({
            "npy": npy_path,
            "text": meta["text"],
            "source_key": source_key,
            "wpm": wpm,
        })

    for entry in sources.values():
        entry["items"].sort(key=lambda x: x["npy"])
        entry["num_chunks"] = len(entry["items"])

    return sources


def build_source_benchmark(num_folds: int = 5):
    sources = _load_arrl_sources()
    if not sources:
        raise RuntimeError(f"No ARRL labeled sources found in {ARRL_ROOT}")

    by_bucket = {bucket: [] for bucket in WPM_BUCKETS}
    for source in sorted(sources.values(), key=lambda x: x["source_key"]):
        by_bucket[source["wpm"]].append(source)

    for bucket, items in by_bucket.items():
        if len(items) < num_folds:
            raise RuntimeError(
                f"Bucket {bucket} has only {len(items)} sources, need at least {num_folds}"
            )

    selected = {bucket: _pick_evenly(by_bucket[bucket], num_folds) for bucket in WPM_BUCKETS}

    folds = [
        {
            "name": f"fold_{i+1}",
            "sources": [],
            "items": [],
            "wpms": [],
            "num_chunks": 0,
        }
        for i in range(num_folds)
    ]

    for bucket in WPM_BUCKETS:
        bucket_sources = sorted(selected[bucket], key=lambda x: x["num_chunks"], reverse=True)
        for source in bucket_sources:
            eligible = [fold for fold in folds if bucket not in fold["wpms"]]
            fold = min(
                eligible,
                key=lambda f: (f["num_chunks"], len(f["sources"]), f["name"]),
            )
            fold["sources"].append(source["source_key"])
            fold["wpms"].append(bucket)
            fold["items"].extend(source["items"])
            fold["num_chunks"] += source["num_chunks"]

    all_sources = []
    all_items = []
    for fold in folds:
        filtered_items = []
        for item in fold["items"]:
            try:
                audio = np.load(item["npy"])
            except Exception:
                continue
            if getattr(audio, "size", 0) == 0:
                continue
            filtered_items.append(item)
        fold["items"] = filtered_items
        fold["sources"].sort()
        fold["items"].sort(key=lambda x: x["npy"])
        fold["wpms"].sort()
        fold["num_chunks"] = len(fold["items"])
        fold["num_sources"] = len(fold["sources"])
        all_sources.extend(fold["sources"])
        all_items.extend(fold["items"])

    payload = {
        "created_at": datetime.now().isoformat(),
        "strategy": "5 fixed source-level folds; each fold holds out one 15wpm, one 20wpm, and one 30wpm ARRL source.",
        "arrl_root": ARRL_ROOT,
        "wpm_buckets": WPM_BUCKETS,
        "num_folds": num_folds,
        "total_sources": len(all_sources),
        "total_chunks": len(all_items),
        "folds": folds,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"Saved source benchmark: {len(all_sources)} sources / {len(all_items)} chunks -> {OUT_PATH}",
        flush=True,
    )
    for fold in folds:
        print(
            f"  {fold['name']}: {fold['num_sources']} sources, "
            f"{fold['num_chunks']} chunks, {', '.join(fold['sources'])}",
            flush=True,
        )
    return payload


if __name__ == "__main__":
    build_source_benchmark()
