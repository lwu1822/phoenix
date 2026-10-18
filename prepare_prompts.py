#!/usr/bin/env python3
"""Build deterministic, disjoint prompt splits from a public HF dataset."""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DATASET = "Tapos-Minmoy/python_programming_questions"
CONFIG = "default"
SOURCE_SPLIT = "train"
QUESTION_COLUMN = "question"
SEED = 42
TRAIN_SIZE = 500
TEST_SIZE = 50
PAGE_SIZE = 100
SOURCE_POOL_SIZE = 1_000


def fetch_page(offset: int) -> list[str]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SOURCE_SPLIT,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                rows = json.load(response)["rows"]
            return [
                row["row"][QUESTION_COLUMN].strip()
                for row in rows
                if isinstance(row["row"].get(QUESTION_COLUMN), str)
                and row["row"][QUESTION_COLUMN].strip()
            ]
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    return []


def fetch_questions() -> list[str]:
    offsets = range(0, SOURCE_POOL_SIZE, PAGE_SIZE)
    with ThreadPoolExecutor(max_workers=5) as executor:
        pages = executor.map(fetch_page, offsets)
    return [question for page in pages for question in page]


def write_jsonl(path: Path, prompts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for prompt in prompts:
            output.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    # Exact-string deduplication ensures no prompt can occur in both splits.
    questions = list(dict.fromkeys(fetch_questions()))
    random.Random(SEED).shuffle(questions)
    required = TRAIN_SIZE + TEST_SIZE
    if len(questions) < required:
        raise RuntimeError(f"Need {required} unique questions; found {len(questions)}")

    selected = questions[:required]
    train_prompts = selected[:TRAIN_SIZE]
    test_prompts = selected[TRAIN_SIZE:]
    if set(train_prompts) & set(test_prompts):
        raise RuntimeError("Train and test prompts overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_prompts.jsonl"
    test_path = args.output_dir / "test_prompts.jsonl"
    write_jsonl(train_path, train_prompts)
    write_jsonl(test_path, test_prompts)

    print(
        f"Dataset: {DATASET} ({CONFIG}/{SOURCE_SPLIT}); "
        f"source pool: first {SOURCE_POOL_SIZE} rows; "
        f"question column: {QUESTION_COLUMN!r}; seed: {SEED}"
    )
    for path, prompts in ((train_path, train_prompts), (test_path, test_prompts)):
        print(f"\n{path}: {len(prompts)} rows")
        for prompt in prompts[:3]:
            print(json.dumps({"prompt": prompt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
