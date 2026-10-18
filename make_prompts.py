"""Download verified MBPP questions and split 500/50 with seed 42."""

import json
from pathlib import Path
import random
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).parent
DATASET = "google-research-datasets/mbpp"
CONFIG = "full"
EXPECTED_COLUMNS = {
    "task_id", "text", "code", "test_list", "test_setup_code", "challenge_test_list"
}
SPLITS = {"train": 374, "test": 500, "validation": 90, "prompt": 10}


def fetch(split, offset, length):
    query = urlencode({
        "dataset": DATASET, "config": CONFIG, "split": split,
        "offset": offset, "length": length,
    })
    with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=60) as response:
        return json.load(response)


def main():
    train_path = ROOT / "train_prompts.jsonl"
    test_path = ROOT / "test_prompts.jsonl"
    if train_path.exists() or test_path.exists():
        raise FileExistsError("Refusing to overwrite existing prompt files")
    questions = []
    ids = set()
    for split, expected_count in SPLITS.items():
        for offset in range(0, expected_count, 100):
            page = fetch(split, offset, min(100, expected_count - offset))
            columns = {column["name"] for column in page["features"]}
            if columns != EXPECTED_COLUMNS or page["num_rows_total"] != expected_count:
                raise ValueError(f"Unexpected MBPP {split} schema or row count: {columns}")
            for item in page["rows"]:
                row = item["row"]
                if row["task_id"] in ids or not row["text"].strip():
                    raise ValueError("Duplicate task ID or empty question in source")
                ids.add(row["task_id"])
                questions.append(row["text"].strip())
    random.Random(42).shuffle(questions)
    train, test = questions[:500], questions[500:550]
    if len(train) != 500 or len(test) != 50 or set(train) & set(test):
        raise ValueError("Train/test overlap or missing prompts")
    for path, rows in ((train_path, train), (test_path, test)):
        path.write_text("".join(json.dumps({"prompt": prompt}) + "\n" for prompt in rows))
    print(f"Verified {len(questions)} public MBPP full questions and columns: {sorted(columns)}")
    print(f"Wrote {len(train)} train and {len(test)} disjoint test prompts (seed 42)")


if __name__ == "__main__":
    main()
