"""Build the self-contained Google Colab notebook without notebook dependencies."""

import json
from pathlib import Path
from textwrap import dedent


cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)})


def code(text):
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    })


markdown("""
# Phoenix | self-tuning speculative decoding

**Colab setup:** Runtime > Change runtime type > T4 GPU. Upload `test_prompts.jsonl`
to `/content` (or mount Drive and change `TEST_PROMPTS`). Run cells top to bottom.
Change only `DRAFT_MODEL` to compare a fine-tuned draft; results are kept in
different folders for different draft models.

The small model proposes several tokens; the large model checks them in a
single target forward pass. A bandit learns which proposal length is fastest.
Greedy decoding keeps the output identical to greedy target-only generation.
""")

code("""
%pip -q install -U "transformers==4.57.6" "accelerate>=1.0" "sentry-sdk>=2.35.0"
""")

code("""
import csv
import json
import logging
import re
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sentry_sdk
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator

TARGET_MODEL = "Qwen/Qwen3-1.7B"
DRAFT_MODEL = "Qwen/Qwen3-0.6B"  # Change to your public merged HF repo after training.
NUM_TEST_PROMPTS = 20
MAX_NEW_TOKENS = 128
TEST_PROMPTS = Path("/content/test_prompts.jsonl")
SEED = 42

assert torch.cuda.is_available(), "Choose a Colab T4 GPU runtime first."
torch.manual_seed(SEED)
print("transformers:", transformers.__version__, "| GPU:", torch.cuda.get_device_name(0))
""")

markdown("""
## Observability

Set `SENTRY_DSN` in Colab Secrets and enable notebook access for it. The
notebook stops with an explicit error if the secret is missing; it never prints
the secret value. Every request gets one trace and a structured Sentry log.
Continuous profiling is tied to the trace; if the Colab profiler cannot start,
Phoenix warns and continues benchmarking.
""")

code("""
logger = logging.getLogger("phoenix")
logging.basicConfig(level=logging.INFO)
from google.colab import userdata
try:
    sentry_dsn = userdata.get("SENTRY_DSN")
except Exception as exc:
    print("SENTRY_DSN loaded from Colab Secrets: NO")
    raise RuntimeError("Add SENTRY_DSN to Colab Secrets and grant notebook access.") from exc
print("SENTRY_DSN loaded from Colab Secrets:", "YES" if sentry_dsn else "NO")
if not sentry_dsn:
    raise RuntimeError("SENTRY_DSN is empty. Set it in Colab Secrets before running.")

try:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        enable_logs=True,
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",
    )
    logger.info("Sentry tracing/logging configured; continuous profiling is best effort.")
except Exception as exc:
    logger.warning("Sentry profiling unavailable; continuing with tracing/logging: %s", exc)
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0, enable_logs=True)

def event(message, **attributes):
    logger.info("%s %s", message, attributes)
    try:
        sentry_sdk.logger.info(message, attributes=attributes)
    except Exception as exc:
        logger.warning("Sentry log unavailable (benchmark continues): %s", exc)

@contextmanager
def request_trace(mode, arm):
    transaction = None
    try:
        transaction = sentry_sdk.start_transaction(op="phoenix.inference", name=mode)
        transaction.__enter__()
        transaction.set_tag("draft_model", DRAFT_MODEL)
        transaction.set_tag("arm", str(arm))
    except Exception as exc:
        logger.warning("Sentry trace/profiler failed to start; continuing: %s", exc)
        transaction = None
    try:
        yield transaction
    finally:
        if transaction is not None:
            try:
                transaction.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Sentry trace/profiler failed to stop; continuing: %s", exc)
""")

markdown("""
## Load both models once

Half precision keeps both small Qwen models within T4 memory. The tokenizer's
chat template disables Qwen3 thinking so the teacher and the benchmark use
ordinary answers, not hidden reasoning tokens.
""")

code("""
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL)
target = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
).eval()
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
).eval()
assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), (
    "Assisted decoding requires the same token IDs in target and draft tokenizers."
)
assert tokenizer.eos_token_id == draft_tokenizer.eos_token_id
draft.generation_config.num_assistant_tokens_schedule = "constant"
draft.generation_config.assistant_confidence_threshold = 0.0
""")

markdown("""
## Count accepted draft tokens

Transformers does not return an acceptance statistic from `generate`. Its
`AssistedCandidateGenerator` does receive both the candidate length and the
number of matches at each verification step. This wrapper counts proposed
tokens in `get_candidates()` and matched tokens in
`update_candidate_strategy()`. It never approximates acceptance by comparing
the final text; only tokens actually proposed and verified are counted.
The temporary patch is safe here because the benchmark runs requests serially.
""")

code("""
class AcceptanceCounter:
    def __init__(self):
        self.proposed = 0
        self.accepted = 0

    def wrapper(self):
        counter = self

        class CountingGenerator(AssistedCandidateGenerator):
            def get_candidates(self, input_ids):
                candidate_ids, candidate_logits = super().get_candidates(input_ids)
                self._pending = max(0, candidate_ids.shape[-1] - input_ids.shape[-1])
                counter.proposed += self._pending
                return candidate_ids, candidate_logits

            def update_candidate_strategy(self, input_ids, scores, n_matches):
                counter.accepted += max(0, min(int(n_matches), self._pending))
                return super().update_candidate_strategy(input_ids, scores, n_matches)

        return CountingGenerator


def infer(prompt, arm=None, mode="target"):
    chat = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
    ).to(target.device)
    counter = AcceptanceCounter()
    kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                  pad_token_id=tokenizer.eos_token_id)
    if arm not in (None, 0):
        draft.generation_config.num_assistant_tokens = arm
        kwargs["assistant_model"] = draft
    with request_trace(mode, arm) as transaction:
        start = time.perf_counter()
        with torch.inference_mode():
            if arm in (None, 0):
                output = target.generate(encoded, **kwargs)
            else:
                # Patch the symbol used by GenerationMixin to construct its candidate generator.
                with patch("transformers.generation.utils.AssistedCandidateGenerator",
                           counter.wrapper()):
                    output = target.generate(encoded, **kwargs)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        tokens = output.shape[-1] - encoded.shape[-1]
        acceptance = counter.accepted / counter.proposed if counter.proposed else None
        if transaction is not None:
            try:
                transaction.set_data("latency_seconds", latency)
                transaction.set_data("output_tokens", tokens)
                transaction.set_data("draft_accepted", counter.accepted)
                transaction.set_data("draft_proposed", counter.proposed)
            except Exception as exc:
                logger.warning("Sentry trace metadata unavailable; continuing: %s", exc)
        event("phoenix request", mode=mode, arm=arm if arm is not None else 0,
              latency_seconds=latency, tokens_per_second=tokens / latency,
              accepted=counter.accepted, proposed=counter.proposed)
    return dict(latency=latency, tokens=tokens, tps=tokens / latency,
                acceptance=acceptance, proposed=counter.proposed,
                accepted=counter.accepted,
                text=tokenizer.decode(output[0, encoded.shape[-1]:],
                                      skip_special_tokens=True))
""")

markdown("""
## Epsilon-greedy bandit

Arm 0 uses only the target (no draft). The bandit first tests each arm twice,
then usually picks the best running-average output tokens/second while
occasionally exploring. Epsilon gradually decreases.
""")

code("""
class Bandit:
    def __init__(self):
        self.arms = (0, 2, 4, 6, 8)
        self.counts = {arm: 0 for arm in self.arms}
        self.means = {arm: 0.0 for arm in self.arms}
        self.rng = np.random.default_rng(SEED)
        self.step = 0

    def choose(self):
        epsilon = max(0.05, 0.3 * (0.97 ** self.step))
        if any(self.counts[arm] < 2 for arm in self.arms):
            arm = next(arm for arm in self.arms if self.counts[arm] < 2)
        elif self.rng.random() < epsilon:
            arm = int(self.rng.choice(self.arms))
        else:
            arm = max(self.arms, key=lambda a: (self.means[a], -a))
        self.step += 1
        return arm

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm]
""")

markdown("""
## Benchmark

Warm up once to exclude first-run GPU overhead. All three modes see the same
first `NUM_TEST_PROMPTS` prompts in the same order. The acceptance column is `accepted / proposed`
across all requests; target-only has no draft tokens, so it shows `-`.
Outputs go to `results/<draft-model>/`, preserving stock and tuned runs.
""")

code("""
all_prompts = [json.loads(line)["prompt"] for line in TEST_PROMPTS.read_text().splitlines()
               if line.strip()]
assert len(all_prompts) >= NUM_TEST_PROMPTS > 0
prompts = all_prompts[:NUM_TEST_PROMPTS]
assert all(isinstance(p, str) and p.strip() for p in prompts)
print("Warm-up (not included in results)...")
infer(prompts[0], mode="warmup")

bandit = Bandit()
results = {}
bandit_log = []
for mode in ("target", "fixed_4", "bandit"):
    rows = []
    for number, prompt in enumerate(prompts, 1):
        arm = None if mode == "target" else (4 if mode == "fixed_4" else bandit.choose())
        row = infer(prompt, arm=arm, mode=mode)
        rows.append(row)
        if mode == "bandit":
            bandit.update(arm, row["tps"])
            bandit_log.append(dict(request_number=number, arm=arm, reward=row["tps"]))
    results[mode] = {
        "mean_latency_s": statistics.mean(r["latency"] for r in rows),
        "p50_latency_s": float(np.percentile([r["latency"] for r in rows], 50)),
        "p95_latency_s": float(np.percentile([r["latency"] for r in rows], 95)),
        "tokens_per_second": sum(r["tokens"] for r in rows) / sum(r["latency"] for r in rows),
        "acceptance_rate": (sum(r["accepted"] for r in rows) /
                            sum(r["proposed"] for r in rows)
                            if sum(r["proposed"] for r in rows) else None),
        "requests": len(rows),
    }

baseline = results["target"]["tokens_per_second"]
print(f'{"mode":<10} {"mean s":>8} {"p50 s":>8} {"p95 s":>8} {"tok/s":>9} {"accept":>9} {"speedup":>9}')
for mode, stats in results.items():
    stats["speedup"] = stats["tokens_per_second"] / baseline
    accept = "-" if stats["acceptance_rate"] is None else f'{stats["acceptance_rate"]:.1%}'
    print(f'{mode:<10} {stats["mean_latency_s"]:>8.2f} {stats["p50_latency_s"]:>8.2f} '
          f'{stats["p95_latency_s"]:>8.2f} {stats["tokens_per_second"]:>9.2f} '
          f'{accept:>9} {stats["speedup"]:>8.2f}x')

folder = Path("/content/results") / re.sub(r"[^a-zA-Z0-9._-]", "_", DRAFT_MODEL)
folder.mkdir(parents=True, exist_ok=True)
(folder / "results.json").write_text(json.dumps({
    "draft_model": DRAFT_MODEL, "target_model": TARGET_MODEL, "max_new_tokens": MAX_NEW_TOKENS,
    "num_test_prompts": NUM_TEST_PROMPTS,
    "transformers_version": transformers.__version__, "modes": results,
}, indent=2))
with (folder / "bandit_log.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["request_number", "arm", "reward"])
    writer.writeheader()
    writer.writerows(bandit_log)
print("Saved:", folder / "results.json", "and", folder / "bandit_log.csv")
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU",
                 "colab": {"gpuType": "T4"}},
    "nbformat": 4, "nbformat_minor": 5,
}
Path(__file__).with_name("phoenix.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
