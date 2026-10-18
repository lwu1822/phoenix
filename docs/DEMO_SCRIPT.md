# Two-minute demo script

## 0:00–0:15 — Hook

**Say:** "LLMs normally generate one token at a time, making a full model pass
for every tiny step. Phoenix accelerates that loop. A small model drafts ahead,
Qwen3-1.7B verifies several tokens in one pass, and a learned scheduler decides
how many tokens are worth drafting."

**Show:** The Phoenix title, one-line architecture, and live endpoint.

## 0:15–0:40 — Live inference

**Say:** "I’ll send the same held-out Python coding question through our three
modes: no draft, a fixed draft length, and our learned bandit."

**Show:** Submit one test prompt. Keep the three outputs or compact result cards
side by side. Point to tokens/sec and latency.

**Say:** "The large model still verifies every accepted token, so the draft
model never becomes the authority over the output. Across our first 50-request
benchmark, the learned scheduler reached 15.72 tokens per second versus 14.98
for a fixed-four-token draft."

If the live request is slow, switch immediately to a pre-recorded response or
the completed benchmark view.

## 0:40–1:05 — Learned scheduler

**Say:** "The key tradeoff is draft length. Too short misses potential
parallelism; too long wastes guesses. Our lightweight bandit measures the
reward after each step, explores candidate lengths, and learns the best setting
online."

**Show:** `bandit_convergence.png`.

**Say:** "It selected four draft tokens on 43 of 50 requests, including 18 of
the last 20. The final two requests explore a two-token arm, so this run shows a
strong preference but also tells us we need a longer learning window."

## 1:05–1:30 — Fine-tuning and results

**Say:** "We also fine-tuned the draft model on Veri so its Python guesses align
more often with the target model."

**Show:** `benchmark.png`, then the results table.

**Say:** "The learned scheduler improves throughput by 4.9 percent and cuts p95
latency by 19.3 percent versus fixed speculation. It does not beat target-only
decoding yet, which tells us exactly where to focus: reducing draft overhead
and raising acceptance with the fine-tuned model. The fine-tuned benchmark is
still [TODO: pending or insert measured result]."

## 1:30–1:52 — Sentry evidence

**Say:** "Sentry made the serving loop visible. This trace breaks one request
into tokenization, generation, and detokenization. The generate span records
the draft length, model, acceptance, and throughput."

**Show:** The trace waterfall, then the dashboard.

**Say:** "The dashboard revealed [TODO: specific finding]. Trace-linked
structured logs led us to [TODO: issue], so we changed [TODO: concrete fix].
This is also how we can catch regressions as the scheduler adapts."

## 1:52–2:00 — Close

**Say:** "Phoenix combines speculative decoding, domain-tuned drafting, and
online learning to make inference adapt instead of relying on one magic
constant. Next, we’ll condition the scheduler on workload and scale it to
batched, larger-model serving."

**Show:** Architecture plus final headline: `Adaptive drafting: +4.9%
throughput vs. fixed drafting`.

## Recording checklist

- Rehearse to 1:50 so transitions and network delay do not exceed two minutes.
- Preload the endpoint, charts, Sentry trace, dashboard, and log search in tabs.
- Use a held-out prompt from `test_prompts.jsonl`, not a training prompt.
- Zoom Sentry so span names and attributes remain legible in the recording.
- Never expose the DSN, environment secrets, or browser notifications.
