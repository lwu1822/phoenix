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
model improves speed without becoming the authority over the output. On this
request, Phoenix reaches [TODO] tokens per second versus [TODO] with no draft."

If the live request is slow, switch immediately to a pre-recorded response or
the completed benchmark view.

## 0:40–1:05 — Learned scheduler

**Say:** "The key tradeoff is draft length. Too short misses potential
parallelism; too long wastes guesses. Our lightweight bandit measures the
reward after each step, explores candidate lengths, and learns the best setting
online."

**Show:** `bandit_convergence.png`.

**Say:** "Here it explores at the start, then settles around [TODO] draft
tokens after about [TODO] requests."

## 1:05–1:30 — Fine-tuning and results

**Say:** "We also fine-tuned the draft model on Veri so its Python guesses align
more often with the target model."

**Show:** `benchmark.png`, then the results table.

**Say:** "Our best setup is [TODO]. It achieves [TODO] tokens per second,
[TODO]x the no-draft baseline, with p95 latency of [TODO] milliseconds. The
fine-tuned draft changes acceptance from [TODO]% to [TODO]%."

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

**Show:** Architecture plus final headline: `[TODO]x faster`.

## Recording checklist

- Rehearse to 1:50 so transitions and network delay do not exceed two minutes.
- Preload the endpoint, charts, Sentry trace, dashboard, and log search in tabs.
- Use a held-out prompt from `test_prompts.jsonl`, not a training prompt.
- Zoom Sentry so span names and attributes remain legible in the recording.
- Never expose the DSN, environment secrets, or browser notifications.
