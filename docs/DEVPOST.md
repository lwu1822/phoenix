# Phoenix

Phoenix is an experimental LLM inference server that combines speculative
decoding with a learned scheduler. It tests whether automatically choosing how
far a small model drafts can reduce response time without changing which model
controls the output.

## Problem

Autoregressive language models normally generate one token at a time. Each
token requires another pass through the model, so even a capable GPU spends
time repeatedly moving model weights and launching work for a very small unit
of progress. That makes interactive code generation feel slow.

Phoenix asks a practical question: can a small model cheaply predict several
tokens ahead, let the larger model verify them together, and automatically
choose how far ahead to predict?

Our target model is Qwen3-1.7B. We evaluate Phoenix on a held-out set of 50
Python programming questions, separated from 500 training prompts with a fixed
seed and no duplicate prompts across the splits.

## How speculative decoding works

Think of the small draft model as an assistant finishing the next few words of
a sentence in pencil. Instead of asking the large model to write every word
from scratch, Phoenix asks it to check the penciled-in block in one pass. It
keeps every token the large model agrees with. At the first disagreement, the
large model supplies the correct token and drafting starts again.

The large model remains the authority, so speculation changes how quickly we
arrive at an answer, not which model decides the answer. Drafting too few tokens
leaves speed on the table; drafting too many wastes work when guesses are
rejected.

## The learned scheduler

The best draft length depends on the prompt, model, and hardware. A fixed value
cannot adapt when acceptance behavior changes, so Phoenix treats candidate
draft lengths as choices in a lightweight bandit.

After each generation step, the scheduler observes the reward
([TODO: exact reward formula, such as verified tokens per millisecond]) and
updates its estimate for the chosen draft length. It explores alternatives
early, then increasingly chooses the setting that produces the best measured
throughput. This keeps adaptation cheap enough to remain in the serving loop.

In our 50-request run, the scheduler selected four draft tokens for 43 requests,
including 18 of the final 20. It still explored two tokens on the final two
requests, so this short run shows a strong preference for four rather than
definitive convergence. Those two-token trials had the highest observed reward,
which is a useful reason to run a longer learning phase.

## Fine-tuning the draft model on Veri

A draft model helps only when its guesses match the target model. We fine-tuned
the draft model on Veri using [TODO: exact Veri subset/configuration], formatted
as [TODO: prompt/response or token-training format]. Training used [TODO]
examples for [TODO] epochs with [TODO: key hyperparameters].

This makes the inexpensive model more familiar with the Python coding patterns
seen by the server. Compared with the stock draft, the fine-tuned model changed
acceptance rate from [TODO]% to [TODO]% and throughput from [TODO] to [TODO]
tokens/sec in [TODO: mode/workload].

## How Sentry shaped Phoenix

We instrumented every inference request as a trace with `tokenize`, `generate`,
and `detokenize` spans. Generate spans carry the selected draft length, draft
model, acceptance rate, and tokens/sec. Structured logs preserve the decision
and outcome for each scheduler update, and link back to the trace that produced
it.

The traces showed [TODO: specific bottleneck and evidence, for example
"generation accounted for X% of p95 request latency while tokenization stayed
below Y ms"]. That led us to [TODO: concrete change].

Grouping throughput by `draft_tokens` showed [TODO: specific pattern]. We used
that finding to [TODO: scheduler or default-setting change]. Comparing
acceptance rate by `draft_model` showed [TODO: stock versus fine-tuned result],
which [TODO: decision it justified].

Sentry also exposed [TODO: outlier/failure found in logs or a trace], and the
trace-linked log entry let us identify [TODO: root cause] and fix it by
[TODO: fix]. These are intentionally concrete findings rather than generic
monitoring claims; fill them from the final dashboard and trace screenshots.

## Results

All numbers below come from [TODO: hardware], using 50 requests with up to 200
new tokens per request, [TODO] warm-up requests, and one reported benchmark
run. The target is Qwen3-1.7B, the stock draft is Qwen3-0.6B, and Transformers
4.57.6 was used. P95 is end-to-end request latency.

| Mode                   | Draft model | Tokens/sec | P95 latency (ms) | Acceptance rate |
| ---------------------- | ----------- | ---------: | ---------------: | --------------: |
| No draft               | N/A         |      21.55 |            9,536 |             N/A |
| Fixed draft (4 tokens) | Stock       |      14.98 |           17,749 |          69.97% |
| Fixed draft            | Fine-tuned  |     [TODO] |           [TODO] |         [TODO]% |
| Learned bandit         | Stock       |      15.72 |           14,322 |          68.95% |
| Learned bandit         | Fine-tuned  |     [TODO] |           [TODO] |         [TODO]% |

The learned scheduler improved throughput by 4.9% and reduced p95 latency by
19.3% compared with the fixed-four speculative configuration. The current
prototype did not yet beat target-only decoding: learned speculation was 27.0%
lower in throughput and had 50.2% higher p95 latency. That result is important,
not something to hide—it points directly to draft-model quality and speculative
loop overhead as the next optimization targets. Fine-tuned-draft results remain
pending and are not inferred from this stock-model run.

## What's next

- Condition the scheduler on prompt length, recent acceptance, and hardware
  load rather than learning one global draft length.
- Batch verification across concurrent requests and evaluate under sustained
  multi-user load.
- Expand fine-tuning and evaluation beyond Python while keeping language- and
  domain-specific scheduler policies.
- Add safe rollback when acceptance or latency drifts, driven by Sentry alerts.
- Test larger target models and additional draft-model sizes to map where the
  speculative speedup is greatest.
