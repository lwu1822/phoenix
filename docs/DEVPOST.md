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

Our target model is Qwen3-1.7B. We shuffled the full MBPP dataset with seed 42,
using 500 prompts for training and 50 held-out prompts for evaluation with no
overlap.

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

Phoenix uses an epsilon-greedy bandit over draft lengths `{2, 4, 6, 8}`. Its
reward is output tokens per second. Epsilon starts at 0.3 and decays during the
run, balancing early exploration with increasingly frequent use of the
best-observed arm. [TODO: insert the exact decay expression and arm-value update
from `phoenix.ipynb` once that notebook is added to the repository.] This keeps
adaptation cheap enough to remain in the serving loop.

In our 50-request run, the scheduler selected four draft tokens for 43 requests,
including 18 of the final 20. It still explored two tokens on the final two
requests, so this short run shows a strong preference for four rather than
definitive convergence. Those two-token trials had the highest observed reward,
which is a useful reason to run a longer learning phase.

## Fine-tuning the draft model on Veri

A draft model helps only when its guesses match the target model. Through Veri,
we shuffled the full MBPP dataset with seed 42 and split it into 500 training
and 50 held-out prompts. Qwen3-1.7B acted as the teacher, answering all 500
training prompts with temperature 0, a 256-token output limit, and thinking
disabled.

We then supervised-fine-tuned Qwen3-0.6B for one epoch on a single NVIDIA L4 on
AWS. The job ID is `1a524a15352e`, and the output model is
`lwu1822/phoenix-qwen3-0.6b-mbpp`. [TODO: add the remaining optimizer, learning
rate, sequence length, and related settings from `train.toml` once it is added
to the repository.] Teacher deployments cost $0.36, and training was estimated
at approximately $1.75.

Fine-tuned acceptance, throughput, and latency results are still pending and
will be added without inferring them from the stock-model benchmark.

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

All numbers below come from one Google Colab NVIDIA T4 in fp16 with a batch size
of one. Each mode used one warm-up followed by one benchmark run over 50
held-out MBPP prompts, greedy decoding, thinking disabled, and up to 200 new
tokens per prompt. The target is Qwen3-1.7B, the stock draft is Qwen3-0.6B, and
Transformers 4.57.6 was used. P95 is end-to-end request latency.

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

- Benchmark the fine-tuned MBPP draft under the same protocol.
- Add a no-draft arm so the bandit can bypass speculation when its overhead is
  higher than target-only decoding.
- Repeat the experiment with a larger target model, where verification has more
  opportunity to amortize target-model work.
- Condition the scheduler on prompt length, recent acceptance, and hardware
  load rather than learning one global draft length.
- Batch verification across concurrent requests and evaluate under sustained
  multi-user load.
- Expand fine-tuning and evaluation beyond Python while keeping language- and
  domain-specific scheduler policies.
- Add safe rollback when acceptance or latency drifts, driven by Sentry alerts.
- Finish the Sentry dashboard and screenshots after resolving the DSN issue.
