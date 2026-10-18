# Benchmark analysis

Source files: `results.json` and `bandit_log.csv`.

## Recorded setup

- Target model: Qwen/Qwen3-1.7B
- Draft model: Qwen/Qwen3-0.6B (stock)
- Requests per benchmark mode: 50
- Maximum new tokens: 200
- Transformers: 4.57.6
- Hardware: Google Colab NVIDIA T4, fp16
- Batch size: 1
- Decoding: greedy, Qwen3 thinking disabled
- Evaluation data: 50 held-out MBPP prompts
- Warm-up and repetition protocol: 1 warm-up, then 1 benchmark run per mode

No fine-tuned-draft measurements are present in this result file.

## Results

| Mode                  | Tokens/sec | Mean latency | P50 latency | P95 latency | Acceptance |
| --------------------- | ---------: | -----------: | ----------: | ----------: | ---------: |
| Target only           |      21.55 |       9.26 s |      9.25 s |      9.54 s |        N/A |
| Fixed, 4 draft tokens |      14.98 |      13.32 s |     12.81 s |     17.75 s |     69.97% |
| Learned bandit        |      15.72 |      12.70 s |     12.75 s |     14.32 s |     68.95% |

Compared with fixed-four speculation, the learned bandit:

- Raises throughput by 4.9%.
- Reduces p95 latency by 19.3%.
- Reduces mean latency by 4.7%.

Compared with target-only decoding, the learned bandit:

- Has 27.0% lower throughput.
- Has 50.2% higher p95 latency.

The honest conclusion from this run is that adaptation improves the
speculative implementation, but speculation has not yet overcome its draft and
verification overhead.

## Bandit behavior

| Draft tokens | Selections | Mean observed reward |
| -----------: | ---------: | -------------------: |
|            2 |          2 |                18.07 |
|            4 |         43 |                15.85 |
|            6 |          4 |                14.84 |
|            8 |          1 |                14.13 |

The scheduler strongly favors four tokens, selecting it for 43 of 50 requests
and 18 of the final 20. However, the final two requests choose two tokens and
produce the highest observed rewards. Two samples are not enough to claim that
two is optimal, but they do mean this run has not established definitive
convergence. A longer run, or an initial forced-exploration phase with equal
samples per arm, would make the convergence claim stronger.

## Missing evidence

- Matching fixed and bandit benchmarks for the fine-tuned draft model.
- Exact epsilon decay and arm-value update code from `phoenix.ipynb`.
- Per-request benchmark output or confidence intervals.
- Sentry trace/log findings for the observability section.

## Next benchmark

The planned follow-up adds a no-draft bandit arm, tests a larger target model,
and includes the fine-tuned draft. This directly tests whether the scheduler
can avoid speculative overhead when target-only decoding is faster.
