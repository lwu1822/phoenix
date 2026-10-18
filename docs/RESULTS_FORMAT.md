# Chart input formats

Run:

```bash
python3 -m pip install matplotlib
python3 scripts/plot_results.py --results results.json --bandit-csv bandit.csv
```

`results.json` may be a JSON list or an object with a `results` list:

```json
[
  {
    "mode": "no_draft",
    "draft_model": "stock",
    "tokens_per_sec": 0,
    "p95_latency_ms": 0
  },
  {
    "mode": "fixed_draft",
    "draft_model": "fine_tuned",
    "tokens_per_sec": 0,
    "p95_latency_ms": 0
  }
]
```

Replace the zeroes with measurements. Valid modes are `no_draft`,
`fixed_draft`, and `learned_bandit`; valid draft models are `stock` and
`fine_tuned`. Repeated mode/model rows are averaged. It is fine to include only
one `no_draft` row because that baseline has no draft model.

The bandit CSV needs one request index and one selected setting:

```csv
step,draft_tokens
1,2
2,4
```

Accepted index aliases are `iteration`, `request`, and `round`. Accepted setting
aliases are `chosen_draft_tokens`, `setting`, and `action`.
