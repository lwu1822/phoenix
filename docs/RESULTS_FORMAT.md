# Chart input formats

Run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/plot_results.py \
  --results results.json \
  --bandit-csv bandit_log.csv
```

The plotter directly accepts the server's nested benchmark output:

```json
{
  "modes": {
    "target": {
      "tokens_per_second": 21.5,
      "p95_latency_s": 9.5
    },
    "fixed_4": {
      "tokens_per_second": 15.0,
      "p95_latency_s": 17.7
    },
    "bandit": {
      "tokens_per_second": 15.7,
      "p95_latency_s": 14.3
    }
  }
}
```

It also accepts a JSON list or an object with a `results` list:

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

Accepted index aliases are `iteration`, `request`, `request_number`, and
`round`. Accepted setting aliases are `chosen_draft_tokens`, `setting`,
`action`, and `arm`.
