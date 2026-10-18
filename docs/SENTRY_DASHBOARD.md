# Phoenix Sentry dashboard

These steps match Sentry's current **Spans** dataset and Trace Explorer. The
Developer plan supports 10 custom dashboards and has a 7-day query window.

Before building widgets, send several requests for every combination you want
to compare. Keep attribute names and types stable:

- `draft_tokens`: number
- `acceptance_rate`: number from 0 to 1
- `tokens_per_sec`: number
- `draft_model`: string such as `stock` or `fine_tuned`

Put these attributes on the `generate` span. If you also need end-to-end request
latency, put them on the root inference span too. Numeric values must be sent as
numbers, not strings, or Sentry cannot apply `avg()`/`p95()` to them.

## Confirm the data

1. Open **Explore > Traces** (called **Trace Explorer** in the docs).
2. Select **Span Samples**, your Phoenix project, and **Last 24 hours**.
3. Filter to the inference traffic, for example `span.op:generate`.
4. Add columns for `span.duration`, `draft_tokens`, `acceptance_rate`,
   `tokens_per_sec`, and `draft_model`.
5. If an attribute does not autocomplete, open a recent generate span and
   confirm the exact spelling and data type. Generate fresh traffic after any
   instrumentation fix.

## Build the dashboard

Open **Dashboards > Create Dashboard**, choose a blank dashboard, and name it
`Phoenix — Speculative Decoding`.

### 1. Latency over time

Add a widget with:

- Dataset: **Spans**
- Display: **Line**
- Query/filter: the inference root operation if available; otherwise use the
  stage spans (`span.op:tokenize OR span.op:generate OR span.op:detokenize`)
- Y-axis: `p95(span.duration)`
- Group by: `span.op` when charting stage spans; none for the root span
- Title: `P95 inference latency over time`

Add `avg(span.duration)` as a second series only if the chart remains readable.
The root inference span is the correct source for end-to-end latency; summing
sampled child-span aggregates is not.

### 2. Tokens/sec by draft token count

Add a widget with:

- Dataset: **Spans**
- Display: **Bar (Categorical)**
- Query/filter: `span.op:generate` (or the exact operation shown in your data)
- Y-axis: `avg(tokens_per_sec)`
- Group by: `draft_tokens`
- Sort: `draft_tokens` ascending if the builder offers it
- Title: `Throughput by draft tokens`

If the categorical bar chart is unavailable in your UI, use a **Table** with
`draft_tokens`, `avg(tokens_per_sec)`, and `count()`.

### 3. Acceptance rate by draft model

Add a widget with:

- Dataset: **Spans**
- Display: **Bar (Categorical)**
- Query/filter: `span.op:generate`
- Y-axis: `avg(acceptance_rate)`
- Group by: `draft_model`
- Title: `Acceptance rate by draft model`

Keep acceptance as a 0–1 fraction unless the SDK currently emits 0–100. Label
the convention in the title if the builder cannot format it as a percentage.

Save the dashboard. Set its global project filter to Phoenix and its time range
to a window containing the benchmark run. Sentry aggregates sampled spans, so
the dashboard is observability evidence rather than the source of the final
benchmark table.

## Submission screenshots

Take these three screenshots after generating representative traffic:

1. **Trace waterfall:** Open a single successful inference trace. Expand the
   request so `tokenize`, `generate`, and `detokenize` are all visible with
   their durations. Select `generate` so the side panel also shows
   `draft_tokens`, `acceptance_rate`, `tokens_per_sec`, and `draft_model`.
   Include the trace ID and total duration; crop out the browser chrome and DSN.
2. **Dashboard:** Show the dashboard title, benchmark time range, project
   filter, and all three widgets at once. Make sure legends, axes, and
   stock/fine-tuned model labels are readable.
3. **Log search:** Open **Explore > Logs** and search for one distinctive
   inference-completion message plus an attribute filter, for example
   `draft_model:fine_tuned draft_tokens:4`. Show several rows, structured
   columns (`tokens_per_sec`, `acceptance_rate`, `draft_tokens`), and the
   trace-ID link. Open one row only if its details do not hide the result list.

Do not show the DSN, auth tokens, user prompts containing secrets, or unrelated
organization data.

## Current documentation

- Widget Builder: https://docs.sentry.io/product/dashboards/widget-builder/
- Custom Dashboards: https://docs.sentry.io/product/dashboards/custom-dashboards/
- Traces and Spans: https://docs.sentry.io/product/trace-explorer/
- Structured Logs: https://docs.sentry.io/product/logs/
- Developer plan: https://sentry.io/pricing/
