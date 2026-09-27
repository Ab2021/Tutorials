# Cheat Sheet: Observability, Tracing & Evaluation

> `T17` · **Transcript coverage:** primary · [Case study](../01-case-studies/T17-observability-evals.md) · [Blueprint](../03-design-blueprints/T17-observability-evals/HLD.md) · [Interview bank](../02-interview-questions/T17-observability-evals.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Metrics that matter for LLM serving | **TTFT, ITL/TPOT, goodput, queue depth, KV usage** | `[T]`/`[D]` |
| The signal that exposes bad routing | prefix cache hit rate, per pod | `[T]` llm-d |
| Saturation examples to alert on | KV **80%**, active requests **> 8** | `[T]` llm-d |
| Non-determinism at temperature 0 | **outputs still differ** | `[T]` CMU lecture 2 |
| Quantization's effect on it | **worsens** non-determinism; also **fingerprints** the model | `[T]` CMU lecture 2 |
| The mandatory quality gate | **"always rerun your evals after quantizing"** | `[T]` LLMOps cost talk |
| Cost visibility | tokens in/out per request, per tenant | `[T]` LLMOps cost talk |

**The organising idea** `[D]`: **three planes, not one.** (1) **System observability** — metrics
(latency, queue, KV, errors), the ops dashboard. (2) **Trace observability** — spans per LLM call,
tool call and retrieval step, so you can see *why* a task was slow or wrong. (3) **Quality
evaluation** — offline and online scoring of outputs and trajectories. A team with only plane 1 can
tell you the GPU is busy and nothing else.

---

## The one-table summary

| Plane | What you collect | Answers | Tool shape |
|---|---|---|---|
| **Metrics** | TTFT, ITL, goodput, queue, KV%, cache hit rate | "is it healthy / fast / affordable?" | Prometheus + Grafana |
| **Traces** | spans per LLM call, tool call, retrieval; `gen_ai.*` attributes | "where did this request spend time / go wrong?" | OpenTelemetry → collector |
| **Logs** | prompts, responses, errors (with PII policy) | "what exactly happened?" | structured logs |
| **Offline eval** | golden sets, judges, regression suites | "did this change make it worse?" | CI-gated eval harness |
| **Online eval** | sampled judge scoring, user feedback, thumbs | "is production quality drifting?" | async scoring pipeline |
| **Trajectory eval** | step sequence, tool-call correctness | "did the agent take the right *path*?" | trace-based assertions |
| **Cost telemetry** | tokens in/out, cost, per tenant/model | "who is spending the money?" | gateway accounting `[T]` |

**Trace ≠ eval.** A trace tells you what happened; an eval tells you whether it was good. You need
both, and they consume different pipelines `[D]`.

---

## Formulas

**The latency decomposition you must be able to draw** `[D]`:
```
end_to_end = queue_wait + prefill(TTFT) + N × ITL + Σ tool_time + retries
```
Every span in the trace should map onto one of these terms. **If a span does not map to a term, your
instrumentation is missing something.**

**Goodput as the headline SLO metric** `[D]`:
```
goodput = |{r : TTFT(r) ≤ T_t AND ITL(r) ≤ T_i}| / |requests|
```
Alert on goodput, not on mean latency. A mean-latency alert fires after users have already suffered.

**Judge agreement — the number that validates your eval** `[D]`:
```
agreement = |judge agrees with human on gold set| / |gold set|
```
**Do not trust a judge you have not measured.** Target agreement on a human-labelled set, and
recheck when the judge model or prompt changes.

**Cost per task** `[T]`:
```
cost_task = (prompt_tokens − cached_tokens) × p_fresh
          + cached_tokens × p_cached          # ~10x cheaper [T]
          + completion_tokens × p_out
```
Attributing this per tenant and per agent is what makes the 100→42→26→11 ladder visible in
production rather than in a blog post `[T]`.

---

## Configuration

```python
# OpenTelemetry — one span per LLM call, GenAI semantic conventions [R]
from opentelemetry import trace
tracer = trace.get_tracer("agent")

with tracer.start_as_current_span("chat") as span:
    span.set_attribute("gen_ai.system", "vllm")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
    span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
    # YOUR attributes: what makes this trace debuggable
    span.set_attribute("app.prefix_cache_hit", hit_tokens)
    span.set_attribute("app.route_replica", replica_id)
    span.set_attribute("app.tool_calls", len(calls))
```

```yaml
# Collector: sample aggressively at the tail, keep 100% of errors and slow requests
processors:
  tail_sampling:
    policies:
      - name: errors            {type: status_code, status_code: {values: [ERROR]}}
      - name: slow              {type: latency, threshold_ms: 10000}
      - name: baseline          {type: probabilistic, percentage: 5}
```

**Eval-gated CI** `[D]`: every prompt, model, quantization or config change runs the eval suite;
merge is blocked on regression. This is the mechanism that makes **"always rerun evals after
quantizing"** `[T]` an enforced rule rather than a good intention.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Cannot explain a slow request | missing spans between gateway and engine | trace propagation `[D]` |
| Quality complaints with green dashboards | measuring only plane 1 | you have no quality eval `[D]` |
| Eval passes, users unhappy | eval set too easy / off-distribution | add hard cases and real traffic samples `[T]` |
| Judge scores swing run to run | judge bias or judge drift | measure agreement on gold set `[D]` |
| Quantization shipped a regression | evals not rerun | this is the corpus's explicit rule `[T]` |
| Agent succeeded but badly | only outcome evaluated, not trajectory | add trajectory assertions `[D]` |
| Same prompt, different output, "bug" | **non-determinism at temperature 0** | expected `[T]` — not a bug |
| Cost unknown until the invoice | no per-request token accounting | gateway + span attributes `[T]` |
| Trace volume unaffordable | sampling everything | tail-sample; keep errors at 100% `[D]` |
| PII found in traces | prompts logged verbatim | redaction policy at the collector `[T]` |

---

## Gotchas

- **Non-determinism at temperature 0 is real.** The corpus states it plainly `[T]`. Do not build
  tests that assert exact string equality; assert properties.
- **Quantization changes quality *and* makes outputs more variable — and fingerprints your model**
  `[T]`. Two reasons it belongs in the eval gate.
- **Always rerun evals after quantizing** `[T]`. This is the single most-repeated rule in the cost
  material, and the most-skipped.
- **Traces and metrics answer different questions.** Metrics say *something is wrong*; traces say
  *what*. Ship both or you will debug blind `[D]`.
- **LLM-as-judge has known biases** — position, verbosity, self-preference `[T]`. Mitigate by
  randomising order, pinning length, using a different judge family, and **measuring agreement**.
- **Trajectory eval is not optional for agents.** A correct answer reached by an unsafe or absurd
  path is a production incident waiting to happen `[T]`.
- **Cost is a first-class observability signal.** Token accounting per request and per tenant is
  what turns the cost ladder into an operational metric `[T]`.
- **Eval-gated CI is the only durable quality mechanism.** Un-gated evals decay into unused scripts
  `[D]`.
- **Sampling policy is a design decision.** Trace everything and you cannot afford it; trace nothing
  and you cannot debug. Tail-sample `[D]`.
- **Prompt changes are code changes.** Version them, review them, gate them `[T]` Prompt Management
  as Code / DSPy.

---

## When to use what

| Situation | Do |
|---|---|
| Any production deployment | metrics + traces + logs, three planes `[D]` |
| Debugging latency | latency decomposition spans; map every span to a term |
| Debugging quality | offline eval suite with a validated judge `[D]` |
| Agent debugging | **trajectory** spans (tool calls, retries), not just final output `[T]` |
| Before any model/quant/config change | eval-gated CI `[T]` |
| Cost control | per-request token + cost telemetry `[T]` |
| High trace volume | tail sampling; 100% of errors, 5% of the rest `[D]` |
| Regulated data | redaction at the collector, before storage `[T]` |

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Prompt_Management_as_Code_Versioning_Injection_DSPy.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
