# Cheat Sheet: FinOps, Token Economics & Sovereignty

> `T19` · **Transcript coverage:** primary · [Case study](../01-case-studies/T19-finops-sovereignty.md) · [Blueprint](../03-design-blueprints/T19-finops-sovereignty/HLD.md) · [Interview bank](../02-interview-questions/T19-finops-sovereignty.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **The cost ladder** | **100 → 42 → 26 → 11** | `[T]` LLMOps cost talk |
| Step 1 → 2 | **100 → 42** — continuous batching | `[T]` |
| Step 2 → 3 | **42 → 26** — quantization (16 → 8 → 4 bit; AWQ, GPTQ) | `[T]` |
| Step 3 → 4 | **26 → 11** — caching + routing | `[T]` |
| Cache reads vs fresh tokens | **~10× cheaper** | `[T]` |
| Precision journey | **16-bit → 8-bit → 4-bit** | `[T]` |
| Mandatory gate on every step | **"always rerun **your** evals after quantizing"** | `[T]` |
| KV utilisation target | fill to **90% of VRAM** | `[T]` NxtGen |
| Rack-scale memory demo | **4 servers**, pooled box — the sovereign/owned path | `[T]` SK hynix |
| Sovereignty dimensions | **control · choice · trust · economics · continuity** — a system property, not a flag | `[T]` sovereign AI talk |
| Token-cost decline (cited study) | Feb **30¢** → Apr **16¢** for a 10k-token task | `[T]` Token Raj |
| Same study: code lines drafted | **630 → 91** | `[T]` Token Raj |
| Same study: files touched | **8.2 → 3.6** | `[T]` Token Raj |
| The catch in that study | billing fell **less than the workload did** — because **thinking tokens rose** | `[T]` Token Raj |

**Read the ladder as a sequence, not a menu** `[T]`: the corpus presents it as an ordered
progression where each step multiplies the last. Doing them out of order leaves savings on the
table — for example, quantizing before fixing batching optimises a smaller bill.

---

## The one-table summary

| Lever | Ladder step | Effort | Risk | Reversible? |
|---|---|---|---|---|
| **Continuous batching** | 100 → 42 | config (free) | very low | yes |
| **Quantization 16→8 bit** | 42 → 26 | config | low, **eval required** | yes |
| **Quantization 8→4 bit** | (part of the same step) | calibration | **higher — quality cliffs** | yes |
| **Prefix caching + routing** | 26 → 11 | integration work | low | yes |
| **KV offload tiering** | not on the ladder | infra | low | yes |
| **Speculative decoding (MTP)** | not on the ladder | training for MTP | low | yes |
| **Right-sizing / autoscaling** | not on the ladder | ops | low | yes |
| **Buy vs build** | not on the ladder | strategic | high | **no** |
| **Sovereignty choices** | not on the ladder | strategic | high | **no** |

**Two different books** `[D]`: the ladder is **efficiency** (run the same system for less). Buy-vs-
build and sovereignty are **structure** (change what system you run). Structure decisions are the
expensive ones because they are hard to reverse — make them deliberately, and make them once.

---

## Formulas

**The ladder, arithmetically** `[T]`:
```
cost_per_token = base × 0.42 × 0.62 × 0.42
               ≈ base × 0.11
```
So the corpus's 100 → 11 is roughly cumulative: batching ×0.42, quantization ×0.62,
caching+routing ×0.42. **Each factor is a separate engineering effort** — and the last one,
routing, is the one teams skip because it is "just plumbing" `[T]`.

**Unit economics you must be able to write down** `[D]`:
```
cost_request = (prompt_tokens − cached) × p_fresh
             + cached × p_cached           # ~10x cheaper [T]
             + completion_tokens × p_out
             + amortised_gpu_hour / throughput
```
The **amortised GPU-hour term is what your batching and parallel-efficiency work moves**; the token
terms are what caching and routing move. Together they are the ladder `[T]`.

**Break-even for self-hosting** `[D]`:
```
cost_build  = GPUs + datacentre + staff + opportunity_cost
cost_buy    = per_token_price × volume
build_wins  ⟺  cost_build / volume_per_year < per_token_price   (sustainably)
```
The corpus's sovereignty talks add terms the pure arithmetic misses: **control, residency,
and the option value of owning the capability** `[T]`.

**Sovereignty is a system property with dimensions — not a flag** `[T]`.

The talk rejects the binary framing outright: *"Think about sovereignty as something like binary —
it's a sovereign setup or non-sovereign setup. I think that's wrong. You need to start looking at it
as a system property … this particular system property has various dimensions."* The five the
speaker enumerates `[T]`:

| Dimension | The question it asks |
|---|---|
| **Control** | Where does the actual inference run? |
| **Choice** | Which model, and which accelerator? (often read as part of *control* — the speaker lists it with "and then the choice…", so the count is **four or five** depending on how strictly you split it `[T]`) |
| **Trust** | What code and artifacts are being executed in this environment? |
| **Economics** | Who controls the token cost? (*"the bill should not come to you from Belarus"*) |
| **Continuity** | Can you operate without a single vendor when geopolitics changes the rules? |

**The load-bearing argument is that the *inference layer itself* is a sovereignty component people
forget** `[T]`: *"if inference leaves your control, can you really call that as sovereign AI?"* You
can have the data in-house, an open model, and infrastructure you own — and still lose sovereignty
at the layer that actually turns the data into answers. Hence: *"it's not about owning a particular
component, but it is all about owning an entire system that delivers the outcome."*

**And inference matters more than training here** `[T]`: *"you train once … but inference is not
like that."* Every user question, every application, every agent turn is inference — which is why
latency, throughput, routing, caching and observability become sovereignty concerns and not just
performance ones.

The full stack the speaker names, bottom-up `[T]`: **Linux → accelerators → models and architectures
→ inference engines (vLLM) and distributed inference → the application** where value is created.

---

## Configuration / practices

```python
# FinOps telemetry: attribute every token, every request, every tenant [T]
span.set_attribute("cost.prompt_fresh_tokens",  fresh)
span.set_attribute("cost.prompt_cached_tokens", cached)   # ~10x cheaper [T]
span.set_attribute("cost.completion_tokens",    out)
span.set_attribute("cost.tenant",               tenant_id)
span.set_attribute("cost.agent_task_id",        task_id)
```

```yaml
# The cost guardrails that prevent surprise invoices [D]
quotas:  {per_tenant_tokens_per_day: N, hard_stop: true}
budgets: {per_agent_task_tokens: M, on_exceed: terminate}
routing: {prefixAware: true}      # the 26 -> 11 step [T]
batching: {continuous: true, max_num_seqs: 256}
quantization: {weights: fp8, kv: fp8, eval_gate: required}   # [T]
```

**Eval-gate every efficiency decision** `[T]`: the cost ladder and quality are coupled. Each rung
must be bought with a passing eval suite, or you are trading quality for a smaller invoice without
noticing.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Bill flat despite optimization work | you optimised the wrong end — check prefill vs decode mix | token accounting `[T]` |
| Cost per request falling, quality too | quantization without an eval gate | rerun evals `[T]` |
| Cache "working" but no savings | routing defeats it (cache misses across replicas) | prefix-aware routing `[T]` |
| Cached tokens still billed at full rate | provider/pricing tier mismatch | check the cached-token price `[T]` |
| Agent costs unbounded | no per-task budget | set token/step budgets `[D]` |
| GPU spend high but utilisation low | no autoscaling on the right signal | see [T15](T15-autoscaling-slo.md) `[D]` |
| Compliance finding despite "sovereign" vendor | sovereignty assessed as a flag, not per dimension | score each dimension `[T]` |
| Data residency assumed, not proven | no audit of where inference ran | require verifiable residency `[T]` |
| Self-hosting cheaper on paper, worse in practice | staff and utilisation underestimated | include ops cost and realistic utilisation `[D]` |
| Cannot attribute spend to a team | no per-tenant tagging | tag at the gateway `[T]` |

---

## Gotchas

- **The ladder is ordered and cumulative, and the last step is the most-skipped.** Caching and
  routing (26 → 11) require engine↔router integration, so teams stop at 26 and assume the rest is
  procurement `[T]`.
- **Cache reads are ~10× cheaper — but only if routing preserves the cache** `[T]`. Caching without
  prefix-aware routing shows a high hit rate in the engine and a terrible one in production `[T]`.
- **Quantization is not free; the corpus's rule is absolute**: **always rerun evals after
  quantizing** `[T]`. It also worsens non-determinism and fingerprints the model `[T]`.
- **8-bit is close to free; 4-bit is a decision.** Take FP8 unless you have measured headroom `[D]`.
- **Structure beats efficiency for total cost, but only after efficiency is exhausted.** Buying GPUs
  to avoid doing batching and caching work is the most expensive way to solve a config problem `[D]`.
- **Sovereignty has several dimensions, not one flag** `[T]`: **control, choice, trust, economics,
  continuity** (choice is often folded into control — see above). The talk explicitly rejects the binary framing and treats it as a system property.
  A vendor can score well on one dimension and badly on another.
- **Self-hosting has an option value beyond the arithmetic** `[T]` — control of the roadmap,
  no per-token vendor risk, and the ability to run models that will never be hosted for you. Price
  that explicitly rather than treating it as sentiment.
- **Rack-scale pooled memory is the infrastructure-ownership play** `[T]` — it is simultaneously a
  performance decision (pooled < RDMA << TCP/IP) and a sovereignty decision.
- **Token pricing is not your cost.** Your cost is `GPU-hours / goodput`, and it can fall while your
  vendor invoice rises (or vice versa) `[D]`.
- **The Token Raj study is the cautionary tale for that last point** `[T]`. Unit cost for a 10k-token
  task fell **30¢ → 16¢** between February and April, and per-task work fell much further — code
  lines drafted **630 → 91**, files touched **8.2 → 3.6**. Yet **billing did not fall
  proportionally, because thinking tokens rose**. Falling unit prices do not guarantee a falling
  bill when the workload mix shifts toward reasoning. *(Speaker attributes the study to Stanford
  researchers; the lab name and the model named are ASR-garbled — treat those two as approximate,
  the figures as reported.)*
- **Attribution is a prerequisite for optimisation.** You cannot reduce what you cannot assign to a
  tenant, agent or feature `[T]`.
- **Confidential computing and model fingerprinting interact** `[T]`: if the inference stack is
  confidential, quantization choices can leak information about it.
- **Budget limits are both a FinOps and a safety control** — an unbounded agent is a self-inflicted
  denial of service `[D]`.

---

## When to use what

| Situation | Do |
|---|---|
| First cost reduction | **continuous batching** — free, large, reversible `[T]` |
| Second | **FP8 quantization**, then 4-bit if evals permit `[T]` |
| Third | **prefix caching + prefix-aware routing** — the 26 → 11 step `[T]` |
| Long-context, memory-bound | KV quantization + KV offload tiering |
| Latency-sensitive, small batch | MTP/speculative decoding — but check it is not prefill-bound `[T]` |
| Unpredictable traffic | right-sizing + saturation autoscaling `[T]` |
| Regulated / public sector | score **every** sovereignty dimension `[T]` |
| Very high stable volume | model buy-vs-build honestly, including ops cost |
| You need control of the roadmap | self-host; price the option value explicitly `[T]` |
| Cost unknown | instrument per-request token and cost telemetry first `[T]` |

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` `[R]`
