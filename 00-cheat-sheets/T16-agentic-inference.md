# Cheat Sheet: Agentic Inference

> `T16` · **Transcript coverage:** primary · **The workload that redefines the whole stack**
> [Case study](../01-case-studies/T16-agentic-inference.md) · [Blueprint](../03-design-blueprints/T16-agentic-inference/HLD.md) · [Interview bank](../02-interview-questions/T16-agentic-inference.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Prefill share of agentic tokens | **~98%** | `[T]` llm-d |
| The rest — decode | **~2%** | `[T]` llm-d |
| Consequence | the workload is **TTFT-bound, not ITL-bound** | `[T]`/`[D]` |
| KV eviction in agentic workloads | frequent enough that the router consumes **per-request** create/evict events, with an offload tier + retention API to manage it `[T]` | `[T]`/`[D]` llm-d |
| KV offload win on session return | **~5× TTFT** | `[T]` llm-d |
| Fairness improvement (session-level) | latencies cut **2–3×** | `[T]` llm-d |
| Concurrency achievable (tuned 1P1D) | **~20,000** | `[T]` ROCm/WideEP |
| The cliff | **28k inputs** ⇒ KV recomputed every turn | `[T]` ROCm/WideEP |
| Cost ladder | **100 → 42 → 26 → 11** | `[T]` LLMOps cost talk |
| Tools the model may choose from | an open, growing set | `[T]` |

**Why agents break serving assumptions** `[T]`: a chat request is one short prompt and one long
output. An agent is the inverse — a **huge, growing prefix and a tiny output**, repeated many times
per task, with **idle gaps** between tool calls during which the KV is evicted. Every optimisation
built for chat (speculative decoding, ITL tuning, output-length batching) is aimed at the 2%.

---

## The one-table summary

| Pattern | What it is | Fits inference how |
|---|---|---|
| **ReAct** | reason → act → observe loop | many short turns; prefix grows every turn `[T]` |
| **CodeAct** | actions are code, executed | same loop, larger observations; sandbox needed `[T]` |
| **Reflexion** | self-critique and retry | multiplies turns — cost multiplies with it `[T]` |
| **Tool calling (MCP)** | standardised tool/resource protocol | the tool result re-enters the prefix `[T]` |
| **A2A** | agent-to-agent protocol | agent B becomes a *tool* of agent A `[T]` |
| **Multi-agent (LangGraph-style)** | explicit graph of specialists | N× the inference calls; needs routing + budget `[T]` |
| **Long-horizon** | hours-to-days tasks | durable execution, checkpointing, memory `[T]` |
| **Harness internalisation** | the scaffolding moves *into* the model | fewer calls per task over time `[T]` |
| **Agent memory** | external store for facts/history | trades inference cost for retrieval cost `[T]` |
| **Context condensation** | summarise history to fit | **reduces prefix-cache hit rate** `[T]` |

**The one-line architecture** `[T]`/`[D]`: **agent loop on top, tool sandbox beside, memory behind,
and a prefix-cache-aware, KV-offloading, fairness-scheduled serving stack underneath.** The agent
framework is the easy part; the serving stack is what determines whether it is affordable.

---

## Formulas

**Why agents are prefill-bound** `[T]`:
```
tokens_per_task = Σ_turns (prefix_tokens + output_tokens)
prefix_tokens grows monotonically with turn count
```
With **98% of tokens in prefill** `[T]`, per-task cost is dominated by re-reading a growing context.
This is why **prefix caching and KV offload are the two highest-leverage optimisations** in an
agentic stack — and why speculative decoding, aimed at decode, barely moves the needle.

**Cost of losing the cache between turns** `[D]`:
```
extra_cost(turn) ≈ prefix_tokens(turn) × fresh_token_cost
                 − prefix_tokens(turn) × cached_token_cost
```
With cache reads ~10× cheaper `[T]` and a 50k-token prefix by turn 10, losing the cache between
turns is the single largest avoidable cost in the system.

**Tool-call concurrency** `[D]`:
```
wall_clock ≈ Σ_sequential_steps (LLM_latency + tool_latency) / parallelism
```
Parallel tool calls are usually the cheapest speedup available — but they multiply **concurrency**
on the serving tier, which is what forces the 20k-concurrency class of deployment `[T]`.

**Turn-vs-session fairness** `[T]`:
```
attained(session) = service_received / demand
```
Schedule by least-attained **session**, not request. This is what produced the 2–3× latency cut for
long agent programs `[T]`.

---

## Configuration

```python
# The loop, with the serving-relevant decisions made explicit [D]
while not done:
    resp = llm.chat(
        messages=history,             # keep the PREFIX STABLE; append, don't rewrite
        tools=TOOLS,                  # MCP-discovered tool list
        cache_hint=stable_prefix_hash # tells the router where to send this
    )
    if resp.tool_calls:
        results = run_parallel(resp.tool_calls, sandbox=True)   # tool latency, not LLM
        history += results            # append-only keeps prefix cache valid
    else:
        done = True
```

```yaml
# Serving side: the agentic-shaped deployment
decode:  {replicas: 4}
prefill: {replicas: 2}             # prefill-heavy ratio
kvOffload: {tier: cpu, retention: session}   # survive idle gaps [T]
routing: {prefixAware: true, kvEvents: true} # never round-robin an agent
scheduling: {policy: leastAttainedService, turnPriority: true}  # [T]
```

**Two rules for the client side** `[D]`: (1) **append to the history, never rewrite it** — rewriting
invalidates the prefix cache; (2) **put volatile content (timestamps, fresh tool output) last** —
anything early in the prompt that changes every turn destroys cacheability.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Costs explode with turn count | prefix cache missing between turns | routing + stable prefix `[T]` |
| Turn 5 is far slower than turn 1 | KV evicted during the idle gap | CPU/SSD offload tier `[T]` |
| Throughput collapses at long context | the 28k-input recomputation cliff | KV capacity vs context `[T]` |
| One agent starves the fleet | FCFS; no session fairness | least-attained service `[T]` |
| P99 terrible despite good mean | no admission control | saturation gate `[T]` |
| Tool calls serialise everything | no parallel tool execution | parallelise independent calls |
| Agent loops forever | no step/token budget | budget + termination check `[T]` |
| Prompt injection from tool output | no rail at the tool boundary | see [T18](T18-guardrails-security.md) `[T]` |
| Session lost mid-task | no durable execution | checkpoint/restore `[T]` |
| Cost/quality drift over versions | no continuous eval of trajectories | see [T17](T17-observability-evals.md) `[T]` |
| Cache hit rate falls after adding summarisation | context condensation rewrites the prefix | expected tradeoff `[T]` |

---

## Gotchas

- **Agentic traffic inverts the chat assumption.** 98% prefill `[T]` — so **TTFT and KV reuse are
  the levers**, and output-side tricks (speculative decoding, ITL tuning) are marginal here.
- **KV eviction is the normal state, not an error.** Long idle gaps plus large prefixes mean the
  cache is gone when the next turn arrives `[T]`. Design for it.
- **Never route an agent round-robin.** Prefix-cache-aware routing is the difference between the
  cost ladder's 26 and 11 `[T]`.
- **Append-only history is a performance requirement, not a style choice.** Rewriting the message
  list invalidates every cached block `[D]`.
- **Context condensation trades cache for space.** Summarising history replaces cached tokens with
  new ones — sometimes correct, but know what you are paying `[T]`.
- **Fairness must be session-level, and the direction of starvation is counter-intuitive.** The
  documented failure is **one large session taking all dispatch cycles while short sessions starve**
  `[T]` — not the reverse. Least-attained service over *agentic programs* fixes it, and the win
  compounds: short sessions finish fast, *"leaving the room for larger sessions to also finish much
  faster"*, at **2–3× lower request latency** `[T]`.
- **Turn priority is counter-intuitive and right**: finish the agents nearest completion so their
  KV is freed `[T]`.
- **Multi-agent multiplies cost by the number of agents.** Every hop is a full inference call with
  its own prefix. Budget it explicitly `[T]`.
- **Harness internalisation is the long-run trend** `[T]`: as models absorb scaffolding, the same
  task needs fewer calls. Architect so your serving layer survives that shift — it changes call
  volume, not the serving primitives.
- **Long-horizon agents need durable execution and checkpointing** — a preempted agent without a
  checkpoint loses hours of work `[T]`.
- **Tool sandboxing is a security boundary, not a convenience.** Treat every tool output as
  untrusted input `[T]` — see [T18](T18-guardrails-security.md).
- **Budget tokens, steps and wall-clock per task.** An agent without a budget is a denial-of-service
  against your own fleet `[D]`.

---

## When to use what

| Situation | Do |
|---|---|
| Any multi-turn agent | stable append-only prefix + prefix-aware routing `[T]` |
| Tool-heavy, many turns | KV offload to CPU/DRAM between turns `[T]` |
| Many concurrent agents | session-level fairness + saturation admission `[T]` |
| Long-horizon (hours+) | durable execution + checkpointing + memory `[T]` |
| Parallel tool calls available | parallelise — biggest cheap win `[D]` |
| Multi-agent topology | justify every hop; each is a full inference call `[T]` |
| Cost pressure | measure prefill share first — it is where the money is `[T]` |
| Untrusted tools / web content | rails plus sandbox before anything else `[T]` |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Jerry_Tworek_-_Opportunities_and_Challenges_for_Long_Horizon_Agents.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Jianfeng_Gao_-_Agentic_Modeling_via_Internalizing_Agent_Harnesses.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Ankit_Sobti_-_From_Agent_Demos_to_Production_How_Postman_Is_Building_Reliable_AI.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Weizhu_Chen_-_Continuous_Model_Improvement.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_AI_Agents_Actually_Work_ReAct_Tools_Reflexion.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Multi-Agent_Systems_with_LangGraph_The_Klarna_Uber_Lessons.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
