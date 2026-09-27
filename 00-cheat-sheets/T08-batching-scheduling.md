# Cheat Sheet: Continuous Batching, Scheduling & Fairness

> `T08` · **Transcript coverage:** partial · [Case study](../01-case-studies/T08-batching-scheduling.md) · [Blueprint](../03-design-blueprints/T08-batching-scheduling/HLD.md) · [Interview bank](../02-interview-questions/T08-batching-scheduling.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **Continuous batching's contribution** | **100 → 42** on the cost ladder | `[T]` LLMOps cost talk |
| Agentic fairness (least-attained service) | request latencies cut **2–3×** (stated as up to **50%** reduction) | `[T]` llm-d talk |
| Cost of that fairness | small overall throughput gain — *not* a throughput loss | `[T]` llm-d talk |
| **FCFS queueing** | **"adds nothing"** | `[T]` llm-d talk |
| Saturation threshold examples | KV cache **80% full**, or **average active requests > 8** | `[T]` llm-d talk |
| Prefill share of agentic tokens | **~98%** | `[T]` llm-d talk |
| Turn-priority strategy | favour agents **near completion** so their KV is freed | `[T]` llm-d talk |

---

## The one-table summary

| Mechanism | What it solves | Cost | Use when |
|---|---|---|---|
| **Static batching** | nothing much — waits for the slowest | poor GPU utilisation | never, in production |
| **Continuous batching** | iteration-level admission; new requests join mid-flight | scheduler complexity | **always** — the baseline |
| **Chunked prefill** | interleaves prefill with decode to stop long prefills stalling everyone | slight TTFT increase | long prompts mixed with short ones |
| **Prefill/decode split** | isolates the two resource profiles | KV transfer | see [T12](../01-case-studies/T12-disaggregation-kv-transfer.md) |
| **Priority bands** | protects interactive traffic from batch jobs | needs a priority source | mixed premium/best-effort tenants `[T]` |
| **Least-attained service** | fairness across *sessions*, not requests | small bookkeeping | agentic workloads `[T]` |
| **Turn priority** | frees KV by finishing near-done agents | needs completion prediction | KV-saturated agentic serving `[T]` |
| **Preemption + recompute** | admits a high-priority request | throws away work | rare, high-priority arrivals |
| **Preemption + swap** | admits without losing work | PCIe/NVLink transfer | KV is expensive to recompute |

---

## Formulas

**Continuous batching throughput gain** — the reason it works:
```
Naive:     batch runs until the LONGEST sequence finishes
Continuous: a slot frees the instant ANY sequence finishes → immediately refilled
Utilisation ≈ (mean sequence length) / (max sequence length)     [naive]
Utilisation ≈ 1 − idle_fraction                                   [continuous]
```
With a skewed length distribution the naive penalty is severe; this is the entire 100→42 step `[T]`.

**Chunked prefill** — split a long prompt into chunks so decode steps interleave:
```
prefill_chunk_size ≈ max_num_batched_tokens − (decode_seqs × 1)
```
Choose `max_num_batched_tokens` so a chunk fits alongside the current decode batch. Too small ⇒
TTFT suffers; too large ⇒ decode stalls and ITL spikes `[D]`.

**Fairness — least-attained service** `[T]`:
```
attained(s) = service_received(s) / demand(s)
next = argmin_s attained(s)
```
Prioritise the *session* furthest behind its fair share, not the oldest request.

**Get the direction of the starvation right — it is counter-intuitive.** The speaker's scenario is
`[T]`: three sessions arrive, *"there is a small session and a **large** session that comes in and
then takes all the dispatch cycles. So the **shorter sessions are starving** now."* The big session
hogs the scheduler; the small ones wait. Least-attained service fixes **that**, and the second-order
effect is what produces the big win — *"short sessions finish much faster and leaving the room for
larger sessions to also finish much faster."* Result: **latencies down up to 50% (2–3×)**, *"overall
token throughput increased a bit but the request latencies came down by a lot"* `[T]`.

So it is not "protect the long agent from short requests" — it is **"stop one big session from
monopolising the scheduler, which then speeds up everyone including the big one."**

**Turn priority — the complementary strategy** `[T]`: an agent at turn ~100 is *likely to finish
soon*, so prioritising the near-done agents lets them **finish and evict their KV**, freeing memory
under saturation. The speaker notes it is in effect the **opposite** mechanism to least-attained
service, and that they were working on integrating the two `[T]`.

**Goodput**
```
goodput = |{r : latency(r) ≤ SLO}| / |requests|
```
The scheduling objective. Every batching decision should be evaluated against this, not throughput.

---

## Configuration

```bash
vllm serve <model> \
  --max-num-seqs 256 \                 # concurrent sequences (raises throughput, raises ITL)
  --max-num-batched-tokens 8192 \      # the chunked-prefill knob
  --enable-chunked-prefill \           # interleave prefill with decode
  --scheduling-policy fcfs \           # or priority
  --preemption-mode recompute          # or swap (swap needs CPU memory)
```

**Tuning order** `[D]`: (1) `--max-num-seqs` to raise throughput until ITL crosses the SLO;
(2) `--enable-chunked-prefill` to tame TTFT under mixed lengths; (3) priority policy; (4) fairness
policy. Stop when goodput stops improving.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| GPU idle while requests queue | static batching somewhere; admission blocked | is continuous batching actually on? |
| ITL spikes intermittently | a long prefill is blocking decode | enable chunked prefill |
| TTFT terrible for short prompts behind long ones | head-of-line blocking on prefill | chunked prefill; prefill/decode split |
| One tenant starves others | FCFS — which "adds nothing" | least-attained service `[T]` |
| Throughput fine, P99 terrible | over-batching | goodput, not throughput |
| Agent sessions never finish | long programs lose to short requests | session-level fairness `[T]` |
| Thrashing — constant preemption | batch too large for available KV | lower `--max-num-seqs`, raise `--gpu-memory-utilization` |
| Latency degrades as KV fills | no admission control | add a saturation threshold (e.g. 80%) `[T]` |

---

## Gotchas

- **Continuous batching is not a tuning option — it is the baseline.** Everything else is layered
  on it `[T]`.
- **FCFS is not a neutral default.** The llm-d team states plainly that it "adds nothing" for
  agentic workloads `[T]`.
- **Session-level fairness ≠ request-level fairness, and the starvation runs the way you may not
  expect.** The transcript's scenario is that **one large session takes all the dispatch cycles and
  the short sessions starve** `[T]`. Fairness is computed per **agentic program** (attained service),
  not per request — and the payoff is that short sessions finish fast, *which frees room for the
  large ones too* `[T]`.
- **Turn priority is counter-intuitive but correct**: favour the agents *closest to finishing*, so
  their KV can be evicted and the memory reused. It improves throughput under KV saturation `[T]`.
- **Preemption cost depends on the mode.** `recompute` throws away work and burns prefill again;
  `swap` costs a transfer. Pick by which resource you are short of `[D]`.
- **Chunked prefill shifts latency, it does not remove it.** You trade a small TTFT increase for
  much better ITL stability.
- **Over-batching is the classic own-goal.** Throughput rises monotonically with batch size; goodput
  does not. Always measure the one that pays the bills.
- **Admission control is part of scheduling.** Without a saturation threshold the system degrades
  gracefully into uselessness rather than shedding load `[T]`.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Opening_Note_vLLM_Inference_Meetup_Bengaluru_September_19_2026.txt`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/04-batching-strategies.md` `[R]`
