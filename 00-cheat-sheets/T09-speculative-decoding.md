# Cheat Sheet: Speculative Decoding

> `T09` · **Transcript coverage:** partial · **Bridge topic** (algorithm ↔ serving)
> [Case study](../01-case-studies/T09-speculative-decoding.md) · [Blueprint](../03-design-blueprints/T09-speculative-decoding/HLD.md) · [Interview bank](../02-interview-questions/T09-speculative-decoding.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **MTP in production** | **~2× throughput** | `[T]` llm-d talk |
| Where MTP appears | the llm-d talk reports it as a production win, not a research result | `[T]` |
| Draft model requirement | must share the **tokenizer** with the target | `[T]`/`[D]` |
| Acceptance condition | `min(1, p_target / p_draft)` | `[D]` standard construction — **not** in CMU lecture 2, which contains no acceptance rule |
| Speculative decoding targets | **decode** only — prefill is unaffected | `[D]` |

---

## The one-table summary

| Variant | Draft source | Extra state | Use when |
|---|---|---|---|
| **Draft-model speculation** | a small separate model | a second model in memory | you have a good small model of the same family |
| **MTP (multi-token prediction)** | extra prediction heads in the target model | heads trained in | **production default** — ~2× `[T]` |
| **Medusa** | multiple decoding heads on the target | heads + tree attention | you can fine-tune heads |
| **EAGLE** | a small autoregressive draft on features | a small draft net | best acceptance rates in practice |
| **N-gram / prompt lookup** | match against the prompt itself | none | **summarisation, RAG, code editing** — output overlaps input |
| **Self-speculation** | skip layers of the target itself | none | no extra model budget |

**N-gram speculation is the underrated one** `[D]`: when the output is likely to quote the input —
summarise this, extract this, edit this file — you can propose candidates by string matching and
pay almost nothing for drafting.

---

## Formulas

**Acceptance rule** — accept the draft token with probability:
```
accept = min(1, p_target(x) / p_draft(x))
```
If rejected, resample from the *residual* distribution
`normalise(max(0, p_target − p_draft))`. This construction is what makes the output **exactly
distributed as the target model** — speculative decoding is not an approximation `[T]`.

**Expected speedup**
```
tokens_per_step = 1 + Σ_{i=1..γ} Π_{j=1..i} α_j
speedup ≈ tokens_per_step / (1 + γ · c)
```
- `γ` = number of draft tokens per step
- `α` = per-token acceptance rate
- `c` = draft cost as a fraction of a target forward pass

**This is the whole engineering problem.** Two consequences people miss:

1. **Acceptance decays with position.** `α` is usually high for token 1 and falls off, so long
   drafts (`γ` large) stop paying. `γ = 4–8` is typical `[D]`.
2. **Speculation only helps when the target is memory-bound.** Decode reads all weights to produce
   one token, leaving tensor cores idle — that idle compute is what verifies the draft "for free".
   **In a compute-bound (large-batch) regime, speculation makes things worse** `[D]`.

**Worked example** `[D]`: `γ=4`, `α=[0.8, 0.7, 0.6, 0.5]`, `c=0.15`.
`tokens_per_step = 1 + 0.8 + 0.56 + 0.336 + 0.168 = 2.86`.
`speedup ≈ 2.86 / (1 + 4×0.15) = 2.86 / 1.6 ≈ 1.79×`.

---

## Configuration

```bash
# vLLM — n-gram speculation (no extra model, great for RAG/summarisation)
vllm serve <model> --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'

# vLLM — draft model
vllm serve <target> --speculative-config '{"model":"<draft>","num_speculative_tokens":5}'

# vLLM — MTP (model must ship MTP heads; this is the ~2x production path)
vllm serve <model-with-mtp> --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

**Tuning `γ`** `[D]`: raise until the measured speedup stops rising. Monitor two counters —
**draft acceptance rate** and **draft tokens per accepted token**. A falling acceptance rate means
`γ` is too high or the draft has drifted from the target.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Speedup is <1 (it got slower) | target is compute-bound, not memory-bound | batch size — speculation helps small batches only |
| Acceptance rate collapses | draft model drifted / different tokenizer | verify tokenizer identity first |
| Great on chat, useless on summarisation | wrong variant | try **n-gram** — output quotes the input |
| Draft model OOMs | a second model's weights plus KV | account for draft weights in the memory budget |
| Output differs subtly from non-speculative | bug — this should be *exact* | check the residual resampling is implemented |
| Latency variance up | acceptance varies per request | report P95; speculation is a distribution shift |
| Works on one model, not another | no MTP heads / no good small sibling | Medusa/EAGLE, or n-gram |

---

## Gotchas

- **Speculative decoding is exact, not approximate.** If your output distribution changes, you have
  a bug. The residual resampling is not optional `[T]`.
- **It attacks decode, never prefill.** Agentic workloads are ~98% prefill tokens `[T]` — so
  speculation does much less for agents than the headline suggests. **TTFT work matters more there.**
- **The speedup depends on batching.** Larger batches move decode *towards* compute-bound (the
  weights are amortised over more sequences), eroding the idle compute that speculation exploits.
  Speculation and large-batch throughput are in tension `[D]`.
- **MTP is not free at training time.** The heads must be trained in. You cannot bolt MTP onto an
  arbitrary checkpoint `[T]`.
- **N-gram speculation needs no model at all** and is often the best cost/benefit for RAG, summarisation
  and code editing, because the output overlaps the input `[D]`.
- **Acceptance rate is the metric to watch, not speedup.** Speedup is downstream of it and of the
  batch shape; acceptance is a property of the draft/target pair `[D]`.
- **Draft model choice is a family question.** A draft from a different family has a different
  tokenizer and cannot be used at all `[D]`.

---

## When to use what

| Situation | Do |
|---|---|
| Latency-sensitive, small batch, decode-bound | **speculative decoding** |
| You control training | **MTP** — the ~2× production path `[T]` |
| Summarisation / extraction / code edit / RAG | **n-gram or prompt-lookup** — nearly free |
| Output is creative and unpredictable | skip it — acceptance will be low |
| Prefill-dominated (agentic) | skip it — optimise TTFT and KV reuse instead |
| Large batch, throughput-oriented | skip it — you are compute-bound already |

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/03-speculative-decoding.md` `[R]`
