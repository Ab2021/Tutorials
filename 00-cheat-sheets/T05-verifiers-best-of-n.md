# Cheat Sheet: Reward Models, Verifiers & Best-of-N

> `T05` · **Transcript coverage:** primary · [Case study](../01-case-studies/T05-verifiers-best-of-n.md) · [Blueprint](../03-design-blueprints/T05-verifiers-best-of-n/HLD.md) · [Interview bank](../02-interview-questions/T05-verifiers-best-of-n.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Typical sample size | `n = 10` or `100` | `[T]` CMU lecture 12 |
| **Why `n = 32` specifically** | it **fits one batch** — the 33rd would need a second pass/GPU | `[T]` CMU lecture 12 |
| KL bound | `log n − (n−1)/n` | `[T]` CMU lecture 12 |
| Temperature 0.2 diversity | only **~20 unique** outputs from 100 draws | `[T]` CMU lecture 12 |
| Critic reranking (SWE-bench) | **20% → 32%**, ~constant gain per doubling of rollouts | `[T]` CMU lecture 11 |
| Critic reranking cost | **~16×** inference | `[T]` CMU lecture 11 |
| Content filter behaviour | applied **post-hoc**, not trained in | `[T]` CMU lecture 12 |

---

## The one-table summary

| Verifier type | Signal | Cost | Use when |
|---|---|---|---|
| **Rule-based / programmatic** | exact | ~0 | you can write the check — **always prefer this** `[T]` |
| **Outcome reward model (ORM)** | learned, sequence-level | 1 forward per candidate | no rule exists, preference data available |
| **Process reward model (PRM)** | learned, per-step | 1 forward per *step* | long reasoning where the error is mid-trajectory |
| **Generative RM / LM-as-judge** | prompted LLM | 1 generation (expensive) | flexibility, no training data |
| **Unit tests / compiler / type checker** | exact | ~0 | code — the strongest verifier that exists |
| **Critic reranking over rollouts** | learned | ~16× | high-value tasks where 2× accuracy is worth 16× cost |

---

## Formulas

**Rejection sampling / Best-of-N**: draw `y_1..y_n ~ P(·|x)`, return `argmax_i R(x, y_i)`.

**The KL bound** `[T]` — the divergence you *create* by selecting with Best-of-N, i.e. between the
selected-output distribution and the **target (preference) distribution** — *not* the base policy:
```
KL(P_bon || P_target) ≤ log n − (n−1)/n
```
The lecture is explicit: it is *"the KL divergence between your best-of-N outputs distribution and
your target distribution"* `[T]`. Intuition: you pay a divergence price that grows only as `log n`,
so **Best-of-N is a cheap way to move the policy** — no gradient step, no training run. It rises
with `n` because more samples buy more aggressive alignment; the bound rises with it.

**Caveat** `[T]`: this bound is commonly quoted as exact but it is **not tight**. Tighter results
exist (the lecture cites a later paper's equation 25). Empirical KL measurements sit below the
dotted bound line.

**Acceptance probability** in rejection sampling: `D(x) / (C · P(x))`, where `C` is the tightest
upper bound on the ratio. Higher `C` ⇒ lower acceptance ⇒ more wasted samples.

---

## The pathologies of learned reward models — memorise this list

`[T]` CMU lecture 12. This is what separates a real answer from a shallow one.

| Pathology | What happens | Mitigation |
|---|---|---|
| **Non-convex preference space** | gradient descent only works locally; no global optimum | don't expect monotone improvement |
| **Intransitivity** | A>B, B>C, but C>A | never rely on a total order |
| **Order sensitivity** | swapping A/B changes the verdict | randomise order; average both |
| **Instruction-template sensitivity** | the prompt wrapper changes the score | pin the template |
| **Non-determinism** | generative RMs disagree across calls **even at temperature 0** | don't cache or compare across calls |
| **Verbosity bias** | longer outputs score higher **even when wrong** | length penalty in RLHF |
| **Style over correctness** | captures formatting/format, not truth | validate with rules where possible |
| **Refusal policy is baked in** | helpfulness vs harmfulness decided by *who annotated* | know your annotator |
| **Not well-defined over subsequences** | per-substring reward is "bouncy" | use a PRM if you need step-level |
| **No distinction for some inputs** | "rainbow green vs blue" — expect ~0.5 | accept the tie; don't force a winner |
| **Same-family bias** | a Qwen RM upweights Chinese text for a Qwen generator | you may not *want* that |

**Content filters are applied post-hoc, not trained in** `[T]`. The lecture's examples: DeepSeek
cutting off mid-response, and a wrist-tattoo story censored as self-harm. If your reward model
never saw the filter, the filter will surprise you at inference time.

---

## Configuration

```python
# Best-of-N with a verifier — the shape that matters
candidates = llm.generate(prompt, n=32, temperature=0.7)   # 32 fits one batch
scores     = [verifier(prompt, c) for c in candidates]     # ORM / PRM / judge
best       = candidates[argmax(scores)]

# Prefer this whenever you can write the check:
def verify(prompt, candidate) -> bool:
    try:
        tree = ast.parse(candidate)          # compiles?
        return run_unit_tests(tree)          # passes tests?
    except SyntaxError:
        return False
```

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Best-of-N gains plateau | samples are near-duplicates | raise temperature; check unique-output count |
| Same answer regardless of `n` | temperature too low | `τ=0.2` gives ~20 unique of 100 `[T]` |
| Reward model prefers longer wrong answers | verbosity bias | add a length penalty |
| Score flips when you swap candidates | order sensitivity | randomise order |
| Reward model contradicts itself run to run | generative RM non-determinism | use a classifier-head RM instead |
| Verifier scores look random | the RM is out of distribution | check RewardBench performance |
| A/B comparison returns a winner for identical inputs | no real distinction exists | expect ~0.5 — treat as a tie |
| Two candidates tie for 90 tokens then diverge | **check after generation, not during** | max-length mismatch `[T]` |

---

## Gotchas

- **`n = 32` is an engineering constraint, not a theory result.** It is the largest batch that
  fits one pass. If your batch is bigger or smaller, your optimal `n` changes `[T]`.
- **Rule-based verifiers beat model-based verifiers** wherever both are possible `[T]`. Write the
  check before you train the model.
- **The KL bound means Best-of-N is a *policy improvement* method**, not just a reranking trick.
  That is why it substitutes for RL in low-data settings.
- **Multi-model or multi-temperature sampling breaks the single-proposal assumption** behind the
  clean KL intuition — though it still works in practice `[T]`.
- **Rejection sampling was rejected for constrained decoding** (risk of generating forever) but is
  fine here because only *relative* behaviour matters and `n` is fixed `[T]`.
- **Best-of-N at the wrong granularity wastes compute.** Verifying the final answer when the error
  was at step 3 costs 32× and finds nothing. Use a PRM `[D]`.
- **A verifier is only as good as its distribution.** RewardBench v2 exists because RMs that look
  great on held-out preference data collapse on adversarial prompts `[T]`.

---

## When to use what

| Situation | Use |
|---|---|
| You can write the check (code, math, schema) | **rule-based verifier** — free and exact |
| No rule, but you have preference data | trained RM + Best-of-N `n≈32` |
| Long reasoning, error is mid-trajectory | **process** reward model |
| No data, no rule, need it today | LM-as-judge — but randomise order, cap length, and never self-judge |
| Maximal accuracy, cost is no object | critic reranking with ~16× budget |
| You want policy improvement without training | **Best-of-N** — bounded by `log n` |

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt`
