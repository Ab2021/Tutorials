# Cheat Sheet: Chain-of-Thought, Self-Correction & Reasoning Models

> `T04` · **Transcript coverage:** primary · [Case study](../01-case-studies/T04-test-time-compute.md) · [Blueprint](../03-design-blueprints/T04-test-time-compute/HLD.md) · [Interview bank](../02-interview-questions/T04-test-time-compute.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Self-consistency cost | **100 samples = 100×** inference cost | `[T]` CMU lecture 7 |
| Adaptive self-consistency prior | Dirichlet/Beta, `α=3` ⇒ posterior `0.5/0.25/0.25` | `[T]` CMU lecture 7 |
| Adaptive confidence threshold | **0.95** | `[T]` CMU lecture 7 |
| DeepSeek R1 scale | **470B** params | `[T]` CMU lecture 9 |
| R1 reasoning gain | AIME **~15% → >70%** | `[T]` CMU lecture 9 |
| R1 thinking length | **~800 tokens → thousands** | `[T]` CMU lecture 9 |
| GRPO clip `ε` | **≈0.1** | `[T]` CMU lecture 9 |
| GRPO temperature | **1.0** — required for on-policy sampling | `[T]` CMU lecture 9 |
| S1 curated examples | **1,000** | `[T]` CMU lecture 9 |
| Stream of Search | **500,000** trajectories (Countdown game) | `[T]` CMU lecture 9 |
| RL plateau | ~**200 steps** flat, then a 250-step cutoff with smaller batch | `[T]` CMU lecture 9 |
| Edit vectors | **512-dim**, capped to force generalisation | `[T]` CMU lecture 8 |
| Self-Debugging | 1 sample, vs self-consistency's **16** | `[T]` CMU lecture 8 |

---

## The one-table summary

| Technique | Mechanism | Works when | **Fails when** |
|---|---|---|---|
| **Zero-shot CoT** | "think step by step" | multi-step arithmetic, symbolic | the model is too small, or the task has no intermediate structure |
| **Few-shot CoT** | worked examples in the prompt | you have good exemplars | exemplars are unrepresentative |
| **Self-consistency** | sample `n`, majority-vote the answer | the task has a **discrete verifiable answer** | open-ended generation — there is nothing to vote on |
| **Adaptive self-consistency** | sample until confidence ≥ 0.95 | you want to cut the 100× cost | distributions are bimodal |
| **Self-Refine** | critique then revise, ≤3 iterations | **style, format, readability** | **deep reasoning errors** — it makes them worse |
| **Self-Debugging** | run the code, use the traceback | code, with an executor | no executor available |
| **Reflection** | verbal self-feedback into a retry | repeated attempts at the same task | the failure was a knowledge gap, not a slip |
| **RL-trained reasoning** (R1, GRPO) | train the model to reason | you have verifiable rewards + compute | no verifier exists |

---

## Formulas

**The latent-variable view** `[T]` — CoT introduces latent variable `Z`:
```
P(Y | X) = Σ_Z P(Y | X, Z) · P(Z | X)
```
Exact marginalisation is intractable: with vocabulary `V` and 100 reasoning tokens you would need
`≈ V^100` terms. **Every CoT method is an approximation to this sum.**

- **Greedy CoT** ≈ mode-seeking: `argmax_Z P(Z|X)` then `argmax_Y P(Y|X,Z)`.
- **Joint argmax `argmax_{Z,Y}`** is **wrong** — the lecture's inches/centimetres example gives
  0.6 vs 0.4, i.e. the jointly-most-likely path picks the wrong answer `[T]`.
- **Self-consistency** ≈ sampling: draw `Z ~ P(Z|X)`, take the majority over `Y`. This is a
  rejection-sampled estimate of the marginal.

**Adaptive self-consistency** `[T]`: maintain a Beta/Dirichlet posterior over answer frequencies.
With prior `α=3` and observed counts, the posterior after one answer each of `{a,b,c}` is
`0.5/0.25/0.25`. Stop when the leader's posterior exceeds **0.95** — check per batch, not per
sample.

**Cosine length-modulated reward** `[T]`:
```
reward = correctness − λ · cos(length_penalty)
```
Wrong **and** short ⇒ larger negative. Right ⇒ encourage *shorter*. This is how reasoning models are
kept from rambling without being truncated mid-thought.

---

## The four cognitive behaviours RL induces

`[T]` CMU lecture 9. These emerge from RLVR — nobody programs them:

1. **Verification** — re-checking its own work
2. **Sub-goal setting** — decomposing before solving
3. **Backtracking** — abandoning a wrong path
4. **Backward chaining** — working from the goal

Their emergence is the main empirical argument that RL teaches *process*, not just answers.

---

## Why self-correction often fails — the negative result

`[T]` CMU lecture 8. The most important caution in this topic:

- **Models cannot reliably identify their own errors.** "Hard-to-do is hard-to-check" — if the
  model could check it, it could probably do it.
- **Intrinsic self-correction degrades reasoning performance.** Asking a model to reconsider a
  correct answer frequently makes it change to a wrong one.
- **Confirmation bias** — the model defends its prior output.
- **What it *does* fix:** grammar, style, formatting, code readability, sentiment, dialogue.
- **What fixes it:** *external* feedback — code execution, a fact-checker, a tool. Self-Debugging
  works precisely because the Python interpreter is an oracle the model is not.
- **Bigger gains on stronger base models**, and GPT-4 was the first to self-correct spontaneously.

**Rule: self-correction without an external oracle is not worth the tokens for reasoning tasks.**

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Model changes a correct answer to a wrong one | intrinsic self-correction | remove the critique step |
| Self-consistency gives no improvement | task has no discrete answer to vote on | confirm answers are comparable |
| Voting is split ~evenly | the model doesn't know — bimodal | more samples won't help; add a verifier ([T05](T05-verifiers-best-of-n.md)) |
| Reasoning loops forever | no max-length control | set `max_tokens` + length reward |
| Output truncated mid-thought | `max_tokens` too tight | watch the **exceed rate** `[T]` |
| RL training plateaus early | over-supervised bootstrapping | check the SFT/RL mix `[T]` |
| Length grows without accuracy | reward not length-aware | add a length penalty |
| Short-CoT model gains nothing from RL | primed with short CoT | needs long-CoT priming first `[T]` |

---

## Gotchas

- **CoT is unfaithful.** The stated reasoning frequently does not reflect the actual computation.
  Never treat the trace as an audit log `[T]`.
- **Longer CoT correlates with better accuracy**, but only up to a point — and only for models
  trained that way. Prompting a non-reasoning model to "think longer" mostly adds latency `[T]`.
- **Over-supervising bootstraps faster and plateaus worse.** Rationalization (SFT on traces) beats
  RL early and loses later `[T]`.
- **Distillation beats RL-from-scratch at small scale** — R1-Distill 32B (Qwen-based) beats 470B V3
  on some measures `[T]`.
- **SFT degrades non-reasoning tasks; RL does not.** Mechanism: GRPO downweights only negative
  sampled sequences, while SFT upweights one sequence and downweights everything else `[T]`.
- **Training on math alone is better at math and worse everywhere else.** RL generalises; narrow SFT
  does not `[T]`.
- **Adaptive parallel search escapes the sequential token-length limit** at the cost of more tokens
  and higher latency — it is a parallelism/latency trade, not free `[T]`.
- **Reasoning models need `temperature=1` for on-policy sampling** during RL. Lower temperatures
  silently turn the algorithm off-policy `[T]`.
- **Exceeding max output length crashes the trajectory.** Monitor the exceed rate as a first-class
  metric `[T]`.

---

## When to use what

| Situation | Use |
|---|---|
| Arithmetic, symbolic, multi-step logic | CoT + self-consistency (`n` 5–20) |
| Cost-sensitive, still wants CoT | **adaptive** self-consistency (stop at 0.95) |
| Code that can be executed | **Self-Debugging** — the interpreter is the oracle |
| Prose style / formatting | Self-Refine, ≤2 iterations |
| Deep reasoning error | **do not self-correct** — add a verifier |
| You have verifiable rewards + compute | RL (GRPO) — train it in |
| No verifier and no training budget | CoT + Best-of-N with a reward model ([T05](T05-verifiers-best-of-n.md)) |

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_8_Self-Refine_and_Self-Correction_Methods.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_9_Reasoning_Models.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
