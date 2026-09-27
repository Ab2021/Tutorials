# Cheat Sheet: Sampling & Decoding Strategies

> `T01` · **Transcript coverage:** primary · [Case study](../01-case-studies/T01-sampling-decoding.md) · [Blueprint](../03-design-blueprints/T01-sampling-decoding/HLD.md) · [Interview bank](../02-interview-questions/T01-sampling-decoding.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| HF `generation_config` defaults | `top_k=50`, `top_p=0.95` | `[T]` CMU lecture 2 |
| Llama recommended | `temperature=0.6`, `top_p=0.9` | `[T]` CMU lecture 2 |
| Temperature 0.2, 100 draws | only **~20 unique** outputs on average | `[T]` CMU lecture 12 |
| Typicality set | the "typical set" is where most of the probability mass sits — not the highest-probability tokens | `[T]` CMU lecture 2 |
| Self-consistency cost | **100 samples = 100× cost** | `[T]` CMU lecture 7 |

---

## The one-table summary

| Method | What it does | Use when | Avoid when |
|---|---|---|---|
| **Ancestral** | sample directly from the model distribution | you want true samples; a base for everything else | you need control over diversity/quality |
| **Greedy** (`temperature→0`) | take argmax | deterministic-ish extraction, classification, structured output | creative text — degenerates, repeats |
| **Temperature** `τ` | rescale logits by `1/τ` before softmax | always — it's the primary knob | as a substitute for truncation |
| **Top-k** | keep k highest-prob tokens | simple, cheap, well-understood | when the distribution is flat — k=50 still admits junk |
| **Top-p / nucleus** | keep the smallest set whose cumulative prob ≥ p | **the default choice** for open-ended generation | very peaked distributions, where it keeps too much |
| **Min-p** | keep tokens with `p ≥ min_p × p_max` | better than top-p at high temperature | very low temperature (degenerates to greedy) |
| **Epsilon sampling** | drop tokens with `p < ε` (absolute floor) | when you want an absolute quality floor | distributions with no clear head |
| **Locally typical** | keep tokens near the *entropy* of the distribution | when you want diversity without incoherence | short/structured outputs |
| **Mirostat** | target a fixed perplexity via feedback control | you want stable perceived quality across prompts | you need predictable latency (it adapts) |
| **eta / ADA** | adaptive truncation from the entropy of the distribution | research; distribution-aware control | production without benchmarking |
| **Repetition / presence / frequency penalties** | subtract from logits of already-seen tokens | loops and verbatim repetition | factual recall where repetition is correct |

---

## Formulas

**Temperature scaling** — applied to logits `z` before softmax:
```
p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
```
`τ → 0` ⇒ argmax (greedy). `τ = 1` ⇒ the model's own distribution. `τ > 1` ⇒ flatter, more random.
Worked: logits `[2.0, 1.0, 0.5]`, `τ=0.5` ⇒ `[4.0, 2.0, 1.0]` ⇒ p ≈ `[0.84, 0.11, 0.04]`.

**Top-p (nucleus)** — sort descending, keep the smallest prefix `S` with `Σ_{i∈S} p_i ≥ p`.
Worked: `p = [0.5, 0.3, 0.15, 0.05]`, `top_p = 0.8` ⇒ keep `{0.5, 0.3}`, renormalise to `[0.625, 0.375]`.

**Min-p** — keep `{i : p_i ≥ min_p × max_j p_j}`, renormalise. Scales with how peaked the
distribution actually is, which is why it holds up better than top-p as temperature rises `[D]`.

**Perplexity** `= exp(cross-entropy)`. Mirostat targets a fixed perplexity rather than a fixed
probability mass — that is the whole idea.

**Entropy** `H = −Σ p_i log p_i`. Locally-typical sampling keeps tokens whose surprisal
`−log p_i` is close to `H`. High-entropy positions admit more tokens; low-entropy positions admit
almost none. This is the principled version of what top-p approximates `[T]`.

---

## Configuration

```python
# The default that works for most open-ended generation
{"temperature": 0.7, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.05}

# Extraction / structured output — do not sample at all
{"temperature": 0.0, "top_p": 1.0, "top_k": 0}

# Creative, diversity wanted
{"temperature": 1.0, "min_p": 0.05, "top_p": 1.0}   # min-p, not top-p, at high temp

# Reasoning / math — sample broadly, then vote (self-consistency)
{"temperature": 0.7, "top_p": 0.95, "n": 100}
```
`top_k=0` means disabled in HuggingFace and vLLM.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Output loops verbatim | no repetition penalty; temperature too low | set `repetition_penalty` 1.05–1.2 |
| Output is incoherent word salad | temperature too high with `top_k=0` | lower τ, or add `min_p` |
| Same answer every time even at high τ | a truncation rule is collapsing the distribution | check `top_p` isn't tiny |
| Different answer every run at `temperature=0` | **GPU float non-associativity**; MoE argmax ties; batch-size-dependent kernels | `[T]` expected, not a bug — see below |
| Structured output breaks intermittently | sampling instead of constrained decoding | `temperature=0` + grammar, see [T03](T03-constrained-generation.md) |
| Model refuses to stop | EOS token suppressed or penalised | check penalties don't include EOS; see EOS absorption `[T]` |

**On temperature-0 non-determinism** `[T]` CMU lecture 2: floating-point addition is not
associative, so reduction order changes results; MoE routing does an argmax over experts and ties
break by layout. **Quantization makes this worse** — and the resulting fingerprints are exploitable
by inference attacks that infer a provider's quantization scheme.

---

## Gotchas

- **Truncation changes the distribution, not just the tail.** Renormalising after truncation
  redistributes mass, so `top_p=0.9` does not mean *the model's distribution minus its tail* — it
  means a *different* distribution `[T]`.
- **`temperature=0` is not determinism.** Never rely on it for reproducibility; you need seeds
  *and* fixed batch shapes, and even then kernels may differ.
- **Top-p and temperature interact.** Raising temperature flattens the distribution, so a fixed
  `top_p` then keeps *more* tokens. Tune them together.
- **The "locally typical" name is literal** — it is defined against the entropy *at that position*,
  not globally.
- **Repetition penalty applies to the prompt too** in some implementations — it can corrupt
  instruction following. Verify.
- **Do not sample when you want the argmax.** Classification, extraction, and structured output
  should be `temperature=0` (ideally with constrained decoding `[T]`).

---

## When not to sample at all

If the task has one right answer — extraction, classification, JSON, code that must compile —
sampling is pure downside. Use greedy or beam search ([T02](T02-search-decoding.md)) and constrain
the output ([T03](T03-constrained-generation.md)). Sampling earns its place only when you want
*diversity* — creative writing, and multi-sample reasoning where you will vote ([T04](T04-test-time-compute.md), [T05](T05-verifiers-best-of-n.md)).

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_3_Common_Sampling_Methods.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_4_Beam_Search_and_Variants.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt`
