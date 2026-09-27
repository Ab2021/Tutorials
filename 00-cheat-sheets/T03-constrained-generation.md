# Cheat Sheet: Constrained & Structured Generation

> `T03` · **Transcript coverage:** primary · [Case study](../01-case-studies/T03-constrained-generation.md) · [Blueprint](../03-design-blueprints/T03-constrained-generation/HLD.md) · [Interview bank](../02-interview-questions/T03-constrained-generation.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| FUDGE candidate set | restricts to **top-200** tokens + a small future discriminator | `[T]` CMU lecture 6 |
| Contrastive decoding cost | **2× compute** (needs a second forward pass) | `[T]` CMU lecture 6 |
| Token healing trigger heuristics | 4 conditions (see below) | `[T]` CMU lecture 6 |
| Semantic constraint failure | Gemini 1.5 regressed on "don't suggest climbing"; GPT-5 succeeded | `[T]` CMU lecture 6 |
| Prompt-format regression | **98% → 71%** valid JSON from one added sentence | `[T]` LLMOps prompt talk |

---

## The one-table summary

| Technique | Guarantee | Cost | Use when |
|---|---|---|---|
| **Logit masking / FSM** | **hard** — only valid tokens emitted | ~0 (a mask per step) | JSON, SQL, regex, any *syntactic* constraint |
| **Grammar / CFG (pushdown)** | **hard**, supports nesting | moderate | nested JSON, balanced brackets, code |
| **Token healing** | none (heuristic) | 1 extra step | the prompt ends mid-token — a boundary artefact |
| **FUDGE** | none | 1 discriminator forward | you want a soft, learned constraint without a grammar |
| **Contrastive decoding** | none | **2×** | you have a small "amateur" model and want to sharpen the expert |
| **Adversarial decoding** | none | 1–2× | safety — steer away from a harmful direction |
| **Reward-augmented decoding** | none (soft) | 1 reward model | you have a verifier and want token-level guidance |
| **Prompt instruction + validation** | none — retry on failure | ~1 call | quick prototypes only |

**Rule:** if the constraint is *syntactic*, use a grammar. If it is *semantic*, no grammar can help
you — use a verifier or a check-and-retry loop.

---

## The theory-of-computation hierarchy — which constraints are even expressible

`[T]` CMU lecture 6. This is the conceptual core.

| Constraint class | Automaton | Examples | Decodable by masking? |
|---|---|---|---|
| **Regular** | FSA | JSON *without* nesting, regex, enums, fixed schemas | ✅ yes, exactly |
| **Context-free** | Pushdown (CFG) | **JSON with nesting**, balanced parens, most config languages | ✅ yes, with a stack |
| **Context-sensitive / Turing** | Turing machine | "variable used before definition" in C, semantic validity | ❌ **no** |

**Worked example from the lecture** `[T]`: a JSON schema is a *pushdown* automaton, not an FSA — so
an FSA-based masker handles flat schemas but breaks on nesting. And "valid C" is **not**
context-free, so no grammar gives you it. Choose your tool by asking which row you are in.

---

## Formulas

**Logit masking**: for each decoding step, given FSM state `s`:
```
allowed(s) = { t ∈ V : δ(s, t) is defined }
z'_i = z_i           if i ∈ allowed(s)
z'_i = −∞            otherwise
p = softmax(z')
```
Then sample normally. The constraint is applied *before* softmax, so it costs one mask lookup per
step and is exact `[T]`.

**Contrastive decoding** `[T]`:
```
p_final ∝ (1 + β) · log p_expert − β · log p_amateur     (clipped to valid tokens)
```
Amplifies what the expert knows that the amateur does not. Requires **identical tokenizers** — this
is the constraint that kills it in practice.

---

## Token healing — the four trigger heuristics

`[T]` CMU lecture 6. The problem: the prompt ends mid-token, so the model's first generated token
is contaminated by the boundary. Heal by backing up and re-generating the boundary.

Trigger a heal when **any** of these holds:
1. the last token is in a known **bad-character list**
2. the last token is a **single character**
3. the last token has **no leading or trailing whitespace**
4. the last token is an **exact prefix of a likely token**

Enabled in HuggingFace via `token_healing=True`. **Cost:** one extra decoding step `[T]`.

---

## Configuration

```python
# vLLM: guided decoding
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

SamplingParams(
    temperature=0.0,
    guided_decoding=GuidedDecodingParams(json=schema)   # or .regex=, .choice=, .grammar=
)

# Outlines — compile a schema to an FSM once, reuse across requests
import outlines
model = outlines.models.transformers("meta-llama/Llama-3.1-8B-Instruct")
generator = outlines.generate.json(model, MySchema)     # FSM compiled at build time
```
**Compile the grammar once, not per request.** Per-request compilation is the usual latency bug `[D]`.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| JSON is valid but semantically wrong | grammar guarantees *syntax* only | validate semantics separately |
| Repeated keys / missing keys | FSA cannot enforce key uniqueness | move to a schema validator, not a grammar |
| Nesting breaks the output | using an FSA where you need a pushdown | use a CFG-based masker |
| Output is valid but nonsensical | the mask left only one path early on | check the grammar isn't over-constrained; add `anyOf` |
| Non-fluent output with contrastive decoding | tokenizer mismatch, or β too high | verify identical tokenizers first |
| Model produces garbage after a good prefix | **token boundary artefact** | enable token healing |
| Latency spike on first request | grammar compiled per request | compile once at startup |
| "Don't do X" constraints ignored | semantic constraint — not expressible | use a verifier / check-and-retry `[T]` |

---

## Gotchas

- **A grammar cannot express "don't".** The lecture's Gemini 1.5 example: told not to suggest
  climbing, it suggested *bouldering* — a synonym no token-level constraint can catch. Pre-generated
  related-term lists cause non-fluency. **Semantic constraints need a semantic check** `[T]`.
- **FSA constraints cannot bound length** or enforce key uniqueness `[T]`. If you need those, you
  need a validator in the loop.
- **Contrastive decoding ≠ HuggingFace `contrastive search`.** Different methods that share a name.
  HF's is a decoding heuristic; contrastive *decoding* is the expert/amateur construction `[T]`.
- **Contrastive decoding shows repetition artefacts** (the lecture's Obama/Honolulu example) and
  needs matched tokenizers — in practice this rules it out across model families `[T]`.
- **FUDGE gives no guarantee.** It is a soft reweighting by a discriminator over the top-200
  candidates, requires logit access, and can still emit invalid output `[T]`.
- **Adversarial decoding needs a safe *and* an unsafe prompt.** You must construct both; it is not
  a drop-in guardrail `[T]`.
- **Train-time constraints are cheaper than inference-time ones.** If a format is required on every
  call, fine-tune it in — masking costs you on every single request `[T]`.
- **Grammar masking can *hurt* quality** by forcing the model off its preferred path. Measure
  task accuracy with and without, not just format compliance `[D]`.

---

## When to use what

| Need | Use |
|---|---|
| Flat JSON, enums, regex, classification labels | **FSM / logit masking** |
| Nested JSON, balanced structures | **CFG grammar** |
| SQL, code with syntax rules | grammar — but validate semantics separately |
| "Don't mention X", "stay on topic" | **verifier or check-and-retry** — not a grammar |
| Sharpen a large model with a small one | contrastive decoding (if tokenizers match) |
| Token-boundary weirdness | **token healing** |
| Required on every request at scale | **fine-tune it in** |

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_6_Other_Controlled_Generation_Methods.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_5_A_and_Best_First_Search.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Prompt_Management_as_Code_Versioning_Injection_DSPy.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/05-prompting-and-context/06-structured-generation.md` `[R]`
