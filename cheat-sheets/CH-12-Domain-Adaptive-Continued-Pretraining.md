# CH-12 — Domain-Adaptive Continued Pretraining (CPT) Cheat Sheet

**One-line purpose:** move a **base** model's distribution toward your domain by resuming the
original pretraining objective — next-token cross-entropy on **raw, unlabelled text**, with the
loss on **every** token — so the model stops being surprised by your vocabulary and register.
**Use when:** you have a large in-domain corpus and the bottleneck is *fluency, vocabulary, or
register* — how the model writes about the domain.
**Do NOT use when:** the bottleneck is a **fact** you could retrieve (RAG — CH-04 §3), a
**behaviour** you could demonstrate (SFT — CH-13), or you have under ~1M tokens of domain text
and no held-out eval. CPT cannot cite, cannot be updated weekly, and cannot be judged by its
training loss.

> **The one sentence that matters.** CPT teaches *knowledge*; SFT teaches *behaviour*. The
> objective is byte-identical to pretraining — no chat template, no masked prompt, loss on every
> token. If you are reaching for CPT to make the model *do* something, you want CH-13 instead.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **CPT is pretraining, resumed.** Same loss, new distribution. | Nothing about the objective changes. Anyone who says it is "a different kind of training" is wrong. |
| 2 | **The loss is on every token — no masking, no chat template.** | Data is documents, not (prompt, response). `labels = input_ids`; masking is an SFT habit that destroys CPT (CS-12 §4.9.3). |
| 3 | **Start from a BASE model, never an Instruct model.** | CPT on raw documents trains the chat template back out. You pay to undo work you already bought (CS-12 §4.1.1b, §10 item 10). |
| 4 | **τ_cpt = 10⁻³…10⁻¹ tokens/param; 10⁻² is the target.** | 8M tokens for a 1.1B model, 80M for an 8B. Chinchilla's ~20 tok/param is a *pretraining* ratio and does not apply (CS-12 §4.10). |
| 5 | **1 epoch. Not 5.** | A domain corpus is small enough to memorise. Held-out PPL bottoms out after ~1 pass and then *rises* (CS-12 §4.10.3). |
| 6 | **Replay 1–10% general text. It is the #1 forgetting dial.** | Mix at the *example* level, and include it in the loss. The video never mentions it (CS-12 §4.9.3). |
| 7 | **LR is lower than SFT: full FT 1e-5…5e-5, LoRA 1e-4…2e-4.** | Forgetting scales roughly with LR × steps. This is the second dial, after replay. |
| 8 | **`target_modules` = all 7 projections for CPT.** | `q_proj`/`v_proj` only changes *routing*. Facts live in the MLP projections — attention-only LoRA cannot write them (CS-12 §4.3.3, §7.5). |
| 9 | **Dedup matters MORE here than in SFT.** | CPT loss is over every token, so a duplicated document is a **re-weighted objective**, invisible in the loss curve (CS-12 §4.7.2). |
| 10 | **Held-out domain PPL is the metric. Training loss lies three ways.** | Padding, duplicates, and memorisation all depress it. The video's run reported `loss 9.66` ≈ 15,760 PPL and called it success (CS-12 §4.3.5, §12.1). |
| 11 | **The data pipeline is 70–80% of the work; extraction alone is 8 silent failure modes.** | Two-column interleaving, tables, headers, ligatures, hyphenation, hard wraps, scans, zero-width chars. None raise an error (CS-12 §4.5). |
| 12 | **CPT + SFT + RAG is the stack. CPT alone is not the answer.** | CPT has no citations, no freshness, no per-user ACLs. Price RAG first — it is usually cheaper (CH-04 §11, CS-12 §13.1). |
| 13 | **A LoRA checkpoint directory is not a model.** | `AutoModelForCausalLM.from_pretrained(adapter_dir)` silently loads base weights, no adapter. Use `PeftModel`; check `print(type(model))` (CS-12 §6.7, §9.4-S3). |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **CPT objective** | `L = −(1/N) Σ log p(x_t \| x_<t)` | identical to pretraining | No prompt term, no response mask |
| **Volume ratio** | `τ_cpt = epochs × corpus_tokens / params` | — | 8B model, 80M tokens, 1 epoch → 1e-2 (T3) |
| **Wrong way to hit τ** | `small_corpus × more_epochs` | — | 4M × 5 = 20M token-passes on 4M *unique* → memorises |
| **Right way to hit τ** | `bigger_corpus × 1 epoch` | — | 20M × 1 = 20M token-passes on 20M unique → generalises |
| **Chinchilla (does not apply)** | `~20 tokens/param` | Hoffmann 2022 | From-scratch pretraining. CPT is 10⁻³–10⁻¹. |
| **Training FLOPs** | `6 × N × D` | N = params, D = tokens | 8B × 80M = 3.84e18 FLOPs |
| **LoRA training FLOPs** | ≈ `6 N D` too, not `3 N D` | backward still runs through the net | See §7 — the case study's `3PT` halves every estimate |
| **Tokens from words** | `tokens ≈ words × 1.33` | English BPE | 1,024 words ≈ 1,362 tokens |
| **Chars from tokens** | `tokens ≈ chars / 4` | English | 4,000 chars ≈ 1,000 tokens |
| **Replay mix** | `D̂ = (1−ρ)·D_domain + ρ·D_general` | ρ by **token** count | 80M domain + 4.2M general ≈ ρ 0.05 |
| **Effective batch** | `micro × accum × seq` | tokens/step | 1 × 8 × 2048 = 16,384 (the script's default) |
| **Steps per epoch** | `ceil(chunks / (micro × accum))` | — | 20,000 chunks / 32 = 625 steps |
| **Warmup steps** | `total_steps × warmup_ratio` | — | 625 × 0.03 ≈ 19 steps |
| **Perplexity** | `PPL = exp(mean NLL)` over **non-pad tokens only** | — | loss 9.66 → `e^9.66` ≈ 15,760 |
| **Uniform-vocab floor** | `ln(V)` | V = vocab | TinyLlama V=32,000 → `ln(32000)` = **10.37** |
| **Bits per byte** | `BPB = total NLL / (bytes × ln 2)` | tokenizer-invariant | The only way to compare PPL after re-tokenising |
| **Dedup similarity** | `P(candidate) = 1 − (1 − s^r)^b` | s = Jaccard, b bands × r rows | num_perm 128 = 32×4 |
| **Full-FT memory** | `14–16 bytes/param` before activations | fp32 load = 16 | 1.1B → 14.5–19.9 GB (see §7) |

> **Correction:** CS-12 §4.11.3 states `FLOPs ≈ 6·P·T` and then tabulates T3/8B at **1.9e18
> FLOPs, 1.8 A100-hours, $4.50**, i.e. it computes `3·P·T` and divides by an effective
> **300 TFLOP/s** while labelling that "~40% MFU" (40% of an A100-80's 312 dense TFLOP/s is
> ~125, not 300). **Both choices are generous in the same direction:** a LoRA backward still
> propagates gradients through the frozen network, so the forward+backward cost stays near
> `6ND` (only the *weight-gradient* term is saved), and 300 TFLOP/s effective is ~96% of dense
> peak. The handbook's own calculator gives **3.84e18 FLOPs, 9.77 h, $24.42** for the same run
> at 35% MFU (`python code/common/memory.py --model 8B --tokens 80000000 --gpu-type A100-80`),
> or $57 at 15% MFU. The §18 takeaway "under $5 of A100 time for an 8B model" is therefore
> ~5–10× optimistic. CS-12 §4.11.3, §18.13.

---

## 3. Decision Tree

### 3.1 The table that decides it — CPT vs SFT vs RAG vs long-context

| Dimension | **CPT (this file)** | **SFT (CH-13)** | **RAG (CH-04 §3)** | Long-context prompt |
|---|---|---|---|---|
| What changes | weights (distribution) | weights (behaviour) | the prompt | the prompt |
| Data shape | **raw text, unlabelled** | (instruction, response) pairs | a document store + index | a document store |
| Loss applied to | **every token** | response spans only (`-100` on prompt) | nothing | nothing |
| Chat template | **none** | required, train == serve | n/a | n/a |
| Data volume | 1M–1B+ tokens | 500–50k pairs | the docs | the docs |
| Data cost | ~$0 (you own the PDFs) | 2–10 min/pair of SME time | near-zero | near-zero |
| Train cost | GPU-hours (§7) | 1–10% of CPT | $0 | $0 |
| Update cadence | **retrain: hours–days** | retrain: hours | **re-index: minutes** | **edit prompt: seconds** |
| Cites sources / auditable | **no** | no | **yes** | yes |
| Per-user access control | **impossible** | impossible | **trivial** | trivial |
| Teaches new **vocabulary** | **yes — unique strength** | no | tokens in prompt only | no |
| Teaches the **register** | **yes — unique strength** | good (if in the pairs) | **none** | none |
| Teaches a **behaviour/schema** | weakly | **yes** | no | weakly |
| Forgetting risk | **high** | low | none | none |
| Primary metric | held-out domain PPL | task accuracy | recall@k + faithfulness | task pass rate |
| Fails when | corpus is small/dirty; no eval | you have no labels | answer needs synthesis across 50 docs | the corpus exceeds the window |

**The composition rule: these are not alternatives.** The production stack is
**CPT → SFT → RAG**. CPT makes the model fluent, SFT makes it useful, RAG makes it correct and
current. Skipping CPT is right when the base model is already fluent in the domain; skipping
RAG is right only when nothing needs to be verifiable (CS-12 §13.1).

### 3.2 The ordered rule (first match wins)

```text
1. Does the knowledge change weekly, or is it per-user, or must it be access-controlled?
      → RAG. CPT cannot be re-run weekly and cannot enforce ACLs. Full stop.

2. Must the answer CITE its source, or survive an audit?
      → RAG (optionally CPT underneath for voice). CPT alone is disqualified.

3. Is the requirement "the model must write/speak like this field"?
      → CPT. This is the case RAG cannot serve at any price.

4. Do you need FACT recall of stable facts, count under ~10,000?
      → RAG is cheaper. CPT only if #3 also applies.

5. Do you need a task-executor (extract / classify / format) over domain text?
      → SFT (CH-13) — on top of CPT if #3 also applies.

6. Is your corpus under ~1M tokens?
      → Do not run CPT (CS-12 §8.2-3). Between 1M and ~5M you get a style/jargon nudge,
        not knowledge (CS-12 §4.2.2 rule 6, §4.10.3 T1–T2).

7. Do you have a held-out domain split at DOCUMENT level AND a replay corpus?
      → No: STOP. These are the two hardest stop conditions (CS-12 §8.2-1/2).

8. ≥10M tokens, a register/vocabulary gap, and hardware to pay for it?
      → RUN CPT. LoRA first, as a probe (§4.2), then SFT on top, then eval against the base.
```

> **Correction:** the "minimum corpus" threshold is stated three different ways inside CS-12,
> and only one of them can be your planning number. §4.2.2 rule 6 says under **~5M tokens** do
> not do CPT; §8.2 item 3 says under **~1M tokens** do not; §4.10.3 calls T2
> (τ = 10⁻³ → **1.1M tokens for a 1.1B model**) "the minimum viable CPT run"; and
> `code/03_continued_pretraining.py` warns below **1M** and calls 1–10M "a modest style/jargon
> shift, not new knowledge". **The truth is that they are measuring different things:** the
> threshold depends on model size (τ, not raw tokens) and on what you are asking for (register
> vs facts). Plan in **τ_cpt**, not tokens: register shift needs τ ≈ 10⁻⁴–10⁻³, facts need
> τ ≈ 10⁻². CS-12 §4.2.2, §4.10.3, §8.2.

### 3.3 When CPT is genuinely the right call — the three conditions

| Condition | Threshold (rule of thumb) | Why it is load-bearing |
|---|---|---|
| **A large in-domain corpus that is not on the open web** | ≥10M tokens, ≥50M if full FT | Below this the gradient never generalises; the model memorises instead (CS-12 §4.10). |
| **A style / vocabulary / register shift, not a fact gap** | Your 500 most frequent domain terms tokenise into ≥3 pieces each | RAG fixes facts. Nothing but gradient descent on the corpus fixes a distributional property (CS-12 §4.2.2). |
| **Hardware and an eval harness you can pay for** | ≥1 GPU-hour of budget *and* 20% of the project in eval work | The GPU is ~10% of the project cost; the pipeline and eval are the other 90% (CS-12 §11.6). |

---

## 4. Hyperparameter Quick Reference

### 4.1 The knobs that actually matter

| Param | LoRA-CPT default | Full-FT-CPT default | Safe range | Effect of getting it wrong |
|---|---|---|---|---|
| `τ_cpt` (tokens/param) | **1e-2** (T3) | 1e-2 | 1e-3 … 1e-1 | Too high → memorisation, held-out PPL rises. Too low → nothing learned. |
| Replay ratio ρ | **0.05** | 0.05 | 0.01 … 0.10 | 0 → forgetting (MMLU-class drops of 5–25% on a long run). >50% → weak adaptation. |
| `learning_rate` | **2e-4** | **2e-5** (1.1B) / 1e-5 (7–8B) | LoRA 1e-4…3e-4; full 1e-5…5e-5 | 10× too high → loss spikes and settles on a *higher* plateau; model left the basin. |
| `num_train_epochs` | **1** | 1 | 1 … 3 (2 is an admission the corpus is too small) | 5+ overfits and memorises; held-out PPL rises while train loss falls. |
| `lr_scheduler_type` | `cosine` | `cosine` | cosine / linear | Constant LR leaves the model at a high-LR endpoint. |
| `warmup_ratio` | **0.03** | 0.03 | 0.01 … 0.05 (0 if <200 steps) | No warmup → the first steps can wreck a pretrained model. |
| `weight_decay` | **0.01** | 0.01 | 0.0 … 0.1 | **HF default is 0.0**, which is a trap for long runs: weights drift, forgetting accelerates. |
| Batch (tokens/step) | 16k–64k | 16k–64k | 4k … 256k | Too small → noisy loss; too large → diminishing returns at linear wall-clock cost. |
| `max_length` | **p99.5 of your chunks** | same | 256 … 2048 | Copying 512 truncates whole documents; too large wastes compute on padding. |
| LoRA `r` | **32** | — | 16 … 64 (r=8 is a *style* rank) | r too low → cannot absorb new vocabulary. Too high → overfits a small corpus. |
| `lora_alpha` | **2 × r** (64) | — | r … 4r | Moving `r` without moving `alpha` silently changes the effective LR. |
| `use_rslora` | **True** for r>16 | — | — | Without it, r 8→64 multiplies the effective LR by 8. |
| `lora_dropout` | 0.05 | — | 0.0 … 0.1 | — |
| `target_modules` | **all 7 projections** | — | q,v (SFT) → all (CPT) | Attention-only cannot change what a token *means* (CS-12 §4.3.3). |
| Chunk overlap | **0** | 0 | 0 … <5% of `max_length` | Overlap duplicates tokens → the near-duplicate re-weighting of §4.7.2, at every boundary. |
| Precision | `bf16` on Ampere+ | bf16 | — | `fp16` needs loss scaling and can NaN; on a T4 fp16 is the only option. |
| Quantization | 4-bit NF4 (QLoRA) | — | 4 or 8 | 4-bit on a small model can cost quality; 8-bit is a middle path. |
| `gradient_checkpointing` | `True` | True | — | ~25–30% slower, ~60% less activation memory. Effectively mandatory. |
| `max_grad_norm` | 1.0 | 1.0 | 0.5 … 1.0 | Clipping is a band-aid over an LR problem. |
| Val split | **5% of documents** | 5% | 3 … 10% | Chunk-level splits leak; eval becomes memorisation. |
| `save_steps` | ≤ ⅕ of an epoch | same | — | `save_steps=500` on a 20-step run writes **no checkpoint at all**. |
| `seed` | 42, fixed | 42 | — | Otherwise you cannot bisect a regression. |

### 4.2 The data-volume tiers — plan in τ, not in tokens

| Tier | τ_cpt | Tokens, 1.1B | Tokens, 8B | What you get | Verdict |
|---|---|---|---|---|---|
| **T0 — Demo** | 1e-6 | 1.5K | 10K | Nothing. Memorisation of the exact chunks. | The video's notebook. Do not ship. |
| **T1 — Register nudge** | 1e-4 | 110K | 800K | Sounds domain-native; vocabulary unchanged. Visible in PPL, not in samples. | A style LoRA on a competent base. |
| **T2 — Style + vocabulary** | 1e-3 | 1.1M | 8M | Terms tokenise into stable concepts; register clearly shifts. Facts still unreliable. | Minimum viable CPT. |
| **T3 — Vocabulary + facts** | **1e-2** | 11M | **80M** | Reliable use *and definition* of domain facts. | **The target for most projects.** |
| **T4 — Deep domain** | 1e-1 | 110M | 800M | Domain-expert behaviour. | Only with ≥5% replay and a real eval harness. |
| **T5 — Continued pretraining proper** | 1–10 | 1.1B–11B | 8B–80B | A domain base model. Multi-GPU, multi-day. | BloombergGPT-class. Not a fine-tune. |

**The planning rule:** if your corpus gives you ≥T3 in a **single pass**, train one pass and stop.
One clean pass at T3 beats five passes at T2 with a memorised model at the end. Compute the tier
from **unique** tokens after dedup — a 40%+ dedup removal rate means your token count is fiction
(CS-12 §15.4).

### 4.3 Corpus-quality rules of thumb

| Signal | Threshold | What it means | Assumption behind it |
|---|---|---|---|
| Extraction yield | <200 chars/page → OCR | Scanned page, silently dropped | Text-native PDFs give 1,500–4,000 chars/page |
| Alphabetic ratio | <0.5 letters | CID/ligature garbage | Ordinary prose is >0.75 |
| Median line length | <90 chars, low variance | Hard-wrapped prose; un-wrap before paragraph splitting | Typeset lines are 60–90 chars |
| Dedup removal | >40% → corpus is mostly repetition | τ_cpt is inflated; re-scope, don't just retrain | CS-12 §15.4; filings removed 55% |
| Distinct 13-grams | <0.5 | More than half the corpus is near-duplicate | Count before you train (CS-12 §2.3) |
| Domain term tokenisation | median ≥3 tokens/term | Tokenizer inflation; consider a bigger-vocab base | Qwen2.5 151k vs Llama 32k dropped 3.1 → 1.9 tok/word |

---

## 5. Copy-Paste Code Snippets

### 5.1 The pipeline is the job — what `code/03_continued_pretraining.py` does

PDFs → extract → clean → dedup → chunk → train. Read the script; its cleaning and chunking
functions are the reference implementation for this section.

```python
# code/03_continued_pretraining.py — the five cleaning steps, in the order that works.
import unicodedata, re

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # 1. ligatures: ﬁ -> fi
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # 2. de-hyphenate a line break
    text = re.sub(r"([^\n.!?:;])\n(?=[a-z])", r"\1 ", text)   # 3. join mid-sentence lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)              # 4. keep paragraph structure
    return "".join(c for c in text                        # 5. strip control chars
                   if c in "\n\t" or unicodedata.category(c)[0] != "C").strip()
```

Four things the script does **right** and the video's notebook does not:

| # | Correct choice | Why it matters |
|---|---|---|
| 1 | `padding=False` + `DataCollatorForLanguageModeling(tok, mlm=False)` | Pads per batch and pads **labels with `-100`**. The notebook's `padding="max_length"` + `labels = input_ids.copy()` trains on pads — ~85% of the loss on a ~75-token chunk at `max_length=512` (CS-12 §4.3.4). |
| 2 | `labels` are **not** pre-shifted | HF's `Trainer` drops the last logit and the first label internally. Pre-shifting gives a doubly-shifted objective that still looks like it converges (CS-12 §4.3.2). |
| 3 | SHA-256 exact + MinHash near-duplicate dedup, before chunking | Chunking after dedup avoids boundary near-duplicates (CS-12 §5.1). |
| 4 | A volume verdict printed before training | `<1M` → "use RAG or SFT-only"; `1–10M` → "style/jargon shift"; `≥10M` → "meaningful adaptation". |

Two things it does **not** do, which you must add (§5.3, §5.4): it prints a replay line but
mixes **no** replay data, and it has **no held-out split and no eval**.

### 5.2 Tokenising for CPT — no masking, pads to `-100`, EOS appended

```python
MAX_TOKENS = 1024          # p99.5 of YOUR chunk lengths, measured, not copied
tok.pad_token = tok.eos_token          # only if the base model ships no pad token

def tokenize_fn(examples):
    out = tok(examples["text"], truncation=True, max_length=MAX_TOKENS,
              padding=False)                    # dynamic padding lives in the collator
    # Append EOS so the model learns where a document ends (CS-12 §4.8.5).
    out["input_ids"]      = [ids + [tok.eos_token_id] for ids in out["input_ids"]]
    out["attention_mask"] = [m + [1]                  for m in out["attention_mask"]]
    # labels = input_ids: HF shifts internally. Do NOT pre-shift (CS-12 §4.3.2).
    out["labels"] = [list(ids) for ids in out["input_ids"]]
    return out

from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
```

> **Caution with `pad_token = eos_token`.** The collator masks every position equal to
> `pad_token_id` to `-100`. When pad **is** EOS they are the same id, so a real `</s>` inside
> your text is masked too, and any "is this a pad?" test by id becomes ambiguous. If you append
> EOS deliberately, prefer a dedicated `<|pad|>` token or mask by `attention_mask` instead
> (CS-12 §10 item 9). The script sidesteps this by never appending EOS explicitly.

**The assertion that catches the pad-as-target class of bug** — run it on the tokenised dataset:

```python
import numpy as np
_lab = np.array(train_tok["labels"])
print(f"supervised fraction: {(_lab != -100).mean():.3f}")   # expect ~1.0 for CPT
```

For CPT the supervised fraction should be **~100%** — every real token is a target. A low
number means padding, or a masked-replay mistake from §5.3.

### 5.3 Replay — the mitigation the script only prints

```python
from datasets import interleave_datasets

# Replay text = general pretraining-ish text: fineweb, dolma, the-pile, or a
# fineweb-edu slice. 1-10% of TOKENS (not examples). (CS-12 §4.9.3)
train_tok = interleave_datasets(
    [domain_tok, replay_tok],
    probabilities=[1.0 - REPLAY_RATIO, REPLAY_RATIO],   # 0.95 / 0.05
    seed=42,
    stopping_strategy="first_exhausted",
)

# The trap: teams arriving from SFT mask "non-target" tokens to -100, which
# deletes the entire mitigation while keeping its cost. Assert on it:
assert (np.array(train_tok["labels"]) != -100).mean() > 0.98, "replay is being masked out"
assert len(replay_tok) > 0, "replay dataset loaded zero rows (silent failure S7)"
```

Interleave at the **example** level, not the batch level: batch-level mixing makes the effective
ratio fluctuate and the loss curve uninterpretable (CS-12 §4.9.3).

### 5.4 The eval you must build before you train

```python
import hashlib, math, torch

# 1. Split by DOCUMENT, and assert no overlap. Chunk-level splits leak. (CS-12 §12.3)
def doc_hash(t): return hashlib.sha1(t.strip().encode()).hexdigest()
assert not ({doc_hash(d) for d in train_docs} & {doc_hash(d) for d in val_docs}), "DOC LEAK"

# 2. Perplexity over NON-PAD tokens only, weighted by the true token count.
@torch.no_grad()
def perplexity(model, tok, texts, max_len=1024, batch=4):
    model.eval(); nll, n_tok = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
        labels = ids.clone(); labels[am == 0] = -100        # never score padding
        loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
        n = (labels[:, 1:] != -100).sum().item()            # HF shifts by one
        nll += loss.item() * n; n_tok += n
    return math.exp(nll / max(n_tok, 1))
```

Report **four** numbers for every run, base vs adapted, in one table: held-out **domain** PPL
(down ≥10%), held-out **general** PPL (up ≤15%), the **task** metric, and a **catastrophe
probe** (§8, CS-12 §12.2). A PPL of 24 is meaningless; "38.1 → 21.4 versus base" is a result.

---

## 6. CLI Commands

Verified against `code/03_continued_pretraining.py`'s argparse. **Exactly one** source flag is
required (`--pdf-dir`, `--text-file`, or `--jsonl-text-field` — they are mutually exclusive).

```bash
# ── This handbook's CPT script ───────────────────────────────────────────────
# Always dry-run first: it prints the plan, the volume verdict, and exits.
python code/03_continued_pretraining.py --dry-run --pdf-dir ./pdfs

# The real run (all flags below are the script's own defaults except --pdf-dir):
python code/03_continued_pretraining.py \
  --pdf-dir ./pdfs \
  --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --out-dir ./out/dapt \
  --chunk-words 1024 --seq-len 2048 \
  --epochs 1 --lr 2e-5 --batch 1 --grad-accum 8 \
  --replay-frac 0.10

# Already-extracted text, or a JSONL with a "text" field:
python code/03_continued_pretraining.py --text-file corpus.txt --epochs 1
python code/03_continued_pretraining.py --jsonl-text-field corpus.jsonl --epochs 1

# ── VRAM / cost, BEFORE you rent anything ───────────────────────────────────
python code/common/memory.py --table
python code/common/memory.py --model 1B --method full --seq-len 2048 --batch 1 --grad-accum 8
python code/common/memory.py --model 8B --method qlora --tokens 80000000 --gpu-type A100-80

# ── Downstream: the SFT stage that makes the CPT checkpoint useful (CH-13) ──
python code/01_sft_lora.py --model ./out/dapt --data code/data/sample_sft.jsonl
python code/09_merge_and_export.py --adapter out/sft-lora --out out/merged
```

**Flag reference (script defaults in bold):**

| Flag | Default | Note |
|---|---|---|
| `--pdf-dir` / `--text-file` / `--jsonl-text-field` | — | Required, mutually exclusive. JSONL rows must have a `text` key. |
| `--model` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | A **base** model. |
| `--out-dir` | `./out/dapt` | Checkpoint + tokenizer. |
| `--chunk-words` | `1024` | Words, not tokens. ≈1,362 tokens. Overlap is hardcoded to 64 (~6%). |
| `--seq-len` | `2048` | Truncation length for tokenisation. |
| `--epochs` | `1.0` | Float. 1 is correct; 5 is the mistake. |
| `--lr` | `2e-5` | Full-FT LR. LoRA would be 1e-4…2e-4. |
| `--batch` / `--grad-accum` | `1` / `8` | Effective 16,384 tokens/step — inside the 16k–64k target. |
| `--replay-frac` | `0.10` | **Prints only. No replay data is mixed — see §11.** |
| `--dry-run` | off | Plan + verdict, no training. |

**The dry-run output you should be able to read** (this is the real shape):

```text
  raw corpus         3,997 chars  (~999 tokens)
  after dedup        1 docs (removed 0 near-duplicates)
  chunks             7  (~150 words each)
  estimated tokens   1,293
  ⚠  Under ~1M tokens, DAPT will change your model very little. Strongly consider RAG or SFT-only instead.
  replay             mixing 10% general text (reduces catastrophic forgetting; set --replay-frac 0 to disable)
```

---

## 7. VRAM / Cost Calculator

From `code/common/memory.py --table`. Single GPU, gradient checkpointing **on**, AdamW, bf16
weights, fp32 optimiser states for full FT. GB:

| Model | Full FT | LoRA r16 | QLoRA r16 | Infer bf16 | Infer 4-bit |
|---|---|---|---|---|---|
| 0.5B | 6.6 | 1.1 | 0.4 | 2.0 | 1.2 |
| 1B | 14.5 | 2.4 | 0.9 | 3.1 | 1.5 |
| 1.5B | 19.7 | 3.2 | 1.1 | 3.9 | 1.7 |
| 3B | 39.4 | 6.4 | 2.2 | 7.0 | 2.5 |
| 7B | 91.6 | 14.7 | 4.9 | 16.0 | 4.8 |
| 8B | 104.7 | 16.7 | 5.6 | 16.4 | 4.9 |
| 13B | 170.0 | 27.1 | 9.0 | 28.3 | 7.8 |
| 32B | 417.9 | 66.2 | 21.5 | 61.6 | 16.2 |
| 70B | 913.8 | 144.5 | 46.8 | 132.6 | 33.9 |

**Add 10–20% for allocator/framework overhead, and leave headroom.**

**What the script's own defaults cost** (`--model TinyLlama…` maps to the 1B preset, full FT,
seq 2048, batch 1 × accum 8 → 16,384 tokens/step):

| Component | GB |
|---|---|
| weights (4 B/param: bf16 + fp32 master) | 4.10 |
| gradients (2 B/param) | 2.05 |
| AdamW state (8 B/param) | 8.20 |
| activations (checkpointed) | 0.15 |
| **Total, 1 GPU** | **14.49** |

Fits 24 GB — and **only** 24 GB. The script is full FT by design; for CPT you almost always want
the LoRA or QLoRA rows instead (§9.2).

**Cost, from the same calculator** (`--gpu-type A100-80`, 35% MFU, $2.50/h):

| Run | Tokens | τ (8B) | FLOPs | Hours | $ @35% MFU | $ @15% MFU |
|---|---|---|---|---|---|---|
| T2, 8B | 8M | 1e-3 | 3.84e17 | 0.98 | $2.44 | $5.70 |
| **T3, 8B** | **80M** | **1e-2** | **3.84e18** | **9.77** | **$24.42** | **$56.98** |
| T4, 8B | 800M | 1e-1 | 3.84e19 | 97.68 | $244.20 | $569.80 |
| T3, 1.1B (A100-40) | 11M | 1e-2 | 7.26e16 | 0.18 | $0.33 | $0.78 |

> **Why you will see different numbers elsewhere — and both are right.** This table prices
> `6 × N × D` at a stated MFU. CS-12 §11.3 prices the same runs at `3 × N × D` and an effective
> 300 TFLOP/s, and lands on **1.8 h / $4.50** for the T3 8B run — ~5× below the line above.
> The gap is entirely assumptions: (1) **the FLOP count.** A LoRA backward still propagates
> gradients through the frozen network; only the *weight-gradient* term is saved, so the cost
> stays near `6ND`, not `3ND`. (2) **Bytes per parameter.** CS-12 §1.3 uses
> `4 + 4 + 8 = 16 B/param` (fp32 load) → 1.1B = 17.6 GB; this table uses
> `4 + 2 + 8 = 14 B/param` → 14.5 GB. (3) **GiB vs GB** — this table is binary (1024³); vendor
> slides are decimal (1000³), another +7.4%. **How to use it:** trust a real measurement over
> any table, and when two numbers disagree by ~20–30% check bytes-per-param and GiB-vs-GB
> before assuming one is wrong. Check the FLOP exponent before assuming a 5× gap is an error.

### 7.1 Where the money actually goes (7B domain project, engineer-days)

| Workstream | Share |
|---|---|
| Corpus acquisition + licensing review | 10% |
| **PDF extraction + the eight failure modes** | **25%** |
| **Cleaning + dedup** | **15%** |
| Chunking + token analysis | 5% |
| **Eval harness (held-out PPL + task eval + probe)** | **20%** |
| Training runs, including the failed ones | 10% |
| Deployment, versioning, rollback | 15% |

**The GPU bill is ~10% of the project.** Budget 10–20× the GPU cost for engineering, and expect
the pipeline to be the part that misses the deadline (CS-12 §11.6).

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| `CUDA out of memory` before step 1 | Full FT: weights + grads + AdamW > VRAM (§7) | QLoRA + LoRA; or 8-bit AdamW; or a smaller model. This is arithmetic, not bad luck. |
| OOM at step 3–10, not step 1 | Activation memory / sequence length | Gradient checkpointing + smaller micro-batch + more accumulation; halve `max_length`. |
| Loss flat at ~10.4 (≈ `ln(32000)`) | Nothing is training | `sum(p.requires_grad for p in model.parameters())`; check LR; confirm the adapter is in the forward path. |
| **Loss ~9.66, "falling nicely"** | Pad-as-target (the video's bug): ~85% of the loss is padding | `padding=False` + `DataCollatorForLanguageModeling`; mask labels `-100`. |
| Loss falls to ~0.1 within a few hundred steps | Memorisation — corpus too small or duplicated | More unique data; dedup; **1 epoch**. |
| Loss spikes to NaN | LR too high, fp16 overflow, or an empty-string batch | LR ÷5; bf16; clip at 1.0; drop zero-length chunks. |
| Loss falls, then **rises** | Overfit past the knee | Stop at the minimum; 1 epoch; more unique tokens. |
| Loss spikes then settles on a **higher** plateau | LR too high — the model left the basin | Roll back to the pre-spike checkpoint; LR ÷2–5. |
| Loss declines smoothly, never plateaus | Run too short or LR too low | Longer run, or LR ×2 — then check that domain PPL actually moved. |
| Held-out PPL barely moves after a full run | LR too low, corpus too similar to pretraining, or τ_cpt too small | Compute τ_cpt; raise LR; or accept that CPT is not needed here. |
| Domain PPL great, model useless on the task | You optimised fluency, not the task | That is expected — add SFT (CH-13). CPT is stage 1 of 3. |
| Domain PPL improves, arithmetic and JSON break | Catastrophic forgetting | Raise replay ρ to 10%; halve epochs; consider merging the adapter at reduced scale. |
| Model repeats the prompt verbatim | Adapter not loaded, rank too low, or too few unique tokens | `print(type(model))` must say `PeftModel`; wider `target_modules`; more data. |
| Model emits HTML / Markdown / LaTeX boilerplate | The corpus contains markup | Strip markup in cleaning — the model learns to emit `<div class="…">` mid-sentence. |
| Model prepends a fluent, confident, *wrong* safety notice | Template-level duplication: a repeated boilerplate section was learned as high-probability continuation | Template/n-gram dedup, down-sample the boilerplate class (CS-12 §15.3). |
| Output is fluent but wrong about the domain | It learned the distribution, not the facts | That is what RAG is for. CPT cannot do this (CS-12 §9.3-L3). |
| Inference output is bad, training looked fine | **Adapter never loaded** | `PeftModel.from_pretrained(base, adapter_dir)`; `print(type(model))`. |
| Inference bad and `type(model)` *is* `PeftModel` | Tokenizer reloaded from the base repo, not the adapter dir | `AutoTokenizer.from_pretrained(adapter_dir)`. |
| Non-English output has garbage characters | Tokenizer lacks the script; NFKC stripped something | Measure `len(tok(text)) / len(text.split())`; if >3, change base model. |
| `KeyError: 'text'` / empty dataset | Column name mismatch between chunker and dataset | `print(ds[0])`; assert `len(ds) > 0` before training. |
| All labels masked / loss never moves | `if t == pad_token_id: -100` with pad == eos masks real EOS too | Mask by `attention_mask`, or add a dedicated pad token (§5.2). |
| Training 10× slower than expected | `padding="max_length"`, no packing, checkpointing without `use_reentrant=False` | Dynamic padding or packing; `gradient_checkpointing_kwargs={"use_reentrant": False}`. |
| GPU utilisation 20% | DataLoader starvation (PDF parsing at train time) | Pre-tokenise to disk; `dataloader_num_workers >= 4`. |
| `save_steps` never fires | `save_steps` > total steps | `save_strategy="epoch"`, or total ÷ 5. |
| Only part of the corpus is used | Truncation at `max_length`, or a dropped final partial batch | Histogram of lengths; `max_length = ceil(p99.5)`; check `drop_last`. |
| Different results on rerun, same config | No seed, or nondeterministic kernels | Fix `seed`/`data_seed`; accept small attention nondeterminism. |
| Dedup removed 0% on a scraped corpus | It did not run (bad threshold, wrong field, empty shingles) | Log `(before, after)` every time; 0% removal on crawled data is a bug, not a clean corpus. |

---

## 9. Comparison Matrix

### 9.1 CPT vs SFT — the spine

| | Pretraining | **CPT — this file** | SFT (CH-13) |
|---|---|---|---|
| Objective | next-token CE | **next-token CE — identical** | next-token CE |
| Data distribution | web, 10–15T tokens | **your documents, 1M–1B tokens** | (prompt, response), 500–50k pairs |
| Supervision mask | every token | **every token — identical** | response spans only |
| Label construction | shift by one | `labels = input_ids`, HF shifts internally | `-100` on prompt tokens |
| Corpus structure | documents | **documents (chunked)** | conversations |
| Data supply | finite but huge | **unbounded — scan more pages** | hard-limited by SME time |
| Gradient cost per new fact | ~1 token | **~1 token** | ~10–100 paraphrases |
| What it changes | — | **vocabulary, register, collocations, facts** | format, tone, task compliance, refusals |
| Forgetting risk | — | **high (four dials, §4.1)** | low |
| Primary metric | — | **held-out domain PPL** | task accuracy |

### 9.2 LoRA vs full FT for CPT

| Dimension | LoRA | Full FT |
|---|---|---|
| Trainable params (1.1B, all 7 projections, r=32) | ~9M (0.8%) | 1.1B (100%) |
| 1B training VRAM | 2.4 GB (r16, §7 table) | **14.5–19.9 GB** |
| 7B training VRAM | 14.7 GB (r16) | 91.6 GB |
| Forgetting | **less** — the frozen base structurally protects general capability | more — every weight moves |
| Domain-vocabulary ceiling | weaker (the frozen embedding cannot move) | **stronger** |
| Best achievable domain PPL | good | **best** |
| Catastrophe probe | usually within 1–2 points of base | often drops 5–15 points |
| Rollback | delete the adapter; base untouched | restore a full checkpoint |
| **Recommended for** | **first run, always; ≥3B models; anything <500M tokens** | ≤3B with ≥500M tokens, after a LoRA probe shows real ΔPPL |

### 9.3 Deduplication methods

| Method | Detects | Cost | Threshold | Use for |
|---|---|---|---|---|
| Exact hash (SHA-256) | byte-identical docs | O(N) | — | Always, first pass |
| **MinHash + LSH** | near-duplicates (Jaccard) | O(N) with LSH indexing | **0.85**, 5-word shingles | **The default** |
| Suffix array | repeated n-grams across docs | O(N log N), memory-heavy | 50-token match | Heavy boilerplate |
| SimHash | near-duplicates (Hamming) | O(N), cheaper | Hamming ≤ 3 | Very large corpora |
| Embedding / semantic | paraphrase-level | O(N) forward passes | cosine ≥ 0.95 | Small corpora where paraphrase matters |
| Template / n-gram profile | same-skeleton documents | O(N) | top-k overlap | Filings, manuals, generated reports |

### 9.4 Chunking strategies

| Strategy | Boundary quality | When to use | Fails when |
|---|---|---|---|
| Fixed token window | **worst** — cuts mid-sentence, mid-table | baseline only | Two-column text; tables |
| Fixed window + overlap | better | RAG, not CPT | Duplicates content → re-weighted objective |
| Recursive (paragraph → sentence) | good | **the practical default** | Tables and lists still collapse |
| Structure-aware (headings) | **best** when structure exists | Markdown/HTML/LaTeX, manuals, standards | PDFs with no detectable headings |
| Document-level (no chunking) | perfect | documents shorter than `max_length` | Anything longer |

### 9.5 Frameworks

| Framework | Interface | Best for | Watch out |
|---|---|---|---|
| This handbook's `code/03_continued_pretraining.py` | CLI | Reference pipeline: clean, dedup, chunk, full FT | No replay implementation, no held-out split |
| HF `Trainer` + `peft` | Python | Full control | You write the replay interleave and the masking |
| Axolotl / LLaMA-Factory | YAML | Reproducible runs, 100+ models | Config keys churn; see CH-15 / CS-15 |
| Unsloth | Python | Single GPU, 2–4× faster | Patched kernels are per model family |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| CPT LR (full FT) | **1e-5 … 5e-5** | One to two orders below pretraining's peak |
| CPT LR (LoRA) | **1e-4 … 2e-4** | 10× the full-FT LR — same split as SFT |
| Epochs | **1** | 3 is the ceiling; 5 is the video's mistake |
| τ_cpt target | **1e-2** | Tokens per parameter; 8M for 1.1B, 80M for 8B |
| τ_cpt viable band | **1e-3 … 1e-1** | Below → nothing; above → memorisation |
| Chinchilla (does not apply) | **~20 tok/param** | From-scratch pretraining only |
| Replay ratio ρ | **0.01 … 0.10** | 0.05 is the default answer; by token count |
| Minimum corpus | **~1M tokens** | Below this, use RAG or SFT-only |
| Meaningful corpus | **≥10M tokens** | Where the script's own verdict flips to ✅ |
| Dedup threshold | **0.85**, 5-word shingles | 0.80 removed 31% and *hurt* a pharma run (CS-12 §15.1) |
| Dedup removal alarm | **>40%** | Your token count is mostly repetition |
| LoRA `r` for CPT | **32** (16–64) | r=8 is a style rank |
| `lora_alpha` | **2 × r** | 64 when r=32 |
| `target_modules` | **all 7** | q,k,v,o,gate,up,down |
| Warmup | **3%** of steps | 0 below 200 total steps |
| Weight decay | **0.01** | HF default of 0.0 is a trap |
| Effective batch | **16k–64k tokens/step** | The script's default gives 16,384 |
| Chunk overlap | **0** | <5% of `max_length` if you must |
| `max_length` | **p99.5 of chunks** | Measure; do not copy 512 |
| Train FLOPs | **6ND** | memory.py's rule; the case study's `3ND` is optimistic |
| A100-80 TFLOP/s | **312** dense | @35% MFU = 109 effective |
| Realistic MFU | **0.35** (0.15 naive) | The single biggest cost lever |
| Tokens/word | **≈1.33** | English |
| Chars/token | **≈4** | English |
| Uniform-PPL floor | **ln(V)** | 32,000-vocab → 10.37 |
| Video's reported loss | **9.66** | ≈ 15,760 PPL. Not a result. |
| Pad-as-target fraction | **~85%** | Chunks ~75 tok at `max_length` 512 |
| Chars/page, text-native PDF | **1,500–4,000** | 0 = scan; 200–800 = chart-heavy |
| Header/footer frequency rule | **>60%** of pages | A line on most pages is furniture |
| Catastrophe-probe budget | **≤2 points** of 40 | More than that = something was forgotten |
| Domain PPL target | **down ≥10%** | Versus the base model, same protocol |
| General PPL budget | **up ≤15%** | Above that, you over-forgot |

---

## 11. Common Errors And Their Exact Messages

| Error message / symptom | Meaning | Fix |
|---|---|---|
| `CUDA out of memory. Tried to allocate X GiB` | Full FT: 14–16 B/param before activations | QLoRA + LoRA; gradient checkpointing; shorter `max_length`; smaller model |
| `torch.cuda.OutOfMemoryError` at *validation* | Eval batch too large | Lower `per_device_eval_batch_size`; eval a fixed 200–1,000-chunk subsample |
| `ValueError: num_samples should be a positive integer value, but got num_samples=0` | Dataset empty after filtering | Your cleaning/dedup removed everything; print counts at every stage |
| `KeyError: 'text'` | The `map` function reads a column your dataset does not have | The HF contract is one column named `text`; align the dict keys |
| `The following columns are unused: [...]` | Trainer dropped your text column | Use a collator, or `remove_unused_columns=False` |
| `element 0 of tensors does not require grad` | Base frozen *and* adapter not attached | Confirm `get_peft_model` ran; print trainable params |
| `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | Optimiser has no trainable params | `sum(p.numel() for p in model.parameters() if p.requires_grad) > 0` |
| `Token indices sequence length is longer than the specified maximum` | A chunk exceeds `max_length` | Expected warning — but confirm you are not silently truncating most of the corpus |
| `UserWarning: pad_token_id is not set` | Base model ships no pad token | `tok.pad_token = tok.eos_token` — then read the §5.2 caution |
| `ImportError: cannot import name 'SFTConfig'` | TRL too old/new | `pip install -U trl`; `SFTConfig` replaced `TrainingArguments` in TRL 0.12 |
| `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'` | Renamed in transformers ≥4.46 | Use `eval_strategy` (or pin the older `transformers`) |
| `OSError: … does not appear to have a file named config.json` | You pointed `from_pretrained` at an **adapter** directory | Load the base, then `PeftModel.from_pretrained(base, adapter_dir)` |
| Loss ≈ 0 on the first log line | Everything masked, or the model is not the one you think | Print the supervised fraction; print `type(model)` |
| Training completes but loss never appears in the log | `logging_steps` > total steps | `logging_steps ≤ total_steps / 10` |
| Run "succeeds" but no checkpoint exists | `save_steps` > total steps | `save_strategy="epoch"` |

### 11.1 Silent failures — no error message, and this is where CPT dies

| # | Failure | Looks like | Detector |
|---|---|---|---|
| S1 | **Pad-as-target** | Loss falls smoothly | Supervised fraction; loss recomputed on non-pad tokens |
| S2 | **Replay that is not replay** | Config says `replay_frac=0.05` | `assert len(replay_tok) > 0` — `interleave_datasets` silently gives 100% domain |
| S3 | **Replay masked to `-100`** | Replay is "on" | `assert (labels != -100).mean() > 0.98` |
| S4 | **Adapter never loaded at inference** | Output is bad | `print(type(model))` must be `PeftModel` |
| S5 | **Dedup that never ran** | Pipeline "is clean" | Log `(before, after)`; 0% removal on scraped data is a bug |
| S6 | **Truncation mid-document** | Training is fine | Length histogram vs `max_length`; the model learned introductions |
| S7 | **Chunk-level train/val split** | Held-out PPL looks great | Split by document hash; assert zero overlap |
| S8 | **Eval on the training corpus** | PPL 3.2, "amazing" | Check for a near-duplicate of each val doc in train |
| S9 | **Loss improving on a frozen model** | Loss drops | Trainable-param count; weight delta after one step |
| S10 | **Different tokenizer at eval** | Weird, degraded output | Save the tokenizer **with** the adapter and reload it from there |
| S11 | **OCR never triggered** | Corpus is "done" | Log chars/page and the OCR fraction as first-class metrics |
| S12 | **The "it works!" demo** | Output is fluent-looking | Read the generation against the prompt. Literally compare strings. |

> **Beyond the video:** S2 is the one that will bite you if you use this handbook's own script.
> `code/03_continued_pretraining.py` accepts `--replay-frac 0.10` and prints
> `replay  mixing 10% general text (reduces catastrophic forgetting…)`, but it **loads no
> general-domain data and mixes nothing** — the flag is a print statement. Grep the file: the
> only use of `a.replay_frac` is inside an `if a.replay_frac > 0:` print. Follow §5.3 to wire it
> up before you trust the number, and note the default is 0.10 where CS-12 recommends 0.05
> (CS-12 §4.9.3, §7.1).

---

## 12. Copy-Paste Starter Config

### 12.1 Run one — the script, dry-run first

```bash
# 1. Validate the corpus and the plan. Costs nothing. Always do this first.
python code/03_continued_pretraining.py --dry-run --pdf-dir ./pdfs

# Read the output: chars/page warning? volume verdict? Does the VRAM plan fit your GPU?
# 2. Only then, the run. These are the script's defaults except --pdf-dir / --out-dir.
python code/03_continued_pretraining.py --pdf-dir ./pdfs --out-dir ./out/dapt \
  --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --chunk-words 1024 --seq-len 2048 --epochs 1 --lr 2e-5 \
  --batch 1 --grad-accum 8 --replay-frac 0.05
```

### 12.2 The config you should graduate to (LoRA-CPT, any 1B–8B base)

```python
# ── CHANGE THESE ────────────────────────────────────────────────────────────
BASE_MODEL   = "meta-llama/Llama-3.1-8B"     # a BASE model, never -Instruct
MAX_TOKENS   = 1024                          # p99.5 of YOUR chunks, measured
REPLAY_RATIO = 0.05                          # 1-10% of tokens, general-domain
# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, lora_alpha=64, use_rslora=True,     # alpha = 2r; rsLoRA for r > 16
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # all 7, not q/v only
    lora_dropout=0.05, bias="none",
)

args = TrainingArguments(
    output_dir="./out/cpt-lora",
    num_train_epochs=1,                       # one clean pass
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,            # 4 x 8 x 1024 = 32,768 tokens/step
    learning_rate=2e-4,                       # LoRA-CPT LR, not 2e-5
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,                        # HF default of 0.0 is a trap
    max_grad_norm=1.0,
    bf16=True,                                # fp16=True only on a T4
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",                    # transformers >= 4.46 name
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,              # pick on held-out PPL, not the last step
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",                         # use "wandb" for anything real
    run_name="cpt-domain-r32",
    seed=42,
)
```

### 12.3 The nine checks before every CPT run

```text
[ ] Corpus audit read: chars/page (1,500-4,000 = text-native), OCR fraction logged.
[ ] Dedup ran and reported: (before, after), and the removal rate. >40% => re-scope.
[ ] tau_cpt computed from UNIQUE tokens: epochs x corpus / params. In 1e-3..1e-1?
[ ] Replay present, non-empty, and included in the loss (labels != -100 for its tokens).
[ ] Held-out split is DOCUMENT-level, frozen, hashed, >=50k tokens for a stable PPL.
[ ] Learning rate: <= 5e-5 full FT / <= 2e-4 LoRA. target_modules = all 7.
[ ] Supervised fraction ~100%; no pad positions trained; EOS appended per chunk.
[ ] Base model checkpoint saved and hash-pinned; rollback path exists.
[ ] The BASE model's four numbers (domain PPL, general PPL, task, probe) recorded FIRST.
```

**Run one.** Then, in this order: read the tokenised sample (`tok.decode(ds[0][:80])`) → confirm
the supervised fraction is ~1.0 → confirm the first loss is far below `ln(vocab)` and falling
smoothly → eval on held-out **domain + general** PPL against the base → *then* touch a
hyperparameter. Do not tune anything until the base numbers exist; a run you cannot compare is
a run you cannot trust.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| Understand *why* CPT works, from first principles | **CS-12 — Domain-Adaptive Continued Pretraining** (the full treatment; §4.3–4.10 and §12 are the core) |
| Learn the behaviour stage that sits on top of CPT | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| Decide between fine-tuning, RAG, and agents | **CH-04 / CS-04 — Fine-Tuning vs RAG vs Agents** |
| Train on preferences after CPT + SFT | **CH-14 / CS-14 — The Alignment Map** |
| Understand LoRA/QLoRA/DoRA mathematically | CS-12 §7.5–7.6 (the CPT-specific corrections), and your LoRA deep-dive |
| Quantize the CPT'd model so it fits at serve time | **CH-10 / CS-10–CS-11 — Quantization** |
| Run the same pipeline without hand-written loops | **CH-15 / CS-15 — LLaMA-Factory**, **CS-16 — Unsloth** |
| Compress the domain model into something servable | CS-08 / CS-09 — Distillation |
| See the exact pipeline code with the traps pre-checked | `code/03_continued_pretraining.py`, `code/common/memory.py` |
| Practice being interviewed on this | IQ-12 — Domain adaptation questions (CS-12 §19 has 10 with answers) |
