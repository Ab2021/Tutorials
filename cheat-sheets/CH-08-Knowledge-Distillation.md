# CH-08 — Knowledge Distillation Cheat Sheet

**One-line purpose:** train a small *student* model to reproduce a large *teacher* model's
behaviour — using the teacher's full probability distribution, not just its hard labels.
**Use when:** you have a model that is too big/slow/expensive to deploy, and you can afford to
run it offline to generate training signal.
**Do NOT use when:** you need to *exceed* the teacher (distillation transfers, it does not
create capability), or the teacher is barely better than the student (there is nothing to
transfer), or you only need a smaller *artifact* rather than a smaller *model* — that is
quantisation (CH-10/CH-11) and it is far cheaper.

> **The one sentence that matters.** A hard label says "this is a 3". A soft label says
> "this is mostly a 3, a bit of an 8, and almost nothing else" — and that *relative* structure
> is the dark knowledge you are actually transferring.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **Soft targets carry more information than hard labels.** | One-hot is ~1.6 bits; a full distribution over 100k classes is far more. |
| 2 | **Temperature T controls how soft.** | T=1 is the raw distribution; high T reveals the small probabilities. |
| 3 | **Multiply the KD loss by T².** | Gradients scale as `1/T²`; without this correction, higher T silently shrinks your learning. |
| 4 | **Typical T = 2–20, α = 0.5–0.9.** | α is the weight on the soft loss. |
| 5 | **A too-large teacher can be worse than a mid-size one.** | The capacity gap. |
| 6 | **Storing full teacher logits is TB-scale.** | 150k vocab × 4 B = 600 KB/token → 600 GB per 1M tokens. |
| 7 | **Sequence-level KD (Kim & Rush) needs no logits at all.** | Just train the student on teacher-*generated text*. Far cheaper. |
| 8 | **Distillation ≠ compression.** | Compression = same architecture, fewer params. Distillation = transfer of behaviour. |
| 9 | **Distillation transfers biases too.** | A biased teacher makes a biased student, with the student's own blind spots on top. |
| 10 | **Self-distillation can work.** | Born-again networks: a student that becomes the next teacher. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Softmax with temperature** | `p_i = exp(z_i/T) / Σ_j exp(z_j/T)` | z = logits | T=1 → raw; T=5 → much flatter |
| **KL divergence** | `KL(p‖q) = Σ p_i log(p_i/q_i)` | p = teacher, q = student | Both computed **at the same T** |
| **The KD loss** | `L = α·T²·KL(p_teacher‖p_student) + (1−α)·CE(y, q_student)` | α ∈ [0,1] | α=0.7, T=4 is a common start |
| **The T² correction** | `∂L/∂z ∝ 1/T²` | — | Without `× T²`, raising T shrinks gradients |
| **Dark-knowledge bits** | `H(p) = −Σ p_i log p_i` | — | A confident one-hot ≈ 0 bits; a soft one carries more |
| **Logit storage** | `vocab × precision × tokens` | — | 150k × 4 B × 1M = **600 GB** |
| **Top-k storage** | `k × (4 + 4) × tokens` (index + value) | k=20 | 20 × 8 × 1M = **160 MB** |
| **Compression ratio** | `teacher_params / student_params` | — | 70B → 1B = 70× |

### 2.1 Why `T²` — the derivation in one line

For high T, the logits are small, so `exp(z/T) ≈ 1 + z/T`. The softmax becomes approximately
`(1 + z_i/T) / (N + Σz_j/T)`, i.e. **linear in `z/T`**. So the gradient of the KD loss with
respect to the logits scales as `1/T²`. Multiplying the loss by `T²` restores the gradient
magnitude — which is why the Hinton paper's loss has that factor and why omitting it is a
silent bug at high T rather than a crash.

### 2.2 What the soft label actually tells the student

```
Teacher (T=1)          Teacher (T=5)
  3  0.90                3  0.55
  8  0.07                8  0.18
  2  0.02                2  0.11
  9  0.01                9  0.09
  ...                    ...
```

At T=5 the *relative* ordering survives but the mass spreads. That spread is the teaching
signal: "an 8 is the plausible confusion, a 2 less so". Hard labels throw all of it away —
they say `3 = 1.0` and everything else `= 0`, which is a statement the teacher itself does
not believe.

---

## 3. Decision Tree

```
What do you actually want?
├─ A smaller ARTIFACT (same architecture, fewer bits)
│     → QUANTISATION (CH-10/CH-11). Cheaper, simpler, no training.
├─ A smaller MODEL (fewer parameters, same behaviour)
│     → DISTILLATION. You are in the right place.
└─ A faster model with the same size
      → Quantisation, or a better serving stack (batching, kernels).

Do you have access to the teacher's LOGITS?
├─ Yes (you can run it locally) → token-level KD (the full method)
└─ No (API-only teacher) ↓
    ├─ Can you generate text from it in bulk? → SEQUENCE-LEVEL KD (Kim & Rush)
    └─ No → you cannot distil from it. Fine-tune instead.

How big is the capacity gap (teacher / student)?
├─ < 10×  → comfortable. Standard KD works.
├─ 10–50× → consider an intermediate "assistant" model between them.
└─ > 50×  → the teacher's distribution is often too complex for the student to
             match; a mid-size teacher frequently beats a huge one. TEST BOTH.

Is your vocabulary huge (>100k)?
├─ Yes → full-logit storage is TB-scale. Use top-k (k=20–100) or
│        sequence-level KD instead.
└─ No  → full logits are feasible; store them.

Is the teacher's tokenizer the same as the student's?
├─ Yes → token-level KD is clean.
└─ No  → you need alignment (a mapping, or a projection layer), or use
        sequence-level KD, which sidesteps the problem entirely.

Is the teacher's quality worth transferring at all?
├─ Teacher barely beats the student → you will transfer mostly noise. Reconsider.
└─ Teacher clearly better            → proceed.
```

---

## 4. Hyperparameter Quick Reference

| Param | Typical | Range | Effect |
|---|---|---|---|
| `temperature` (T) | **4** | 2–20 | Higher = softer, more dark knowledge; too high → noise dominates |
| `alpha` (α) | **0.7** | 0.5–0.9 | Weight on the soft (KD) loss vs hard-label CE |
| `T²` correction | **on** | — | Turn it OFF only if you know why |
| Student LR | task-dependent | — | Often the same as normal training; KD is not special here |
| `top_k` logits stored | 20–100 | — | Storage vs fidelity trade |
| Teacher precision | fp16/bf16 | — | Store logits at the same precision you train at |
| Assistant model size | geometric mean-ish | — | Intermediate teacher for a big gap |
| Epochs | 1–3 | — | KD overfits too |

### 4.1 Choosing T for your teacher's confidence profile

| Teacher behaviour | Symptom | T to try |
|---|---|---|
| Very confident (near one-hot everywhere) | Soft labels ≈ hard labels; no benefit | **Higher T (10–20)** | 
| Well-calibrated | Good spread at T=1 | **T = 2–5** |
| Very flat / underconfident | Already diffuse at T=1 | **T = 1–3**; higher adds noise |
| Overconfident (a known pathology) | Sharp even when wrong | **T = 5–10**, and consider label smoothing |

> **T is not a free win.** Too high and you are training the student on the teacher's residual
> noise floor; too low and you have paid for soft labels and received hard ones. Sweep it.

---

## 5. Copy-Paste Code Snippets

### 5.1 Token-level KD — the core loop

```python
import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """The textbook KD loss, with the T^2 correction that people forget.

    student_logits, teacher_logits: (batch, seq, vocab)
    labels: (batch, seq) with IGNORE_INDEX (-100) on prompt positions
    """
    B, L, V = student_logits.shape

    # Flatten to (B*L, V) so the reduction divides by TOKENS, not by vocab.
    s_flat = student_logits.view(-1, V)
    t_flat = teacher_logits.view(-1, V)

    # Soft loss: both distributions computed AT THE SAME T.
    # F.kl_div expects LOG-probabilities for the input, probabilities for the target.
    s_log = F.log_softmax(s_flat / T, dim=-1)
    t_soft = F.softmax(t_flat / T, dim=-1)

    # THE REDUCTION TRAP: with a (N, V) input, 'mean' divides by N*V while
    # 'batchmean' divides by N. So 'mean' is V times too small (~150,000x at a
    # 150k vocab) — the soft term vanishes and you have trained a plain
    # supervised model that looks like it is working.
    soft = F.kl_div(s_log, t_soft, reduction="batchmean", log_target=False)

    # Hard loss: standard next-token CE, ignoring the masked positions.
    hard = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
    )

    # The T^2 factor restores the gradient magnitude that the 1/T softmax scaling removed.
    return alpha * (T * T) * soft + (1.0 - alpha) * hard
```

> **Two traps in a dozen lines.**
>
> **(1) The reduction, and the shape it depends on.** `F.kl_div` reduces elementwise:
> `'mean'` divides by `numel`, `'batchmean'` divides by `input.size(0)`. The ratio therefore
> depends entirely on how you shaped the tensor — which is why this bug is so slippery.
>
> | Input shape | `'mean'` divides by | `'batchmean'` divides by | `'mean'` is too small by |
> |---|---|---|---|
> | `(B, L, V)` | `B·L·V` | `B` | **`L·V`** (≈3×10⁸ at L=2048, V=150k) |
> | `(B·L, V)` (flattened) | `B·L·V` | `B·L` | **`V`** (≈150,000) |
>
> The code above flattens first, so the penalty is a factor of `V`. **Either way the soft
> term is annihilated** — it is just annihilated by a different number. Always flatten
> deliberately and use `'batchmean'`.
>
> **(2) Forgetting `T*T`.** Without it the soft term shrinks as you raise T, so "more
> temperature" appears to hurt and you conclude, wrongly, that soft targets do not help.

### 5.2 Generating and caching teacher logits

```python
import torch, numpy as np

@torch.no_grad()
def cache_logits(teacher, batches, top_k=50, path="teacher_logits.npz"):
    """Store only the TOP-K logits per position.

    Full-vocab storage at 150k vocab x 4 bytes = 600 KB per token. For a 1M-token
    corpus that is 600 GB. Top-50 index+value pairs are 50 x 8 = 400 bytes per
    token -> 400 MB. Same training signal for a fraction of the disk.
    """
    teacher.eval()
    out = []
    for batch in batches:
        logits = teacher(**batch).logits                      # (B, L, V)
        vals, idx = logits.topk(top_k, dim=-1)                # (B, L, k)
        out.append({
            "idx": idx.to(torch.int32).cpu().numpy(),
            "val": vals.to(torch.float16).cpu().numpy(),
        })
    np.savez_compressed(
        path,
        idx=np.concatenate([o["idx"] for o in out]),
        val=np.concatenate([o["val"] for o in out]),
    )
    print(f"saved to {path}")
```

```python
def kd_loss_topk(student_logits, idx, val, labels, T=4.0, alpha=0.7):
    """KD from stored top-k logits. Everything outside the top-k is treated as
    -inf, so the teacher distribution is renormalised over the kept entries."""
    B, L, V = student_logits.shape
    k = idx.size(-1)

    t_vals = val.float() / T
    t_soft = F.softmax(t_vals, dim=-1)                        # renormalised over top-k

    s_at_k = student_logits.gather(-1, idx.long()) / T
    s_log = F.log_softmax(s_at_k, dim=-1)                     # over the SAME k entries

    soft = F.kl_div(s_log, t_soft, reduction="batchmean")
    hard = F.cross_entropy(student_logits.view(-1, V), labels.view(-1),
                           ignore_index=-100)
    return alpha * (T * T) * soft + (1.0 - alpha) * hard
```

> Note the renormalisation: truncating to top-k and then applying `softmax` over the
> *truncated* vector is the correct treatment. Comparing a k-length student distribution
> against a full-vocab teacher distribution would be comparing different objects.

### 5.3 Sequence-level KD — no logits at all (Kim & Rush)

```python
"""The cheapest form of distillation, and often the most practical.

You never touch the teacher's logits. You generate text with the teacher and train
the student on (prompt, teacher_generation) with ordinary SFT. This works with an
API-only teacher and sidesteps vocabulary mismatch entirely.
"""
import json

prompts = [json.loads(l) for l in open("data/prompts.jsonl", encoding="utf-8")]

with open("data/distil_sft.jsonl", "w", encoding="utf-8") as f:
    for p in prompts:
        teacher_out = call_teacher_api(p["prompt"])       # your API call
        f.write(json.dumps({
            "messages": [{"role": "user",      "content": p["prompt"]},
                         {"role": "assistant", "content": teacher_out}],
        }, ensure_ascii=False) + "\n")

# Then: train the student on data/distil_sft.jsonl with ordinary SFT (CH-13).
print("now run: python code/01_sft_lora.py --data data/distil_sft.jsonl")
```

> **What you lose.** Sequence-level KD gives the student only the teacher's *argmax path*, so
> no dark knowledge and no gradient signal on the non-chosen tokens. It is a much weaker
> signal than token-level KD — but it is available when nothing else is, and it is strong
> enough to be the basis of most practical "make a small model behave like a big one" work.

### 5.4 Checking the teacher/student vocabulary alignment

```python
from transformers import AutoTokenizer

t_tok = AutoTokenizer.from_pretrained(TEACHER)
s_tok = AutoTokenizer.from_pretrained(STUDENT)

if t_tok.get_vocab() == s_tok.get_vocab():
    print("✅ identical vocabularies — token-level KD is clean")
else:
    t_v, s_v = t_tok.get_vocab(), s_tok.get_vocab()
    shared = set(t_v) & set(s_v)
    print(f"⚠  vocabularies differ:")
    print(f"     teacher {len(t_v):,}  student {len(s_v):,}  shared {len(shared):,}")
    print(f"     {len(shared)/len(t_v):.1%} of the teacher's vocabulary is usable")
    print()
    print("  A token-level comparison across different vocabularies is comparing")
    print("  different objects at the same index. Options:")
    print("    1. Use a student with the teacher's tokenizer (often the best fix).")
    print("    2. Learn a projection between the two (fiddly, degrades quality).")
    print("    3. Use sequence-level KD, which needs no alignment at all.")
```

> **Same-vocabulary is the single biggest practical constraint on token-level KD.** Most
> same-family pairs satisfy it (Llama-3.2-1B and Llama-3.1-70B share a tokenizer); cross-family
> pairs usually do not. Check before you plan anything else.

---

## 6. CLI Commands

```bash
# ── The handbook's script ───────────────────────────────────────────────────
# The two modes are MUTUALLY EXCLUSIVE and one of them is REQUIRED. There is no
# --mode flag. --from-teacher takes --prompts (a .jsonl of prompts); --token-kd
# takes --text (a raw .txt corpus). Passing the wrong one is the common mistake.
python code/07_distillation.py --help

# --dry-run FIRST, every time. On --token-kd it still loads the tokenizers (the
# vocab-alignment check is the go/no-go and needs the real vocabs) but NOT the
# weights — so it plans a 32B teacher from a laptop.
python code/07_distillation.py --from-teacher --dry-run --prompts data/prompts.jsonl
python code/07_distillation.py --token-kd     --dry-run --text    data/corpus.txt \
    --size-hint 32B -T 4 --alpha 0.7 --top-k 20

# ── Sequence-level KD: generate, then SFT the student on the result ─────────
# 1. Dry-run to size the job, then generate 20 and READ them before committing.
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl --dry-run
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl
# 2. Filter the raw generations (refusals, boilerplate, instruction-echoing survive
#    generation and are imitated faithfully), then train as ordinary SFT:
python code/data/make_instruction_data.py --filter --in data/seqkd.jsonl
python code/01_sft_lora.py --data data/seqkd.jsonl --out out/student

# ── Token-level KD: the soft-target loss, and its storage problem ───────────
# Requires a SHARED VOCABULARY — the script exits early if the tokenizers differ.
python code/07_distillation.py --token-kd --teacher Qwen/Qwen2.5-32B-Instruct \
    --student Qwen/Qwen2.5-0.5B-Instruct --text data/corpus.txt --dry-run
```

Note the two different temperatures: `-T` is the **KD** temperature (softening, default
2.0); `--teacher-temp` is the **sampling** temperature for generation (default 0.8). They
appear in the same script and mean unrelated things.

# ── Check vocab alignment between teacher and student BEFORE planning ───────
python -c "
from transformers import AutoTokenizer
T = 'meta-llama/Llama-3.1-70B-Instruct'
S = 'meta-llama/Llama-3.2-1B-Instruct'
tv, sv = AutoTokenizer.from_pretrained(T).get_vocab(), AutoTokenizer.from_pretrained(S).get_vocab()
print('teacher', len(tv), 'student', len(sv), 'shared', len(set(tv)&set(sv)))
print('usable fraction:', f'{len(set(tv)&set(sv))/len(tv):.1%}')
"

# ── Generate a distillation corpus from a teacher (sequence-level) ──────────
python -c "
import json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
mid='meta-llama/Llama-3.1-70B-Instruct'
tok=AutoTokenizer.from_pretrained(mid); m=AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map='auto')
prompts=[json.loads(l) for l in open('data/prompts.jsonl',encoding='utf-8')]
with open('data/distil_sft.jsonl','w',encoding='utf-8') as f:
    for p in prompts:
        ids=tok.apply_chat_template([{'role':'user','content':p['prompt']}], tokenize=True, add_generation_prompt=True, return_tensors='pt').to(m.device)
        out=m.generate(ids, max_new_tokens=512, do_sample=False)
        txt=tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        f.write(json.dumps({'messages':[{'role':'user','content':p['prompt']},{'role':'assistant','content':txt}]}, ensure_ascii=False)+'\n')
print('done')
"

# ── Measure how much dark knowledge the teacher actually has (is KD worth it?) ──
python -c "
import torch, torch.nn.functional as F
# If the teacher's own entropy is ~0, its soft labels ARE hard labels and
# token-level KD buys you nothing over plain SFT.
logits = torch.load('teacher_batch.pt')          # any (B, L, V) tensor
p = F.softmax(logits, dim=-1)
ent = -(p * torch.log(p + 1e-9)).sum(-1).mean()
import math
print(f'teacher mean entropy {ent:.3f} nats  (uniform would be {math.log(logits.size(-1)):.2f})')
print('near 0   -> teacher is near one-hot; KD gives little. Consider higher T.')
print('moderate -> good dark knowledge; token-level KD is worth the storage.')
"
```

---

## 7. VRAM / Cost Calculator

### 7.1 The logit-storage wall — why token-level KD is rare in practice

| Vocab | Precision | Per token | Per 1M tokens | Per 1B tokens |
|---|---|---|---|---|
| 32k | fp32 | 128 KB | 128 GB | 128 TB |
| 128k | fp32 | 512 KB | 512 GB | 512 TB |
| **150k** | **fp32** | **600 KB** | **600 GB** | **600 TB** |
| 150k | fp16 | 300 KB | 300 GB | 300 TB |
| 150k | top-20, fp16 | 120 B | 120 MB | 120 GB |
| 150k | top-50, fp16 | 300 B | 300 MB | 300 GB |
| 150k | top-100, fp16 | 600 B | 600 MB | 600 GB |

**For comparison, the raw text is about 4 bytes per token** (≈4 characters/token in English).
Full-vocab fp32 logits at a 150k vocabulary are **~150,000× the size of the corpus they
describe.** That single ratio is the reason sequence-level KD and top-k caching exist.

### 7.2 The compute cost

| Stage | Cost driver | Note |
|---|---|---|
| Teacher forward passes | `2 × N_teacher × tokens` FLOPs | One-off, and the dominant term |
| Student training (KD) | `6 × N_student × tokens` | Same as normal training, plus the KL |
| Teacher generation (seq-level) | `2 × N_teacher × tokens + KV cache` | Autoregressive — much slower per token than a forward pass |
| Storage | §7.1 | Often the practical blocker |

### 7.3 Rough wall-clock for generating a distillation corpus

| Teacher | Tokens to generate | Rough order |
|---|---|---|
| 7B, one A100 | 1M | ~1–3 hours (batched, greedy) |
| 70B, 4×A100 | 1M | ~4–12 hours |
| API teacher at $3/1M output | 1M | minutes, ~$3 + prompt cost |

> **The API route is often the fastest and cheapest way to build a distillation corpus** — you
> trade the ability to get logits for a huge saving in wall-clock. If your teacher is
> API-only, sequence-level KD is not a compromise; it is the plan.

### 7.4 Teacher size vs student quality — the capacity gap

| Gap | Outcome | Recommendation |
|---|---|---|
| Student ≈ teacher | Little to gain | Not worth it |
| 2–10× | **The sweet spot** | Standard KD works well |
| 10–50× | Workable, sometimes unstable | Try an intermediate teacher |
| > 50× | The teacher is often *worse* than a mid-size one | **Test a mid-size teacher too** |
| > 200× | Usually a waste | Train a mid-size student on data instead |

> **This is the most counter-intuitive result in the field**, and it is well documented: for a
> very small student, distilling from a 70B teacher frequently loses to distilling from a 7B
> teacher. The bigger teacher's distribution is so far from anything the student can
> represent that the KL signal is mostly noise the student cannot fit.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Student is no better than plain SFT** | Soft loss is effectively zero | Check `reduction="batchmean"` and the `T*T` factor |
| Student gets *worse* as T rises | Missing `T*T` correction | Multiply the soft term by `T*T` |
| Student is much worse than the teacher | Capacity gap too large | Use a smaller/mid-size teacher, or a bigger student |
| Student outputs are degenerate/repetitive | T too high; training on the teacher's noise floor | Lower T to 2–5 |
| Student mimics the teacher's *mistakes* | Distillation transfers bias faithfully | Curate teacher outputs; consider filtering by a reward |
| `RuntimeError: shape mismatch` in the KL | Different vocabularies, or vocab sizes differ | Align tokenizers (§5.4) or switch to sequence-level KD |
| Soft loss is NaN | `log_softmax` on fp16 overflow, or T=0 | Use bf16; ensure `T > 0` |
| Disk fills during logit caching | Full-vocab fp32 storage | Top-k (§5.2); verify with §7.1 |
| Student trains but generations are truncated | EOS not in the distillation data | Append EOS; check the teacher's generation config |
| Student loses general ability | Distilled too narrowly | Mix in general instruction data as replay |
| KD helps on teacher-domain data, hurts elsewhere | Expected — you copied the teacher's distribution | Measure on both; decide which you care about |
| Student can't match teacher on long outputs | Sequence-level KD gives only the argmax path | Token-level KD, or accept the ceiling |
| Very slow generation of the corpus | Autoregressive teacher generation | Batch it; use greedy; or use an API |
| `kl_div` returns a negative number | You passed probabilities instead of log-probabilities as `input` | `input=log_softmax(...)`, `target=softmax(...)` |

> **That last row is a genuine trap.** `F.kl_div` expects **log**-probabilities as `input` and
> **probabilities** as `target`. Passing probabilities for both gives a mathematically
> meaningless number that can be negative — and nothing raises.

---

## 9. Comparison Matrix

### 9.1 The three families of distillation

| Family | What is matched | Needs | Quality | Used when |
|---|---|---|---|---|
| **Response-based** | The output distribution (logits) | Same vocab, logits access | Strong | Teacher is local, vocabularies match |
| **Feature-based** | Intermediate hidden states | Layer alignment between models | Strong but fiddly | Same architecture family, research |
| **Relation-based** | Relationships *between* examples | Batches / similarity structure | Moderate | Cross-architecture, or no logits |
| **Sequence-level** (Kim & Rush) | The teacher's generated text | Only generated text | Weaker but robust | API teacher, or different vocabularies |

### 9.2 Distillation vs the other ways to get a smaller model

| | Distillation | Quantisation | Pruning | Train small from scratch |
|---|---|---|---|---|
| Mechanism | Transfer behaviour | Fewer bits per weight | Remove weights | Learn from data |
| Needs a teacher | ✅ | ❌ | ❌ | ❌ |
| Needs training | ✅ | PTQ: no / QAT: yes | Usually | ✅ |
| Parameter count | Reduced | Same | Reduced (sparse) | Reduced |
| Real speedup | Proportional to param cut | 2–4× (hardware-dependent) | Only with sparse kernels | Proportional |
| Quality risk | Ceiling = teacher | Low at 8-bit, real at 4-bit | High without retraining | High — needs more data |
| Typical cost | Teacher inference + student train | Minutes to hours | Hours | Large |
| Best for | Behaviour transfer, cheap inference | Same model, less memory | Rarely worth it today | When you have a lot of data |

### 9.3 Where distillation sits in the pipeline

| Goal | Right tool |
|---|---|
| Smaller *artifact*, same model | **Quantisation** (CH-10/CH-11) |
| Smaller *model*, similar behaviour | **Distillation** (this card) |
| Same model, faster on your hardware | Quantisation + better serving |
| Teach a small model a *task* | **SFT on the small model** — often beats distillation |
| Teach a small model a *reasoning style* | Distillation from a reasoning teacher, or RL (CS-26) |

> **The most important row:** if you have labelled task data, plain SFT on the small model is
> frequently better *and* simpler than distilling from a large one. Distillation earns its
> keep when you do **not** have labels and the teacher does — which is precisely the
> situation sequence-level KD solves.

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Temperature T | **2–20** | 4 is a common default |
| α (weight on soft loss) | **0.5–0.9** | 0.7 typical |
| T² correction | **required** | Gradients scale as `1/T²` |
| Softmax approximation (high T) | `exp(z/T) ≈ 1 + z/T` | Why the T² arises |
| Logit storage, 150k vocab fp32 | **600 KB/token** | → 600 GB per 1M tokens |
| Logit storage, 150k fp16 | **300 KB/token** | |
| Top-20 fp16 | **120 B/token** | 5000× smaller than full fp32 |
| Text itself | **~4 B/token** | Full logits are ~150,000× the corpus |
| Capacity-gap sweet spot | **2–10×** | Teacher:student params |
| Capacity-gap danger | **> 50×** | Mid-size teacher often wins |
| Compression ratio example | 70B → 1B = **70×** | |
| Teacher forward FLOPs | **2 × N × tokens** | One-off |
| Student training FLOPs | **6 × N × tokens** | Same as normal |
| `kl_div` argument order | `input` = **log**-probs | `target` = probs |
| Typical distillation corpus | 1k – 1M prompts | |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `RuntimeError: The size of tensor a (V1) must match the size of tensor b (V2) at non-singleton dimension 2` | Teacher and student vocabs differ | Align tokenizers, or use sequence-level KD |
| `RuntimeError: Expected all tensors to be on the same device` | Teacher output left on its own device | Move teacher logits to the student's device |
| `ValueError: Expected input to be a floating point tensor` | Integer logits (e.g. from a quantised path) | `.float()` before the softmax |
| `CUDA out of memory` during the teacher pass | Teacher + student both resident | Run the teacher on a separate process and cache; or use `device_map` and offload |
| `kl_div` returns a negative value | Probabilities passed where log-probabilities were expected | `input=log_softmax(...)` |
| Soft loss is ~1e-6 while hard loss is ~2.0 | `reduction="mean"` ate the soft term | `"batchmean"` on a flattened tensor (§5.1) |
| Loss fine, generations nonsense | Masking or template mismatch in the student | CH-13 §8 |
| `np.savez_compressed` disk-full | Full-vocab fp32 caching | §5.2 top-k |
| `OverflowError` in `softmax` at large T | T applied to already-large logits in fp16 | bf16, or subtract the max first |
| Teacher generates truncated text | `max_new_tokens` hit | Raise it, or filter truncated rows out of the corpus |
| Student mimics a refusal the teacher never gave | The corpus contains template artifacts | Inspect 20 random generations manually before training |
| Distillation loss decreases but eval does not improve | You are fitting the teacher's noise | Lower T; curate the corpus; check teacher quality |

---

## 12. Copy-Paste Starter Config

### 12.1 The decision first

```
Do you have the teacher's logits AND a shared vocabulary?
├─ Yes → token-level KD (§5.1 + §5.2). Best quality per token of training signal.
└─ No  → sequence-level KD (§5.3). Generate text, then run ordinary SFT.
```

### 12.2 Sequence-level KD, end to end (the practical default)

```bash
# 1. Build the prompt set (500 - 50,000 prompts is the usual range)
#    Use REAL user prompts if you have them; they define the distribution that matters.

# 2. Generate with the teacher  (see §6 for the script)
#    -> data/distil_sft.jsonl  in {messages:[...]} form

# 3. INSPECT. Always. Read 20 random rows yourself.
python -c "
import json, random
rows = [json.loads(l) for l in open('data/distil_sft.jsonl',encoding='utf-8')]
print(f'{len(rows)} rows')
for r in random.sample(rows, 5):
    print('=' * 70)
    print('USER:', r['messages'][0]['content'][:200])
    print('ASST:', r['messages'][1]['content'][:300])
"

# 4. Filter: drop truncated, empty, or refusal-shaped generations
python -c "
import json
keep = []
for l in open('data/distil_sft.jsonl', encoding='utf-8'):
    r = json.loads(l)
    a = r['messages'][1]['content']
    if len(a.split()) < 5:            continue   # too short
    if a.rstrip().endswith(('...','—')): continue   # likely truncated
    keep.append(l)
open('data/distil_sft.clean.jsonl','w',encoding='utf-8').writelines(keep)
print(f'kept {len(keep)}')
"

# 5. Train the student with ORDINARY SFT — no special loss needed
python code/01_sft_lora.py --data data/distil_sft.clean.jsonl --out out/student

# 6. Evaluate: student vs teacher vs a small model trained on the SAME prompts
#    from your own labels. If supervised SFT wins, use that instead.
```

### 12.3 The six checks before you trust a distillation run

| # | Check | Pass condition |
|---|---|---|
| 1 | **Teacher is genuinely better** | Measured, on your task, not assumed |
| 2 | **Vocabulary alignment** (§5.4) | Identical, or you chose sequence-level KD deliberately |
| 3 | **Teacher's entropy is non-trivial** (§6) | Otherwise soft labels ≈ hard labels |
| 4 | **Reduction and `T²`** (§5.1) | `batchmean` on a flattened tensor, `× T*T` |
| 5 | **Capacity gap** (§7.4) | ≤ ~10×, or you tested a mid-size teacher |
| 6 | **A supervised-SFT baseline** | Distillation beat it — or you should ship the SFT model |

> Check 6 is the one that saves you the most time and gets skipped the most. If you have any
> labelled data at all, train the student on it directly and compare. Distillation is a way
> to manufacture labels from a teacher; if you already have labels, its main advantage is gone.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full foundations treatment | **CS-08 — Knowledge Distillation: Foundations** |
| The applied LLM → SLM module | **CH-09 / CS-09 — Distillation: LLM → SLM** |
| To compress without training | **CH-10 / CH-11 — Quantization** |
| To train the student once you have data | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To go beyond the teacher's ceiling | **CH-14 / CS-14 — The Alignment Map**, **CS-26 — GRPO** |
| The repo's implementation | `code/07_distillation.py` |
| Practice being interviewed on this | **IQ-08 — Interview Questions: Knowledge Distillation** |

