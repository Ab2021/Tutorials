# CH-09 — Distillation: LLM → SLM Cheat Sheet

**One-line purpose:** make a 0.5–3B *student* behave like a 70B-class *teacher* by training it
on the teacher's **outputs** — which in 2025–26 normally means generating text with the teacher
and running ordinary SFT, with **no distillation loss anywhere in the pipeline**.
**Use when:** you have a teacher that is too big, too slow or too expensive to serve; you can
run it offline or reach it over an API; and you have no labels (or far too few). Distillation
*manufactures* labels from a teacher.
**Do NOT use when:** you only need a smaller *artefact* (that is quantisation — CH-10 §9), you
only need a smaller *latency* (speculative decoding keeps the model), your task is a small
classification with ≥5k human labels (train a small encoder — CS-07), you need new *facts*
(RAG — CH-04), or your monthly API bill is under a few hundred dollars (the engineering never
amortises — CS-09 §16.4).

> **The one sentence that matters.** 85–90 % of what the industry calls "distilling an LLM" is
> **sequence-level KD**: the teacher writes the answers, you filter them, and you SFT the
> student on the result. That is not a hack around distillation — it *is* forward-KL
> distillation in sequence space with a Monte-Carlo estimator (CS-09 §4.3).

> **This card covers only what is *different* when the teacher is a modern LLM and the student
> is a 0.5–3B SLM.** Hinton's loss, the `T²` factor, dark knowledge, the α-convention trap and
> the three KD families are **CH-08 §1–§5** and are not repeated here. §1–§13 below mirror the
> CH-13 cheat-sheet structure.

> **Convention used in this card.** A row marked **RoT** is a rule of thumb with no published
> measurement behind it — treat it as a starting point, not a result. Unmarked rows are
> documented results with a citation, a repo file, or a case-study section. Vendor model-card
> figures are release-time numbers and are prompt-format sensitive; re-verify before quoting
> (CS-09 §4.9).

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **Sequence-level KD is what "distillation" means today.** ~85–90 % of production. | It needs no logits, so it works with an API teacher and crosses tokenizer families (CS-09 §13.1). |
| 2 | **Token-level (logit) KD needs an identical tokenizer *mapping*, not just size.** | Two 32k BPE vocabs line up perfectly and mean different things. Same `V` ≠ same tokenizer (CS-09 §1.3, failure 3). |
| 3 | **Feature / hidden-state KD needs architectural access to both models.** | It buys +1–2 points over logit KD when it applies, and you own a projection module forever (CS-09 §13.1). |
| 4 | **The capacity gap is real — and narrower than the folklore.** | For logit KD a too-large teacher measurably loses to a mid-size one (CS-08 §4.6). For sequence-level KD on a *verifiable narrow* task the opposite holds (R1: 671B → 1.5B). §2.4 and §4.4. |
| 5 | **Full logit caching is infeasible; top-k truncation is the standard workaround.** | 150k vocab × 4 B = **600 KB/token** → 25.5 TB for a 42.5M-token corpus. Top-100 makes it 34 GB (CS-09 §4.4). |
| 6 | **The teacher runs offline, once.** It is a *data generator*, not a loss term. | That is why response KD needs one GPU, not two (CS-09 §4.1). |
| 7 | **The teacher's flaws are copied faithfully: verbosity, hedging, refusals, instruction-echoing.** | Every token of a teacher response carries gradient weight 1.0. Unlike a soft label, a teacher's *confident wrong answer* is a full-strength target (CS-09 §2.2). |
| 8 | **Generate 20, then READ them, before you generate 5,000.** | The script prints this warning at `--dry-run` for a reason (CS-09 §5.3). |
| 9 | **The training half is ordinary SFT.** Mask the prompt, 1–3 epochs, LoRA LR. | `completion_only_loss=True` + a prompt-level split. CH-13 §4 and §10 apply unchanged. |
| 10 | **Judge family ≠ teacher family, and never report a bare win rate.** | Self-preference bias means a win rate against your own teacher's family measures mimicry, not quality (CS-09 §12.4). |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Why "just SFT" *is* KD** | `KL(p_T‖p_θ) = −H(p_T) − E_{y~p_T}[log p_θ(y|x)]` | `H(p_T)` is constant in `θ` | `argmin_θ KL = argmax_θ` likelihood of teacher samples → SFT (CS-09 §4.3) |
| **The KD loss (Hinton)** | `L = α·T²·KL(p_T‖p_θ) + (1−α)·CE(y, q_θ)` | α weights the **soft** term | CH-08 §2. Full derivation: CH-08 §2.1 |
| **The repo script's loss** | `L = α·T²·KL + (1−α)·hard_CE` | `--alpha` weights the **soft** term, default **0.7** | **Same convention as Hinton.** `code/07_distillation.py` (see the `KD_MIX` block and the `--alpha` argparse help) |
| **Forward KL** | `KL(p_T‖p_θ) = Σ p_T log(p_T/p_θ)` | mode-covering | Punishes `p_θ = 0` where `p_T > 0` → the student hallucinates on the teacher's tail (CS-09 §4.3) |
| **Logit storage (full)** | `V × bytes × tokens` | 4 B fp32 | 150,000 × 4 B = **600 KB/token**; 42.5M tokens = **25.5 TB** |
| **Logit storage (top-k)** | `k × (idx + val bytes) × tokens` | 8 B/entry in the script | k=100 → 800 B/token → 42.5M tokens = **34 GB** |
| **Softmax buffer, both models resident** | `B × L × V × 4 B × 3` | 3 tensors for backward | B=8, L=2048, V=128,256 → **8.41 GB/tensor ≈ 25 GB** |
| **Teacher weights resident** | `P × bytes/param` | 2 / 1 / 0.5 B | 32B → 64 / 32 / 16 GB (decimal) |
| **Teacher generation cost** | `N × k × (in·P_in + out·P_out) / 1e6` | k = samples per prompt | 50k × 3 × (200 in + 600 out) at GPT-4o rates = **$975** |
| **Student training FLOPs** | `6 × N × D` | D = tokens seen | 1.5B × 127.5M tokens = 1.148e18 FLOPs |
| **Compression ratio** | `teacher_params / student_params` | — | 70B → 1.5B = **47×** |
| **Capacity gap** | `teacher_params / student_params` | — | 2–10× comfortable; >50× test a mid-size teacher too (CH-08 §7.4) |
| **Data volume target** | `examples × tokens/example × epochs` | **RoT** | 50k × 850 × 3 = 127.5M response tokens — inside the 20M–200M band (CS-09 §7.2) |
| **Tokens from words / chars** | `words × 1.33` / `chars / 4` | English BPE | 1,000 words ≈ 1,330 tokens |

### 2.1 The reduction convention — read the code before you cite a loss number

The single most-copied bug in KD (CH-08 §5.1) has a *second* form once you move to a
truncated top-k tensor, because `batchmean` divides by `input.size(0)` — and what
`size(0)` *is* depends on the shape you built.

```python
# code/07_distillation.py, lines 250-261 — the actual convention
t_soft = F.softmax(t_top / T, dim=-1)          # t_top: (seq_len, top_k)
s_log  = F.log_softmax(s_top / T, dim=-1)      # s_top: (seq_len, top_k)
kd   = F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)
hard = F.cross_entropy(s_logits[:, :-1].reshape(-1, V).float(), ids[:seq][1:].reshape(-1))
loss = a.alpha * hard + (1 - a.alpha) * kd
```

| Fact about this code | Consequence |
|---|---|
| The KD tensors are **not flattened** — they are `(seq_len, top_k)` | `batchmean` divides by **`seq_len`** (positions), *not* by the batch or by examples |
| The hard term is `cross_entropy` on a `(-1, V)` reshape | It divides by **`seq_len − 1`** positions (the shift drops one), with **no `ignore_index`** |
| The two terms therefore average over different position counts | A one-position discrepancy at `seq_len=512` — cosmetic, but do not describe them as "the same mean" |
| The teacher's top-k is renormalised over the *kept* entries, and the student is gathered **at the teacher's indices** | Both distributions live on the same support. This is correct practice (CH-08 §5.2) |

> **Beyond the video.** The instructor's notebook (CS-09 §6.3) computes `kl_loss(s_log_soft,
> t_soft)` over a `[B, V]` tensor and correctly uses `batchmean`. Once you truncate to top-k,
> the tensor becomes `(seq, k)` and the same reduction silently changes meaning — from
> "per-example mean" to "per-position mean". Neither is wrong; they are different numbers. Log
> `kd` and `hard` separately, as the script does, so you can see which one is doing the work
> (CH-08 §12.3, check 3).

---

## 3. Decision Tree

```
Can you get the teacher's LOGITS over the STUDENT's vocabulary?
├─ No (API-only, or a different family) ──────────────────────────────────────────┐
│                                                                                │
└─ Yes (you can run it locally, same family) ↓                                   │
   ├─ assert tok_T.get_vocab() == tok_S.get_vocab()   ← mapping, not size!       │
   │  ├─ Fails → you are in the right-hand branch. Go there.                     │
   │  └─ Passes → TOKEN-LEVEL KD (CH-08 §5.1). Best signal per token.            │
   │        └─ Corpus > ~5M tokens? → cache TOP-K LOGITS OFFLINE, one pass.      │
   │             └─ Full-vocab cache impossible: 600 KB/token (§2).              │
   │                                                                             │
   └─────────────────────────────────────────────────────────────────────────────┘
                                                                                 │
                                          ┌──────────────────────────────────────┘
                                          ▼
              Can you GENERATE text from the teacher in bulk?
              ├─ No  → you cannot distil from it. Fine-tune on your own data instead.
              └─ Yes → SEQUENCE-LEVEL KD (this is the default answer; ~85–90 % of
                       production — CS-09 §13.1)
                   1. Curate a PROMPT SET (real traffic > Self-Instruct seeds > Magpie)
                   2. Generate 20, READ them. Then generate the rest at T = 0.7–1.0
                   3. Filter: dedup → decontaminate → verify → judge  (CS-09 §5.3)
                   4. SFT the student on it with CH-13's recipe

Do you have your OWN labelled data for this task?
├─ Yes, and it is ≥ ~5k examples → train the student on it directly FIRST.
│     Distillation's only structural advantage is manufacturing labels.
│     If hard-label SFT wins, ship that — it is simpler and cheaper. (CH-08 §12.3, check 6)
└─ No / far too few → distillation is the right tool. Continue.

How big is the capacity gap (teacher / student)?
├─ < 10×   → comfortable.
├─ 10–50×  → workable. Logit KD gets unstable here; sequence-level KD does not.
└─ > 50×   → RUN A TEACHER-SIZE SWEEP (3 teachers × 1 student, everything else fixed).
             A mid-size teacher frequently wins for a ≤1B student (CH-08 §7.4).
             Exception: a *verifiable narrow task* (math, code) with a reasoning
             teacher — the gap stops mattering (R1 671B → 1.5B, CS-09 §15.4).

Is the teacher genuinely competent at YOUR task?
├─ No (≤ chance, or untuned) → STOP. You will distil noise and the loss will still fall
│                                smoothly. Measure first (CS-09 §14.3).
└─ Yes → proceed.

Is the prompt distribution at inference the same as the one you distilled on?
├─ No / unknown → you are shipping a confidently wrong model on unseen inputs.
│                 Build a drift monitor, or do not ship (CS-09 §16.5).
└─ Yes → ship, and monitor anyway.
```

---

## 4. Hyperparameter Quick Reference

### 4.1 Loss knobs (token-level KD only)

| Param | Script default | Range | Effect of getting it wrong | Marker |
|---|---|---|---|---|
| `-T` / `--temperature` (**KD** temperature) | **2.0** | 1.5–6 for LLMs; 2 is the floor | `T→1`: target is near one-hot, you paid for soft labels and got hard ones. `T→∞`: both distributions go uniform, information destroyed, soft term → 0 | CS-09 §4.5 |
| `--alpha` (**weight on the SOFT / KD term**) | **0.7** | 0.5–0.9. Hinton's own grid ran α ∈ {0.1 … 0.9} and found the soft-heavy end best when the teacher is strong; the classic `α=0.7` from CH-08 §4 is a good default | α is **not** symmetric in effect: `--alpha 0.0` is plain SFT wearing a distillation costume (you pay for the teacher forward pass and get nothing), `--alpha 1.0` discards the ground truth entirely and lets the student inherit the teacher's hallucinations on tokens the teacher never saw. On a weak/quantised teacher, *lower* α toward the hard side | `code/07_distillation.py` `--alpha` help; CH-08 §4 |
| `T²` on the soft term | **on** (hard-coded) | — | Off → raising `T` silently shrinks the KD gradient; the run trains as plain SFT and looks fine | CH-08 §2.1 |
| `--top-k` | **100** | 20–100 | Lower = smaller cache, slightly lossier. k=20 captures >99.9 % of the KL mass on real distributions | CH-08 §5.2 |
| `--seq-len` | **512** | 512–2048 | The script's token-KD mode runs ONE forward pass over `min(seq_len, corpus)` tokens and prints the loss terms — it does not train | `code/07_distillation.py` L237 |
| `--quant-bits` | **1.0** (int8) | 2 / 1 / 0.5 | Only affects the VRAM *estimate* and (in the dry-run tip) the recommendation. Never quantise a teacher for logit KD — it perturbs the soft targets | CS-09 §9.4 |

### 4.2 The α convention — one table, three sources, three meanings

| Source | Formula | What `α = 0.7` gives you |
|---|---|---|
| Hinton 2015 / CH-08 §2 | `α·T²·KL + (1−α)·CE` | 70 % **soft** / 30 % hard |
| CS-09 §6.3 notebook cell | `alpha_soft * loss_soft + (1 - alpha_soft) * loss_hard` | 70 % **soft** / 30 % hard (named `alpha_soft`, so it is readable) |
| `code/07_distillation.py` | `a.alpha * hard + (1 - a.alpha) * kd` | 70 % **hard** / 30 % soft |

**Practical rule:** never pass a bare `--alpha` without checking which term it multiplies. The
script's own `--dry-run` does not print α, but the token-KD path prints
`alpha  0.7 hard-label / 0.3 KD` — read that line, not the number you typed. **RoT.**

### 4.3 Data-generation knobs (sequence-level KD)

| Param | Script default | Range | Effect of getting it wrong |
|---|---|---|---|
| `--n` (prompts kept) | **1000** | 500 – 50,000 | Below ~500 prompts the student memorises them (CS-09 §8.2) |
| `--teacher-temp` (**sampling** temperature) | **0.8** | 0.7–1.0 | 0 → zero diversity, N near-copies. >1.1 → rambling; dedup kill rate spikes |
| `--max-new-tokens` | **512** | 600–4096 | Too low → truncated responses teach the student to cut off mid-sentence. Reasoning traces need 2048+ |
| `k` (samples per prompt) | — (not in the script) | 1–8 chat; 8–64 math/code with a verifier | k=1 keeps the teacher's tail errors at full weight. Gains flatten after k≈4 for chat |
| Keep rate after filters | the script drops responses < 5 words | 10–70 % | Under-filter → the student learns hallucinations. Over-filter → 2k examples and the distribution is gone |
| Decontamination n-gram | — (external) | 13-gram | Skipping it inflates every number you will report (CS-09 §12.1) |

### 4.4 Model-choice knobs

| Knob | Default answer | Why |
|---|---|---|
| Teacher family | **Same family as the student** if you want logit KD | The default pair in the script (`Qwen2.5-32B-Instruct` → `Qwen2.5-0.5B-Instruct`) shares a vocabulary *by construction*, which is what makes `--token-kd` pass its guard |
| Teacher size | **RoT: 7–32B** for a 0.5–3B student on a generalist task; sweep three sizes | The capacity-gap curve is real for logit KD and flat-to-inverted for sequence-level KD (§1 row 4) |
| Student size | **1.5B is the modern sweet spot**; 0.5B for narrow tasks | Floors: ~0.3B holds a chat format; ~1.5B for reliable JSON / tool calls; long-chain reasoning ~1.5B *if distilled from a reasoning teacher* (CS-09 §4.9) |
| Student starting point | An **Instruct** model, not a base model | You are teaching behaviour, not capability. Same rule as CH-13 §3 |
| Teacher precision | **Never quantise for logit KD.** Fine for sequence-level KD | Logit KD: quantisation noise raises the KL floor. Response KD: it only affects text quality, which you filter (CS-09 §9.4) |
| Rejection sampling vs a judge | **A verifier beats a judge** | A judge is a proxy; a verifier is ground truth (CS-09 §5.3) |

---

## 5. Copy-Paste Code Snippets

### 5.1 Sequence-level KD, end to end (the practical default)

```bash
# 1. Prompts: REAL traffic if you have it — it defines the distribution that matters.
#    Accepted keys in the script: "instruction", "prompt", or "question" (+ optional "input").
python code/07_distillation.py --from-teacher --dry-run \
    --teacher Qwen/Qwen2.5-32B-Instruct --prompts data/prompts.jsonl --n 5000 \
    --size-hint 32B --quant-bits 1.0 --out data/seqkd.jsonl

# 2. Generate 20 first and READ them. Then the full run (drop --n to 20 to do exactly that).
python code/07_distillation.py --from-teacher \
    --teacher Qwen/Qwen2.5-32B-Instruct --prompts data/prompts.jsonl --n 5000 \
    --teacher-temp 0.8 --max-new-tokens 512 --out data/seqkd.jsonl

# 3. Filter (the repo's quality_filter — it has no CLI, see §5.2b), then train with
#    ORDINARY SFT. There is no special loss anywhere in step 3.
python code/01_sft_lora.py --data data/seqkd.train.jsonl --output out/student
```

The script writes **Alpaca schema** (`{"instruction", "input", "output"}`), which is exactly
what `01_sft_lora.py` loads by default (`format="alpaca"`). The two are compatible — see §6 for
the one flag in the script's printed hint that is wrong.

### 5.2 What step 3 must add that the generator does not do for you

```python
import json, random

# (a) The 20-generate-and-read discipline. Do this BEFORE the full run.
rows = [json.loads(l) for l in open("data/seqkd.jsonl", encoding="utf-8")]
for r in random.sample(rows, min(20, len(rows))):
    print("=" * 78)
    print("PROMPT:", r["instruction"][:200])
    print("TEACHER:", r["output"][:400])

# (b) The filter stack, cheapest first (CS-09 §5.3). Reuse the repo's own filter rather
#     than rewriting it — quality_filter() in code/data/make_instruction_data.py already
#     implements the seven rules that matter. It has NO CLI, so import it.
import sys; sys.path.insert(0, "code/data")
from make_instruction_data import quality_filter

kept, report = quality_filter(rows)
print("rejected:", report)   # too_short / too_long / refusal / bad_opener /
                             # placeholder / echoes_instruction / duplicate

# The repo's thresholds, so you know what you just applied:
#   too_short          < 8 words            (stricter than the generator's own < 5)
#   too_long           > 400 words
#   refusal            any marker in REFUSAL_MARKERS
#   bad_opener         any BAD_OPENERS prefix
#   placeholder        {slot} / [INSERT / XXX / TODO / '....'
#   echoes_instruction instruction[:40] appears anywhere in the output
#   duplicate          instruction normalised to [a-z0-9][:80] already seen
# Add what it does NOT cover, because both are sequence-level-KD-specific:
kept = [r for r in kept
        if not r["output"].strip().endswith(("...", "—", ","))      # truncated at max_new_tokens
        and r["output"].count("\n\n") <= len(r["output"].split()) / 20]  # boilerplate padding

# (c) Split by PROMPT, never by example — otherwise the same prompt appears in train and val.
prompts = sorted({r["instruction"] for r in kept})
random.seed(0); random.shuffle(prompts)
val_prompts = set(prompts[: max(1, len(prompts) // 20)])
train = [r for r in kept if r["instruction"] not in val_prompts]
val   = [r for r in kept if r["instruction"] in val_prompts]

for path, data in (("data/seqkd.train.jsonl", train), ("data/seqkd.val.jsonl", val)):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"kept {len(kept)}/{len(rows)}  train {len(train)}  val {len(val)}")
```

> **The teacher's flaws are the ones you must filter.** A teacher's *uncertain* answer arrives
> as a full-strength hard target, unlike a soft label — response distillation gives every
> sampled token weight 1.0 whether the teacher was certain or guessing (CS-09 §2.2). Verbosity,
> hedging, refusals and instruction-echoing all survive generation and are imitated faithfully.

### 5.3 The token-level KD loss, with the script's convention made explicit

```python
import torch, torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, ids, T=2.0, alpha=0.5, top_k=100):
    """Top-k KD, matching code/07_distillation.py's convention.

    alpha weights the HARD term. Hinton's alpha weights the SOFT term. This is inverted
    relative to CH-08 §2 — check which convention your code uses before passing a value.

    student_logits, teacher_logits : (seq, V)   [the script works on one sequence]
    ids                            : (seq,)    the corpus itself; no prompt masking needed
    """
    seq, V = student_logits.shape

    # Select with the TEACHER, then gather the SAME indices from the student, so both
    # distributions live on the same support. Selecting with the student would be a
    # different (and wrong) objective.
    top = torch.topk(teacher_logits, top_k, dim=-1)
    t_idx, t_vals = top.indices, top.values                      # (seq, k)
    s_vals = torch.gather(student_logits, -1, t_idx)              # (seq, k)

    t_soft = F.softmax(t_vals / T, dim=-1)                        # renormalised over top-k
    s_log  = F.log_softmax(s_vals / T, dim=-1)

    # reduction="batchmean" on a (seq, k) tensor divides by SEQ LEN, not by examples.
    # F.kl_div wants LOG-probabilities as `input` and PROBABILITIES as `target`.
    kd = F.kl_div(s_log, t_soft, reduction="batchmean") * (T * T)

    # Shifted next-token CE. No ignore_index: this is a raw corpus, not a prompt/response
    # pair, so every position is supervised. For an instruction set you MUST mask.
    hard = F.cross_entropy(student_logits[:-1].float(), ids[1:])

    return alpha * hard + (1.0 - alpha) * kd, hard.detach(), kd.detach()

# Sanity gate before you train: print both terms. If kd << hard * 0.1 the soft term is off
# (wrong alpha side, wrong reduction, or a quantised teacher) — code/07_distillation.py L267.
```

### 5.4 The teacher-size sweep — the one diagnostic almost nobody runs

```python
"""Three teachers, one student, everything else fixed. Costs three generation runs and
tells you whether you are left or right of the capacity-gap curve (CH-08 §4.6)."""
TEACHERS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct",   "7B"),
    ("Qwen/Qwen2.5-32B-Instruct",  "32B"),
]
for model_id, label in TEACHERS:
    print(f"python code/07_distillation.py --from-teacher --teacher {model_id} \\")
    print(f"    --prompts data/prompts.jsonl --n 5000 --out data/seqkd_{label}.jsonl")
print("# Then train the SAME student on each and compare on YOUR held-out set.")
print("# Report all three. 'The 70B must be better' is the assumption this sweep tests.")
```

### 5.5 Is token-level KD even worth it? Measure the teacher's entropy

```python
import torch, torch.nn.functional as F, math
logits = torch.load("teacher_batch.pt")          # any (B, L, V) tensor from YOUR teacher
p = F.softmax(logits.float(), dim=-1)
ent = -(p * torch.log(p + 1e-9)).sum(-1).mean()
print(f"mean entropy {ent:.3f} nats (uniform = {math.log(logits.size(-1)):.2f})")
# near 0   -> the teacher is effectively one-hot; its soft labels ARE hard labels.
#             Token-level KD buys you almost nothing over plain SFT.
# moderate -> real dark knowledge; the top-k cache and the extra plumbing are justified.
```

---

## 6. CLI Commands

```bash
# ── 0. The modes are MUTUALLY EXCLUSIVE and ONE IS REQUIRED. There is no --mode flag. ──
#    --from-teacher takes --prompts (a .jsonl of prompts)
#    --token-kd     takes --text    (a raw .txt corpus)  → sys.exit if missing
#    Passing the wrong one is the most common mistake.
python code/07_distillation.py --help

# ── 1. Sequence-level KD: plan, then generate ─────────────────────────────────────────
# --dry-run does not load any weights, so it plans a 32B teacher from a laptop.
python code/07_distillation.py --from-teacher --dry-run --prompts data/prompts.jsonl

python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl --dry-run
python code/07_distillation.py --from-teacher --teacher Qwen/Qwen2.5-32B-Instruct \
    --prompts data/prompts.jsonl --n 5000 --out data/seqkd.jsonl

# ── 2. Token-level KD: the vocab guard is THE go/no-go, and it runs before any weights ─
# It needs transformers installed even for --dry-run (real tokenizers), but not torch.
python code/07_distillation.py --token-kd --dry-run --text data/corpus.txt \
    --teacher Qwen/Qwen2.5-32B-Instruct --student Qwen/Qwen2.5-0.5B-Instruct \
    --size-hint 32B -T 4 --alpha 0.7 --top-k 20
# NOTE the alpha convention: on this script --alpha 0.7 = 70 % HARD / 30 % KD.
# CH-08 §4 states α as the weight on the SOFT loss. The two cards disagree; the script wins.
# NOTE also: -T is the KD temperature; --teacher-temp is the SAMPLING temperature. Unrelated.

# ── 3. Train the student on the generated data (this IS the distillation step) ─────────
# Filter first with quality_filter() — it is a FUNCTION, not a CLI (§5.2b); the script at
# code/data/make_instruction_data.py --help only offers --from-docs / --template.
python code/01_sft_lora.py --data data/seqkd.train.jsonl --output out/student
python code/09_merge_and_export.py --base <student-id> --adapter out/student --out out/student-merged

# ── 4. Vocab alignment, before you plan anything (the LLM-era precondition) ────────────
python -c "
from transformers import AutoTokenizer
T = 'meta-llama/Llama-3.3-70B-Instruct'; S = 'meta-llama/Llama-3.2-1B-Instruct'
tv, sv = AutoTokenizer.from_pretrained(T).get_vocab(), AutoTokenizer.from_pretrained(S).get_vocab()
shared = set(tv) & set(sv)
print('teacher', len(tv), 'student', len(sv), 'shared', len(shared))
print('usable fraction:', f'{len(shared)/len(tv):.1%}')
print('IDENTICAL' if tv == sv else 'MISMATCH -> token-level KD is UNDEFINED')
"

# ── 5. Size the plan ─────────────────────────────────────────────────────────────────
python code/common/memory.py --table        # the student-side VRAM reference (CH-13 §7)
```

**Reproduce the teacher-side VRAM numbers in §7 yourself:**

```bash
python -c "
import sys; sys.path.insert(0,'code')
from common.memory import inference_gb
for m in ['1.5B','7B','14B','32B','70B']:
    print(m, [round(inference_gb(m, 512, 1, b), 1) for b in (2, 1, 0.5)])
"
```

---

## 7. VRAM / Cost Calculator

### 7.1 The teacher, resident (for generation or for logit KD)

GiB, `code/common/memory.py:inference_gb(model, seq_len=512, batch=1)` — weights + KV cache
(512 tokens) + **1 GiB framework overhead**. Same GiB convention as CH-13 §7.

| Teacher | fp16 (2 B/param) | int8 (1 B) | int4 (0.5 B) | Fits on |
|---|---|---|---|---|
| 1.5B | **3.8** | 2.4 | 1.7 | any 8 GB card |
| 7B | **14.3** | 7.6 | 4.3 | 16 GB (fp16) / 8 GB (int8) |
| 8B | **16.0** | 8.5 | 4.7 | 24 GB (fp16) / 12 GB (int4) |
| 14B | **27.2** | 14.1 | 7.5 | 40 GB (fp16) / 16 GB (int4) |
| 32B | **60.7** | 30.9 | 15.9 | 2×48 GB (fp16) / 24 GB (int4) |
| 70B | **131.5** | 66.3 | 33.6 | 2×80 GB (fp16) / 1×48 GB (int4) |

**The sequence-level KD punchline:** for `--from-teacher` the teacher is a *separate,
offline* process, so this table is a *generation* budget, not a training budget. The training
run holds **only the student** — CH-13 §7's table applies unchanged.

### 7.2 The two-model case (token-level KD only)

| Item | Formula | 32B int8 teacher + 1.5B QLoRA student, seq 512 |
|---|---|---|
| Teacher weights | `P_t × bytes` | 29.8 GiB |
| Student (QLoRA) | CH-13 §7 | 1.1 GiB |
| Framework overhead | — | 1.0 GiB |
| **Full-vocab softmax buffers** | `3 × seq × V × 4 B`; V=151,936 → 311 MB/tensor | **+0.93 GiB** |
| **Top-k buffers** | `3 × seq × (k + V_cast)` with k=100 | **~0** |
| **Total** | | **~32 GiB** on one device |

The top-k truncation is not only a *storage* trick — it is what makes the two-model forward
pass fit. Without it you are holding three full-vocab fp32 tensors per sequence (CS-09 §4.4).

### 7.3 The storage wall — full logits vs top-k vs the text itself

For one 42.5M-token corpus (≈ 50k examples × 850 tokens):

| What you store | Per token | 42.5M tokens | vs the text |
|---|---|---|---|
| Raw text (UTF-8, ~4 B/token) | 4 B | **~170 MB** raw / ~250 MB as JSONL | 1× |
| Top-20 logits (8 B/entry) | 160 B | **6.8 GB** | ~40× |
| Top-100 logits (8 B/entry, the script's default) | 800 B | **34 GB** | ~200× |
| **Full vocab fp32 (V = 150k)** | **600 KB** | **25.5 TB** | **~150,000×** |

> **Why 8 bytes per entry and not 6.** The script's arithmetic (`a.top_k * 8`) charges 4 B for
> the value and 4 B for the index. CH-08 §7.1's table charges 6 B (fp16 value + int32 index),
> and CH-08 §5.2's code comment also says 8. Both conventions are in the handbook; **state
> which one you used** before comparing two storage estimates — the difference is 33 %.

### 7.4 Generation cost (sequence-level KD)

50k examples, 200 input + 600 output tokens each. Prices are the CS-09 §11.2 list prices
(verify before budgeting — they move).

| Scale | Tokens out | GPT-4o (k=1) | GPT-4o (k=3) | GPT-4o-mini (k=1) | R1 (k=1) | Self-hosted 70B (k=1) |
|---|---|---|---|---|---|---|
| 1k prompts | 0.6M | $6.50 | $19.50 | $0.39 | $1.42 | $0.58 |
| 10k prompts | 6M | $65 | $195 | $3.90 | $14.24 | $5.82 |
| **50k prompts** | **30M** | **$325** | **$975** | **$19.50** | **$71.20** | **$29.10** |
| 50k + two-stage judge | — | +$20 cheap over candidates +$100 strong over survivors | **≈ $1,095 total** | ≈ $85 | ≈ $214 | ≈ $87 |

| Stage | Cost driver | Note |
|---|---|---|
| Teacher generation | `k × (in·P_in + out·P_out)` | **The dominant term — 90 %+ of the bill** |
| Judge / reward model | runs on **candidates**, not survivors | Why the two-stage (cheap then strong) shape is standard |
| Student training | `6 × N × D` FLOPs | **0.8 % of the bill** (see §7.5) |
| Wall clock | 150k generations at 8-way concurrency | ~6 hours (CS-09 §11.3) |

**Self-hosting breakeven (CS-09 §11.2):** a 70B on 2×A100-80 at $7/hr, 2,000 out-tok/s → $0.97
per 1M output tokens, i.e. **≈ hosted API price — do not self-host a 70B to save money.** A 7B
on 1×A100 at $3.50/hr, 5,000 out-tok/s → $0.19 per 1M, 4.6× cheaper; self-host when the
teacher is ≤32B **or** utilisation exceeds ~50 %.

### 7.5 Student training cost (the part everyone over-estimates)

```
50k examples × 850 tokens = 42.5M tokens/epoch; 3 epochs = 127.5M tokens seen
FLOPs = 6 × N × D;  A100-80 bf16 peak 312 TFLOPS at 40 % MFU → 1.248e14 FLOP/s

1.5B student : 6 × 1.5e9 × 1.275e8 = 1.148e18 FLOPs → 2.6 A100-hours ≈  $9
8B   student : 6 × 8.0e9 × 1.275e8 = 6.120e18 FLOPs → 13.6 A100-hours ≈ $48
```

### 7.6 The honest comparison, and the breakeven you must compute first

| Source of the same 50k examples | Unit cost | Total | Wall clock |
|---|---|---|---|
| Expert human (8 min @ $25/hr loaded) | $3.33 | **$166,500** | ~12 weeks, 4 FTEs |
| Crowdworker (3 min @ $12/hr) | $0.60 | $30,000 | ~10 weeks |
| **Distilled from GPT-4o (k=3, two-stage judge) + training** | **$0.022** | **≈ $1,084** | **~6 h + 3 h** |
| Distilled from a self-hosted 70B | $0.0017 | ≈ $87 | ~25 h |

**30×–150× cheaper than human labelling, ~2,000× faster.** Then the breakeven, before you
start: `generation + engineering_hours × rate + training` vs `monthly_api_saving × months`.
$5,075 against a $2,000/month API bill → ~2.5 months. **Against a $200/month bill, the
engineering time never amortises — do not distil** (CS-09 §16.4).

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| Student is no better than plain SFT | Soft term effectively zero (token KD), or the teacher is not better than the data you had | Check `reduction` + `T*T`; train a hard-label baseline on your own data and compare (§12, check 6) |
| Output sounds exactly like the teacher and is wrong in the same way | **Style transferred, substance did not** | Judge win rate ↑ with flat task accuracy = style transfer. Add a *verifiable* filter and train on traces, not answers (CS-09 §12.5) |
| Student refuses far more than the teacher | Refusals were ~5 % of candidates and you kept them all | Count refusal prefixes in the dataset; rebalance or filter. Then re-run a safety stage on human data — distillation *thins* alignment |
| Student rambles, restates the question, writes headers | Verbosity and instruction-echoing copied faithfully | Filter by length + echo check; `completion_only_loss=True`; consider trimming boilerplate openings |
| Loss falls fast, generations are fluent but the model ignores the format | The student learned to generate the *prompt* too | Mask prompt tokens; verify by decoding a masked example (CH-13 §5.2) |
| Judge win rate 90 %, objective accuracy flat | Judge shares the teacher's family (self-preference), or length bias | Judge from a different family; position-swap; report a length-controlled number **and** a task metric (CS-09 §12.4) |
| Eval loss 0.02, generations reproduce dataset rows verbatim | Memorisation: 2.5k prompts × 3 epochs | More unique prompts; 1–2 epochs; hold out **prompts**, not examples (CS-09 §9.4 #12) |
| Disk fills during logit caching | Full-vocab fp32 storage | Top-k (§7.3); the full cache is ~150,000× the corpus |
| `RuntimeError: The size of tensor a (V1) must match … b (V2)` | Teacher/student vocabularies differ | You cannot logit-distil this pair. Sequence-level KD, or a same-family student (CH-08 §5.4) |
| KL is finite and falling but the student learns nothing | Same *size*, different *mapping* — the silent version | `assert tok_T.get_vocab() == tok_S.get_vocab()` |
| Soft loss NaN from step 1 | fp16 overflow in the softmax over 150k classes | Compute the softmax/log-softmax in **fp32** even when the model is fp16; clip grads at 1.0 |
| Student is great on teacher-domain prompts, poor elsewhere | Expected — you distilled a prompt distribution | Measure on both; add a held-out prompt cluster; monitor drift (CS-09 §16.5) |
| Everything is fine, then quality drops after 4 months | The traffic mix moved; the dataset did not | Drift monitor on input embeddings + a quarterly canary against the pinned teacher id |
| Student beats the teacher by 20+ points | The teacher was never fine-tuned for the task (the classic silent failure) | Measure teacher accuracy **before** generating. ≤ chance → stop (CS-09 §14.3) |
| Costs are 3× the estimate | Retries are billed; the judge ran on candidates; reasoning tokens bill as output | Log `usage` per call; two-stage judge; cap `max_tokens` |
| `openai.RateLimitError: 429` four hours in | No backoff, no concurrency cap | Exponential backoff + `max_workers=8`; checkpoint every 500 examples |

---

## 9. Comparison Matrix

### 9.1 The three KD modes, and which to reach for

| Dimension | Token-level (logit) KD | **Sequence-level KD** | Feature / hidden-state KD |
|---|---|---|---|
| Teacher requirement | Logits, **identical tokenizer mapping** | **Text output only** (API works) | Architectural access + a projection |
| Teacher at train time | Yes | **No** | Yes |
| Signal per position | ~V numbers | 1 token | a hidden vector |
| Implementation | Low (the loss is 6 lines) | Lowest (it *is* SFT) | High (you now train three things) |
| Scales to 100k+ examples | Only with offline top-k caching | **Trivially** | Hard |
| Quality vs sequence-level | +1–3 points *when it applies* | baseline | +1–2 points |
| Cross-tokenizer fix if you insist | ULD / MinED / DSKD — research-grade | n/a | DSKD |
| Failure mode | Silent vocab mismatch, OOM | Teacher errors, contamination, collapse | Projection misalignment |
| **2025–26 production share** | **~10 %** | **~85 %** | **<1 %** |
| Reach for it when | You already run both models, same family, and want the last 1–3 points | **Everything else — the default** | You are a lab with a capacity-gap problem |

### 9.2 KD vs SFT-on-hard-labels vs quantisation — the three "smaller/faster" routes

| | **Distillation (KD)** | **SFT on hard labels** | **Quantisation (CH-10 §9)** |
|---|---|---|---|
| What shrinks | **Parameters and FLOPs** | nothing (you pick the model) | **Bytes only** — same params, same FLOPs |
| Needs a teacher | ✅ | ❌ | ❌ |
| Needs training | ✅ (SFT) | ✅ (SFT) | PTQ: no. QAT: a short one |
| Needs labels | ❌ — manufactures them | ✅ | ❌ |
| Data cost | **$0.002–$0.022/example** (filtered synthetic) | $0.60–$3.33/example (human) | $0 — 128–512 calibration samples |
| Time to a result | hours of generation + hours of training | days of labelling + hours of training | **minutes to hours** |
| Quality ceiling | **the teacher** | your labels | the fp16 model (−0.05 Δppl at 4-bit) |
| Typical outcome at 7B | 1.5B student ≈ 80–90 % of teacher on the distilled task | task-specific, needs volume | 70B: 140 → 35 GB, near-lossless |
| Failure mode | inherits teacher errors; prompt-distribution-bound; collapse if recursed | overfits; needs volume | degrades on rare tokens/code/JSON at 3-bit and below |
| **Choose when** | You need fewer FLOPs, have no labels, and the teacher is strong | You have ≥5k good labels and only need a task model | You need a smaller *artefact* and can keep the architecture |
| Compose them? | Yes — **distil, then quantise the student** is the standard deployment recipe | — | Yes |

> **The row that decides most projects:** if you have ≥5k labelled examples, train the student on
> them directly and compare. Distillation's structural advantage is *manufacturing labels*; if
> you already have labels, you have paid for the expensive part already (CH-08 §12.3, check 6).

### 9.3 When NOT to distil (STOP conditions, CS-09 §8.2)

| STOP signal | Do this instead |
|---|---|
| Teacher accuracy on your task ≤ chance | Fine-tune the teacher first, or pick another teacher |
| < ~500 prompts and no way to get more | Hand-write a smaller, better set; few-shot; RAG |
| Classification with ≤50 labels and ≥5k labelled examples | Train a small encoder (CS-07), or quantise (CH-10) |
| Output must be verifiably correct and you have no verifier | Build the verifier first — you will otherwise distil confident errors |
| Your latency budget is met by quantisation alone | Quantise. It needs no training and no teacher |
| The teacher's ToS forbids competing-model training and no open-weights teacher will do | Use an open-weights teacher (CS-09 §16.7) |
| You need the student to be *safer* than the teacher | Distillation thins alignment. Re-run a safety stage afterwards |
| Monthly API bill < ~$200 | The engineering never amortises (CS-09 §16.4) |

---

## 10. Numbers To Memorize

| Number | Value | Context | Marker |
|---|---|---|---|
| Full logit storage, 150k vocab | **600 KB/token** | 42.5M tokens → 25.5 TB | documented |
| Top-20 logits, 8 B/entry | **160 B/token** | 5000× smaller than full fp32 (CH-08 §7.1) | documented |
| Softmax buffers, B=8/L=2048/V=128k | **8.41 GB/tensor**, ~25 GB for 3 | the real reason logit KD does not scale | documented (CS-09 §4.4) |
| Teacher VRAM, per param | **2 / 1 / 0.5 B** | fp16 / int8 / int4 — never quantise for logit KD | documented |
| KD temperature `T` (LLM) | **2 is the floor**, 2–4 typical | CH-08's 4–20 is the encoder-era range | RoT |
| `alpha` in the repo script | **weight on the HARD term** | inverted vs Hinton; default 0.5 | documented (L74) |
| `--top-k` default in the script | **100** | 20–100 is the useful range | documented |
| `--seq-len` default | **512** | one forward pass, prints the loss terms, does not train | documented (L237) |
| `--teacher-temp` default | **0.8** | SAMPLING temperature, unrelated to `-T` | documented |
| Response tokens for a 1.5–8B student | **20M–200M** | below → undertrained; above → reproduces surface form | RoT (CS-09 §7.2) |
| Student LR (full FT) | **1e-5…2e-5** | R1's distillation used this band | documented |
| Epochs on synthetic data | **2–3** | 5 epochs reproduces teacher phrasings verbatim | documented |
| Capacity-gap sweet spot | **2–10×** | teacher:student params | RoT |
| Capacity-gap danger | **>50×** | test a mid-size teacher too | RoT |
| Student floors | **~0.3B** chat format; **~1.5B** JSON/tools; **~1.5B** reasoning *if distilled* | below each, the named failure | documented (CS-09 §4.9) |
| Sizing law, 1–10B | **+4–6 MMLU per doubling** | plus +5–15 for a data-quality tier, one-time | RoT |
| R1-Distill-Qwen-32B vs direct RL | **72.6 vs 47.0** AIME 2024 | distillation beats RL on the small model | documented (CS-09 §15.4) |
| R1-Distill-Qwen-1.5B | **83.9 MATH-500** vs GPT-4o's 74.6 | the strongest "why distil" number | documented |
| R1 distillation data | **800k traces, SFT only, no logits** | across a tokenizer boundary | documented |
| 50k-example corpus cost | **≈$1,075–$1,095** (GPT-4o, k=3, two-stage judge) | + $9 of student training | documented (CS-09 §11.3) |
| Same corpus, human labels | **$166,500** | 30×–150× cheaper to distil | documented |
| Teacher training share | **0.8 %** of the pipeline cost | cost is a credit-card line item | documented |
| Judge family | **≠ teacher family** | self-preference bias is structural | documented |
| Vocab sizes to know | Llama-3 128,256 · Qwen2.5 151,936 · Phi-3 32,064 · CodeGen 51,200 | why cross-family logit KD is undefined | documented |
| Decontamination n-gram | **13-gram** | against *every* benchmark you will report | documented |
| Human replay in the mix | **≥5–10 %** | the anti-collapse rule | documented (CS-09 §4.8) |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `RuntimeError: The size of tensor a (128256) must match the size of tensor b (32064) at non-singleton dimension 2` | Llama-3.3-70B teacher vs Phi-3-mini student: `V_T ≠ V_S` | Sequence-level KD, or a same-family student |
| *(no error — the silent variant)* | Same `V`, different id→token mapping. The KL is finite, falling and meaningless | `assert tok_T.get_vocab() == tok_S.get_vocab()` |
| `NameError: name 'tokenizer' is not defined` | The notebook built the CE loss before any tokenizer existed (CS-09 §6.4 #1) | Use `student_tokenizer.pad_token_id`, or `ignore_index=-100` |
| `RuntimeError: Expected all tensors to be on the same device` | `device_map="auto"` put the teacher and student on different GPUs and you gathered the student at the teacher's indices | Force one device, or `.to(s_logits.device)` on the index tensor *before* `gather` |
| `ValueError: Expected input batch_size (…) to match target batch_size (…)` in `KLDivLoss` | Teacher and student were tokenised separately, so `T_t ≠ T_s` | Tokenise once, feed both models the same `input_ids`, slice `[:, :-1, :]` on both |
| `IndexError` slicing `t_logits[:, :-1, :]` vs `s_logits[:, 1:, :]` | One model got a prefix, the other a full sequence | Same `input_ids`, same slice |
| Soft loss is `nan` from step 1 | fp16 overflow in a 150k-way softmax, or `log(0)` on a zero-probability target | Do the softmax/log-softmax in **fp32**; keep the clamp; `clip_grad_norm_(1.0)` |
| Soft loss ≈ 1e-6 while hard loss ≈ 2.0 | `reduction="mean"` divided by `n·V` instead of `n` | `batchmean`, deliberately shaped (CH-08 §5.1) |
| `kl_div` returns a negative number | Probabilities passed where log-probabilities were expected | `input=log_softmax(...)`, `target=softmax(...)` |
| Soft loss constant at exactly `log(V)` | The teacher's distribution collapsed to uniform | Lower `T`; check a quantised teacher; confirm the teacher generates sane text |
| `torch.cuda.OutOfMemoryError` on the *second* `from_pretrained` | Both models loaded in fp32 — `from_pretrained` does not infer fp16 | `torch_dtype=torch.float16`. The phi-2/phi-1.5 pair is 16.0 GB in fp32 vs 8.0 GB in fp16 |
| `UserWarning: pad_token_id is not set` | Most Llama/Qwen tokenizers have no pad token | `tok.pad_token = tok.eos_token` — **and then** mask those pad positions out of the loss |
| Loss falls, generations are `!!!!!!` or whitespace | Training on the prompt tokens *and* the pads | `completion_only_loss=True`; `ignore_index = pad_token_id` |
| `AssertionError: Tokenizer mismatch` | The §6.5-style guard fired. It just saved you a wasted run | Pick a same-family pair, or switch to sequence-level KD |
| `TypeError: '<' not supported between instances of 'NoneType' and 'int'` in the scheduler | `num_training_steps` not passed | Compute the total step count first, including gradient accumulation (CH-13 §10) |
| `openai.RateLimitError: 429` mid-run | No backoff, no concurrency cap | Exponential backoff + `max_workers=8`; checkpoint every 500 examples |
| The student emits the teacher's *system prompt* | The system prompt was stored in the response field | Keep it separate; assert the response contains no prompt prefix |
| Two runs on the same prompt give different teacher output | `temperature > 0`, no seed, and the alias moved | Pin the **dated** teacher id per example; accept that API generation is not bit-reproducible |

---

## 12. Copy-Paste Starter Config

### 12.1 Run one — response distillation of a 1.5B student from a 32B teacher

```bash
# ── CHANGE THESE ────────────────────────────────────────────────────────────────────
TEACHER=Qwen/Qwen2.5-32B-Instruct       # a 32B, same family as the student: keeps the
STUDENT=Qwen/Qwen2.5-1.5B-Instruct      # token-level option open if you want it later
PROMPTS=data/prompts.jsonl              # REAL traffic sample, 1k-10k prompts
N=5000
# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────────────

# 0. Size the job. Loads no weights, so it runs on a laptop.
python code/07_distillation.py --from-teacher --dry-run \
    --teacher $TEACHER --prompts $PROMPTS --n $N --size-hint 32B --quant-bits 1.0

# 1. Generate 20. READ THEM. This is not optional and not a formality.
python code/07_distillation.py --from-teacher --teacher $TEACHER --prompts $PROMPTS \
    --n 20 --teacher-temp 0.8 --max-new-tokens 512 --out data/peek.jsonl

# 2. The full generation run (concurrency/retries: §5.3 of CS-09 for the API shape).
python code/07_distillation.py --from-teacher --teacher $TEACHER --prompts $PROMPTS \
    --n $N --teacher-temp 0.8 --max-new-tokens 512 --out data/seqkd.jsonl

# 3. Filter (dedup / refusal / echo / length — §5.2b), then split by PROMPT.
python -c "import sys; sys.path.insert(0,'code/data');
from make_instruction_data import quality_filter; import json
rows=[json.loads(l) for l in open('data/seqkd.jsonl',encoding='utf-8')]
kept,report=quality_filter(rows); print(len(kept),'/',len(rows),report)"

# 4. Train. Ordinary SFT — the prompt must be masked or you teach prompt generation.
python code/01_sft_lora.py --data data/seqkd.train.jsonl --output out/student \
    --model $STUDENT --epochs 2 --lr 2e-4 --lora-r 16 --lora-alpha 32 \
    --max-seq-len 2048 --quant 4bit

# 5. Merge (`--base` is REQUIRED — the adapter alone does not identify a model), then gate.
python code/09_merge_and_export.py --base $STUDENT --adapter out/student \
    --out out/student-merged --verify
```

The equivalent YAML if you prefer LLaMA-Factory (`template:` must match the student — CH-13 §8):

```yaml
# train.yaml — distilled SFT. Only the marked lines differ from CH-13 §12.
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset: seqkd                    # the teacher-generated set, registered in data/dataset_info.json
template: qwen                    # MUST match the student, not the teacher
cutoff_len: 2048
learning_rate: 2.0e-4
num_train_epochs: 2.0
train_on_prompt: false            # the whole point: response-only loss
# everything else identical to CH-13 §12
```

### 12.2 The six checks before you trust a distillation run

| # | Check | Pass condition | How |
|---|---|---|---|
| 1 | **The teacher is genuinely better** | Measured on your task, not assumed. ≤ chance → stop | teacher accuracy on a held-out set (CS-09 §14.3) |
| 2 | **Vocab alignment** (token-level only) | `get_vocab()` identical, or you *chose* sequence-level | §6, command 4 |
| 3 | **The teacher has non-trivial entropy** (token-level only) | Not near one-hot | §5.5 |
| 4 | **Reduction + `T²` + α side** | `batchmean` on a deliberately shaped tensor; `× T*T`; α's side stated | §2.1, §4.2 |
| 5 | **Capacity gap** | ≤ ~10×, **or** you ran the three-teacher sweep | §5.4 |
| 6 | **A supervised-SFT baseline** | You trained the student on your own labels too, same schedule, and distillation won | CH-08 §12.3, check 6 |
| 7 | **An objective metric next to any win rate** | Task accuracy + a judged number + the contamination overlap | CS-09 §12.1 |
| 8 | **Prompt-level holdout** | Val prompts are *disjoint*, not just val examples | §5.2(c) |

> Check 6 saves the most time and is skipped the most. Distillation is a way to manufacture
> labels; if you already have labels, its main advantage is gone.

### 12.3 The 20-generate-and-read discipline

The script refuses to let you skip it — `--dry-run` on `--from-teacher` prints, verbatim:

> *"Generate 20 first and READ them. Teacher generations that are repetitive, over-long, or
> refuse the prompt will be faithfully imitated. Distillation copies the teacher's flaws as
> reliably as its strengths — including its verbosity and its hedging."*

What to look for in the 20: over-long responses, hedging boilerplate ("It's important to
note…"), refusals on benign prompts, the instruction echoed back, markdown headers where you
want prose, language mixing, and truncation at `max_new_tokens`. Every one of those is a
training target at full weight.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The theory half — soft labels, `T²`, dark knowledge, the three KD modes | **CH-08 / CS-08 — Knowledge Distillation: Foundations** |
| The full applied treatment of every number on this card | **CS-09 — Distillation: LLM → SLM** |
| The SFT recipe the student actually trains with | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To compress without training at all (the rival route) | **CH-10 — Quantization**, **CS-10 / CS-11** |
| To know whether you need a small model in the first place | **CH-04 / CS-04 — Fine-Tuning vs RAG vs Agents** |
| The no-code path to the training run | **CH-15 / CS-15 — LLaMA-Factory** |
| To distil on a single consumer GPU | **CS-16 — Unsloth** |
| To go *beyond* the teacher's ceiling | **CH-14 / CS-14 — The Alignment Map** |
| The runnable implementation | `code/07_distillation.py` (both modes), `code/01_sft_lora.py` (step 3) |
| Practice being interviewed on this | **CS-09 §19 — Self-Check Questions**; worked answers in **`interview-questions/IQ-09-Distillation-LLM-to-SLM.md`** |

> **Settled — two errors this card used to report were fixed in the script, not in the card.**
> Both are kept here because the failure mode is the lesson, and because a card that
> quietly deletes its own corrections teaches you to trust the next one too much.
>
> **(1) α was inverted.** `code/07_distillation.py` originally defined the mixture as
> `L = α·hard_CE + (1−α)·T²·KL` and documented `--alpha` as *"weight on the HARD-label CE term"*,
> defaulting to 0.5 — the **opposite** of CH-08 §4's `L = α·T²·KL + (1−α)·CE` with `α = 0.7`.
> A reader who copied CH-08 §6's `--token-kd … --alpha 0.7` therefore trained 70 % hard labels
> and 30 % KD while believing the reverse, which is precisely the failure the script's own
> docstring warns about: *the run looks like it trains, but the student mostly just learns the
> hard labels.* The script now uses Hinton's convention (α weights the **soft** term, default
> **0.7**) and says so in the `--alpha` help text, the `DEFAULTS` comment, the `KD_MIX`
> print, and the docstring. **The generalisable rule: `alpha` is the single most
> convention-dependent name in the whole field — Hinton's α, the `alpha_soft` of CS-09 §6.3 and
> the LoRA `alpha` of CH-05 are three different quantities that all spell "alpha". Read the
> mixture expression, never the flag name.**
>
> **(2) The closing hint named a flag that does not exist.** The script printed
> `python 01_sft_lora.py --data {out_path} --out out/student`, but `01_sft_lora.py`'s argparse
> has no `--out` — the output-directory flag is `--output` (default `./out/sft-lora`). Copied
> verbatim, that line exits with `unrecognized arguments: --out`. The hint now prints
> `--output`. The data schema was never the problem: `07` writes Alpaca
> (`instruction`/`input`/`output`) and `01` defaults to `--format alpaca`.
>
> Both were found by *running the printed command rather than reading it* — which is the only
> way this class of defect is ever found. Prose and code drift in opposite directions: fixing
> the script makes the card stale, and fixing the card leaves the script lying.

> **Settled — a third correction, also fixed in the script.** `07_distillation.py`'s closing
> advice used to say *"Run them through the same quality filter as any synthetic data (see
> data/make_instruction_data.py)"*. That was a dead end: `quality_filter()` in
> `data/make_instruction_data.py` is real (7 rejection reasons, with a kill-rate report) but
> its argparse exposes only `--from-docs`, `--template`, `--out`, `--n`, `--provider`,
> `--model`, `--chunk-words` and `--seed` — **no `--filter` and no `--in`** — and the function
> is only ever called on data that script generated itself, in the same process. Pointing it
> at an existing file is not a supported operation. The script now says so outright.
>
> The lesson generalises past this file: **a function existing is not the same as a CLI
> existing.** Four of the five corrections on this card were of that shape — the prose named a
> capability the code had, but not a way to reach it. Before you write "run X with flag Y" into
> a card or a script, run X with flag Y. `--help` is the only authority.

> **Beyond the video:** the instructor's LLM demo only runs because `microsoft/phi-2` and
> `microsoft/phi-1_5` both use the **CodeGen tokenizer (V = 51,200)** — a fact the video never
> states (CS-09 §4.6). Every LLM example in that notebook is *accidentally* a same-vocabulary
> example, including the commented-out `Llama-2-7b-chat` → `TinyLlama` pair (both V = 32,000).
> The moment you try "distil phi-2 into TinyLlama" you meet the shape error and have no model
> of why. That single missing sentence is the difference between the BERT-era recipe and the
> LLM-era one.

> **Beyond the video:** the video presents the capacity gap not at all, and the sibling card
> (CH-08 §7.4) states it as a general law — *"for a very small student, distilling from a 70B
> teacher frequently loses to distilling from a 7B teacher."* That is **documented for logit
> KD** (Mirzadeh's TAKD; Cho & Hariharan's early-stopped-teacher result; CS-08 §4.6) and it
> should govern your token-level runs. For **sequence-level KD the 2025 evidence points the
> other way**: DeepSeek distilled a 671B MoE into a 1.5B student by SFT on 800k sampled traces
> and got 83.9 on MATH-500, a >400× parameter gap (CS-09 §15.4). The reconciliation is
> mechanical, not mysterious: sampling from `p_T` gives the student *high-probability
> trajectories only*, so it never has to represent the teacher's full distribution — the
> representational-mismatch mechanism that drives the capacity gap in logit KD is largely
> bypassed (CS-09 §4.3). The practical rule that follows: **sweep the teacher size on a
> verifiable narrow task rather than assuming either direction.** Note also that CS-09 §3's
> glossary row for "Teacher capacity gap" cross-references "§12" for the counter-evidence; §12
> is *Evaluation* — the counter-evidence is in §15.4 and §4.9.

**Line count:** see `README.md` module 09. If a number here and a number in CS-09 disagree by
~20–30 %, check **GiB vs GB** (1024³ vs 1000³, a 7.4 % difference), **bytes-per-param** (logit
entries at 6 B vs 8 B, a 33 % difference — §7.3), and **whether the teacher's weights are
included in the figure**, before assuming either is wrong.
