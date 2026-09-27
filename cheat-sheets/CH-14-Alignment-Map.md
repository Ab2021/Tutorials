# CH-14 — The Alignment Map: RLHF, PPO, DPO, ORPO Cheat Sheet

**One-line purpose:** move a model from "follows instructions" (SFT) to "prefers the answers
you actually want" — using *comparisons* rather than demonstrations.
**Use when:** the model already *can* do the task but does not reliably *choose* to do it the
way you want. Tone, helpfulness, refusal behaviour, conciseness, safety.
**Do NOT use when:** the model cannot do the task at all (that is SFT — CH-13), or the
problem is missing knowledge (that is RAG or continued pretraining — CH-04/CH-12), or you
have a verifiable reward and no preference data (that is GRPO — CS-26).

> **The one sentence that matters.** SFT says *"do this"*. Preference optimisation says
> *"between these two, this one"*. If you cannot point at a concrete pair where one answer is
> better, you do not have a preference-optimisation problem.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **SFT first, always.** | Preference methods refine a model; they cannot install a capability. |
| 2 | **DPO is the default.** PPO only if DPO plateaus or you need an online loop. | DPO deletes the reward model and the value model. |
| 3 | **β ≈ 0.1 for DPO.** | Controls how far you may drift from the reference model. |
| 4 | **DPO LR is ~10–100× lower than SFT LR.** | 5e-7…5e-6 full FT. Using the SFT LR destroys the model. |
| 5 | **Length bias is the #1 silent failure.** | If `chosen` is systematically longer, you trained "be verbose". |
| 6 | **Reward-model accuracy below ~70% is noise.** | Measure it. A bad RM makes PPO optimise garbage. |
| 7 | **DPO can lower the likelihood of *both* answers.** | "Likelihood displacement". Fix with IPO, or mix in an SFT loss (ORPO). |
| 8 | **ORPO needs one model. DPO needs two. PPO needs four.** | Memory decides your method more often than theory does. |
| 9 | **You need ~5k–100k pairs.** | Below ~1k, DPO overfits hard. |
| 10 | **Evaluate with a swapped-position judge.** | Otherwise you are measuring position bias, not quality. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Bradley–Terry** | `P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))` | r = reward | The assumption *all* of these methods rest on |
| **DPO loss** | `−log σ(β·log(π(y_w)/π_ref(y_w)) − β·log(π(y_l)/π_ref(y_l)))` | π = policy, ref = frozen SFT model | The whole method, in one line |
| **DPO implicit reward** | `r(x,y) = β·log(π(y\|x)/π_ref(y\|x))` | — | No reward model is trained; it is *implied* |
| **PPO clipped surrogate** | `min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)` | ρ = π_new/π_old, ε=0.2 | Clipping is what keeps PPO from exploding |
| **PPO total** | `E[surrogate] − c₁·KL(π‖π_ref) + c₂·H(π)` | c₁ ≈ 0.01–0.05 | The KL term is the leash |
| **KL penalty (per token)** | `log(π_ref/π) ` averaged | — | Reported as `kl` in PPO logs; watch it |
| **ORPO loss** | `L_SFT + λ·L_OR` | λ = odds-ratio weight | SFT and preference in one pass |
| **ORPO odds ratio** | `OR = (odds_θ(y_w) / odds_θ(y_l))` | `odds = p/(1−p)` | Length-normalised, so less length bias |
| **GRPO advantage** | `A_i = (r_i − mean(r)) / std(r)` | over a group of G samples | No value model needed |
| **SimPO implicit reward** | `(β/\|y\|)·log π(y)` | length-normalised | No reference model at all |
| **IPO loss** | `(log(π(y_w)/π_ref(y_w)) − log(π(y_l)/π_ref(y_l)) − 1/(2β))²` | — | Squared loss fixes DPO's overfitting |
| **KTO** | logistic on a *single* label | desirable / undesirable | No pairs needed |
| **Win rate** | `wins / (wins + losses)`, ties excluded | — | Run it BOTH ways round to remove position bias |

### 2.1 The four-model / two-model / one-model picture

```
PPO   [policy] [reference] [reward model] [value model]   ← 4 models, most VRAM
DPO   [policy] [reference]                                 ← 2 models
ORPO  [policy]                                             ← 1 model
GRPO  [policy] + a rule function                           ← 1 model, no RM
SimPO [policy]                                             ← 1 model, no RM, no ref
```

### 2.2 β, in plain terms

| β | Effect | When to pick it |
|---|---|---|
| 0.01 | Very loose — allows large drift from the reference | You want a big behaviour change and have lots of clean pairs |
| **0.1** | **The default** | Start here |
| 0.5 | Tight — stays close to the reference | Small dataset, or you fear degradation |
| >1 | Barely moves | Almost never right; you are just doing expensive nothing |

> **Intuition:** β is the price of moving away from the SFT model. Low β = cheap drift =
> more change and more risk. High β = expensive drift = safety and stagnation.

---

## 3. Decision Tree

```
Can the model already produce a CORRECT answer, just not reliably?
├─ No  → SFT first (CH-13). Preference methods cannot teach a missing skill.
└─ Yes ↓

Is the reward VERIFIABLE by a program (unit tests, a math checker, a regex)?
├─ Yes → GRPO (CS-26). No reward model, no pairs, no reference model needed.
└─ No ↓

Do you have PAIRS (this answer is better than that one)?
├─ No, only thumbs-up / thumbs-down on individuals
│     → KTO. Unpaired feedback is far cheaper to collect at scale.
└─ Yes ↓

Is your dataset LARGE and CLEAN (>10k pairs, low label noise)?
├─ Yes ↓
└─ No  → DPO will overfit. Prefer ORPO (SFT + preference in one pass) or
          a high β (0.3–0.5) with early stopping.

Do you have VRAM for a second full model?
├─ Yes → DPO with β=0.1. This is the answer most of the time.
│         └─ Plateauing? → PPO (needs 4 models and a real RM) or
│                          GRPO-style sampling with a reward function.
└─ No  → ORPO. One model, one pass, and it keeps the SFT signal alive.

Do you have an SFT-quality dataset as well as pairs?
├─ No  → ORPO lets you use both at once (its L_SFT term).
└─ Yes → run SFT then DPO. More control, two knobs, two evals.

Is your chosen answer systematically LONGER than the rejected one?
├─ Yes → you are about to train "be verbose". FIX THE DATA FIRST:
│        (a) rebalance lengths, or (b) use SimPO / length-normalised DPO,
│        or (c) use ORPO, whose odds ratio is length-normalised.
└─ No  → proceed.

Do you have a held-out preference eval with a swapped-position judge?
├─ No  → build it first. Otherwise win-rate numbers are measurement error.
└─ Yes → train.
```

---

## 4. Hyperparameter Quick Reference

| Param | DPO | ORPO | PPO | GRPO | Notes |
|---|---|---|---|---|---|
| `learning_rate` | **5e-7 … 5e-6** (full FT); 1e-5 … 1e-4 (LoRA) | 5e-7 … 1e-6 | **1e-6 … 3e-6** | 1e-6 | **All far below SFT LRs.** This is the most common mistake. |
| `beta` (β / λ) | **0.1** | **0.1** (the λ) | — | — | 0.01–0.5 sweep. |
| `num_train_epochs` | **1–2** | 1–3 | 1–4 PPO epochs *per batch* | 1 | DPO overfits past 2 fast. |
| `per_device_train_batch_size` | 2–8 | 2–8 | 1–4 | 1–8 | PPO needs rollout memory on top. |
| `gradient_accumulation_steps` | 4–16 | 4–16 | 4–16 | 4–16 | Effective batch 32–128. |
| `max_length` / `max_prompt_length` | 1024 / 512 | 1024 | 512 / 256 | 1024 | Longer = much more memory (O(L²)). |
| `kl_coef` (PPO) | — | — | **0.01–0.05** | 0.001–0.04 | The leash on the policy. |
| `cliprange` (PPO) | — | — | **0.2** | 0.2 | Standard. |
| `gamma` / `lam` (PPO) | — | — | 1.0 / 0.95 | — | Standard. |
| `num_generations` (GRPO) | — | — | — | **4–16** | Group size; more = lower-variance advantage, more compute. |
| `loss_type` (DPO) | `sigmoid` | — | — | — | Also `ipo`, `hinge`, `bco_pair`, `orpo`. |
| `rpo_alpha` (DPO) | 0–1 | — | — | — | Mixes an SFT loss into DPO. `1.0` is a strong regulariser. |
| `label_smoothing` | 0–0.1 | — | — | — | Helps with noisy preference labels. |
| `precompute_ref_log_probs` | `True` | n/a | n/a | n/a | Big speed + memory win; reference is frozen anyway. |
| `reference_free` | — | ✅ always | — | — | ORPO/SimPO structurally have no reference model. |

### 4.1 Preference dataset sizing

| Pairs | What to expect |
|---|---|
| < 500 | DPO overfits; use ORPO or a large β and stop early |
| 1k – 5k | Workable for a narrow, well-defined behaviour |
| **5k – 50k** | **The practical sweet spot** |
| 50k – 200k | Diminishing returns unless the pairs are genuinely diverse |
| > 200k | Useful only if label noise is low; otherwise you amplify the noise |

### 4.2 The reward model (only needed for PPO)

| Property | Guidance |
|---|---|
| Base | The SFT model, with a scalar head replacing the LM head |
| Data | ~10k–100k *comparisons* (more than the policy pairs — the RM is the bottleneck) |
| Accuracy target | **>70%** on held-out pairs. 50% = a coin flip = a useless RM. 60% is marginal. |
| Loss | Bradley–Terry: `−log σ(r_w − r_l)` |
| Failure mode | Reward hacking — the policy finds the RM's blind spot, and *reported reward keeps rising while real quality falls* |

---

## 5. Copy-Paste Code Snippets

### 5.1 DPO with TRL (the default path)

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

# The dataset needs exactly three fields: prompt / chosen / rejected
ds = load_dataset("json", data_files="data/sample_preference.jsonl", split="train")

trainer = DPOTrainer(
    model=model,
    ref_model=None,          # None + PEFT => the adapter is disabled for the ref pass.
                             # This is the "reference model memory trick": you get the
                             # reference distribution for free, at zero extra VRAM.
    args=DPOConfig(
        output_dir="out/dpo",
        beta=0.1,                       # the knob that matters most
        learning_rate=5e-6,             # LOW. Not 2e-4.
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=1024,
        max_prompt_length=512,
        bf16=True,
        gradient_checkpointing=True,
        precompute_ref_log_probs=True,  # reference is frozen — compute it once
        loss_type="sigmoid",            # try "ipo" if you see likelihood displacement
        report_to="none",
    ),
    train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                           task_type="CAUSAL_LM", target_modules="all-linear"),
)
trainer.train()
trainer.save_model("out/dpo")
```

> **`ref_model=None` with a PEFT config** is not a shortcut, it is the correct choice. TRL
> disables the adapter to recover the base model's log-probs, so the reference is the frozen
> base — exactly what DPO wants — with no second copy of the weights in memory.

### 5.2 ORPO (no reference model, one pass)

```python
from trl import ORPOConfig, ORPOTrainer

trainer = ORPOTrainer(
    model=model,
    args=ORPOConfig(
        output_dir="out/orpo",
        beta=0.1,               # this is lambda (λ) in the ORPO paper
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_length=1024,
        max_prompt_length=512,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    ),
    train_dataset=ds,           # same prompt/chosen/rejected shape
)
trainer.train()
```

### 5.3 GRPO with a verifiable reward (no reward model)

```python
import re
from trl import GRPOConfig, GRPOTrainer

def correctness_reward(completions, answer, **kwargs):
    """A RULE, not a model. This is the whole point of GRPO."""
    rewards = []
    for c, a in zip(completions, answer):
        m = re.search(r"####\s*(-?[\d,]+)", c)   # the GSM8K answer format
        got = m.group(1).replace(",", "") if m else None
        rewards.append(1.0 if got == a.replace(",", "") else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    return [0.5 if re.search(r"####\s*-?[\d,]+", c) else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward, format_reward],
    args=GRPOConfig(
        output_dir="out/grpo",
        learning_rate=1e-6,
        num_generations=8,            # G — the group that replaces the value model
        max_completion_length=512,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    ),
    train_dataset=ds,                 # needs a 'prompt' and an 'answer' column
)
trainer.train()
```

### 5.4 Auditing your preference pairs for length bias (do this first)

```python
import json, statistics as st

rows = [json.loads(l) for l in open("data/sample_preference.jsonl", encoding="utf-8")]
gaps = []
for r in rows:
    cw = len(r["chosen"].split())
    rw = len(r["rejected"].split())
    gaps.append(100 * (rw / max(cw, 1) - 1))

print(f"pairs                 {len(rows)}")
print(f"median length gap     {st.median(gaps):+.1f}%   "
      f"(positive = rejected is longer)")
print(f"mean length gap       {st.mean(gaps):+.1f}%")
longer_chosen = sum(1 for r in rows if len(r['chosen'].split()) > len(r['rejected'].split()))
print(f"chosen is longer in   {longer_chosen}/{len(rows)} = "
      f"{longer_chosen/len(rows):.0%}")
print()
print("If 'chosen is longer' is above ~65%, you are training verbosity, not quality,")
print("regardless of what your win-rate eval says. Fix the data, not the hyperparameters.")
```

> This is exactly the check `code/data/make_preference_data.py --only-neutral` automates.
> A symmetric tolerance of ±25% is what the handbook's generator enforces by default.

### 5.5 A position-bias-corrected win-rate judge

```python
def win_rate(judge_fn, items, model_a_key="model_a", model_b_key="model_b"):
    """items: [{'prompt','model_a','model_b'}]   judge_fn(rendered_prompt) -> 'A'|'B'|'TIE'

    Run every pair TWICE with the answers SWAPPED. A is only credited with a win
    if it wins in BOTH orderings; a split result counts as a TIE. This is the
    strict version: a judge with first-position bias that always answers "A"
    yields all ties and win_rate 0.5, rather than a fake 100% win rate.
    """
    wins = losses = ties = 0
    for it in items:
        p, a, b = it["prompt"], it[model_a_key], it[model_b_key]

        fwd = judge_fn(RENDER.format(prompt=p, a=a, b=b)).strip().upper()[:3]
        rev = judge_fn(RENDER.format(prompt=p, a=b, b=a)).strip().upper()[:3]

        # Map reverse-order answers back onto the original assignment.
        rev = {"A": "B", "B": "A", "TIE": "TIE"}.get(rev, "TIE")

        if fwd == rev == "A":
            wins += 1                 # agreed, in both orders
        elif fwd == rev == "B":
            losses += 1
        else:
            ties += 1                 # disagreement, or an explicit tie

    decided = wins + losses
    return {"win_rate": wins / decided if decided else 0.5,
            "wins": wins, "losses": losses, "ties": ties,
            "tie_rate": ties / len(items)}
```

> **Why the strict rule matters.** The permissive version — score each ordering
> independently and count a split as one win plus one loss — gives nearly the same headline
> number but hides *how much* the judge flipped. The strict version turns every flip into a
> tie, so `tie_rate` becomes a direct readout of judge noise. High `tie_rate` means you are
> measuring position bias, not quality.
>
> This mirrors `code/common/eval_utils.py` exactly (`win_rate(judge_fn, items, ...)`, with
> the judge receiving the fully rendered prompt).

> **A note on the sample fixture.** Run `python code/04_dpo.py --dry-run
> --data code/data/sample_preference.jsonl` and the script prints the pair-level length gap
> itself: `mean words chosen 67 rejected 62 (+7.5% length gap)`. That number being small is
> the whole point of the generator's length-matching — the same check on naively-built pairs
> routinely shows ±60–160%.

---

## 6. CLI Commands

```bash
# ── TRL ─────────────────────────────────────────────────────────────────────
python -m trl.scripts.dpo --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --dataset_name my_prefs --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \
  --max_length 1024 --max_prompt_length 512 --bf16 --use_peft --lora_r 16

# ── This handbook's scripts ─────────────────────────────────────────────────
python code/04_dpo.py  --dry-run --data code/data/sample_preference.jsonl
python code/05_orpo.py --dry-run --data code/data/sample_preference.jsonl
python code/06_grpo.py --dry-run --data code/data/toy_grpo.jsonl

# ── LLaMA-Factory ───────────────────────────────────────────────────────────
llamafactory-cli train dpo.yaml        # stage: dpo
llamafactory-cli train orpo.yaml       # stage: orpo
llamafactory-cli train ppo.yaml        # stage: ppo  (needs a reward model path)

# ── Make preference data (with length-bias control) ─────────────────────────
python code/data/make_preference_data.py --template --out code/data/prefs.jsonl
python code/data/make_preference_data.py --template --only-neutral   # filter biased pairs
python code/data/make_preference_data.py --template --no-match-lengths  # show the raw bias

# ── Inspect whether a run drifted too far ───────────────────────────────────
python -c "
import json
for l in open('out/dpo/trainer_state.json'):
    s=json.loads(l)
    if 'log_history' in s:
        for e in s['log_history']:
            if 'rewards/chosen' in e:
                print(e['step'], e['rewards/chosen'], e['rewards/rejected'],
                      e.get('kl'))
        break
"
# What you want to see: rewards/chosen RISING, rewards/rejected FALLING,
# and the GAP widening. If both fall, that is likelihood displacement.
```

---

## 7. VRAM / Cost Calculator

### 7.1 The model count is the cost

| Method | Models in memory | Relative VRAM (7B, bf16) | Notes |
|---|---|---|---|
| ORPO | 1 | ~16 GB weights + activations | No reference model |
| SimPO | 1 | ~16 GB | No reference model |
| DPO | 2 (policy + ref) | ~32 GB | Or **~16 GB** with the PEFT trick (`ref_model=None`) |
| PPO | 4 (policy, ref, RM, value) | ~48–64 GB | Plus rollout buffers |
| GRPO | 1–2 | ~16–32 GB | No value model; RM is a *function*, so free |

### 7.2 Adding preference tuning on top of an SFT plan

| Base plan | + DPO (PEFT ref trick) | + ORPO | + PPO |
|---|---|---|---|
| QLoRA 7B SFT: ~5 GB | ~6 GB | ~5 GB | ~20 GB+ |
| LoRA 7B SFT: ~15 GB | ~16 GB | ~15 GB | ~35 GB+ |
| Full FT 7B: ~92 GB | ~110 GB (two fp32 copies) | ~92 GB | ~160 GB+ |

> **The practical rule:** preference tuning with LoRA costs barely more than the SFT run it
> follows. PPO is the only one that forces you onto a bigger box.

### 7.3 Data cost

| Source | Cost per 1k pairs | Quality | Notes |
|---|---|---|---|
| Human annotation | $500 – $5,000 | Highest | 5–20 min per pair; needs clear guidelines |
| LLM judge (strong model) | $1 – $20 | Good if the judge is much stronger than the policy | Watch for the judge's own length bias |
| Frontier-model outputs vs base outputs | ~$0 | Often too easy — the model learns nothing | The pairs must be *close* |
| Rejection sampling from your own model | compute only | Good for GRPO-style; needs a reward function | Best pairs come from your model's own near-misses |

> **The pair-difficulty point.** If `chosen` is obviously better than `rejected`, the gradient
> is near zero and you learn nothing. The useful pairs are the ones where a careful reader has
> to think. Easy pairs inflate your dataset size and teach almost nothing.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| Outputs got much longer after DPO | **Length bias in the pairs** | Rebalance pair lengths; check the §5.4 audit. |
| Outputs got much shorter/terser | Length bias in the other direction | Same audit, symmetric tolerance. The generator's `--only-neutral` filter is symmetric for this reason. |
| Both `rewards/chosen` and `rewards/rejected` falling | **Likelihood displacement** | Raise β; switch `loss_type="ipo"`; add `rpo_alpha=1.0`; or use ORPO. |
| Loss goes to ~0 immediately at epoch 1 | Pairs are trivially separable (or duplicated) | Harder pairs; check for exact duplicate prompts. |
| Model becomes repetitive / degenerate | β too low, LR too high, or too many epochs | Raise β to 0.3, cut LR 5×, 1 epoch. |
| Model loses general ability | Over-optimised away from the reference | Raise β; mix general instruction data into the pairs. |
| Win rate stuck at exactly 50% with high tie rate | Your judge picks a position | Swap positions and average — §5.5. |
| Win rate 95%+ | Your eval set is too easy, or leaked into training | Build a harder held-out set; decontaminate (`eval_utils`). |
| PPO `kl` climbing past ~10 and never recovering | KL coefficient too low | Raise `kl_coef`; lower LR; shorten PPO epochs per batch. |
| PPO reward rising but samples getting worse | **Reward hacking** | The RM is being exploited. Retrain/ensemble the RM, raise the KL penalty, add a length penalty. |
| RM accuracy ≈ 50% | The RM learned nothing | Check label balance and pair quality; the RM data is likely too easy or too noisy. |
| `expected a list of length 2` / shape error | `chosen`/`rejected` are not strings | Confirm the file has plain strings, not nested message dicts. |
| OOM in DPO but not SFT | Two models are resident | Use `ref_model=None` + PEFT; or `precompute_ref_log_probs=True`. |
| Loss NaN in PPO | LR too high, or fp16 overflow | LR ≤ 1e-6; bf16; gradient clipping at 1.0. |
| GRPO reward never moves | Reward function returns a constant | Print the reward distribution across the group; if `std=0`, the advantage is 0 and there is no gradient. |
| Adapter merged, quality dropped | Merged in fp16 and lost precision | Merge in bf16/fp32, then quantise if needed. |

> **The GRPO constant-reward trap** deserves emphasis: GRPO's advantage is
> `(r − mean) / std` *within a group*. If every sample in the group gets the same reward —
> all right, or all wrong — `std = 0` and the advantage is 0. **No gradient, no learning,
> and a loss that looks perfectly calm.** Always log the per-group reward spread.

---

## 9. Comparison Matrix

| Method | Ref model | Reward model | Data shape | Stages | Best for |
|---|---|---|---|---|---|
| **RLHF / PPO** | ✅ | ✅ | comparisons → RM, then prompts | 3 | Max quality, online loop, big budget |
| **DPO** | ✅ | ❌ | (prompt, chosen, rejected) | 2 | **The default** |
| **IPO** | ✅ | ❌ | pairs | 2 | When DPO overfits |
| **ORPO** | ❌ | ❌ | pairs (+ SFT data) | **1** | Low VRAM; one-pass SFT + preference |
| **SimPO** | ❌ | ❌ | pairs | 2 | No reference; length-normalised |
| **KTO** | ✅ | ❌ | **unpaired** binary labels | 2 | Cheap feedback at scale |
| **GRPO** | ⚠️ optional | ❌ (a rule) | prompts + verifier | 2 | Math, code, anything checkable |
| **RLAIF** | ✅ | ✅ (an LLM judge) | prompts | 3 | No human labellers available |

| Dimension | PPO | DPO | ORPO | GRPO |
|---|---|---|---|---|
| Models in memory | 4 | 2 | **1** | 1–2 |
| Hyperparameter sensitivity | **Very high** | Medium | Low | Medium |
| Training stability | Poor — the classic complaint | Good | Good | Good |
| Needs online generation | ✅ (expensive) | ❌ | ❌ | ✅ |
| Can exceed the demonstration ceiling | ✅ (explores) | ❌ | ❌ | ✅ |
| Implementation complexity | High | **Low** | Low | Medium |
| Reported in papers as the SOTA baseline | Historically | Widely | Sometimes | Recent reasoning work |

**How to read the "exceeds the demonstration ceiling" row** — this is the real theoretical
difference. PPO and GRPO *sample* from the policy and score those samples, so they can find
answers better than anything in the training data. DPO/ORPO/SimPO only reweight what is
already in your pairs: they cannot invent a better answer, only make the good ones likelier.
For reasoning tasks where a correct chain of thought may not exist in your data, that
distinction is decisive.

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| β (DPO) | **0.1** | Default; sweep 0.01–0.5 |
| λ (ORPO) | **0.1** | In TRL this is the `beta` argument |
| DPO LR (full FT) | **5e-7 … 5e-6** | ~10–100× below SFT |
| DPO LR (LoRA) | **1e-5 … 1e-4** | |
| PPO LR | **1e-6 … 3e-6** | Lower still |
| PPO `kl_coef` | **0.01–0.05** | Watch the logged KL |
| PPO `cliprange` | **0.2** | Standard |
| PPO `gamma` / `lam` | **1.0 / 0.95** | Standard |
| GRPO `num_generations` | **4–16** | Group size G |
| DPO epochs | **1–2** | Overfits past 2 |
| Pairs needed | **5k – 100k** | Sweet spot |
| RM accuracy floor | **>70%** | Below = noise |
| RM training pairs | **10k – 100k** | More than the policy pairs |
| Bradley–Terry | `σ(r_w − r_l)` | The shared assumption |
| Tie rate alarm | **>30%** | Judge likely position-biased |
| Likelihood-displacement fix | `loss_type="ipo"` / `rpo_alpha=1.0` | |
| Length-gap tolerance | **±25%** | The handbook generator's default |
| Reference-model VRAM saving | **~50%** | `ref_model=None` + PEFT |
| Typical β intuition | price of drift | low = cheap drift = more change |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `KeyError: 'chosen'` / `'rejected'` | Field names wrong | DPO needs `prompt`/`chosen`/`rejected` (or a conversational form) |
| `ValueError: You cannot use a reward model and a reference model` | Conflicting args | Pick one path |
| `TypeError: 'NoneType' object is not subscriptable` in the ref pass | `ref_model=None` but no PEFT config | Either supply a ref model or a `peft_config` |
| `RuntimeError: CUDA out of memory` only in DPO | Two models resident | `ref_model=None`, `precompute_ref_log_probs=True` |
| `AssertionError: beta must be > 0` | β = 0 | β > 0 always; 0.1 to start |
| `ValueError: max_prompt_length must be < max_length` | | Set prompt ≤ length/2 |
| `ImportError: cannot import name 'DPOConfig'` | TRL version | `pip install -U trl` |
| `UserWarning: Some weights of ... were not initialized` | Expected if you add an RM head | Harmless for an RM; not for a policy |
| `IndexError: index out of range` in the collator | A row's prompt exceeds `max_prompt_length` | Filter or raise the limit — silently truncated prompts are worse |
| `torch.distributed` errors in PPO | PPO expects a specific topology | Use the TRL example config rather than hand-rolling |
| `AssertionError` on `num_generations` | G must divide the effective batch | `per_device_batch × accum` must be a multiple of G |
| `reward is constant, advantage is zero` (your own log) | GRPO with no variance | See §8 — the task is too easy or too hard for the current policy |

---

## 12. Copy-Paste Starter Config

`dpo.yaml` — LLaMA-Factory, after a completed SFT run.

```yaml
# ── CHANGE THESE ────────────────────────────────────────────────────────────
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
adapter_name_or_path: out/sft-lora     # START FROM YOUR SFT ADAPTER, not the base
dataset: my_prefs
output_dir: out/dpo
template: llama3
# ── CHANGE NOTHING BELOW FOR RUN ONE ────────────────────────────────────────
stage: dpo
do_train: true

finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

pref_beta: 0.1                 # β
pref_loss: sigmoid             # or 'ipo' if you see likelihood displacement

cutoff_len: 1024
pref_batch_size: 2
gradient_accumulation_steps: 8

learning_rate: 5.0e-6          # LOW. Two orders of magnitude below SFT.
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

val_size: 0.05
eval_strategy: epoch
logging_steps: 5
save_strategy: epoch
report_to: none
```

### The five checks before you trust a preference run

| # | Check | Pass condition |
|---|---|---|
| 1 | **Start from the SFT model**, not the base | `adapter_name_or_path` points at your SFT output |
| 2 | **Length-bias audit on the pairs** | chosen-is-longer < ~65% (§5.4) |
| 3 | **`rewards/chosen` up AND `rewards/rejected` down** | Both moving, gap widening |
| 4 | **Win rate with swapped positions** | Reported alongside the tie rate (§5.5) |
| 5 | **General-ability regression test** | A few general prompts still answered well |

> Check 5 is the one people skip and the one that matters most in production. A model that
> wins your preference eval while becoming unable to follow an unrelated instruction is a
> **regression**, and your preference eval will never tell you.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full treatment with maths and failure modes | **CS-14 — The Alignment Map** |
| RL fundamentals and PPO specifically | **CS-24 — RL Fundamentals & RLHF with PPO** |
| DPO in depth | **CS-25 — Direct Preference Optimization** |
| The verifiable-reward path | **CS-26 — GRPO** |
| The single-stage path | **CS-27 — ORPO** |
| How to build the pairs | `code/data/make_preference_data.py` (it enforces the length control) |
| How to evaluate without fooling yourself | `code/common/eval_utils.py` (position bias, reward hacking, bootstrap CIs) |
| What SFT has to do first | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| Whether you need any of this | **CH-04 / CS-04 — Fine-Tuning vs RAG vs Agents** |
| Practice being interviewed on this | **IQ-14 — Interview Questions: The Alignment Map** |

