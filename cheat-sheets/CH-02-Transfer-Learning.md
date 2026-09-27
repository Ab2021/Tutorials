# CH-02 — Transfer Learning Cheat Sheet

**One-line purpose:** pick the right adaptation strategy (probe / freeze / partial / full FT / PEFT), the right learning rate, and the right anti-forgetting controls — in under five minutes.
**Use when:** you have a pretrained checkpoint and a target dataset and you are about to spend GPU money.
**Do NOT use when:** the task is already solved by prompting (Q1), the knowledge you need is factual and volatile (use RAG, CS-04), or the input modality differs from the pretrained one (no adaptation strategy fixes that).

---

## 1. The 10-Second Summary

| Fact | Value |
|---|---|
| Transfer learning vs fine-tuning | Strategy vs tactic. Two sides of one coin — you transfer *by* fine-tuning; zero-shot inference is transfer with no fine-tuning |
| The universal default | Touch the top, leave the bottom alone. Early layers = primitive features; late layers = task-specific features |
| **The number** | **Fine-tuning LRs are 10–100× smaller than pretraining LRs** — 1e-5–5e-5 full FT vs 1e-4–6e-4 pretraining |
| The baseline you must run | Linear probe: freeze 100% of the backbone, train one linear layer. ~70 s, often within 2 points of the best config, often *best* OOD |
| The failure you cannot see | Catastrophic forgetting. Target metric rises while general capability falls. Instrument KL-to-base or ship blind |
| The cheapest real fix for forgetting | Replay 1–10% general-domain data (5% default) into every batch |
| Memory | Full FT = 16 bytes/trainable param; 7B ≈ 112 GB. QLoRA 7B ≈ 6–8 GB |
| Decision rule | <500 labels → probe. 500–5k → LP-FT or LoRA. 5k–50k → full FT + LLRD or LoRA r=32–64. ≥3B params → LoRA/QLoRA, always |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Full-FT memory | `bytes = 16 × N_trainable` | 2 bf16 w + 2 bf16 g + 8 fp32 Adam m,v + 4 fp32 master | 7e9 × 16 = **112 GB** + activations |
| Loss at init | `L₀ ≈ ln(k)` | `k` = `num_labels` | 3-class ⇒ 1.099; flat at 1.386 ⇒ `num_labels=4` bug |
| Forgetting drift | `Δ ∝ η · ‖g‖ · t` | `η` LR, `t` optimizer steps | 3 ep × 9,749 ex / bs 16 = 1,828 steps of pure target signal |
| LLRD | `η_layer = η_base · decay^(L−1−i)` | `decay` 0.8–0.9 (ULMFiT: 1/2.6) | `2e-5 × 0.8^12 ≈ 1.4e-6` embeddings, head `4e-5` |
| STLR | warmup to `η` over `cut_frac=0.1`, then decay to `η/32` | `ratio=32` | 2,000 steps ⇒ 200 warmup, 1,800 decay |
| LoRA update | `W' = W + (α/r)·BA` | `A` random, `B=0` ⇒ `ΔW=0` at step 0 | `r=16, α=32` ⇒ effective scale 2.0 |
| L2-SP | `L + (λ/2)·‖θ − θ*‖²` | `λ ∈ [1e-4, 1e-2]` | Toward `θ*`, **not** toward zero |
| EWC | `L + (λ/2)·Σ Fᵢ(θᵢ − θ*ᵢ)²` | `F` = diagonal Fisher | `λ ∈ [1e2, 1e4]` — Fisher values are tiny |
| KL-to-base | `L_task + β·KL(p_base ‖ p_θ)` | `β ∈ [0.01, 0.5]` SFT | Needs base resident or cached logits |
| Effective batch | `bs × grad_accum × n_gpu` | — | bs 4 × accum 8 = 32 |
| Activation memory | `bs × seq × hidden × layers × ~16 B` | checkpointing off | Doubling `max_len` quadruples attention cost |
| Data floor (head-only) | `10 × n_classes` | — | 3 classes ⇒ 30 examples minimum |

---

## 3. Decision Tree — Feature Extraction vs Fine-Tuning

```
START: pretrained checkpoint + target dataset.
0. BASELINE?  No -> (a) zero-shot/5-shot prompt (b) majority class (c) TF-IDF+SVM.
   Three numbers before any GPU is rented. ................................. [STOP]
1. QUADRANT?  (domain same/diff x task same/diff)
   Q1 same/same -> head only, or just calibrate. 100-2,000 ex.
   Q2 same/diff -> head + last 1-4 blocks, or LoRA. 1k-100k ex.
   Q3 diff/same -> continued pretraining on unlabeled target text FIRST, then head
                   + upper blocks. 10k-1B unlabeled tokens.
   Q4 diff/diff -> continued pretraining + head + upper blocks, or PEFT (bigger rank).
2. LABELS?    <500 -> linear probe (full FT memorizes 200 examples)
              500-5,000 -> LP-FT, or LoRA r=8-16
              5,000-50,000 -> full FT + LLRD + early stop, or LoRA r=32-64
              >100,000 in-dist -> full FT; >1M -> from-scratch is competitive
                                  (He 2019: pretraining buys speed, not ceiling)
3. MODEL SIZE?  >=3B -> LoRA/QLoRA, always (memory decides, not quality).
                <3B  -> either; on <=5k examples PEFT often wins anyway.
4. TARGET OOD vs the base's pretraining?  Yes -> LP-FT or a linear probe. Full FT
   distorts features and can LOSE to a frozen probe (Kumar 2022).
5. DOMAIN LANGUAGE differs (legal/clinical/code/new tokenizer)?  Yes -> continued
   pretraining is the first move regardless of label count — labels cannot fix
   features built on the wrong distribution.
6. FACTUAL + VOLATILE need (a catalog, a policy)?  Yes -> STOP. Use retrieval (CS-04)
   or 1B+ tokens of continued pretraining. SFT teaches behavior, not facts.
7. LATENCY <20 ms?  Yes -> distill to a smaller model or an encoder. Fine-tuning
   does not make anything faster.
8. MULTI-TENANT (N behaviors, one GPU)?  Yes -> one LoRA adapter per tenant,
   UNMERGED, vLLM --enable-lora. r and target_modules are fixed at train time.
ALWAYS: 3 x 20-minute experiments before committing — (a) prompt baseline, (b) linear
probe, (c) LoRA r=16. Only if (c) >> (b) AND you have >10k in-distribution labels
should you pay for full FT.
```

---

## 4. Layer-Freezing Reference Table

Work bottom-up. Each row adds to the previous. Stop when your labels run out.

| # | Strategy | Code | Trainable (BERT-base) | Trainable (VGG16) | Labels | Gain | Use when |
|---|---|---|---|---|---|---|---|
| 0 | **Zero-shot** | no gradients | 0 | 0 | 0–20 | baseline | always measure it |
| 1 | **Linear probe / head only** | `for p: p.requires_grad=False`; head `True` | 2,307 (0.002%) | 4,097 (0.003%) | 100–2,000 | reference | data scarce, OOD risk, the default first experiment |
| 2 | **+ pooler / penultimate** | also unfreeze `bert.pooler`, last LayerNorm | ~592K (0.5%) | — | 500+ | +0.3–0.8 pt | small data, the video's snippet *misses* this |
| 3 | **+ last block** | `layer[-1:]` | ~7.09M (6.5%) | 7,079,424 (block5) | 1k–5k | +0.5–1.5 pt | the video's Way 3 |
| 4 | **+ last 2–4 blocks** | `layer[-4:]` | 14.18M (12.95%) | — | 2k–20k | +0.3–1.0 pt | the video's BERT variant: 0.925 acc / 0.917 F1 |
| 5 | **+ LLRD** | per-layer param groups, decay 0.8–0.9 | same as #4 | same | any | +0.3–0.8 pt | any multi-block unfreeze |
| 6 | **+ STLR / discriminative warmup** | custom scheduler | same | same | any | +0.2–0.6 pt | any full FT |
| 7 | **+ gradual unfreezing over epochs** | per-epoch unfreeze callback | grows | grows | small/medium | +0.3–1.0 pt, less forgetting | encoder classification |
| 8 | **Full FT, all of the above** | `requires_grad=True` everywhere | 109.5M (100%) | 138.4M (100%) | ≥10k | +0.5–1.5 pt | you have the GPU and the labels |

**Count before you choose.** `sum(p.numel() for p in model.parameters() if p.requires_grad)` ÷ total. **Above ~10% you are doing full FT with extra steps.**

| Model | Total params | "Last block" | "Last block" as % | Notes |
|---|---|---|---|---|
| VGG16 | 138,357,544 | 7,079,424 | 5.1% (9.76% with head) | 123.6M lives in the FC layers, not the convs |
| BERT-base | 109,482,240 | 7,087,872 | 6.5% (12.95% for last-2 + head) | pooler 590,592 — do not forget it |
| **Llama-3-8B** | 8,030,261,248 | ~218M | **~17% for last-4 + head** | embeddings ≈ 525M, tied to `lm_head` |

> **The intuition does not survive the transition to LLMs.** "Unfreeze the last block, it's cheap" was true for 2014-era CNNs. On a 7B decoder, one block is ~218M parameters and the embedding table is ~525M. This is exactly why PEFT exists.

**Torchvision index map (VGG16 `features`):** block5 = `24:31` (block5_conv1=24, block5_conv2=26, block5_conv3=28, maxpool=29, avgpool=30). Print `[(i, l) for i, l in enumerate(m.features)]` once and hard-code the slice — guessing indices accidentally unfreezes block4.

---

## 5. Learning-Rate Delta Table (Pretraining vs Full FT vs LoRA vs Head-Only)

| Param | Pretraining | Full FT | Head-only / probe | LoRA / QLoRA | Too high → | Too low → |
|---|---|---|---|---|---|---|
| **Peak LR (AdamW)** | 1e-4 – 6e-4 (LLM); 1e-3 SGD (CV) | **1e-5 – 5e-5** | **1e-3 – 1e-2** | **1e-4 – 3e-4** | Loss spikes, embedding drift, forgetting within 200 steps | ~1/10 speed; looks stuck for the first 200 steps |
| **Schedule** | Cosine or WSD → 10% | Linear → 0 | Constant or cosine | Cosine | Constant @ high LR = late divergence | Schedule never reaches peak |
| **Warmup** | 1–2% of steps | **6–10% of steps** | 0–5% | 5–10% | Wasted epochs at low LR | Cold-start spike on step 1; occasional NaN |
| **Epochs** | < 1 | 2–4 NLP, 10–30 CV | 3–10 | 3–5 | Memorization + forgetting; val loss turns up | Underfit head |
| **Weight decay** | 0.1 | 0.01 | 0.0–0.01 | 0.0–0.01 | Underfit, over-regularized | Overfit on small data |
| **Head dropout** | 0.0–0.1 | 0.1 | 0.1–0.3 | 0.05–0.1 | Slow convergence | Overfit |
| **Batch / device** | 1M–4M tokens | 16–64 NLP, 32–256 CV | 32–256 | 8–32 + accum | OOM | Noisy gradients |
| **Precision** | bf16 / fp16 + scaling | bf16 preferred | fp32 fine | 4-bit NF4 + bf16 compute | fp16 w/o scaling → NaN | fp32 = 2× memory, no gain |

**The one-line rule:** *pretraining LRs are for models with nothing to lose; fine-tuning LRs are for models with everything to lose.*

| LR value | Verdict for full FT of a pretrained encoder |
|---|---|
| 1e-6 | Too conservative; you will conclude "fine-tuning doesn't work" |
| **1e-5** | Conservative, safe — the default for a first attempt |
| **2e-5** | The BERT-era default. Use this. |
| 3e-5 | Aggressive; watch the general metric |
| 5e-5 | Instability starts here on small data |
| ≥1e-4 | A pretraining LR. Destroys the checkpoint in a few hundred steps. |

**LoRA is the exception:** adapters run *higher* than full FT (1e-4–3e-4) because `A` and `B` initialize such that `ΔW = 0` — they start further from a good solution than a pretrained weight does.

---

## 6. Forgetting-Mitigation Table

Catastrophic forgetting = target metric rises while general capability falls. Your task metric **cannot** see it.

| Mitigation | Mechanism | Cost | Knob / magnitude | When it is right |
|---|---|---|---|---|
| **Low LR** | Drift ∝ `η·‖g‖·t`; halve `η`, halve the drift | Free | 1e-5–2e-5 full FT | Always. The first thing to try |
| **Warmup** | Prevents large early steps at a random-init head | Free | 6–10% of steps | Always |
| **Fewer epochs / early stop** | Cuts `t`, the dominant factor on narrow data | Free | 2–4 epochs; stop when the general metric declines | Always |
| **Early stopping on a general metric** | The general metric is your canary | Free | `load_best_model_at_end=True` on target **and** general | Always |
| **LP-FT** | Head settles before the backbone moves | 1 extra short phase | Phase 1 lr 1e-3 frozen; phase 2 lr 2e-5 | Small/medium data with OOD risk |
| **Progressive / gradual unfreezing** | Descending schedule; head first | ~25 lines | epoch 1 last layer, epoch 2 last 2, … | Medium data, encoder, classification |
| **LLRD (discriminative LR)** | Per-layer rates; bottom barely moves | One param-group loop | `decay 0.8–0.9`; head ×2 | Any multi-block unfreeze |
| **LoRA / PEFT** | Base weights *cannot* move; update confined to a low-rank subspace | Slightly lower ceiling | `r=16, α=32`, `lr=2e-4` | Almost always ≥3B; ≤5k labels at any size |
| **Replay / rehearsal** | Restores a source-distribution gradient in every batch | Data curation | **1–10% general data; 5% default** | The single most reliable fix; standard in production SFT |
| **L2-SP** | Penalty toward `θ*` instead of toward zero | Free | `λ ∈ [1e-4, 1e-2]` | Simple baseline for any full FT |
| **KL-to-base** | Directly optimizes "stay close to the base" on general prompts | Base resident in VRAM (or cache logits) | `β ∈ [0.01, 0.5]` SFT | When you can afford two models; the direct fix |
| **EWC** | Fisher-weighted quadratic penalty | Extra Fisher pass over source data | `λ ∈ [1e2, 1e4]` | Classical continual learning; rarely worth it for LLM SFT |
| **Data mixing / curriculum** | Same as replay, at the sampler level | Free | `interleave_datasets([t, g], p=[0.95, 0.05])` | Same as replay |
| **Switch base model** | A domain-adapted or larger base has less to distort | Retraining | — | When everything else has failed |

**Ranked by cost-to-benefit:** lower LR → fewer epochs → warmup → LoRA → replay → L2-SP/KL → EWC. **The cheapest that actually works at scale is replay.**

> **Replay must be distributionally different from the target.** Mixing in more of your own tickets is not replay; it is just more data. The source signal is the point.

---

## 7. The Anti-Catastrophic-Forgetting Recipe (step by step)

Run this in order. Stop as soon as the general metric is stable.

```
STEP 0 — INSTRUMENT BEFORE YOU TRAIN  (non-negotiable)
  0.1 Pick a general probe: LLM -> lm_eval mmlu,arc_challenge,hellaswag + WikiText ppl;
      encoder -> base-model pooled embeddings over 5,000 held-out general sentences.
  0.2 Record the BASE numbers. Without a base number you cannot detect anything.
  0.3 Build a 2,000-prompt general set, held out, distributionally DIFFERENT from target.
  0.4 Freeze the target eval set and hash it.
STEP 1 — LOW LR
  1.1 Full FT of a transformer: 2e-5 (conservative: 1e-5). Never >=1e-4.
  1.2 LoRA: 2e-4. Head-only: 1e-3.
STEP 2 — WARMUP 6-10%
  2.1 HF: warmup_ratio=0.06 — a fraction of OPTIMIZER steps, not micro-batches.
  2.2 Custom schedulers: match it, or warmup is 8x too short under grad accumulation.
STEP 3 — SHORT SCHEDULE + EARLY STOP ON THE GENERAL METRIC
  3.1 2-4 epochs classification, 1-3 epochs SFT. Not more.
  3.2 save_strategy="epoch" AND eval_strategy="epoch" AND load_best_model_at_end=True —
      mismatched strategies raise, or silently never save the best checkpoint.
  3.3 Select the checkpoint on (target UP, general FLAT). Never on target alone.
STEP 4 — ADD REPLAY 5%
  4.1 interleave_datasets([target, general], probabilities=[0.95, 0.05], seed=42)
  4.2 Verify the general stream is NOT from your target domain — otherwise it is just
      more data, not replay.
STEP 5 — LP-FT IF THE DATA IS SMALL OR THE TARGET IS OOD
  5.1 Phase 1: freeze everything, train the head, lr=1e-3, 3 epochs.
  5.2 Phase 2: unfreeze all, lr=2e-5, 1-2 epochs, warmup_ratio=0.1.
STEP 6 — DROP TO LoRA IF STEPS 1-5 ARE NOT ENOUGH
  6.1 r=16, alpha=32, lr=2e-4, target_modules expanded to the MLP projections.
  6.2 Base weights cannot move — forgetting becomes structurally impossible.
  6.3 Accept a slightly lower ceiling on the target. It is usually 1-2 points.
STEP 7 — OPTIONAL ADD-ONS, IN THIS ORDER
  7.1 L2-SP: +(lam/2)*||theta - theta*||^2, lam in [1e-4, 1e-2]  (toward theta*, not zero)
  7.2 KL-to-base: +beta*KL(p_base || p_tuned) on the 2,000 prompts, beta 0.01-0.5
  7.3 EWC only with a genuine continual-learning requirement and a source dataset for
      the Fisher pass.
STEP 8 — GATE THE RELEASE
  MMLU delta > -0.5 | WikiText ppl ratio < 1.05 | KL < 0.02 nats | cos drift < 0.03
  Any signal in the "stop" band => do not ship. Lower the LR, cut epochs, raise replay
  to 10%, or switch to LoRA and re-run.
```

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Diagnostic | Fix |
|---|---|---|---|
| Loss flat at 0.693 (binary) | Head not learning; often everything frozen | `sum(p.requires_grad for p in model.parameters())` | Unfreeze the head; check the optimizer holds it |
| Loss flat at 1.099 (3-class) | Model predicting uniform | `print(model.classifier)` | Head trainable; labels ∈ {0,1,2} |
| Loss flat at 1.386 with 3 classes | `num_labels=4` with 3 present | `model.config.num_labels` vs the dataset's class count | Set `num_labels` correctly; remap labels |
| Loss decreasing, accuracy stuck at majority class | Class imbalance without weighting, or misaligned labels | `Counter(labels)`, confusion matrix | `class_weight`, weighted sampler, focal loss; verify label↔index |
| Loss → NaN in the first 20 steps | LR too high; no warmup; fp16 without loss scaling | Log the first 10 loss values | LR ÷ 10, add 6–10% warmup, bf16 or enable loss scaling |
| Loss oscillates, never descends | LR above the stability threshold for the effective batch | Reduce LR 3× and re-run 50 steps | 2e-5 → 7e-6; increase effective batch |
| Val loss rises, train falls | Overfitting (big LR × many epochs × small data) | Train/val curves; `trainable%` | Fewer epochs, early stop, weight decay ↑, dropout ↑, freeze more |
| **Val loss falls but the general benchmark falls** | **Catastrophic forgetting** | MMLU / WikiText ppl / KL-to-base probe | 5% replay, LR ÷ 3, fewer epochs, switch to LoRA |
| Val accuracy 0.99 on a hard task | Train/val leakage (duplicate or near-duplicate rows) | Hash raw inputs, count overlap | Dedup by content hash *before* splitting |
| Accuracy far below the reported paper | LR/schedule mismatch; papers omit warmup and epochs | Reproduce the paper's exact LR and epoch count | Mosbach et al. 2021: *lower* LR and *more* epochs than the original recipe |
| `CUDA out of memory` on a model that should fit | Two model copies resident (HF Trainer + a custom class) | `torch.cuda.memory_allocated()`, `nvidia-smi` | `del model; gc.collect(); torch.cuda.empty_cache()`; bs ÷ 2 + accumulation |
| OOM only at eval | `per_device_eval_batch_size` left at default with `max_len` 512 | Print the args | Set eval batch size explicitly; `eval_accumulation_steps=1` |
| Grad is `None` for a parameter you expect to train | Layer unreachable from the loss, or frozen after optimizer construction | `[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]` | Rebuild the optimizer after freezing; check the forward pass |
| Trainable count is 0 | Frozen everything, never un-froze the head | Print before `trainer.train()` | Add the unfreeze step; assert count > 0 |
| Keras: freezing has no effect | Flags set after `model.compile()` | `model.summary()` trainable count | Re-`compile()` after changing `trainable` |
| Keras VGG16 accuracy plateaus ~0.84 | Wrong input normalization | Compare to the backbone's training-time transform | `tf.keras.applications.vgg16.preprocess_input` (RGB→BGR + mean subtraction) |
| All one class in production, fine in eval | Model left in `train()` mode (dropout + BN batch stats) | `model.training` | `model.eval()` in the serving path |
| Batched inference differs from single-example | Padding side or `attention_mask` handling | Encode one vs many, compare logits | Set `tokenizer.padding_side`; always pass `attention_mask` |
| F1 is 0 on one class | Class never predicted (rare + unweighted) | Confusion matrix | Class weights, oversampling, threshold tuning |
| Eval metrics identical across epochs | `eval_strategy` never fired | Count `eval` lines in the log | Match `eval_strategy` and `save_strategy`; set `eval_steps` |
| Accuracy drops after merging a LoRA adapter | `lora_alpha`/scaling mismatch, or merged in the wrong dtype | Compare logits pre/post-merge on 10 examples | `merge_and_unload()` on CPU in fp32, then cast |
| Worked in staging, fails at load in production | Base model revision moved on the Hub | Check the pinned `revision=` | Pin every artifact: base SHA + adapter SHA + tokenizer SHA |
| Fine-tune succeeded but is worse than the prompt baseline | You never measured the prompt baseline | Run the 5-shot baseline on the same eval set | If the prompt wins, ship the prompt |
| `IndexError: Target 3 is out of bounds` | `Dataset.filter()` did not renumber labels | `print(sorted(set(dataset['label'])))` | Remap `{0:0, 1:1, 3:2}` after filtering |

---

## 9. Comparison Matrix — The Six Adaptation Strategies

| Dimension | Zero-shot / prompt | Linear probe | Freeze base + dense head | Partial (last-N) + LLRD | Full fine-tune | LoRA / QLoRA |
|---|---|---|---|---|---|---|
| Trainable (BERT-base) | 0 | 2,307 (0.002%) | 2,307–590K (~0.5%) | 14.2M (13%) | 109.5M (100%) | 0.3M (0.3%) |
| Labels needed | 0–20 | 100–2,000 | 500–5,000 | 2,000–50,000 | 10,000+ | 500–50,000 |
| Quality in-distribution (rel.) | 0.70–0.85 | 0.90 | 0.92 | 0.95 | **1.00** | 0.96–0.99 |
| Quality **OOD** (rel.) | 0.70 | **0.94 (best)** | 0.93 | 0.90 | 0.88 | 0.92 |
| Forgetting risk | none | none | none | low | **high** | low |
| Wall-clock (9.7k ex, T4) | 0 | 1 min | 2 min | 4 min | 11 min | 3 min |
| VRAM (BERT-base, bs 16) | 0.5 GB | 0.8 GB | 0.9 GB | 1.4 GB | 4.2 GB | 2.5 GB |
| VRAM (8B LLM) | 16 GB (inference) | n/a | n/a | n/a | 112 GB | **6–8 GB** |
| Multi-tenant artifact | n/a | head 9 KB | head 9 KB | 56 MB | 28 GB fp16 | 20–200 MB |
| Implementation complexity | lowest | low | low | medium | low (resource-heavy) | medium |
| Best when | the task is already solved | data scarce, OOD risk | data scarce, in-domain | medium data, encoder | abundant in-domain data + GPU | **always >3B; usually <5k labels** |
| Worst when | task is genuinely new | head capacity insufficient | ceiling too low | over-tuning small data | data <10k or domain-shifted | you need max accuracy on huge in-domain data |

**Measured head-to-head (CS-02, BERT-base, 3-class emotion, 9,749/2,438):**

| Run | Trainable | Wall-clock (T4) | Val acc | Val macro-F1 |
|---|---|---|---|---|
| Head only | 2,307 (0.002%) | ~70 s | 0.885 | 0.874 |
| Last 2 blocks + head | 14,178,051 (12.95%) | ~4 min | **0.925** | **0.917** |
| Full FT @ 2e-5, 3 epochs | 109,482,243 (100%) | ~11 min | 0.933 | 0.926 |
| Full FT @ 5e-5, 10 epochs | 100% | ~35 min | 0.918 | 0.906 |

**Read rows 1, 2 and 4 together:** 96% of the benefit for 3% of the compute (probe vs last-2), and 3× the compute at a higher LR produced a *worse* model than 3 epochs. That is the whole cost/quality story in one table.

---

## 10. Numbers To Memorize

| Category | Number | Why |
|---|---|---|
| **LR** | 1e-5 – 5e-5 full FT; 2e-5 default | 10–100× below pretraining |
| **LR** | 1e-3 – 1e-2 head-only; 1e-4 – 3e-4 LoRA | Head is from scratch; adapters start at zero |
| **LR** | ≥1e-4 = destruction for full FT | The single most common way to kill a checkpoint |
| **Warmup** | 6–10% of steps | More important than in pretraining |
| **Epochs** | 2–4 classification, 1–3 SFT | Past this you are training the model to forget |
| **Weight decay** | 0.01 | Not the 0.1 of pretraining |
| **Memory** | 16 bytes / trainable param | 2+2+8+4 (bf16 w, bf16 g, fp32 m&v, fp32 master) |
| **Memory** | 7B full FT ≈ 112 GB; QLoRA ≈ 6–8 GB; 70B QLoRA ≈ 38–48 GB | The reason PEFT exists |
| **Memory rule** | Full FT ≈ 20 × param count in GB | 7B ⇒ ~140 GB with activations |
| **Data** | 10 × n_classes = head-only floor | 3 classes ⇒ 30 examples |
| **Data** | 500–2,000 = format; 10k+ = behavior; — = facts | The trichotomy |
| **Data** | 100× the labels buys ~1 point past the knee | Spend it on the OOD eval set instead |
| **Loss floors** | ln(2)=0.693, ln(3)=1.099, ln(4)=1.386 | Flat loss ⇒ head/label bug, not LR |
| **VGG16** | 138,357,544 total; 14,714,688 conv base; block5 7,079,424 | Head + Dense(256) = 6,423,041 (4.64%) |
| **BERT-base** | 109,482,240 total; block 7,087,872; pooler 590,592 | last-2 + head = 14,178,051 (12.95%) |
| **Llama-3-8B** | 8,030,261,248 total; ~218M/block; embeddings ≈ 525M | last-4 + head ≈ 1.4B ≈ 17% |
| **Emotion dataset** | 12,187 → 9,749 / 2,438 | Verified against the video's on-screen counts |
| **Forgetting gates** | MMLU > −0.5; ppl ratio < 1.05; KL < 0.02 nats; cos drift < 0.03 | CI thresholds |
| **Replay** | 1–10%, 5% default | Cheapest reliable anti-forgetting measure |
| **Cost** | QLoRA 8B 3 h = $1.35; full FT 8B = $150 | ~100× for 1–3 points |
| **LoRA** | r=16, alpha=32, lr=2e-4; adapter 21 MB | `A` random, `B` zero ⇒ `ΔW=0` at init |
| **ULMFiT** | LLRD 1/2.6; STLR cut_frac=0.1, ratio=32 | Production LLRD decay is 0.8–0.9 |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `IndexError: Target 3 is out of bounds` | Labels `{0,1,3}` with `num_labels=3` — `Dataset.filter()` does not renumber | Remap `{0:0, 1:1, 3:2}` after filtering |
| `RuntimeError: CUDA out of memory` | Multiple model copies resident, or eval batch too large | `del model; gc.collect(); torch.cuda.empty_cache()`; bs ÷ 2 + accumulation |
| `ValueError: Expected input batch_size (X) to match target batch_size (Y)` | Shape mismatch in a rebuilt Keras head — usually a missing `Flatten` | Add `layers.Flatten()` between the conv base and the first `Dense` |
| `ValueError: Unknown activation function: ...` / wrong output shape with `include_top=False` | `VGG16(include_top=False)` without an explicit `classes`/head — or `include_top` left `True` | `include_top=False` and build your own head; verify with `model.summary()` |
| `RuntimeError: element 0 of tensors does not require grad` | Everything is frozen — no trainable parameter reaches the loss | Unfreeze the head; `assert trainable_count > 0` before `train()` |
| `AssertionError` from `Trainer` about eval/save strategy | `load_best_model_at_end=True` with `save_strategy != eval_strategy` | Make them match; set `eval_steps` |
| `AttributeError: 'Dataset' object has no attribute 'set_format'` | `datasets` major-version change | Pin versions; use `with_format("torch")` on newer `datasets` |
| `TypeError: filter() got an unexpected keyword argument 'load_from_cache_file'` | `datasets` API change across majors | Pin `datasets`; check the changelog for your pinned version |
| `KeyError: 'label'` after `set_format` | The column list omitted `label` | `set_format("torch", columns=["input_ids","attention_mask","label"])` |
| `OSError: Can't load tokenizer for ...` at serving | Tokenizer not saved with the model, or a different revision | Save the tokenizer alongside the checkpoint; pin its SHA |
| `Token indices sequence length is longer than ... 512` | Sequences exceed the model's positional limit | Set `max_len` and `truncation=True`; check the length percentiles first |
| Warnings about `padding_side`/`pad_token_id` | Padding side differs from training | Set `tokenizer.padding_side` explicitly in the serving path |
| `RuntimeError: one of the variables needed for gradient computation has been modified` | In-place op on a tensor needed for backward (common in custom heads) | Replace in-place ops; clone the tensor before mutating |
| Silent: `Non-trainable params:` equals `Total params:` | Freezing applied but never un-froze anything | Print the first trainable tensor name; the video's snippet also leaves `bert.pooler` frozen |
| Silent: eval accuracy never changes | `eval_strategy` never fires | Count the `eval` lines in the training log |

---

## 12. VRAM / Cost Calculator

**Full FT (bf16 + AdamW, before activations): 16 bytes × parameters.**

| Model | Full FT | LoRA (bf16) | QLoRA (NF4) | Min card for QLoRA |
|---|---|---|---|---|
| 1B | 16 GB | 4 GB | 2 GB | 4 GB |
| 3B | 48 GB | 9 GB | 4 GB | 6 GB |
| 7–8B | **112 GB** (+activations) | 18–22 GB | **6–8 GB** | 8 GB (12 GB comfortable) |
| 13B | 208 GB | 30 GB | 10 GB | 12 GB |
| 34B | 544 GB | 72 GB | 22 GB | 24 GB |
| 70B | 1.12 TB | 145 GB | **38–48 GB** | 48 GB (or 2×24 GB) |
| 405B | ~6.5 TB | 830 GB | 220 GB | Multi-node |

**Encoder models (BERT-base 110M, bs 16, len 128):** head-only ~0.8 GB (fits any 4 GB card, CPU viable) · last-2 + head ~1.4 GB · full FT bs 16 ~4.2 GB · full FT bs 64 ~10.8 GB · LoRA r=16 ~2.5 GB (activations dominate, not optimizer state).

**Worked budgets** (rental: Vast.ai 4090 $0.35–0.60/h, RunPod A100 80 GB $1.50–2.50/h, H100 $2.50–4.00/h; free tiers: Colab T4 12.7 GB, no bf16; Kaggle 2×T4 30 h/week):

| Job | Config | Time | Cost |
|---|---|---|---|
| BERT-base, 9,749 ex, 3 epochs | last-2 unfrozen, T4 | 4 min | **$0.00** |
| Llama-3-8B QLoRA, 10k ex, 3 epochs | bs 4 × accum 8, len 1024, NF4, 4090 | 2.5–4 h | **$1.20** |
| Llama-3-8B full FT, 100k ex, 2 epochs | bs 32, len 2048, FSDP on 4×A100 | 18–26 h | $130–$200 |
| Clinical DAPT (MLM), 2.1B tokens | bert-base, 2×A100 80 GB | 19 h | $76 → +10.8 F1 |
| Clinical NER SFT, 3,100 notes, 4 epochs | bert-base, 1 GPU | 18 min | $0.60 |
| LoRA adapter, 300–800 ex | r=8, 4090 | ~12 min | $0.08 → 21 MB |

**The economic rule:** full FT costs ~100× a QLoRA run for what is typically 1–3 points of task accuracy. Spend the difference on evaluation and data cleaning. Do not buy a GPU — $0.40/h × 4 h = $1.60 versus a $2,000 card.

---

## 13. Copy-Paste Code Snippets

### 13.1 Freeze / unfreeze by name, with the receipt and the leak check

Full implementation in §15 (`set_trainable_by_unfreezing_top`). The invariants: resolve modules
by **name**, not index (indices break across `transformers` versions); unfreeze the last n
blocks of `bert.encoder.layer` **plus** `bert.pooler` and `classifier` — the pooler is the step
the video's snippet misses (590,592 params, and it feeds the head).

```python
def assert_no_gradient_leak(model, batch):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert tr > 0, "nothing is trainable"
    out = model(**{k: v[:2] for k, v in batch.items()}); out.loss.backward()
    leaked = [n for n, p in model.named_parameters()
              if p.grad is not None and not p.requires_grad]
    assert not leaked, f"gradient leaked into frozen tensors: {leaked[:3]}"
    first = next(n for n, p in model.named_parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"trainable {tr:,}/{tt:,} = {100*tr/tt:.2f}%  |  first: {first}")
    return out.loss.item()
# -> trainable 14,178,051/109,484,547 = 12.95%  |  first: bert.encoder.layer.10....
```

### 13.2 Freeze the prefix AND drop its activations

```python
# requires_grad=False alone does NOT save activation memory — wrap the frozen prefix.
for p in model.conv_base.parameters():
    p.requires_grad = False
with torch.no_grad():                       # <-- the line people forget
    features = model.conv_base(x)
features = features.detach()
loss = criterion(model.head(features), y); loss.backward()
```

### 13.3 LLRD + slanted triangular LR (the ULMFiT pair)

```python
def param_groups_llrd(model, base_lr, decay=0.8, num_layers=12):
    """Bottom layers smallest LR, head largest."""
    groups = [{"params": model.bert.embeddings.parameters(),
               "lr": base_lr * (decay ** num_layers)}]
    for i in range(num_layers):
        groups.append({"params": model.bert.encoder.layer[i].parameters(),
                       "lr": base_lr * (decay ** (num_layers - 1 - i))})
    groups.append({"params": model.classifier.parameters(), "lr": base_lr * 2.0})
    return groups

def stlr(step, total_steps, peak_lr, cut_frac=0.1, ratio=32):
    """Short linear warmup to peak, long linear decay back to peak/ratio."""
    cut = int(total_steps * cut_frac)
    if step < cut:
        return peak_lr * step / max(cut, 1) * (1 / ratio) + peak_lr / ratio
    p = (step - cut) / max(total_steps - cut, 1)
    return peak_lr * (1 - p * (1 - 1 / ratio))

# optimizer = torch.optim.AdamW(param_groups_llrd(model, base_lr=2e-5, decay=0.8),
#                               weight_decay=0.01)
```

### 13.4 Replay + KL-to-base (the forgetting pair)

```python
from datasets import interleave_datasets
import torch.nn.functional as F

# Replay: 5% general data in every batch. The two streams MUST be distributionally different,
# or the replay carries no source signal and you have only added compute.
train = interleave_datasets([target_ds, general_ds], probabilities=[0.95, 0.05], seed=42)

# KL-to-base: the measurement that is also the mitigation. beta in [0.01, 0.5] for SFT.
with torch.no_grad():
    base_logits = base_model(**batch).logits
kl = F.kl_div(F.log_softmax(tuned_logits, -1),
              torch.softmax(base_logits, -1),   # log_target=False => pass PROBABILITIES
              reduction="batchmean", log_target=False)
loss = task_loss + beta * kl
```

### 13.5 The 30-second pre-flight

```python
def preflight(model, tokenizer, dataset, texts, batch, n_unfreeze=2):
    print(model.config.num_labels, model.config.hidden_size, model.config.num_hidden_layers)
    print(tokenizer(texts[:3])["input_ids"])                # is it shredding your text?
    lens = [len(t) for t in tokenizer(list(texts))["input_ids"]]
    print(np.percentile(lens, [50, 95, 99]))                # -> set max_len
    print(Counter(dataset["label"]))                        # imbalance?
    set_trainable_by_unfreezing_top(model, n_unfreeze)      # freeze, then print the receipt
    loss = assert_no_gradient_leak(model, batch)            # §13.1 — grads only where intended
    assert abs(loss - math.log(model.config.num_labels)) < 0.1, loss
    assert max(dataset["label"]) < model.config.num_labels, "label remap missing"
    return True
```

**Serving: merge vs hot-swap.** One behavior per GPU → `model.merge_and_unload()` (on CPU in fp32, then cast to fp16). N behaviors on one GPU → do **NOT** merge: `vllm serve <base> --enable-lora --lora-modules a=./adapter_a b=./adapter_b` (~+11 ms p95 per swap). Merging destroys the ability to swap; `r` and `target_modules` are fixed at train time, so decide before you train.

---

## 14. CLI Commands

```bash
pip install transformers datasets accelerate peft bitsandbytes evaluate "lm-eval"

# --- general capability BEFORE training (the alignment-tax number) ---
lm_eval --model hf --model_args pretrained=meta-llama/Meta-Llama-3.1-8B \
        --tasks mmlu,arc_challenge,hellaswag --batch_size 8 --output_path base.json

# --- AFTER training, then diff the two result files task by task ---
lm_eval --model hf --model_args pretrained=./my-ft-8b \
        --tasks mmlu,arc_challenge,hellaswag --batch_size 8 --output_path tuned.json
python -c "import json;b=json.load(open('base.json'))['results'];t=json.load(open('tuned.json'))['results'];
[print(f'{k:16s} {t[k][\"acc,none\"]-b[k][\"acc,none\"]:+.3f}') for k in b]"

# --- perplexity drift on held-out general text (gate: ratio < 1.05) ---
lm_eval --model hf --model_args pretrained=./my-ft-8b --tasks wikitext \
        --num_fewshot 0 --output_path tuned_ppl.json

# --- token-length percentiles: measure before you set max_len ---
python -c "
from transformers import AutoTokenizer; import numpy as np, sys
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
L=[len(x) for x in tok([l.strip() for l in sys.stdin if l.strip()])['input_ids']]
print(np.percentile(L,[50,95,99]), max(L))" < corpus.txt

# --- the VGG16 block5 boundary, confirmed (torchvision indices 24:31) ---
python -c "
from torchvision import models
m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print([(i, l.__class__.__name__) for i, l in enumerate(m.features)][20:31])"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5   # VRAM watch
```

---

## 15. Copy-Paste Starter Config

Runnable end-to-end: probe → last-2-blocks → full FT, on the video's own dataset, with the
pre-flight, the receipts, and the forgetting hooks. T4 / 16 GB, ~15 min for all three.

```python
# ch02_starter.py — pip install transformers datasets evaluate scikit-learn torch
import math, time, torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

MODEL_ID, MAX_LEN, SEED = "bert-base-uncased", 128, 42
KEEP = ["sadness", "joy", "anger"]              # the video's 3-class subset  [53:45]
torch.manual_seed(SEED)

# ---- 1. data -------------------------------------------------------------
ds = load_dataset("dair-ai/emotion")
label_names = ds["train"].features["label"].names   # sadness,joy,love,anger,fear,surprise
ds = ds.filter(lambda ex: label_names[ex["label"]] in KEEP)
# CRITICAL: filter() does NOT renumber — the column now holds {0,1,3}, not {0,1,2}.
remap = {label_names.index(n): i for i, n in enumerate(KEEP)}   # {0:0, 1:1, 3:2}
ds = ds.map(lambda ex: {"label": remap[ex["label"]]})
split = ds["train"].train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
train_ds, val_ds = split["train"], split["test"]
print(len(train_ds), len(val_ds), Counter(train_ds["label"]))
#   -> 9749 2438   Counter({1: 4290, 0: 3732, 2: 1727})   (the video's counts)
assert max(train_ds["label"]) < 3, "label remap missing"

# ---- 2. tokenize ---------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
enc = lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
train_ds, val_ds = train_ds.map(enc, batched=True), val_ds.map(enc, batched=True)
cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format("torch", columns=cols); val_ds.set_format("torch", columns=cols)
lens = [len(x) for x in tok(list(train_ds["text"]))["input_ids"]]
print("len p50/p95/p99:", np.percentile(lens, [50, 95, 99]))

# ---- 3. freezing utility + metrics ---------------------------------------
def set_trainable_by_unfreezing_top(model, n_unfreeze, prefix="bert.encoder.layer"):
    for p in model.parameters():
        p.requires_grad = False
    blocks = [m for n, m in model.named_modules() if n.startswith(prefix) and m is not model]
    for blk in blocks[-n_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
    for n, p in model.named_parameters():        # pooler is the step the video misses
        if n.startswith("classifier") or n.startswith("bert.pooler"):
            p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    first = next(n for n, p in model.named_parameters() if p.requires_grad)
    print(f"  trainable {tr:,}/{tt:,} = {100*tr/tt:.3f}% | first: {first}")
    assert tr > 0
    return model

def metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": float((preds == p.label_ids).mean()),
            "macro_f1": float(f1_score(p.label_ids, preds, average="macro"))}

# ---- 4. one run, parameterized over the freezing depth -------------------
def run(name, n_unfreeze, lr, epochs):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)
    set_trainable_by_unfreezing_top(model, n_unfreeze)
    b = {k: v[:2] for k, v in next(iter(torch.utils.data.DataLoader(train_ds, batch_size=2))).items()}
    l0 = model(**b).loss.item()                  # pre-flight: the loss@init gate
    print(f"  loss@init {l0:.3f} (expect {math.log(3):.3f})")
    assert abs(l0 - math.log(3)) < 0.15, "head/label problem"
    args = TrainingArguments(
        output_dir=f"./{name}", num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,       # set EXPLICITLY or you OOM only at eval
        learning_rate=lr, warmup_ratio=0.06, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", logging_steps=50, report_to="none", seed=SEED,
        fp16=torch.cuda.is_available(),      # T4 has no bf16; bf16 on Ampere+
    )
    t0 = time.time()
    tr = Trainer(model=model, args=args, train_dataset=train_ds,
                 eval_dataset=val_ds, compute_metrics=metrics)
    tr.train(); m = tr.evaluate()
    print(f"{name}: acc={m['eval_accuracy']:.4f} f1={m['eval_macro_f1']:.4f} ({time.time()-t0:.0f}s)")
    del model, tr; torch.cuda.empty_cache()
    return m

if __name__ == "__main__":
    run("probe", 0, 1e-3, 3)     # 0.002% trainable
    run("last2", 2, 2e-5, 3)     # 12.95% trainable  <-- the video's variant, completed
    run("full", 12, 2e-5, 3)     # 100% trainable
# Expected on a T4 (CS-02 §6.4):
#   probe : acc 0.885  f1 0.874  ~70 s        <- 96% of the benefit, 3% of the compute
#   last2 : acc 0.925  f1 0.917  ~4 min
#   full  : acc 0.933  f1 0.926  ~11 min
# Then, ALWAYS, before shipping:
#   - held-out target metric, prompt baseline, OOD slice, general-capability delta, ECE
#   - KL-to-base on 2,000 general prompts < 0.02 nats
#   - pin (model_id, base_revision_sha, adapter_sha, tokenizer_sha, data_snapshot_id)
```

---

## 16. What To Read Next

| Topic | Module |
|---|---|
| Pretraining and the model lifecycle | CS-01 |
| Why fine-tuning was hard pre-transformer | CS-05 |
| BERT fine-tuning for NER, sentiment, QA | CS-07 |
| Domain-adaptive continued pretraining on your own PDFs | CS-12 |
| Instruction fine-tuning (SFT) | CS-13 |
| Embedding fine-tuning — the same transfer logic in a bi-encoder | code/10_embedding_finetune.py (CS-22 planned, not yet written) |
| Fine-tuning vs RAG vs agents | CS-04 |
| LoRA & QLoRA, the PEFT deep dive | CS-13 §6.8, CS-11 §4.11 (CS-23 planned, not yet written) |
| Quantization, and the serving economics of everything above | CS-10, CS-11 |
| Distillation as an alternative to fine-tuning | CS-08, CS-09 |
| The interview bank for this module | IQ-02 |

**Papers to have read, by claim:** Yosinski 2014 (early-vs-late transferability); Kumar 2022 (LP-FT and the OOD inversion); Kornblith 2019 (feature extraction is a strong baseline); He 2019 (from-scratch matches with enough data); Howard & Ruder 2018 (LLRD, STLR, gradual unfreezing); Kirkpatrick 2017 (EWC); Xuhong 2018 (L2-SP); Mosbach 2021 (BERT fine-tuning instability — lower LR, more epochs); Zhang 2021 (re-initializing top layers on few-sample data); Ouyang 2022 (the alignment tax); Gururangan 2020 (DAPT/TAPT); Biderman 2024 (LoRA learns less and forgets less).
