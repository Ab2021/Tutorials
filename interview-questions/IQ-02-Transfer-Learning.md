# IQ-02 — Interview Questions: Transfer Learning & Fine-Tuning

| Field | Value |
|---|---|
| **Module** | Foundations / Transfer Learning |
| **Pairs with** | CS-02 (Transfer Learning & Model Fine-Tuning), CH-02 (Transfer Learning cheat sheet) |
| **Total questions** | 94 (28 L1 + 28 L2 + 20 L3 + 8 L4 + 10 L5) |
| **Levels covered** | Screen / Intermediate / Advanced / System Design / Debug |
| **Source** | LLM Fine-Tuning 03: Transfer Learning and Model Fine-Tuning (Sunny Savita); CS-02 carries the verified transcript claims |

---

## How To Use This File

- **L1** = phone screen / recruiter filter (30-second answers). If you cannot answer these without hedging, do not book the loop.
- **L2** = working engineer (2–3 min answers, expects implementation detail: flags, layer names, exact numbers).
- **L3** = senior / specialist (5 min, expects mechanism, trade-offs, and the paper the claim comes from).
- **L4** = staff / system design (15-min whiteboard). Answer in the fixed order **requirements → constraints → design → trade-offs → failure modes**; skipping to "design" is the most common way to fail these.
- **L5** = debugging & incident (war stories). Answer as an *ordered checklist*, not a bag of guesses.
- `[Company style: ...]` tags mark the loop where a question typically appears.

**The three-sentence version of this module**, in case you only remember one thing: transfer learning is the strategy and fine-tuning is the tactic [23:37]; early layers compute primitive features and late layers compute task-specific ones, so you freeze the bottom and move the top [21:00]–[21:41]; and the failure mode that matters is catastrophic forgetting, which your task metric cannot see, because fine-tuning learning rates are 10–100× smaller than pretraining learning rates for a reason.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. Define transfer learning and fine-tuning, and state precisely how they differ.**
- **Answer:** Transfer learning is the *strategy* of reusing knowledge acquired on a source domain/task `(D_S, T_S)` to improve performance on a target domain/task `(D_T, T_T)`. Fine-tuning is the *tactic*: additional training of a pretrained model on target data. The instructor's framing is exact — "this fine-tuning, this transfer learning, and this fine-tuning are the two aspects of a single task" [23:37], restated at [25:29] as two sides of a single coin. They co-occur; they are not synonyms. You cannot fine-tune without transferring; you can transfer without fine-tuning (zero-shot inference, or a frozen backbone feeding a classical classifier).
- **Why the interviewer asks:** it is the single most-failed vocabulary distinction in the field. Candidates who use the two words interchangeably will also confuse "we fine-tuned" with "we prompted," which corrupts every later design conversation.
- **Trap:** "Fine-tuning is a type of transfer learning, so they mean the same thing." They nest, they do not equate. The follow-up that catches this answer is "give me transfer learning with no fine-tuning" — the correct answer is zero-shot/few-shot inference on a pretrained model, or a frozen feature extractor plus an SVM.

**Q2. What is the difference between a source domain/task and a target domain/task, formally?**
- **Answer:** `D = (X, P(X))` — an input space plus its marginal distribution. `T = (Y, P(Y|X))` — a label space plus the conditional mapping. The source pair is what the checkpoint was trained on (ImageNet-1k classification for VGG16 [11:43]–[12:38]; web-scale MLM for BERT [51:22]); the target pair is what you actually need. Domain shift means `P_S(X) ≠ P_T(X)`; task shift means `Y` or `P(Y|X)` changed.
- **Why the interviewer asks:** it separates people who think "my data is different" from people who can say *which half* is different. The distinction decides whether you spend money on labels or on unlabeled text.
- **Trap:** describing the target only by its labels ("we classify tickets into 42 categories") and never stating the input distribution. The input distribution is half the problem and, in practice, the expensive half.

**Q3. Name the four quadrants of the domain × task taxonomy and what each implies operationally.**
- **Answer:** Crossing same/different domain with same/different task:
  - **Q1 same/same** — only priors or label noise moved. Tune the head only, or just calibrate. 100–2,000 examples.
  - **Q2 same/different** — classic inductive transfer (the video's cat/dog and emotion tasks). Head + last 1–4 blocks or LoRA. 1k–100k examples.
  - **Q3 different/same** — domain adaptation: head + upper blocks **after** continued pretraining on unlabeled target text. 10k–1B unlabeled tokens + 1k–10k labeled.
  - **Q4 different/different** — continued pretraining + head + upper blocks, or PEFT. 100k+ labeled, 1B+ unlabeled.
- **Why the interviewer asks:** the quadrant is the entire decision procedure compressed. Getting it right is the difference between a 19-hour DAPT run worth +10.8 F1 and a month of labeling that buys nothing.
- **Trap:** treating everything as Q2 because "we have labels." Labeling 50k examples for a Q3 problem produces 88% validation accuracy and a model that collapses on slightly-further-out production inputs.

**Q4. What is feature extraction, and how is it different from a linear probe?**
- **Answer:** Feature extraction uses the frozen backbone purely as a fixed featurizer and trains a new head on its outputs. A **linear probe** is the special case where that head is a single linear layer trained with a high LR (1e-3 to 1e-2). Feature extraction with an MLP head is still feature extraction. Both are the cheapest, most forgetting-proof, strongest-regularizer option — and both have the lowest ceiling in-distribution.
- **Why the interviewer asks:** it tests whether you know the baseline everyone skips. If a candidate proposes full fine-tuning as a first experiment on 800 examples, that is a red flag.
- **Trap:** "linear probing = last-layer fine-tuning." It does not: last-layer fine-tuning lets gradients flow into the last block, which is a materially different (and riskier) configuration. On BERT-base the linear probe is 2,307 trainable parameters (0.002%); last-2-blocks + head is 14,178,051 (12.95%).

**Q5. Why is the linear probe a mandatory baseline rather than an optional one?**
- **Answer:** Three reasons. (1) It costs ~70 seconds on a T4 for 9,749 examples, so there is no budget argument against it. (2) It is frequently within 2 points of the best configuration — CS-02's BERT run measured 0.885 accuracy / 0.874 macro-F1 for the probe versus 0.925 / 0.917 for the last-2-blocks variant: **96% of the benefit for 3% of the compute**. (3) It is often the *best* option out-of-distribution (Kumar et al., 2022). If LP ≈ full FT, your problem is data or evaluation, not fine-tuning.
- **Why the interviewer asks:** it reveals whether the candidate has an experimental discipline or just reaches for the largest GPU. It also creates an easy follow-up: "what if the probe beats your fine-tune?"
- **Trap:** "We skipped it — full FT is strictly better." On OOD targets the ordering can invert, and on <1k examples full FT frequently loses to the probe because it has enough capacity to memorize.

**Q6. What is full fine-tuning and what is its memory floor?**
- **Answer:** Updating every parameter with a small learning rate. With Adam-family optimizers in mixed precision the steady-state cost is **16 bytes per trainable parameter**: 2 (bf16 weights) + 2 (bf16 grads) + 8 (fp32 Adam m and v) + 4 (fp32 master weights). A 7B model is therefore ~112 GB before activations. No flag changes that arithmetic — only PEFT or ZeRO/FSDP does.
- **Why the interviewer asks:** the arithmetic is the reason PEFT exists. A candidate who cannot produce the 16 bytes/param figure cannot size a training job.
- **Trap:** "bf16 halves it to 8 bytes." bf16 halves the weights and gradients; Adam's moments stay fp32 and the master copy stays fp32. And the "master weights" line is a property of the mixed-precision optimizer, not of the dtype you saved the checkpoint in.

**Q7. What is PEFT, and name four members of the family.**
- **Answer:** Parameter-Efficient Fine-Tuning — any method that trains ≤1–5% of parameters while freezing the rest. Members: LoRA (`ΔW = BA`, low-rank), QLoRA (LoRA on a 4-bit NF4 base), DoRA (magnitude/direction decomposition), adapters (bottleneck modules), prefix/prompt tuning (virtual tokens), IA³ (rescaling vectors), BitFit (bias terms only, ~0.08% of parameters).
- **Why the interviewer asks:** PEFT is the default above ~3B parameters, and the follow-up ("so is PEFT worse?") is a genuine discriminator.
- **Trap:** "PEFT is a worse full FT." On ≤5k examples LoRA often beats full FT, because the frozen base acts as a regularizer — Biderman et al. (2024), *LoRA Learns Less and Forgets Less*.

**Q8. What is catastrophic forgetting, and why is it dangerous specifically?**
- **Answer:** Degradation of previously acquired capabilities caused by gradient updates for a new task. Each step moves `θ` along `−∇L_target`; nothing in the target objective says "stay good at the source distribution," so the fastest way to lower target loss on a small dataset is often to repurpose features doing source-task work. It is dangerous because **your target metric cannot see it** — accuracy goes up while the model quietly gets worse at everything else. The industrial name is the **alignment tax** (InstructGPT, Ouyang et al. 2022).
- **Why the interviewer asks:** it is the dominant silent production failure in this module. Candidates who only know "don't overfit" have no vocabulary for it.
- **Trap:** confusing it with overfitting. Overfitting is worse *on the target* (train loss falls, val loss rises). Forgetting is worse *everywhere else* (val loss can fall while MMLU falls).

**Q9. Distinguish domain shift from covariate shift.**
- **Answer:** Domain shift is the general case `P_S(X) ≠ P_T(X)`. Covariate shift is the subtype where the `X`-marginal changes while `P(Y|X)` is unchanged. The practical significance: under covariate shift your labels remain valid, so you adapt features and reuse them; you do not re-derive the task.
- **Why the interviewer asks:** using "covariate shift" to mean "any shift" destroys the diagnostic value of the term, and the correct remedy differs per shift type.
- **Trap:** calling any distribution change "covariate shift." If the label mapping changed, it is concept shift, and domain adaptation will not fix it.

**Q10. What is label shift, and how do you fix it?**
- **Answer:** Label shift (prior shift) is `P(Y)` changing while `P(X|Y)` is unchanged — the class proportions in production differ from training. It breaks accuracy *and* calibration even when the model is perfect. Fix with logit adjustment (add `log π_target(y) − log π_train(y)` to the logits), class weights, or threshold tuning. **Do not retrain.** It is usually detected late, in production, as "the model suddenly predicts one class."
- **Why the interviewer asks:** it is the shift that tempts teams into an unnecessary training run.
- **Trap:** "the model degraded, retrain it." If the conditional is unchanged there is nothing to learn; retraining on shifted priors bakes the skew in.

**Q11. What is concept shift, and why is it expensive?**
- **Answer:** `P(Y|X)` changes — the same input now has a different label (a policy change, a taxonomy change, a redefinition of "churn"). It is the only shift that genuinely requires new labels. It is often misdiagnosed as covariate shift, which costs teams a month of domain adaptation for a labeling-policy change.
- **Why the interviewer asks:** it is the highest-cost misdiagnosis in the family, and it is a business conversation as much as a modeling one.
- **Trap:** "we'll do continued pretraining on recent data." Continued pretraining fixes `P(X)`; concept shift is about `P(Y|X)` and needs annotated examples under the new definition.

**Q12. What does layer freezing actually save, and what does it not save?**
- **Answer:** It saves backward-pass compute, gradient memory, and optimizer state (2 fp32 moments per trainable parameter). It does **not** save the forward pass — a frozen layer still computes its activations — and it does not change inference latency at all. In PyTorch, `requires_grad=False` alone does not even drop the activation memory for the frozen prefix; you must also wrap it in `torch.no_grad()`. On a 7B model, freezing 90% cuts optimizer state from 84 GB to 8.4 GB.
- **Why the interviewer asks:** "freezing saves memory" is true but incomplete in exactly the way that causes a failed training job.
- **Trap:** "freezing means it doesn't compute them." It computes them; it just does not differentiate through them. This is the misconception the video's own rule depends on getting right.

**Q13. What is gradual unfreezing?**
- **Answer:** ULMFiT's third technique. Train with the top layer only (epoch 1), then the last two (epoch 2), then the last three (epoch 3), descending toward the embeddings. It gives the head time to become useful before the backbone moves, so the large gradients from a randomly-initialized head never reach the pretrained body. It also happens to be strictly cheaper for the first epochs. Typical gain: +0.3 to +1.0 point with measurably less forgetting.
- **Why the interviewer asks:** it is a schedule that composes with everything else, and ~25 lines of code.
- **Trap:** conflating it with discriminative learning rates. They are different mechanisms — one is a schedule over *time*, the other is a rate per *layer* — and they stack.

**Q14. What is discriminative fine-tuning (LLRD)?**
- **Answer:** Layer-wise learning-rate decay: each lower layer receives a smaller LR than the layer above, typically by a constant factor. ULMFiT's rule is `η^{l−1} = η^l / 2.6`; production PyTorch usually uses a decay of 0.8–0.9 with the head at `base_lr × 2`. Rationale: lower layers hold general features that should barely move; the head needs to move fast. One LR for the whole model forces a compromise between those two requirements.
- **Why the interviewer asks:** it is the difference between a truncated full FT and a scheduled one, and it is the highest gain-per-line-of-code item in CS-02's ranked freezing table (+0.3–0.8 pt for ~15 lines).
- **Trap:** "discriminative LR = a lower LR." It is not lower, it is *layered*: the head may get `4e-5` while the embeddings get `4e-5 × 0.85^12 ≈ 6.5e-6`.

**Q15. What is ULMFiT and why does it matter in 2025?**
- **Answer:** Universal Language Model Fine-tuning (Howard & Ruder, ACL 2018) — the paper that introduced discriminative learning rates, the slanted triangular LR schedule (`cut_frac=0.1`, `ratio=32`), and gradual unfreezing for LM fine-tuning. It predates BERT by roughly eight months, and essentially every production layer-freezing schedule descends from it. It is also the source of the data-efficiency anchor: 100 labeled examples matching from-scratch training on 10× more data.
- **Why the interviewer asks:** credit for these schedules is usually given to BERT; a candidate who knows the provenance has actually read the literature.
- **Trap:** "ULMFiT is an old LSTM method, not relevant." The schedules are the relevance; they are the ones people re-derive badly from scratch.

**Q16. What is LP-FT and what does it claim?**
- **Answer:** Linear-probe then fine-tune (Kumar et al., ICLR 2022). Phase 1: freeze the whole backbone, train the head at a high LR (1e-3) for a few epochs. Phase 2: unfreeze everything and fine-tune with a *small* LR (2e-5) for a short time. Claim: it beats both pure linear probing and pure fine-tuning, in-distribution and out-of-distribution, because phase 1 replaces the random head before any gradient reaches the backbone, so phase 2 starts from a low-loss point and the gradients reaching the backbone are small and task-aligned instead of large and noise-driven.
- **Why the interviewer asks:** it is the highest-value six lines of code in the module and a good proxy for whether the candidate reads papers or only blog posts.
- **Trap:** "that's just gradual unfreezing." Related mechanism, different schedule: LP-FT unfreezes *everything at once* in phase 2, whereas gradual unfreezing descends block by block.

**Q17. What is the alignment tax?**
- **Answer:** The measurable loss of general capability incurred by aligning or fine-tuning a model. Named in InstructGPT (Ouyang et al., 2022), which reports the tax on public NLP benchmarks and shows that mixing pretraining gradients into the RL objective (`PPO-ptx`) recovers most of it. It is the industrial name for catastrophic forgetting. "We only did SFT" is not an exemption — narrow SFT carries a tax too.
- **Why the interviewer asks:** it is the term that appears in an actual production review, and CS-02's case study reports concrete magnitudes (MMLU −0.6, WikiText ppl ratio 1.07).
- **Trap:** treating it as an RLHF-only phenomenon. Any gradient update on a narrow distribution produces it.

**Q18. What is the "head" of a model, and what fixes its input and output dimensions?**
- **Answer:** The task-specific module on top of the backbone — `Dense(1)`, `Linear(768, num_labels)`, a causal-LM `lm_head`. Its **input** dimension is fixed by the backbone's hidden size; its **output** dimension is fixed by your label space. It is always trainable. On VGG16 the video's head is `Flatten → Dense(256) → Dense(1)` = 6,423,041 parameters (4.64% of 138,357,544); on BERT-base it is a single `Linear(768, 3)` = 2,307 parameters.
- **Why the interviewer asks:** the head is where most of the practical mistakes live (`num_labels`, sigmoid vs softmax, loss function choice).
- **Trap:** initializing the new head from a slice of the old one. Replacing a 1,000-class head with a 3-class head throws away 4M parameters and *should* — the old head's rows encode ImageNet semantics that have nothing to do with your labels.

**Q19. What does `include_top=False` do in Keras, and what is the silent trap?**
- **Answer:** It drops the pretrained classification head so you can bolt on your own. The video uses it at [43:52]. The trap: `tf.keras.applications.VGG16(include_top=False)` also exposes a `classes` argument and a `classifier_activation`; leaving `weights='imagenet'` with an inconsistent `classes` value, or forgetting `include_top=False` entirely, silently gives you the 1,000-class ImageNet head and a shape mismatch (or worse, a silently wrong output space). Always confirm with `model.summary()`.
- **Why the interviewer asks:** it is the first line of the first vision fine-tuning notebook anyone writes.
- **Trap:** "`include_top=False` gives me a model with no output layer, so `num_classes` is irrelevant." It is not irrelevant — it is what sizes the replacement.

**Q20. What does `requires_grad` do, and what are the three ways it is not enough?**
- **Answer:** PyTorch's per-tensor flag; `False` freezes that parameter so no gradient is computed or applied. It is not enough because: (1) it does not drop the frozen prefix's activations unless you also use `torch.no_grad()` or `torch.utils.checkpoint`; (2) it does nothing to BatchNorm running statistics, which keep updating in PyTorch unless you call `.eval()`; (3) in Keras the equivalent `trainable` flag is baked at `compile()` time — setting it afterwards is a no-op until you recompile.
- **Why the interviewer asks:** this is where the "I froze it and it still OOM'd / still drifted" incidents come from.
- **Trap:** "I set requires_grad=False, so the layer is frozen." Verify it: print the trainable count, print the first trainable tensor name, and assert `p.grad is None` for every frozen tensor after one step.

**Q21. State the video's three ways to fine-tune a CNN, in increasing cost.**
- **Answer:** [18:28]–[19:28] (1) **Replace the output layer** — 4,097 trainable parameters for a 2-class head on VGG16, 0.003% of the model. (2) **Freeze the convolutional base and train a new dense head** — `Flatten → Dense(256) → Dense(1)` = 6,423,041 trainable, 4.64%. (3) **Unfreeze the last convolution block(s) and retrain them** — block5 adds 7,079,424, for 13,502,465 trainable, 9.76%. The transformer case is structurally identical [30:01]–[31:32], applied to encoder blocks instead of conv blocks: `model.bert.encoder.layer[-2:]` plus `model.classifier`.
- **Why the interviewer asks:** it is the module's spine, and it lets the interviewer move straight to "what are the parameter counts, and which one would you pick?"
- **Trap:** presenting them as a strict quality ranking. They are a ranking by *capacity*. By expected quality on small or shifted data the ordering can invert (linear probe > full FT out-of-distribution).

**Q22. Why does the instructor say we should never fine-tune the earliest layers?**
- **Answer:** Two arguments, and he only gives the first. **Feature hierarchy:** early layers fetch primitive features — "edge, texture, shape" — that are task-agnostic, so there is nothing task-specific to gain by moving them [21:17]. **Anti-forgetting:** the early layers hold the most general and most fragile representation; gradient steps taken to fit a small target set will overwrite them, degrading everything else. Both point the same direction, which is why the rule survives. His operational form: "we never train the entire model. We just unfreeze some last layer" [21:41], with a claimed 99% success rate [22:07].
- **Why the interviewer asks:** the candidate should be able to supply the second argument, which the lecture omits — it is the one that tells you what to do when freezing is not an option.
- **Trap:** repeating the 99% number as a general truth. It refers to the frozen-base + new-head configuration on an in-distribution Q2 task (cat vs dog, ImageNet-pretrained, 25k images) — the easiest possible quadrant. Do not read it into a Q4 problem.

**Q23. What is BERT's pooler output and why does it matter to the head?**
- **Answer:** The `tanh`-squashed transformation of the `[CLS]` token, `tanh(W · cls + b)` with `W` of shape 768×768. `BertForSequenceClassification` feeds `pooler_output` — not the raw `[CLS]` vector, and not a mean pool — into `classifier`. That is why the head is a single `Linear(768, num_labels)`. It matters operationally because the pooler is 590,592 parameters and the video's freeze-then-unfreeze snippet leaves it frozen while training `encoder.layer[-2:]` and `classifier`, which is a real bottleneck on adaptivity.
- **Why the interviewer asks:** it distinguishes people who have read the transformers source from people who have read the docs.
- **Trap:** assuming the `[CLS]` vector is used raw, or that mean pooling is what HF does for classification. `sentence-transformers` uses mean pooling; `BertForSequenceClassification` uses the pooler.

**Q24. What does `num_labels` control, and what happens if it is wrong?**
- **Answer:** It sizes the classification head: `Linear(hidden, num_labels)`. Wrong value = wrong loss baseline, silently. If `num_labels` exceeds the number of distinct labels present, the model trains an extra head row that is never supervised — the loss floors near `ln(k')` and one class is never predicted. If it is below the max label id, you get a loud `IndexError: Target k is out of bounds` on the first batch containing that class. The video changes it from the default 2 to 3 at [1:01:33], and that single line is where the emotion-notebook bugs originate.
- **Why the interviewer asks:** it is the cheapest, most common, most diagnosable bug in the module, and it has a signature (the `ln(k)` loss floor) that an experienced engineer recognizes instantly.
- **Trap:** "the loss floor tells me the LR is wrong." A perfectly flat loss at exactly `ln(2) = 0.693` (binary) or `ln(3) = 1.099` (3-class) is a head/label problem, not an optimization problem.

**Q25. In one sentence, what is LoRA?**
- **Answer:** Low-Rank Adaptation: freeze `W`, learn `ΔW = BA` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `r ≪ min(d,k)`, so you train 0.1–1% of parameters, store 10–200 MB adapters per task, and can merge or hot-swap them at inference. The base weights *cannot* move, which makes LoRA a regularizer as well as a memory trick.
- **Why the interviewer asks:** it is the default adaptation method above ~3B parameters and it is directly forward-referenced by the video [29:00]–[32:34].
- **Trap:** "LoRA selects a subset of the existing weights." It does not select — it *adds* a pair of low-rank matrices alongside a weight that is frozen and fully present. The subset family (BitFit: bias terms only, ~0.08%) is a different mechanism, and the distinction matters at merge/serve time.

**Q26. What is continued pretraining (DAPT) and when is it the first move?**
- **Answer:** Domain-Adaptive Pretraining — running the *pretraining* objective (MLM or causal LM) over unlabeled target-domain text before any supervised fine-tuning. It is the correct first move for Q3/Q4 (domain shift). Gururangan et al., *Don't Stop Pretraining*, ACL 2020. CS-02's clinical de-identification case: 2.1B tokens of unlabeled notes, `lr=5e-5`, 1 epoch, 19 hours on 2×A100 = $76, which bought +10.8 entity-level F1 over skipping it, while the entire supervised phase took 18 minutes and $0.60.
- **Why the interviewer asks:** it is the decision that separates an 0.804 F1 from a 0.912 F1, and it is the answer to "the domain is different, what do you do?"
- **Trap:** "we fine-tuned on our PDFs" usually means SFT on QA pairs extracted from the PDFs — a different and much weaker intervention for domain shift. The unlabeled-data phase is where the accuracy lives; the labeled-data phase is where the product lives.

**Q27. Is fine-tuning a good way to teach a model new facts?**
- **Answer:** No. SFT teaches *behavior* — format, tone, decision boundaries, refusal policy. It encodes facts probabilistically, so if your eval asks "what is our refund window?" and the training set contained the answer 500 times, you may get it right, and you will also get it wrong at a rate nobody can predict. New facts need continued pretraining (1B+ tokens) or retrieval (RAG). LoRA can learn new behavior; new facts need a knowledge-injection mechanism.
- **Why the interviewer asks:** it is the most expensive wrong turn in enterprise fine-tuning — months of SFT on a product catalog.
- **Trap:** "we'll fine-tune on our documentation and the model will know it." Facts are volatile and revocable; retrieval is exact and revocable. Encoded facts are neither.

**Q28. Does fine-tuning make a model faster?**
- **Answer:** No. Fine-tuning changes *what* the model says, not how fast it says it. A frozen layer still runs its forward pass; the trained artifact has the same architecture and the same parameter count. Serving latency is unchanged, with one caveat: unmerged LoRA adapters add 1–15 ms of overhead per request (CS-02 measured +11 ms p95 for multi-LoRA serving on a 40-tenant workload). If you have a latency budget, the answer is distillation or a smaller model, not fine-tuning.
- **Why the interviewer asks:** it is a common stakeholder expectation and a candidate should be able to refute it in one sentence.
- **Trap:** "we merged the adapter so it's faster than the base model." Merging removes adapter overhead; it does not make the model faster than the original base. In CS-02's case, serving the merged model in fp16 without re-quantizing was 1.7× *slower* than the quantized base — a serving choice, not a training one.

---

## Level 2 — Applied & Implementation

**Q29. Walk me through freezing a Hugging Face encoder classifier and unfreezing only the last two blocks.**
- **Answer:**
```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

for p in model.parameters():
    p.requires_grad = False                      # freeze everything first  [1:05:47]

for layer in model.bert.encoder.layer[-2:]:      # "minus2 colon" — last two blocks [1:06:37]
    for p in layer.parameters():
        p.requires_grad = True

for p in model.classifier.parameters():
    p.requires_grad = True                       # [1:08:19]

# ALWAYS print the receipt:
tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
tt = sum(p.numel() for p in model.parameters())
print(f"{tr:,}/{tt:,} = {100*tr/tt:.3f}%")
# -> 14,178,051 / 109,484,547 = 12.950%
```
- **Why the interviewer asks:** this is the exact 8 lines the video writes, and it contains a known bug (see Q33). Asking for it verbatim-then-critical is a reliable seniority probe.
- **Trap:** building the optimizer *before* setting `requires_grad`. Adam will keep moments for parameters that no longer receive gradients — they never change but they still consume memory, and if you unfreeze after building the optimizer those parameters are silently absent from the param groups and never train.

**Q30. How do you freeze a backbone in PyTorch such that you actually save the activation memory?**
- **Answer:** `requires_grad=False` stops gradients but not activation retention. Wrap the frozen prefix:
```python
for p in model.conv_base.parameters():
    p.requires_grad = False

with torch.no_grad():                    # <-- the line people forget
    features = model.conv_base(x)        # no graph retained
features = features.detach()
logits = model.head(features)            # graph exists only from here
loss = criterion(logits, y)
loss.backward()
```
This is the difference between a VGG16 fine-tune fitting in ~4 GB and needing ~11 GB.
- **Why the interviewer asks:** it separates people who have watched a training job OOM from people who have only read tutorials.
- **Trap:** "`requires_grad=False` on the first frozen layer's output is enough." It is not — you need the `no_grad()` context (or `torch.utils.checkpoint`) around the whole frozen segment, and `.detach()` so the head's graph does not reach back through the features.

**Q31. What is the Keras equivalent, and what is the #1 reason "freezing didn't work" in Keras?**
- **Answer:** `conv_base.trainable = False`, or a per-layer loop flipping `layer.trainable`. The #1 failure is that **Keras bakes the trainable set into the compiled train function** — setting `layer.trainable = False` after `model.compile()` has no effect until you recompile. In Keras, `trainable=False` also runs frozen layers in inference mode, so BatchNorm running statistics stop updating (unlike PyTorch).
- **Why the interviewer asks:** it is the most common Keras-specific incident, and the symptom (trainable count looks right, nothing learns) mimics three other bugs.
- **Trap:** checking `model.summary()` and trusting it. The summary reflects the flags, not the compiled function. Recompile, then re-summary.

**Q32. Give me learning rates for pretraining, full fine-tuning, head-only, and LoRA.**
- **Answer:**
| Regime | Peak LR (AdamW) | Why |
|---|---|---|
| Pretraining from scratch (LLM) | 1e-4 – 6e-4 | Random init; nothing to destroy |
| Full fine-tuning, transformer | **1e-5 – 5e-5** | 10–100× smaller; protecting the pretrained basin |
| Linear probe / frozen base + head | 1e-3 – 1e-2 | The head is effectively from scratch; nothing pretrained is at risk |
| LoRA / QLoRA adapters | 1e-4 – 3e-4 | Higher than full FT — `B` and `A` start at zero and must travel further |
| CV fine-tuning (SGD) | 1e-3 – 1e-4 | VGG-16 Way 2 used Keras's Adam default 1e-3 with the base frozen; Way 3 unfroze block5 at **1e-5** |
Anything ≥1e-4 for full fine-tuning of a pretrained encoder destroys the checkpoint faster than the task loss can rebuild it. The video does not say the 1e-5 aloud, but unfreezing pretrained convolutions at 1e-3 is the single most common way people destroy a good checkpoint.
- **Why the interviewer asks:** "fine-tuning LRs are 10–100× smaller than pretraining LRs" is the single most load-bearing number in the module. An answer of "2e-4 for everything" ends the interview.
- **Trap:** "LoRA should use a smaller LR because there are fewer parameters." Backwards. Adapter matrices initialize at zero, so they start further from a good solution than a pretrained weight does; LoRA runs *higher* than full FT, not lower.

**Q33. The video's freeze snippet freezes `layer[-2:]` and `classifier`. What does it get wrong?**
- **Answer:** It leaves `bert.pooler` frozen. `model.parameters()` is frozen wholesale, then only the encoder tail and the classifier are re-enabled — but `BertForSequenceClassification` feeds `pooler_output = tanh(W·cls + b)` into `classifier`. A frozen pooler is a real bottleneck on how much the head can adapt: 590,592 parameters (768×768 + 768). The fix is one line — unfreeze anything whose name starts with `bert.pooler` — or better, unfreeze by name rather than index.
- **Why the interviewer asks:** it is a subtle correctness bug in code most candidates have copied, and it demonstrates whether you read the model's forward pass or just its layer list.
- **Trap:** "the pooler has 0 parameters, like a pooling layer." It is a dense layer with a `tanh`; 590,592 parameters on BERT-base. Only the *attention* pooling/max-pool layers in vision have zero parameters.

**Q34. How do you make unfreezing robust across `transformers` versions?**
- **Answer:** Resolve modules by name, not by index:
```python
def set_trainable_by_unfreezing_top(model, n_unfreeze, prefix="bert.encoder.layer"):
    for p in model.parameters():
        p.requires_grad = False
    blocks = [m for n, m in model.named_modules() if n.startswith(prefix) and m is not model]
    for blk in blocks[-n_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
    for n, p in model.named_parameters():
        if n.startswith("classifier") or n.startswith("bert.pooler"):
            p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"trainable {tr:,}/{tt:,} = {100*tr/tt:.2f}%")
    return model
```
- **Why the interviewer asks:** index-based unfreezing breaks silently on architecture changes, and a silent mis-slice is hard to detect downstream.
- **Trap:** trusting `blocks[-n:]` without checking registration order. Print the resolved names once and assert the first trainable tensor is the one you expect; the same trap exists in torchvision, where VGG16's block5 is indices `24:31`, not "the last 3 layers."

**Q35. How do you prove freezing actually took effect, in four checks?**
- **Answer:**
```python
# 1. Trainable parameter count, and its fraction of the total
print(count_trainable(model))                     # e.g. (14178051, 109484547)

# 2. The FIRST trainable tensor — catches "we unfroze the embeddings by accident"
first = next(n for n, p in model.named_parameters() if p.requires_grad)
print("first trainable tensor:", first)

# 3. One dry forward+backward; no gradient may exist on a frozen tensor
out = model(**{k: v[:2] for k, v in batch.items()})
out.loss.backward()
for n, p in model.named_parameters():
    if p.grad is not None and not p.requires_grad:
        raise RuntimeError(f"gradient leaked into frozen tensor {n}")

# 4. The loss at initialization ≈ ln(num_labels)
assert abs(out.loss.item() - math.log(3)) < 0.1
```
- **Why the interviewer asks:** a 30-second pre-flight that catches four of the ten silent failures in CS-02's table. Candidates who do not have such a ritual will ship a frozen model and report a flat loss.
- **Trap:** "the summary showed the count, so it worked." The count proves the *flags*; the dry step proves the *optimizer* sees them. Both are needed, because freezing after optimizer construction produces the right count and no learning.

**Q36. How much warmup for fine-tuning, and why is it more important than in pretraining?**
- **Answer:** 6–10% of optimizer steps — ~150 steps on a 2,000-step run, i.e. `warmup_ratio=0.06`–`0.1`. It is more important than in pretraining, counter-intuitively, because a pretrained model sits in a sharp, well-adapted basin: the first Adam step can move a weight by a full `η` in the direction of a randomly-initialized head's gradient, applied simultaneously to every parameter. A random-init model has nothing to damage. Zero warmup with a random head is the second most common cause of "fine-tuning didn't work."
- **Why the interviewer asks:** warmup is the free knob nobody sets, and the candidate should know the interaction with `gradient_accumulation_steps` (HF's `warmup_ratio` is a fraction of *optimizer* steps, not micro-batches).
- **Trap:** "warmup matters less for fine-tuning because the LR is smaller." The magnitude of the first step scales with `η`, but the *damage* scales with how fragile the starting point is; the pretrained point is far more fragile than random init.

**Q37. How many epochs for a classification fine-tune versus instruction SFT?**
- **Answer:** 2–4 epochs on a classification set (the video runs 1 for time [1:03:40] and 2 in the vision notebook — both are demos, not recipes); 1–3 epochs on instruction data; 10–30 for CV on small data. The failure signature of too many epochs is *not* overfitting on the target — target metrics keep climbing — but a rising general loss and a falling general benchmark score. Set `save_strategy="epoch"` with `load_best_model_at_end=True` on a metric that includes the general suite, or you will ship the most-forgotten checkpoint.
- **Why the interviewer asks:** "more epochs is better" is the most expensive misconception in the module. CS-02's measured row: full FT at LR 5e-5 for 10 epochs scored 0.918 versus 0.933 for LR 2e-5 for 3 epochs — 3× the compute for a worse model.
- **Trap:** early-stopping on the target metric alone. The target metric is monotone because it cannot see forgetting. Early stop on the *pair* (target up, general flat).

**Q38. What weight decay do you use for fine-tuning, and how does it relate to L2-SP?**
- **Answer:** 0.01 for transformer fine-tuning, not the 0.1 of pretraining; 0.0–0.01 for a head-only run. Weight decay is L2 toward *zero*, which is a different objective from staying near the pretrained weights. If your goal is anti-forgetting, use **L2-SP** — `+ (λ/2)·‖θ − θ*‖²` with `λ ∈ [1e-4, 1e-2]` — which directly encodes "stay near the pretrained solution." Turning up weight decay on a pretrained model degrades it in a way that looks like forgetting but is not.
- **Why the interviewer asks:** the distinction is the difference between a regularizer that helps and one that actively hurts a fine-tune.
- **Trap:** "weight decay and L2-SP are both L2 penalties so they're equivalent." They pull toward different points: zero versus `θ*`.

**Q39. What batch size fits on a 16 GB card for BERT-base at `max_len=128`, and what do you do if you need a bigger effective batch?**
- **Answer:** Batch 16 fits comfortably (~4.2 GB total for full FT per CS-02's table); batch 64 does not (~10.8 GB, plus fragmentation). Use `gradient_accumulation_steps` to reach an effective batch of 32–64: `per_device_train_batch_size=8, gradient_accumulation_steps=4`. Note that batch-64 eval needs `per_device_eval_batch_size` set explicitly — leaving it at the default with `max_len=512` is the classic "OOM only at eval."
- **Why the interviewer asks:** it tests whether the candidate has actually run a training job on a small card, plus the LR/batch interaction (double the batch → scale LR by √2 in a linear-schedule regime).
- **Trap:** "raise the LR to compensate for the bigger batch and skip accumulation." Scaling LR 10× without scaling batch gives divergence, not a faster run.

**Q40. bf16 or fp16 for fine-tuning?**
- **Answer:** bf16 on any Ampere-or-later GPU: same memory as fp16, no loss-scaling fragility, negligible accuracy delta. On a T4/P100 (no bf16 support) use fp16 with `fp16=True` in `TrainingArguments` (which enables loss scaling) or just fp32. Head-only runs are fine in fp32 — the memory is dominated by the frozen backbone, not the optimizer. For QLoRA, compute dtype is bf16 on top of an NF4 base.
- **Why the interviewer asks:** precision is where silent NaNs come from, and the T4 (Colab free tier, no bf16) is the most common fine-tuning GPU in the world.
- **Trap:** "bf16 and fp16 both just halve the memory, so use fp16 because it's more precise." fp16 has a narrow exponent range and overflows without loss scaling; bf16 has fp32's exponent range and fp16's mantissa, so it is the safer default wherever it is supported.

**Q41. How do you choose `max_len`?**
- **Answer:** Measure, do not guess: `np.percentile([len(t) for t in tok(texts)["input_ids"]], [50, 95, 99])`. Classification rarely needs more than 128–256 tokens; SFT typically 1k–4k. Doubling `max_len` doubles activation memory *and* quadruples attention cost (O(n²)). Two failure modes: too short → you truncate the label-bearing tokens and the model plateaus low with a healthy-looking loss curve; too long → you OOM and pay 2× for tokens that were padding.
- **Why the interviewer asks:** it is the most under-thought hyperparameter, and truncation is a *silent* failure.
- **Trap:** "512 is the max BERT supports so I use 512." BERT's positional limit is not the same as your data's requirement, and paying 512 tokens for a 22-token emotion sentence is 4× the attention cost for zero signal.

**Q42. How do you decide which layers to unfreeze? Give me the arithmetic.**
- **Answer:** Count parameters; do not use intuition.
| Model | Total | Last block | Head-only trainable | "Last block + head" as % |
|---|---|---|---|---|
| VGG16 | 138,357,544 | 7,079,424 (block5) | 4,097 (0.003%) | 9.76% |
| BERT-base | 109,482,240 | 7,087,872 (block 11) | 2,307 (0.002%) | 12.95% for last-2 + head |
| Llama-3-8B | 8,030,261,248 | ~218M/block | — | ~17% for last-4 + head (embeddings ≈ 525M) |
Then print `sum(p.numel() for p in model.parameters() if p.requires_grad) / total`. **If the fraction exceeds ~10%, you are doing full fine-tuning with extra steps** — either commit to full FT (with its 16 bytes/param cost) or switch to LoRA. This is exactly why the instructor pivots to PEFT for Llama/Mistral [32:09]–[32:41]: the CNN-era intuition "the last block is small" was true in 2014 and is false for a 7B decoder.
- **Why the interviewer asks:** it tests whether the candidate ports 2014-era intuition into 2025-scale models. The number is the answer, and candidates who answer with adjectives fail.
- **Trap:** "unfreezing the last block keeps it cheap on any model." On Llama-3-8B one decoder block is ~218M parameters and the embedding table is ~525M tied to `lm_head` — "just the last few blocks" is 1.4B parameters, 17% of the model.

**Q43. How do you implement replay/rehearsal in a fine-tuning job, and how much general data do you mix in?**
- **Answer:** Mix 1–10% general-domain data into every batch (5% is a common production default). Implementation is at the sampler or dataset level:
```python
# HF: interleave the target and general datasets at a fixed ratio
from datasets import interleave_datasets
train = interleave_datasets([target_ds, general_ds], probabilities=[0.95, 0.05], seed=42)
```
It is the single most reliable anti-forgetting method because it restores a source-distribution gradient signal in every step. It is standard in production SFT everywhere (PPO-ptx in InstructGPT is the same idea inside RL).
- **Why the interviewer asks:** it is the cheapest mitigation that actually works at scale and it requires no coefficient tuning, unlike EWC or KL-to-base.
- **Trap:** replaying data from the *same* domain. If both streams come from your support tickets, you have added compute and changed nothing about the forgetting, because replay must be distributionally different from the target to carry source signal.

**Q44. What is EWC, and why is it rarely used for LLM SFT?**
- **Answer:** Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017) adds `(λ/2)·Σ_i F_i (θ_i − θ*_i)²`, where `F` is the diagonal Fisher information — the penalty is weighted by how important each parameter was to the source task. It requires an extra pass over source data to estimate the Fisher, and `λ` is finicky (`λ_EWC ∈ [1e2, 1e4]` because Fisher values are tiny). For LLM SFT it is rarely used: the Fisher pass is expensive at scale, `λ` needs tuning per dataset, and replay plus a low LR gets most of the benefit for none of the machinery.
- **Why the interviewer asks:** it is the canonical continual-learning answer, and the follow-up is "so why don't you use it?" — the answer should be cost, not ignorance.
- **Trap:** "EWC is free — it's one extra term in the loss." The Fisher estimate is an additional forward/backward pass over source data plus storage for one Fisher diagonal per parameter.

**Q45. What is L2-SP and how is it coded?**
- **Answer:** `L_total = L_task + (λ/2)·‖θ − θ*‖²` — an L2 penalty toward the *pretrained* weights rather than toward zero (Xuhong et al., ICML 2018). One extra term, two lines of code:
```python
for n, p in model.named_parameters():
    if n in pretrained:                       # snapshot of θ* taken before training
        loss = loss + (lam / 2) * ((p - pretrained[n]) ** 2).sum()
```
`λ ∈ [1e-4, 1e-2]`. It is the simplest baseline that directly encodes "stay near the pretrained solution."
- **Why the interviewer asks:** it is the one-line version of the anti-forgetting idea, and it distinguishes "I know L2 regularization" from "I know the fine-tuning literature."
- **Trap:** "I already use weight decay, that IS L2-SP." Weight decay's reference point is the origin, not `θ*`.

**Q46. How do you implement KL-to-base regularization and what does it cost?**
- **Answer:** Add `β · KL(p_base ‖ p_tuned)` on a held-out general corpus to the training loss:
```python
with torch.no_grad():
    base_logits = base_model(**batch).logits        # base must stay resident
kl = F.kl_div(F.log_softmax(tuned_logits, -1),
              torch.softmax(base_logits, -1),
              reduction="batchmean", log_target=False)
loss = task_loss + beta * kl                        # beta in [0.01, 0.5] for SFT
```
Cost: the base model stays resident in VRAM (or you cache its logits over the general corpus and replay them). It is simultaneously the most direct measurement and the most direct mitigation of forgetting.
- **Why the interviewer asks:** it is the method that makes the forgetting quantity you report in CI identical to the quantity you optimize.
- **Trap:** "log_target=True is the default." `kl_div`'s default is `log_target=False`, meaning the target argument must be *probabilities*, not log-probabilities. Getting this backwards produces a nonsensical but finite loss, which is worse than an error.

**Q47. How do you measure forgetting for an encoder classifier versus a decoder LLM?**
- **Answer:** For an LLM: `lm-evaluation-harness` on MMLU, ARC-Challenge, HellaSwag before and after, plus WikiText perplexity. For an encoder classifier: freeze the *base* model's pooled embeddings over 5,000 held-out sentences before and after, and report mean cosine distance. Thresholds that have held up:
| Signal | OK | Investigate | Stop and re-plan |
|---|---|---|---|
| MMLU delta | > −0.5 | −0.5 to −2.0 | < −2.0 |
| WikiText ppl ratio (tuned/base) | < 1.05 | 1.05–1.20 | > 1.20 |
| KL-to-base, 2k general prompts | < 0.02 nats | 0.02–0.10 | > 0.10 |
| Base-embedding cosine drift | < 0.03 | 0.03–0.10 | > 0.10 |
Level 0 (target metric only) is not a measurement — it *cannot* see forgetting.
- **Why the interviewer asks:** it tests whether the candidate has an actual instrument, not an opinion. "We watched the loss" is the wrong answer.
- **Trap:** reporting only the target metric and calling it evaluation. Every level-0 report has the same blind spot, which is exactly the one you hired the eval for.

**Q48. How much labeled data do you need for a few-class classification fine-tune?**
- **Answer:** In-distribution, the knee is **10–30 examples per class (~100 total)** and the plateau arrives at 1k–5k. Head-only is often optimal below 500. Fine-grained classification (100+ subtle classes) needs 50–100 per class and 10k–50k total. The rules of thumb: 10 × the number of classes is the absolute floor for a head-only fine-tune; 100× the labels buys you ~1 point past the knee; if you cannot reach the plateau with 10× your current data, the problem is the representation (domain shift), not the volume.
- **Why the interviewer asks:** it is the budget question every project starts with. A candidate who says "more data is always better" has not read the curves.
- **Trap:** "below 500 examples, fine-tune harder." Below ~1,000 examples the optimum shifts toward *more* frozen layers and lower LR, not more capacity — CS-02's VGG case measured unfreezing block4 on 2,000 images and losing accuracy (0.978 → 0.964).

**Q49. How much data for instruction SFT — and does the answer differ for format versus behavior?**
- **Answer:** Yes, sharply.
| Goal | Labels needed | Plateau |
|---|---|---|
| Style / format / JSON schema | **500–2,000** | 5k–10k |
| New domain behavior | 5,000–50,000 | 100k+ |
| New factual knowledge | — (not achievable by SFT) | Use continued pretraining or RAG |
The anchors: LIMA (1,000 curated instructions → competitive chat model), Alpaca (52k self-instruct for <$600), InstructGPT (13k SFT demonstrations). 500–2,000 examples teach *format*; 10k+ teach *behavior*; nothing teaches *facts*.
- **Why the interviewer asks:** it converts a vague "we need more data" into a budget, and the format/behavior/fact trichotomy is the most useful planning heuristic in the module.
- **Trap:** "1,000 examples got LIMA a good chat model, so 1,000 is enough for our new domain behavior." LIMA's 1,000 examples were curated for *format and style* against an already-strong base. A new decision boundary is a behavior, and behavior is the 10k+ column.

**Q50. The video's emotion notebook shows 9,749 train / 2,438 validation. Reproduce that arithmetic.**
- **Answer:** `dair-ai/emotion`'s train split has 16,000 rows over 6 classes. Filter to sadness (4,666), joy (5,362), anger (2,159) → **12,187** rows. An 80/20 stratified split gives **9,749 / 2,438**, byte-for-byte the counts on screen at [54:39] and [55:10]. Per-class: sadness 3,732/934, joy 4,290/1,072, anger 1,727/432. Verified with `train_test_split(test_size=0.2, seed=42, stratify_by_column="label")`.
- **Why the interviewer asks:** it proves the candidate actually touched the data rather than paraphrasing the video, and it opens the label-remap follow-up (Q51) which is the module's best trap.
- **Trap:** taking the split from the dataset's own `validation`/`test` splits and getting different numbers — or, worse, splitting the *full* dataset including the authors' held-out test set, which silently contaminates the final number.

**Q51. What is the label-remap bug in the emotion notebook, and what are its two symptoms?**
- **Answer:** `Dataset.filter()` does **not** renumber labels. `dair-ai/emotion`'s `ClassLabel` order is `['sadness','joy','love','anger','fear','surprise']`, so keeping sadness/joy/anger leaves the label column holding `{0, 1, 3}` while `num_labels=3` expects `{0, 1, 2}`. The video's claim "0 means sadness, 1 means joy and second means anger" [54:55] is wrong without an explicit remap.
  - **Symptom A (loud, good):** `IndexError: Target 3 is out of bounds` on the first batch containing an anger example.
  - **Symptom B (silent):** if you set `num_labels=4` to dodge the error, you train a four-way head with class 2 permanently empty — loss floors near `ln(4) = 1.386`, accuracy caps below 1.0, and the model never predicts the third class.
  - **Fix:** `remap = {0: 0, 1: 1, 3: 2}` applied with `ds.map()` after filtering.
- **Why the interviewer asks:** it is the single most likely reason a candidate's reproduction diverges from a tutorial's description, and the two-symptom structure tests diagnostic reasoning rather than memorization.
- **Trap:** "`filter` keeps the label ids contiguous because it rebuilds the dataset." It rebuilds the *features*, not the `ClassLabel`'s integer mapping.

**Q52. The video's VGG16 notebook plateaus at ~0.84. What is the one-line fix?**
- **Answer:** The input normalization does not match the backbone's training-time transform. Keras's `VGG16` was trained with `preprocess_input` — RGB→BGR conversion plus ImageNet channel-mean subtraction — while the notebook uses `ImageDataGenerator(rescale=1./255)` [41:25]. The fix is one line: `tf.keras.applications.vgg16.preprocess_input`. Measured cost of getting it wrong: **8–12 points of accuracy**, with a completely healthy-looking loss curve.
- **Why the interviewer asks:** it is the highest-value one-line fix in the module and it mimics "fine-tuning doesn't work for my problem."
- **Trap:** "rescaling to [0,1] is the standard normalization." It is the standard for *your own* network trained from scratch on those images; a pretrained backbone must see the transform it was trained with.

**Q53. How do you configure a multi-label classification head?**
- **Answer:** `num_labels=k` with `problem_type="multi_label_classification"` in the model config, which switches the loss to `BCEWithLogitsLoss`. The head shape is identical to multi-class (`Linear(hidden, k)`); the loss and the metric are not — use per-label sigmoid thresholds and macro-F1/mAP, never argmax over softmax.
- **Why the interviewer asks:** it is a two-line config difference that produces a silently wrong model if you leave it on softmax cross-entropy, and it is a common real-world setting (topic tags, multi-symptom coding).
- **Trap:** "softmax cross-entropy with k classes handles multi-label if labels are mutually exclusive in my data." If they are mutually exclusive you have multi-class, not multi-label — and if they are not, softmax forces the model to choose one and caps recall.

**Q54. The video's BERT run OOMs twice. Diagnose it.**
- **Answer:** The code is fine — a 110M-parameter BERT-base at batch 16 × 128 tokens fits comfortably in 12 GB. The instructor's Colab had exhausted its GPU quota [1:09:52]–[1:10:10], and separately, the notebook holds *two* fully-resident copies of the model (the `BertForSequenceClassification` and a second inside the hand-written `BertClassifier`) plus the tokenized dataset tensors on a T4. The transferable fixes: `del model; gc.collect(); torch.cuda.empty_cache()` between experiments; `per_device_train_batch_size=8` with `gradient_accumulation_steps=2`; and never keep a second model resident to compute a KL diagnostic on the same GPU without freeing the first.
- **Why the interviewer asks:** it tests whether the candidate can separate a *resource* failure from a *code* failure — the single most useful debugging instinct for training jobs.
- **Trap:** "the batch size was too large, the notebook is wrong." Nothing in the notebook is wrong for a 12 GB card; the failure is environmental. A candidate who "fixes" it by shrinking the model has misdiagnosed it.

**Q55. You must serve 40 customers' behaviors from one 24 GB GPU. Design it.**
- **Answer:** One base model (`Llama-3.1-8B-Instruct`) plus 40 unmerged LoRA adapters. Per tenant: 300–800 curated examples, `r=8`, `alpha=16`, `lr=1e-4`, 3 epochs → ~12 min and ~$0.08 per adapter, 21 MB each (840 MB total for all 40). Serve with vLLM's multi-LoRA (`--enable-lora`) and swap the adapter per request: measured p95 overhead **+11 ms**. Keep them unmerged — merging destroys the ability to hot-swap. Two customers with genuinely different output schemas needed `r=32` and `target_modules` expanded to the MLP projections.
- **Why the interviewer asks:** it is the workload where the training decision and the hosting decision are the same decision, and the candidate has to name that coupling.
- **Trap:** "merge the adapters into 40 checkpoints." That is 40 × ~16 GB = 640 GB of artifacts and no per-request swap. `r` and `target_modules` are fixed at train time, so this must be decided *before* training.

**Q56. How much VRAM does a 7B full fine-tune need, and what about QLoRA? Show the arithmetic.**
- **Answer:**
  - **Full FT:** 16 bytes/trainable parameter × 7e9 = **112 GB** (bf16 weights 2 + bf16 grads 2 + fp32 Adam m,v 8 + fp32 master weights 4), plus activations — at bs 4 × len 2048 across 32 layers, several more GB. Realistically ~120–130 GB, needing 2×A100 80 GB with FSDP/ZeRO-3 at minimum. The shorthand: full FT needs about 20× the parameter count in GB, so 7B ≈ 140 GB.
  - **QLoRA:** 4-bit NF4 base ≈ 7e9 × 0.5 bytes = **3.5 GB**, plus adapter optimizer/gradients (~1% of parameters × 16 bytes ≈ 1.1 GB), plus activations (~2–3 GB at bs 4/len 1024 with `gradient_checkpointing=True`) ≈ **7–8 GB** — one 8–12 GB card, or a 24 GB 4090 with headroom.
  - Cost ratio: a 3-hour QLoRA run on a rented 4090 is ~$1.35; a comparable full FT is ~$150 — for what is typically 1–3 points of task accuracy.
- **Why the interviewer asks:** the arithmetic decides the hardware, and the hardware decides whether the project happens. It is also the strongest economic argument for PEFT and the reason the instructor pivots to it [29:00].
- **Trap:** "QLoRA is LoRA with a smaller `r`." It is LoRA on a *4-bit quantized frozen base* with paged optimizers; `r` is independent. Also note the quality caveat: NF4 on a sub-1B base can cost measurable quality, so QLoRA is the right tool at scale, not on small models.

---

## Level 3 — Advanced, Internals & Theory

**Q57. Mechanically, why can full fine-tuning lose to a linear probe out-of-distribution?**
- **Answer:** Kumar et al. (ICLR 2022), *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*. Full FT has enough capacity to fit the target's idiosyncrasies, and the cheapest path to lowering target loss on a small dataset is often to repurpose or overwrite the general features that made the backbone useful elsewhere. On an in-distribution target that is fine — those features were partly redundant. On an OOD target the distorted features are exactly the ones you needed, so FT's higher capacity becomes a liability. The linear probe cannot distort anything: the backbone is frozen, so its features are preserved by construction.
- **Why the interviewer asks:** it is the counter-intuitive result that invalidates the naive "more trainable parameters = better" ranking, and it is the theoretical justification for the probe baseline.
- **Trap:** "that only happens with tiny datasets." It is a property of the distortion, not just the data volume — Kumar et al. show it on ViT and BERT at real dataset sizes when the target is OOD.

**Q58. Why does LP-FT beat both of its own components?**
- **Answer:** Phase 1 (freeze, train head at high LR) replaces the random head with a competent one *before* any gradient reaches the backbone. Phase 2 (fine-tune everything at a small LR) therefore begins from a point where the loss is already low, so the gradients reaching the backbone are small and task-aligned rather than large and noise-driven. Compare a pure fine-tune: on step 0 the head is random, the loss is ~`ln(k)`, and the large resulting gradients flow straight into the pretrained body in whatever direction reduces the loss of a garbage predictor. Compare a pure probe: the features never adapt, so the ceiling is the frozen representation. LP-FT gets the low-loss start *and* the feature adaptation.
- **Why the interviewer asks:** it is the mechanism the video never states, and "which phase does what" is a clean way to test whether the candidate understands gradient flow rather than just recipes.
- **Trap:** "LP-FT is just warmup." Warmup scales the LR schedule; LP-FT changes *which parameters receive gradient* in a separate phase with a separate optimizer state.

**Q59. Why does starting from a pretrained checkpoint converge so much faster than random init?**
- **Answer:** A randomly-initialized deep network sits in a region of parameter space with high curvature, many saddle points, and gradients dominated by input-scale effects. A pretrained network sits in a **flat basin** that already encodes useful features. Gradient descent starting from a good basin finds a good solution in a few thousand steps; the same optimizer from random init needs orders of magnitude more data and steps to find even an acceptable basin. He et al. (ICCV 2019) confirmed the corollary: with enough target data *and a long enough schedule*, from-scratch matches pretrained — pretraining's advantage is **convergence speed**, not a higher ceiling.
- **Why the interviewer asks:** it explains why fine-tuning is short (2–4 epochs) and why it works on 1k examples, and it sets up the "when does from-scratch win" question.
- **Trap:** "pretraining gives a higher ceiling." With enough data it does not — it gives the same ceiling, faster. Zhai et al. (2019) found VTAB task families where *no* pretraining method beat from-scratch.

**Q60. Name the three factors that make catastrophic forgetting worse.**
- **Answer:** (1) **High LR** — drift is roughly proportional to `η·‖g‖·t`, so halving the LR halves the drift. (2) **Many steps on narrow data** — `t` is the killer: 3 epochs × 9,749 examples / batch 16 = 1,828 steps of pure target signal and zero source signal. (3) **A randomly-initialized head feeding gradients into the backbone** — early in training the head produces near-uniform garbage, and the gradients it sends down are large and uninformative. Each factor maps to a mitigation: lower LR, fewer epochs/early stop, LP-FT or gradual unfreezing.
- **Why the interviewer asks:** the three factors are the diagnostic frame — given a forgetting incident, you check these three in order, and it is far more useful than a list of ten techniques.
- **Trap:** "it's about the learning rate." LR is only the first factor; the number of steps on narrow data is the one teams routinely get wrong because their target metric rewards every additional epoch.

**Q61. Why is LoRA a regularizer and not just a memory optimization?**
- **Answer:** Because the base weights *cannot* move. The update is confined to an additive low-rank subspace `ΔW = BA`, so the pretrained features are preserved by construction, and the hypothesis class the optimizer searches is structurally smaller. Biderman et al. (2024), *LoRA Learns Less and Forgets Less*, measured exactly this: LoRA has a lower ceiling on the target (code/math continued pretraining) but measurably less forgetting on general benchmarks than full FT. That is why LoRA frequently *wins* at ≤5k examples — the frozen base is doing the regularization work that weight decay and dropout approximate.
- **Why the interviewer asks:** it reframes PEFT from a cost compromise to a modeling choice, which changes when you would pick it.
- **Trap:** "LoRA is strictly a memory win with a small accuracy cost." At small data the accuracy delta is often *positive*. The right framing is: LoRA trades ceiling for stability, and which one you want depends on your data volume.

**Q62. What is the evidence for "early layers general, late layers specific," and where does it break?**
- **Answer:** The empirical origin is Yosinski et al. (NeurIPS 2014), *How transferable are features in deep neural networks?* — they measured layer-by-layer transferability and found the first layers transfer well and the last layers are specialized, with a fragility boundary where transferring both halves hurts. The instructor's version: early layers "fetch primitive features — edge, texture, shape"; later layers "fetch specific features — how Sunny's nose looks" [21:17]. It breaks in two places: (a) when the target's **low-level statistics** differ from the source's (grayscale medical scans, thermal imagery, 1-channel inputs, spectrograms) — then the primitive features are the *wrong* primitive features and you should re-initialize the stem or freeze less; (b) when the target **modality** differs entirely, the shared part is small and transfer yields less than the hype implies.
- **Why the interviewer asks:** the candidate should know the rule comes from a 2014 vision paper, and should be able to name the exception — every rule has one.
- **Trap:** treating it as a theorem. It is a robust empirical regularity on natural-image and natural-text data, not a guarantee, and the whole "never touch the bottom" heuristic is its crude operational form.

**Q63. What does the BERTology literature actually say about layer-wise function?**
- **Answer:** Tenney et al. (ACL 2019), *BERT Rediscovers the Classical NLP Pipeline*, found a rough ordering — surface features low, then syntax, then semantics, then task-specific — using probing classifiers, but with heavy overlap and no clean boundaries. Rogers et al. (TACL 2020), *A Primer in BERTology*, is the honest survey: the layer-function picture is messier than the slogan, depends on the task, and probing results are affected by probe capacity and by whether the information is *linearly* decodable. Operationally this means: "syntax is in the middle layers" is a defensible heuristic for choosing a freeze boundary, and it is not a guarantee that freezing layer 6 is right for your task.
- **Why the interviewer asks:** it separates a candidate who cites a nice story from one who has read the caveats. The follow-up is "so how do you choose the boundary?" — the answer is *empirically*, by sweeping 0/1/2/4 blocks.
- **Trap:** quoting the pipeline ordering as a law. CS-02's own measurement contradicts the "more is better" reading: on 2,000 images, unfreezing block4 of VGG16 hurt (0.978 → 0.964).

**Q64. Why is warmup *more* important for fine-tuning than for pretraining, counter-intuitively?**
- **Answer:** Because the starting point is more fragile, not less. A pretrained model sits in a sharp, well-adapted basin that is a local optimum for a different distribution; a single Adam step with a large second-moment estimate can move a weight by a full `η` regardless of the gradient's magnitude, and that step is applied simultaneously to 110M parameters in the direction of a randomly-initialized head's gradient. A randomly-initialized model has nothing worth preserving, so a large first step costs it nothing. Warmup converts a blind, model-wide perturbation into a gradual one over the first 6–10% of steps (~150 steps on a 2,000-step run).
- **Why the interviewer asks:** it is the free knob nobody sets, and the counter-intuitive framing identifies people who have reasoned about the mechanism rather than memorized a recipe.
- **Trap:** "warmup is a pretraining thing; fine-tuning runs are short so it doesn't matter." Short runs are precisely where 150 warmup steps are 7.5% of the run and where the first step is the most damaging.

**Q65. Zhang et al. (2021) found that *re-initializing* the top layers helps on few-sample data. Reconcile that with "never break the pretrained representation."**
- **Answer:** The top transformer layers were optimized for the pretraining objective — MLM next-token-style statistics — and on 500 examples those MLM-specialized weights are actively harmful: they encode a task the target does not share, and there is not enough target signal to overwrite them, so they act as a bad prior. Re-initializing the top 2–3 layers to random init before fine-tuning performs better in that regime. The reconciliation is that the pretrained representation is only an asset where the source and target tasks actually share structure; the top of an MLM encoder is the least shared part. It is an exception for very small datasets (hundreds of examples), not a general recommendation.
- **Why the interviewer asks:** it is a genuinely counter-intuitive published result that tests whether the candidate holds beliefs as probabilities or as dogma.
- **Trap:** generalizing it ("re-init the top layers and fine-tune" on 50k examples). With enough data the pretrained top layers are an asset, and re-initializing throws away signal.

**Q66. Why can unfreezing more layers *hurt* on small data, mechanically?**
- **Answer:** Capacity without signal is variance. With 2,000 images and 13.5M trainable parameters, the model has enough degrees of freedom to fit the training set's label noise rather than the task; the extra parameters raise the variance of the estimator faster than they lower its bias. CS-02 measured it: unfreezing block5 took VGG16 from 0.961 → 0.978, then unfreezing block4 as well took it *down* to 0.964. The boundary is empirical, not theoretical, and the way to find it is a sweep over freeze depth with a fixed seed.
- **Why the interviewer asks:** it is the misconception behind most "we fine-tuned a big model on our small data and it got worse" projects.
- **Trap:** "more trainable parameters can't reduce accuracy if the LR is low enough." A low LR slows the drift but the optimizer still travels; and at some point the LR is so low that nothing learns, which is a different failure with a similar symptom.

**Q67. At the gradient level, why must fine-tuning learning rates be 10–100× smaller than pretraining rates?**
- **Answer:** Because the cost function has changed reference points. Pretraining descends from random init where any descent direction is an improvement. Fine-tuning descends from a point `θ*` that is already a local optimum for the source distribution; the target loss surface near `θ*` has a well-adapted curvature, and a step of size `η` in the direction `−∇L_target` moves the parameters by `η·‖g‖`. When `η` is a pretraining-scale 1e-4–6e-4, the early steps are large enough to leave the basin entirely — the model lands in a region that fits the target but has destroyed the source features, and it never returns because the target loss is lower there. Drift is roughly proportional to `η·‖g‖·t`, so the LR is the linear control. Empirically: full FT of a transformer lives at 1e-5–5e-5, `≥1e-4` destroys the checkpoint within a few hundred steps.
- **Why the interviewer asks:** the candidate should give the mechanism (leaving the basin) rather than the rule, and should be able to connect it to the drift-proportionality argument.
- **Trap:** "Adam normalizes the step size so the LR is the only thing that matters." Adam bounds the step per-parameter but the *direction* is still the target gradient, and the aggregate displacement over `t` steps is what destroys the representation.

**Q68. What is the intrinsic-dimensionality argument for why a low-rank update suffices?**
- **Answer:** Aghajanyan et al. (ACL 2021), *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*: fine-tuning a large pretrained model can be reparameterized to optimize only a small number of parameters in a randomly-projected subspace — on the order of hundreds to a few thousand dimensions — while retaining most of the fine-tuning performance, and the intrinsic dimension *shrinks as the pretrained model gets larger*. That is the theoretical license for LoRA: if the task-specific update lives in a low-dimensional subspace, `ΔW = BA` with small `r` is not an approximation of the update you wanted, it is a reasonable parameterization of it.
- **Why the interviewer asks:** it is the "why does this work at all" question for PEFT, and it distinguishes a candidate who can connect mechanism to method.
- **Trap:** "intrinsic dimensionality means the update matrix is low-rank in the original basis." It is low-rank in a *random projection*; LoRA's `BA` is a different (but empirically effective) parameterization of the same small-subspace intuition.

**Q69. When does training from scratch match pretraining?**
- **Answer:** He et al. (ICCV 2019), *Rethinking ImageNet Pre-training*: with enough target data *and a long enough schedule*, from-scratch training matches ImageNet-pretrained initialization on COCO detection and segmentation. Pre-training's advantage is convergence speed, not a higher ceiling. Zhai et al. (2019) went further: on some VTAB task families (structured/geometric), no pretraining method beat from-scratch. The practical rule: past ~1M in-distribution labeled examples, budget for from-scratch; below that, fine-tune.
- **Why the interviewer asks:** it kills the "fine-tuning is always better" reflex and frames the decision as economic — 1.2M labeled images and 8 GPUs for a month is a from-scratch budget, not a fine-tuning budget.
- **Trap:** "you need to re-pretrain the backbone to get the best result." Neither paper says that; they say the *initialization* stops mattering once the schedule and data are sufficient.

**Q70. Why does the unlabeled-data phase buy more than labels under domain shift?**
- **Answer:** Because domain shift is a statement about `P(X)`, and labels carry information about `P(Y|X)`. If your inputs do not look like the pretraining corpus, every feature the backbone built is tuned to the wrong statistics; labeling 50k examples teaches the head to map *wrong features* onto the right answers, which fits the validation distribution (drawn from the same pool) and collapses on production inputs that are merely slightly further out. Continued pretraining on unlabeled target text re-tunes the features themselves. CS-02's clinical case measured it: DAPT 2.1B tokens, 19 hours, $76 → +10.8 entity-level F1; the SFT phase after it, 18 minutes and $0.60 → the shipped artifact.
- **Why the interviewer asks:** it is the single most expensive misconception in the module and it maps to a real budget line.
- **Trap:** "we'll just label more and it'll generalize." The generalization failure is in the representation; more labels at the same representation raise the validation number and widen the validation-production gap.

**Q71. Distinguish LoRA from BitFit from adapters at the mechanism level.**
- **Answer:** **LoRA** freezes `W` and *adds* `ΔW = BA` (`B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `A` initialized random and `B` at zero so `ΔW = 0` at step 0) alongside it — the whole original weight remains present and active. **BitFit** trains only the bias terms (~0.08% of parameters) — this is the genuine "subset of the existing weights" family. **Adapters** insert bottleneck modules (down-project → nonlinearity → up-project) *between* transformer sublayers rather than modifying them in parallel. LoRA and adapters are both additive structure; BitFit is selective modification. The distinction matters at merge/serve time: LoRA merges exactly (`W + BA`), adapters add latency unless folded, BitFit needs no merge at all because the base is unchanged apart from biases.
- **Why the interviewer asks:** the video's "some subset of the weight" [32:23] is a loose description of LoRA, and a candidate who repeats it uncritically has not understood the method. It is a directly corrected claim in CS-02.
- **Trap:** calling LoRA a subset method. The correction is explicit: LoRA does not select a subset of existing weights; it adds low-rank matrices next to frozen weights.

**Q72. Why is initialization of `A` and `B` asymmetric in LoRA?**
- **Answer:** `A` is initialized from a random Gaussian and `B` at zero, so `ΔW = BA = 0` at step 0. The model therefore *starts* as the exact pretrained model — no disruption at initialization, no cold-start transient, and no need for the warmup gymnastics that a random head requires. It also gives the optimizer a well-conditioned start: `A` receives gradient immediately (through `B`'s zero weights the gradient path exists but the output is zero), and the pair grows out of the zero solution rather than out of noise. This is a large part of why LoRA behaves so stably.
- **Why the interviewer asks:** it is a small implementation detail with a large behavioral consequence, and it explains why LoRA tolerates a *higher* LR than full FT.
- **Trap:** "both matrices are initialized at zero." If both were zero, the gradient to both would be zero (for `B` it is `∂L/∂B ∝ A^T x`, and for `A` it is `∂L/∂B`-dependent), and the adapter could never leave the origin — a symmetric zero init is a saddle. The asymmetry is load-bearing.

**Q73. Why is unfreezing the embedding table on a 7B model a bad idea?**
- **Answer:** Two reasons. (1) Cost: Llama-3-8B's embedding table is 128,256 × 4,096 ≈ 525M parameters, tied to `lm_head` — unfreezing it is not a small operation, it is ~6.5% of the model on its own, and the full-parameter-count reality means "last few blocks + head" is already ~1.4B parameters (17%). (2) Signal: each token's embedding row receives gradient only from the batches containing that token, so rare tokens get few updates and frequent tokens get many — the embedding table is the most unevenly-optimized part of the model and the one most prone to drift on a narrow corpus. This is precisely why the instructor pivots to PEFT for Llama/Mistral/Gemini [32:09]–[32:41].
- **Why the interviewer asks:** it tests whether the candidate ports CNN-era intuitions to LLM scale, which is the module's central "the intuition does not survive the transition" claim.
- **Trap:** "'backbone' means the encoder blocks, so the embeddings aren't part of it." In transformers the backbone includes the embedding table, and it holds the most parameters per token of anything.

**Q74. Your validation loss is rising and your MMLU score is falling. Is that overfitting or forgetting, and how do you tell?**
- **Answer:** Check *where* the degradation lives. **Overfitting:** train loss falls, validation loss (from the target distribution) rises — worse *on the target*. **Forgetting:** validation loss can fall, target metrics climb, but the general-capability suite declines — worse *everywhere else*. They co-occur often, so the diagnostic is the pair of curves: plot target validation loss and the general metric on the same axis. If target loss still falls while the general metric falls, you are in the forgetting regime and more regularization will not help; you need lower LR, fewer epochs, replay, or a switch to LoRA. If target loss is rising too, it is overfitting and you need regularization or more frozen layers.
- **Why the interviewer asks:** the two failures are routinely conflated and they have **opposite** remedies — this is the highest-signal diagnostic question in the module.
- **Trap:** reaching for weight decay and dropout because the target metric plateaued. Those help overfitting; they do nothing about forgetting and, in the case of increased weight decay on a pretrained model, actively hurt.

**Q75. Why can fine-tuning an instruction-tuned model degrade it faster than fine-tuning a base model?**
- **Answer:** Because the instruct behavior is itself a thin, fragile layer of learned behavior sitting on top of the base's capabilities. Fine-tuning a narrow task with a full-capacity update overwrites that thin layer quickly — you see it as format-lock (the model emits JSON-shaped output for unrelated prompts) and as lost refusal behavior. CS-02's QLoRA case hit exactly this: 8 epochs reached 0.89 on val but lost 4.1 MMLU points and started emitting JSON for unrelated prompts; dropping to 3 epochs plus 5% general instruction replay kept the metric and cut the loss to 0.6. If you must fine-tune an instruct model, use LoRA with a low rank and always replay general instructions.
- **Why the interviewer asks:** it is the most common production fine-tune shape in 2025 (everyone starts from an Instruct checkpoint) and the failure is both quiet and embarrassing.
- **Trap:** "the instruct model is already aligned, so it's more robust." It is *less* robust to narrow fine-tuning than a base model, because the alignment is a thin behavioral layer with less redundancy behind it.

**Q76. Why is the plateau of a fine-tune set by the checkpoint and not by your data volume?**
- **Answer:** Because past the knee, the features are the binding constraint. The pretrained representation defines which functions are *reachable* by training the head and upper blocks; adding labels moves you along the curve toward that ceiling and cannot raise it. This is why the rules of thumb read the way they do: 100× the labels buys ~1 point past the knee; if you cannot reach the plateau with 10× your current data, the problem is the representation (domain shift), not the volume. Changing checkpoints — a larger model, a domain-adapted one, a better-matched tokenizer — moves the plateau; adding data moves you along it.
- **Why the interviewer asks:** it is the mental model that prevents a six-figure labeling project from being authorized to fix what is actually a checkpoint problem.
- **Trap:** "we'll reach the plateau eventually with more data." The plateau is asymptotic; if the representation is wrong for your inputs, no volume of labels reaches a plateau that is above your requirement.

---

## Level 4 — System Design & Scenario

Each of these is a 15-minute whiteboard prompt. Answer in the fixed order **requirements → constraints → design → trade-offs → failure modes**. Naming the quadrant before naming a method is the single highest-signal move you can make.

**Q77. [Company style: big-tech ML eng / applied scientist] A support org has 14,000 historical tickets, each with a free-text body and a final category chosen by a human agent from 42 categories. A 3B model gets 61% top-1 with few-shot prompting. The business needs 85%. Latency budget is 800 ms p95. Design the system.**
- **Answer:**
  1. **Requirements.** 42-way classification, macro-F1 ≥ ~0.85 measured on a held-out slice, 800 ms p95 at serving, per-category precision/recall visible (agents will route on it), and a rollback path. Non-requirements: generation, multi-turn, and any new factual knowledge.
  2. **Constraints.** 14k labeled tickets — a Q2-with-a-twist problem: the task is new, the domain is mostly in-distribution English product chatter, but the label space is internal and skewed toward the company's own product names. Taxonomy is a *format* problem, which is exactly what SFT is good at. Compute: one rented 4090; serving: one L40S. Label noise is the ceiling: two annotators on 2,000 tickets gave Cohen's κ = 0.81, which bounds achievable accuracy near 89%, not 100% — so 85% is demanding but arithmetically reachable.
  3. **Design.** `Meta-Llama-3.1-8B-Instruct`, QLoRA NF4 + bf16 compute, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=[q,k,v,o,gate,up,down]_proj`, `lr=2e-4` cosine, `warmup_ratio=0.03`, bs 4 × accum 8 (effective 32), `max_len=1024`, 3 epochs, `bf16=True`, `gradient_checkpointing=True`. Trainable ~42M / 8.03B = 0.52%. Data prep: dedup 14,000 → 13,200, stratified 90/5/5, plus 2,000 double-annotated tickets to bound the label-noise ceiling and 5% general instruction replay. Serve merged fp16 on the L40S.
  4. **Trade-offs.** QLoRA over full FT: ~$1.35 and 3 h 10 min on a 4090 versus ~$150 and a multi-GPU node, for a delta that at 13k examples is 1–3 points at most. `r=16`/`alpha=32` over `r=8`/`alpha=16`: the first run at r=8 hit 0.83 macro-F1; r=16 gained 3 points — the low rank underfits a 42-way taxonomy. Merged fp16 over keeping the adapter unmerged: one behavior per deployment, so merge; but it costs 1.7× latency versus the quantized base, which is a serving decision that must be made against the 800 ms budget. Measured result: macro-F1 0.871 from 0.612 prompt baseline, 42-way top-1 0.884, 240 ms p95 — inside budget with 3× headroom.
  5. **Failure modes.** (a) **Forgetting** — 8 epochs reached 0.89 on val but lost 4.1 MMLU points and produced JSON-shaped output for unrelated prompts; fix is 3 epochs + 5% general instruction replay, which kept the metric and cut the loss to 0.6. (b) **Label noise as the ceiling** — without the κ measurement the team would chase 95% and never understand why it is unreachable. (c) **Class-prior drift** — the taxonomy changes as the business changes, which is label shift and needs logit adjustment, not retraining. (d) **Taxonomy leakage** — a ticket mentioning a product name that appears only in one category's label; the model learns the keyword, not the intent, and the OOD slice catches it.
- **Why the interviewer asks:** it is the full stack in one prompt — quadrant reasoning, PEFT selection, hyperparameters, label-noise ceiling, forgetting, and serving. It also has a trap: candidates who propose full fine-tuning of a 70B, or who never quantify the label-noise ceiling, fail on the budget axis.
- **Trap:** "fine-tune a 70B because accuracy matters most." 14k examples on a 70B costs ~1.1 TB of optimizer state for full FT; QLoRA on an 8B hits the target for $1.35. The second-most-common wrong answer is to skip the dedup and the κ measurement, which makes the 85% target unmeasurable.

**Q78. [Company style: healthcare / regulated industry] A payer must redact 18 PHI entity types (names, MRNs, dates, locations) from clinical notes. They have 3,100 annotated notes and 4 TB of unlabeled historical notes. Design it.**
- **Answer:**
  1. **Requirements.** Token-level entity extraction (37 BIO labels = 18 types × B/I + O), entity-level micro-F1 ≥ ~0.90, zero-PHI leakage into the shipped artifact, auditable provenance, and a de-identification pass that itself cannot leak. Latency is not binding (batch).
  2. **Constraints.** This is **Q3 — different domain, same task**. The task (token classification/NER) is unchanged, but `P(X)` is clinical shorthand (`pt c/o SOB`, `H/O DM2`, `q.d.`), abbreviations, and section headers that are largely absent from Wikipedia + BooksCorpus. **Labels will not fix it.** Compute: 2×A100 80 GB for ~a day, then minutes.
  3. **Design.** Two phases. **Phase 1 — DAPT:** regex de-identify the raw notes first (non-negotiable), then continued MLM on 2.1B tokens, `bert-base-uncased`, MLM 15%, `lr=5e-5`, 1 epoch, `max_len=512`, bf16 → 19 hours on 2×A100 = $76. **Phase 2 — NER SFT:** `BertForTokenClassification.from_pretrained(domain_ckpt, num_labels=37)`, `lr=3e-5`, `warmup_ratio=0.1`, bs 16, 4 epochs → 18 minutes, $0.60. Measure entity-level micro-F1 against both a no-DAPT SFT baseline and a zero-shot off-the-shelf NER model.
  4. **Trade-offs.** DAPT costs 98% of the wall-clock and 99% of the budget and buys **+10.8 F1** (0.912 with DAPT vs 0.804 without); the SFT phase is what produces the artifact. Skipping DAPT and raising the LR to compensate *loses* accuracy (0.742 at lr=5e-5 for 3 epochs) because that is forgetting, not adaptation. Unlabeled data is free here (4 TB) and labels are expensive, so the DAPT investment is obviously correct — the reverse of the usual trade.
  5. **Failure modes.** (a) **PHI memorization** — DAPT on unde-identified notes bakes real MRNs into the weights and ships PHI in the artifact; de-identify with a regex pass *before* tokenization. (b) **Wrong features blamed on the head** — the first attempt skipped DAPT and the model tagged every capitalized token as a name (0.804 F1); teams usually respond by labeling more, which does not help. (c) **Label schema drift** — 37 BIO labels must be stable across annotation waves, or the model learns the annotator, not the entity. (d) **Silent recall collapse on a rare entity type** — report per-entity F1, never just micro-F1.
- **Why the interviewer asks:** it is the canonical Q3 case and it is fully quantified, so the candidate can be graded on whether they spend the budget in the right phase. "Label more notes" is the failing answer.
- **Trap:** "fine-tune `bert-base-uncased` for NER" — the team's own first instinct in CS-02, worth 0.804 F1 versus 0.912. The second trap is doing DAPT on the raw notes without a de-identification pass.

**Q79. [Company style: startup ML eng] A SaaS product personalizes tone, format, and refusal policy per customer. 40 customers, same base model, one 24 GB GPU for serving. Design training and serving.**
- **Answer:**
  1. **Requirements.** 40 distinct behaviors from one GPU; per-tenant isolation (customer A's behavior must not appear for customer B); per-tenant update cadence of days; per-tenant artifacts that can be added and removed without retraining anything else; p95 latency within existing SLA.
  2. **Constraints.** The delta per customer is small and **behavioral** — format, tone, escalation policy, a handful of house rules. 300–800 curated examples per tenant. One 24 GB GPU. This is the workload where the training decision and the hosting decision are the same decision.
  3. **Design.** Base `Llama-3.1-8B-Instruct`, one unmerged LoRA adapter per tenant: `r=8`, `alpha=16`, `lr=1e-4`, 3 epochs, ~12 min per adapter on a 4090 ($0.08 each, $3.20 for all 40). Adapter size 21 MB → 840 MB for all 40. Serve with vLLM multi-LoRA (`--enable-lora`) with adapter swap per request; measured p95 overhead **+11 ms**. Two tenants with genuinely different output schemas needed `r=32` and `target_modules` expanded to the MLP projections.
  4. **Trade-offs.** Adapters over per-tenant checkpoints: 21 MB versus ~16 GB each, and one base model resident instead of 40. **Unmerged over merged:** merging makes per-request swapping impossible, so keep them separate — but that fixes `r` and `target_modules` at train time, so the hosting choice must be made *before* training. Mean tenant-specific rubric score 4.2/5 versus 3.1/5 for a per-tenant system prompt, which is the honest comparison and the one that justifies the project. Distillation into an 8B is the alternative if the base is ever 70B.
  5. **Failure modes.** (a) **Adapter collapse to a common mean** — training all 40 on the same general instruction mix made them nearly identical; tenant-specific data only helped when it contained the tenant's actual escalation examples. (b) **Rank underfit** — `r=8` was too small for two tenants; the symptom is a tenant whose rubric score plateaus below the others at the same epoch count. (c) **Cross-tenant contamination** — one base plus per-request adapters means a routing bug serves the wrong behavior; log the adapter id on every request and assert it against the tenant. (d) **Silent refusal removal** — a one-sided tenant dataset can delete refusal behavior; re-run the safety eval per adapter, not per base.
- **Why the interviewer asks:** it tests whether the candidate understands that adapters are a hosting architecture, not just a training trick — the exact lesson CS-02 draws. Candidates who propose 40 full checkpoints or 40 merged models fail.
- **Trap:** "train 40 adapters and merge them into 40 models." That is 640 GB of artifacts, no per-request swap, and no ability to add tenant 41 without another full merge.

**Q80. [Company style: startup ML eng] A team has 2,000 labeled images (cat vs dog), one A100, and one afternoon. They want 99% validation accuracy. Design the run.**
- **Answer:**
  1. **Requirements.** Binary classification, 2,000 labeled images, one afternoon of GPU time, a number that is defensible on a held-out slice rather than a single IID split.
  2. **Constraints.** Q2 — same domain (natural images; ImageNet pretraining is directly on-distribution), different task (binary instead of 1,000-way). 2,000 images is small: below ~1,000 the optimum is "freeze all but the head"; at 2,000 you can afford one block. Overfitting is the dominant risk, not underfitting.
  3. **Design.** VGG16 `include_top=False`, conv base frozen (14,714,688 params), `Flatten → Dense(256, relu) → Dense(1, sigmoid)` = 6,423,041 trainable (4.64% of 138,357,544), Adam 1e-3 (safe *only* because the base is frozen), `binary_crossentropy`, batch 32, 224×224, **`preprocess_input` — not `rescale=1./255`**, 10 epochs with `EarlyStopping(patience=3, monitor='val_loss')`. Then, and only then, unfreeze `block5` at LR **1e-5** for 3 more epochs. Report the 5-number protocol: held-out accuracy, prompt/zero-shot baseline, an OOD slice, calibration (ECE), and the baseline embedding-drift check.
  4. **Trade-offs.** Freeze-then-unfreeze-one-block over full FT: with 2,000 images the whole-model update memorizes. Measured: 0.961 at epoch 6 for the frozen-base run, 0.961 → 0.978 after unfreezing block5, then **0.978 → 0.964 after also unfreezing block4** — more capacity hurt. The freeze boundary is a hyperparameter to sweep (0, 1, 2 blocks) on a fixed seed, not a theoretical choice. Total cost: 4 min 20 s on a T4, so the A100 is not the constraint — the eval set is.
  5. **Failure modes.** (a) **Normalization mismatch** — `rescale=1./255` instead of `preprocess_input` plateaued at 0.84; the one-line fix gave +12 points with a healthy-looking loss curve. **The video's own notebook has this bug** [41:25]. (b) **Overfitting after unfreezing block4** — the val loss turns up while train loss keeps falling. (c) **Train/val leakage** — if the images were crawled and near-duplicates straddle the split, val accuracy hits 0.99 and production hits 0.7; hash and dedup before splitting. (d) **Alphabetical folder-order label mapping** [40:55] — `class_mode="binary"` assigns 0/1 by folder name order, so a renamed folder silently swaps the classes. (e) **1 sigmoid neuron with `categorical_crossentropy`** — silently wrong loss, trains, and produces a bad model.
- **Why the interviewer asks:** it is the video's own configuration, so it tests faithful reproduction *and* the judgment to deviate (the preprocess fix, the block4 stop). The 99% target is arithmetically plausible only with the preprocess fix, which is the point.
- **Trap:** "unfreeze the last block because that's better." On 2,000 images it is better for exactly one block, and worse for two. Also: "I'll add dropout before the head's first dense layer" — that underfits rather than helping.

**Q81. [Company style: big-tech ML platform] Design the fine-tuning platform a 20-team company will use: one shared base model family, per-team adaptation, shared evaluation, and no drift into ungoverned artifacts.**
- **Answer:**
  1. **Requirements.** 20 teams, each with its own behavior; one shared serving fleet; a single evaluation harness; every artifact reproducible and rollback-able; no team able to ship a model that regresses general capability.
  2. **Constraints.** Teams have wildly different data volumes (a few hundred to a few hundred thousand examples) and different skills. Training budget is shared and finite. Serving must be multi-tenant on a small number of GPUs. Base model revisions are managed centrally and must not break 20 downstream artifacts.
  3. **Design.** (a) **Training:** LoRA/QLoRA by default for every team above 1B base parameters, with a shared launcher and a per-team config checked into the repo; full FT allowed only with a written justification and a GPU budget approval. (b) **Artifacts:** version the **quintuple** `(model_id, base_revision_sha, adapter_sha, tokenizer_sha, data_snapshot_id)`, plus an eval-set hash — a changed eval set invalidates every historical number. Pin `revision=` to a commit SHA on every `from_pretrained`. (c) **Serving:** one base per fleet node, adapters hot-swapped (`vLLM --enable-lora`), merged only for single-behavior high-throughput deployments. (d) **Evaluation:** a shared harness exposing the five-number protocol — held-out target, prompt baseline, OOD slice, general-capability delta, ECE — plus the level-3 forgetting diagnostic (KL-to-base on 2,000 general prompts, thresholds < 0.02 OK / 0.02–0.10 investigate / > 0.10 stop). (e) **CI:** the regression tests below, run on every PR that touches training code.
  4. **Trade-offs.** LoRA-by-default trades 1–2 points of ceiling for a 100× cost reduction and far less forgetting; teams that can demonstrate >10k in-distribution labels and a real accuracy gap can escalate. Unmerged adapters trade ~11 ms p95 overhead and a routing-bug risk for per-tenant flexibility. Central base-revision management trades team autonomy for the ability to patch a base-model vulnerability once.
  5. **Failure modes.** (a) **Silent forgetting across the fleet** — without the general-capability gate, 20 teams each shave 2 points off MMLU and nobody notices until the base is unusable; the gate is the only structural defense. (b) **Revision drift** — `main` moves on the Hub and a rollback silently loads a different model; pin SHAs. (c) **Evaluation-set mutation** — someone "fixes" the golden set and every historical number becomes incomparable; hash it and refuse to run if it changed. (d) **Version sprawl** — 20 teams × weekly runs × 14–140 GB checkpoints; need an explicit registry with retention policy, not a shared bucket. (e) **Memorization / data-protection exposure** — a fine-tune can reproduce training rows; canary tests in CI.
  ```python
  # tests/test_model_regression.py — runs on every PR that touches training code
  def test_task_metric_floor():
      assert evaluate(model, GOLDEN)["macro_f1"] >= 0.90
  def test_no_forgetting():
      assert mean_kl_to_base(model, base_model, FORGET) < 0.05     # nats
  def test_label_mapping_is_stable():
      assert model.config.id2label == json.load(open("eval/id2label.json"))
  def test_output_schema():
      for ex in load_jsonl("eval/schema_50.jsonl"):
          assert parses_as_json(generate(model, ex["prompt"]))
  def test_determinism():
      assert generate(model, "hello", seed=0, temperature=0) == generate(model, "hello", seed=0, temperature=0)
  ```
- **Why the interviewer asks:** it is a platform-design question where the technology is already settled (LoRA, adapters, vLLM) and the difficulty is governance: evaluation gates, provenance, and rollback. Candidates who design only the training path fail the prompt's "no drift into ungoverned artifacts."
- **Trap:** "give every team full fine-tuning so they get the best model." Twenty teams × weekly runs × 16 bytes/param × a base model of choice is an unbounded bill, and it is how a company loses the ability to say what its models do.

**Q82. [Company style: consulting / enterprise] A client must cut inference cost by 3× while *improving* task accuracy on a document-classification workload currently served by a prompted 70B. Design it.**
- **Answer:**
  1. **Requirements.** 3× cost reduction at fixed or better accuracy, stable p95 latency, an accuracy number defensible to an auditor, and a fallback if the small model misses the bar. The client currently has zero fine-tuning infrastructure.
  2. **Constraints.** The 70B is expensive per token; the task is classification, which means a *small encoder* is a legitimate target rather than a compromise. Data volume unknown at the start — that is the first thing to establish, because it determines the method. Documents can be long, so truncation is a real risk.
  3. **Design.** (a) **Establish the floor first:** measure the prompted-70B accuracy on a held-out slice, plus a TF-IDF+SVM baseline and a zero-shot/5-shot probe of a small model. No training starts without these three numbers. (b) **Distill the behavior:** use the 70B as a labeler on unlabeled documents to create a large teacher set, then fine-tune a small encoder (`bert-base` or a DeBERTa-v3-small) on it. This is the "fine-tune a smaller model" path that the module endorses for latency-bound work — fine-tuning does not make a model faster, so the speed must come from the model choice. (c) **Configuration:** head-only linear probe first (70 seconds, often within 2 points), then last-2-blocks + head at `lr=2e-5` with `warmup_ratio=0.06`, 3 epochs, early stopping on target *and* general. (d) **Serving:** the encoder at fp16/int8, batched, replacing the 70B call for the classification path; keep the 70B behind a router for low-confidence cases.
  4. **Trade-offs.** Distillation trades the teacher's knowledge for a small, fast, cheap student and introduces teacher-label noise — the student cannot exceed the teacher's agreement rate with the human labels, so measure the teacher's ceiling first. Classification on an encoder trades generality for ~10–50× the throughput and a fraction of the cost. A LoRA on a 3B instruct model is the middle path if the task needs generative output. Cost per 1M tokens falls by far more than 3×; the honest statement is that the saving is in the model choice and batching, and fine-tuning is what lets you *keep* the accuracy while taking it.
  5. **Failure modes.** (a) **Truncation** — with `max_len=512` on long documents the label-bearing section can be in the truncated tail; the loss decreases and accuracy plateaus low. Fix with head+tail truncation (first 256 + last 256), sliding-window pooling, or a long-context backbone. (b) **Teacher-label noise** — distilled labels inherit the teacher's OOD blind spots; sample and human-review a few hundred. (c) **Distillation destroying calibration** — the student is overconfident; fit a temperature on validation before setting thresholds. (d) **The 3× cost claim being unmeasured** — instrument cost per 1,000 documents before and after, not just latency. (e) **Silent regression on a rare class** — per-class breakdown, never a single accuracy number.
- **Why the interviewer asks:** it forces the candidate to state the module's key insight — **fine-tuning does not make anything faster; it makes a *smaller* model good enough** — and to reach for distillation rather than a bigger fine-tune.
- **Trap:** "fine-tune the 70B harder." The cost problem is the 70B, and fine-tuning it, if anything, adds serving artifacts without changing the per-token economics.

**Q83. [Company style: startup ML eng] A team has 400 examples of a new JSON output schema and wants to apply it to a 70B model. Their GPU budget is one 24 GB card. Design it.**
- **Answer:**
  1. **Requirements.** The model must emit a specific JSON schema reliably, 400 examples of the target format exist, no larger GPU is available, and existing general capability must be preserved (the model is also used for other prompts).
  2. **Constraints.** 400 examples is *format* territory, not behavior: 500–2,000 examples teach format; 400 is at the low edge but format is the cheapest thing to teach. 70B full FT needs ~1.1 TB of optimizer state — categorically impossible on one card. LoRA/QLoRA is the only option, and the question to settle first is whether fine-tuning is needed at all.
  3. **Design.** (a) **Try prompting first** — 20 few-shot prompts with a schema example. If the task is a prompt-format problem, that is the entire solution and it costs nothing. (b) If it genuinely fails: **QLoRA on the 70B** — NF4 4-bit base (~38–48 GB at 70B, so this needs 2×24 GB or a 48 GB card — check the arithmetic honestly and say so) — or, realistically on *one* 24 GB card, **distill into an 8B**: use the 70B to generate schema-conformant outputs for a few thousand prompts, then QLoRA the 8B on those at `r=16`, `alpha=32`, `lr=2e-4`, 3 epochs, `max_len=1024`. (c) Add a **constrained decoder** (grammar/JSON-schema-constrained sampling) on top regardless — it makes schema validity a hard guarantee rather than a learned tendency.
  4. **Trade-offs.** QLoRA on the 70B gives the highest fidelity to the base's reasoning and cannot fit the stated hardware; the 8B distillation fits comfortably and loses some general capability but the schema is a *format* task, which transfers well. Schema-constrained decoding alone is cheaper than either and is the correct first answer if the requirement is validity rather than content. The honest design says: "one 24 GB card rules out the 70B; here are the two paths that fit, and here is the decision criterion."
  5. **Failure modes.** (a) **Format lock** — over-training on a single schema causes the model to emit JSON for unrelated prompts; replay general instructions at ~5% and keep epochs ≤ 3. (b) **Schema drift** — the schema changes and the adapter is stale; version the schema id with the adapter. (c) **Silent schema violation on long inputs** — validate every generation against the schema in CI, not just on a sample. (d) **Catastrophic forgetting on a 400-example run** — small data plus a narrow format is exactly the high-forgetting regime; measure KL-to-base.
- **Why the interviewer asks:** it tests budget honesty — the candidate must recognize that the stated hardware excludes the stated model, say so, and still produce a design. It also tests whether they consider the cheaper non-training answer first.
- **Trap:** "400 examples on a 70B, no problem, LoRA is parameter-efficient." Parameter efficiency does not fix base-model memory: a 70B in NF4 is ~38–48 GB before adapters, activations, or optimizer state.

**Q84. [Company style: big-tech ML eng] Design the evaluation and release process for a fine-tuned model that will be deployed to production, given that the team currently has one accuracy number on an IID split.**
- **Answer:**
  1. **Requirements.** Detect task regression, detect forgetting, detect distribution shift in production, enable rollback as a config change, and keep every historical number comparable.
  2. **Constraints.** The team's single IID accuracy number has ±3 points of noise at n=500 and hides per-class collapse; it cannot detect forgetting at all (level 0). Model selection *is* evaluation — if the eval sets do not exist before training, the team selects on training loss and ships the most-forgotten checkpoint.
  3. **Design.** (a) **Five-number protocol**, reported together: held-out target metric (stratified, ≥50 examples/class, macro-F1), the prompt/zero-shot baseline on the same set, an OOD slice from a different source/annotator/time period, the general-capability delta, and calibration (ECE). (b) **Forgetting instrument:** `lm-evaluation-harness` for MMLU/ARC/HellaSwag before and after plus WikiText perplexity for LLMs; base-embedding cosine drift for encoders. Thresholds: MMLU delta > −0.5 OK / −0.5 to −2.0 investigate / < −2.0 stop; ppl ratio < 1.05 / 1.05–1.20 / > 1.20; KL-to-base < 0.02 / 0.02–0.10 / > 0.10 nats. (c) **Held-out protocol:** split before any leaky preprocessing, hash and dedup across splits, stratify, use a temporal slice if production data is time-ordered (random splits overstate by 5–15 points on drifting domains), keep a final set touched once, fix seeds and run 3 seeds for any claim you will defend. (d) **CI:** the five regression tests in Q81, on a frozen golden set never used for selection. (e) **Release:** shadow-deploy against the current model, compare on task metrics *and* a guardrail panel simultaneously, then canary; rollback is a config change with the previous artifact kept warm. (f) **Monitoring:** input-length distribution (truncation rate), input embedding drift (MMD/PSI > 0.2), prediction distribution (class proportion shift > 20% relative), confidence distribution (mean max-prob drop > 0.1), refusal/format-violation rate, shadow eval on the golden set, and latency percentiles.
  4. **Trade-offs.** Three eval sets cost labeling and infrastructure that the team does not currently have, and that cost is the price of being able to say what happened. A temporal split is more honest and gives lower numbers — expect a 5–15 point drop versus a random split on drifting domains, and budget for the political conversation. Keeping a general-capability gate may block a release that would have shipped on target metrics alone; that is the intended behavior.
  5. **Failure modes.** (a) **Eval-set reuse** — the same set used 40 times is overfitted by selection; freeze it or hold a final set touched once. (b) **`eval_strategy` never firing** — "eval accuracy" is actually the last training batch; count the `eval` lines in the log. (c) **Benchmark contamination** — the base model may have been pretrained on your benchmark, so a "+2 on MMLU" can be memorization; check the model card against your eval set. (d) **Judge drift** — LLM-as-judge correlates with length and with the judge's own fine-tuning and drifts between judge versions; version the judge. (e) **A model that wins on accuracy and loses 4 MMLU points shipping without a written acceptance of the tax** — the A/B must compare task metrics and the guardrail panel together.
- **Why the interviewer asks:** it is the section of CS-02 that teams skip, and the answer is graded on whether the candidate produces an *instrument* (thresholds, sets, CI) rather than an intention to "evaluate carefully."
- **Trap:** "we'll evaluate after training." Model selection is evaluation. Also, reporting one accuracy number on an IID split to leadership is the specific failure this prompt is constructed around.

---

## Level 5 — Debugging & Incident Response

Answer as an **ordered checklist**. The interviewer is grading the order: cheapest and most-likely-first, and each step should eliminate a class of causes rather than a single guess.

**Q85. A 3-class fine-tune shows a loss that is flat at 1.386 and never moves. What do you check, in what order?**
- **Answer:**
  1. **`model.config.num_labels`** — 1.386 is `ln(4)`. With 3 classes present and `num_labels=4`, the fourth head row is never supervised and the model sits at uniform. This is the specific signature, so check it first. `print(model.config.num_labels)` vs `dataset.features['label'].num_classes`.
  2. **The label id set** — `print(sorted(set(dataset['label'])))`. If it is `{0, 1, 3}` with `num_labels=3`, you have the emotion-filter bug and the *loud* `IndexError` counterpart is hiding behind the silent 4-class config.
  3. **Whether the head is trainable at all** — `sum(p.numel() for p in model.parameters() if p.requires_grad)`. A flat loss is also what a fully-frozen model produces.
  4. **Whether the optimizer holds the head's parameters** — if `requires_grad` was set *after* optimizer construction, the count looks right and nothing learns. `len(optimizer.param_groups[0]['params'])`.
  5. **LR** — only now. A too-low LR produces a nearly-flat loss, but it is not *exactly* `ln(k)`; the value is the discriminator, and checking LR before the config wastes the most informative signal you have.
- **Why the interviewer asks:** the exact value of a flat loss is the highest-information number in a training log, and candidates who ignore it guess at learning rates for hours.
- **Trap:** "the LR is too low, raise it." Raising the LR on a `num_labels` bug makes the loss noisy around `ln(4)` and can destroy the checkpoint — a worse outcome with the same root cause.

**Q86. Loss is flat at exactly 0.693 on a binary task. Same drill.**
- **Answer:**
  1. **Is `ln(2) = 0.693`?** Then the model is predicting uniform — the head is random and not learning.
  2. **Trainable count and the first trainable tensor name** — a frozen-everything model (the classic "I froze the whole model and forgot to re-enable the head") produces exactly this.
  3. **Optimizer construction order** — frozen after the optimizer was built ⇒ head absent from the param groups ⇒ no updates, right count.
  4. **`num_labels`/head shape** — a binary head with `num_labels=2` gives `Linear(hidden, 2)` + softmax CE, or `Linear(hidden, 1)` + sigmoid BCE. Mixing them (1 sigmoid neuron with `categorical_crossentropy`) trains, but toward nonsense.
  5. **Label dtype** — BCE expects float targets; integer labels in `{0, 1}` usually work in PyTorch but fail silently in some Keras configurations.
- **Why the interviewer asks:** it is the same signature as Q85 at `k=2`, and the interviewer is checking that the candidate has a *general* rule (`ln(k)` floor ⇒ head/label/freeze problem) rather than two memorized cases.
- **Trap:** "the head LR is too small at 1e-3." If the head is genuinely trainable, 1e-3 on a linear layer moves the loss within 20 steps. A flat loss means no gradient is arriving, not that it is arriving slowly.

**Q87. Loss spikes to NaN in the first 20 steps of a fine-tune. Order of checks?**
- **Answer:**
  1. **Log the first 10 loss values, not just the ones that survived** — a loss that goes 2.1 → 4.7 → 21k → NaN is a divergence; one that is NaN from step 1 is an initialization or data problem.
  2. **LR** — `≥1e-4` for full FT of a pretrained encoder is a pretraining LR and diverges. Divide by 10 and re-run 50 steps.
  3. **Warmup** — zero warmup with a randomly-initialized head is the second most common cause. Add 6–10%.
  4. **Precision** — fp16 on a T4 without loss scaling overflows. Switch to bf16 (Ampere+) or confirm `fp16=True` is enabling scaling.
  5. **Gradient clipping** — should be 1.0; check it is actually wired to the optimizer, not just configured.
  6. **Data** — a NaN or `-inf` in the input tensor, or a label outside `[0, num_labels)`; check `torch.isnan(batch['input_ids']).any()` and the label range on the first batch.
- **Why the interviewer asks:** NaN is the most over-diagnosed symptom in training. The ordered checklist separates "numerical" from "data" from "schedule" in six steps instead of six hours.
- **Trap:** "it's a data problem, rebuild the dataset." NaN in the first 20 steps on a fine-tune is overwhelmingly an LR/warmup/precision issue. Rebuilding the dataset is the expensive way to not fix it.

**Q88. Validation loss falls, target metrics rise, and MMLU fell 4 points. What do you check, and what is the fix?**
- **Answer:**
  1. **Confirm the direction of the two curves** — target up, general down is the forgetting signature. Target loss falling is *not* reassurance; it is part of the signature.
  2. **Epoch count** — how many passes over the task data? 8 epochs on a narrow set is the classic over-training regime.
  3. **LR** — anything ≥5e-5 for a full FT is drifting fast; drift is roughly proportional to `η·‖g‖·t`.
  4. **Replay fraction** — is any general-domain data in the batch? If the pipeline replays general data, check it is *distributionally different* from the target (replaying the same domain does nothing).
  5. **Which parameters moved** — trainable fraction and the freeze depth. Full FT is the high-forgetting configuration; LoRA is the low one.
  6. **The exact numbers** — 4 points on MMLU is past the "stop and re-plan" threshold (< −2.0), so this is not a tuning exercise; the artifact should not ship as-is.
  - **Fix, in order:** add 5% general instruction replay and drop to 3 epochs; if the loss persists, divide the LR by 3; if it persists, cut the epochs further and early-stop on the *general* metric; if it persists, switch to LoRA (`r=16`), which cannot move the base weights at all.
- **Why the interviewer asks:** it is the incident this entire module is built around. The correct first instinct is "stop the run," not "add regularization," and the interviewer is listening for that.
- **Trap:** raising weight decay or adding dropout. Neither addresses forgetting, and higher weight decay on a pretrained model degrades it in a way that *looks* like forgetting but is not — you will chase a phantom.

**Q89. Validation accuracy comes back at 0.99 on a task you expected to be hard. What is your first move?**
- **Answer:**
  1. **Assume leakage before you celebrate.** Hash the raw inputs (exact and near-duplicate via MinHash/SimHash) and count the overlap between train and validation. Scraped datasets routinely contain the same row in both splits.
  2. **Check the split logic** — was the split taken before or after deduplication, before or after any target encoding or tokenizer-statistics fitting?
  3. **Check the label distribution** — a single class dominating both splits (the emotion `joy` class alone is 44%) gives a high number to a model that learned nothing. Compare against the majority-class baseline.
  4. **Check the evaluation code** — `eval_strategy` firing, `load_best_model_at_end` selecting on the right metric, no label leakage through an ID column, `padding_side` consistent.
  5. **Check for base-model contamination** — if the eval set is a public benchmark, the base model may have been pretrained on it; check the model card.
  6. **Then** hold out a temporal slice or a different-source slice and re-measure; random splits overstate by 5–15 points on drifting domains.
- **Why the interviewer asks:** the instinct to disbelieve a good number is the single strongest signal of a production-seasoned engineer, and 0.99 on a hard task is a data bug roughly always.
- **Trap:** "great, ship it." The second-worst answer is "the task was easier than I thought" — possible, but you check leakage first because it is far more common and far more expensive.

**Q90. A Keras VGG16 fine-tune plateaus at 0.84 and the loss curve looks healthy. Diagnose.**
- **Answer:**
  1. **Check the input preprocessing first.** `ImageDataGenerator(rescale=1./255)` is *not* what VGG16 was trained with; the correct call is `tf.keras.applications.vgg16.preprocess_input` (RGB→BGR + ImageNet channel-mean subtraction). Cost of getting it wrong: 8–12 points, with a completely normal-looking loss curve. This is the single most likely cause, so check it first — and note the video's own notebook has this bug [41:25].
  2. **Check `model.summary()`'s trainable count** — was freezing applied before `compile()`? Keras bakes `trainable` at compile time, so a post-compile freeze is a silent no-op.
  3. **Check `include_top=False`** — if it was left `True`, you are fine-tuning a 1,000-class ImageNet head against 2 classes.
  4. **Check the class index mapping** — `flow_from_directory` assigns labels in alphabetical folder order [40:55]; a renamed or reordered folder swaps the classes and caps accuracy near chance on a balanced set (0.5, not 0.84, so this is a lower-probability cause here).
  5. **Check the loss/metric pairing** — 1 sigmoid neuron must use `binary_crossentropy`; `categorical_crossentropy` with 1 output is silently wrong.
  6. **Check `target_size`** — must be `(224, 224)` for VGG16 or the first block's receptive-field statistics are invalidated.
  7. **Only then** consider capacity: unfreeze `block5` at LR 1e-5 and re-measure.
- **Why the interviewer asks:** it is a real, one-line, high-cost bug in the source video, so a candidate who has actually run this notebook knows it and one who has only read about transfer learning does not.
- **Trap:** "the model needs more capacity, unfreeze more layers." Unfreezing at LR 1e-3 destroys the pretrained filters within ~200 steps; unfreezing correctly at 1e-5 buys a couple of points and leaves the 12-point normalization problem in place.

**Q91. `RuntimeError: CUDA out of memory` on a BERT-base fine-tune that should fit in 12 GB. What do you check?**
- **Answer:**
  1. **Count the resident model copies** — `torch.cuda.memory_allocated()` and `nvidia-smi`. The video's own notebook OOMs because it holds a `BertForSequenceClassification` *and* a second full copy inside a hand-written `BertClassifier`. This is the most common cause and the least obvious.
  2. **Free aggressively between experiments** — `del model; gc.collect(); torch.cuda.empty_cache()`. A KL-to-base diagnostic that keeps the base model resident doubles the footprint by design.
  3. **The eval batch size** — `per_device_eval_batch_size` defaults are often 8 with `max_len=512`; "OOM only at eval" is this. Set it explicitly and `eval_accumulation_steps=1`.
  4. **`max_len`** — activation memory scales linearly and attention cost quadratically. 512 → 128 is a 4× attention reduction.
  5. **Batch size and accumulation** — `per_device_train_batch_size=8` with `gradient_accumulation_steps=2` reaches an effective batch of 16 for a third of the peak memory.
  6. **Frozen-prefix activation retention** — if the base is "frozen" but not wrapped in `torch.no_grad()`, you are still storing the full graph.
  7. **The environment** — if all six are clean, it is a quota/driver/environment problem, which is what the video's second OOM actually was [1:09:52]–[1:10:10]. Say so explicitly rather than shrinking the model.
- **Why the interviewer asks:** the ordering matters — the candidate must not "fix" a resource problem by changing the model, which silently changes the experiment.
- **Trap:** "reduce the batch size." It works, but it changes the effective batch, the LR/batch interaction, and the wall clock — and it does not address the two-model-copies cause, so the next run OOMs again with a different batch size.

**Q92. A model gives all-one-class predictions in production but looks fine in evaluation, and batched inference differs from single-example inference. Diagnose.**
- **Answer:**
  1. **`model.eval()`** — set it in the serving path, once, at load. A model left in `train()` mode has dropout active and BatchNorm using batch statistics, which is the classic "fine in eval, garbage in production" cause.
  2. **Padding side** — `tokenizer.padding_side` must match training. Padding on the wrong side with a causal model shifts every position and silently changes logits; padding on the wrong side with an encoder changes attention masking if `attention_mask` is mishandled.
  3. **The attention mask** — always pass it. Encoding a batch without it makes pad tokens attendable, and the damage grows with batch size, which is exactly the reported symptom.
  4. **Label↔index alignment at serving** — compare `model.config.id2label` with the label encoder used in production. CS-02 lists this as "very common in multi-class": training accuracy is high and production output is garbage because class 3 is being read as class 0.
  5. **Preprocessing parity** — the serving path must reproduce the training-time transform exactly (for vision, `preprocess_input`; for text, the same tokenizer revision and the same `max_length`/truncation).
  6. **Then** the model itself — compare logits for one example encoded alone versus in a batch of 32; if they diverge, it is 2–4, not the weights.
- **Why the interviewer asks:** it is the highest-frequency production incident in the module and it is almost never the model. The candidate must resist the instinct to retrain.
- **Trap:** "the model is overfit / the data drifted, retrain." A model that is correct in eval and wrong in production is a *serving-path* bug until proven otherwise, and retraining will reproduce it exactly.

**Q93. The fine-tune "succeeded" but the resulting model is worse than the prompt baseline, and a teammate says their reproduction is 4 points below the paper's number. What do you check?**
- **Answer:**
  1. **Did you measure the prompt baseline at all?** If you cannot state the zero-shot and majority-class numbers on the same eval set, the "success" is unmeasured. This is STOP condition #1 in CS-02 and the first thing to fix.
  2. **Label mapping and class index alignment** — a systematically wrong mapping costs several points and looks exactly like "the paper doesn't reproduce."
  3. **LR and epoch count versus the paper** — papers routinely omit warmup and epoch counts. Mosbach et al. (ICLR 2021), *On the Stability of Fine-tuning BERT*: use a *lower* LR and *more* epochs than the original BERT recipe; the standard recipe is unstable on small data.
  4. **Freeze depth and what actually trained** — print the trainable count and the first trainable tensor. A 4-point gap is often "the pooler never moved."
  5. **Preprocessing and tokenization** — the same `preprocess_input`/tokenizer-revision class of bug as Q90 and Q92.
  6. **The eval protocol** — different split, different metric (micro vs macro), different `max_len`, different threshold. A 4-point gap between two honest numbers is often two different metrics being compared.
  7. **Data volume and quality** — duplicates across splits, label noise, class imbalance. CS-02's κ measurement bounded the ceiling at ~89% for a task the team was targeting at 95%.
- **Why the interviewer asks:** it is the reproducibility question, and the correct answer is a checklist that starts with "was the comparison even valid" rather than with hyperparameter tinkering.
- **Trap:** "raise the LR / train longer to catch up to the paper." CS-02's measured row is the counterexample: full FT at 5e-5 for 10 epochs scored 0.918 versus 0.933 at 2e-5 for 3 epochs. More aggressive training moves you away from the paper's number, not toward it.

**Q94. After merging a LoRA adapter into the base, accuracy drops. Separately, a model that worked in staging fails at load time in production. Handle both.**
- **Answer:**
  **Part A — merge degradation:**
  1. **Check `lora_alpha` / scaling** — the effective update is `(alpha / r) · BA`; a mismatch between the trained config and the merge script rescales the adapter. This is the most common cause.
  2. **Check the dtype of the merge** — merge on CPU in fp32 (`merge_and_unload()`), then cast to fp16 for serving. Merging in fp16 accumulates rounding into every weight.
  3. **Compare logits pre- and post-merge on 10 fixed examples** — this localizes the bug to the merge rather than to serving; if the logits match, the problem is downstream.
  4. **Check for a stale adapter** — the SHA of the adapter you merged must equal the SHA you trained (see the quintuple below).
  5. **Check serving precision** — CS-02 found serving the merged model in fp16 without re-quantizing was 1.7× *slower* than the quantized base; a "drop in accuracy" can also be a different quantization path being exercised.
  **Part B — load-time failure in production:**
  6. **Base model revision drift.** The Hub serves `main`, and `main` moves. Pin `revision=` to a commit SHA on every `from_pretrained` call.
  7. **Version the quintuple** — `(model_id, base_revision_sha, adapter_sha, tokenizer_sha, data_snapshot_id)`, plus the eval-set hash. A changed eval set invalidates every historical number.
  8. **Rollback must be a config change, not a retrain** — keep the previous artifact warm, and record the base SHA *and* adapter SHA in the merged artifact's card, because merging destroys provenance otherwise.
- **Why the interviewer asks:** it pairs the two halves of the release problem — the merge is a numerical operation, the deployment is a provenance problem — and a candidate who only knows one half ships a silent regression.
- **Trap:** "re-train the adapter." If the failure is a revision move or an alpha mismatch, retraining reproduces it. Diagnose by comparing logits on fixed examples before touching training.

---

## Rapid Fire — True / False / One-Liner

Say "true" or "false" and then the one-line why. Ten seconds per row.

| # | Statement | Verdict |
|---|---|---|
| 1 | Transfer learning and fine-tuning are synonyms. | **False** — transfer learning is the strategy, fine-tuning one tactic that implements it; zero-shot inference is transfer learning with no fine-tuning [23:37], [25:29]. |
| 2 | Early layers compute primitive features and late layers compute task-specific ones. | **True** — the empirical result from Yosinski et al. 2014; the instructor's version at [21:17]–[21:32]. It is a heuristic, not a theorem. |
| 3 | Freezing a layer means its forward pass is skipped. | **False** — it still computes; freezing saves backward/optimizer memory and time, never forward time or inference latency. |
| 4 | Fine-tuning learning rates are 10–100× smaller than pretraining rates. | **True** — 1e-5 to 5e-5 for full FT of an encoder versus 1e-4 to 6e-4 for pretraining. The module's most load-bearing number. |
| 5 | LoRA uses a lower learning rate than full FT because it has fewer parameters. | **False** — LoRA runs *higher* (1e-4 to 3e-4) because `B` and `A` start at zero and must travel further than a pretrained weight. |
| 6 | A linear probe is a waste of time if you plan to full-fine-tune. | **False** — it is 70 seconds, frequently within 2 points of the best config, and often the *best* option OOD (Kumar et al. 2022). |
| 7 | Catastrophic forgetting shows up as rising validation loss. | **False** — it shows up as a falling *general* metric while the target metric improves. Target validation loss can keep falling throughout. |
| 8 | Weight decay is the same thing as L2-SP. | **False** — weight decay pulls toward zero; L2-SP pulls toward the pretrained `θ*`. Different objectives, different effects. |
| 9 | `requires_grad=False` is sufficient to save activation memory. | **False** — you also need `torch.no_grad()` (or checkpointing) around the frozen prefix, or the graph is still retained. |
| 10 | Keras applies a `trainable` change made after `compile()`. | **False** — `trainable` is baked into the compiled train function; changing it afterwards is a silent no-op until you recompile. |
| 11 | Full fine-tuning always beats PEFT on accuracy. | **False** — below ~5k examples LoRA often wins, because the frozen base regularizes (Biderman et al. 2024: LoRA learns less and forgets less). |
| 12 | Domain shift is fixed by labeling more data. | **False** — it is fixed by *unlabeled* target text (continued pretraining). CS-02's clinical case: +10.8 F1 from 19 h of DAPT, +0 from labels. |
| 13 | A flat loss at exactly `ln(k)` usually means the LR is too low. | **False** — it means the head is not learning: wrong `num_labels`, frozen head, head missing from the optimizer, or missing label remap. |
| 14 | Label shift should be fixed by retraining on the new class proportions. | **False** — if `P(X\|Y)` is unchanged there is nothing to learn; use logit adjustment or threshold tuning. |
| 15 | Fine-tuning is a reliable way to inject new facts into a model. | **False** — SFT teaches behavior and format; facts need continued pretraining or RAG. "Nothing teaches facts." |
| 16 | Merging a LoRA adapter makes it faster than the base model. | **False** — merging removes adapter overhead (1–15 ms); it does not make the model faster than the original base. |
| 17 | On a 7B model, unfreezing "the last few blocks" is a cheap operation. | **False** — one Llama-3-8B decoder block is ~218M parameters and the embedding table is ~525M; last-4 + head ≈ 1.4B ≈ 17% of the model. |
| 18 | More epochs improve the target metric, so more epochs are safe. | **False** — target metrics keep climbing past 3–4 epochs while general capability falls. You are training the model to forget. |
| 19 | Warmup matters less for fine-tuning than for pretraining because the LR is smaller. | **False** — it matters *more*, because the pretrained basin is more fragile than random init and the randomly-initialized head sends the first large gradients. |
| 20 | Unfreezing more layers on a small dataset can reduce accuracy. | **True** — capacity without signal is variance. Measured: VGG16 at 2,000 images went 0.978 → 0.964 when block4 was also unfrozen. |
| 21 | A `Dataset.filter()` call renumbers integer labels to stay contiguous. | **False** — it does not; the emotion notebook keeps `{0,1,3}` while `num_labels=3` expects `{0,1,2}`. |
| 22 | Replay works as long as you mix in extra data from your target domain. | **False** — replay must be *distributionally different* from the target, or it carries no source signal. General-domain data, 1–10%, 5% typical. |
| 23 | Fine-tuning reduces inference latency. | **False** — the architecture and parameter count are unchanged. Speed comes from a smaller model, quantization, or distillation. |
| 24 | The `dair-ai/emotion` split in the video gives 9,749 train / 2,438 validation. | **True** — verified exactly: 12,187 rows after filtering to sadness/joy/anger, then an 80/20 stratified split [54:39]–[55:10]. |
| 25 | The video's BERT notebook proves the configuration only reaches ~0.70 accuracy. | **False** — the video *never produces a result*; it OOMs twice on an exhausted Colab GPU. The working equivalent reaches ~0.925 on a T4. |
| 26 | Rescaling inputs to `[0,1]` is the correct preprocessing for a pretrained VGG16. | **False** — you must call `preprocess_input` (RGB→BGR + ImageNet mean subtraction); the mismatch costs 8–12 points with a healthy-looking loss curve. |

---

## Coding / Whiteboard Tasks

### Task 1 — Implement freeze-then-unfreeze, with the receipt

**Prompt.** Given a `BertForSequenceClassification`, write a function that freezes everything, unfreezes the last `n` encoder blocks, the pooler, and the classifier, and prints the trainable fraction. Then write the assertion that proves no gradient reaches a frozen tensor.

**Reference solution.**
```python
import torch, math

def set_trainable_by_unfreezing_top(model, n_unfreeze, prefix="bert.encoder.layer"):
    for p in model.parameters():
        p.requires_grad = False
    blocks = [m for n, m in model.named_modules() if n.startswith(prefix) and m is not model]
    for blk in blocks[-n_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True
    for n, p in model.named_parameters():
        if n.startswith("classifier") or n.startswith("bert.pooler"):
            p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"trainable {tr:,}/{tt:,} = {100*tr/tt:.2f}%")
    first = next(n for n, p in model.named_parameters() if p.requires_grad)
    print("first trainable tensor:", first)
    return model

def assert_no_gradient_leak(model, batch):
    model(batch["input_ids"][:2], attention_mask=batch["attention_mask"][:2]).loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None and not p.requires_grad:
            raise RuntimeError(f"gradient leaked into frozen tensor {n}")
    return True
```
Expected for `n_unfreeze=2` on BERT-base: `14,178,051 / 109,484,547 = 12.95%`.

**Grading notes.** Award full marks only if: (a) the freeze loop comes *before* any unfreeze, (b) `bert.pooler` is explicitly re-enabled — omitting it is the bug in the video's own snippet, (c) modules are resolved by name, not by hard-coded index, (d) the trainable fraction is printed, and (e) the gradient-leak assertion exists. Penalize a solution that freezes only `bert.encoder` and leaves `embeddings`/`pooler` in their default state, because that is unfreezing the wrong things rather than choosing what to unfreeze. **Bonus point:** returning a `param_groups` list for LLRD, and noting the optimizer must be constructed *after* this call.

### Task 2 — The pre-flight script

**Prompt.** Write the eight-step, 30-second pre-flight that must run before any fine-tuning job starts.

**Reference solution.**
```python
import math, torch, numpy as np

def preflight(model, tokenizer, dataset, texts, batch):
    # 1. Config sanity
    print(model.config.num_labels, model.config.hidden_size, model.config.num_hidden_layers)
    # 2. Tokenizer on real domain examples — is it shredding your text?
    print(tokenizer(texts[:3])["input_ids"])
    # 3. Token length percentiles -> choose max_len
    lens = [len(t) for t in tokenizer(list(texts))["input_ids"]]
    print(np.percentile(lens, [50, 95, 99]))
    # 4. Class distribution
    from collections import Counter
    print(Counter(dataset["label"]))
    # 5. Freeze and print the receipt
    set_trainable_by_unfreezing_top(model, n_unfreeze=2)
    # 6. Loss at init must be ~ln(k)
    out = model(batch["input_ids"][:2], attention_mask=batch["attention_mask"][:2], labels=batch["label"][:2])
    assert abs(out.loss.item() - math.log(model.config.num_labels)) < 0.1, out.loss.item()
    # 7. No gradient on frozen tensors
    assert_no_gradient_leak(model, batch)
    # 8. Twenty steps and the loss is falling
    # (delegate to the trainer; assert history[0] > history[-1])
    return True
```
**Grading notes.** The eight steps map one-to-one onto CS-02's pre-flight list. Grade on *completeness of the checklist* more than on elegance: candidates who skip step 3 (token length) or step 6 (`ln(k)` check) have never debugged a real run. Full marks require the `ln(num_labels)` assertion — it is the cheapest detector of the module's most common bug. **Bonus point** for asserting `labels.max() < model.config.num_labels` explicitly, which catches the remap bug before the first batch.

### Task 3 — Four-level forgetting measurement

**Prompt.** Implement the forgetting diagnostic for an encoder classifier, and state the pass/fail thresholds.

**Reference solution.**
```python
import torch, numpy as np

@torch.no_grad()
def embed_drift(base_model, tuned_model, sentences, tok, batch_size=64):
    """Mean cosine DISTANCE between base and tuned pooled embeddings."""
    base_model.eval(); tuned_model.eval()
    dists = []
    for i in range(0, len(sentences), batch_size):
        b = tok(sentences[i:i+batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=128)
        b = {k: v.to(base_model.device) for k, v in b.items()}
        a = base_model(**b).logits            # or .pooler_output for the representation
        c = tuned_model(**b).logits
        cos = torch.nn.functional.cosine_similarity(a, c, dim=-1)
        dists.extend((1 - cos).cpu().tolist())
    return float(np.mean(dists))

@torch.no_grad()
def kl_to_base(base_model, tuned_model, sentences, tok, batch_size=32):
    """Mean KL(p_base || p_tuned) on general-domain prompts, in nats."""
    acc, n = 0.0, 0
    for i in range(0, len(sentences), batch_size):
        b = tok(sentences[i:i+batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=256)
        b = {k: v.to(base_model.device) for k, v in b.items()}
        lp_tuned = torch.log_softmax(tuned_model(**b).logits, -1)
        p_base   = torch.softmax(base_model(**b).logits, -1)
        acc += torch.nn.functional.kl_div(lp_tuned, p_base,
                                          reduction="batchmean", log_target=False).item() * len(sentences[i:i+batch_size])
        n += len(sentences[i:i+batch_size])
    return acc / n
```
Thresholds: base-embedding cosine drift < 0.03 OK, 0.03–0.10 investigate, > 0.10 stop. KL-to-base < 0.02 nats OK, 0.02–0.10 investigate, > 0.10 stop. For an LLM, replace this with `lm_eval` MMLU/ARC/HellaSwag (delta > −0.5 OK, < −2.0 stop) and a WikiText perplexity ratio (< 1.05 OK, > 1.20 stop).

**Grading notes.** Full marks require `log_target=False` with *probability* targets — a candidate who passes log-probabilities with `log_target=False` gets a finite but meaningless number, which is worse than an error. The second full-marks criterion is stating explicit thresholds; "compare before and after" without a cutoff is not a diagnostic. **Bonus point** for noting that the general corpus must be held out and *different* in distribution from the target, and that 2,000 prompts is enough for a stable estimate.

### Task 4 — LLRD + slanted triangular LR

**Prompt.** Write the two ULMFiT schedule helpers: layer-wise LR decay with the head boosted, and the STLR schedule. State the decay factor you would use and why it differs from ULMFiT's 2.6.

**Reference solution.**
```python
def param_groups_llrd(model, base_lr, decay=0.8, num_layers=12):
    """Layer-wise LR decay: embeddings smallest, head largest."""
    groups = [{"params": model.bert.embeddings.parameters(),
               "lr": base_lr * (decay ** num_layers)}]
    for i in range(num_layers):
        groups.append({"params": model.bert.encoder.layer[i].parameters(),
                       "lr": base_lr * (decay ** (num_layers - 1 - i))})
    groups.append({"params": model.classifier.parameters(), "lr": base_lr * 2.0})
    return groups

def stlr(step, total_steps, peak_lr, cut_frac=0.1, ratio=32):
    cut = int(total_steps * cut_frac)
    if step < cut:
        return peak_lr * step / max(cut, 1) * (1 / ratio) + peak_lr / ratio
    p = (step - cut) / max(total_steps - cut, 1)
    return peak_lr * (1 - p * (1 - 1 / ratio))

optimizer = torch.optim.AdamW(param_groups_llrd(model, base_lr=2e-5, decay=0.8), weight_decay=0.01)
```
ULMFiT's `η^{l−1} = η^l / 2.6` is a per-layer ratio of 0.385. Production uses 0.8–0.9 because 2.6 compounds to `0.385^12 ≈ 4e-5` of the base LR at the embedding table — effectively frozen, which is rarely what you want when you are also replaying general data and want the bottom to move slightly. With `decay=0.8` and `base_lr=2e-5`, the embeddings get `2e-5 × 0.8^12 ≈ 1.4e-6` and the head gets `4e-5`.

**Grading notes.** Three things separate a pass from a fail: (a) the embedding group exists and is the *smallest*, (b) the head is boosted above `base_lr` rather than equal to it, and (c) the STLR warmup reaches `peak_lr/ratio` at step 0 and `peak_lr` at `cut`, then decays back to `peak_lr/ratio` — candidates routinely invert the ratio. **Bonus point** for noting that `warmup_ratio` in HF's `TrainingArguments` counts *optimizer* steps (accounting for `gradient_accumulation_steps`), not micro-batches, and that a hand-rolled schedule must match or the warmup will be 8× too short. **Red flag:** applying one flat LR and calling it LLRD.

---

## Cheat Sheet of Numbers To Memorize

Instant recall, no derivation. If a number here is asked for and you hedge, that is a failed screen.

### Learning rates

| Regime | Peak LR (AdamW) | Note |
|---|---|---|
| Pretraining, LLM from scratch | 1e-4 – 6e-4 | 1e-3 for CV with SGD |
| **Full fine-tuning, transformer** | **1e-5 – 5e-5** | 2e-5 is the BERT-era default; 3e-5 aggressive; ≥1e-4 destroys the checkpoint |
| Head only / linear probe | 1e-3 – 1e-2 | Nothing pretrained is at risk |
| LoRA / QLoRA adapters | 1e-4 – 3e-4 | Higher than full FT, not lower |
| VGG16, unfrozen block5 (video Way 3) | 1e-5 | The video's load-bearing silent detail |

### Warmup, epochs, batch

| Param | Pretraining | Fine-tuning |
|---|---|---|
| Warmup | 1–2% of steps (2,000 typical) | **6–10% of steps** |
| Epochs | < 1 (single pass over tokens) | 2–4 NLP classification, 1–3 SFT, 10–30 CV small data |
| Batch | 1M–4M tokens | 16–64/device NLP; effective 32–64 with accumulation |
| Weight decay | 0.1 | **0.01** |
| Gradient clipping | 1.0 | 1.0 |
| Head dropout | 0.0–0.1 | 0.1 (0.1–0.3 head-only) |

### Memory

| Quantity | Value |
|---|---|
| Full FT, mixed precision + Adam | **16 bytes / trainable parameter** (2 bf16 w + 2 bf16 g + 8 fp32 m,v + 4 fp32 master) |
| Rule of thumb | full FT needs ~**20×** the parameter count in GB ⇒ 7B ≈ 140 GB |
| 7B full FT | 112 GB + activations ⇒ 2×A100 80 GB with FSDP/ZeRO-3 |
| 7B/8B QLoRA | **6–8 GB** (4-bit base 3.5 GB + adapters ~1.1 GB + activations) |
| 70B QLoRA | 38–48 GB ⇒ one 48 GB card or 2×24 GB |
| Freezing 90% of a 7B | optimizer state 84 GB → 8.4 GB |
| Optimizer state, fp16 grads + fp32 m,v | 10 bytes / trainable parameter (100% at 7B = 84 GB) |

### Parameters and trainable fractions

| Model | Total | Last block | Head only | Configuration |
|---|---|---|---|---|
| VGG16 | 138,357,544 | 7,079,424 (block5) | 4,097 = 0.003% | head + Dense(256) = 6,423,041 = 4.64% |
| VGG16, block5 + head | — | — | — | 13,502,465 = 9.76% |
| BERT-base | 109,482,240 | 7,087,872 | 2,307 = 0.002% | last-2 + classifier = 14,178,051 = **12.95%** |
| BERT-base pooler | 590,592 | — | — | 768×768 + 768; frozen in the video's snippet |
| Llama-3-8B | 8,030,261,248 | ~218M | — | last-4 + head ≈ 1.4B ≈ 17%; embeddings ≈ 525M |
| LoRA r=16 on 8B | — | — | 0.3M = ~0.5% | adapters 21 MB at r=16 on 8B (7B-family) |

### Data volume

| Task | Knee (labels) | Plateau |
|---|---|---|
| Binary / few-class classification, in-distribution | 10–30/class (~100 total) | 1k–5k |
| Fine-grained classification (100+ classes) | 50–100/class | 10k–50k |
| NER / sequence labelling | 500–2,000 sentences | 10k–20k |
| Extractive QA | 1,000–5,000 triplets | 20k–50k |
| SFT — style / format / schema | **500–2,000** | 5k–10k |
| SFT — new domain behavior | 5,000–50,000 | 100k+ |
| Continued pretraining | 0 labels | 1e8–1e10 unlabeled tokens |

Rules: 10 × number of classes is the head-only floor. 100× the labels buys ~1 point. 500–2,000 teaches format, 10k+ teaches behavior, nothing teaches facts.

### Datasets and case-study results

| Quantity | Value |
|---|---|
| `dair-ai/emotion` train split | 16,000 rows, 6 classes |
| Filtered to sadness/joy/anger | **12,187** rows (4,666 / 5,362 / 2,159) |
| 80/20 split | **9,749 / 2,438** (video [54:39], [55:10]) |
| BERT-base, head only | 0.885 acc / 0.874 macro-F1, ~70 s on T4 |
| BERT-base, last-2 + head | **0.925 acc / 0.917 macro-F1**, ~4 min, 12.95% trainable |
| BERT-base, full FT @ 2e-5, 3 ep | 0.933 / 0.926, ~11 min |
| BERT-base, full FT @ 5e-5, 10 ep (the mistake) | 0.918 / 0.906, ~35 min |
| Majority-class baseline (joy) | 0.440 |

### Forgetting thresholds

| Signal | OK | Investigate | Stop |
|---|---|---|---|
| MMLU delta | > −0.5 | −0.5 to −2.0 | < −2.0 |
| WikiText ppl ratio | < 1.05 | 1.05–1.20 | > 1.20 |
| KL-to-base (2k prompts) | < 0.02 nats | 0.02–0.10 | > 0.10 |
| Base-embedding cosine drift | < 0.03 | 0.03–0.10 | > 0.10 |

Regularizer magnitudes: `λ_L2SP ∈ [1e-4, 1e-2]`, `λ_EWC ∈ [1e2, 1e4]`, `β_KL ∈ [0.01, 0.5]` SFT (~0.01–0.1 RLHF), replay fraction 1–10% (5% default).

### Cost

| Job | Config | Time | Cost |
|---|---|---|---|
| BERT-base, 9.7k examples, 3 epochs | last-2 unfrozen, T4 | 4 min | $0.00 |
| Llama-3-8B QLoRA, 10k examples, 3 epochs | NF4, 4090 | 2.5–4 h | **$1.20** |
| Llama-3-8B full FT, 100k examples, 2 epochs | 4×A100 80 GB, FSDP | 18–26 h | $130–$200 |
| Clinical DAPT, 2.1B tokens | 2×A100 80 GB | 19 h | $76 → +10.8 F1 |
| Clinical NER SFT | 3,100 notes, 4 epochs | 18 min | $0.60 |
| LoRA adapter, 300–800 examples | 4090 | ~12 min | $0.08 → 21 MB |

Rental prices (2025): Vast.ai 4090 $0.35–0.60/h; RunPod A100 80 GB $1.50–2.50/h; H100 80 GB $2.50–4.00/h; AWS p4d.24xlarge (8×A100) $32–40/h. Free tiers: Colab T4 12.7 GB (no bf16), Kaggle 2×T4 30 h/week.

### Other numbers worth having on instant recall

- `ln(2) = 0.693`, `ln(3) = 1.099`, `ln(4) = 1.386` — the flat-loss signatures.
- ULMFiT discriminative ratio `1/2.6 = 0.385`; production LLRD decay 0.8–0.9; STLR `cut_frac=0.1`, `ratio=32`.
- LoRA: `A` random Gaussian, `B` zero ⇒ `ΔW = 0` at step 0; effective scaling `alpha/r`.
- ILSVRC: 1.2M train / 50k val / 100k test, 1,000 classes; ImageNet: 14M images, 21,841 synsets.
- Pretraining anchors: BERT-base 110M params / 33B tokens / 4 days on 16 Cloud TPU v3; Llama-3-8B 1.3M H100-hours / 15T tokens.
- Alignment tax measured in CS-02: MMLU −0.6, WikiText ppl ratio 1.07 — acceptable; the 8-epoch variant lost 4.1 MMLU points.
- Adapter serving overhead: +11 ms p95 for vLLM multi-LoRA on a 40-tenant fleet.

---

## Answers To The Self-Check Questions From CS-02

**1. Define transfer learning and fine-tuning, and state precisely how they differ. Give one example of transfer learning with no fine-tuning.**
Transfer learning is the strategy of reusing knowledge from a source domain/task to improve a target domain/task; fine-tuning is continuing training of a pretrained model on target data. Transfer learning is the frame, fine-tuning one realization of it — the instructor's "two sides of a single coin" [25:29]. Examples with no fine-tuning: running a pretrained model zero- or few-shot on a new task, or feeding frozen ImageNet features into a classical SVM. Both reuse source knowledge with zero gradient steps on the target.

**2. In the four-quadrant taxonomy, which quadrant does "clinical NER using a Wikipedia-pretrained BERT" occupy, and what is the correct first move?**
**Q3 — different domain, same task.** The task (token classification) is unchanged; the input distribution is not (clinical shorthand `pt c/o SOB`, `H/O DM2`, `q.d.`, abbreviations, section headers, largely absent from Wikipedia + BooksCorpus). Correct first move: **continued pretraining (MLM) on unlabeled clinical text**, then supervised NER. In CS-02's §15.4 that ordering was worth +10.8 entity-level F1 (0.912 vs 0.804), with the DAPT phase costing 19 hours / $76 and the SFT phase 18 minutes / $0.60.

**3. Why does the instructor say we should never fine-tune the earliest layers? Give the feature-hierarchy answer *and* the anti-forgetting answer, then name one case where the rule should be broken.**
Feature hierarchy: early layers compute primitive features — edge, texture, shape — which are task-agnostic, so there is nothing task-specific to gain by moving them [21:17]. Anti-forgetting: the early layers hold the most general and most fragile representation, and gradient steps taken to fit a small target set overwrite them, degrading everything else; drift scales with `η·‖g‖·t`. Break the rule when the target's low-level statistics differ from the source's — grayscale medical scans, thermal imagery, spectrograms, or a different input channel count. Then either re-initialize and train the stem/first block, or start from a checkpoint trained on comparable low-level statistics.

**4. A colleague fine-tunes BERT-base for 3-class classification and gets a training loss that decreases but a validation loss that rises after epoch 2. List four things to check, in order.**
(1) **Count trainable parameters and print `model.summary()`/the first trainable tensor** — a freeze/optimizer-construction mistake explains everything downstream. (2) **Validate the split and the labels** — duplicates across train/val, label-index mismatch, class imbalance, near-duplicate leakage. (3) **Lower the LR and cut the epochs** — rising val loss after epoch 2 at LR 2e-5 on a few thousand examples is the classic overfit-with-drift signature; Mosbach et al. (2021) recommend a *lower* LR and *more* epochs than the original BERT recipe. (4) **Check regularization and early stopping** — `weight_decay=0.01`, head dropout, and early stopping on a metric that includes a general-capability term. Only if all four are clean should you conclude the model is too large for the data.

**5. You have 800 labeled examples and one 16 GB GPU. Which of the video's three configurations do you choose, and what learning rate? Justify the LR choice.**
**Way 2 — freeze the conv base / encoder body and train a fresh head**, with Way 1 (head only) as the first experiment and Way 3 (last block) only if the frozen run plateaus below requirement. LR: **1e-3 for the head** with the body frozen, or **1e-5 with 10% warmup** if you unfreeze the last block. Justification: with the body frozen there is no pretrained parameter at risk, so the head is effectively trained from scratch and needs a from-scratch LR; the moment *any* pretrained weight receives gradient, the LR must drop ~100× or the checkpoint is destroyed within a few hundred steps. Note the measured context: head-only on BERT-base was 0.885 vs 0.925 for last-2 + head — the last block is worth points if you can afford the risk, and at 800 examples you usually cannot.

**6. What is LP-FT, mechanically, and why does it beat both of its components?**
Phase 1: freeze the entire backbone, train only the head at a high LR (1e-3), a few epochs. Phase 2: unfreeze everything and fine-tune with a *small* LR (2e-5) for a short run with ~10% warmup. It beats pure linear probing because phase 2 adapts the features to the task (raising the ceiling above the frozen representation); it beats pure fine-tuning because phase 1 replaces the random head *before* any gradient reaches the backbone, so phase 2 starts from a low-loss point and the gradients that reach the backbone are small and task-aligned instead of large and noise-driven. Kumar et al. (ICLR 2022) show it wins in both the in-distribution and OOD regimes.

**7. Name six mitigations for catastrophic forgetting, ranked by implementation cost, and say which is the cheapest that actually works.**
(1) **Lower the LR** to 1e-5–2e-5 — free, one number. (2) **Fewer epochs / early stopping on a general metric** — free. (3) **Warmup 6–10%** — free. (4) **LoRA/PEFT** — one dependency and a config change; the base weights cannot move. (5) **Replay 1–10% general data** (5% default) — needs a data pipeline. (6) **L2-SP or KL-to-base** — a training-loop change, and KL needs the base resident. (7) **EWC** — an extra Fisher pass over source data plus λ tuning. The cheapest that actually works at scale is **replay**: it restores a source-distribution gradient signal in every batch, needs no coefficient tuning, and is standard in production SFT (PPO-ptx in InstructGPT is the same idea inside RL).

**8. Your ImageNet-pretrained VGG16 fine-tune plateaus at 0.84 accuracy on a cat/dog task that should be easy. Name the single most likely cause and the one-line fix.**
The input normalization does not match the backbone's training-time transform. `tf.keras.applications.VGG16` was trained with `preprocess_input` — RGB→BGR conversion plus ImageNet channel-mean subtraction — while the video's notebook uses `ImageDataGenerator(rescale=1./255)` [41:25]. Fix: `tf.keras.applications.vgg16.preprocess_input`. Cost when wrong: 8–12 points of accuracy with a completely healthy-looking loss curve. **The video's own notebook has this bug**; in CS-02's case study the fix took the run from 0.84 to 0.961.

**9. The `dair-ai/emotion` dataset is filtered to sadness/joy/anger and `num_labels=3` is passed. What breaks, and what are the two possible symptoms?**
`Dataset.filter()` does not renumber labels. The source `ClassLabel` order is `['sadness','joy','love','anger','fear','surprise']`, so keeping sadness/joy/anger leaves the label column holding `{0, 1, 3}` while `num_labels=3` expects `{0, 1, 2}`. **Symptom A (loud):** `IndexError: Target 3 is out of bounds` on the first batch containing an anger example. **Symptom B (silent):** if you set `num_labels=4` instead, you train a four-way head with class 2 permanently empty — the loss floors near `ln(4) = 1.386`, accuracy caps below 1.0, and the model never predicts the third class. Fix: remap `{0:0, 1:1, 3:2}` after filtering, and assert `labels.max() < model.config.num_labels` in the pre-flight.

**10. Estimate the VRAM needed to full-fine-tune a 7B model with Adam in bf16, and the same for QLoRA. Show the arithmetic.**
**Full FT:** 16 bytes per trainable parameter × 7e9 = **112 GB** (bf16 weights 2 + bf16 gradients 2 + fp32 Adam m,v 8 + fp32 master weights 4), plus activations — at bs 4 × len 2048 across 32 decoder layers this is several more GB. Realistically ~120–130 GB, requiring 2×A100 80 GB with FSDP/ZeRO-3 as a floor. The shorthand to quote: full FT needs ~20× the parameter count in GB. **QLoRA:** 4-bit NF4 base ≈ 7e9 × 0.5 bytes = **3.5 GB**, adapter gradients/optimizer ≈ 1% of parameters × 16 bytes ≈ **1.1 GB**, activations ≈ **2–3 GB** at bs 4 / len 1024 with `gradient_checkpointing=True` ⇒ **~7–8 GB total**, i.e. one 8–12 GB card or a 24 GB 4090 with substantial headroom. The economic consequence: a 3-hour QLoRA run on a rented 4090 costs ~$1.35 versus ~$150 for the full FT, for what is typically 1–3 points of task accuracy.
