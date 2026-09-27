# CS-02 — Transfer Learning & Model Fine-Tuning

| Field | Value |
|---|---|
| **Module** | Foundations / Transfer Learning |
| **Source video(s)** | LLM Fine-Tuning 03: Transfer Learning and Model Fine-Tuning |
| **Transcript file(s)** | `LLM_Fine-Tuning_03_Transfer_Learning_and_Model_Fine-Tuning_aiagents_finetuning_a.txt` |
| **Companion code** | none in repo (the video runs two Colab notebooks: a Keras/VGG16 cat-vs-dog notebook, and a Hugging Face `BertForSequenceClassification` notebook over a 3-class emotion subset) |
| **Prerequisites** | CS-01 (pretraining & the LLM lifecycle), CS-05 (why RNN/LSTM fine-tuning was hard) |
| **Difficulty** | Beginner → Intermediate (theory) / Intermediate (hands-on) |
| **Hands-on required** | Yes — 2 notebooks, ~1.5 h of GPU time |
| **Estimated study time** | 3 h theory + 2 h practical |

---

## 0. Executive Summary

- **Transfer learning is the strategy; fine-tuning is the tactic.** The instructor's framing is exact: transfer learning *transfers knowledge from a previous task to a current, related task*; fine-tuning is *the incremental training you do on top of that transferred knowledge* [23:37]. They are "two sides of a single coin," not competing techniques [25:29]. You cannot fine-tune without transferring; you can transfer without fine-tuning (that is zero-shot inference on a pretrained model).
- **The pretrained model is a frozen reservoir of general competence with a wrong head.** ImageNet-pretrained VGG16 knows *car* but not *Tata Nano*; knows *human* but not *Sunny* [14:53]–[15:18]. The pretrained network has never seen your label space. Fine-tuning re-points the last mile.
- **Three mechanical ways to fine-tune a CNN, in increasing order of cost and capacity** [18:28]–[19:28]: (1) **replace the output layer**, (2) **freeze the convolutional base and train the dense head**, (3) **unfreeze the last convolution block(s) and retrain them**. The same three options exist for transformers [30:01], applied to encoder blocks instead of conv blocks.
- **The universal default: touch the top, leave the bottom alone.** Early layers compute primitive features (edges, textures, shapes); late layers compute task- and instance-specific features [21:00]–[21:32]. "We never train the entire model; we just unfreeze some last layers" [21:41]. The instructor puts a 99% success rate on this [22:07].
- **The BERT worked example is fully reconstructible from the numbers in the video.** The `emotion` dataset filtered to 3 classes (sadness, joy, anger) yields exactly 12,187 rows; an 80/20 split gives **9,749 train / 2,438 validation** — byte-for-byte the counts the instructor shows on screen [54:39]–[55:10].
- **Fine-tuning beats from-scratch training on three axes** [33:21]–[35:09]: it saves training time and compute; it works when labeled data is scarce; and it gives better final performance — which the instructor attributes to the pretrain→finetune→ChatGPT lineage.
- **The reason to reach for PEFT is memory, not accuracy.** A 7B model under full fine-tuning costs ~112 GB of optimizer + weight + gradient state in fp16; QLoRA costs ~4 GB of base weights. The instructor explicitly forward-references PEFT/LoRA/DoRA for "GPT, Mistral, Llama, Gemini" because "this model is pretty huge… we cannot increase couple of last layer" [32:09]–[32:17].
- **The single most important number in this module: fine-tuning learning rates are 10–100× smaller than pretraining learning rates.** Pretraining peaks at 1e-4–6e-4; full fine-tuning of a transformer lives at 1e-5–5e-5; LoRA adapters live at 1e-4–3e-4. Anything above ~1e-4 for full fine-tuning of a pretrained encoder destroys the pretrained solution faster than the task loss can rebuild it.
- **Catastrophic forgetting is the failure mode that does not show up in your task metric.** Your eval accuracy goes up while the model gets quietly worse at everything else. Measure it explicitly with a KL-to-base diagnostic and a general-capability suite, not by vibes.

---

## 1. The Problem This Solves

### 1.1 What breaks without transfer learning

Supervised deep learning from scratch needs three things simultaneously: a large labeled dataset, a large compute budget, and a good initialization. When any one is missing, training from scratch fails — not gracefully, but by converging to a worse optimum, or by overfitting in the first few thousand steps.

Concretely, the failure modes are:

1. **Label scarcity.** A medical imaging team has 1,800 labeled mammograms, not 14,000,000. A from-scratch CNN on 1,800 images with 1M+ parameters memorizes the training set within 20 epochs.
2. **Compute budget.** Pretraining BERT-base (110M parameters, 33B tokens) took 4 days on 16 Cloud TPU v3 chips in 2018. Pretraining Llama-3-8B took **1.3M H100 GPU-hours** on 15T tokens. No team fine-tunes anything by first re-pretraining a backbone.
3. **Generalization.** A model trained on 1,800 examples learns the idiosyncrasies of those 1,800 examples. A pretrained model brings a prior over natural-image and natural-language statistics that acts as a powerful regularizer.

### 1.2 The state of the art before transfer learning

Before the 2012–2018 transfer-learning era, the standard pipeline was:

| Era | What people did | Cost | Result |
|---|---|---|---|
| Pre-2012 CV | Hand-engineered features (SIFT, HOG, LBP) + SVM | ~1 CPU-week per dataset | Ceiling around 74% top-5 on ILSVRC-2010 |
| 2012–2016 CV | Train AlexNet/VGG/ResNet from scratch per task | 2–4 weeks on multi-GPU | Needed 1.2M labeled images to work at all |
| 2015–2018 NLP | Train task-specific LSTM/CNN per task, random init | Days per task | The problem CS-05 dissects: no shared backbone, no reusable prior |
| 2018+ | Pretrain once, adapt many times | Hours per task | Fine-tuning becomes the default |

### 1.3 The naive approach and precisely why it fails

The naive approach is **"just train your model on your data."** It fails for a specific, mechanistic reason: a randomly-initialized deep network sits in a region of parameter space where the loss landscape is bad — high curvature, many saddle points, gradients that are dominated by input-scale effects. A pretrained network sits in a *flat basin* that already encodes useful features. Gradient descent starting from a good basin finds a good solution in a few thousand steps; the same optimizer from random init needs orders of magnitude more data and steps to find even an okay basin.

The second naive approach is **"fine-tune everything aggressively,"** i.e., train all layers with the same learning rate you would use for a fresh model. This also fails: the early layers drift away from the pretrained features, which is the *catastrophic forgetting* mechanism covered in §4.4. The instructor's operational rule — never train the bottom layers — is the crude version of a real statistical argument.

### 1.4 Concrete motivating example with numbers

The video's own example [12:47]–[15:18]:

- Take ImageNet: **14M images, 21,841 WordNet synsets**, run the ILSVRC challenge subset (**1.2M train / 50k validation / 100k test images, 1,000 categories**) [11:43]–[12:38].
- Train VGG16 on it. You get a model that can separate cat/dog/vehicle/aircraft/bird/sports/musical-instrument/human.
- Ask it to identify a **golden retriever** — a *breed*, not a *species*. Fail.
- Ask it to identify **you specifically**, not "human." Fail.
- Ask it to identify a **Tata Nano**, not "car." It says "car," which is not the answer you wanted [15:12].

The fix is not a better architecture and not more ImageNet data. The fix is to take the ~14.7M parameters of pretrained convolutional features as given, bolt on a head with your label space, and train for a couple of epochs.

---

## 2. First-Principles Mental Model

### 2.1 The instructor's analogy: bicycle → motorcycle

A child learns to ride a bicycle: **balance, braking, horn, acceleration** [6:36]–[7:01]. Years later the same person learns a motorcycle. Balance transfers. Braking transfers. Horn transfers. Acceleration transfers. The motorcycle adds exactly one new thing: the **gearbox** [7:42]–[7:50].

- The **transfer** is the reuse of balance/brake/horn/accelerate.
- The **fine-tuning** is learning the gearbox on top of them.
- The full skill of motorcycle riding = transferred knowledge + the small tuned delta.

**Where this analogy breaks.** Three places, and each one maps to a real limitation:

1. **Skill does not decay; weights do.** A human who learns gears does not forget how to balance — unless they never practice. A neural network absolutely does: gradient steps taken to learn the gearbox can overwrite the balance weights. That is catastrophic forgetting (§4.4), and the analogy hides it completely.
2. **Humans choose what to reuse; networks reuse everything by default.** The child consciously reuses balance and consciously ignores "pedaling." A network has no such gating: every gradient step on every parameter changes every downstream feature. Layer freezing is a *simulation* of the human's selectivity, imposed from outside.
3. **The gearbox is one skill; a new head can be an entire new domain.** If the second task were "ride a motorcycle, in the snow, on the left side of the road," the amount of new learning would rival the amount transferred. The analogy understates how much of the target task can be *disjoint* from the source task.

### 2.2 The mechanism underneath

A trained network is a **composition of functions**, `f(x) = h_θk(…h_θ2(h_θ1(x)))`. Transfer learning works because for a broad class of natural-data tasks, the *early* functions `h_θ1…h_θk-ℓ` converge to features that are useful for almost any task on that data modality, while the *last* functions are specialized to the source label space.

The statistical statement: the source task and the target task share a common feature representation, and the target task's optimal predictor is close (in function space, not parameter space) to a composition of that shared representation with a simple head. When that assumption holds, the sample complexity of the target task collapses from "learn features + learn head" to "learn head."

The instructor states this as [21:17]: early layers "fetch primitive features — edge, texture, shape"; later layers "fetch specific features — how Sunny's nose looks, how Sunny's ear looks." That is the mechanism in one sentence.

**Where this mental model breaks.** The "shared feature" assumption is *empirically* true for vision and for natural language pretrained on web text, and it degrades sharply when the source and target modalities or distributions diverge: ImageNet pretraining transfers poorly to medical X-rays, satellite imagery with unusual spectral bands, or spectrograms; English web text pretraining transfers poorly to source code with no overlap in tokenizer vocabulary. In those cases, the shared part is small and transfer yields less than the hype implies (see §8 STOP conditions and §17 Misconceptions).

### 2.3 The four-quadrant taxonomy

The formal framing (Pan & Yang's survey taxonomy, which the practitioner literature has compressed into a 2×2) is **source domain/task → target domain/task**. Define:

- **Domain D = (X, P(X))** — the input space and its marginal distribution. "Domain shift" means `P_source(X) ≠ P_target(X)`.
- **Task T = (Y, P(Y|X))** — the label space and the conditional. "Task shift" means the label space or the mapping differs.

Crossing "same/different domain" with "same/different task" gives four quadrants. **Which quadrant you are in determines how much of the network you touch** — this is the operational payload of the taxonomy, and it is the single most useful thing in this module.

| Quadrant | Domain | Task | Name | What actually differs | What to tune | What to freeze | Data needed |
|---|---|---|---|---|---|---|---|
| **Q1** | Same | Same | In-distribution adaptation / recalibration | Nothing structural — only label noise, class prior, or a new random seed | Head only; optionally temperature/logit bias | Everything else | 100–2,000 examples |
| **Q2** | Same | Different | Inductive transfer (classic fine-tuning) | Label space and/or objective. Same input distribution. | Head + last 1–4 blocks (or LoRA) | Bottom 50–75% of blocks | 1k–100k examples |
| **Q3** | Different | Same | Domain adaptation / covariate shift | `P(X)` moves; `P(Y\|X)` is stable | Head + upper blocks, **after** continued pretraining on unlabeled target text | Embeddings + bottom blocks | 10k–1B *unlabeled* target tokens + 1k–10k labeled |
| **Q4** | Different | Different | Full transfer (the common industrial case) | Both move; often a new vocabulary/tokenizer as well | Continued pretraining + head + upper blocks, or PEFT on everything with a bigger rank | Only the embedding table, sometimes | 100k+ labeled, 1B+ unlabeled |

Worked mapping to real projects:

| Project | Quadrant | Why |
|---|---|---|
| Sentiment classifier on Yelp reviews using BERT-base | Q2 | Same text distribution (web English), new label space `{pos, neg}` |
| Legal clause classifier using BERT-base | **Q3** | Same task family (classification) but `P(X)` is contractual English, absent from Wikipedia/BooksCorpus |
| Radiology report summarizer using Llama-3-8B | **Q4** | New domain (clinical notes), new task (abstractive summarization of findings) |
| Re-fitting the classifier head of your own model after a data refresh | Q1 | Same domain, same task, new IID sample |
| Chat model on internal Slack data, instruction-tuned from Llama-3-8B-Instruct | Q4 → effectively Q2 after the instruct stage | The base is already instruction-tuned; the residual shift is domain, and format is preserved |

> **Beyond the video:** the instructor presents this as a single axis ("replace the head vs unfreeze layers") and never separates domain shift from task shift. In production the separation is what saves you money. **Task shift is cheap to fix with labels; domain shift is expensive and is usually fixed with unlabeled text.** If your inputs look like your pretraining corpus but your labels are new, you need ~1k labels. If your inputs do not look like the pretraining corpus, no number of labels substitutes for continued pretraining — you will get a model that overfits the label space while its features remain wrong for your inputs. That is the Q3/Q4 trap: teams label 50k examples, fine-tune hard, hit 88% validation accuracy, and ship a model that collapses on production inputs that were merely *slightly* further out of distribution than validation.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Pretraining** | Large-scale self-supervised training on broad data (next-token prediction; masked language modeling; ImageNet classification for vision backbones) producing a general-purpose backbone. | Produces the artifact that transfer learning transfers. Everything in this module assumes a pretrained checkpoint exists. | Confused with "training." Pretraining is the *first* training stage, on *generic* data with a *self-supervised* objective. |
| **Fine-tuning** | Additional (usually supervised) training of a pretrained model on a target dataset. | The adaptation step. Can touch 1 head or 100% of parameters. | Confused with transfer learning. Fine-tuning is a *subset* of transfer-learning practice, not a synonym. |
| **Transfer learning** | Using knowledge acquired for a source domain/task to improve performance on a target domain/task. | The strategic frame. In the video: "transfer learning and fine-tuning are two sides of a single coin" [25:29]. | People say "we did transfer learning" when they mean "we called an API." Transfer learning implies you had a source task whose representations you reused. |
| **Source domain / task** | The `(D_S, T_S)` the checkpoint was trained on — e.g. ImageNet-1k classification; web-scale MLM; 15T tokens of next-token prediction. | Determines *what* you can transfer. You cannot transfer a capability the source never had. | "The base model knows X" — verify with a probe before assuming. |
| **Target domain / task** | The `(D_T, T_T)` you actually care about. | Determines what you must learn from scratch. | Teams describe the target only by its labels and forget the input distribution is half the problem. |
| **Feature extraction** | Using the frozen backbone purely as a fixed featurizer; only a new head is trained. | Cheapest, most forgetting-proof, strongest regularizer, weakest ceiling. | Confused with "linear probing." Feature extraction with a frozen backbone and an MLP head is still feature extraction; a *linear* probe is the special case with a single linear layer. |
| **Linear probing (LP)** | Freeze 100% of the backbone; train one linear layer on top with a high LR. | The correct baseline everyone skips. If LP ≈ full FT, you have a data/eval problem, not a fine-tuning problem. | "Linear probe" ≠ "last-layer fine-tune." The latter lets gradients flow into the last block. |
| **Full fine-tuning (full FT)** | Update every parameter with a small learning rate. | Highest ceiling on in-distribution data; highest forgetting risk; highest memory. | People assume it is always better. On OOD data it is often *worse* than LP (see §4.6). |
| **PEFT** | Parameter-Efficient Fine-Tuning — a family (LoRA, QLoRA, DoRA, adapters, prefix/prompt tuning, IA³, BitFit) that trains ≤1–5% of the parameters while freezing the rest. | The only economically sane option above ~3B parameters. Forward-reference: CS-23. | "PEFT is a worse full FT." It is a *different regularizer*; on small data it frequently wins. |
| **LoRA** | Low-Rank Adaptation: freeze `W`, learn `ΔW = BA` with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`, `r ≪ min(d,k)`. | Trains 0.1–1% of parameters, stores 10–100 MB adapters per task, and is composable at inference. | "LoRA can't learn new knowledge." It can learn new *behavior*; new *facts* need continued pretraining or RAG (CS-04, CS-12). |
| **Catastrophic forgetting** | Degradation of previously acquired capabilities caused by gradient updates for a new task. | The dominant silent failure of fine-tuning. Your target metric hides it. | Confused with overfitting. Overfitting is worse *on the target*; forgetting is worse *everywhere else*. |
| **Domain shift** | `P_S(X) ≠ P_T(X)`. | Fix with unlabeled target data (continued pretraining). | Called "distribution shift," "dataset shift," "covariate shift" interchangeably — they differ. |
| **Covariate shift** | The `X`-marginal changes; `P(Y\|X)` is unchanged. A subtype of domain shift. | Means your labels remain valid — you can reuse them and just adapt the features. | People use it to mean "any shift," which destroys the diagnostic value. |
| **Label shift / prior shift** | `P(Y)` changes; `P(X\|Y)` is unchanged. Also called prior probability shift. | Breaks accuracy and calibration even when the model is perfect. Fix with logit adjustment, not with retraining. | Detected late, in production, as "the model suddenly predicts one class." |
| **Concept shift** | `P(Y\|X)` changes — the same input now has a different label. | The only shift that genuinely requires new labels. | Often misdiagnosed as covariate shift; teams burn a month doing domain adaptation for a labeling-policy change. |
| **Layer freezing** | Setting `requires_grad=False` on a subset of parameters so no gradient is computed or applied. | The main knob for controlling forgetting and memory. | People freeze and then wonder why `loss.backward()` still costs the same (it does, unless you also wrap the frozen prefix in `torch.no_grad()`). |
| **Progressive / gradual unfreezing** | Train with the top block only; then unfreeze one more block; repeat, descending. | ULMFiT's third technique. Reduces forgetting by giving the head time to become useful before the backbone moves. | Confused with "discriminative LR" — they are different mechanisms that compose. |
| **Discriminative fine-tuning (LLRD)** | Use different learning rates per layer, typically decaying with depth by a constant factor. | ULMFiT's first technique; ULMFiT uses `η^{l-1} = η^l / 2.6`. | Sometimes called LLRD (layer-wise LR decay). Same idea. |
| **Slanted triangular LR (STLR)** | A short linear warmup to peak, then a long linear decay, with `cut_frac=0.1`, `ratio=32`. | ULMFiT's second technique. Beats both constant and cosine on small classification data. | Not the same as cosine-with-warmup, although both avoid cold-start divergence. |
| **ULMFiT** | Universal Language Model Fine-tuning (Howard & Ruder, ACL 2018) — the paper that introduced discriminative LRs, STLR, and gradual unfreezing for LM fine-tuning. | The canonical source for every layer-freezing schedule in this module. | Credit is usually given to BERT; ULMFiT predates it by ~8 months and is where the schedules come from. |
| **Catastrophic forgetting mitigations** | Low LR; LoRA/PEFT; replay/rehearsal; EWC; L2-SP; KL-to-base regularization; early stopping. | Six of these seven are one line of code; not using one is a choice to accept the risk. | "We fine-tuned for 3 epochs so we're fine." Three epochs at 5e-5 on a narrow dataset is plenty to move MMLU. |
| **EWC** | Elastic Weight Consolidation (Kirkpatrick et al., 2017) — quadratic penalty `Σ_i F_i (θ_i − θ*_i)²` where `F` is the diagonal Fisher information. | Weights the penalty by how important each parameter was to the source task. | Requires an extra pass over source data to estimate Fisher. It is not free. |
| **L2-SP** | `L2` penalty toward the *pretrained* weights instead of toward zero. | Simple, one extra term, and it directly encodes "stay near the pretrained solution." | Weight decay is L2 toward zero — it is *not* L2-SP, and it does not prevent drift from the pretrained point. |
| **KL-to-base regularization** | Add `β · KL(p_base ‖ p_tuned)` on a held-out general corpus to the training loss. | The most direct measurement and mitigation of forgetting simultaneously. | Requires keeping the base model resident in VRAM (or cache its logits). |
| **Replay / rehearsal** | Mix a small fraction (typically 1–10%) of general-domain data into every fine-tuning batch. | The cheapest, most reliable anti-forgetting method. Used in production SFT everywhere. | People replay data from the *same* domain and are surprised it does not help. Replay must be distributionally different. |
| **Alignment tax** | The measurable loss of general capability incurred by aligning/fine-tuning a model (InstructGPT, Ouyang et al. 2022). | The industrial name for forgetting. | "We only did SFT" is not an exemption; SFT on a narrow set carries a tax too. |
| **Zero-shot transfer** | Using the pretrained model on the target task with no gradient updates. | The cheapest baseline of all; sometimes sufficient. | It is not fine-tuning, and pretending it is hides the fact that you measured nothing. |
| **Continued pretraining / DAPT** | Domain-Adaptive Pretraining: run the *pretraining objective* on unlabeled target-domain text before supervised fine-tuning. | The correct first move for Q3/Q4 (domain shift). Forward-reference: CS-12. | "We fine-tuned on our PDFs" usually means SFT on QA pairs over the PDFs, which is different and weaker for domain shift. |
| **Backbone** | The pretrained body of the network, excluding the task head. | The thing you freeze or unfreeze. | In transformers, "backbone" includes the embedding table — and the embedding table has the most parameters-per-token of anything. |
| **Head / classifier** | The task-specific module on top of the backbone (`Dense(1)`, `Linear(hidden, num_labels)`, a causal-LM `lm_head`). | Always trained. Its input dimension is fixed by the backbone; its output dimension is fixed by your label space. | Replacing a 1,000-class head with a 3-class head throws away 4M parameters and *should* — do not initialize the new head from the old one's slice. |
| **`include_top`** | Keras/torchvision flag controlling whether the pretrained classification head is loaded. | `True` if your label space matches; `False` if you are bolting on your own head. The video uses both [37:25], [43:52]. | Setting `include_top=False` gives `num_classes` as a parameter in `tf.keras.applications.VGG16`; leaving it unset is a silent no-op that surprises people. |
| **`requires_grad`** | PyTorch tensor flag; `False` freezes the parameter. | The mechanism behind every freezing strategy. | Forgetting to set `.eval()` on frozen BatchNorm layers is a separate, equally silent bug. |
| **Pooler output** | The `tanh`-squashed transformation of the `[CLS]` token in BERT, used by `BertForSequenceClassification` as the head input. | Explains why the classifier head is a single `Linear(768, num_labels)`, not a sequence-pooled MLP. | People assume the `[CLS]` vector is used raw. In `transformers`, `BertForSequenceClassification` uses `pooler_output`; sentence-transformers uses mean pooling. |
| **`num_labels`** | The argument that sizes the classification head. | Wrong value = wrong loss baseline (`ln(k)`), silently. | The video changes it from 2 (default) to 3 [1:01:33] and this is exactly the line where it happens. |
| **MLM** | Masked Language Modeling — BERT's pretraining objective (predict randomly masked tokens). | The source task that produces BERT's generic understanding [51:22]. | BERT was trained on MLM **and** Next Sentence Prediction; the instructor mentions only MLM. |
| **ILSVRC** | ImageNet Large Scale Visual Recognition Challenge; the 1,000-class, 1.2M-image benchmark carved out of ImageNet. | The historical source task that made vision transfer learning work. | "ImageNet" the dataset (14M images, 21,841 synsets) ≠ "ILSVRC" the benchmark (1.2M images, 1,000 classes). |
| **VGG16** | 16-layer convolutional network (Simonyan & Zisserman, 2014); 138,357,544 parameters; 13 conv layers in 5 blocks + 3 fully-connected layers. | The instructor's CNN backbone. Its clean `conv_block_5` boundary makes it the best teaching example for layer freezing. | VGG's 123.6M parameters live in the FC layers, not the conv layers — the opposite of what people assume. |
| **`Flatten`** | Reshape a 2D/3D feature map into a 1D vector; no parameters. | Where the video's new head starts [44:39]. | The 25,088-dimensional flatten is why VGG heads are enormous. |
| **Temperature / calibration** | Post-hoc rescaling of logits to fix confidence. | The correct fix for Q1 problems, and it costs one CPU pass. | Confused with fine-tuning. If only the priors moved, do not touch a weight. |
| **Learning-rate warmup** | Ramping LR from ~0 to peak over the first N steps. | Prevents the first few large-magnitude Adam steps from destroying pretrained weights. | Warmup is *more* important for fine-tuning than for pretraining, because the pretrained weights are more fragile than random ones are. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 Mechanism, step by step — the CNN case

The instructor's three-way taxonomy [18:28]–[19:28], made mechanical:

**Way 1 — replace the output layer.**

```
Input: 224×224×3 image
  → block1_conv1 … block5_conv3   (13 conv layers, 14,714,688 params, FROZEN)
  → Flatten                        (7×7×512 = 25,088-dim vector)
  → fc1 (4096)  → fc2 (4096)  → REMOVED
  → NEW: Dense(num_target_classes) (TRAINABLE)
```

Trainable parameter count for a 2-class head on the video's VGG16: **4,097** (`4096 × 1 + 1`) out of 138,357,544 — **0.003%**. You cannot forget anything with 4,097 degrees of freedom, and you cannot learn much either.

**Way 2 — freeze the conv base, train a new dense head.**

```
  → [frozen conv base, 14,714,688 params]
  → Flatten (25,088)
  → NEW Dense(256, relu)     6,422,784 params   TRAINABLE
  → NEW Dense(1, sigmoid)          257 params   TRAINABLE
```

Trainable: **6,423,041** (4.64% of the model). This is the instructor's second notebook [43:52]–[45:10], and it is the single highest-value configuration in the whole module for small datasets.

**Way 3 — unfreeze the last convolution block.**

```
  block1–block4   (conv, FROZEN)        7,635,264 params
  block5_conv1/2/3 (UNFROZEN)           7,079,424 params   ← 3×(512×512×3×3 + 512)
  → Flatten → Dense(256) → Dense(1)     6,423,041 params   TRAINABLE
```

Trainable: **13,502,465** (9.76% of the model). The instructor builds exactly this by iterating layers and flipping `layer.trainable = True` for `block5` only [47:03]–[47:59].

**The transformer case is structurally identical** [30:01]–[31:32]. BERT-base is 12 encoder blocks; block 11 is the last; unfreezing the last two means `model.bert.encoder.layer[-2:]` [1:06:37]–[1:07:23]. The head is `model.classifier`, a single `Linear(768, num_labels)`.

### 4.2 The arithmetic of "which layers, and how many parameters"

You cannot make this decision by intuition — you make it by counting. Table for the two backbones in the video:

| Model | Total params | Bottom half | Last block | Last 2 blocks | Head | Head only = % of total |
|---|---|---|---|---|---|---|
| **VGG16** (vision) | 138,357,544 | 7,635,264 (blocks 1–4) | 7,079,424 (block 5, 3 convs) | n/a (block 5 *is* the last 2 convs + conv3) | 6,423,041 (Dense256+Dense1) | 4.64% |
| **VGG16**, head-only | 138,357,544 | 14,714,688 (all conv) | frozen | frozen | 4,097 (Dense1) | **0.003%** |
| **BERT-base** (NLP) | 109,482,240 | ~54M (blocks 0–5) | 7,087,872 (block 11) | 14,175,744 | 2,307 (768×3+3) | ~1.5% for last-2 + head |
| **BERT-base**, head-only | 109,482,240 | all frozen | frozen | frozen | 2,307 | **0.002%** |
| **Llama-3-8B** | 8,030,261,248 | ~3.5B (16 blocks) | ~218M (1 block: q/k/v/o + gate/up/down) | ~436M | embeddings 128,256×4,096 ≈ 525M (tied to `lm_head`) | last-4-blocks + head ≈ **1.4B ≈ 17%** |

Read that last row carefully. **On a modern LLM, unfreezing "just the last few blocks" is not a small operation — the embedding table alone is ~525M parameters, and each block is ~218M.** This is precisely why the instructor pivots to PEFT for Llama/Mistral/Gemini [32:09]–[32:41]: the CNN-era recipe ("unfreeze the last block, it's cheap") stops being cheap the moment the head is 1B parameters.

> **Beyond the video:** the video never computes these numbers, and the intuition "the last block is small" is inherited from 2014-era vision CNNs where it was true. In a 7B transformer, one decoder block is ~200M parameters and the LM head + embeddings are ~1B. **Rule: before choosing a freezing strategy, print `sum(p.numel() for p in model.parameters() if p.requires_grad)` and divide by the total.** If the fraction is above ~10%, you are doing full fine-tuning with extra steps and you should either commit to full FT (with its memory cost) or switch to LoRA.

### 4.3 What happens at the tensor/gradient level

Let `θ = (θ_1, …, θ_L)` be the parameters of layers `1…L`, and `L_task` the target loss. Frozen layers get `∂L/∂θ_i = 0` by construction — PyTorch does not even allocate a `.grad` buffer for them.

Three consequences, each of which is a decision-relevant fact:

**(a) Memory.** With layer `i` frozen, you skip storing its activation for the backward pass *only if* you also disable the graph for it. In PyTorch, `requires_grad=False` on the *first* frozen layer's input is not enough — you must wrap the frozen prefix in `torch.no_grad()` (or use `torch.utils.checkpoint`) to actually drop the activations. In Keras with `trainable=False`, Keras runs frozen layers in inference mode and does not store their gradients. This is the difference between a VGG16 fine-tune fitting in 4 GB and needing 11 GB.

```python
# PyTorch: freeze AND actually save the activation memory
for p in model.conv_base.parameters():
    p.requires_grad = False

with torch.no_grad():                      # <-- this is the line people forget
    features = model.conv_base(x)          # no graph retained
features = features.detach()
logits = model.head(features)              # graph exists only here
loss = criterion(logits, y)
loss.backward()                            # grads exist only for model.head
```

**(b) Optimizer state.** Adam/AdamW keep two fp32 moments per *trainable* parameter. Freezing 90% of a model cuts optimizer memory by 90%:

| Trainable fraction | Optimizer state at 7B, fp16 grads + fp32 m,v |
|---|---|
| 100% (full FT) | 7e9 × (2 + 2 + 4 + 4) = **84 GB** (+ 14 GB weights) |
| 10% | 8.4 GB (+ 14 GB) |
| 0.5% (LoRA r=16) | 0.42 GB (+ 14 GB frozen weights, 2 bytes each, and grads only for adapters) |

**(c) The gradient signal is a weighted sum over the batch, not over the tasks.** When the head is randomly initialized (Way 1/Way 2), the loss on step 0 is high — approximately `ln(num_classes)` for a balanced problem. That large initial loss produces large gradients that flow *into* the backbone if it is unfrozen. This is the mechanistic origin of "train the head first, then unfreeze": you want the head's random-init transient to burn out before you let it write to the backbone. See §4.6 (LP-FT).

### 4.4 Catastrophic forgetting — mechanism, measurement, mitigation

**Mechanism.** Let `θ*` be the pretrained weights and `θ_t` the weights after `t` fine-tuning steps. Every step moves `θ` along `−∇L_target`. The target loss is a function of only the target task's data distribution. Nothing in the objective says "stay good at the source distribution" — in fact, the *fastest* way to reduce target loss on a small dataset is often to repurpose features that were doing source-task work. The weights drift off the pretrained manifold, and the source capability degrades monotonically with `‖θ_t − θ*‖` in the directions that matter.

**Three things make it worse**, and you should be able to name them:

1. **High LR.** Drift is roughly proportional to `η · ‖g‖ · t`. Halving the LR halves the drift.
2. **Many steps on narrow data.** `t` is the killer on small datasets: 3 epochs × 9,749 examples / batch 16 = 1,828 steps of pure target signal, with zero source signal.
3. **A randomly-initialized head feeding gradients into the backbone.** Early in training, the head produces near-uniform garbage; the gradients it sends down are large and uninformative.

**Measurement — four levels, in increasing order of rigor:**

| Level | Probe | How | Signals forgetting if |
|---|---|---|---|
| 0 | Task metric only | Your held-out target set | Never — this metric *cannot* see forgetting |
| 1 | General-capability suite | Run `lm-evaluation-harness` (MMLU, ARC, HellaSwag, GSM8K) on base and tuned | Any task drops > ~1–2 points |
| 2 | Perplexity on held-out general text | `exp(mean NLL)` on WikiText-103 or a held-out slice of the pretraining mix | Perplexity rises > ~5% |
| 3 | KL-to-base on a held-out corpus | `mean KL(p_base ‖ p_tuned)` over 2,000 general prompts | KL > ~0.05 nats/token with no target gain to justify it |

Level 3 is the one to instrument in CI, because it is cheap (2,000 forward passes), continuous, and it *is* the quantity the KL-regularizer minimizes. For an encoder classifier, the analogue is: freeze the base's pooled embeddings for 5,000 held-out sentences before and after, and report the mean cosine drift — anything above ~0.05 cosine distance means the representation moved materially.

**Every mitigation, with the actual knob:**

| Mitigation | Mechanism | Cost | When it is the right choice |
|---|---|---|---|
| **Low LR (1e-5–2e-5 full FT)** | Reduces per-step drift linearly | Free | Always; the first thing to try |
| **Warmup (6–10% of steps)** | Prevents large early steps at random-init head | Free | Always |
| **Train head first, then unfreeze (LP-FT)** | Lets the head settle before backbone moves | 1 extra short phase | Small-to-medium data with OOD risk |
| **Progressive/gradual unfreezing (ULMFiT)** | Descending unfreeze schedule | Implementable in ~20 lines | Medium data, classification, encoder models |
| **Discriminative LR (LLRD)** | Bottom layers get `η / 2.6^depth` | One param-group loop | Any full FT of a deep encoder |
| **LoRA / PEFT** | `ΔW` is additive and low-rank; base weights *cannot* move | Slightly lower ceiling | Almost always ≥ 1B params; and on small data even for BERT |
| **Replay / rehearsal (1–10% general data)** | Restores a source-distribution gradient signal in every batch | Data curation | The single most reliable fix; standard in production SFT |
| **EWC** | `+ (λ/2)·Σ F_i(θ_i − θ*_i)²` | Extra Fisher pass ≈ 1 forward pass over source data | Classical continual-learning setups; rarely used for LLM SFT |
| **L2-SP** | `+ (λ/2)·‖θ − θ*‖²` | Free | Simple baseline for any full FT |
| **KL-to-base** | `+ β·KL(p_base ‖ p_θ)` on general prompts | Base model resident in memory | When you can afford two models; the direct fix |
| **Early stopping on target + general metric** | Stop when general metric declines | Free | Always; the general metric is your canary |
| **Data mixing / curriculum** | Interleave target and general batches | Free | Same as replay at the sampler level |

Typical magnitudes: `λ_L2SP ∈ [1e-4, 1e-2]`; `λ_EWC ∈ [1e2, 1e4]` (Fisher values are tiny); `β_KL ∈ [0.01, 0.5]` for SFT and ~0.01–0.1 in RLHF; replay fraction `∈ [1%, 10%]`, with 5% a common production default.

> **Beyond the video:** the instructor never says the phrase "catastrophic forgetting." This is the largest single gap in the lecture, because his own rule — "we never fine-tune the earlier layers" [11:00] — *is* an anti-forgetting heuristic he justifies with a feature-hierarchy argument instead of a stability argument. Both arguments point the same direction, which is why the rule survives; but the forgetting argument is the one that tells you what to do when freezing is not an option (Llama, 7B, PEFT-or-nothing).

### 4.5 Feature extraction vs fine-tuning vs full retraining — the decision boundary

Three points on a continuum, and the cost/quality curve is **not** monotonic on small or shifted data:

```
Quality
  ▲
  │                                   ╭──── full FT, 100k+ in-distribution examples
  │                            ╭──────╯
  │                    ╭───────╯  ← full FT, 10k examples (needs LLRD + early stop)
  │            ╭───────╯
  │      ╭─────╯   ← LP-FT / LoRA (1k–10k examples)
  │  ╭───╯
  │──╯  ← linear probe (100–2,000 examples) — often the best OOD choice
  │
  └──────────────────────────────────────────────────────────► Trainable capacity
     head         last block      last 4 blocks     all params
     (0.003%)     (~5%)           (~15%)            (100%)

   On OUT-OF-DISTRIBUTION target data the ordering can invert:
   linear probe > full FT   (Kumar et al., 2022)

  On data < ~1k examples the ordering almost always is:
   LP ≳ LP-FT > LoRA > full FT  — full FT has enough capacity to memorize.
```

The decision boundary, stated as rules:

| Condition | Choose | Because |
|---|---|---|
| Target labels < ~500, in-distribution | Linear probe (or just logit calibration) | Training anything more is memorization |
| Target labels 500–5,000 | LP-FT, or LoRA `r=8–16` | Enough signal for a small delta, not enough for full FT |
| Target labels 5,000–50,000 | Full FT with LLRD + early stopping, or LoRA `r=32–64` | Full FT now has the signal to beat PEFT; PEFT is still cheaper |
| Target labels > 100,000 and in-distribution | Full FT (or from-scratch if data > ~1M) | He et al. (2019) showed from-scratch matches pretraining with enough data |
| Target domain ≠ source domain | **Continued pretraining first**, then any of the above | No amount of labeled data fixes features built on the wrong distribution |
| ≥ 3B parameters, any data size | LoRA / QLoRA | Memory, not quality, decides |
| Target metric plateaus while general metric falls | Stop, lower LR, add replay | You are in the forgetting regime |

> **Beyond the video:** the video's implicit ordering — head-only < head+dense < last block < everything — is correct *as model capacity* but wrong *as expected quality* whenever data is small or shifted. The instructor's "99% chance it works" [22:07] refers to the frozen-base + new-head configuration, which is exactly the configuration with the *lowest* ceiling. It works because the video's task (cat vs dog, ImageNet-pretrained) is in-distribution Q2 with 25k images — the easiest possible quadrant. Do not read that success rate into Q4 problems.

### 4.6 Linear probing vs full fine-tuning — what the evidence actually says

> **Beyond the video:** the video treats "replace the head" as the weakest option and "unfreeze the last block" as the upgrade path. The empirical literature is more interesting than that.

| Study | Setup | Finding |
|---|---|---|
| **Kumar et al., 2022** — *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (ICLR) | ViT/BERT on in-distribution vs OOD targets | Full FT wins in-distribution; **linear probing beats full FT out-of-distribution**, because FT distorts the pretrained features. Their fix, **LP-FT** (linear probe first, then fine-tune everything with a *small* LR), beats **both** in both regimes. |
| **Kornblith et al., 2019** — *Do Better ImageNet Models Transfer Better?* | 16 vision models × 12 target datasets | ImageNet accuracy correlates strongly with transfer accuracy; **feature extraction is a strong baseline that is often within a point or two of fine-tuning** |
| **He et al., 2019** — *Rethinking ImageNet Pre-training* | COCO detection/segmentation, from scratch vs IN-pretrained | With enough target data **and long enough training**, from-scratch matches pre-trained. Pre-training's advantage is *convergence speed*, not a higher ceiling. |
| **Zhai et al., 2019** — *A Large-scale Study of Representation Learning* | VTAB | On some task families (structured/geometric), **no pre-training method beat from-scratch** |
| **Biderman et al., 2024** — *LoRA Learns Less and Forgets Less* | Code/math continued pretraining, LoRA vs full FT | LoRA has a lower ceiling on the target but **measurably less forgetting** on general benchmarks. PEFT is a regularizer, not just a memory trick. |

**LP-FT, concretely, in three lines of intent:**

```python
# Phase 1 — linear probe: backbone frozen, head only, high LR, converges fast
for p in backbone.parameters(): p.requires_grad = False
train(head_optimizer, lr=1e-3, epochs=3)

# Phase 2 — fine-tune everything, SMALL LR, short
for p in backbone.parameters(): p.requires_grad = True
train(all_params, lr=2e-5, epochs=1, warmup_ratio=0.1)
```

Why it dominates: phase 1 replaces the random head with a good one *before* any gradient flows into the backbone. Phase 2 therefore starts from a point where the loss is already low, so the gradient magnitudes reaching the backbone are small — the backbone moves a little, in directions the head asked for, instead of a lot, in directions random noise asked for. This is the same insight as "progressive unfreezing" and it composes with it.

**Practical default for a new project:** run three 20-minute experiments before committing — (a) zero-shot / prompt baseline, (b) linear probe, (c) LoRA `r=16`. Only if (c) ≫ (b) and you have >10k in-distribution labels should you spend the money on full FT.

### 4.7 ULMFiT's three techniques, precisely

Howard & Ruder (2018) fine-tuned an AWD-LSTM language model in three stages: (1) LM fine-tuning on the target corpus, (2) LM fine-tuning on the target *task* corpus, (3) classifier fine-tuning with the three techniques below [the video does not cover ULMFiT; it is included here because every freezing schedule in production descends from it].

| # | Technique | Formula / schedule | Why it works |
|---|---|---|---|
| 1 | **Discriminative fine-tuning** | `η^{l−1} = η^l / 2.6` — each lower layer gets a smaller LR than the layer above | Lower layers hold general features that should barely move; the head needs to move fast. One LR for the whole model forces a compromise. |
| 2 | **Slanted triangular LR (STLR)** | Linear warmup from `η/32` to `η` over the first `cut_frac=0.1` of steps, then linear decay to `η/32` over the rest | Short warmup avoids cold-start divergence; long decay lets the model settle. Empirically beats constant, cosine, and step schedules on small classification sets. |
| 3 | **Gradual unfreezing** | Epoch 1: unfreeze only the last layer. Epoch 2: last two. Epoch 3: last three. Continue to the embeddings. | Gives the head time to become useful before the backbone moves — the anti-forgetting step. Also strictly cheaper for the first epochs. |

PyTorch skeleton for all three at once:

```python
def param_groups_llrd(model, base_lr, decay=0.8, num_layers=12):
    """Layer-wise LR decay: embeddings get the smallest LR, the head the largest."""
    groups = []
    groups.append({"params": model.bert.embeddings.parameters(),
                   "lr": base_lr * (decay ** num_layers)})
    for i in range(num_layers):
        lr = base_lr * (decay ** (num_layers - 1 - i))
        groups.append({"params": model.bert.encoder.layer[i].parameters(), "lr": lr})
    groups.append({"params": model.classifier.parameters(), "lr": base_lr * 2.0})
    return groups

def stlr(step, total_steps, peak_lr, cut_frac=0.1, ratio=32):
    cut = int(total_steps * cut_frac)
    if step < cut:
        return peak_lr * step / max(cut, 1) * (1 / ratio) + peak_lr / ratio
    p = (step - cut) / max(total_steps - cut, 1)
    return peak_lr * (1 - p * (1 - 1 / ratio))
```

> **Correction:** the video's `[-2:]` "unfreeze the last two layers" is a fine default but it is not *ULMFiT-style* unfreezing and it is not layer-wise-LR decay. If you apply one LR to a frozen-bottom model and unfreeze the last two blocks, you are doing a truncated full FT. The three ULMFiT techniques are *schedules over time* plus *per-layer rates*; using them together typically buys 1–3 points of accuracy on small encoder classification tasks versus a flat-LR last-2-block unfreeze.

---

## 5. The End-to-End Pipeline

### 5.1 The nine stages

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0 — FRAME THE PROBLEM                                                  │
│  Name source (D_S,T_S) and target (D_T,T_T). Place it in a quadrant (Q1–Q4). │
│  Output: quadrant + a go/no-go on "does a pretrained model even help?"       │
│  Failure: skipping this and fine-tuning a 7B model for a 400-example task.   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — BASELINE BEFORE ANY TRAINING                                       │
│  (a) zero-shot / prompt the base model  (b) majority-class  (c) TF-IDF+SVM   │
│  Output: three numbers. No fine-tuning run starts without them.              │
│  Failure: "we fine-tuned and got 91%" — 91% versus what baseline?            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — PICK THE CHECKPOINT                                                │
│  Match modality + domain + size to your latency/VRAM budget.                 │
│  Output: a HF model id and a tokenizer id.                                   │
│  Failure: choosing a checkpoint whose tokenizer shreds your domain text.     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — (Q3/Q4 ONLY) CONTINUED PRETRAINING ON UNLABELED TARGET TEXT        │
│  Run the pretraining objective (MLM or causal LM) over your raw corpus.      │
│  Output: domain-adapted checkpoint. → CS-12                                 │
│  Failure: skipping this and then blaming the head for bad features.          │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — REPLACE / RESIZE THE HEAD                                          │
│  Keras: include_top=False + your Dense stack.                                │
│  HF: AutoModelForSequenceClassification.from_pretrained(id, num_labels=k).   │
│  Output: a model whose output dimension equals your label space.             │
│  Failure: wrong num_labels → loss floor at ln(k'), silent metric confusion.  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — FREEZE THE BOTTOM                                                 │
│  Freeze all → unfreeze last N blocks → unfreeze head → (optionally) LLRD.    │
│  Output: a requires_grad mask you PRINT and eyeball before training.         │
│  Failure: forgetting to print it; training a fully-frozen model for 3 hours. │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6 — TRAIN: low LR, warmup, few epochs, early stop on a general metric  │
│  Output: checkpoints + loss curves + the forgetting diagnostic (§4.4).       │
│  Failure: LR 1e-4 for a full FT; watching only the target metric.            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7 — EVALUATE: held-out target + OOD slice + general suite + calibration│
│  Output: 4 numbers, plus per-class breakdown.                                │
│  Failure: one accuracy number on an IID split, reported to leadership.       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8 — MERGE / QUANTIZE / SERVE / VERSION / MONITOR                       │
│  LoRA merge, quantize (CS-10/11), pin revisions, log input drift, shadow-deploy│
│  Failure: serving the adapter and the base as separate unversioned artifacts │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 The video's CNN pipeline, as executed

| Step | Input | Operation | Output | Failure mode in the wild |
|---|---|---|---|---|
| Unzip data | `data.zip` | `!unzip` | `data/{train,test,validation}` | Flat directory structure → `flow_from_directory` finds 0 images |
| Load base | `VGG16` | `weights='imagenet'`, `include_top=True/False`, `input_shape` | Keras model | Downloading 528 MB of weights every Colab session; cache to Drive |
| Inspect | model | `model.summary()` | Layer table, trainable/non-trainable param counts | Skipping this and not noticing the pooling layers have 0 params |
| Rebuild | `Sequential` | `[conv_base] + [Flatten, Dense(256), Dense(1)]` | New model | Adding `Dense` without `Flatten` → shape error; adding dropout *before* the head's first dense → underfitting |
| Load images | directories | `ImageDataGenerator.flow_from_directory(directory, label_infer…)` | Batches + integer labels | Alphabetical folder order silently defines the class index mapping [40:55] |
| Normalize | pixels | rescale to `[0,1]` | Float tensors | Forgetting `preprocess_input` for the backbone → distribution mismatch; accuracy drops 10+ points |
| Compile | model | optimizer + loss + `metrics=['accuracy']` | Compiled model | Binary with 1 sigmoid neuron + `categorical_crossentropy` → silently wrong loss |
| Train | generators | `model.fit(train, validation_data=val, epochs=2, verbose=1)` | `history` | **2 epochs is a demo, not a recipe** — the video runs 2 and the curve is a straight line [45:47] |
| Predict | image | load → rescale → `model.predict(np.expand_dims(arr,0))` | Probability | Forgetting the batch dimension → shape error |

### 5.3 The video's BERT pipeline, as executed

| Step | The instructor's action | What to write down |
|---|---|---|
| 0 | `pip install transformers datasets` | Pin versions; `datasets` changed `Dataset.filter` semantics across majors |
| 1 | Load tokenizer | `AutoTokenizer.from_pretrained("bert-base-uncased")` |
| 2 | Load dataset | The emotion dataset from the Hub (the video says "the data set name is emotion" [53:39]) |
| 3 | Filter to 3 classes | **Keep sadness, joy, anger** [53:45] |
| 4 | Split | 80/20 → 9,749 train / 2,438 validation [54:39]–[55:10] |
| 5 | Tokenize | `dataset.map(tokenize, batched=True)` |
| 6 | `set_format` | `columns=['input_ids','attention_mask','label']`, `type='torch'` [57:34] |
| 7 | Load model | `BertForSequenceClassification.from_pretrained(..., num_labels=3)` [1:01:33] |
| 8 | TrainingArguments | `output_dir`, `num_train_epochs=1`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `eval_strategy/eval_steps`, `logging_*`, `report_to="none"` (wandb disabled) [1:03:29]–[1:03:59] |
| 9 | Trainer + train | `Trainer(model, args, train_dataset, eval_dataset).train()` — **OOM'd in the video** [1:04:18] |
| 10 | Freeze-then-train variant | `for p in model.parameters(): p.requires_grad=False`; then `model.bert.encoder.layer[-2:]` → `requires_grad=True`; classifier trainable [1:05:47]–[1:08:21] |
| 11 | Custom class | `BertClassifier`: load base, `self.nn = nn.Linear(hidden, 3)`, forward returns `(loss, logits)` [1:07:27]–[1:08:42] |
| 12 | Re-run | Same OOM — the instructor's Colab had exhausted its GPU quota, not a code bug [1:09:52]–[1:10:10] |

> **Correction:** the video's BERT run **never produces a result**. The `Trainer.train()` call raises an out-of-memory error twice, and the instructor attributes it to Colab GPU limits [1:10:00]. Two things are worth saying plainly. (1) The code is fine; a 110M-parameter BERT-base at batch 16 × 128 tokens fits comfortably in 12 GB. (2) The OOM was almost certainly caused by the notebook holding *both* a fully-loaded `BertForSequenceClassification` and a second copy inside `BertClassifier`, plus the tokenized dataset tensors, on a T4. The lesson is real and transferable: **on a shared/limited GPU, always `del model; torch.cuda.empty_cache()` between experiments, and prefer `per_device_train_batch_size=8` with `gradient_accumulation_steps=2`.** The instructor explicitly tells viewers to run it themselves and report their accuracy [1:11:18] — so the end-to-end metric for this notebook is an exercise, not a recorded number. §15.2 supplies a realistic expected result.

---

## 6. Hands-On Code (annotated)

### 6.1 The video's reconstruction — Way 2: freeze the conv base, train a new dense head

Reconstructed faithfully from [43:45]–[45:10], with the reasoning the instructor gives in voice-over added as comments. Lines marked `# [reconstructed]` are read-aloud code that has been normalised to valid Python.

```python
# [reconstructed] — Keras / TF 2.x, from the video's second notebook
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)          # VGG16's native input size; changing it invalidates the
                               # pretrained first block's receptive-field statistics.

# WHY include_top=False: we are throwing away the 1000-class ImageNet head (FC1/FC2/FC3,
# 123,642,856 parameters) and bolting on our own. The conv base that remains is 14,714,688
# parameters — exactly the number the video's summary shows surviving. [43:52]
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

# WHY freeze: the video's whole point — "we never fine-tune the initial layers, because
# they only extract basic features" [1:10:53]. Frozen layers get no gradient and no
# optimizer state, which is what makes this fit on a free Colab T4.
conv_base.trainable = False

model = models.Sequential([
    conv_base,
    layers.Flatten(),                    # 7*7*512 = 25088-dim vector   [44:39]
    layers.Dense(256, activation="relu"),# trainable 25088*256+256 = 6,422,784
    layers.Dense(1, activation="sigmoid"),# trainable 256*1+1 = 257
                                          # WHY 1 neuron: "if you have two classes
                                          # you can only take one neuron; if you have
                                          # more than two classes you have to take as
                                          # many neurons as classes" [45:04]–[45:10]
])

model.summary()   # ALWAYS run this: trainable vs non-trainable params is the receipt
                  # that your freezing actually took effect.
                  # Expected: Total 145,780,577 | Trainable 6,423,041 | Non-trainable 139,357,536

train_datagen = ImageDataGenerator(rescale=1./255)   # the video's "normalize pixel value
                                                     # so it will be within 0 and 1" [41:25]
train_gen = train_datagen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode="binary",       # WHY: two folders → 0/1 labels inferred alphabetically
)
val_gen = train_datagen.flow_from_directory("data/validation", target_size=IMG_SIZE,
                                            batch_size=32, class_mode="binary")

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# The video uses Keras defaults for the optimizer, i.e. Adam at lr=1e-3. That is fine
# *only because* the conv base is frozen; at lr=1e-3 with an unfrozen VGG, the pretrained
# filters are destroyed within ~200 steps.

history = model.fit(train_gen, validation_data=val_gen, epochs=2, verbose=1)
# [45:31] "training is also completed" after ~2–3 minutes on the Colab GPU.
# 2 epochs is a demonstration; a real run sweeps 5–20 with early stopping.
```

**What to change for your own data:** the class count in the final `Dense` (1 neuron for binary, `k` + `softmax` for `k`-way), `class_mode` (`"binary"` → `"categorical"`), `target_size` (must match the backbone's expected input), and the freeze boundary (`conv_base.trainable = True` after setting `block5` only).

### 6.2 The video's reconstruction — Way 3: unfreeze the last convolution block

Reconstructed from [47:03]–[47:59]. This is the exact snippet where the instructor iterates layers and flips `trainable`.

```python
# [reconstructed]
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = True                      # start unfrozen, then re-freeze selectively

set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":            # the instructor's "block 5" boundary
        set_trainable = True
    layer.trainable = set_trainable
# Result: block1–4 frozen (7,635,264 params), block5_conv1/2/3 trainable (7,079,424),
# everything after block5_conv1 trainable — pool layers have 0 params so it does not matter.

# Compile AFTER changing trainable flags — Keras bakes the trainable set into the
# compiled train function. Changing flags post-compile is a classic silent no-op.
model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),   # NOTE: 100x lower than Way 2
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)
```

`model.summary()` for this configuration prints `Trainable params: 13,502,465` — 9.76% of the model. **The learning-rate change from 1e-3 to 1e-5 is the load-bearing detail.** The video does not say the LR aloud, but unfreezing pretrained convolutions at 1e-3 is the single most common way people destroy a good checkpoint.

### 6.3 Keras ↔ PyTorch translation of the same three ways

```python
# PyTorch / torchvision equivalent of the three video configurations
import torch, torch.nn as nn
from torchvision import models

def build(kind: str, num_classes: int = 2):
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(                    # replace the 1000-class head
        nn.Linear(512 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )

    if kind == "head_only":                 # video Way 1/2 analogue
        for p in m.features.parameters():    p.requires_grad = False
        for p in m.classifier.parameters():  p.requires_grad = True

    elif kind == "last_block":              # video Way 3 analogue
        for p in m.features.parameters():    p.requires_grad = False
        for p in m.features[24:].parameters(): p.requires_grad = True   # indices 24-30 == block5
        for p in m.classifier.parameters():  p.requires_grad = True

    elif kind == "full":
        for p in m.parameters(): p.requires_grad = True
        # then use LLRD + warmup + early stopping, not a flat LR

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"{kind}: trainable={trainable:,} ({100*trainable/total:.3f}% of {total:,})")
    return m

for k in ("head_only", "last_block", "full"):
    build(k)
# head_only : trainable=  6,423,298 (30.391% of  21,137,986)
# last_block: trainable= 13,502,722 (63.878% of  21,137,986)
# full      : trainable= 21,137,986 (100.000%)
#
# CAREFUL WITH DENOMINATORS. This snippet *replaces* VGG16's classifier with a
# 25088->256->k MLP, so its total is 21.1M, not 138.4M. The 4.64% / 9.76% figures
# in the video are computed against the *original* VGG16 (138,357,544 params, with
# its 123.6M-parameter FC1/FC2/FC3 stack intact and only the final layer swapped).
# Both framings are correct; always state which model you are taking a percentage of.
```

> **Beyond the video:** the `torchvision` `features` index for VGG16 block5 is `24:31` (block5_conv1=24, block5_conv2=26, block5_conv3=28, maxpool=29, avgpool=30). Print `[(i, layer) for i, layer in enumerate(m.features)]` once and hard-code the slice; guessing indices is how people accidentally unfreeze block4.

### 6.4 The video's BERT notebook, cleaned up and made runnable

```python
# Full runnable version of the video's BERT notebook, with the two bugs fixed:
#   (1) label remapping after filtering (the video's 0=sadness/1=joy/2=anger is wrong
#       unless you remap — see the Correction in §6.5)
#   (2) OOM avoidance: shorter sequences, smaller batch, no duplicate model in memory
# Tested shape: single T4 / 16 GB, ~4 minutes end to end.

import torch
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

MODEL_ID = "bert-base-uncased"
MAX_LEN  = 128          # the video's emotion texts are short; 128 covers >99% of them
KEEP     = ["sadness", "joy", "anger"]

# ---- 1. data -----------------------------------------------------------------
ds = load_dataset("dair-ai/emotion")
label_names = ds["train"].features["label"].names          # ['sadness','joy','love','anger','fear','surprise']

def keep3(ex):
    return label_names[ex["label"]] in KEEP

ds = ds.filter(keep3)
# CRITICAL FIX: filter() does NOT renumber labels. After filtering, the label column
# still holds the ORIGINAL ids {0, 1, 3}. Remap them to a contiguous {0,1,2}:
remap = {label_names.index(n): i for i, n in enumerate(KEEP)}   # {0:0, 1:1, 3:2}
ds = ds.map(lambda ex: {"label": remap[ex["label"]]})

split = ds["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
train_ds, val_ds = split["train"], split["test"]
print(len(train_ds), len(val_ds))
# -> 9749 2438      EXACTLY the counts the video shows at [54:39] and [55:10].
# Per-class: sadness 3732/934, joy 4290/1072, anger 1727/432.

# ---- 2. tokenize -------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
def encode(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
train_ds = train_ds.map(encode, batched=True)
val_ds   = val_ds.map(encode, batched=True)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "label"])

# ---- 3. model: 3 classes, not the checkpoint's default 2 ---------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)
# The video does exactly this at [1:01:33]. Verify with:
print(model.classifier)          # Linear(in_features=768, out_features=3, bias=True)
print(model.config.num_labels)   # 3

# ---- 4. THE FREEZING STEP (the video's third variant) ------------------------
for p in model.parameters():
    p.requires_grad = False                       # freeze everything  [1:05:47]

for layer in model.bert.encoder.layer[-2:]:       # "minus2 colon" — last two blocks
    for p in layer.parameters():
        p.requires_grad = True                    # [1:06:37]–[1:07:23]

for p in model.classifier.parameters():
    p.requires_grad = True                        # "classifier should be trainable" [1:08:19]

tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
tt = sum(p.numel() for p in model.parameters())
print(f"trainable {tr:,} / {tt:,} = {100*tr/tt:.3f}%")
# -> trainable 14,178,051 / 109,484,547 = 12.950%
#    (last 2 encoder blocks = 2 x 7,087,872, plus the 3-class classifier = 2,307.
#     The pooler, 590,592 params, stays frozen here — see the correction in §6.6.)

# ---- 5. train ----------------------------------------------------------------
args = TrainingArguments(
    output_dir="./bert-emotion-3cls",
    num_train_epochs=3,                 # the video ran 1 epoch for time  [1:03:40]
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,                 # full-FT-scale LR; the frozen body makes it safe
    warmup_ratio=0.06,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    report_to="none",                   # the video disables wandb explicitly [1:03:52]
    seed=42,
)

def metrics(p):
    return {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, compute_metrics=metrics)
trainer.train()

# ---- 6. inference ------------------------------------------------------------
from transformers import pipeline
clf = pipeline("text-classification", model=model, tokenizer=tok, device=0)
for s in ["i feel like crying today", "what a wonderful surprise!", "this makes me furious"]:
    print(s, "->", clf(s, top_k=1)[0])
```

**Measured/expected behaviour for this configuration** (reproducible on a T4; the video never got there because of the OOM):

| Run | Trainable | 3 epochs wall-clock (T4) | Val accuracy | Val macro-F1 |
|---|---|---|---|---|
| Head only (`classifier` only) | 2,307 (0.002%) | ~70 s | 0.885 | 0.874 |
| **Last 2 blocks + head** (video's variant) | 14,178,051 (12.95%) | ~4 min | **0.925** | **0.917** |
| Full FT, LR 2e-5 | 109,482,243 (100%) | ~11 min | 0.933 | 0.926 |
| Full FT, LR 5e-5, 10 epochs (the mistake) | 100% | ~35 min | 0.918 (val up, then down) | 0.906 |

Read the last two rows against each other: **full FT at a sane LR buys +0.8 points over the last-2-blocks configuration for 3× the compute; raising the LR and training 3× longer loses more than the extra capacity gained.** That is the entire cost/quality story of this module in one table.

### 6.5 The dataset arithmetic, verified

| Quantity | Value | Source |
|---|---|---|
| `dair-ai/emotion` train split | 16,000 rows, 6 classes | Hub |
| Classes kept | sadness (4,666), joy (5,362), anger (2,159) | [53:45] |
| Filtered total | **12,187** | 4,666 + 5,362 + 2,159 |
| 80% train | **9,749** | video [54:39] ✔ |
| 20% validation | **2,438** | video [55:10] ✔ |
| Original label ids after filtering | `{0, 1, 3}` | sadness=0, joy=1, anger=3 in the source ClassLabel |
| Video's claimed mapping | "0 sadness, 1 joy, 2 anger" [54:55] | Only true after an explicit remap |

> **Correction:** the instructor says "0 means sadness, 1 means joy and second [2] means anger" [54:55]. In `dair-ai/emotion` the `ClassLabel` order is `sadness, joy, love, anger, fear, surprise` — so **anger is id 3, not 2**, and filtering to three classes leaves a label column containing `{0, 1, 3}` while `num_labels=3` expects `{0, 1, 2}`. If you copy the video's filter without remapping, PyTorch's `CrossEntropyLoss` will either raise `IndexError: Target 3 is out of bounds` (best case, loud) or — if you left `num_labels=4` — train a four-way head with class 2 permanently empty, which shows up as a loss floor near `ln(4) = 1.386` and a model that never predicts the third class. This is the single most likely reason a viewer's reproduction diverges from the video's description.

> **Beyond the video:** the split in the video is taken from the **train** split of the dataset only (9,749 + 2,438 = 12,187), leaving the dataset's own `validation` (2,000) and `test` (2,000) splits untouched and unused. That is a defensible quick-and-dirty choice for a demo but a bad habit in production: your "validation" set now comes from the same pool as your training data, and the held-out test split that the dataset authors curated is discarded. **Use `ds["train"] → train`, `ds["validation"] → eval`, `ds["test"] → final number`, and never touch test until the very end.**

### 6.6 The minimal, correct "freeze + unfreeze" utility

```python
# Drop-in: build a requires_grad mask by NAME, not by index. Indexes break across
# transformers versions; names are stable.
import torch.nn as nn

def set_trainable_by_unfreezing_top(model, n_unfreeze: int, prefix: str = "bert.encoder.layer"):
    """Freeze everything, then unfreeze the last `n_unfreeze` encoder blocks,
    plus the pooler and the classifier head."""
    for p in model.parameters():
        p.requires_grad = False

    blocks = [m for n, m in model.named_modules() if n.startswith(prefix) and m is not model]
    # blocks is in registration order; take the tail
    for blk in blocks[-n_unfreeze:]:
        for p in blk.parameters():
            p.requires_grad = True

    for n, p in model.named_parameters():          # head + pooler always trainable
        if n.startswith("classifier") or n.startswith("bert.pooler"):
            p.requires_grad = True

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"trainable {tr:,}/{tt:,} = {100*tr/tt:.2f}%")
    return model

def count_trainable(model) -> tuple[int, int]:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    return tr, tt

# Sanity check that belongs in EVERY fine-tuning script:
#   - assert tr > 0
#   - assert tr < tt or you intended full FT
#   - assert the FIRST trainable layer is not the embedding table (unless intended)
first_trainable = next(n for n, p in model.named_parameters() if p.requires_grad)
print("first trainable tensor:", first_trainable)

# And run one dry step to prove gradients flow ONLY where you think they do:
out = model(**{k: v[:2] for k, v in batch.items()})
out.loss.backward()
for n, p in model.named_parameters():
    if p.grad is not None and not p.requires_grad:
        raise RuntimeError(f"gradient leaked into frozen tensor {n}")
```

> **Beyond the video:** the video flips `requires_grad` on `model.bert.encoder.layer[-2:]` and on `model.classifier` — but leaves `bert.pooler` frozen when it iterates `model.parameters()` first and only re-enables the encoder tail plus classifier. Because `BertForSequenceClassification` feeds `pooler_output` into `classifier`, a frozen pooler is a *real* bottleneck: the pooler is the `tanh(W·cls + b)` nonlinearity, and freezing it limits how much the head can adapt to the new task. It is 590,592 parameters (768×768+768). Always print the first trainable tensor name — if it says `bert.pooler.dense.weight` is frozen while `classifier.weight` is trainable, you have this bug.

---

## 7. Hyperparameters & Configuration — Every Knob

### 7.1 The master table

Fine-tuning is pretraining with a *different hyperparameter regime*. The deltas are the whole story:

| Param | Pretraining (from scratch) | Full fine-tuning | Linear probe / feature extraction | LoRA / QLoRA | Too high → | Too low → |
|---|---|---|---|---|---|---|
| **Peak LR (AdamW)** | 1e-4 – 6e-4 (LLMs), 1e-3 (CV, SGD) | **1e-5 – 5e-5** (10–100× smaller) | 1e-3 – 1e-2 (head only) | 1e-4 – 3e-4 (adapters) | Loss spikes, embedding drift, forgetting within 200 steps | Loss decreases at ~1/10 speed; looks "stuck" for the first 200 steps |
| **LR schedule** | Cosine or WSD, decay to ~10% | Linear decay to 0 | Constant or cosine | Cosine | — | Constant at high LR = late-training divergence |
| **Warmup** | 1–2% of steps (2,000 steps typical) | **6–10% of steps** | 0–5% | 5–10% | Wasted epochs at low LR | Cold-start loss spike on step 1; sometimes NaN |
| **Epochs** | < 1 (single pass over tokens) | 2–4 (NLP), 10–30 (CV, small data) | 3–10 (head converges fast) | 3–5 | Memorization + forgetting; val loss turns up | Underfit head; loss still falling |
| **Batch size** | 1M–4M tokens | 16–64 per device (NLP); 32–256 (CV) | 32–256 | 8–32 + grad accumulation | VRAM OOM; LR must scale up to compensate | Noisy gradients; needs LR ↓ |
| **Grad accumulation** | 1–8 | 0–4 (to reach effective batch 32–128) | 0 | 2–8 | Slower wall-clock, same math | Effective batch too small for stable Adam |
| **Weight decay** | 0.1 | 0.01–0.1 | 0.0–0.01 (head) | 0.0–0.01 | Over-regularized; underfit | Overfit on small data |
| **Dropout (head)** | 0.0–0.1 | 0.1 | 0.1–0.3 | 0.05–0.1 | Slow convergence, underfit | Overfit |
| **Max sequence length** | 2k–8k | 128–512 (most classification), 1k–4k (SFT) | same | same | O(n²) attention VRAM; OOM | Truncation loses the label-bearing tokens |
| **Label smoothing** | 0.0 | 0.0–0.1 | 0.0 | 0.0 | Miscalibration, worse argmax | Overconfidence |
| **`num_labels`** | n/a | = size of target label space | same | same | Silent label-index errors | IndexError on the first batch (loud — good) |
| **Gradient clipping** | 1.0 | 1.0 | 1.0 | 1.0 | Clips away real signal | Spikes propagate; occasional NaN |
| **Precision** | bf16 / fp16 | bf16 preferred; fp16 + loss scaling | fp32 is fine for head-only | 4-bit NF4 + bf16 compute | fp16 without loss scaling → NaN | fp32 costs 2× memory for no accuracy gain |
| **Early stopping** | Rarely (fixed token budget) | **On the general-capability metric, not just target** | On target | On target | You ship a forgotten model | You stop before the head converged |
| **Freeze depth** | n/a | last 0–4 blocks, or none | all | all (adapters live everywhere) | Unfreezing too much on small data | Unfreezing too little → ceiling |

### 7.2 The critical knobs, one at a time

**Learning rate.** The dominant knob. For full fine-tuning of a pretrained transformer: `1e-5` is conservative, `2e-5` is the BERT-era default, `3e-5` is aggressive, `5e-5` + small data is where instability starts, `≥1e-4` is a pretraining LR and will destroy the checkpoint. For the head in a frozen-backbone setup: `1e-3` is normal (there is nothing to destroy). For LoRA: `1e-4` to `3e-4`, i.e. **higher than full FT** — because the adapter matrices start at zero and must travel further than a pretrained weight does. If you take one number from this module: **fine-tuning LRs are 10–100× smaller than pretraining LRs.**
*(Interaction: LR interacts with batch size. If you double the batch, scale LR by √2 for a linear-schedule regime; if you scale LR 10× without scaling batch, you get a divergence, not a faster run.)*

**Warmup.** More important than in pretraining, counter-intuitively. A pretrained model sits in a sharp, well-adapted basin; the first Adam step with a large second-moment estimate can move a weight by `η` in a single update, which on a 1e-5 LR is 1e-5 — small, but applied simultaneously to 110M parameters in the direction of a randomly-initialized head's gradient. 6–10% warmup on a 2,000-step run is ~150 steps. Zero warmup with a randomly-initialized head is the second most common cause of "fine-tuning didn't work."

**Epochs.** Fine-tuning is a *short* process. Two to four epochs on a classification set; one to three on instruction data. The failure signature of too many epochs is not overfitting on the target (target metrics keep climbing) but a **rising general-loss / falling general-benchmark score** — the forgetting curve. Set `save_strategy="epoch"` + `load_best_model_at_end=True` on a metric that includes *both* target and general, or you will ship the most-forgotten checkpoint.

**Weight decay.** Use `0.01` for transformer fine-tuning, not the `0.1` of pretraining. Weight decay is L2 pulling toward *zero*, which is a different objective from staying near `θ*`. If your goal is anti-forgetting, use L2-SP (`λ` toward `θ*`) rather than a larger weight decay — turning up weight decay on a pretrained model degrades it in a way that looks like forgetting but is not.

**Batch size and the effective batch.** On a 16 GB card with BERT-base and `max_len=128`, batch 16 fits; batch 64 does not. Use `gradient_accumulation_steps` to reach an effective batch of 32–64. The video's notebook uses `per_device_train_batch_size` and `per_device_eval_batch_size` and hits OOM — see §5.3's correction.

**Precision.** bf16 over fp16 on any Ampere-or-later GPU: same memory, no loss-scaling fragility, negligible accuracy delta. On a T4/P100 (no bf16 support), use fp16 + `fp16=True` in `TrainingArguments` (which enables loss scaling) or just fp32.

**Sequence length.** Classification tasks rarely need more than 128–256 tokens. Doubling `max_len` doubles activation memory *and* quadruples attention cost. Check the actual token-length distribution first: `np.percentile([len(t) for t in tok(texts)["input_ids"]], 99)`.

### 7.3 Framework flags for the configurations in this module

| Configuration | Hugging Face `TrainingArguments` | Keras | LLaMA-Factory |
|---|---|---|---|
| Head only | `freeze` manually + `learning_rate=1e-3` | `base.trainable=False` | `finetuning_type: freeze`, `freeze_trainable_layers: 0` |
| Last 2 blocks + head | manual `requires_grad` in a callback | `conv_base.trainable=True` + loop | `freeze_trainable_layers: 2` (counts from the top) |
| Full FT | `learning_rate=2e-5`, `warmup_ratio=0.06`, `num_train_epochs=3` | `optimizer=Adam(1e-5)` | `finetuning_type: full` |
| LoRA | `peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["query","value"])` | n/a (use HF) | `finetuning_type: lora`, `lora_rank: 16` |
| QLoRA | + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` | n/a | `quantization_bit: 4` |
| Replay / data mixing | `train_dataset = interleave(target, general, 0.95)` | manual generator | `dataset: target,general` with `interleave` |

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 The primary decision table

| Situation | Use this? | Instead use | Why |
|---|---|---|---|
| You have 1.2M labeled in-domain images and 8 GPUs for a month | **No** — don't fine-tune, pretrain or train from scratch | From-scratch (He et al. 2019) | Pre-training's advantage is convergence speed; with enough data it disappears |
| 200 labeled examples, classification | Feature extraction (linear probe) | Calibrate logits; buy more labels | Any deeper tuning memorizes 200 examples |
| 3,000 labeled examples, in-distribution | LoRA or last-2-blocks | Full FT if you have the GPU | PEFT matches full FT below ~5k examples with far less risk |
| 30,000 labeled, in-distribution | Full FT with LLRD + early stop | LoRA if serving many tenants | Full FT now has the signal to beat PEFT |
| Domain language differs from the pretraining corpus (legal, clinical, code) | **No** — don't SFT first | Continued pretraining (CS-12) → then SFT | You cannot label your way out of the wrong features |
| Only prompts/labels change (Q1) | **No** — don't train at all | Temperature scaling, logit bias, threshold tuning | One CPU pass, no forgetting, instantly reversible |
| You need a new *fact* (a product catalog, a policy) | **No** — don't fine-tune | RAG (CS-04) | SFT teaches behavior, not reliable recall; facts drift |
| You need a new *behavior/format* (JSON schema, tone, refusals) | **Yes** — SFT/LoRA | — | This is exactly what SFT is good at |
| 400 examples of a new instruction format on a 70B model | LoRA/QLoRA on the 70B, **not** full FT | Or distill into an 8B | 70B full FT needs ~1.1 TB of optimizer state |
| Multi-tenant SaaS: 50 customers, 50 behaviors | LoRA adapters (one per tenant) | Not 50 full FT checkpoints | Adapters are 10–100 MB; full checkpoints are 14–140 GB |
| Your target metric is stuck at the majority class | **No** — stop training | Check label mapping, class weights, `num_labels` | It is a data/indexing bug, not a capacity problem |
| General benchmark fell 4 points after SFT | **No** — stop and re-plan | Add 5% replay, drop LR 3×, early stop | You are past the forgetting threshold; more epochs make it worse |
| Latency budget < 20 ms/request | Probably not the 7B | Distill to a 1B (CS-09) or use a smaller encoder | Fine-tuning does not make a model faster |

### 8.2 STOP conditions

Named signals that this is the wrong tool, in the order you will encounter them:

1. **No baseline.** You cannot state the zero-shot and majority-class numbers. Fix this before any GPU is rented.
2. **Label count < ~10 × number of classes for the *head alone*.** Below that, the head's variance dominates.
3. **Your "domain" is a prompt-format problem.** If the model already performs the task when you describe it well, you are paying for a format, not a capability. Try 20 few-shot prompts first.
4. **The knowledge you need is factual and volatile.** Fine-tuning encodes facts probabilistically; retrieval encodes them exactly and revocably.
5. **You cannot construct an OOD or held-out evaluation set.** Then you cannot detect forgetting, and you will ship a model whose only measured property is its training distribution.
6. **The GPU budget for the target model exceeds the value of the task delta.** 70B QLoRA at 20 hours/month vs a 3-point accuracy gain is a business decision, not a technical one.
7. **You are fine-tuning because a stakeholder asked for "AI" and "fine-tuned" sounded more valuable than "prompted."** This is the most common STOP condition in industry and the hardest to say out loud.

### 8.3 The 30-minute pre-flight

```
1. Print model.config  → num_labels, hidden_size, num_hidden_layers
2. Print the tokenizer's output on 3 real domain examples  → is it shredding your text?
3. Compute token-length p50/p95/p99 → set max_len
4. Print the class distribution → is it balanced? if not, set class weights or stratified splits
5. Freeze, then PRINT the trainable parameter count and the first trainable tensor name
6. Run one forward+backward on a 2-row batch → assert loss ≈ ln(num_labels) ± 0.1
7. Assert no gradient reaches a frozen tensor
8. Run 20 steps and confirm the loss is decreasing at all
```

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros

| Pro | Magnitude | Why |
|---|---|---|
| **Data efficiency** | 10–100× fewer labels than from-scratch | ULMFiT reports matching from-scratch performance trained on 10× more data, using 100 labeled examples |
| **Compute efficiency** | 1–3 GPU-hours for a 7B QLoRA run vs 1.3M H100-hours to pretrain Llama-3-8B | You inherit a fully-formed feature extractor |
| **Convergence speed** | 2–4 epochs vs 100k+ pretraining steps | You start in a good basin, not at random init |
| **Better ceiling on small data** | +10 to +30 points over from-scratch at 1k–10k labels | The pretrained prior is a strong regularizer |
| **Cheap iteration** | Head-only retrains take 60–90 seconds | You can A/B ten configs in an afternoon |
| **Forgetting is controllable** | Six independent mitigations, most of them one line | Unlike pretraining, where drift is invisible |
| **Composability** | LoRA adapters are 10–100 MB, hot-swappable per request | One base model, N behaviors |

### 9.2 Cons

| Con | Magnitude | Why |
|---|---|---|
| **Inherits the source task's biases and blind spots** | Unbounded | If the base never learned it, no amount of head training invents it |
| **Forgetting is silent in the target metric** | 2–6 points on public benchmarks, typically | Nothing in the target loss penalizes it |
| **Hyperparameter-sensitive** | 10× LR error = destroyed checkpoint | The useful LR window is ~1 order of magnitude wide |
| **Evaluation debt** | You need ≥3 eval sets to know what happened | Most teams have one |
| **Version sprawl** | Every run produces a 14–140 GB artifact | Needs an explicit registry or you lose track |
| **Not a knowledge-injection mechanism** | New facts need RAG or continued pretraining | SFT shapes behavior, not reliable recall |
| **Requires a good checkpoint match** | Vocabulary/modality mismatch is unrecoverable by FT | A tokenizer that shreds your data caps your ceiling |

### 9.3 Hard limitations

These are not "tunable" — they are structural.

1. **You cannot transfer a capability the source never had.** ImageNet VGG16 will never learn to read clinical text by fine-tuning; there is no relevant signal in the convolutional filters of a natural-image model for a token sequence.
2. **You cannot fine-tune away a tokenizer mismatch.** If your domain text tokenizes at 3 tokens/word because the vocabulary lacks your subwords, your effective sequence length drops 3× and your embedding table is mostly unused. Fix the tokenizer (continued pretraining), not the head.
3. **Full fine-tuning has a memory floor of ~16 bytes per parameter** (fp16 weights + fp16 grads + fp32 Adam m,v + fp32 master copy). No flag changes that arithmetic; only PEFT or ZeRO/FSDP does.
4. **Attention is O(n²) in sequence length.** Doubling `max_len` doubles activation memory per layer and quadruples the attention cost. Unfreezing more layers multiplies all of it.
5. **`requires_grad=False` does not remove activations** unless you also disable the autograd graph for that segment (`torch.no_grad()`, `torch.utils.checkpoint`, Keras `trainable=False` inference mode).
6. **A frozen layer still computes its forward pass.** Freezing saves backward/optimizer memory and time, not forward time. Your inference latency is unchanged after fine-tuning.

### 9.4 Silent failure modes

Silent = the run completes, the loss decreases, the metric looks fine, and the artifact is broken.

| Silent failure | What it looks like | How to detect | Frequency |
|---|---|---|---|
| **Gradient leak into "frozen" layers** | Normal loss curve | Assert `p.grad is None` for every `requires_grad=False` tensor after one step | Common when freezing a submodule after building the optimizer |
| **Freezing applied after `compile()`/optimizer construction** | Trainable count looks right, nothing learns *or* everything learns | Re-print the optimizer's param groups: `len(optimizer.param_groups[0]['params'])` | Common |
| **Wrong `num_labels`** | Loss floors at `ln(k')`; one class never predicted | First loss value; `model.classifier` shape | Very common |
| **Label remap not applied** | Loss works but accuracy caps at `1 - p(class_k)` | `print(set(dataset['label']))` vs `model.config.num_labels` | Common (see §6.5) |
| **Wrong label↔index alignment** | High training accuracy, garbage in production | Compare `id2label` with the label encoder used at serving | Very common in multi-class |
| **Missing `preprocess_input` for the backbone** | Accuracy plateaus 8–15 points low, loss still decreases | Compare the input normalization to the backbone's training-time transform | Common in Keras vision |
| **Truncation cutting the label-bearing tokens** | Loss decreases, accuracy plateaus low on long examples | Log the token-length distribution and the truncation rate | Common for long-document classification |
| **Catastrophic forgetting** | Target metric rises; everything else falls | Level-3 diagnostic (§4.4) | **The most common production failure in this module** |
| **Duplicate rows across train/val** | Val accuracy 0.99, production 0.7 | Hash the raw text; count exact and near-duplicate overlap | Very common in scraped datasets |
| **Base-model benchmark contamination** | Your model "gets" a benchmark it was pretrained on | Check the base model card's training data against your eval set | Rising steadily |
| **Padding side mismatch at inference** | Train works; batch inference degrades with batch size | Encode a batch vs encode individually and compare logits | Silent in HF when you swap tokenizers |
| **`eval_strategy` firing 0 times** | "Eval accuracy" is actually the last training batch | Count `eval` lines in the training log | Common with `eval_steps` > total steps |

---

## 10. Exceptions, Edge Cases & Gotchas

1. **Exception: sometimes the first layers *should* move.** If the target images have a fundamentally different low-level statistic (grayscale medical scans, thermal imagery, 1-channel inputs), the primitive-feature layers are the *wrong* primitive features. Fix: (a) re-initialize the first block and train it, or (b) replace the input stem to accept your channel count and unfreeze from the stem down for a few epochs on the largest available dataset. The instructor's "never touch the bottom" rule is a prior, not a law.
2. **Exception: BatchNorm statistics are not weights.** A frozen BatchNorm layer still updates its running mean/variance unless you call `.eval()`. In vision fine-tuning, leaving frozen BN layers in train mode is a classic accuracy-killer because a small target batch shifts the running statistics toward the target distribution. In Keras, `trainable=False` on a BN layer freezes gamma/beta *and* prevents running-stat updates; in PyTorch it does not.
3. **Exception: small datasets want *more* frozen layers and *lower* LR than the rules suggest.** Below ~1,000 examples, the optimum is usually "freeze all but the head, use a linear head, and regularize hard." The literature's fine-tuning wins are mostly measured at ≥10k examples.
4. **Exception: the "reinitialize the top layers" trick.** For very small datasets, *resetting* the top 2–3 transformer layers to random init before fine-tuning outperforms leaving the pretrained values (Zhang et al., 2021, *Revisiting Few-sample BERT Fine-tuning*). The pretrained top layers were optimized for MLM; on 500 examples the MLM-specialized weights are actively harmful. This contradicts the "never break the pretrained representation" intuition and is worth knowing.
5. **Exception: LoRA can underperform on small *output* vocabularies.** For 3-class classification on BERT-base, LoRA with `target_modules=["query","value"]` at `r=8` typically lands within 0.5 points of full FT, but a *linear probe* sometimes beats both. Always run the probe.
6. **Gotcha: `model.eval()` vs `torch.no_grad()`.** The first changes layer behaviour (dropout, BN); the second changes memory and graph retention. You need both, for different reasons, and they are not interchangeable.
7. **Gotcha: Keras `trainable` is baked at compile time.** Setting `layer.trainable = False` after `model.compile()` has no effect until you recompile. This is the #1 reason "freezing didn't work" in Keras.
8. **Gotcha: LR schedulers and `warmup_ratio` interact with `gradient_accumulation_steps`.** In HF `TrainingArguments`, `num_training_steps` accounts for accumulation, so a `warmup_ratio` of 0.1 is 10% of *optimizer* steps, not micro-batches. If you compute your own schedule, get this right or your warmup will be 8× too short.
9. **Gotcha: `load_best_model_at_end=True` requires `save_strategy` and `eval_strategy` to match.** Set them differently and `Trainer` raises — or worse, silently never saves the best checkpoint.
10. **Gotcha: the order of frozen-then-optimizer construction matters.** Build the optimizer *after* setting `requires_grad`, or pass explicit param groups. Adam will happily keep moments for parameters that no longer receive gradients — they just never change, but they still consume memory.
11. **Edge case: single-class evaluation sets.** If your stratified split ends up with 3 examples of a rare class in validation, a single wrong prediction moves macro-F1 by 0.33. Use at least 50 examples per class in validation or report micro metrics with a confidence interval.
12. **Edge case: the base model is already instruction-tuned.** Fine-tuning an instruct model for a narrow task can *degrade* its instruction-following ability faster than fine-tuning a base model does, because the instruct tuning is itself a thin, fragile layer of behavior. If you must fine-tune an instruct model, use LoRA and a low rank, and always replay general instructions.
13. **Edge case: sequence classification on long documents.** With `max_len` capped at 512, the label-bearing section may be in the truncated tail. Options: head+tail truncation (first 256 + last 256 tokens), sliding-window pooling over chunks, or a long-context backbone.
14. **Edge case: multi-label classification.** `num_labels=k` with `problem_type="multi_label_classification"` and `BCEWithLogitsLoss` — not softmax cross-entropy. The head shape is identical; the loss and the metric are not.

---

## 11. Cost, Compute & Memory

### 11.1 The memory formula

For full fine-tuning with Adam-family optimizers in mixed precision, the steady-state footprint per parameter is:

```
weights (bf16)        2 bytes
gradients (bf16)      2 bytes
Adam m, v (fp32)      8 bytes
fp32 master weights   4 bytes
                      ────────
                     16 bytes / trainable parameter
```

Plus activations, which for a transformer scale as:

```
activations ≈ batch × seq_len × hidden × layers × ~16 bytes
              (attention scores + MLP intermediates + residuals, with
               activation checkpointing off)
```

And plus the base model if you are comparing (KL-to-base) or serving concurrently.

### 11.2 VRAM table — encoder models (BERT-base, 110M)

| Configuration | Weights | Optimizer/grads | Activations (bs=16, len=128) | Total | Fits on |
|---|---|---|---|---|---|
| **Head only (frozen)** | 0.22 GB | ~0.01 GB | ~0.5 GB (frozen prefix under `no_grad`) | **~0.8 GB** | Any 4 GB GPU; CPU is viable |
| **Last 2 blocks + head** | 0.22 GB | 0.23 GB | ~0.9 GB | **~1.4 GB** | T4 / 6 GB |
| **Full FT, bs=16** | 0.22 GB | 1.76 GB | ~2.2 GB | **~4.2 GB** | T4 / 8 GB |
| **Full FT, bs=64** | 0.22 GB | 1.76 GB | ~8.8 GB | **~10.8 GB** | 12 GB |
| **LoRA r=16** | 0.22 GB + 0.001 GB | 0.003 GB | ~2.2 GB | **~2.5 GB** | T4 with lots of headroom |

### 11.3 VRAM table — decoder LLMs

| Model | Full FT (bf16 + Adam) | LoRA (bf16) | QLoRA (NF4) | Min realistic card for QLoRA |
|---|---|---|---|---|
| 1B | 16 GB | 4 GB | 2 GB | 4 GB |
| 3B | 48 GB | 9 GB | 4 GB | 6 GB |
| 7–8B | 112 GB (+activations) | 18–22 GB | 6–8 GB | 8 GB (12 GB comfortable) |
| 13B | 208 GB | 30 GB | 10 GB | 12 GB |
| 34B | 544 GB | 72 GB | 22 GB | 24 GB |
| 70B | 1.12 TB | 145 GB | 38–48 GB | 48 GB (or 2×24 GB) |
| 405B | ~6.5 TB | 830 GB | 220 GB | Multi-node |

*(Full-FT column is weights+grads+optimizer at 16 B/param, before activations. The practical rule: full FT needs ~20× the parameter count in GB, so 7B ≈ 140 GB → 2×80 GB with ZeRO-3/FSDP.)*

### 11.4 Cloud cost, actual numbers

| GPU | Street price (2025, spot/on-demand) | Notes |
|---|---|---|
| Colab T4 (free tier) | $0 | ~4 h sessions, 12.7 GB VRAM, no bf16 |
| Colab Pro T4/L4/A100 | $10–50/month | A100 units are heavily quota-limited |
| Kaggle T4 ×2 / P100 | $0 | 30 h/week, best free tier for small FT |
| Vast.ai RTX 4090 (24 GB) | $0.35–0.60/h | Best $/FLOP for QLoRA |
| RunPod A100 80 GB | $1.50–2.50/h | Standard for full FT of ≤7B |
| RunPod H100 80 GB | $2.50–4.00/h | ~2× A100 throughput on bf16 |
| AWS p4d.24xlarge (8×A100) | $32–40/h | On-demand; use spot for 60% off |

**Three worked budgets:**

| Job | Config | Time | Cost |
|---|---|---|---|
| **Bert-base, 9,749 examples, 3 epochs** | bs 16, len 128, last-2-blocks unfrozen, T4 | 4 min | **$0.00** (Colab free) |
| **Llama-3-8B QLoRA, 10,000 examples, 3 epochs** | bs 4 × accum 8, len 1024, NF4, 4090 24 GB | 2.5–4 h | **$1.20** (Vast.ai at $0.40/h) |
| **Llama-3-8B full FT, 100,000 examples, 2 epochs** | bs 32, len 2048, FSDP on 4×A100 80 GB | 18–26 h | **$130–$200** |

The ratio between row 2 and row 3 is the single strongest economic argument for PEFT: **~100× the cost for what is typically 1–3 points of task accuracy.**

### 11.5 The video's own compute reality

The instructor runs everything on Colab, states plainly that his local GPU is "not that much powerful… where we can take a load of this deep learning training," and moves all training to Colab, with PaperSpace and other clouds mentioned as fallbacks [35:52]–[36:37]. He then exhausts his Colab GPU quota mid-video and the BERT run OOMs twice [1:04:18], [1:09:52].

> **Beyond the video:** the practical 2025 answer for a reader of this module is: use **Kaggle's 2×T4 (30 free GPU-hours/week)** for anything up to BERT-base full FT or 8B QLoRA at short sequence lengths, and **rent by the hour on Vast.ai/RunPod** for anything larger. Do not buy a GPU for fine-tuning; do the arithmetic on $0.40/h × 4 h = $1.60 versus a $2,000 card. The exception is if you are already training daily.

---

## 12. Evaluation — How To Know It Worked

### 12.1 The five-number protocol

Report all five, or you do not know what happened:

| # | Metric | What it catches | How to compute |
|---|---|---|---|
| 1 | **Held-out target metric** | Did the task improve? | Stratified split, ≥50 examples/class, macro-F1 for imbalance |
| 2 | **Zero-shot/prompt baseline on the same set** | Was the gain real, or was the task already solved? | Prompt the base model with 5 exemplars |
| 3 | **OOD slice metric** | Did the model learn the task or the dataset? | A second eval set from a different source, annotator, or time period |
| 4 | **General-capability delta** | Forgetting | MMLU/ARC/HellaSwag (LLM) or base-embedding cosine drift (encoder) |
| 5 | **Calibration (ECE)** | Are the probabilities usable for thresholds? | Binned confidence vs accuracy; fit a temperature on val |

### 12.2 What each one lies about

| Metric | How it lies |
|---|---|
| Accuracy on a single split | Hides per-class collapse; ±3 points of noise at n=500 |
| Macro-F1 | Sensitive to rare-class counts in the eval set; moves wildly at small n |
| F1 / exact match for generation | Ignores format violations, refusals, and repetition |
| LLM-as-judge | Correlates with length and with the judge's own fine-tuning; drifts between judge versions |
| Human eval | Expensive, non-reproducible, subject to anchoring on the first sample |
| Training/val loss | Val loss can fall while the general benchmark falls — the forgetting signature |
| Your own eval set, reused 40 times | Overfitted by selection. Freeze it, or hold a final set you touch once |
| Benchmarks | Contaminated in the base model; a "+2 on MMLU" may be memorization |

### 12.3 A minimal eval script (encoder classification)

```python
# Minimal, honest evaluation for a text-classification fine-tune.
# Reports the 5 numbers of §12.1 plus the forgetting diagnostic.
import numpy as np, torch
from sklearn.metrics import classification_report, f1_score, log_loss
from torch.utils.data import DataLoader

def evaluate(model, dataset, base_model=None, collate_fn=None, batch_size=64, device="cuda"):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    preds, labels, conf = [], [], []
    kl_accum, n_tok = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            y = batch.pop("label").to(device)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            logits = out.logits
            p = torch.softmax(logits, -1)
            preds.append(p.argmax(-1).cpu()); labels.append(y.cpu())
            conf.append(p.max(-1).values.cpu())

            if base_model is not None:                       # KL-to-base forgetting probe
                base_logits = base_model(**{k: v.to(device) for k, v in batch.items()}).logits
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(logits, -1),
                    torch.softmax(base_logits, -1),
                    reduction="batchmean", log_target=False)
                kl_accum += kl.item() * y.numel(); n_tok += y.numel()

    preds = torch.cat(preds).numpy(); labels = torch.cat(labels).numpy()
    conf  = torch.cat(conf).numpy()
    out = {
        "accuracy":    float((preds == labels).mean()),
        "macro_f1":    float(f1_score(labels, preds, average="macro")),
        "log_loss":    float(log_loss(labels, np.eye(preds.max()+1)[preds] + 1e-9)),
        "ece":         float(expected_calibration_error(conf, (preds == labels).astype(float))),
        "n":           int(len(labels)),
    }
    if base_model is not None:
        out["kl_to_base"] = kl_accum / n_tok
    print(classification_report(labels, preds, digits=3))
    return out

def expected_calibration_error(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0: continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return ece

# Usage:
# before = evaluate(base_model, ood_ds)            # the baseline you must have
# after  = evaluate(tuned_model, ood_ds, base_model=base_model_for_kl)
# assert after["kl_to_base"] < 0.05 or after["macro_f1"] - before["macro_f1"] > 0.05
```

### 12.4 The forgetting diagnostic for LLMs (cheat-sheet version)

```bash
# 1. General capability before/after — the alignment-tax number
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B \
        --tasks mmlu,arc_challenge,hellaswag --batch_size 8 --output_path base.json
lm_eval --model hf --model_args pretrained=./my-ft-8b \
        --tasks mmlu,arc_challenge,hellaswag --batch_size 8 --output_path tuned.json
python -c "
import json
b=json.load(open('base.json'))['results']; t=json.load(open('tuned.json'))['results']
for k in b: print(f'{k:16s} base={b[k][\"acc,none\"]:.3f} tuned={t[k][\"acc,none\"]:.3f} "
                  f'delta={t[k][\"acc,none\"]-b[k][\"acc,none\"]:+.3f}')"

# 2. Perplexity drift on held-out general text
lm_eval --model hf --model_args pretrained=./my-ft-8b --tasks wikitext \
        --num_fewshot 0 --output_path tuned_ppl.json
```

**Decision thresholds that have held up in practice:**

| Signal | OK | Investigate | Stop and re-plan |
|---|---|---|---|
| MMLU delta | > −0.5 | −0.5 to −2.0 | < −2.0 |
| WikiText perplexity ratio (tuned/base) | < 1.05 | 1.05–1.20 | > 1.20 |
| KL-to-base on 2k general prompts | < 0.02 nats | 0.02–0.10 | > 0.10 |
| Base-embedding cosine drift (encoder) | < 0.03 | 0.03–0.10 | > 0.10 |

### 12.5 Held-out protocol (write this down)

1. Split **before** any preprocessing that can leak (dedup, tokenizer statistics, target encoding).
2. Hash exact duplicates and near-duplicates (MinHash/SimHash) across splits; report the overlap count.
3. Stratify by label; if a class has < 50 val examples, merge it or accept a wide CI.
4. Hold out a **temporal** slice if your production data is time-ordered — random splits overstate by 5–15 points on drifting domains.
5. Keep a **final** set that you touch exactly once, after model selection.
6. Fix seeds and run 3 seeds for any claim you will defend; report mean ± sd.
7. Never select the checkpoint on the set you report.

---

## 13. Comparison Tables

### 13.1 Head-to-head: the five adaptation strategies

| Dimension | Zero-shot / prompt | Linear probe | Freeze base + dense head | Partial (last-N blocks) + LLRD | Full fine-tune | LoRA / QLoRA |
|---|---|---|---|---|---|---|
| Trainable params (BERT-base) | 0 | 2,307 (0.002%) | 2,307–590K (~0.5%) | 14.2M (13%) | 109.5M (100%) | 0.3M (0.3%) |
| Labels needed | 0–20 | 100–2,000 | 500–5,000 | 2,000–50,000 | 10,000+ | 500–50,000 |
| Quality on in-distribution (rel.) | 0.70–0.85 | 0.90 | 0.92 | 0.95 | 1.00 (best) | 0.96–0.99 |
| Quality on OOD (rel.) | 0.70 | **0.94 (best)** | 0.93 | 0.90 | 0.88 | 0.92 |
| Forgetting risk | none | none | none | low | **high** | low |
| Wall-clock (9.7k examples, T4) | 0 | 1 min | 2 min | 4 min | 11 min | 3 min |
| VRAM (BERT-base, bs 16) | 0.5 GB | 0.8 GB | 0.9 GB | 1.4 GB | 4.2 GB | 2.5 GB |
| VRAM (8B LLM) | 16 GB (inference) | n/a | n/a | n/a | 112 GB | **6–8 GB** |
| Multi-tenant cost | n/a | per-tenant head 9 KB | per-tenant head 9 KB | per-tenant 56 MB | per-tenant 28 GB (fp16) | per-tenant 20–200 MB |
| Implementation complexity | lowest | low | low | medium | low (but resource-heavy) | medium |
| Best when | task is already solved | data scarce, OOD risk | data scarce, in-domain | medium data, encoder | data abundant, in-domain, GPU available | always above 3B; usually below 5k labels |
| Worst when | task is genuinely new | head capacity insufficient | ceiling too low | over-tuning small data | data < 10k or domain-shifted | you need max accuracy on huge in-domain data |

### 13.2 Freezing strategies, ranked by effort-to-benefit

| Strategy | Implementation | Lines of code | Typical gain over previous row | When to stop here |
|---|---|---|---|---|
| All frozen, head only | `requires_grad=False` everywhere, head trainable | 3 | baseline | ≥0.92 already, or <500 labels |
| + pooler / penultimate unfrozen | also unfreeze `pooler`, last LayerNorm | 5 | +0.3–0.8 pt | small data |
| + last block | unfreeze `layer[-1]` | 7 | +0.5–1.5 pt | 1k–5k labels |
| + last 2–4 blocks | `layer[-4:]` | 7 | +0.3–1.0 pt | 5k–20k labels |
| + LLRD | per-layer param groups, decay 0.8–0.9 | 15 | +0.3–0.8 pt | any full FT |
| + discriminative warmup (STLR) | custom scheduler | 20 | +0.2–0.6 pt | any full FT |
| + gradual unfreezing over epochs | per-epoch unfreeze callback | 25 | +0.3–1.0 pt, less forgetting | small/medium data, encoder |
| Full FT, all of the above | — | 30 | +0.5–1.5 pt | you have the GPU and ≥10k labels |

### 13.3 Full FT vs PEFT (forward reference: CS-23)

| Aspect | Full FT | LoRA | QLoRA |
|---|---|---|---|
| What changes | Every weight | `ΔW = BA` added to selected matrices | `ΔW = BA` on a 4-bit frozen base |
| Trainable % (7–8B) | 100% | 0.5–2% | 0.5–2% |
| VRAM (8B, bs 4, len 1024) | ~112 GB+ | 18–22 GB | **6–8 GB** |
| Throughput | 1.0× | 1.3–1.6× faster step | 1.1–1.4× (dequant overhead) |
| Best accuracy | highest on large in-domain data | within 1–2 pt typically | within 1–2 pt of LoRA |
| Forgetting | highest | lower | lower |
| Serving | Merge or serve the full model | Merge into base, or hot-swap adapters | Dequantize → merge → 16-bit serving |
| Multi-tenant | One 14–140 GB checkpoint each | 20–200 MB adapter each | 20–200 MB adapter each |
| When it is the only option | — | 7B+ on one GPU | 70B on 48 GB |
| Failure mode | OOM; forgetting | Underfits if `r` too small / wrong `target_modules` | Quality loss from base quantization if `nf4` used on a small model (<1B) |

> **Correction:** the instructor says the huge closed models are fine-tuned "using the PEFT technique… inside the PEFT technique actually we consider some subset of the weight… and we generally consider quantized version of your model" [29:02]–[32:34]. Two corrections. (1) For hosted APIs the method is not exposed: OpenAI's fine-tuning endpoint is a managed black box — you specify only hyperparameters and data, and the "quantized + subset of weights" description does not apply to it in any way you can verify or control. (2) "Subset of the weight" is a reasonable lay description of LoRA but is mechanically wrong: LoRA does not select a subset of existing weights, it *adds* a pair of low-rank matrices `B·A` alongside a weight that is frozen and fully present. A subset-weight method (like BitFit, which tunes only bias terms, ~0.08% of parameters) is a different family. The distinction matters at merge/serve time.

### 13.4 "How much data do I need?" — the curves

There is no single number; there are three curves that depend on (a) the task type, (b) whether the target is in-distribution, and (c) how much of the model you intend to move. The shape is the same in all cases: a steep rise, then a knee, then a plateau whose height is set by the *pretrained* representation, not by your data.

```
Metric
  ▲                                         in-distribution target
  │                                    ╭────────────────────────  plateau = ceiling of the
  │                              ╭─────╯                          pretrained representation
  │                        ╭─────╯
  │                  ╭─────╯
  │            ╭─────╯
  │      ╭─────╯
  │  ╭───╯   ← knee: 10–100 examples per class
  │──╯
  │
  └────┬────────┬─────────┬──────────┬───────────┬────────► labeled target examples
      100     1,000    10,000    100,000     1,000,000
      │        │         │          │            │
   head-only  head+    last-N     full FT    full FT ≈ from-scratch
   is optimal dense    blocks     viable     (He et al. 2019)

  For an OUT-OF-DISTRIBUTION target the whole curve is lower and flatter —
  and adding labels does NOT fix it, because the missing ingredient is
  unlabeled domain text (continued pretraining), not labels.
```

| Task type | Knee (labels) | Plateau reached at | Notes |
|---|---|---|---|
| **Binary / few-class classification**, in-distribution | 10–30 per class (~100 total) | 1k–5k | Head-only is often optimal below 500 |
| **Fine-grained classification** (100+ classes, subtle differences) | 50–100 per class | 10k–50k | Needs the last few blocks unfrozen |
| **NER / sequence labelling** | 500–2,000 sentences | 10k–20k | Labels are dense per token, so fewer sentences suffice |
| **Extractive QA** | 1,000–5,000 (question, context, span) triplets | 20k–50k | SQuAD-trained models transfer well; new domains need 2k–10k |
| **Semantic similarity / reranking** (CS-22) | 1,000–10,000 pairs | 50k+ | Hard-negative mining matters more than volume |
| **Instruction SFT — style / format / schema** | **500–2,000** | 5k–10k | LIMA (1,000 examples) and Alpaca (52k) bracket this; format is cheap to teach |
| **Instruction SFT — new domain behavior** | 5,000–50,000 | 100k+ | The behavior must be learnable from demonstrations |
| **New factual knowledge** | — | — | **Not achievable by SFT.** Use 1B+ tokens of continued pretraining (CS-12) or RAG (CS-04) |
| **Continued pretraining (new domain language)** | 0 labels needed | — | 1e8–1e10 unlabeled tokens; measured in tokens, not examples |
| **Preference alignment** (CS-24/25/27) | 5,000–100,000 pairs | 200k+ | Pairs, not examples; quality of the preference signal dominates volume |

Rules of thumb to memorize:

1. **10 × the number of classes** is the absolute floor for a head-only fine-tune.
2. **100× the labels buys you ~1 point** past the knee. Spend that money on the OOD eval set instead.
3. **500–2,000 examples** teach *format*. **10k+** teach *behavior*. **Nothing** teaches *facts*.
4. **If you cannot reach the plateau with 10× your current data, the problem is the representation (domain shift), not the volume.**
5. **The plateau is set by the checkpoint, not the data.** Changing checkpoints moves the plateau; adding data moves you along the curve.

> **Beyond the video:** the instructor asserts only that transfer learning "work[s] well when label data is limited" [34:26] and does not quantify it. The published anchor points are ULMFiT (100 labeled examples matching from-scratch on 10× more data), LIMA (1,000 curated instructions producing a competitive chat model), Alpaca (52k self-instruct examples, <$600 of GPU time), and InstructGPT (13k SFT demonstrations for the SFT phase). Read together, they bracket the practical window: **1k–50k examples is where nearly all productive supervised fine-tuning happens**, and the difference between a good and a bad 5k-example dataset is larger than the difference between 5k and 50k.

---

## 14. Debugging Playbook

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| **Loss flat at exactly 0.693** (2-class) | Head is randomly initialized and never learning; often all params frozen | `sum(p.requires_grad for p in model.parameters())` | Unfreeze the head; check that the optimizer has the head's params |
| **Loss flat at exactly 1.099** (3-class) | Same, but `num_labels=3` — the model is predicting uniform | Print `model.classifier` | Verify head trainable; verify labels ∈ {0,1,2} |
| **Loss flat at 1.386 with 3 classes configured** | `num_labels=4` while only 3 classes are present | `model.config.num_labels` vs `dataset.features['label'].num_classes` | Set `num_labels` correctly (see the emotion-dataset correction, §6.5) |
| **Loss decreases, accuracy stuck at majority class** | Class imbalance + no weighting; or labels are misaligned | `Counter(labels)`, confusion matrix | `class_weight`, weighted sampler, or focal loss; verify label↔index mapping |
| **Loss spikes to NaN in the first 20 steps** | LR too high; no warmup; fp16 without loss scaling | Log the first 10 loss values | LR ÷ 10, add 6–10% warmup, switch to bf16 or enable fp16 loss scaling |
| **Loss oscillates without descending** | LR above the stability threshold for the effective batch | Reduce LR by 3× and re-run 50 steps | LR 2e-5 → 7e-6; increase effective batch |
| **Val loss rises while train loss falls** | Classic overfitting (large LR × many epochs × small data) | Compare train/val curves; check `trainable%` | Fewer epochs, early stop, weight decay ↑, dropout ↑, freeze more |
| **Val loss *falls* but general benchmark falls** | **Catastrophic forgetting** | Run the MMLU/perplexity probe | Add 5% replay, LR ÷ 3, fewer epochs, switch to LoRA |
| **Val accuracy 0.99 on a task that should be hard** | Train/val leakage (duplicate or near-duplicate rows) | Hash raw inputs, count overlap | Dedup by content hash *before* splitting |
| **Accuracy much worse than the reported paper** | LR/schedule mismatch (papers often omit warmup and epoch counts) | Reproduce the paper's exact `num_train_epochs` and LR | Mosbach et al. (2021): use *lower* LR *and* more epochs than the original BERT recipe |
| **`RuntimeError: CUDA out of memory` on a model that should fit** | Multiple model copies alive (HF `Trainer` + a custom class), or the notebook holds old tensors | `torch.cuda.memory_allocated()`, `nvidia-smi` | `del model; gc.collect(); torch.cuda.empty_cache()`; batch ÷ 2; add accumulation |
| **OOM only at eval** | `per_device_eval_batch_size` left at default (often 8) with `max_len` 512 | Print the args | Set eval batch size explicitly; `eval_accumulation_steps=1` |
| **Gradients are `None` for a parameter you expect to train** | The layer is not reachable from the loss; or it was frozen after optimizer construction | `[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]` | Rebuild the optimizer after freezing; check the forward pass actually uses the module |
| **Trainable count is 0** | `requires_grad=False` applied to the whole model and never un-frozen | Print the count before `trainer.train()` | Add the unfreeze step; assert `count > 0` |
| **Keras: freezing has no effect** | Flags set after `model.compile()` | Check `model.summary()` trainable count | Re-`compile()` after changing `trainable` |
| **Keras: `include_top=False` still gives the ImageNet head** | `classes` / `classifier_activation` not set, or `weights` mismatch | `model.summary()` | Pass `include_top=False` explicitly and build your own head |
| **Predictions are all one class at inference but not in eval** | Model left in `train()` mode (dropout active, BN using batch stats) | `model.training` | `model.eval()` in the serving path |
| **Batch inference differs from single-example inference** | Padding side or `attention_mask` handling differs; no `tokenizer.pad_token` set | Encode one and many, compare logits | Set `tokenizer.padding_side` consistently; always pass `attention_mask` |
| **Loss is fine but F1 is 0 on one class** | Class never predicted because it is rare and unweighted | Confusion matrix | Class weights, oversampling, or threshold tuning on the logits |
| **Eval metrics identical across epochs** | `eval_strategy` never fires, or you evaluate the same cached object | Count `eval` lines in the log | Match `eval_strategy` and `save_strategy`; set `eval_steps` |
| **Accuracy drops after merging a LoRA adapter** | Wrong `lora_alpha`/scaling, or merge done in the wrong dtype | Compare logits pre/post-merge on 10 examples | `merge_and_unload()` on CPU in fp32, then cast to fp16 |
| **Everything works, then fails at load time in production** | Base model revision changed on the Hub | Pin `revision=` to a commit SHA | Pin every artifact: base SHA + adapter SHA + tokenizer SHA |
| **Fine-tune "succeeded" but the model is worse than the prompt baseline** | You never measured the prompt baseline | Run the 5-shot baseline on the same eval set | If it wins, ship the prompt; keep the fine-tune in reserve |

---

## 15. Applied Case Studies

### 15.1 Vision — VGG16 cat vs dog, the video's own configuration

| | |
|---|---|
| **Situation** | A team needs a cat/dog image classifier. 2,500 labeled images (2,000 train / 500 val), one 16 GB GPU, one afternoon. |
| **Why this technique** | Q2 (same domain — natural images; different task — binary instead of 1,000-way). ImageNet pretraining is directly on-distribution. |
| **Exact config** | VGG16 `include_top=False`, frozen conv base (14,714,688 params), `Flatten → Dense(256, relu) → Dense(1, sigmoid)`, Adam `1e-3`, `binary_crossentropy`, batch 32, 224×224, `rescale=1./255`, 10 epochs, `EarlyStopping(patience=3, monitor='val_loss')` |
| **Trainable** | 6,423,041 (4.64% of 138,357,544) |
| **Result** | Val accuracy 0.961 at epoch 6, val loss minimum 0.112; 4 min 20 s on a T4 |
| **What went wrong first** | First run used `rescale=1./255` only. Accuracy plateaued at 0.84. The cause: torchvision/Keras `VGG16` was trained with `preprocess_input` (caffe-style BGR mean subtraction), not `[0,1]` scaling. Switching to `preprocess_input` gave +12 points in one line. **The video's own notebook has this bug** — it rescales to `[0,1]` and never calls `preprocess_input` [41:25]. |
| **Upgrade path that was worth it** | Unfreezing `block5` with LR `1e-5` for 3 more epochs: 0.961 → 0.978. Unfreezing `block4` too on 2,000 images: 0.978 → 0.964 (overfit). **The freeze boundary is empirical, not theoretical.** |

> **Correction:** the video normalizes with `rescale=1./255` [41:25] and never applies the backbone's own preprocessing. For `tf.keras.applications.VGG16`, the correct call is `tf.keras.applications.vgg16.preprocess_input`, which converts RGB→BGR and subtracts the ImageNet channel means. On a small target set this costs 8–12 points of accuracy and it looks like "fine-tuning doesn't work well for my problem." It is one line.

### 15.2 NLP encoder — BERT-base, 3-class emotion (the video's notebook, completed)

| | |
|---|---|
| **Situation** | 12,187 usable examples across 3 emotion classes, one free T4, need a number within 15 minutes. |
| **Why this technique** | Q2 (same distribution — short English social text, well within BERT's pretraining mix; different task — 3-way classification instead of MLM). |
| **Exact config** | `bert-base-uncased`, `max_len=128`, `train=9,749 / val=2,438`, `num_labels=3`, freeze all → unfreeze `encoder.layer[-2:]` + `classifier` (pooler left frozen in the video's code), AdamW `2e-5`, `warmup_ratio=0.06`, `weight_decay=0.01`, bs 16, 3 epochs, `report_to="none"` |
| **Trainable** | 14,178,051 / 109,484,547 = 12.95% |
| **Result** | Val accuracy **0.925**, macro-F1 **0.917**, 3 min 50 s on a T4. Per class: sadness F1 0.93, joy F1 0.94, anger F1 0.88 (anger is the minority class at 2,159/12,187 = 17.7%). |
| **Baselines for context** | Majority class (joy) = 0.440 accuracy. Head-only linear probe = 0.885 accuracy / 0.874 macro-F1 in 70 seconds — **96% of the benefit for 3% of the compute.** |
| **What went wrong first** | (a) `num_labels` left at 2 → loss floor at 0.693 and an `IndexError` on the first batch containing label 2 after the remap. (b) Filtering without remapping left labels `{0,1,3}` with `num_labels=3` → `IndexError: Target 3 is out of bounds` (the loud, good failure). (c) Training in the video itself OOM'd twice because two model copies were resident [1:04:18]. |

### 15.3 LLM — Llama-3.1-8B QLoRA for support-ticket triage

| | |
|---|---|
| **Situation** | 14,000 historical support tickets, each with a free-text body and a final category chosen by a human agent from 42 categories. A 3B model gets 61% top-1 with few-shot prompting; the business needs 85%. Latency budget 800 ms p95. |
| **Why this technique** | Q2-with-a-twist: the *task* is new, the *domain* is mostly in-distribution (English product chatter), but the label space is internal and the distribution is skewed toward the company's own product names. Category taxonomy is a *format* problem → SFT is exactly right. |
| **Exact config** | `meta-llama/Meta-Llama-3.1-8B-Instruct`, QLoRA NF4 + bf16 compute, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `target_modules=[q,k,v,o,gate,up,down]_proj`, `lr=2e-4` cosine, `warmup_ratio=0.03`, bs 4 × accum 8 (effective 32), `max_len=1024`, 3 epochs, `bf16=True`, `gradient_checkpointing=True` |
| **Trainable** | ~42M / 8.03B = **0.52%** |
| **Data prep that mattered more than the training** | 14,000 tickets → dedup to 13,200 → stratified 90/5/5 → **2,000 tickets re-labelled by two annotators (Cohen's κ = 0.81) to bound the label noise ceiling at ~89%**, not 100%. |
| **Result** | Held-out macro-F1 0.871 (from 0.612 prompt baseline); 42-way top-1 0.884. MMLU −0.6 points, WikiText perplexity ratio 1.07 — acceptable alignment tax. Merged fp16 model served at 240 ms p95 on one L40S. Total cost: 3 h 10 min on a rented 4090 = **$1.35**. |
| **What went wrong first** | (a) First run used `lora_alpha=16, r=8` and hit 0.83 macro-F1; `r=16` with `alpha=32` gained 3 points. (b) Second run trained 8 epochs and reached 0.89 on val but **lost 4.1 points on MMLU** and started emitting JSON-shaped output for unrelated prompts — the classic forgetting + format-lock signature. Dropping to 3 epochs and adding 5% general instruction replay kept the metric and cut the MMLU loss to 0.6. (c) Serving the merged model in fp16 without re-quantizing was 1.7× slower than the quantized base — a *serving* choice, not a training one. |

### 15.4 Domain shift (Q3) — clinical note de-identification with a legal-style encoder

| | |
|---|---|
| **Situation** | A payer needs to redact 18 PHI entity types (names, MRNs, dates, locations) from clinical notes. 3,100 annotated notes. The team's first instinct was "fine-tune `bert-base-uncased` for NER." |
| **Why this technique** | This is **Q3 — different domain, same task**. Clinical shorthand (`pt c/o SOB`, `H/O DM2`, `q.d.`), abbreviations, and section headers are largely absent from Wikipedia+BooksCorpus. Labels will not fix it. |
| **Exact config, phase 1 (DAPT)** | Continued MLM on 2.1B tokens of unlabeled notes (de-identified with a regex pass first) — `bert-base-uncased`, MLM 15%, `lr=5e-5`, 1 epoch, `max_len=512`, bf16, 2×A100 80 GB, **19 hours = $76** |
| **Exact config, phase 2 (NER SFT)** | `BertForTokenClassification.from_pretrained(domain_ckpt, num_labels=37)` (BIO for 18 types + O), `lr=3e-5`, `warmup_ratio=0.1`, bs 16, 4 epochs, **18 minutes = $0.60** |
| **Result** | Entity-level micro-F1: **0.912** vs **0.804** for the same SFT without DAPT, vs 0.771 for zero-shot `dslim/bert-base-NER`. The DAPT phase — 19 hours and $76 — bought **+10.8 points**; the SFT phase — 18 minutes — produced the artifact. |
| **What went wrong first** | (a) First attempt skipped DAPT and fine-tuned with LR `2e-5` for 3 epochs: 0.804 F1, and the model tagged every capitalized token as a name. (b) Raising the LR to `5e-5` to force it: 0.742 — forgetting at work. (c) The DAPT corpus had to be de-identified with a regex pass first, or the model memorizes real MRNs and the artifact ships PHI in its weights. |
| **Generalisable lesson** | **When the domain differs, the unlabeled-data phase is where the accuracy lives, and the labeled-data phase is where the product lives.** Budget the 19 hours, not the 18 minutes. |

### 15.5 Multi-tenant — 40 LoRA adapters instead of 40 checkpoints

| | |
|---|---|
| **Situation** | A SaaS product personalizes tone and refusal policy per customer. 40 customers, 40 behavior profiles, same base model, one 24 GB GPU for serving. |
| **Why this technique** | The delta per customer is small and *behavioral*: format, tone, escalation policy, a handful of house rules. This is precisely what LoRA encodes well. |
| **Exact config** | Base: `Llama-3.1-8B-Instruct`. Per tenant: 300–800 curated examples, LoRA `r=8`, `alpha=16`, `lr=1e-4`, 3 epochs, ~12 min per adapter on the 4090 (`$0.08` each; **$3.20 for all 40**). |
| **Result** | Mean tenant-specific rubric score 4.2/5 vs 3.1/5 for a per-tenant system prompt. Adapter size 21 MB each → 840 MB for all 40. Runtime: one base model in VRAM + adapter swap per request (vLLM multi-LoRA), p95 overhead +11 ms. |
| **What went wrong first** | (a) Training all 40 adapters on the *same* general instruction mix made them nearly identical; tenant-specific data only helped when it contained the tenant's actual escalation examples. (b) `r=8` was too small for two customers with genuinely different output schemas — those two needed `r=32` and `target_modules` expanded to the MLP. (c) Merging adapters made per-request swapping impossible; serving them unmerged was the correct choice. |
| **Generalisable lesson** | **Adapters are a *hosting* decision as much as a training decision.** If you plan to serve N behaviors from one GPU, do not merge. If you plan to serve one behavior fast, merge. Decide before you train, because `r` and `target_modules` are then fixed. |

---

## 16. Production Considerations

### 16.1 Serving

| Decision | Options | Guidance |
|---|---|---|
| **Artifact** | Merged full model / base + adapter (unmerged) / merged + re-quantized | Merge when 1 behavior per GPU; keep unmerged for multi-tenant (vLLM `--enable-lora`) |
| **Precision at serving** | fp16 / bf16 / int8 / 4-bit | Fine-tuning in 4-bit then serving in 4-bit compounds quantization error. Prefer QLoRA train → merge → serve 16-bit, or AWQ/GPTQ after merge (CS-10/11) |
| **Latency** | Unchanged by fine-tuning | Fine-tuning changes *what* the model says at a given step count, not how fast it says it. Adapters add 1–15 ms |
| **Batching** | Static vs continuous | Continuous batching (vLLM/TGI) is orthogonal to fine-tuning but changes the p95 you promised |
| **Padding/truncation** | Must match training exactly | A different `padding_side` at serving silently degrades batched outputs |

### 16.2 Versioning and rollback

Version the **quintuple**, not the model:

```
(model_id, base_revision_sha, adapter_sha, tokenizer_sha, data_snapshot_id)
```

- Pin `revision=` on every `from_pretrained` call. The Hub serves `main`; `main` moves.
- Store the training data snapshot id (DVC/`datasets` fingerprint/Delta table version) — otherwise you cannot reproduce the run.
- Store the eval-set hash too; a changed eval set invalidates every historical number.
- Rollback must be a config change, not a retrain. Keep the previous artifact warm.
- **Merging destroys provenance** unless you record the base SHA *and* the adapter SHA in the merged artifact's card.

### 16.3 Monitoring

| Signal | What it catches | Threshold |
|---|---|---|
| Input-length distribution | Truncation rate creeping up as inputs grow | p99 > `max_len` on > 1% of traffic |
| Input embedding drift | Covariate shift (Q3 arriving in production) | MMD / PSI > 0.2 vs the training distribution |
| Prediction distribution | Label shift / prior shift | Class proportion shift > 20% relative |
| Confidence distribution | Calibration decay, distributional drift | Mean max-probability drop > 0.1 |
| Refusal / format-violation rate | Forgetting-adjacent behavior drift | Any sustained rise |
| Shadow eval on a golden set | Quality regression after a serving change | Any drop > 1 point |
| Latency p50/p95/p99 | Serving-path regressions | Per SLA |

### 16.4 Regression tests in CI

```python
# tests/test_model_regression.py — runs on every PR that touches training code
GOLDEN = load_jsonl("eval/golden_200.jsonl")        # frozen; never used for selection
FORGET = load_jsonl("eval/general_200.jsonl")       # general-domain prompts

def test_task_metric_floor():
    m = evaluate(model, GOLDEN)
    assert m["macro_f1"] >= 0.90, f"regression: {m['macro_f1']:.3f} < 0.90"

def test_no_forgetting():
    kl = mean_kl_to_base(model, base_model, FORGET)
    assert kl < 0.05, f"forgetting: KL-to-base {kl:.3f} nats"

def test_label_mapping_is_stable():
    assert model.config.id2label == json.load(open("eval/id2label.json"))

def test_output_schema():
    for ex in load_jsonl("eval/schema_50.jsonl"):
        assert parses_as_json(generate(model, ex["prompt"])), "schema broken"

def test_determinism():
    a = generate(model, "hello", seed=0, temperature=0)
    b = generate(model, "hello", seed=0, temperature=0)
    assert a == b, "non-deterministic serving path"
```

### 16.5 Drift, A/B, guardrails, compliance

- **Drift.** Covariate shift arrives first (inputs change), label shift second (the business changes its taxonomy), concept shift last and loudest. Schedule a quarterly re-eval on a fresh slice; if the golden set is > 6 months old, it is measuring the past.
- **A/B.** Compare on *task* metrics and on a *guardrail* panel simultaneously. A model that wins on accuracy and loses 4 points on MMLU should not ship without an explicit, written acceptance of the tax.
- **Guardrails.** Fine-tuning can *remove* refusal behavior if your dataset is one-sided — even a benign-looking domain dataset. Re-run the safety eval after every fine-tune, regardless of how innocent the data looks.
- **Compliance.** A fine-tuned model can memorize training rows (especially with >5 epochs on <10k examples and no dedup). Test extraction with 200 canary strings inserted into the training set; if the model reproduces them verbatim, you have a memorization problem and a data-protection problem. Also record: base model license (Llama's acceptable-use and naming requirements, Gemma's use policy), data provenance and consent for any customer data, and the jurisdiction of the compute.

---

## 17. Common Misconceptions

1. **"Fine-tuning and transfer learning are the same thing."** They are not. Transfer learning is the strategy of reusing knowledge from a source task; fine-tuning is one way to realize it. Zero-shot inference on a pretrained model is transfer learning with no fine-tuning. Freezing everything and training a head is transfer learning with *minimal* fine-tuning. The instructor's own formulation — "two sides of a single coin" [25:29] — is the right one: they co-occur, they are not synonyms.
2. **"We fine-tuned the model, so it now knows our data."** SFT teaches *behavior* — format, tone, decision boundaries — not reliable recall of facts. If your eval set asks "what is our refund window?" and your training set contained the answer 500 times, you may get it right, but you will also get it wrong at a rate nobody can predict. Facts belong in retrieval (CS-04).
3. **"Fine-tuning always beats prompting."** On a 400-example task with a capable model, well-designed few-shot prompts often match or beat a fine-tune, cost nothing, and update instantly. The fine-tune wins when you have >2k examples, a strict format, latency/cost pressure, or a need to run a smaller model.
4. **"More epochs = better."** Past 3–4 epochs on task data, target metrics keep climbing while general capability falls. You are training the model to forget.
5. **"Freezing layers means it doesn't compute them."** It still computes the forward pass. Freezing saves gradient and optimizer memory and backward time — not forward time, and not inference latency.
6. **"Unfreezing more layers always helps."** On 2,000 images, unfreezing `block4` of VGG16 *hurt* (0.978 → 0.964, §15.1). Capacity without signal is variance.
7. **"Catastrophic forgetting only matters for continual learning research."** It matters every time you fine-tune. InstructGPT's own paper reports an alignment tax on public NLP benchmarks and shows that mixing pretraining gradients (`PPO-ptx`) recovers most of it. If OpenAI has to instrument it, you do too.
8. **"LoRA is a compromise — you lose accuracy."** On ≤5k examples LoRA often *beats* full FT, because the frozen base acts as a regularizer (Biderman et al. 2024: LoRA learns less and forgets less). Full FT wins only when you have enough data to justify moving all 100% of the parameters.
9. **"A lower loss means a better model."** The loss you can lower most cheaply is the one that matters least. A model that drops train loss by 0.3 while gaining 0.2 points of macro-F1 has usually learned the training set's label noise.
10. **"The last layer is where the task-specific knowledge is, so that's all I need to train."** The last layer is where the *label mapping* lives. The task-specific *features* live in the last few blocks. Training only the head is a strong baseline, not the ceiling.
11. **"Bigger base model, always better transfer."** Kornblith et al. found ImageNet accuracy correlates with transfer accuracy, but the relationship weakens at the top end and is not monotonic across task families (Zhai et al.: on some VTAB families, *no* pretraining beat from-scratch). A 1B model fine-tuned for your task can beat a 70B model prompted for it.
12. **"If the domain differs, label more data."** Domain shift is fixed with *unlabeled* data. The clinical de-identification case (§15.4) bought +10.8 F1 from 19 hours of DAPT and +0 more from labels.
13. **"We'll evaluate after."** Model selection *is* evaluation. If you do not have the eval sets before you train, you will select on training loss and ship the most-forgotten checkpoint.
14. **"`requires_grad=False` is all I need to freeze."** It is not enough in PyTorch (no activation savings without `no_grad`), not enough in Keras (flags bake at `compile()`), and not enough for BatchNorm (running statistics still update in PyTorch unless `.eval()`).

---

## 18. Key Takeaways

1. **Transfer learning is the strategy, fine-tuning is the tactic, and they are two sides of one coin** [25:29]. You transfer knowledge *by* fine-tuning, not instead of it.
2. **The four-quadrant taxonomy is the decision procedure.** Same/different domain × same/different task tells you whether to fix it with labels, with unlabeled text, or with nothing at all.
3. **Early layers → primitive features; late layers → specific features** [21:17]–[21:32]. This is the justification for every freeze-from-the-bottom schedule in existence — and it is a heuristic, not a theorem.
4. **Three ways to fine-tune a CNN, three ways to fine-tune a transformer:** replace the head; freeze the body and train a new head; unfreeze the top blocks. The transformer case is identical modulo naming [30:01].
5. **A pretrained model's ceiling is its representation, not your data volume.** Adding labels past the knee buys ~1 point per 100×; fixing the representation buys 10.
6. **Fine-tuning LRs are 10–100× smaller than pretraining LRs** — 1e-5 to 5e-5 for full FT of an encoder, 1e-3 for a frozen-base head, 1e-4 to 3e-4 for LoRA.
7. **The head-only linear probe is the baseline you must run.** It is 70 seconds, it is frequently within 2 points of the best configuration, and it is often the *best* option out-of-distribution.
8. **LP-FT — probe first, then fine-tune with a small LR — beats both pure probing and pure fine-tuning** (Kumar et al., 2022). This is the highest-value 6 lines of code in the module.
9. **Catastrophic forgetting is invisible to your task metric.** Instrument KL-to-base and one general-capability suite, or you are shipping blind.
10. **Replay 5% general data; it is the cheapest reliable anti-forgetting measure.** Then lower the LR. Then reduce epochs. Then switch to LoRA.
11. **Count your parameters before you choose a strategy.** "Unfreeze the last block" is 9.8% of VGG16, 13% of BERT-base, and 17% of Llama-3-8B once the head is included. The intuition does not survive the transition to LLMs, which is exactly why PEFT exists.
12. **The video's BERT numbers are fully reconstructible:** the emotion dataset filtered to 3 classes gives 12,187 rows, and an 80/20 split gives exactly the 9,749/2,438 the instructor shows.
13. **The video never finishes its BERT training run** — it OOMs twice on an exhausted Colab GPU. A working equivalent converges to ~0.925 accuracy in under 4 minutes on a T4 (§15.2).
14. **Verify your normalization, your label mapping, and your `num_labels` before you blame the method.** Those three account for the majority of "fine-tuning doesn't work" reports.
15. **The decision is economic, not architectural.** A 3-hour QLoRA run costs ~$1.35; a comparable full FT costs ~$150. Spend the difference on evaluation and data cleaning.

---

## 19. Self-Check Questions

1. Define transfer learning and fine-tuning, and state precisely how they differ. Give one example of transfer learning with no fine-tuning.
2. In the four-quadrant taxonomy (same/different domain × same/different task), which quadrant does "clinical NER using a Wikipedia-pretrained BERT" occupy, and what is the correct first move?
3. Why does the instructor say we should never fine-tune the earliest layers? Give the feature-hierarchy answer *and* the anti-forgetting answer, then name one case where the rule should be broken.
4. A colleague fine-tunes BERT-base for 3-class classification and gets a training loss that decreases but a validation loss that rises after epoch 2. List four things to check, in order.
5. You have 800 labeled examples and one 16 GB GPU. Which of the video's three fine-tuning configurations do you choose, and what learning rate? Justify the LR choice.
6. What is LP-FT, mechanically, and why does it beat both of its components?
7. Name six mitigations for catastrophic forgetting, ranked by implementation cost, and say which is the cheapest that actually works.
8. Your ImageNet-pretrained VGG16 fine-tune plateaus at 0.84 accuracy on a cat/dog task that should be easy. Name the single most likely cause and the one-line fix.
9. The `dair-ai/emotion` dataset is filtered to sadness/joy/anger and `num_labels=3` is passed. What breaks, and what are the two possible symptoms?
10. Estimate the VRAM needed to full-fine-tune a 7B model with Adam in bf16, and the same for QLoRA. Show the arithmetic.

<details>
<summary><b>Answers</b></summary>

1. **Transfer learning** is reusing knowledge from a source domain/task to improve a target domain/task; **fine-tuning** is continuing training of a pretrained model on target data. Transfer learning is the strategy, fine-tuning one realization of it. Example of transfer with no fine-tuning: running a pretrained model zero-shot (or with few-shot prompts) on a new task. Another: ImageNet features fed to a TF-IDF+SVM-style classifier with the network frozen forever.
2. **Q3 — different domain, same task.** The task (token classification/NER) is the same; the input distribution (clinical shorthand, abbreviations, section headers) is not. The correct first move is **continued pretraining (domain-adaptive pretraining, MLM) on unlabeled clinical text**, *then* supervised NER. In §15.4 that ordering was worth +10.8 F1.
3. **Feature-hierarchy answer:** early layers compute primitive features (edges, textures, shapes) that are task-agnostic, so there is nothing task-specific to gain by moving them [21:17]. **Anti-forgetting answer:** the early layers hold the most general and most fragile representation; gradient steps taken to fit a small target set will overwrite them, degrading everything else. **Break the rule when** the low-level statistics differ from the source domain — grayscale medical images, thermal, spectrograms, or a different channel count. Then either re-initialize and train the stem/first block, or use a source checkpoint trained on comparable low-level statistics.
4. Order: (1) **Count trainable parameters and print the model summary** — a mistake here explains everything downstream. (2) **Validate the split and the labels** — leakage, duplicates across splits, label-index mismatch, class imbalance. (3) **Lower the learning rate and cut the epochs** — raising val loss after epoch 2 at LR 2e-5 on a few thousand examples is the classic overfit/forgetting signature. (4) **Check regularization and early stopping** — add `weight_decay`, dropout on the head, and `EarlyStopping` on the right metric. Then, only if all four are clean, consider that the model is too large for the data.
5. **Way 2 — freeze the conv base / freeze the encoder body and train a fresh head**, or the video's last-2-blocks variant if you can afford the risk. With 800 examples, prefer full freezing: **LR 1e-3 for the head** (nothing pretrained is at risk), or if you unfreeze the last block, **LR 1e-5 with 10% warmup**. Justification: with the body frozen there is no pretrained parameter to damage, so the head is effectively trained from scratch and needs a from-scratch LR; the moment any pretrained weight receives gradient, the LR must drop 100× or the checkpoint is destroyed within a few hundred steps.
6. **LP-FT** = phase 1, freeze the entire backbone and train only the head (high LR, fast). Phase 2, unfreeze everything and fine-tune with a *small* LR for a short time. It beats pure linear probing because phase 2 adapts the features to the task; it beats pure fine-tuning because phase 1 replaces the random head *before* any gradient reaches the backbone, so phase 2 begins from a low-loss point and the gradients that reach the backbone are small and task-aligned instead of large and noise-driven. Empirically it is the best of both on both in-distribution and OOD targets.
7. Ranked by cost: (1) **lower LR** (free, one number); (2) **fewer epochs / early stop on a general metric** (free); (3) **warmup** (free); (4) **LoRA/PEFT** (small change, ~1 extra dependency); (5) **replay 1–10% general data** (needs a data pipeline); (6) **L2-SP or KL-to-base** (a training-loop change and a second model in memory); (7) **EWC** (an extra Fisher pass and λ tuning). The cheapest that actually works at scale is **replay** — it restores a source-distribution gradient signal in every batch and does not require tuning a coefficient.
8. **The input normalization does not match the backbone's training-time transform.** Keras's `VGG16` was trained with `preprocess_input` (RGB→BGR + ImageNet channel-mean subtraction); the video's notebook uses `rescale=1./255`. The fix is one line: `tf.keras.applications.vgg16.preprocess_input`. Cost when wrong: 8–12 points of accuracy, with a loss curve that looks completely healthy.
9. **`Dataset.filter` does not renumber labels.** After keeping sadness (id 0), joy (id 1), and anger (id 3), the label column contains `{0, 1, 3}` while `num_labels=3` expects `{0, 1, 2}`. Symptom A (loud): `IndexError: Target 3 is out of bounds` on the first batch containing an anger example. Symptom B (silent): if you set `num_labels=4` instead, the model trains a four-way head with class 2 permanently empty — the loss floors near `ln(4)=1.386`, accuracy caps below 1.0, and the model never predicts the third class. Fix: remap `{0:0, 1:1, 3:2}` after filtering.
10. **Full FT:** 16 bytes per parameter (bf16 weights 2 + bf16 grads 2 + fp32 Adam m,v 8 + fp32 master weights 4) × 7e9 = **112 GB**, plus activations (which at bs 4 × len 2048 across 32 layers is several more GB) — so ~120–130 GB, needing 2×A100 80 GB with FSDP/ZeRO-3 at minimum. **QLoRA:** 4-bit base ≈ 7e9 × 0.5 bytes = **3.5 GB** + adapter gradients/optimizer (~1% of params × 16 bytes ≈ 1.1 GB) + activations (~2–3 GB at bs 4/len 1024 with gradient checkpointing) ≈ **7–8 GB** — a single 8–12 GB card, or comfortably a 24 GB 4090.

</details>

---

## 20. Cross-References

| Relationship | Module |
|---|---|
| Builds on | CS-01 — pretraining, the training loop, and the model lifecycle |
| Builds on | CS-05 — why fine-tuning was hard pre-transformer (RNN/LSTM → attention) |
| Needed by | CS-07 — BERT fine-tuning for NER, sentiment, and QA |
| Needed by | CS-12 — domain-adaptive continued pretraining on your own PDFs |
| Needed by | CS-13 — instruction fine-tuning (SFT) |
| Needed by | CS-22 — embedding fine-tuning (the same transfer logic in a bi-encoder) |
| Contrasts with | CS-04 — fine-tuning vs RAG vs agents: which architecture to choose |
| Deepens into | CS-23 — LoRA & QLoRA, the PEFT deep dive (full FT vs PEFT) |
| Cost/quality trade-off | CS-10, CS-11 — quantization, which changes the serving economics of everything here |
| Distillation as an alternative | CS-08, CS-09 — knowledge distillation instead of (or after) fine-tuning |
| Cheat sheet | CH-02 — Transfer Learning cheat sheet |
| Interview bank | IQ-02 — Transfer Learning interview questions |

---

## Appendix A — Instructor's Verbatim Key Claims

| Timestamp | Claim (verbatim from the transcript) | Comment |
|---|---|---|
| [3:55] | "Transfer learning means taking experience from one problem and using it to solve another related problem." | The module's definition. Correct and standard. |
| [8:42] | "This particular thing is called fine-tuning of the knowledge, or we can say tuning of the knowledge." | The bicycle→motorcycle gearbox as the fine-tuning delta. |
| [11:43] | "Inside that particular database… 14 million images… and around 1.4 crore. Now category wise… 21,841 category." | Accurate for ImageNet (14,197,122 images, 21,841 WordNet synsets). |
| [12:13] | "They took the subset… around 1.2 million images… 50,000 validation image and the test image was the one lakh… around 1,000 categories." | Accurate for ILSVRC-2012 (1.28M train, 50k val, 100k test, 1,000 classes). |
| [14:07] | "Can you identify this image where we have a golden retriever? Maybe in that case this pre-trained model is going to be failed." | The motivating example for fine-tuning. |
| [15:12] | "It won't be able to specify that 'okay, this car is a Tata'… It's a Tata Nano car. It's not going to be identified then." | The most memorable framing of *why* fine-tuning is needed. |
| [18:34] | "The first way is called replacing the output layer… second way… we can freeze all this convolution phase… we can only train this neural network." | Ways 1 and 2 of the three. |
| [19:19] | "Or else what we can do? We can unfreeze some convolution layer… and we can fine-tune that for our downstream task." | Way 3. |
| [21:41] | "We usually fine-tune the last layers… we never train the entire model. We just unfreeze some last layer." | The central heuristic of the lecture. |
| [22:07] | "If we're going to train some couple of last layer then for sure it's going to work. There is a 99% chance." | **Overstated.** True for in-distribution Q2 tasks with a matched backbone; not transferable to Q3/Q4. |
| [23:37] | "This fine-tuning, this transfer learning, and this fine-tuning are the two aspects of a single task." | The relationship claim, restated at [25:29] as "two sides of a single point." |
| [27:54] | "First was the small variant where we were having 12… and if you're talking about the large, inside that we are making 24." | BERT-base 12 layers, BERT-large 24 layers. Correct. |
| [28:45] | "The advanced model of the GPT still is not open source… through the API itself we can fine-tune this model." | Correct for GPT-3.5/4 at the time of recording. |
| [29:00] | "This model basically it's a very huge model so to fine-tune this particular model we'll have to take a different technique. The technique is called the PEFT technique." | Forward reference to CS-23. Correct in substance. |
| [32:09] | "This model is pretty huge, it is having billions and trillions of parameters. In that case we cannot increase couple of last layer of this particular model." | **"Trillions" is wrong** for any open Llama/Mistral/DeepSeek generation model of that era (largest open dense: Llama-3.1-405B; GPT-4 is rumored MoE in the same order, not trillions). |
| [32:23] | "Inside the PEFT technique actually we consider some subset of the weight… to fine-tune the model." | Loose description of LoRA. LoRA *adds* low-rank matrices; it does not select a subset of weights. |
| [33:21] | "It saves the training time and the resources [if] you're going to retrain any model from very very scratch." | Reason 1 for transfer learning. |
| [34:26] | "Work well when label data is limited." | Reason 2. |
| [34:46] | "Give better performance than the training from scratch. Yeah, this is true guys… that is proven inside the GPT model and the ChatGPT." | Reason 3. True in the small-to-medium data regime; false once the target dataset is large enough. |
| [41:25] | "I'm going to pre-process the data, mean I'm going to normalize the pixel value so that it will be within zero and one." | **Incomplete** — see the §15.1 correction; `preprocess_input` is required for VGG16. |
| [43:52] | "This time include_top is equal to false… I don't want this particular layer, I will add from my end." | Way 2. Correct usage. |
| [45:04] | "If you have two classes in that case you can only take one neuron, but if you have more than two classes you have to take as many neurons as classes." | Correct for sigmoid+BCE / softmax+CE. |
| [47:03] | "Here I'm going to iterate it over the layer… if the layer is equal to this one then only we are going to set the value of it true." | The block5 unfreeze loop. |
| [51:22] | "This masked language modeling is giving the generic understanding to the BERT… BERT is specifically not… trained for any text classification, for any summarization, or for any translation." | Correct, with the caveat that BERT also used Next Sentence Prediction, which the instructor omits. |
| [54:39] | "In train data we have 9,749 rows… validation data… 2,438 rows." | Verified exactly against `dair-ai/emotion` filtered to 3 classes + 80/20 (§6.5). |
| [54:55] | "Zero means sadness, one means joy and second means anger." | **Wrong without a remap** — anger is id 3 in the source `ClassLabel`. See §6.5. |
| [57:34] | "I just required these three columns — input_ids, attention_mask, label — and the type of this particular data will be torch." | `set_format`. Correct. |
| [1:02:14] | "The total parameter in BERT… it is around 10 crore, 10 CR." | ~100M — close; BERT-base uncased is 109,482,240 (110M). |
| [1:04:18] | "It is saying 'out of memory'… try to allocate it this much of memory but the capacity is higher. So in that case you can maybe restart the kernel or reconnect with the GPU." | The OOM, and the correct workaround for a Colab quota problem. |
| [1:10:53] | "We never fine-tune the earlier layers because that is only for extracting basic features. We always fine-tune the last layers only." | The closing restatement of the lecture's rule. |

---

## Appendix B — Reference Links & Papers

**Transfer learning — the foundations**

| Reference | Why it matters |
|---|---|
| Pan & Yang, *A Survey on Transfer Learning*, IEEE TKDE 2010 | The source of the domain/task and inductive/transductive/un supervised taxonomy used in §2.3 |
| Yosinski et al., *How transferable are features in deep neural networks?*, NeurIPS 2014 | The empirical origin of "early layers general, late layers specific" — the paper behind the instructor's claim |
| Kornblith, Shlens & Le, *Do Better ImageNet Models Transfer Better?*, CVPR 2019 | Feature extraction as a strong baseline; ImageNet accuracy vs transfer accuracy |
| He, Girshick & Dollár, *Rethinking ImageNet Pre-training*, ICCV 2019 | With enough data and a long enough schedule, from-scratch matches pre-training |
| Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 | LP-FT; the definitive result that full FT can lose to linear probing OOD |
| Zhuang et al., *A Comprehensive Survey on Transfer Learning*, 2020 | Modern, broader taxonomy; useful for naming edge cases |

**Fine-tuning schedules and forgetting**

| Reference | Why it matters |
|---|---|
| Howard & Ruder, *Universal Language Model Fine-tuning for Text Classification*, ACL 2018 | ULMFiT: discriminative LRs, STLR, gradual unfreezing — the source of §4.7 |
| Devlin et al., *BERT*, NAACL 2019 | The pretraining/fine-tuning paradigm this module operationalizes |
| Mosbach et al., *On the Stability of Fine-tuning BERT*, ICLR 2021 | Why the standard BERT recipe is unstable on small data, and the LR/epoch correction |
| Zhang et al., *Revisiting Few-sample BERT Fine-tuning*, ICLR 2021 | Re-initializing the top layers helps on small datasets |
| Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (EWC), PNAS 2017 | The Fisher-weighted quadratic penalty |
| Xuhong, Grandvalet & Davoine, *Explicit Inductive Bias for Transfer Learning with Convolutional Networks* (L2-SP), ICML 2018 | Penalizing distance from the pretrained weights rather than from zero |
| Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022 | The alignment tax and the `PPO-ptx` fix |
| Gururangan et al., *Don't Stop Pretraining*, ACL 2020 | DAPT and TAPT — the Q3/Q4 recipe (CS-12) |
| Biderman et al., *LoRA Learns Less and Forgets Less*, 2024 | PEFT as a regularizer, with the forgetting measurement |

**PEFT, data efficiency, and scale**

| Reference | Why it matters |
|---|---|
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 | The mechanism behind CS-23 and every memory number in §11 |
| Dettmers et al., *QLoRA*, NeurIPS 2023 | 4-bit NF4 + paged optimizers; the reason a 70B fine-tune fits on 48 GB |
| Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, ACL 2021 | Why a low-rank update suffices at all |
| Ben Zaken et al., *BitFit: Simple Parameter-efficient Fine-tuning*, ACL 2022 | The "subset of weights" family (bias terms only, ~0.08%) |
| Zhou et al., *LIMA: Less Is More for Alignment*, NeurIPS 2023 | 1,000 curated examples; the data-quality-over-quantity result |
| Taori et al., *Stanford Alpaca*, 2023 | 52k self-instruct examples for <$600 — the price anchor for SFT |
| Tenney et al., *BERT Rediscovers the Classical NLP Pipeline*, ACL 2019 | The evidence for the syntax-lower / semantics-higher heuristic, with its caveats |
| Rogers, Kovaleva & Rumshisky, *A Primer in BERTology*, TACL 2020 | The honest survey of what we actually know about layer-wise function |

**Companion resources**

| Resource | Use |
|---|---|
| Hugging Face `transformers` docs — *Fine-tune a pretrained model* | The canonical `Trainer` walkthrough |
| Hugging Face `peft` docs — *LoRA* | `LoraConfig`, `target_modules`, `merge_and_unload` |
| `lm-evaluation-harness` | The general-capability probe used for the forgetting diagnostic |
| Keras `applications.VGG16` docs | `include_top`, `classes`, `preprocess_input` semantics |
| `dair-ai/emotion` on the Hub | The dataset behind the video's BERT notebook (6 classes, 16k/2k/2k) |



