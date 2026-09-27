# CS-01 — Foundations: Pretraining, Training & the LLM Lifecycle

| Field | Value |
|---|---|
| **Module** | Foundations (entry module — defines the vocabulary every later module assumes) |
| **Source video(s)** | `LLM Fine-Tuning: 01 LLM Fine-Tuning From Scratch—Full Playlist Coming Your Way` (~20 min, syllabus) · `LLM Fine-Tuning: 02 Understanding Model Pretraining and Training in AI` (~64 min, the real foundations lecture) |
| **Transcript file(s)** | `LLM_Fine-Tuning_01_LLM_Fine-Tuning_From_ScratchFull_Playlist_Coming_Your_Way_aia.txt`, `LLM_Fine-Tuning_02_Understanding_Model_Pretraining_and_Training_in_AI_aiagents_f.txt` |
| **Companion code** | `LLM Fine-Tuning-01\LLM Finetuning  01 syllabus.pdf`, `LLM Fine-Tuning-02\LLM Finetuning  02 Introduction of Finetuning.pdf` — both are **image-only screenshots of the instructor's OneNote** (no text layer, not machine-readable). The three demos he runs on camera are reconstructed and modernised in §6. |
| **Prerequisites** | None. This is CS-01. |
| **Difficulty** | Beginner for the map, Intermediate for the arithmetic (§4.2, §11) |
| **Hands-on required** | Yes — §6 ships three runnable scripts (closed-label CNN inference, masked-LM probing, tokenizer forensics) |
| **Estimated study time** | 4h theory + 2h practical |

---

## 0. Executive Summary

- **Fine-tuning is not a technique, it is the second half of a lifecycle.** You cannot fine-tune what you do not have, and what you have is a *pretrained* foundation model. The single sentence that unlocks the whole discipline: **pretraining teaches the model the language; fine-tuning teaches the model the job.**
- The instructor's own framing — *"finetuning is just not a finetuning, just not a training the model... it is more than that"* [4:09] — is correct and is the thesis of this module. Fine-tuning is a stage in a pipeline (`data → tokenizer → pretrain → SFT → preference alignment → eval → deploy`), not an isolated operation.
- **The lifecycle has exactly one direction of information flow you must respect:** data quality set at the bottom determines the ceiling at the top. No amount of SFT fixes pretraining that never saw your domain; no amount of preference alignment fixes SFT that was trained on the wrong chat template.
- **The 6ND rule is the most useful formula in this entire handbook.** Training compute FLOPs ≈ `6 × params × tokens`. Inference is `2 × params` FLOPs per token.
- **The Chinchilla rule is the second:** compute-optimal training wants **~20 tokens per parameter**. A 7B model wants ~140B tokens. Almost nobody does this anymore for production models — modern models are deliberately *overtrained* (Llama-3 8B: ~1,875 tokens/param) because **inference cost dominates training cost over the model's life**.
- **Memory beats FLOPs in most real workloads.** Decode is memory-bandwidth bound: a 7B model in bf16 on a 3.35 TB/s H100 has a hard ceiling of ~240 tokens/s for a batch of 1, no matter how many FLOPs the chip has.
- **Full fine-tuning of 7B needs ~112 GB just for weights, gradients and Adam state** — before activations. That number is why LoRA/QLoRA (CS-13 §6.8, CS-11 §4.11) exist, and why "fine-tune a 7B on a free Colab T4" is a QLoRA claim, never a full-FT claim.
- **The base-vs-instruct distinction is the most consequential and most misunderstood thing in the field.** A base model is a text continuer that has never been taught to answer. Evaluating a base model on a multiple-choice benchmark and concluding "it is stupid" is the field's most common self-inflicted error.
- **Tokenizers are the #1 source of silent, catastrophic bugs** in fine-tuning: wrong chat template, missing pad token, resized embeddings, truncation that eats the label. Four of the twelve debug entries in §14 are tokenizer bugs.
- **The most important thing fine-tuning cannot do: install new facts.** Fine-tuning teaches *form* (style, format, schema, refusal behaviour, domain vocabulary, tone). RAG supplies *substance* (current, verifiable, citable facts). Choosing wrong between them is CS-04.

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world without this mental model

Engineers who skip this module fail in four predictable, expensive ways:

1. **They fine-tune to add knowledge.** They feed 5,000 PDF chunks into an SFT run, the model learns the *prose style* of the documents and starts inventing plausible-sounding facts in that style. The hallucination rate goes **up**, not down. They needed RAG (CS-04), and they spent $400 and three weeks learning it.
2. **They fine-tune the wrong checkpoint.** They grab `meta-llama/Llama-3.1-8B` (base) instead of `...-Instruct`, train on 2,000 instruction pairs, and get a model that answers questions *and then keeps generating three more fake questions*. The chat template was never applied because base models do not ship one. See §10.3.
3. **They size the job from the parameter count instead of from bytes-per-parameter.** "It's only 7B, it'll fit on my 24 GB card" is true for inference in 4-bit, false for LoRA training at seq 4096, and catastrophically false for full FT (112 GB + activations).
4. **They compare their model to GPT-4 on the wrong axis.** They report "our fine-tuned 8B beats GPT-4 on our eval" when the eval was 200 examples of the same distribution they trained on, with no held-out split and no contamination check. The model shipped; the regression showed up in production six weeks later.

### 1.2 The state of the art **before** fine-tuning, and why it failed

Before transfer learning, every NLP task meant training a network from scratch on task-specific labels. The instructor's board walks through exactly this: *"I'm doing a task specific training... if I want to do text classification then I'll train a model on top of it. Then question answering — if I want to do question answering then I will train my model again from scratch"* [17:23–17:41]. He labels the old regime **"training from various branch"** and **"task-specific training"** [17:14–17:25], and the failure mode is structural:

| Property | From-scratch, task-specific (pre-2018 NLP) | Pretrain → fine-tune (post-2018) |
|---|---|---|
| Labels needed | 10k–1M task-labeled examples | 100–5,000 task examples |
| Data required | Task data only — but a *lot* of it | Unlabeled general text (free) + a little task data |
| Reuse across tasks | Zero. New task = new model | One backbone, N task heads |
| Compute per new task | Full training run | A fraction of a run |
| Ceiling on small datasets | Low — overfits | High — pretraining supplied the representation |

The instructor gives the *raison d'être* for the whole paradigm as an interview answer at [42:28]: *"Why use this pre-trained model? Deep learning models require a massive amount of data... we cannot train it again and again from scratch... there will be so many computational resources associated... it will take a very huge time, effort."* The pretrained model is a **shortcut**: someone else paid the compute, and the community reuses it.

### 1.3 The naive approach and precisely why it fails

The naive approach is *"just train on my data."* It fails for a reason that is measurable, not philosophical: **your data is too small to specify the function you want.**

Work the numbers. Suppose your task is 2,000 labeled examples of "classify this support ticket into 14 categories." A from-scratch transformer-base model (~110M params) has roughly `110e6` free parameters. With 2,000 examples you have ~1,400 training examples after a 70/15/15 split. That is ~13 examples per thousand parameters. The model will reach ~100% train accuracy and ~40% validation accuracy — it has memorised the training set and learned nothing generalisable, because the gradient signal from 1,400 examples cannot constrain a 110M-dimensional function space.

Now take `bert-base-uncased` (110M params), pretrained on ~3.3B words of BooksCorpus + English Wikipedia. Fine-tune the same 1,400 examples for 3 epochs at `lr=2e-5`. On a 14-class ticket task, typical F1 lands in the 0.80–0.88 range. **Same architecture, same data, same compute budget for the fine-tune — 2x the accuracy**, because 110M parameters arrived already carrying a representation of English, and 1,400 examples were only needed to *point that representation at your 14 classes*.

That gap — 40% vs 85% from the same amount of labeled data — is the entire justification for the field.

### 1.4 Concrete motivating example: the two images that explain out-of-distribution

The instructor runs two CNN inferences and the second one is the best teaching moment in either video. He loads a ResNet with ImageNet weights, feeds a dog photo, and gets `Labrador retriever > golden retriever` [1:00:53–1:01:03]. Then he feeds a **tomato** photo using the same ImageNet weights and gets *"strawberry, pitcher, orange"* [1:01:40–1:01:47] — and notices, correctly, that `tomato` is simply not in the label set: *"tomato will not be there inside the particular data. That's why it is not coming"* [1:01:47].

This is the closed-label-set failure mode, and it is *exactly* the failure mode of an LLM asked about a topic outside its pretraining distribution — except the LLM hides it in fluent prose instead of returning a silly label. The lesson: **a pretrained model does not say "I don't know"; it returns its nearest neighbour in the space it was trained on.** Everything downstream (RAG, fine-tuning, evals, guardrails) exists to manage that fact.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: a self-taught reader, then an apprenticeship

The instructor's own analogy is at [57:05–57:19]: pretraining is *"a self-taught student"* — it reads everything, teaches itself, then it is *"time to predict"*, and only later does it specialise. Extend it into the full lifecycle:

| Stage | Real-world analogue | What changes in the model |
|---|---|---|
| Pretraining | A child reading the entire internet for 12 years, no teacher, no labels | All weights. Raw language statistics, grammar, world co-occurrence, some reasoning circuits |
| SFT / instruction tuning | Apprenticeship: a human demonstrates "when asked X, answer like Y" | All weights (or adapters). Turns the continuer into a responder |
| Preference alignment (RLHF/DPO) | A supervisor ranking two attempts and saying "more like A, less like B" | All weights (or adapters). Reshapes the *distribution over* responses, not the knowledge |
| RAG | Handing the apprentice the correct manual at the moment of the question | **Nothing in the weights.** The context window changes |
| Prompt engineering | Telling the apprentice "answer in 3 bullets, formal tone" | **Nothing.** The instruction text changes |
| Inference | The apprentice answering the question | Nothing. Only activations are computed |

**Where this analogy breaks.** A human apprentice generalises from a handful of demonstrations and can *refuse* to answer from a manual they haven't read. An LLM (a) needs 1,000–100,000 demonstrations where a human needs three, and (b) has no introspective access to where a fact came from — so the "did I read that manual?" boundary does not exist inside the model. That is precisely why RAG still hallucinates and why citation-grounded prompts matter. The analogy also breaks in the other direction: fine-tuning *can* install facts (a model can memorise a fixed FAQ), it is just an expensive and unreliable way to do it — the fact becomes unremovable, stale and unauditable.

### 2.2 The actual mechanism

Strip away every framework and the mechanism is one loop, which the instructor draws explicitly for the LLM case at [53:32–53:45]: **collect data → tokenize → pass through transformer → predict next token → calculate loss → backpropagation → repeat.**

Formally, training is:

```
θ ← θ - η · ∇θ L(f(x; θ), y)
```

performed several hundred thousand times, where `θ` is the full parameter set, `x` is a batch of token sequences, `y` is the *same* sequences shifted one position left, `L` is cross-entropy, and `η` is a schedule rather than a constant. That is it. Every framework in this handbook (HF Trainer, Unsloth, Axolotl, LLaMA-Factory, TRL, PEFT, OpenAI's API) is a convenience wrapper around that one line plus the systems engineering needed to run it on hardware with finite memory.

The reason it works at all is that next-token prediction is a *surprisingly hard* objective. To predict the next token well across a trillion tokens you must implicitly learn syntax, morphology, entities, arithmetic patterns, translation, code semantics, and some amount of world modelling — because all of those reduce next-token entropy. The instructor states the intuition directly: pretraining lets the model *"understand the grammar inside the data... the word semantic relationship... the crux of the language, and because of that it's going to work like magic"* [57:22–57:40].

---

## 3. Core Concepts — Exhaustive Glossary

Later modules (CS-02 through CS-19) assume every term below. This is the reference table for the whole handbook. (CS-20–CS-28 are planned but were never written; where a row names one, the nearest *written* material is given inline instead.)

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **AI / ML / DL** | Nesting sets: AI ⊃ machine learning ⊃ deep learning (learning via neural networks) | The instructor's whole board map [8:09–8:35]; orients you on where fine-tuning lives | Treating "AI" and "LLM" as synonyms |
| **ANN** | Artificial neural network — fully connected layers | The base unit all other architectures compose | Confused with "artificial general intelligence" |
| **CNN** | Convolutional neural network — grid/image data. Where fine-tuning was *born* (2011–2012) | The historical origin of transfer learning (§5.1) | Assuming fine-tuning started with LLMs. It did not |
| **RNN** | Recurrent neural network — sequential state, one step at a time. Dominant 2015–2018 for NLP | The instructor's "why fine-tuning was hard" module (CS-05) | Believing the Transformer is a variant of RNN |
| **LSTM / GRU** | Gated RNN variants that mitigate vanishing gradients | Sequentially dependent ⇒ poor long-range recall and no parallelism | "LSTMs can't do long context" — they can, badly and slowly |
| **Transformer** | Architecture using self-attention with no recurrence. Vaswani et al., 2017 | The substrate of every model in this handbook | Called a "variant of RNN" — it replaced recurrence entirely |
| **Self-attention** | Each token computes a weighted sum over all tokens; weights from Q·Kᵀ scaled by 1/√d, applied to V | Enables parallelism and O(1) path length between any two tokens | Confused with *cross*-attention (query from decoder, keys/values from encoder) |
| **Attention (origin)** | Bahdanau et al., 2014, "Neural machine translation by jointly learning to align and translate" | The paper the instructor names as the source of the concept [27:47–28:03] | Crediting attention to "Attention Is All You Need" — that paper *replaced* recurrence, it did not invent attention |
| **Foundation model** | The pretrained model itself. Instructor: *"this pre-trained model only it is called the foundation model"* [36:04–36:08] | The artefact you fine-tune | Thinking "foundation model" means a special architecture |
| **Pretrained model** | Synonym for foundation model in this course: trained on huge general data, before any task specialisation | The prerequisite for fine-tuning. No pretrained model ⇒ no fine-tuning | Believing any saved checkpoint is "pretrained" |
| **Base model** | A raw pretrained checkpoint, no instruction tuning. `Llama-3.1-8B` | Text continuer, not an assistant | Using it as a chat model |
| **Instruct model** | Base + SFT (+ alignment) + a chat template. `Llama-3.1-8B-Instruct` | The correct default starting point for most SFT work | Assuming it fine-tunes identically to base |
| **Pretraining** | The first, largest training stage: next-token prediction over trillions of tokens of general text | Where ~99% of the model's capability and ~99% of its compute live | Treated as a synonym for "training" |
| **Unsupervised pretraining** | Legacy label for pretraining because there is no human-provided `y` | Historical name only | The instructor's correction: the right name is self-supervised [55:23] |
| **Self-supervised learning** | Labels are derived *from the data itself* (token *t+1* is the label for tokens *1..t*) | The reason internet-scale text is usable as training data at all | Believing it means "no labels" |
| **Causal LM (CLM)** | Autoregressive objective: predict token *t+1* from tokens *≤t*. Decoder-only. GPT, Llama, Mistral, Qwen, Gemma | The objective of every modern generative LLM | Called "regressive modelling" in the transcript [31:03] — the term is **autoregressive** |
| **Masked LM (MLM)** | Mask ~15% of tokens, predict them from both sides. Encoder-only. BERT | Produces strong bidirectional *understanding* representations | "MLM is obsolete" — false for classification/NER/embedding (CS-07; embedding FT is `code/10_embedding_finetune.py`) |
| **Span corruption / span masking** | Mask contiguous spans, decoder reconstructs them with sentinel tokens. T5, UL2 | Encoder-decoder objective; efficient and strong for seq2seq | Confused with MLM (which masks individual tokens) |
| **Seq2seq** | Sequence in → sequence out, via encoder + decoder. T5, BART, original translation models | The right shape for translation, summarisation, structured extraction | Assumed dead — T5-family and encoder-decoder models still ship |
| **Autoregressive** | Generating one token at a time, feeding output back as input | The inference-time behaviour of every decoder-only LLM | Confused with "recursive"; also confused with the *training* objective (they match in CLM, which is the point) |
| **Teacher forcing** | During training, feed the *ground-truth* previous tokens, not the model's own predictions | Makes training parallel and stable | Confused with inference behaviour (exposure bias) |
| **Token** | Sub-word unit produced by the tokenizer; the atomic input/output unit | All costs, context limits and VRAM numbers are per-token | Confused with "word": ~1 token ≈ 0.75 English words |
| **Tokenizer** | The `text ↔ integer ids` map. BPE, WordPiece, Unigram (SentencePiece), byte-level BPE | Must match the model. Mismatch = garbage output | Swapping tokenizers between models |
| **Vocabulary size** | Number of distinct token ids. BERT 30,522 · GPT-2 50,257 · Llama-2 32,000 · Llama-3 128,256 · Qwen2 151,646 · Gemma 256,000 | Trades sequence length against embedding params and rare-token coverage | Bigger assumed strictly better — it costs params and softmax compute |
| **Special tokens** | Reserved ids: `[CLS] [SEP] [MASK] <s> </s> <|endoftext|> <|im_start|>` | Control structure; wrong ones silently break training | Treating them as literal text that will be tokenized normally |
| **Chat template** | The exact string format wrapping system/user/assistant turns for a given model | Wrong template = the #1 silent SFT failure | Assuming all models use ChatML |
| **Embedding** | Dense vector for a token, or for a whole text. `Embedding = a set of numbers = a vector` [17:33–17:42] | Input layer of the model; also the retrieval primitive in RAG (embedding FT: `code/10_embedding_finetune.py`) | Confusing token embeddings (inside the model) with sentence embeddings (retrieval) |
| **Logits** | Raw unnormalised scores over the vocabulary before softmax | Where temperature, top-k, top-p act | Confused with probabilities |
| **Softmax** | `exp(z_i)/Σ exp(z_j)` — turns logits into a distribution | The final layer of the LM head; the instructor's BERT demo does *"softmax over vocabulary, take top prediction"* [1:03:02] | Numerically unstable without max-subtraction |
| **Loss (cross-entropy)** | `-log p(correct token)` averaged over tokens; the training objective | The only number you actually steer training by | Confused with accuracy — a model can have low loss and be useless |
| **Perplexity** | `exp(cross-entropy loss)`. "How surprised is the model, in effective choices" | The standard pretraining metric the instructor names [35:16–35:18] | Reported across *different tokenizers* — not comparable |
| **Token accuracy** | Fraction of positions where argmax prediction = true next token | Second pretraining metric named at [35:16]. Typical 0.6–0.75 on general text | Read as "the model is right 70% of the time" — it is 70% at the *token* level, not the answer level |
| **Hyperparameter** | Set by you: LR, batch size, LoRA rank, epochs, warmup | Controls *how* training proceeds | Confused with parameters |
| **Parameter** | Learned weight/bias. 7B model = 7e9 of them | Sets VRAM, compute and capability | Confused with hyperparameter |
| **Activation** | Intermediate tensor produced in the forward pass and kept for the backward pass | Usually the largest *variable* memory term | Assumed to be small — it is not (see §4.4) |
| **Gradient** | ∂L/∂θ — the direction and magnitude to nudge each parameter | What the optimiser consumes | Confused with the optimiser state |
| **Backpropagation** | Reverse-mode automatic differentiation: compute ∂L/∂θ for all θ in one backward sweep | Makes training 7e9-parameter models feasible at all | Confused with the optimiser step (separate) |
| **Optimiser** | Rule that turns gradients into parameter updates. Adam/AdamW dominant | Adam adds 2 fp32 states per parameter — the dominant VRAM cost | Confused with the LR schedule |
| **Adam / AdamW** | Adam with *decoupled* weight decay. Default for LLM training | `betas=(0.9, 0.95)` typical for LLMs; `eps=1e-8` | Using the PyTorch default `betas=(0.9,0.999)` — works, but LLM practice differs |
| **Epoch** | One full pass over the training set | For SFT, 2–3 is typical; >3 overfits fast | Carried over from classical ML where 50 epochs is normal |
| **Step / iteration** | One optimiser update on one micro-batch (or an accumulated group) | The unit of the LR schedule | Counted as "one forward/backward" in gradient-accumulation discussions |
| **Batch size** | Examples per step. *Micro-batch*: per GPU, per forward. *Global*: micro × grad-accum × GPUs | Global batch controls optimisation quality; micro-batch is what must fit in VRAM | Confusing the two when reading a config |
| **Gradient accumulation** | Run N micro-batches, sum gradients, then step once | Buys large effective batch on small VRAM at ~equal compute cost | Believed to be free — it costs N× the forward/backward wall clock of one step, and interacts with LR |
| **Gradient checkpointing** | Recompute activations during backward instead of storing them | Cuts activation memory ~5–10x for ~20–33% extra compute | Assumed to be a speed optimisation — it is a memory optimisation |
| **Learning rate (LR)** | Step size `η` | The single most important hyperparameter | Set too high from classical-ML habits: 1e-3 kills an LLM fine-tune |
| **LR schedule** | LR as a function of step: warmup → (cosine | linear) decay | Warmup prevents early divergence; decay improves final loss | Using a constant LR |
| **Warmup** | LR ramps from ~0 to peak over the first N steps | Essential for Adam; typically 3–10% of total steps | Skipping it |
| **Weight decay** | L2-style shrinkage, decoupled in AdamW. 0.01–0.1 for LLMs | Regularisation | Applied to LayerNorm/bias — should be excluded |
| **Mixed precision** | bf16/fp16 compute with fp32 master weights | ~2x speed and ~half the memory | fp16 without loss scaling → NaNs. bf16 preferred on A100+ |
| **bf16 vs fp16** | bf16: 8 exponent bits, 7 mantissa, same range as fp32. fp16: 5 exponent, 10 mantissa, overflows | bf16 is the default for training on Ampere+; fp16 still common for inference | Assuming fp16 has more "precision" so it is safer — the opposite is true for stability |
| **Quantization** | Store/compute in fewer bits. int8, int4/NF4, FP8 | Cuts VRAM 2–8x; enables 70B on one GPU (CS-10, CS-11) | Believed free — it costs accuracy, and QLoRA pays it only in the frozen base |
| **VRAM** | GPU memory. The binding constraint on everything | Determines which technique is even possible | Sized from parameter count alone |
| **FLOPs** | Floating-point operations. `6ND` train, `2N` per token inference | Converts a plan into a time and dollar estimate | Confused with FLOPS (per second) |
| **TFLOPs** | Tera (1e12) FLOPs, or FLOPs/s for a chip | The numerator in any time estimate | Peak vs achieved confused |
| **MFU** | Model FLOPs Utilisation = achieved ÷ peak. 35–50% is good | Turns peak TFLOPs into realistic TFLOPs | Quoting peak TFLOPs as throughput |
| **Memory bandwidth** | Bytes/second the GPU can move from HBM. A100 2.0 TB/s, H100 3.35 TB/s | **Bounds decode.** Time/token ≈ weight bytes ÷ bandwidth | Ignored entirely by people who only count FLOPs |
| **Context window** | Max tokens the model attends over (4k → 128k → 1M) | Sets what RAG can pack in and how much activation memory you need | Assumed to be usable at full length — quality degrades well before the limit |
| **Scaling laws** | Predictable power-law relations between loss, params, data, compute. Kaplan 2020; Chinchilla 2022 | Tell you how to spend a fixed compute budget | Kaplan's exponents used as current truth — Chinchilla revised them |
| **Chinchilla-optimal** | ~20 tokens per parameter for a fixed training-compute budget | The reference point for "is my model undertrained?" | Applied to inference-cost-sensitive deployments, where overtraining is correct |
| **Emergent ability** | A capability that appears only above some scale (Wei et al. 2022) | Used to justify scale | Contested as a metric artefact (Schaeffer et al. 2023) — see §4.6 |
| **SFT** | Supervised fine-tuning on (instruction, response) pairs | The stage that turns a base model into an assistant (CS-13) | Called "instruction tuning" and "fine-tuning" interchangeably |
| **Instruction tuning** | Synonym for SFT with instruction-formatted data | Same | Same |
| **PEFT** | Parameter-efficient fine-tuning: train <1% of params. LoRA, QLoRA, DoRA, adapters, prefix tuning | Makes fine-tuning affordable (CS-13 §6.8, CS-11 §4.11) | Believed to always match full FT — it does not, at high data volumes |
| **LoRA** | Freeze base; learn low-rank `BA` updates, `ΔW = BA` with rank `r` | 100x fewer trainable params, ~3x less VRAM | Believed to reduce compute. It reduces *memory* |
| **QLoRA** | LoRA on top of a 4-bit NF4 quantized frozen base | 7B on 12 GB, 70B on 48 GB | Believed to be ~2x slower than expected — it is ~20–40% slower than bf16 LoRA |
| **Full fine-tuning** | Update every parameter | Max quality/plasticity; needs ~16 bytes/param plus activations | Tried first by beginners on hardware that cannot hold it |
| **Catastrophic forgetting** | Fine-tuning degrades capabilities the base had | Why you must run regression evals, not just task evals | Believed avoidable with a low LR — mitigable, not avoidable |
| **Alignment collapse** | Narrow fine-tuning strips safety behaviour. Qi et al. 2023: 10 adversarial examples, ~$0.20 | A safety incident, not a quality issue | Assumed to require a big dataset |
| **Transfer learning** | Reusing representation learned on task A for task B. Instructor: *"Transfer learning is nothing, it is just a way to perform a fine tuning"* [41:44–41:46] | The umbrella concept | Confused with fine-tuning — fine-tuning is the *mechanism*, transfer is the *goal* |
| **Feature extraction** | Freeze everything, train a new head. Instructor's "way 1": change only the last layer [40:28–40:33] | Cheapest, least plastic | Believed to be a form of fine-tuning; it is a degenerate case |
| **Partial fine-tuning** | Freeze early layers, train late layers. Instructor's "way 2" [40:38–40:52] | Middle ground: less compute, more plasticity than a head swap | Assumed always worse than full FT — often it is better on small data |
| **RAG** | Retrieval-augmented generation: fetch documents, put them in context | Injects facts without touching weights (CS-04) | Believed to be a fine-tuning alternative rather than a complement |
| **RLHF** | Reinforcement learning from human feedback: reward model + PPO | ChatGPT's alignment recipe (CS-14 §4.6.1) | Confused with SFT; also believed to add knowledge |
| **DPO** | Direct Preference Optimization — closed-form preference objective, no reward model, no RL loop (CS-14 §4.6.3) | Now the default preference method | Called "direct reference optimization" in the transcript [37:01] — it is **Direct Preference Optimization** |
| **PPO** | Proximal Policy Optimization — the RL algorithm used inside RLHF | Requires 4 models in memory (policy, ref, reward, value) | Confused with DPO as "the same thing" |
| **RLHF vs DPO** | RLHF = online RL with a reward model; DPO = offline, direct on preference pairs | DPO is simpler, cheaper, and now usually preferred | Believed to be strictly better — DPO can overfit preferences; see CS-14 §4.6.3 |
| **Hallucination** | Fluent, confident, wrong output | The failure mode everything else is defending against | Believed fixable by fine-tuning |
| **Contamination** | Test-set leakage into training data | Invalidates benchmark numbers | Checked rarely; assume it is present |
| **Dedup** | Removing near-duplicate training documents | Improves loss at fixed compute, reduces memorisation | Believed cosmetic — it is one of the highest-ROI data operations |
| **Data quality vs quantity** | LIMA (1,000 examples), Phi ("textbooks are all you need") | Small, curated data can beat large, noisy data | Read as "data volume doesn't matter" — it does, for pretraining |
| **Benchmark** | Standardised eval: MMLU, GSM8K, HumanEval, GLUE, SST-2, MMLU-Pro | Comparability across models | Treated as ground truth; almost all are contaminated and format-sensitive |
| **Zero-shot / few-shot** | Task described with 0 / a handful of examples in the prompt | The prompting baseline you must beat before fine-tuning | Skipped, which is how teams fine-tune to solve a prompting problem |
| **ICL** | In-context learning — learning a task from prompt examples, no weight change | An emergent-ish property of scale | Confused with fine-tuning |
| **Catastrophic forgetting of format** | The model stops following the old chat template after SFT on a new one | Breaks the app, not the benchmark | Discovered in production |
| **Inference** | Forward pass only, no gradients, no optimiser state | Memory = weights + KV cache, not weights + grads + Adam | Sized with the training formula |
| **KV cache** | Stored keys/values for generated tokens, reused each step | Grows linearly with context and batch; the main serving memory cost | Forgotten in serving capacity planning |
| **TTFT / TPS** | Time to first token (prefill, compute-bound) / tokens per second (decode, bandwidth-bound) | The two halves of serving latency | Reported as one number |
| **Checkpoint** | Serialised model state saved during/after training | Rollback, resume, and the thing you actually ship | Confused with the adapter |
| **Adapter** | LoRA-style small weight set saved separately from the base | Portable, hot-swappable; must be merged or loaded with the base | Believed to be a standalone model |
| **Merge (of adapters)** | `W' = W + BA`, folding the adapter into the base weights | Faster inference, no PEFT dependency | Assumed lossless — it is lossless only for plain LoRA at the same dtype |
| **Seed** | RNG seed governing init, shuffling, dropout | Reproducibility | Believed to make runs bit-identical on GPU — it does not |

---

## 4. Deep Dive — How It Actually Works

### 4.1 Mechanism, step by step: the seven stages of the lifecycle

The instructor's board presents the sequence twice — once for the CNN era [19:40–20:33] and once for LLMs [34:54–35:44, 37:20–37:40]. Merged and completed, the lifecycle is:

```
                                                          ┌─────────────────────┐
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │ STAGE 1: DATA       │
  │ 1. COLLECT   │──▶│ 2. CLEAN &   │──▶│ 3. TOKENIZER │  │ COLLECTION          │
  │ raw text     │   │    FILTER    │   │    TRAINING  │  │ Common Crawl, Wiki, │
  │ (trillions)  │   │ dedup, PII,  │   │ BPE/Unigram  │  │ books, GitHub, news │
  └──────────────┘   │ quality, lic.│   │ vocab 32k-256k│ └─────────────────────┘
                     └──────────────┘   └──────┬───────┘
                                               │
                                               ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STAGE 4: PRETRAINING  (self-supervised, ~99% of total compute)              │
  │   loop: tokenize → forward through transformer → predict next token →       │
  │         cross-entropy loss → backprop → optimizer step → repeat             │
  │   objective: causal LM (GPT/Llama/Qwen) | MLM (BERT) | span (T5)            │
  │   in: 1T–15T tokens          out: FOUNDATION / BASE MODEL                   │
  └──────────────────────────────────┬──────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  use as-is (few-shot /      STAGE 5: SFT                 STAGE 6: PREFERENCE
  ICL — zero weight          (instruction tuning)          ALIGNMENT
  change)                    on 1k–1M (prompt, response)   RLHF/PPO · DPO · ORPO
                             pairs. LoRA or full FT.       GRPO  (CS-14, 24-27)
                                     │                            │
                                     ▼                            ▼
                             INSTRUCT MODEL  ─────────────▶  ALIGNED INSTRUCT MODEL
                                                                  │
                                                                  ▼
                                            ┌─────────────────────────────────────┐
                                            │ STAGE 7: EVAL & DEPLOY              │
                                            │ held-out task eval · regression     │
                                            │ suite · safety eval · LLM-judge ·   │
                                            │ human · quantize · serve · monitor  │
                                            └─────────────────────────────────────┘
```

| Stage | Input | Operation | Output | Failure mode if skipped or botched |
|---|---|---|---|---|
| 1. Collect | Crawls, licensed corpora, code, books | Source selection, ToS/robots compliance | Raw text shards | Legal exposure; domain gap the model can never close |
| 2. Clean & filter | Raw shards | Dedup (exact + MinHash/LSH), quality classifier, PII scrub, language ID, benchmark decontamination | Clean token-ready text | Memorised duplicates; contaminated evals; worse loss at equal compute |
| 3. Tokenizer | Clean text | Learn BPE/Unigram merges; reserve special tokens | `tokenizer.json`, `vocab.json` | Tokenizer/model mismatch = garbage. Non-English text costs 3–10x more tokens |
| 4. Pretrain | Token stream | 6ND FLOPs of next-token prediction | **Base / foundation model** | Undertrained (too few tokens) or wasteful (too many params for the token budget) |
| 5. SFT | (instruction, response) pairs | Cross-entropy on response tokens only | **Instruct model** | Wrong chat template, loss on prompt tokens, over-fitting to 3 epochs of 500 examples |
| 6. Align | Preference pairs or reward model | RLHF/DPO/ORPO/GRPO | Aligned model | Over-optimisation: safe but hedgy, or sycophantic. Reward hacking |
| 7. Eval & deploy | Model + eval sets | Quantize, serve, monitor, A/B | Production endpoint | Silent regression, drift, latency blowout, no rollback path |

**Where fine-tuning sits:** stages 5 and 6, plus a special case of stage 4 — *continued pretraining* on domain text (CS-12), which is stage 4 repeated on a narrower corpus.

### 4.2 The mathematics, with a fully worked numeric example

**Notation.** `x = (x₁, …, x_T)` is a token-id sequence. `θ` is the parameter set. `V` is vocabulary size. `h` is the model (hidden) dimension. `L` is the number of transformer blocks. `d_head = h / n_heads`. `D` is total training tokens. `N` is parameter count.

**Causal LM objective.** The model defines `p(x_t | x_1..x_{t−1}; θ)`. Training maximises the log-likelihood of the observed corpus, equivalently minimises the mean negative log-likelihood:

```
L(θ) = − (1/T) Σ_{t=1..T} log p(x_t | x_<t ; θ)

p(x_t | x_<t) = softmax(z_t)_x_t     where  z_t = W_lm · h_t^(L) + b_lm ,  z_t ∈ R^V
```

The instructor states this objective in plain words and draws it on the board: *"We are training our model for next prediction... on top of this next word prediction, my model is trying to learn... trying to understand the language, trying to understand each and every pattern"* [32:51–33:22].

**The insight the board example makes visible.** His example sentence is *"Sunny is AI master"* [56:01–56:10]. He shows that the "label" for the first two words is taken from the data itself — *"we'll keep this last two word and we'll keep this particular word as a target variable"* [56:20–56:25]. Here is the full supervision map, which is the single most important mechanical fact about CLM training:

| Position *t* | Input seen (x_<t) | Target x_t | Supervision? |
|---|---|---|---|
| 1 | `[BOS]` | `Sunny` | Yes |
| 2 | `[BOS] Sunny` | `is` | Yes |
| 3 | `[BOS] Sunny is` | `AI` | Yes |
| 4 | `[BOS] Sunny is AI` | `master` | Yes |

**A 4-token sequence yields 4 supervised predictions, not 1.** Every position is a training signal. This is why causal LM is more sample-efficient per token than MLM, which supervises only the ~15% of positions that were masked:

```
Causal LM supervision  : 100% of positions
MLM  supervision       : ~15% of positions (the masked ones)
```

**Worked example — forward pass, loss, and perplexity.** Vocabulary `V = {Sunny, is, AI, master, the, …}` of size 32,000. Say the model produces these probabilities for the three *new* predictions (positions 2–4):

| Predicted token | Model probability p(correct) | Cross-entropy `−ln p` |
|---|---|---|
| `is` | 0.60 | 0.5108 |
| `AI` | 0.35 | 1.0498 |
| `master` | 0.12 | 2.1203 |

```
mean loss = (0.5108 + 1.0498 + 2.1203) / 3 = 1.2270
perplexity = exp(1.2270) = 3.41
```

Interpretation: the model is behaving like a system choosing between ~3.4 equally likely options per token. For calibration:

| Model state | Loss | Perplexity |
|---|---|---|
| Uniform random over 32,000-token vocab (Llama-2 tokenizer) | `ln 32000 = 10.373` | 32,000 |
| Uniform random over 128,256-token vocab (Llama-3 tokenizer) | `ln 128256 = 11.762` | 128,256 |
| A weak LM | 3.0 | 20.1 |
| A good general LM on Wikipedia | 2.2–2.6 | 9.0–13.5 |
| Strong modern LM on curated web text | 1.7–2.0 | 5.5–7.4 |

> **Watch out:** perplexity is computed in the tokenizer's units. A perplexity of 6.0 on a 128k vocabulary is *not* comparable to 6.0 on a 32k vocabulary — the larger vocabulary has more ways to be wrong. Never compare perplexity across tokenizers, and never compare it across different test sets.

**Why cross-entropy and not accuracy.** Cross-entropy is differentiable and penalises confident wrong answers enormously (p = 0.001 → loss 6.9) while giving diminishing reward for being more confident once correct (0.9 → 0.105, 0.99 → 0.010). Accuracy is piecewise-constant, so its gradient is zero almost everywhere. You cannot backprop through accuracy.

**The gradient of cross-entropy has a famously clean form.** For logits `z` and one-hot target `y`:

```
∂L/∂z = softmax(z) − y
```

Worked: `z = [2.0, 1.0, 0.1]` for `[the, cat, sat]`, true token `cat` (index 1).
```
softmax = [0.659, 0.242, 0.099]
∂L/∂z  = [0.659, 0.242 − 1, 0.099] = [0.659, −0.758, 0.099]     (sums to ~0)
```
Read it as an instruction: *"raise the logit of `cat` by 0.758, lower `the` by 0.659, leave `sat` alone."* Every LLM ever trained is doing this, at 32,000-way rather than 3-way scale, across 10¹² tokens.

**The optimiser step.** AdamW:

```
m_t = β₁·m_{t−1} + (1−β₁)·g_t                  # 1st moment  (β₁ = 0.9)
v_t = β₂·v_{t−1} + (1−β₂)·g_t²                 # 2nd moment  (β₂ = 0.95 for LLMs)
m̂_t = m_t / (1 − β₁^t)                          # bias correction
v̂_t = v_t / (1 − β₂^t)
θ_t = θ_{t−1} − η_t · ( m̂_t / (√v̂_t + ε) + λ·θ_{t−1} )   # ε = 1e-8, λ = weight decay
```

The `m` and `v` tensors are the **two extra fp32 states per parameter** that dominate memory (§4.4). The division by `√v̂` is why Adam is scale-free per-coordinate and why a learning rate of 2e-5 works for a 7B model but 2e-5 would be absurd for logistic regression.

**Worked single-parameter update.** `w = 0.500`, `g = 0.50`, `η = 0.1`, `λ = 0.01`:
```
θ_new = 0.500 − 0.1 · (0.50 + 0.01 · 0.500)
      = 0.500 − 0.1 · 0.5050
      = 0.500 − 0.0505
      = 0.4495
```

**Step, batch, epoch, gradient accumulation — worked numbers.**

Config: 4 × A100-80GB, `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8`, `max_seq_length = 2048`, dataset = 100,000 examples averaging 1,024 tokens.

```
micro-batch tokens      = 2 × 2048            = 4,096 tokens
step tokens (global)    = 4,096 × 8 × 4 GPUs  = 131,072 tokens
dataset tokens (1 epoch)= 100,000 × 1,024     = 102,400,000 tokens
steps per epoch         = 102,400,000 / 131,072 ≈ 781 steps
forward/backward passes = 781 × 8 accum × 2 per-device = 12,496 micro-steps
```

`gradient_accumulation_steps = 8` means the optimiser sees a 131k-token batch while the GPU only ever holds a 4,096-token batch's worth of activations. **The compute cost is unchanged** (you still run every forward and backward); what you bought is optimisation stability and a memory-fit. Each *step* is 8× slower in wall-clock than it would be with `accum = 1`.

**Learning-rate schedule.** Linear warmup then cosine decay, the HF default shape:

```
step  0                       warmup_steps (5% of 781 ≈ 39)             781
LR    0 ────────linear ramp────────▶ 2e-4 ──────cosine decay──────▶ ~2e-5
```

> **Correction:** the instructor describes the pretraining objective as *"a regressive modeling"* [30:25, 31:03]. The correct term is **autoregressive** (or "causal"). "Regressive" means something else entirely (regression on a continuous target). Use "autoregressive/causal LM" in an interview.

### 4.3 What happens at the tensor/gradient level

Take a single transformer block, `h = 4096`, `n_heads = 32`, `d_head = 128`, sequence `s`, batch `b`, in bf16.

```
x                 : (b, s, 4096)                 input
x1  = LayerNorm(x): (b, s, 4096)                 pre-norm
Q = x1 @ Wq       : (b, s, 4096)                 Wq: (4096, 4096)
K = x1 @ Wk       : (b, s, 4096)
V = x1 @ Wv       : (b, s, 4096)
Q,K,V reshaped    : (b, n_heads=32, s, 128)
S = Q @ Kᵀ / √128 : (b, 32, s, s)                ← the quadratic term
P = softmax(S)    : (b, 32, s, s)
A = P @ V         : (b, 32, s, 128) → (b, s, 4096)
x2 = x + A @ Wo   : (b, s, 4096)                 residual
x3 = LayerNorm(x2): (b, s, 4096)
g  = SiLU(x3 @ Wg): (b, s, 11008)                SwiGLU gate   Wg: (4096, 11008)
u  = x3 @ Wu      : (b, s, 11008)                SwiGLU up
m  = g * u        : (b, s, 11008)
x4 = x2 + m @ Wd  : (b, s, 4096)                 Wd: (11008, 4096)  output
```

| Fact | Consequence |
|---|---|
| Every tensor above is **stored** for the backward pass in vanilla PyTorch | Activation memory is dominated by `(b, 32, s, s)` = the attention matrices |
| `S` and `P` are `s²` | At `s=2048`: 268 MB each per layer. At `s=8192`: 4.3 GB each per layer |
| FlashAttention never materialises `S`/`P` | Removes the dominant activation term entirely |
| The backward pass needs each block's **input** at minimum | Gradient checkpointing stores only those and recomputes the rest |
| Residual stream is 4096-wide throughout | `x → x4` are *additive*; this is why LoRA on `Wq,Wk,Wv,Wo,Wg,Wu,Wd` covers nearly all adaptivity |

The backward sweep produces `∂L/∂W` for each of the seven matrices per block (`Wq,Wk,Wv,Wo,Wg,Wu,Wd`) — for `L = 32` blocks that is 224 weight-gradient tensors. **Full fine-tuning means storing all 224 plus their Adam moments.** LoRA means storing and updating 224 `(B, A)` pairs of rank `r` instead — for `r = 16` and `h = 4096`, `BA` has `2 × 4096 × 16 = 131,072` values versus `4096 × 4096 = 16.8M` for the full matrix, a **0.78%** trainable fraction per matrix.

### 4.4 Memory and compute accounting — the arithmetic

#### (a) Memory for full fine-tuning

```
M_fullFT  = P × (w_bytes + g_bytes + opt_bytes + master_bytes)     +  activations + comm buffers
```

| Training mode | Weights | Gradients | Adam m+v | fp32 master | **Total /param** | **7B** | **13B** | **70B** |
|---|---|---|---|---|---|---|---|---|
| SGD, no momentum (bf16) | 2 | 2 | 0 | 0 | **4 B** | 28 GB | 52 GB | 280 GB |
| SGD + momentum (bf16) | 2 | 2 | 4 | 0 | **8 B** | 56 GB | 104 GB | 560 GB |
| Adam, all fp32 | 4 | 4 | 8 | 0 | **16 B** | 112 GB | 208 GB | 1,120 GB |
| **AdamW + AMP (bf16 compute, fp32 master)** | 2 | 2 | 8 | 4 | **16 B** | **112 GB** | **208 GB** | **1,120 GB** |

The last row is the production reality and the famous number: **~16 bytes per parameter, so 7B full FT needs ≈112 GB before a single activation.** A 24 GB card cannot do it. Neither can two of them. This is not a tuning problem; it is arithmetic.

Add activations from §4.3: for `b=1, s=2048`, flash attention, no checkpointing → ~26 GB → **~9 GB per billion parameters**, i.e. ~63 GB for 7B. With gradient checkpointing, ~0.7 GB for 7B plus transient recomputation. Practical rule:

```
Full FT, 7B, seq 2048, batch 8, gradient checkpointing:
  112 GB (states) + ~5 GB (activations) + ~3 GB (fragmentation/comm) ≈ 120 GB
  → 2 × A100-80GB is the floor, 4 × A100-80GB is comfortable.
```

#### (b) Memory for LoRA / QLoRA

The base weights are frozen, so no gradients, no Adam state, no master weights for them:

| Mode | Base weights | Trainable (r=16, ~0.4% of 7B ≈ 30M) | Activations | **Realistic 7B total** |
|---|---|---|---|---|
| LoRA, bf16 base | 14 GB (2 B/param) | 30M × 16 B = 0.5 GB | 5–12 GB | **20–27 GB** (24 GB card, tight at seq 2048) |
| QLoRA, NF4 base | 3.9 GB (0.55 B/param) | 0.5 GB + paged optimiser | 5–12 GB | **10–16 GB** (16 GB card feasible) |

> **Beyond the video:** the numbers the transcript never gives. These are the ones to put on a whiteboard.

| Model | Full FT (AdamW+AMP) | LoRA (bf16) | QLoRA (NF4) |
|---|---|---|---|
| 1B | 16 GB + act → **~24 GB** | ~2 GB + act → **8–10 GB** | ~0.6 GB + act → **5–6 GB** |
| 3B | 48 GB + act → **~60 GB** | ~6 GB + act → **12–16 GB** | ~1.7 GB + act → **7–9 GB** |
| **7B** | 112 GB + act → **~128 GB** (2×A100-80) | 14 GB + act → **20–27 GB** (1×A100-40 / 1×RTX 4090 24 GB tight) | 3.9 GB + act → **10–16 GB** (1×T4 16 GB tight, 1×A10G fine) |
| 13B | 208 GB + act → **~230 GB** (4×A100-80) | 26 GB + act → **32–40 GB** (1×A100-80 / 1×A6000-48 fine) | 7.2 GB + act → **16–22 GB** (1×4090 24 GB fine) |
| 70B | 1,120 GB + act → **~1.3 TB** (16–32×A100-80) | 140 GB + act → **170–220 GB** (3–4×A100-80) | 39 GB + act → **48–64 GB** (1×A100-80, or 2×A6000-48 with care) |

All figures assume gradient checkpointing on for LoRA/QLoRA, seq length 1024–2048, micro-batch 1–2. **Sequence length is the hidden multiplier**: activations and attention memory scale with `s²` before flash attention, `s` after it. Doubling seq length can double activation memory and can turn a fit into an OOM.

#### (c) Compute for training — the 6ND rule

Per token, per parameter:
```
forward pass       : 2 FLOPs  (1 multiply + 1 add per weight)
backward pass      : 4 FLOPs  (gradient wrt input  +  gradient wrt weight — two matmuls)
                     ─────
total              : 6 FLOPs per parameter per token
```
Therefore:
```
Training FLOPs  ≈ 6 × N × D          (N = params, D = tokens)
Inference FLOPs ≈ 2 × N × tokens     (forward only)
```
Attention adds an extra term that matters at long context:
```
attention FLOPs ≈ 12 × L × s² × h  per sequence of length s    (prefill)
```

**Worked example — 7B model, 1B tokens:**
```
6 × 7e9 × 1e9 = 4.2e19 FLOPs
A100-80GB bf16 dense peak = 312 TFLOP/s; at 40% MFU → 125 TFLOP/s effective
4.2e19 / 1.25e14 = 3.4e5 s = 93 GPU-hours  ≈ 4 days on 1 × A100
```

**Worked example — the Chinchilla run (see §4.5):**
```
D_opt = 20 × 7e9 = 1.4e11 tokens (140B)
FLOPs = 6 × 7e9 × 1.4e11 = 5.88e21
5.88e21 / 1.25e14 = 4.7e7 s = 13,060 GPU-hours
on 64 × A100-80GB: 204 h ≈ 8.5 days
```

**Worked example — a realistic SFT job.** 10,000 examples × 500 tokens = 5M tokens; 3 epochs = 15M tokens; 7B:
```
6 × 7e9 × 1.5e7 = 6.3e17 FLOPs
1 × RTX 4090 (165 TFLOP/s bf16 dense; QLoRA overhead → ~50 TFLOP/s effective)
6.3e17 / 5e13 = 1.26e4 s = 3.5 hours
```
The same job on a rented A100-80GB at 125 TFLOP/s effective: `6.3e17/1.25e14 = 5,040 s = 1.4 h ≈ $2.50`. **Fine-tuning a 7B on 10k examples costs about the price of a sandwich.** That fact, not any architecture change, is why fine-tuning became ubiquitous in 2023–2025.

> **Beyond the video:** **LoRA does not reduce training FLOPs.** The base weights are frozen, but you still run the forward pass through them (2N) and you still run a backward pass to compute input-gradients for earlier layers (~4N, minus the weight-gradient matmuls for frozen matrices). Total remains ≈6ND. LoRA buys **memory**, not compute. Reported "2–4x faster" claims from frameworks like Unsloth come from kernel fusion, custom attention, and skipping frozen-weight gradient buffers — not from the rank decomposition itself.

#### (d) Compute for inference — and why bandwidth usually wins

```
Inference FLOPs = 2 × N per generated token
Decode time per token ≈ (weight bytes) / (memory bandwidth)     ← the real constraint
```

| Model (bf16) | Weight bytes | A100 80GB (2.04 TB/s) | H100 SXM (3.35 TB/s) | RTX 4090 (1.01 TB/s) |
|---|---|---|---|---|
| 7B | 14 GB | 6.9 ms → **145 tok/s** | 4.2 ms → **239 tok/s** | 13.9 ms → **72 tok/s** |
| 13B | 26 GB | 12.7 ms → **78 tok/s** | 7.8 ms → **129 tok/s** | 25.8 ms → **39 tok/s** |
| 70B | 140 GB | 68.6 ms → **15 tok/s** | 41.8 ms → **24 tok/s** | does not fit |

Real-world throughput is 40–70% of these ceilings. **Arithmetic intensity** explains when compute matters instead:

```
Decode: FLOPs = 2NB, bytes moved ≈ 2N  →  intensity = B FLOP/byte  (B = batch size)
A100 ridge point = 312e12 / 2.039e12 = 153 FLOP/byte  → need batch ≈153 to saturate
H100 ridge point = 989e12  / 3.35e12  = 295 FLOP/byte  → need batch ≈295 to saturate
```

Below that batch size — which is *always* true for single-user chat — **you are buying memory bandwidth, not TFLOPS.** Batch the requests and the same GPU gets dramatically more throughput. This is the single most valuable hardware insight in this module.

> **Correction:** the instructor frames the GPU question purely as "which model can I fit," i.e. capacity. Capacity is necessary but not sufficient: for serving, the buying decision is **bandwidth per dollar**, and for training it is **bf16 TFLOPs × achievable MFU per dollar**. An L40S (48 GB, 864 GB/s, 362 bf16 dense) beats an A100 on capacity-per-dollar but loses ~2.4x on bandwidth and ~2.7x on bf16 compute — so it is a fine inference box for a 13B and a poor training box for a 70B.

#### (e) KV cache — the memory term everyone forgets in serving

```
KV bytes per token = 2 × L × n_kv_heads × d_head × bytes_per_element
```

| Model | Configuration | KV per token | At 2,048 ctx | At 32,768 ctx |
|---|---|---|---|---|
| Llama-2-7B | L=32, 32 KV heads, d_head=128, fp16 | 512 KiB | 1.0 GB | 16.8 GB |
| Llama-3-8B (GQA) | L=32, **8** KV heads, d_head=128, fp16 | 128 KiB | 0.26 GB | 4.2 GB |
| Llama-3-70B (GQA) | L=80, **8** KV heads, d_head=128, fp16 | 320 KiB | 0.65 GB | 10.5 GB |

**Grouped-query attention (GQA) exists because of this table.** Llama-2-7B's 0.5 MiB/token means 8 concurrent 4k-context users need 16 GB of KV cache — as much as the weights. Llama-3's 8 KV heads cut that 4x. When you plan a serving deployment, `VRAM = weights + (KV/token × ctx × concurrency) + overhead`, and at long context or high concurrency **the KV cache is the larger term**.

---

### 4.5 Scaling laws — Kaplan vs Chinchilla, and why modern models ignore both

Two papers set the field's budget rules. They disagree, and the disagreement is instructive.

**Kaplan et al. 2020 (OpenAI), "Scaling Laws for Neural Language Models."** Loss is a power law in `N` (parameters), `D` (data), and `C` (compute), with each factor saturating once the others become the bottleneck. The practical output, for a fixed compute budget:

```
Kaplan:  N_opt ∝ C^0.73 ,  D_opt ∝ C^0.27      → put ~90% of the budget into parameters
```

This is why GPT-3 (2020) is 175B parameters trained on only 300B tokens — **1.7 tokens per parameter.** Kaplan's prescription was to build bigger models, not to feed them more.

**Hoffmann et al. 2022 (DeepMind), "Training Compute-Optimal Large Language Models" (Chinchilla).** Re-ran the study with a corrected methodology (three independent estimation approaches; importantly, learning-rate schedules tuned per run rather than fixed, which had biased Kaplan against data). The revised result:

```
Chinchilla:  N_opt ∝ C^0.5 ,  D_opt ∝ C^0.5
             → parameters and tokens should grow in EQUAL PROPORTION
             → D_opt ≈ 20 × N
```

**The demonstration:** Chinchilla, 70B parameters trained on **1.4T tokens**, matched or beat Gopher (280B / 300B tokens), GPT-3 (175B / 300B tokens), and Megatron-Turing NLG (530B / 300B tokens) — using roughly the same training compute as Gopher, i.e. a **4x smaller model**. GPT-3, at 175B/300B, was ~4x oversized relative to its data.

**The Chinchilla rule with arithmetic for a 7B model:**

```
D_opt = 20 × 7e9 = 1.4e11 tokens = 140,000,000,000 tokens

Corpus size sanity check:
  140B tokens ≈ 105 GB as uint16 token ids (2 bytes/token)
              ≈ 560 GB as uint32 token ids (4 bytes/token)
              ≈ 100–140B words of English  (≈0.75 words/token)

Compute:  6 × 7e9 × 1.4e11 = 5.88e21 FLOPs
Time:     5.88e21 / 1.25e14 (A100 @ 40% MFU) = 4.7e7 s = 13,060 A100-hours
Money:    13,060 × $1.79/h (Lambda A100-80GB on-demand ballpark) ≈ $23,400
Wall:     64 × A100-80GB → 204 h ≈ 8.5 days
```

**Why essentially nobody follows it today.** Chinchilla optimises *training* compute per unit of quality. It ignores **inference**. A smaller model that was trained longer is permanently cheaper to serve:

```
Total cost of ownership  =  training cost  +  (inference cost per token × lifetime tokens)

Llama-3-8B: 15T tokens ÷ 8e9 params = 1,875 tokens/param   → 94× over Chinchilla
Llama-2-7B:  2T tokens ÷ 7e9 params =   286 tokens/param   → 14× over Chinchilla
GPT-3:     300B tokens ÷ 175e9      =   1.7 tokens/param   → 12× UNDER Chinchilla
```

| Model | Params | Tokens | Tokens/param | vs Chinchilla (20) | Rationale |
|---|---|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | 1.7 | 0.09x (severely undertrained) | Kaplan-era belief that params dominate |
| Chinchilla (2022) | 70B | 1.4T | 20 | **1.00x (the reference)** | Compute-optimal training |
| Llama-2-7B (2023) | 7B | 2.0T | 286 | 14x | Inference-optimal at a fixed serve cost |
| Llama-3-8B (2024) | 8B | 15T | 1,875 | 94x | Inference-optimal + data pipeline maturity |
| Phi-3-mini (2024) | 3.8B | 3.3T | 868 | 43x | Data-quality thesis + on-device target |

> **Beyond the video:** the instructor never mentions scaling laws. This is the gap that most often shows up in interviews. The two-sentence summary to memorise: **"Kaplan (2020) said buy parameters; Chinchilla (2022) corrected it to buy parameters and tokens in equal measure, ~20 tokens per parameter; and everyone since 2023 deliberately overtrains past Chinchilla because inference cost dominates lifetime cost."**

> **Correction:** a common interview answer — "Chinchilla says 20 tokens per parameter, so a 7B model needs 140B tokens" — is right about training-optimal and **wrong as a production prescription**. If you are going to serve the model to a million users, the correct computation is total cost of ownership, and the answer is usually "train smaller, train longer." Chinchilla is the baseline you compute *against*, not the target you hit.

### 4.6 Emergent abilities — and the honest counter-argument

**The claim (Wei et al., 2022, "Emergent Abilities of Large Language Models").** Certain capabilities are ~absent in smaller models and appear abruptly past a scale threshold: 3-digit arithmetic, chain-of-thought reasoning, instruction following, word-in-context retrieval, and specific benchmark tasks. Plotted against log-compute, the curves are flat then near-vertical.

**The counter-argument (Schaeffer, Miranda & Koyejo, 2023, "Are Emergent Abilities of Large Language Models a Mirage?").** The apparent discontinuity is largely an artefact of the *metric*, not the model. Their argument:

1. Capabilities are assessed with **discontinuous, all-or-nothing metrics** — exact match, multiple-choice accuracy, pass@k for a whole program.
2. The underlying per-token/per-step capability improves **smoothly and predictably**.
3. A nonlinear metric applied to a smooth capability produces a sharp jump. Example: if per-token accuracy rises linearly from 0.5 to 0.9, the probability of getting a 5-token answer *exactly* right goes from `0.5⁵ = 3%` to `0.9⁵ = 59%` — a curve that looks like a step function.
4. **Change the metric to a continuous one** (token-level edit distance, Brier score, log-likelihood of the correct answer) and the same models show smooth, predictable improvement with no threshold.

| Position | Evidence for | What it implies for a practitioner |
|---|---|---|
| Emergence is real | Abrupt benchmark jumps replicate across model families; some abilities (ICL) genuinely require scale | Capability planning must account for non-linear returns to scale |
| Emergence is a metric artefact | Metric-swapping flattens the curves (Schaeffer et al.); predictions from smooth metrics transfer across scales | Benchmark jumps mislead; prefer continuous metrics and per-token diagnostics |

**The practitioner's honest position:** both are partly right, and the disagreement is exactly why you should *never* conclude "our model can't do this task" from a single all-or-nothing benchmark score. Evaluate with a *continuous* metric first (log-probability of the right answer, token-level F1), then apply the strict metric for reporting. If the continuous metric improves smoothly across your checkpoints but the strict one stays at zero, you have a **prompting/format problem**, not a capability problem — the fix is a chat template and few-shot examples, not more parameters. This single diagnostic saves more wasted fine-tuning budget than any other in this handbook.

> **Beyond the video:** also relevant but outside the transcript: **grokking** (delayed generalisation long after train loss converges — training longer than looks necessary sometimes pays), **double descent** (test loss improves again past the interpolation threshold — why bigger models can overfit *less*), and **inverse scaling** (some tasks get worse with scale, e.g. certain bias/sycophancy behaviours). None are in the videos; all are fair game in a senior interview.

### 4.7 Base models vs instruct models — the most misunderstood distinction in the field

**Mechanically:**
```
BASE MODEL     = transformer + tokenizer + LM head.  Trained ONLY on next-token prediction.
                 No chat template in the model card.  No notion of "user" or "assistant".
INSTRUCT MODEL = base + SFT on (instruction, response) pairs [+ preference alignment],
                 shipped WITH a chat template and stop tokens.
```

**The behavioural difference is stark.** Given `"What is the capital of France?"`:

| | Typical base-model continuation | Typical instruct-model response |
|---|---|---|
| Output | `"What is the capital of Germany?\nWhat is the capital of Italy?\n..."` — it treats the line as one item in a list of questions, because that is the pattern it saw on the web | `"The capital of France is Paris."` |
| Why | It was trained to continue *text*, and question lists are a common text pattern | It was trained on demonstrations of *answering* |

**Why this is the field's most consequential confusion:**

1. **You cannot evaluate a base model with a chat benchmark.** MMLU is multiple-choice ("Answer with A, B, C, or D"). A base model will not emit a bare letter; it will continue the question. Base models therefore score near chance on MMLU-format evaluation while possessing the knowledge being tested. Teams conclude the model is bad; the model is fine, the harness was wrong. (The LM-eval-harness "base model" task variants exist for exactly this reason.)
2. **You cannot fine-tune a base model on instruct data and expect instruct behaviour from 500 examples.** You can — but you are teaching the *entire* format from scratch. Expect to need 10k–100k+ examples and to lose the base's ICL ability. Fine-tuning an *instruct* model means you are *refining* a behaviour that already exists, which takes 1k–10k examples.
3. **The chat template must be the model's own.** Applying ChatML to a Llama-2 model, or `[INST]` to a Qwen model, produces training loss that falls and validation quality that does not. Silent failure. See §10.3.
4. **Alignment is fragile in the direction you would not expect.** Starting from an instruct model does not protect you: Qi et al. (2023) showed **10 adversarially crafted examples** break the safety alignment of production models, and Lermen et al. (2023) showed a few hundred benign examples can degrade it via LoRA. Conversely, Wei et al. (2023) showed safety behaviour lives in a low-rank, *removable* direction in weight space. **Every fine-tune of an aligned model is a safety regression test away from an incident** (see §16).

**Interview-grade one-liner:** *"Base model = the language. Instruct model = the language plus a job description. Fine-tuning the wrong one is a bug, not a preference."*

### 4.8 Tokenization, deeply

#### The three families

| Algorithm | Used by | Mechanism | Vocab (typical) | Notes |
|---|---|---|---|---|
| **Byte-Pair Encoding (BPE)** | GPT-2/3/4 (tiktoken `cl100k_base`, `o200k_base`), Llama (via SentencePiece-BPE), Mistral, Qwen | Start from bytes; repeatedly merge the most frequent adjacent pair | 32k–200k | Frequency-driven; deterministic; merges are ordered and stable |
| **WordPiece** | BERT, DistilBERT, ELECTRA | Greedy longest-match-first against a learned vocabulary, scored by likelihood rather than frequency; `##` prefix marks continuations | 30,522 | Word-boundary aware; has `[UNK]`, so it *can* fail on unseen characters |
| **Unigram / SentencePiece** | T5, ALBERT, XLNet, Gemma, Llama (SP-BPE variant) | Start from a large candidate set, prune to maximise corpus likelihood; language-agnostic, operates on raw text with `▁` for space | 32k (T5) – 256k (Gemma) | No pre-tokenization needed; handles any script; byte fallback |

**Byte-level BPE means there is no `[UNK]`.** GPT-2/Llama-3 tokenizers can encode *any* byte string, because the base alphabet is the 256 bytes. That is why an emoji or a rare Devanagari glyph becomes 3–8 tokens rather than `[UNK]` — more expensive, never impossible. WordPiece models (BERT family) genuinely can emit `[UNK]`, which is a real information loss.

#### Vocabulary-size trade-offs

| Vocab size | Embedding params (`V × h`) | Softmax cost | Tokens per English word | Winner in |
|---|---|---|---|---|
| 30,522 (BERT) | 30,522 × 768 = **23.4M** | Low | ~1.4 | English-only encoder tasks |
| 32,000 (Llama-2) | 32,000 × 4096 = **131M** | Low | ~1.3 | English-centric LLMs |
| 50,257 (GPT-2) | 50,257 × 768 = 38.6M | Moderate | ~1.3 | Legacy GPT family |
| 100,256 (GPT-4 `cl100k`) | 100,256 × 4096 = **411M** | Higher | ~0.75 | Multilingual, code |
| 128,256 (Llama-3) | 128,256 × 8192 = **1.05B** | Higher | ~0.75 | Multilingual; 15% fewer tokens than Llama-2's tokenizer |
| 256,000 (Gemma) | 256,000 × 2048 = 524M | Highest | ~0.7 | Multilingual, non-Latin scripts |

The trade is explicit: **a bigger vocabulary means more embedding parameters and a more expensive softmax, but fewer tokens per document** — which shortens sequences (cheaper attention, longer effective context) and, for API-metered inference, reduces cost per word. Llama-3's move from 32k to 128k vocabulary cut token counts ~15% on English and far more on non-Latin scripts, at the cost of ~900M extra parameters.

**Rules of thumb to memorise:**
```
English text, GPT-4/cl100k-class tokenizer :  1 token ≈ 4 characters ≈ 0.75 words
                                             1,000 words ≈ 1,333 tokens
English text, Llama-2 32k tokenizer        :  1 token ≈ 3.5 characters ≈ 0.7 words
Hindi / Thai / Amharic, 32k English-centric:  1 word  ≈ 3–10 tokens  (cost blowup)
Source code                                :  1 token ≈ 3 characters (worse than prose)
```

Non-English cost blowup is not academic. With an English-centric 32k tokenizer, the same Hindi document consumes 3–10x the tokens of its English translation — 3–10x the inference cost, 3–10x the context, 3–10x the training tokens. Llama-3's 128k vocabulary exists substantially to fix this. If your product is non-English, **the tokenizer is a first-class cost decision, not a detail.**

#### Why tokenizers are the source of many bugs

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| Tokenizer/model mismatch | Fluent nonsense; loss never falls below ~4 | Loaded `AutoTokenizer` from a different repo than the model | Always load tokenizer from the same `model_id` as the model |
| Missing pad token | `ValueError: Asking to pad but the tokenizer does not have a padding token` — or silent garbage from position 0 | Llama/Mistral tokenizers ship with no `pad_token` | `tokenizer.pad_token = tokenizer.eos_token` (and set `pad_token_id`) |
| Padding side wrong | Loss decreases, but generations are incoherent | Decoder-only models require **left** padding for batched generation; Trainer defaults to right padding | `tokenizer.padding_side = "left"` at inference; right padding is fine for training if the loss mask is correct |
| Chat template mismatch | Loss falls smoothly; the model answers badly or not at all | Template from the wrong model family applied to the data | `tokenizer.apply_chat_template(..., tokenize=False)` and inspect the raw string |
| Loss computed on prompt tokens | Model learns to generate the *questions*; eval loss looks great, chat is useless | Labels not masked to `-100` on prompt tokens | Mask the prompt span; verify by decoding the supervised span |
| Truncation eats the answer | Model trained on truncated inputs, produces truncated outputs; eval loss spikes on long examples | `truncation=True, max_length=512` cuts the *end*, which is where the response lives | Truncate on the left for SFT, or filter examples over the limit |
| Resized embedding without training | Random outputs for the new tokens forever | `tokenizer.add_tokens(...)` + `model.resize_token_embeddings(...)` but the new rows are never trained (or are in a frozen base) | Ensure new embedding rows are trainable; initialise from the mean of existing embeddings |
| Special token stripped by cleaning | Model never learns to stop; generations run to `max_new_tokens` | A regex cleaning step removed `<|im_end|>` / `</s>` | Exclude special-token strings from all text cleaning |

#### Special tokens and chat templates by family

| Model family | BOS / EOS | Turn delimiters | Template shape |
|---|---|---|---|
| Llama-2 | `<s>` / `</s>` | `[INST] … [/INST]`, `<<SYS>> … <</SYS>>` | `<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>` |
| Llama-3 / 3.1 | `<\|begin_of_text\|>` / `<\|eot_id\|>` | `<\|start_header_id\|>role<\|end_header_id\|>` | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n{sys}<\|eot_id\|>…<\|start_header_id\|>assistant<\|end_header_id\|>\n\n{resp}<\|eot_id\|>` |
| Mistral / Mixtral | `<s>` / `</s>` | `[INST] … [/INST]` | `<s>[INST] {user} [/INST]{resp}</s>` |
| Qwen2 / Qwen2.5 | none / `<\|im_end\|>` | **ChatML**: `<\|im_start\|>role … <\|im_end\|>` | `<\|im_start\|>system\n{sys}<\|im_end\|>\n<\|im_start\|>user\n{u}<\|im_end\|>\n<\|im_start\|>assistant\n{r}<\|im_end\|>` |
| Gemma / Gemma-2 | `<bos>` / `<eos>` | `<start_of_turn>role … <end_of_turn>` | `<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{r}<end_of_turn>` |
| Phi-3 | `<s>` / `<\|endoftext\|>` | `<\|user\|> … <\|end\|>` `<\|assistant\|>` | `<\|user\|>\n{u}<\|end\|>\n<\|assistant\|>\n{r}<\|end\|>` |

**Always do this before training:**

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
msgs = [{"role": "system", "content": "You are terse."},
        {"role": "user",   "content": "Capital of France?"},
        {"role": "assistant","content": "Paris."}]
print(tok.apply_chat_template(msgs, tokenize=False))   # INSPECT THE RAW STRING
ids = tok.apply_chat_template(msgs, tokenize=True, return_assistant_tokens_mask=True)
print(len(tok), tok.special_tokens_map)                # vocab size + specials
```

If the printed string does not look exactly like the model card's template, stop. Fix it before spending a GPU-hour.

### 4.9 Data quality — dedup, filtering, contamination, licensing

#### Deduplication

Duplicate documents waste compute, inflate memorisation, and cause train/test leakage. The standard pipeline:

| Method | How | Cost | Catches |
|---|---|---|---|
| Exact hash | SHA-256 of the document (or of a normalised form) | Trivial | Byte-identical reposts |
| MinHash + LSH | Jaccard similarity over n-gram shingles; banded LSH for sub-linear candidate lookup | Moderate | Near-duplicates, boilerplate, mirrors |
| Suffix array | Substring-level dedup across the whole corpus (Lee et al. 2021) | High (needs the full index) | Long repeated spans *within* distinct documents |
| Semantic dedup (SemDeDup) | Embed, cluster, drop near-identical members | Highest | Paraphrase-level duplicates |

**Evidence:** Lee et al. (2021), "Deduplicating Training Data Makes Language Models Better" — deduplication reduces verbatim memorisation substantially and improves perplexity at fixed compute. Modern open corpora (FineWeb, Dolma, SlimPajama, RedPajama-v2) all document a dedup stage; FineWeb's ablation study is the best public reference for how much each stage buys.

#### Filtering

| Filter | Purpose | Typical tool |
|---|---|---|
| Language ID | Keep target languages only | fastText `lid.176` |
| Quality classifier | Drop spam, ads, boilerplate, listicles | fastText classifier trained on a small curated positive set |
| Heuristic rules | Length, symbol-to-word ratio, repetition ratio, stop-word presence, mean line length | Custom (Gopher/MassiveText-style rules) |
| Toxicity / NSFW | Remove harmful content | Jigsaw/Detoxify classifiers |
| PII | Remove emails, phones, national IDs | Regex + NER |
| Benchmark decontamination | Remove eval-set n-grams | n-gram overlap with a 13-gram rule (GPT-3), or exact-substring |

#### Contamination

Contamination is the presence of evaluation data in the training set. It is *assumed present* in every public model and is the reason benchmark leaderboards must be read sceptically. Detection:

- **n-gram overlap** between the corpus and the test set (GPT-3 used a 13-gram rule). Report the overlap rate.
- **Canary strings**: BIG-bench and others embed a GUID canary in eval data; models can be prompted to emit it if it leaked into training.
- **Round-trip probing**: ask the model to complete the *first half* of a test item; if it completes it verbatim, it memorised it.
- **Held-out-by-construction**: build your own eval set from data the base model provably never saw (post-cutoff, internal, synthetic). This is the only robust defence and is mandatory for a production fine-tune.

#### Licensing — the part that becomes a legal problem

| Layer | What to check | Examples |
|---|---|---|
| **Training corpus** | ToS, robots.txt, copyright | Common Crawl ToS; the Books3 / LAION-5B takedowns (2023); NYT v. OpenAI; Doe v. GitHub (Copilot); EU AI Act training-data disclosure obligations |
| **Base-model licence** | Commercial use? MAU caps? Redistribution? Derivative-training permitted? | Llama 3.x Community License (**700M MAU cap**, naming/attribution requirement); Gemma Terms of Use (use restrictions + Google's Prohibited Use Policy); Mistral-7B-v0.1 **Apache-2.0** but Mistral-Large under a research licence; Qwen: Apache-2.0 for the small sizes, Tongyi Qianwen licence for 72B; Falcon TII licence |
| **Your adapter / merged model** | **Inherits the base model's terms** | A LoRA trained on Llama-3 is a Llama-3 derivative. Publishing it on the Hub is redistributing the base's terms |

> **Beyond the video:** **fine-tuning does not launder a licence.** This is the most expensive misconception in the commercial fine-tuning space. If a client says "we'll fine-tune an open model and the output is ours," the correct answer is: check the base model's licence, check whether your *derivative* inherits it, and check the training data's provenance. Also note the **EU AI Act** requires disclosure of training data summaries for general-purpose AI models (in force with phased obligations through 2025–2027), which changes what is legal to *not* document.

#### Quality vs quantity — the LIMA and Phi evidence

**LIMA (Meta, May 2023, "Less Is More for Alignment").** 1,000 carefully curated (prompt, response) pairs — 750 mined from Stack Exchange, wikiHow and similar high-quality community sources, plus 250 hand-written by the authors — applied as pure SFT to LLaMA-65B, with **no RLHF and no preference optimisation at all**. Reported human preference win rates against baselines: ~59% vs Alpaca-65B, ~58% vs DaVinci003, and **~43% against GPT-4** — i.e. competitive with, but not superior to, GPT-4. The paper's thesis sentence is the one to quote: **almost all knowledge in an LLM comes from pretraining; instruction tuning teaches it *how* to surface that knowledge, not *what* to know.**

**The Phi series (Microsoft, "Textbooks Are All You Need", 2023–2024).** The claim that data *curation* substitutes for scale:

| Model | Params | Training tokens | Notable result |
|---|---|---|---|
| phi-1 | 1.3B | ~7B (≈1B "textbook-quality" synthetic + 180M exercises + ~6B filtered code) | ~50% HumanEval — unprecedented at 1.3B |
| phi-1.5 | 1.3B | ~30B synthetic textbook data | Outperformed much larger models on reasoning benchmarks |
| phi-2 | 2.7B | 1.4T | Matched 7B–13B models of its era |
| phi-3-mini | 3.8B | 3.3T | ~69% MMLU; reported trained in roughly a week on 512 H100-80GB |

| Lesson | Applies to | Does NOT apply to |
|---|---|---|
| 1,000 curated examples can produce a well-behaved assistant | **SFT** on top of a strong base | Pretraining. LIMA starts from LLaMA-65B, not from scratch |
| Synthetic "textbook" data can replace bulk web data at small scale | Small-model **pretraining** and **continued pretraining** | Large-scale pretraining, where diversity still requires real corpora |
| Quality beats quantity at the SFT stage | Your fine-tune | Your dataset size *ceiling* — 200 examples cannot teach a new language |

**The practical synthesis, and the rule this handbook will repeat in CS-13:** *at the fine-tuning stage, quality and diversity beat volume, and 1k excellent examples reliably beat 100k scraped ones. At the pretraining stage, volume still wins, and curation is a multiplier rather than a substitute.*

### 4.10 Hardware reality

| GPU | Year | VRAM | Bandwidth | bf16 dense TFLOP/s | bf16 sparse | Interconnect | Fine for |
|---|---|---|---|---|---|---|---|
| T4 | 2018 | 16 GB GDDR6 | 320 GB/s | 65 | 130 | PCIe | QLoRA ≤3B, inference ≤7B int4 |
| V100 | 2017 | 16/32 GB HBM2 | 900 GB/s | 125 | — | NVLink 300 GB/s | Legacy; 7B inference fp16 |
| RTX 3090 | 2020 | 24 GB GDDR6X | 936 GB/s | ~71 | — | **NVLink (pair)** | QLoRA 7B–13B, hobby |
| A10G | 2021 | 24 GB GDDR6 | 600 GB/s | 125 | 250 | PCIe | QLoRA 7B; inference |
| RTX 4090 | 2022 | 24 GB GDDR6X | 1,008 GB/s | ~165 | — | **None** | QLoRA 7B–13B; best consumer inference |
| A100 40 GB | 2020 | 40 GB HBM2 | 1,555 GB/s | 312 (SXM) / 156 (PCIe) | 624 | NVLink 600 GB/s | 7B LoRA; 13B QLoRA; small full-FT |
| **A100 80 GB** | 2020 | 80 GB HBM2e | 2,039 GB/s | 312 (SXM) | 624 | NVLink 600 GB/s | The workhorse: 7B full FT (2–4 cards), 70B QLoRA (1 card) |
| A6000 | 2020 | 48 GB GDDR6 | 768 GB/s | 155 | 310 | NVLink (pair) | 13B LoRA, 70B QLoRA (2 cards) |
| L40S | 2023 | 48 GB GDDR6 | 864 GB/s | 362 | 733 | PCIe | Inference-heavy; weak for multi-GPU training |
| **H100 80 GB** | 2022 | 80 GB HBM3 | 3,350 GB/s | 989 (SXM) | 1,979 | NVLink 900 GB/s | ~3x A100 throughput; FP8 support |
| H200 | 2024 | 141 GB HBM3e | 4,800 GB/s | 989 | 1,979 | NVLink 900 GB/s | Long-context serving |
| B200 | 2024 | 192 GB HBM3e | 8,000 GB/s | ~2,250 | ~4,500 | NVLink 1.8 TB/s | 70B+ full FT, FP4/FP6 support |
| MI300X (AMD) | 2023 | 192 GB HBM3 | 5,300 GB/s | ~1,300 | — | Infinity Fabric | Cost-per-GB leader; software maturity is the tax |

**Three rules that decide real deployments:**

1. **For training, buy bf16 TFLOPs × achievable MFU.** You will get 35–50% of peak on a well-tuned run, 20–30% on a naive one. A100 80GB at 40% MFU = ~125 real TFLOP/s.
2. **For serving, buy bandwidth, and count the KV cache.** See §4.4(d)(e). A 7B on a 4090 (1,008 GB/s) generates ~72 tok/s at batch 1 — better than an A100 40GB PCIe (1,555 GB/s) would suggest per dollar, which is why consumer cards dominate single-user local inference.
3. **Consumer multi-GPU is a trap for training.** RTX 4090s have **no NVLink**; gradient all-reduce goes over PCIe (~32 GB/s Gen4 x16) and often through the CPU, which can make 2×4090 slower than 1×4090 for anything that needs communication. A100/H100 with NVLink 600–900 GB/s is why rented A100s cost more and are still cheaper per unit of *useful* work.

> **Correction:** the transcript treats GPU memory purely as "can I load the model" [12:44–13:00] — the quantization framing ("store the huge model in the lower memory"). That is only the *capacity* axis. The two missing axes that decide real performance: **bandwidth** (bounds decode; §4.4d) and **interconnect** (bounds multi-GPU scaling). A model that "fits" on four consumer cards may train slower than the same model on one rented datacentre card.

---

## 5. The End-to-End Pipeline

### 5.1 The instructor's syllabus, mapped to this handbook

Video 01 is 20 minutes of syllabus. The instructor works through his OneNote page in order and states his rationale plainly: *"I never do [one or two videos] because I believe to give you the entire content with a particular sequence, with a complete sequence"* [5:46–5:55], and *"this is the requirement of the industry — they might ask you the question from the finetuning"* [18:53–19:03]. He estimates 20–25 videos over 2–3 months [14:56, 2:31–2:37].

Here is his spoken syllabus, verbatim in intent, mapped to the module that delivers it. **What each stage solves** is the column that matters — it is the answer to "why does this topic exist at all."

| # | Instructor's syllabus item (video 01) | Timestamp | What problem it solves | Module |
|---|---|---|---|---|
| 1 | Introduction to fine-tuning: model training, pre-training, transfer learning, fine-tuning, why it matters | [6:22–6:45] | Establishes the vocabulary — you cannot reason about fine-tuning without the pretraining/transfer distinction | **CS-01** |
| 2 | Fine-tuning frameworks: HF Trainer/TRL, Unsloth, LLaMA-Factory, Axolotl | [6:45–6:54] | You should not hand-roll a training loop; each framework optimises a different constraint (speed, VRAM, no-code, reproducibility) | CS-03, CS-15, CS-16, CS-17 |
| 3 | Important research papers on fine-tuning | [6:54–7:01] | Primary sources beat blog posts; interviewers ask "which paper?" | CS-01 App. B, CS-14; LoRA/QLoRA in CS-13 §6.8 + CS-11 §4.11 |
| 4 | Fine-tuning vs RAG vs AI agents — which to choose; how to build RAG/agents **on top of** a fine-tuned model | [7:01–7:34] | The architecture decision that precedes every technical decision. Wrong answer = wasted quarter | **CS-04** |
| 5 | Fine-tuning and deep learning: what training is, CNN training in PyTorch and Keras, layers/parameters | [7:39–8:19] | Builds the tensor-level intuition that makes every later module legible | CS-01, CS-05 |
| 6 | Why fine-tuning was not possible in RNN/LSTM; LSTM vs RNN vs Transformer | [8:19–9:30] | Explains *why the Transformer unlocked the field* rather than just asserting it | CS-05 |
| 7 | Hugging Face: installation, loading models via the API, and the library zoo — `transformers`, `datasets`, `accelerate`, `TRL`, `bitsandbytes`, `sentence-transformers` | [9:33–10:40] | The actual toolchain of the trade. LangChain supports RAG/agents but **not** training [2:33–2:39] | CS-06 |
| 8 | Fine-tuning classical LMs: BERT (and T5-class) from the Hub | [10:42–11:36] | Encoder/seq2seq fine-tuning for classification, NER, QA — still the highest-ROI fine-tuning in industry | CS-07 |
| 9 | Knowledge distillation: DistilBERT, then LLM → SLM (LLaMA, Phi) | [11:39–11:55] | Turns an expensive teacher into a cheap student; the main lever on serving cost | CS-08, CS-09 |
| 10 | The LLM era begins; unsupervised pretraining (he promises a separate from-scratch video) | [11:55–12:39] | Where the capability actually comes from | CS-01 |
| 11 | Quantization: GGUF, GGML, GPTQ, AWQ, int4, int8 | [12:41–13:22] | Makes a model that does not fit, fit. The enabler of local and cheap serving | CS-10, CS-11 |
| 12 | Fine-tuning Llama, Mistral, Gemma, Phi-3 and other open LLMs — "one code, any model" | [13:23–13:34] | The generic pipeline that generalises across bases | CS-13 (*"CS-20" is planned, unwritten*) |
| 13 | LoRA, QLoRA (LoRA + quantization), DoRA, RAFT | [13:34–13:50] | The memory revolution: fine-tune on one GPU | **CS-13 §6.8, CS-11 §4.11** |
| 14 | Data preparation; full fine-tuning vs parameter-efficient fine-tuning | [13:52–14:00] | The data is the model. Full FT vs PEFT is the first branching decision | CS-13, CS-13 §6.8 |
| 15 | Ready-made tools: Axolotl, MLX, Unsloth, and the rest | [14:00–14:09] | Removes the training-loop boilerplate entirely | CS-15, CS-16, CS-17 |
| 16 | Deployment: Ollama, RAG on top of the model, HF Hub, cloud | [14:09–14:27] | A model in a notebook is not a product | *planned (CS-28)* — CS-13 §15 is nearest |
| 17 | API-based fine-tuning: OpenAI, Gemini, instruction-based FT | [14:27–14:44] | Rent capability instead of building it; the right answer when your data cannot leave your control *less* than your need for control | CS-18, CS-19 |
| 18 | Vision-language models: ViT, Qwen-VL, LLaMA-V; multimodal data (image, audio, video) | [15:10–15:56] | The fastest-growing slice of applied fine-tuning | *planned (CS-21), not yet written*; `code/12_multimodal_vlm.py` |
| 19 | RLHF — "introduced by ChatGPT, now the backbone of fine-tuning"; PPO vs DPO; preference alignment | [15:59–17:12] | Turns a model that *can* answer into one that answers *the way you want* | CS-14 §4.6.1, §4.6.3, §4.6.9, §4.6.10 |
| 20 | Embedding fine-tuning — "embedding is just a set of numbers, a vector" | [17:12–17:44] | Retrieval quality is the ceiling on RAG quality | `code/10_embedding_finetune.py` (*"CS-22" planned, unwritten*) |
| 21 | Bonus: adapters; evaluation metrics | [17:47–18:04] | Adapters make one base serve many tasks; metrics are how you prove any of it worked | CS-13 §6.8; metrics in CS-13 §12 (*capstone CS-28 planned*) |

> **On the "Module" column:** CS-20–CS-28 were planned but never written. Each cell above points at the nearest *written* material; *planned* marks a syllabus item with no written home yet.

**The instructor's own summary of why the topic matters** — worth reading twice, because it is the pitch for the entire handbook:

> *"This fine-tuning seems very underrated — most people don't discuss it... but no one talks about the fundamental. Believe me, this LLM fine-tuning, it's a fundamental topic of the generative AI."* [3:07–3:29]
>
> *"Companies are spending their money, their resources for fine-tuning their models. They are using the model from various APIs like Claude and OpenAI, but along with that they are fine-tuning according to their requirement... People are fine-tuning open-source models, then on top of it they are creating a RAG, they are creating agents."* [3:37–4:00]
>
> *"Fine-tuning is just not a fine-tuning, just not a training the model — it is more than that."* [4:09–4:17]
>
> *"This could be expensive... so how we can train it at a minimal cost, what frameworks, what new research — with minimal cost, with no cost, we can fine-tune our model."* [4:32–4:54]

**Why the instructor is right about the framing and where the framing is incomplete.** He correctly identifies fine-tuning as a *lifecycle* discipline rather than a single technique, and correctly names the cost constraint as the binding one. The three things he does not say, which you need:

> **Beyond the video:**
> 1. **Fine-tuning is not the default.** The correct order of escalation is: prompt engineering → few-shot prompting → structured output constraints → RAG → fine-tune → agent. Fine-tuning is fourth because it is the first step that is expensive to *reverse*: you cannot A/B a weight change as cheaply as you can A/B a prompt, and rollback means re-serving a different checkpoint.
> 2. **The best fine-tuning datasets are usually built from logged production traffic**, not curated by hand. Your own prompt/response logs, filtered by user satisfaction signals, beat any public dataset because they are in-distribution by construction.
> 3. **The unit economics decide.** A fine-tune that reduces a 2,000-token few-shot prompt to a 200-token zero-shot prompt pays for itself in inference cost within weeks (see §11.4). A fine-tune that only improves a benchmark and not the token count usually does not.

### 5.2 Stage-by-stage: the fine-tuning pipeline in production

```
[1] DEFINE THE TASK                    ─── can prompting/RAG solve it? if yes, STOP.
        │
        ▼
[2] COLLECT DATA                       ─── production logs > curated > public datasets
        │                                  target: 1k–10k high-quality pairs (SFT)
        ▼
[3] FORMAT & TEMPLATE                  ─── apply_chat_template → INSPECT THE STRING
        │                                  mask prompt tokens to -100
        ▼
[4] SPLIT                              ─── train / val / test, deduped ACROSS splits
        │                                  val loss ≠ your product metric
        ▼
[5] CHOOSE METHOD                      ─── QLoRA (default) → LoRA → full FT (only if needed)
        │
        ▼
[6] TRAIN                              ─── monitor loss, grad-norm, LR, throughput
        │                                  early-stop on val loss
        ▼
[7] EVALUATE                           ─── task metric + regression suite + safety suite
        │                                  baseline = base model + few-shot prompt
        ▼
[8] MERGE + QUANTIZE                   ─── merge adapter; quantize for serving
        │
        ▼
[9] SERVE + MONITOR                    ─── vLLM/TGI/Ollama; log prompts; drift alarms
        │
        ▼
[10] ITERATE                           ─── new failures → new training data → back to [2]
```

| Stage | Input | Operation | Output | Failure mode |
|---|---|---|---|---|
| 1 Define | Business problem | Decide FT vs RAG vs prompt vs agent | A decision, with a STOP condition | Fine-tuning to fix a prompting problem |
| 2 Collect | Logs, docs, human writers | Filter, dedup, label, curate | 1k–10k pairs | 200 examples of one template; class imbalance; no negatives |
| 3 Format | Raw pairs | Apply the model's chat template; mask prompt tokens | Tokenised dataset | Wrong template; loss on prompt tokens; truncation eating the answer |
| 4 Split | Dataset | Split by *source document / conversation*, not by row | train/val/test | Leakage: near-identical rows in train and test inflate metrics |
| 5 Choose | Constraints | QLoRA / LoRA / full FT | A config | Full FT on hardware that cannot hold it; LoRA where full FT is needed |
| 6 Train | Config + data | Forward, loss, backward, step | Checkpoint + logs | Loss goes to 0 (overfit) or stays flat (LR/template bug) |
| 7 Evaluate | Checkpoint | Task metric, regression, safety, LLM-judge, human | A verdict | Only measuring the task; ignoring catastrophic forgetting |
| 8 Package | Adapter | Merge, quantize | Serveable artefact | Merging in the wrong dtype; quantizing after merging loses quality |
| 9 Serve | Artefact | vLLM/TGI/Ollama, autoscaling | Endpoint | No version pinning; no rollback; KV cache OOM at concurrency |
| 10 Iterate | Production failures | Mine failures → new data | Next dataset version | No feedback loop ⇒ the model is frozen while the world moves |

---

## 6. Hands-On Code (annotated)

The video's demos are live-coded on Colab and are not shipped as notebooks. Reconstructed faithfully from the narration at [59:01–1:03:50], modernised, and extended with the tokenizer forensics the video shows but does not explain.

### 6.1 Demo 1 — closed-label-set inference (the tomato problem)

This is the instructor's ResNet demo [59:19–1:01:57]. He loads ImageNet weights, predicts a dog, then predicts a tomato and gets "strawberry, pitcher, orange."

```python
# requires: torch, torchvision, pillow, requests
# pip install torch torchvision pillow requests
import io, requests, torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

# WHY: weights=... both downloads the pretrained weights AND attaches the
# correct preprocessing transform. Using a hand-written transform that does not
# match training is one of the top sources of silent inference-quality loss.
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()          # eval() disables dropout/batchnorm updates
preprocess = weights.transforms()                 # resize 256 -> center-crop 224 -> normalize
categories = weights.meta["categories"]           # the 1000 ImageNet class names

def predict(url: str, topk: int = 5):
    img = Image.open(io.BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
    with torch.inference_mode():                  # inference_mode > no_grad: no version counters
        logits = model(preprocess(img).unsqueeze(0))       # (1, 3, 224, 224) -> (1, 1000)
        probs = logits.softmax(dim=-1)[0]                   # softmax over the 1000 classes
    return [(categories[i], float(probs[i]))
            for i in probs.topk(topk).indices]

# The instructor's dog image -> Labrador / golden retriever.
# A tomato image -> "strawberry", "pitcher", "orange" -- because "tomato" is NOT
# one of the 1000 ImageNet classes. The model does not say "unknown"; it returns
# the nearest neighbour in a label space that excludes the right answer.
for label, p in predict("https://images.unsplash.com/photo-1543466835-00a7907e9de1"):
    print(f"{p:6.2%}  {label}")
```

**What to change for your own data.** Replace the final linear layer (`model.fc = nn.Linear(2048, n_classes)`) and train only it — that is "feature extraction," the instructor's transfer-learning "way 1" [40:28]. Unfreeze the last block (`model.layer4`) as well and you have "way 2" [40:38–40:52].

**The transferable lesson for LLM work:** every model, including a frontier LLM, is a closed-label-set classifier at heart. Ask it about something outside its training distribution and it returns a confident nearest neighbour. That is what hallucination *is*.

### 6.2 Demo 2 — masked language modelling with BERT (the instructor's `[MASK]` demo)

He runs `"The capital city of France is [MASK]"`, tokenises it, runs a forward pass, applies *"softmax over vocabulary plus take top prediction for the position"* [1:03:02–1:03:08], and gets **Paris** as the highest-probability token [1:03:15–1:03:28].

```python
# requires: transformers, torch   (pip install "transformers>=4.44" torch)
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_ID = "google-bert/bert-base-uncased"        # 110M params, WordPiece, vocab 30,522
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID).eval()

text = "The capital city of France is [MASK]."

# WHY tokenizer(model_id) and not a hand-built vocab: the id for [MASK] must equal
# the id the model was trained with (103 for BERT). A mismatch = silent garbage.
enc = tokenizer(text, return_tensors="pt")
mask_pos = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.inference_mode():
    logits = model(**enc).logits                 # (1, seq_len, 30522) -- one score per vocab item
probs = logits[0, mask_pos, :].softmax(dim=-1)   # restrict to the masked position

topk = probs.topk(5)
for score, idx in zip(topk.values[0], topk.indices[0]):
    print(f"{float(score):6.2%}  {tokenizer.decode([idx])!r}")

# Expected: ~0.99 'paris', then 'france'/'lyon'/'nice'-class continuations.
# Note what this model CANNOT do: generate. It has no causal mask and no
# left-to-right decoder. BERT fills one blank in a fixed-length sequence.
```

**Why this demo matters more than it looks.** It shows three things at once: (1) a pretrained encoder already *knows* facts, so the value of fine-tuning is access, not knowledge; (2) the vocab-sized softmax at the masked position is literally the same operation as an LLM's next-token head, just bidirectional and applied to 15% of positions; (3) `AutoModelForMaskedLM` vs `AutoModelForCausalLM` vs `AutoModelForSeq2SeqLM` is the architecture decision that determines which pretraining objective you can fine-tune.

### 6.3 Demo 3 — the pipeline the instructor mentions but does not show: tokenizer forensics

He mentions the SST-2 sentiment demo at [1:03:34–1:03:47] and says *"we'll discuss the tokenizer in upcoming sessions."* Here is the check every practitioner should run **before** any fine-tune, because it catches four of the eight tokenizer bugs in §4.8.

```python
# pip install transformers
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"   # gated: needs an HF token
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# --- CHECK 1: does it have a pad token? (Llama-class tokenizers do NOT) ---
print("pad:", tok.pad_token, "| eos:", tok.eos_token, "| pad_id:", tok.pad_token_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token          # standard fix for decoder-only models
print("vocab size:", len(tok))             # 128256 for Llama-3.1

# --- CHECK 2: inspect the RAW chat-template string, never trust it blindly ---
msgs = [{"role": "system",    "content": "You are terse."},
        {"role": "user",      "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris."}]
rendered = tok.apply_chat_template(msgs, tokenize=False)
print(repr(rendered))
# Llama-3.1 should emit:
# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are terse.<|eot_id|>'
# '<|start_header_id|>user<|end_header_id|>\n\nCapital of France?<|eot_id|>'
# '<|start_header_id|>assistant<|end_header_id|>\n\nParis.<|eot_id|>'
# If you see ChatML (<|im_start|>) or [INST], the WRONG model/template pair is in play.

# --- CHECK 3: token cost per word, per language (this is your inference bill) ---
for lang, s in [("english", "The capital of France is Paris."),
                ("hindi",   "फ्रांस की राजधानी पेरिस है।"),
                ("code",    "def f(x): return x**2")]:
    n = len(tok(s)["input_ids"])
    print(f"{lang:8s} {n:3d} tokens | {n/len(s.split()):5.2f} tokens/word")

# --- CHECK 4: build the training example and CONFIRM the supervised span ---
enc = tok.apply_chat_template(msgs, tokenize=True, return_assistant_tokens_mask=True,
                              return_dict=True)
ids, amask = enc["input_ids"], enc["assistant_masks"]
labels = [t if m else -100 for t, m in zip(ids, amask)]   # -100 = ignored by CrossEntropyLoss
print("supervised tokens:", sum(1 for l in labels if l != -100), "/", len(labels))
print("decode of supervised span:", repr(tok.decode([t for t in ids if t in
      [i for i, l in zip(ids, labels) if l != -100]])))
```

**What to change for your own data.** Point `MODEL_ID` at your base; if the supervised span does not decode to exactly your assistant response (plus `<|eot_id|>`), **do not start training.** That single check prevents the most expensive class of silent SFT failure.

### 6.4 The instructor's pretraining loop, in the smallest form that is honest

He draws this loop on the board at [53:32–53:45] and again at [34:54–35:10]: *"we collect the data, we tokenize the data, we pass it to the transformer, we predict the next token, we calculate the loss, we do the back propagation, then repeat."* Here it is as executable code — a 2-layer character-level transformer trained on one sentence, which is the smallest thing that exhibits the real mechanism.

```python
# pip install torch
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
text = "sunny is ai master. sunny is an ai master."   # the instructor's example sentence [56:04]
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)                                          # vocabulary size
data = torch.tensor([stoi[c] for c in text])

BLOCK, D_MODEL, N_HEAD, N_LAYER = 16, 64, 4, 2
x = data[:-1].unfold(0, BLOCK, 1)                       # (n_blocks, BLOCK) inputs
y = data[1:].unfold(0, BLOCK, 1)                        # (n_blocks, BLOCK) targets -- SHIFTED BY 1

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))
    def forward(self, h):
        causal = torch.triu(torch.ones(h.size(1), h.size(1), dtype=bool), diagonal=1)
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=causal)
        h = h + a                                        # residual
        return h + self.mlp(self.ln2(h))                 # residual

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, V, bias=False)     # the LM head, tied in real models
    def forward(self, idx):
        h = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln_f(self.blocks(h)))       # (batch, BLOCK, V) -- logits

model = TinyLM()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)

for step in range(400):
    logits = model(x)                                     # forward:  (n, BLOCK, V)
    loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))   # cross-entropy over all positions
    opt.zero_grad(set_to_none=True)
    loss.backward()                                       # backprop:  dL/dW for every weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # stabilise
    opt.step()                                            # AdamW update
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}  ppl {loss.exp().item():6.2f}")

# Sample: feed a prompt, take the argmax, append, repeat -- that is autoregressive decoding.
model.eval()
idx = torch.tensor([[stoi[c] for c in "sunny is "]])
with torch.inference_mode():
    for _ in range(12):
        nxt = model(idx[:, -BLOCK:])[0, -1].argmax().item()
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
print("generated:", "".join(chars[i] for i in idx[0]))
```

**What to change for your own data.** Nothing about the *shape* — this is the architecture of every decoder-only LLM. What changes at scale: `D_MODEL 64 → 4096`, `N_LAYER 2 → 32`, `V 20 → 128,256`, `BLOCK 16 → 8192`, one sentence → 15T tokens, one CPU → thousands of GPUs, and the `for step` loop becomes a distributed training run with gradient accumulation and checkpointing. **The mechanism is identical.**

---

## 7. Hyperparameters & Configuration — Every Knob

The videos name no hyperparameters — this is a foundations module, and the instructor defers them to later sessions. The table below is therefore the practitioner baseline you need in order to read every later module's config.

| Param | What it does | Typical | Safe range | Too high → | Too low → | Framework flag |
|---|---|---|---|---|---|---|
| **learning_rate** | Step size `η` | 2e-5 (full FT) · 2e-4 (LoRA) · 1e-4 (QLoRA) | full FT 5e-6–5e-5 · LoRA 1e-4–3e-4 | Loss spikes/NaN, then divergence | Learning nothing visible in 200 steps | `learning_rate` |
| **lr_scheduler_type** | Shape of `η(t)` | `cosine` | `cosine`, `linear`, `constant_with_warmup` | With `constant`, final loss is worse | — | `lr_scheduler_type` |
| **warmup_ratio / warmup_steps** | Ramp before peak LR | 0.03–0.05 (3–5% of steps) | 0.01–0.10 | Wasted steps at low LR | Early instability with Adam | `warmup_ratio` |
| **num_train_epochs** | Passes over the data | 1–3 SFT | 1–3; >3 needs evidence | Memorisation; val loss diverges from train loss | Underfit; the model does not adopt the behaviour | `num_train_epochs` |
| **per_device_train_batch_size** | Micro-batch per GPU | as large as VRAM allows (1–8) | 1–16 | OOM | Slow, noisy steps | `per_device_train_batch_size` |
| **gradient_accumulation_steps** | Micro-batches per step | 4–16 | 1–64 | Very slow wall clock; LR may need raising | Noisy global batch | `gradient_accumulation_steps` |
| **global batch (tokens)** | micro × accum × GPUs × seq | 128k–2M tokens | 64k–4M | Diminishing quality; LR must rise | Unstable optimisation | derived |
| **max_seq_length** | Truncation length | 1024–4096 SFT | 512–8192 | `s²` memory blowup; truncation still bites | Answers truncated — data destroyed | `max_seq_length` |
| **weight_decay** | Decoupled L2 (AdamW) | 0.01 | 0.0–0.1 | Over-regularised, underfits | Overfits on small data | `weight_decay` |
| **adam_beta1 / beta2** | Moment decay rates | 0.9 / **0.95** | 0.9/0.95–0.999 | beta2=0.999 adapts too slowly for LLMs | beta2 too low ⇒ noisy | `adam_beta1/2` |
| **max_grad_norm** | Gradient clipping threshold | 1.0 | 0.3–1.0 | Slow, never escapes plateaus | Spikes propagate ⇒ NaN | `max_grad_norm` |
| **lr / LoRA `r`** | Adapter rank | 16 | 8–64 (128+ rarely helps) | Overfits small data; more params | Underfits complex new behaviours | `r` |
| **LoRA `alpha`** | Scaling `α/r` | 16 (= r) or 32 | 8–64 | Effective LR too high | Adapter too weak | `lora_alpha` |
| **LoRA `dropout`** | Regularisation on adapters | 0.05 | 0.0–0.1 | Underfits | Overfits small data | `lora_dropout` |
| **LoRA `target_modules`** | Which matrices get adapters | all linear layers | attention-only is a common mistake | More VRAM | `q,v` only ⇒ notably worse quality | `target_modules` |
| **gradient_checkpointing** | Recompute activations | True for LLM FT | On/off | ~20–33% slower | OOM at useful batch sizes | `gradient_checkpointing` |
| **bf16 / fp16** | Compute precision | bf16 on Ampere+ | bf16 preferred | — | fp16 without loss scaling ⇒ NaN | `bf16`, `fp16` |
| **packing** | Concatenate short examples into full-length sequences | True for pretraining; **careful** for SFT | depends | Cross-contamination between unrelated examples if not masked | 40–70% wasted padding compute | `packing` / `packing_strategy` |
| **eval_steps / save_steps** | Cadence | 50–200 | — | Slow training | Coarse curves; missed overfit point | `eval_steps` |
| **seed** | RNG seed | 42 | any | — | — | `seed` |

### 7.1 The knobs that actually matter, and their interactions

**Learning rate × method.** This is the most common pairing error. Full fine-tuning with `lr=2e-4` destroys a pretrained model within 100 steps — the update magnitude relative to the pretrained weights is far too large, and the model forgets the pretraining distribution before it learns the new task. LoRA tolerates `2e-4` because the adapters start at zero and the base is frozen. QLoRA often wants a slightly lower LR than LoRA (`1e-4`) because NF4 dequantization introduces quantization noise into the gradient path.

**Global batch × learning rate.** These are coupled. Scaling the global batch by 4x without raising the LR means 4x fewer steps at the same step size — strictly less optimisation. The practical rule (linear scaling, then damped):

```
LR_new = LR_base × sqrt(global_batch_new / global_batch_base)     # conservative
LR_new = LR_base ×     (global_batch_new / global_batch_base)     # aggressive, needs warmup
```

**Sequence length × everything.** VRAM scales roughly linearly with `s` (activations, flash attention) or `s²` (attention without flash attention, KV cache grows linearly). Doubling `max_seq_length` from 1024 to 2048 roughly doubles activation memory and doubles per-step time. Before raising it, ask whether your data actually needs it: if your p99 example is 900 tokens, `max_seq_length=2048` costs 2x for nothing.

**Epochs × dataset size.** With a small, high-quality dataset (1k–5k), 3 epochs is often right and 10 will memorise. With a large dataset (100k+), 1–2 epochs is usually enough. The diagnostic is the **gap between train and eval loss**, not the absolute value.

**Gradient checkpointing × wall clock.** Turning it on costs ~20–33% more compute but cuts activation memory ~5–10x. It is almost always the correct trade for LLM fine-tuning, because the alternative is a smaller batch — and a smaller global batch hurts quality more than the extra compute hurts throughput.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 The escalation ladder (the instructor's RAG/agent question, CS-04 in one table)

| Situation | Use this? | Instead use | Why |
|---|---|---|---|
| Model does not know the facts | **No** — not fine-tuning | RAG | Fine-tuning teaches form, not retrievable, citable, updatable facts |
| Facts change weekly | **No** | RAG | A fine-tune is a snapshot; updating means retraining |
| Model ignores your output format | **Yes** | — | Format is exactly what SFT is best at. This is the canonical fine-tuning case |
| Model needs a specific tone/persona | **Yes** | Prompting first | SFT reliably locks style; prompting is cheaper to try first |
| Domain jargon is mangled | **Yes** | RAG + glossary | Domain vocabulary is a form/behaviour problem; continued pretraining (CS-12) for heavy jargon |
| Task needs reasoning the base lacks | **No** | Bigger model, or a better prompt | Fine-tuning on 1k examples will not install reasoning capability |
| 2,000-token few-shot prompt costs too much per call | **Yes** | — | The strongest financial case for fine-tuning. See §11.4 |
| Latency-critical, no room for a big prompt | **Yes** | Distillation (CS-09) | Fewer prompt tokens = faster TTFT |
| You have 50 labeled examples | **No** | Few-shot prompting | 50 examples is a prompt, not a dataset |
| You have no eval set | **No** | Build one first | You cannot know if the fine-tune worked, so you cannot ship it |
| You need structured JSON output reliably | **Yes** | Constrained decoding (grammar/JSON-mode) | Constrained decoding is cheaper and exact — try it first, then SFT for *semantic* correctness |
| The model refuses benign domain queries | **Yes** | System prompt | Refusals that survive prompt iteration are an alignment-behaviour problem — SFT fixes it |
| You need to remove a capability (safety) | **Yes** | — | Abliteration-style SFT is the mechanism; treat it as a safety-critical change with its own eval suite |

### 8.2 STOP conditions — signals fine-tuning is the wrong tool

1. **You cannot write down the metric you expect to move.** Fine-tuning without a metric is expensive guessing.
2. **The failure is factual, not behavioural.** "It gets our product SKUs wrong" → retrieve them. "It formats SKUs wrong" → fine-tune.
3. **Your examples are all the same template.** 5,000 rows of one question shape trains a template-follower, not a general behaviour. Add diversity or do not bother.
4. **Your best prompt already passes the eval.** Then the remaining gap is data quality, not model weights.
5. **You have fewer than ~200 examples and no ability to get more.** Use few-shot prompting and spend the budget on retrieval.
6. **No held-out set exists and nobody will build one.** You will ship a regression you cannot see.
7. **The base model has never seen your language or domain at all.** You need continued pretraining (CS-12), not SFT.
8. **The plan is "fine-tune to fix hallucinations."** This reliably makes them worse by teaching the model the *style* of confident wrongness.

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros of the pretrain → fine-tune paradigm

| Pro | Quantified |
|---|---|
| Massive label efficiency | 100–5,000 examples vs 10k–1M from scratch |
| One backbone, many tasks | One 7B base + N adapters instead of N models |
| Cheap at the margins | A 7B QLoRA on 10k examples ≈ 3.5 GPU-hours ≈ $2–5 (§4.4c) |
| Composable with RAG and agents | The instructor's own recommended architecture [3:51–4:00] |
| Deployable at the edge | A quantized 3B beats a hosted API on latency, cost and privacy |
| Versionable | Adapters and checkpoints are artefacts you can A/B and roll back |
| No data leaves your control | The decisive argument in regulated industries |

### 9.2 Cons

| Con | Quantified |
|---|---|
| Not reversible cheaply | Rollback = re-serve a checkpoint; A/B needs two deployments |
| High fixed cost of the *pipeline* | Data, eval, serving, monitoring — not the GPU bill |
| Needs an eval set to be meaningful | Building one is often 60% of the project |
| Catastrophic forgetting | Measurable drops on unrelated benchmarks are common after narrow SFT |
| Alignment regression | 10 adversarial examples can break safety (Qi et al. 2023) |
| Data must be in-distribution | Out-of-distribution SFT data teaches nothing useful |
| Frozen snapshot | Facts age; the model does not |

### 9.3 Hard limitations — things fine-tuning cannot do

| Limitation | Why | Correct alternative |
|---|---|---|
| Install reliable, updatable, citable knowledge | The fact is diffused across billions of weights with no retrieval address | RAG (CS-04) |
| Add capability the base lacks | SFT reshapes a distribution; it does not add reasoning circuits | A bigger base, or distillation (CS-09) |
| Exceed the base's ceiling | The ceiling is set by pretraining | Start from a stronger base |
| Fix a tokenizer/format problem | Those are harness bugs | Fix the harness (§4.8) |
| Guarantee correctness | A model is a distribution, not a database | Guardrails, validation, retrieval with citations |
| Make a 1B model act like a 70B | The capacity is not there | Distillation gets closer; nothing closes the gap |
| Train on data you have no right to use | Licence law applies to your fine-tune too | Licence review before training (§4.9) |

### 9.4 Silent failure modes — looks fine, is broken

| Failure | What it looks like | Why it is silent | Detection |
|---|---|---|---|
| **Wrong chat template** | Loss falls smoothly to 0.8; model is unusable in the app | The model learns *some* format perfectly — just not yours | Print the rendered template string; chat with the model before/after |
| **Loss on prompt tokens** | Eval loss looks excellent | The model learned to generate questions, which is 90% of the tokens | Decode the supervised span; check label distribution |
| **Train/test leakage** | Test accuracy 0.97 on a 2k-example dataset | Near-duplicate rows in both splits | Dedup across splits, not within |
| **Truncation eating the answer** | Eval loss is fine; long inputs produce truncated outputs | Truncation is applied at the *right* by default | Log input length distribution; count truncated examples |
| **LR too low** | Loss decreases 0.02 over an epoch; "training works, slowly" | Nothing errors | Compare to a 5x-higher-LR run for 100 steps |
| **LR too high** | Loss decreases, then plateaus above where it should | Divergence is partial, not total — the model is in a bad basin | Watch grad-norm; try 5x lower |
| **Overfitting (small data, many epochs)** | Train loss 0.2, eval loss 2.1 | Teams report train loss | Always plot both; early-stop on eval |
| **Adapter saved but never merged/loaded** | Inference uses the base model; results identical to baseline | You get a plausible, unremarkable answer | Diff base vs adapter output on a probe prompt |
| **Quantized the wrong thing** | Quality dropped 8 points after "optimisation" | Nobody re-ran the eval after quantization | Re-evaluate post-quantization, always |
| **Frozen embedding for added tokens** | New domain tokens produce random output forever | Loss still falls on the rest of the vocabulary | Confirm new embedding rows are in `requires_grad` |
| **Eval set contaminated by the base model** | Fantastic benchmark scores that do not reproduce in your app | You tested on the model's own training data | Build an internal post-cutoff eval set |
| **No version pinning in the training data** | Results are unreproducible months later | Data was regenerated from a live source | Hash and version the dataset |

---

## 10. Exceptions, Edge Cases & Gotchas

1. **The exception to "SFT needs thousands of examples": format-only alignment.** If the base model already has the capability and you are only changing the output *shape*, 200–500 examples is genuinely enough (LIMA's thesis at 1,000). The exception to the exception: those examples must be diverse — 500 paraphrases of one prompt teach a template.
2. **The exception to "never fine-tune for facts": fixed, small, high-value fact sets.** If your product's entire knowledge is 40 product names and their specs, SFT on them works and is simpler than a retrieval stack. It becomes wrong the moment the facts change, so gate it on a re-training SLA.
3. **The exception to "base models are useless": continued pretraining.** For a new language or a genuinely new domain (protein sequences, a programming language that did not exist at training time), you *must* start from a base model and do continued pretraining (CS-12). Starting from an instruct model here fights the alignment.
4. **The exception to "3 epochs is the cap": pretraining and continued pretraining.** Those run for a *fraction* of one epoch over a corpus. Epoch counts are meaningless at corpus scale; token counts are the unit.
5. **The exception to "lower LR is safer": LoRA.** Because adapters initialise at zero, LoRA tolerates and often needs a 10x higher LR than full FT.
6. **The exception to "gradient checkpointing costs 33%": PyTorch's `use_reentrant=False` and selective checkpointing.** Modern implementations can checkpoint only the expensive blocks, recovering most of the compute.
7. **The exception to "more data is better": duplicated data.** Duplicating a subset to rebalance classes effectively raises its weight and causes overfitting to it. Dedup first, then re-weight with a sampler.
8. **The exception to "bigger vocab is better": small, single-language deployments.** A 32k English-only vocabulary gives a smaller embedding matrix and a cheaper softmax. If you serve one language, a 256k multilingual vocabulary is 900M wasted parameters.
9. **Gotcha: `padding_side` differs between training and generation.** Right padding is the Trainer default and is fine for training when the label mask is correct; batched *generation* on a decoder-only model requires left padding, or generation starts from pad tokens.
10. **Gotcha: the `loss` you see is the mean over non-ignored tokens.** If you mask 80% of tokens as prompt, the reported loss is over 20% of tokens — a different quantity from a run that masks nothing. Two configs' loss curves are not comparable unless the masking is identical.
11. **Gotcha: `tokenizer.save_pretrained` and `model.save_pretrained` must go to the same directory.** Shipping an adapter with the base's tokenizer but a modified chat template silently mis-serves.
12. **Gotcha: merging LoRA in fp16 loses quality.** Merge into fp32 weights if your framework allows, then cast down. For QLoRA, merging into an NF4 base is not supported by most toolchains — keep the base quantized and load the adapter at serve time, or dequantize to 16-bit, merge, then re-quantize.
13. **Gotcha: the base model's licence attaches to your adapter.** See §4.9. Publishing a LoRA trained on a licence-restricted base is redistribution of a derivative.
14. **Gotcha: `eos_token` is not always the stop token.** Llama-3 uses `<|eot_id|>` at turn boundaries and `<|end_of_text|>` at document boundaries. Setting only one in `generation_config.stop_strings` produces models that either never stop or stop mid-turn.

---

## 11. Cost, Compute & Memory

### 11.1 The three formulas, in the order you should use them

```
1. VRAM   : M = P × bytes_per_param  +  activations  +  overhead
2. FLOPs  : train = 6 × N × D        inference = 2 × N × tokens
3. Time   : wall_clock = FLOPs / (n_gpus × peak_TFLOPs × MFU)
4. Money  : $ = n_gpus × hours × price_per_gpu_hour
```

Worked in sequence for a **7B full FT on 20,000 examples averaging 800 tokens, 3 epochs**:

```
D  = 20,000 × 800 × 3                     = 48,000,000 tokens = 4.8e7
FLOPs = 6 × 7e9 × 4.8e7                    = 2.016e18
VRAM  = 112 GB (states) + ~5 GB (act, ckpt) + ~3 GB (overhead) ≈ 120 GB
      → 2 × A100-80GB minimum, 4 for comfort and speed
Time  = 2.016e18 / (2 × 312e12 × 0.40)     = 8,077 s ≈ 2.24 h
Money = 2 GPUs × 2.24 h × $1.79/h          ≈ $8.03      (Lambda A100-80GB, ballpark)
```

The same job as **QLoRA on one rented A100-40GB**:
```
VRAM  ≈ 4 GB (NF4 base) + 0.5 GB (adapters+Adam) + 6 GB (act) ≈ 11 GB
Time  = 2.016e18 / (1 × 312e12 × 0.28)     = 23,080 s ≈ 6.4 h   (NF4 dequant overhead)
Money = 1 × 6.4 h × $1.29/h                ≈ $8.26
```
**Nearly identical cost.** QLoRA's advantage is not price per FLOP — it is that you can *run it at all* on a 16–24 GB card you already own, where the full-FT job has no on-ramp.

### 11.2 Real VRAM numbers (the beyond-the-video table, repeated here for lookup)

| Model | Full FT (AdamW+AMP) | LoRA (bf16, r=16) | QLoRA (NF4, r=16) | Inference bf16 | Inference 4-bit |
|---|---|---|---|---|---|
| 1B | 16 GB + act ≈ **24 GB** | **8–10 GB** | **5–6 GB** | 2 GB | 0.6 GB |
| 3B | 48 GB + act ≈ **60 GB** | **12–16 GB** | **7–9 GB** | 6 GB | 2 GB |
| 7B | 112 GB + act ≈ **128 GB** | **20–27 GB** | **10–16 GB** | 14 GB | 4 GB |
| 8B | 128 GB + act ≈ **145 GB** | **22–30 GB** | **11–17 GB** | 16 GB | 4.5 GB |
| 13B | 208 GB + act ≈ **230 GB** | **32–40 GB** | **16–22 GB** | 26 GB | 7 GB |
| 34B | 544 GB + act ≈ **580 GB** | **80–100 GB** | **24–32 GB** | 68 GB | 18 GB |
| 70B | 1,120 GB + act ≈ **1.3 TB** | **170–220 GB** | **48–64 GB** | 140 GB | 38 GB |

Assumptions: seq 1024–2048, micro-batch 1–2, gradient checkpointing on for LoRA/QLoRA. Add 30–50% headroom for fragmentation and framework overhead before you commit to a GPU tier.

**Minimum-viable hardware, by job:**

| Job | Minimum | Comfortable |
|---|---|---|
| QLoRA 7B | 1 × T4 16 GB (slow) | 1 × RTX 4090 24 GB / A10G |
| QLoRA 13B | 1 × RTX 3090/4090 24 GB | 1 × A6000 48 GB |
| QLoRA 70B | 2 × A6000 48 GB | 1 × A100 80 GB |
| LoRA 7B | 1 × A100 40 GB | 1 × A100 80 GB |
| Full FT 7B | 2 × A100 80 GB | 4 × A100 80 GB |
| Full FT 13B | 4 × A100 80 GB | 8 × A100 80 GB |
| Full FT 70B | 16 × A100 80 GB | 32–64 × H100 80 GB |
| Pretrain 7B (Chinchilla) | 8 × A100 80 GB | 64 × A100 80 GB |

### 11.3 What GPU-hours actually cost

> **Beyond the video:** the transcript says only *"this could be expensive"* [4:32] and *"how we can train it at a minimal cost"* [4:45]. Here are the actual numbers to budget with. **All figures are ballpark on-demand list prices as of 2025–2026 and must be re-verified before you commit a budget** — spot and reserved pricing can be 50–70% lower, and provider pricing changes frequently.

| Provider / SKU | GPU | $/GPU-hour (ballpark) | Notes |
|---|---|---|---|
| **RunPod** (community cloud) | RTX 4090 24 GB | $0.34–0.69 | Cheapest practical 24 GB |
| RunPod | A100 80 GB | $1.64–2.19 | Community vs secure cloud |
| RunPod | H100 80 GB | $2.49–3.79 | PCIe vs SXM |
| **Lambda Labs** | A100 40 GB | ~$1.29 | On-demand |
| Lambda Labs | A100 80 GB | ~$1.79 | The reference workhorse price |
| Lambda Labs | H100 80 GB | ~$2.49–3.29 | On-demand |
| **AWS** | p4d.24xlarge (8 × A100 40 GB) | $32.77/hr → **$4.10/A100-h** | On-demand; spot ~50–70% off |
| AWS | p5.48xlarge (8 × H100 80 GB) | ~$98.32/hr → **$12.29/H100-h** | On-demand; spot dramatically cheaper |
| **GCP** | a2-highgpu-1g (1 × A100 40 GB) | ~$3.67 | On-demand |
| GCP | a3-highgpu-8g (8 × H100 80 GB) | ~$88/hr → **$11/H100-h** | On-demand |
| **Colab Pro+** | A100 40 GB / L4 | ~$50/month | Compute-unit limited; fine for QLoRA experiments |
| **Kaggle** | 2 × T4 16 GB or P100 | free, ~30 GPU-h/week | Enough for QLoRA ≤3B and all coursework |

**Cost per job, computed:**

| Job | Config | Time | Cost (ballpark) |
|---|---|---|---|
| QLoRA 7B, 10k examples, 3 epochs (15M tokens) | 1 × RTX 4090 (RunPod) | ~3.5 h | **~$1.50–2.40** |
| Same on A100-80GB | 1 × A100 | ~1.4 h | **~$2.50** |
| LoRA 7B, 100k examples, 1 epoch (100M tokens) | 4 × A100-80GB | ~2.4 h | **~$17** |
| Full FT 7B, 20k examples, 3 epochs (48M tokens) | 2 × A100-80GB | ~2.2 h | **~$8** |
| Full FT 13B, 100k examples, 1 epoch (100M tokens) | 4 × A100-80GB | ~5.8 h | **~$42** |
| Full FT 70B, 1B tokens | 32 × A100-80GB | ~19 h | **~$1,090** |
| Pretrain 7B, Chinchilla-optimal (140B tokens) | 64 × A100-80GB | ~204 h (8.5 d) | **~$23,400** |
| Pretrain 8B, Llama-3-style (15T tokens) | 2,048 × H100 | ~150 h (6.3 d) | **~$4.7M** (published-class estimate) |

The last two rows are the point of this whole module: **fine-tuning costs tens of dollars; pretraining costs tens of thousands to millions.** You are renting the second one for free by starting from a pretrained checkpoint. That is the economic content of the word "foundation model."

### 11.4 The break-even calculation that justifies fine-tuning (do this arithmetic before every project)

Suppose your app sends a 2,500-token few-shot prompt per request and receives 300 tokens. At `gpt-4o-mini`-class pricing (input ~$0.15/1M, output ~$0.60/1M, list price ballpark):

```
Baseline per request : 2,500 × $0.15/1e6 + 300 × $0.60/1e6 = $0.000375 + $0.000180 = $0.000555
At 1M requests/month : $555/month = $6,660/year
```

After fine-tuning, a 300-token prompt suffices:
```
Fine-tuned per request: 300 × $0.15/1e6 + 300 × $0.60/1e6 = $0.000045 + $0.000180 = $0.000225
At 1M requests/month  : $225/month = $2,700/year
Saving                : $330/month = $3,960/year
```

Now compare to the *self-hosted* option. A QLoRA 7B on one A10G-class instance (~$0.75–1.00/hr, 24/7 ≈ $550–730/month) serving ~50 req/s at batch — the compute cost is often comparable to or below the API bill at that volume, but you now own uptime, autoscaling, evals and on-call. **The fine-tuning decision is almost never "can we afford the training run." It is "is the prompt 8x longer than it needs to be, and can we live with owning the endpoint."**

The instructor's version of this instinct is at [4:32–4:54]: *"This could be expensive... so how we can train it at a minimal cost, what frameworks, what new research — with minimal cost, with no cost, we can fine-tune our model."* The modern answer is: rent by the hour, start with QLoRA, and validate with a small run before you scale.

### 11.5 The Chinchilla arithmetic, consolidated

```
Rule            : D_opt ≈ 20 × N                (compute-optimal training)
7B model        : D_opt = 20 × 7e9              = 1.4e11 tokens = 140B tokens
Corpus volume   : ≈ 105 GB as uint16 token ids  (2 bytes/token)
                  ≈ 560 GB as uint32 token ids  (4 bytes/token)
                  ≈ 100–140B English words      (0.75 words/token)
Pretrain FLOPs  : 6 × 7e9 × 1.4e11              = 5.88e21 FLOPs
A100-80GB hours : 5.88e21 / (312e12 × 0.40)     = 4.71e7 s = 13,060 GPU-hours
On 64 × A100-80 : 204 h ≈ 8.5 days
Cost @ $1.79/h  : ≈ $23,400
```

For comparison, the **inference** cost of that same model if it serves 1M requests/month at 500 output tokens:
```
2 × 7e9 FLOPs/token × 5e8 tokens/month = 7e18 FLOPs/month
At 125 TFLOP/s effective → 56,000 GPU-seconds/month ≈ 15.6 GPU-hours/month  (trivially cheap)
```
**Both numbers matter, and that is the Chinchilla critique in one line:** training cost is paid once for 13,060 GPU-hours; serving cost recurs forever. Optimising the wrong one is how you end up with a 175B model nobody can afford to run.

---

## 12. Evaluation — How To Know It Worked

### 12.1 The metric hierarchy

| Layer | Metric | What it catches | How it lies |
|---|---|---|---|
| **Training** | Loss, perplexity, token accuracy, grad-norm | Broken config, divergence, overfit | Loss is over *tokens*, not answers; a good loss can coexist with a useless model |
| **Task** | Accuracy, F1, exact match, ROUGE, pass@k, schema-validity rate | Whether the target behaviour improved | Overfits to the eval's distribution; contaminated sets inflate it |
| **Regression** | Capability suite (MMLU-lite, GSM8K-lite, safety suite, refusal suite) | Catastrophic forgetting, alignment collapse | Rarely run; expensive; the failure it catches is the expensive one |
| **Judged** | LLM-as-judge pairwise win rate vs baseline | Qualitative improvement, style, helpfulness | Position bias, length bias, self-preference; judge is a model with its own failure modes |
| **Human** | Pairwise preference, Likert, task success | What users actually experience | Expensive, low-N unless designed carefully, rater drift |
| **Production** | Task success rate, escalation rate, user thumbs, latency p95, cost/request | The truth | Slow feedback loop; confounding from other releases |

**The only metric that decides a launch is the production one.** Everything above it is a proxy. Design the proxies so they cannot all be wrong in the same direction.

### 12.2 The held-out protocol (do this, or the rest is theatre)

1. **Split by source, not by row.** If two rows come from the same document, conversation, or user, they belong in the same split. Row-wise splitting of near-duplicate SFT data is the most common cause of "our fine-tune got 97%."
2. **Dedup across splits.** Run the same MinHash pass over the union and remove any test item with a near-duplicate in train.
3. **Always include a base-model baseline**, evaluated with the *best prompt you can write* for it. "Our fine-tune beats the base model" is meaningless if the base model was evaluated zero-shot.
4. **Include a prompt-only baseline.** If `base + 8-shot` matches your fine-tune, you did not need the fine-tune.
5. **Check base-model contamination.** If the base was trained on your eval set, the numbers are fiction. Build a post-cutoff or internal set.
6. **Report a confidence interval.** On 200 examples, a 3-point difference is noise. Use bootstrap CIs; a ±5% band on n=200 is normal.
7. **Freeze the eval and version it.** An eval that changes with the data is a metric that cannot be compared over time.

### 12.3 LLM-as-judge, honestly

**Use it for:** pairwise preference with a strong judge (GPT-4-class) and a rubric; regression detection at scale; cheap pre-screening before human review.
**Do not use it for:** absolute scores, fine-grained numeric quality, anything safety-critical without human confirmation.

**Its known biases:** *position bias* (prefers the first option — always evaluate in both orders and average), *verbosity bias* (prefers longer answers), *self-preference* (prefers its own family's style), *rubric drift* (score distributions shift as you tweak prompts).

### 12.4 A minimal, honest eval script

```python
# pip install transformers datasets torch
# Evaluates: base vs fine-tuned, on a held-out set, with a continuous metric
# (token-level accuracy on the answer) AND a strict metric (exact match).
# This dual-metric design is the practical response to the emergence debate (Sec 4.6).
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER = "./out/lora-sft"           # or None to evaluate the base only
EVAL   = "eval.jsonl"                # one {"messages":[...]} per line, NEVER seen in training

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"            # REQUIRED for batched decoder-only generation

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
if ADAPTER:
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def score(prompt_msgs, reference):
    ids = tok.apply_chat_template(prompt_msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    # STRICT metric: exact match after normalisation
    em = float(pred.strip().lower() == reference.strip().lower())
    # CONTINUOUS metric: token-level F1 -- smooth, sensitive, comparable across scales
    p, r = set(pred.lower().split()), set(reference.lower().split())
    f1 = 2 * len(p & r) / (len(p) + len(r)) if (p or r) else 0.0
    return em, f1, pred

ems, f1s, worst = [], [], []
for line in open(EVAL, encoding="utf-8"):
    row = json.loads(line)
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    ref  = [m["content"] for m in row["messages"] if m["role"] == "assistant"][-1]
    em, f1, pred = score(msgs, ref)
    ems.append(em); f1s.append(f1)
    if f1 < 0.3:
        worst.append((ref, pred))
    print(f"EM={em:.2f} F1={f1:.2f} | {pred[:80]!r}")

n = len(ems)
print(f"\nn={n}  exact-match={sum(ems)/n:.3f}  token-F1={sum(f1s)/n:.3f}")
print(f"failures (F1<0.3): {len(worst)}")
for ref, pred in worst[:5]:
    print(f"  ref={ref[:90]!r}\n  got={pred[:90]!r}\n")
# THE DIAGNOSTIC FROM Sec 4.6: if token-F1 improved smoothly over your checkpoints
# while exact-match stayed at 0.0, you have a FORMAT/TEMPLATE problem, not a
# capability problem. Fix the template before you spend another GPU-hour.
```

**What to change for your own data.** Swap the metric for your task (schema-validity rate for JSON extraction, entity-level F1 for NER, pass@1 for code). Keep the dual strict/continuous structure — it is the cheapest diagnostic you will ever add.

---

## 13. Comparison Tables

### 13.1 Pretraining vs continued pretraining vs SFT vs preference alignment

| Dimension | Pretraining | Continued pretraining (DAPT/TAPT) | SFT / instruction tuning | Preference alignment (RLHF/DPO) |
|---|---|---|---|---|
| Objective | Next-token prediction | Next-token prediction | Cross-entropy on responses | Relative preference between two responses |
| Data | 1T–15T tokens of general text | 1B–100B tokens of domain text | 1k–1M instruction/response pairs | 10k–1M preference pairs |
| Data labels | None (self-supervised) | None | Human/LLM-written responses | Rankings or chosen/rejected pairs |
| Compute | 10³–10⁶ GPU-hours | 10²–10⁴ GPU-hours | 1–10² GPU-hours | 10–10³ GPU-hours |
| Cost (ballpark) | $23k (7B Chinchilla) – $5M+ | $100–$10,000 | **$2–$200** | $50–$2,000 |
| Teaches | Language, world co-occurrence, reasoning circuits | Domain vocabulary and style | Task format, behaviour, refusal patterns | Which of two acceptable answers is preferred |
| Weights changed | All | All | All or adapters | All or adapters |
| Typical epochs | < 1 | 1–3 | 1–3 | 1–2 |
| Failure mode | Undertrained / contaminated | Catastrophic forgetting of general ability | Wrong template; overfitting | Reward hacking; sycophancy; hedginess |
| Module | CS-01 | CS-12 | CS-13 | CS-14 §4.6.1–§4.6.10 |

### 13.2 Full fine-tuning vs LoRA vs QLoRA (the decision that recurs in every module)

| Dimension | Full FT | LoRA | QLoRA |
|---|---|---|---|
| Trainable params (7B) | 7,000M (100%) | ~20–40M (0.3–0.6%) | ~20–40M |
| Optimiser memory | 112 GB | 0.5 GB | 0.5 GB |
| Base weights memory | included above | 14 GB (bf16) | 3.9 GB (NF4) |
| Realistic total (7B) | **~128 GB** | **20–27 GB** | **10–16 GB** |
| Minimum hardware | 2 × A100-80 | 1 × A100-40 / 1 × 4090 (tight) | 1 × T4 16 GB (slow) / 1 × 4090 |
| Training FLOPs | 6ND | 6ND (unchanged!) | 6ND (unchanged) |
| Wall clock vs full FT | 1.0x | ~1.0–1.5x | ~1.2–1.8x (dequant overhead) |
| Quality at low data (<10k) | Good | **Comparable or better** | ~1–3 points below LoRA |
| Quality at high data (>1M) | **Best** | 1–3 points below | 2–5 points below |
| Serving | Native | Merge (free) or adapter (small overhead) | Must keep quantized or re-quantize after merge |
| Multi-adapter serving | No | **Yes** — one base, N adapters | Yes |
| Best for | Max quality, big data, big budget | **The default.** 90% of use cases | One-GPU, VRAM-starved, prototypes |

**The rule:** start with QLoRA to prove the pipeline and the data; move to LoRA when you have proven the *data*; go to full FT only when you can demonstrate that LoRA is the bottleneck (measurable gap at ≥100k examples). The exception is a genuinely new behaviour or language, where full FT or continued pretraining is warranted from the start.

### 13.3 Prompting vs RAG vs fine-tuning vs agents

| Dimension | Prompt engineering | RAG | Fine-tuning | Agent |
|---|---|---|---|---|
| Weights change | No | No | **Yes** | No (tools change) |
| Adds knowledge | Some (in-context) | **Yes — citable, updatable** | Unreliably | Yes (via tools) |
| Fixes format/style | Partly | No | **Yes** | No |
| Time to first result | Minutes | Days | Days–weeks | Weeks |
| Per-request cost | High (long prompts) | High (retrieval + context) | **Low** (short prompts) | Highest (multi-turn) |
| Upfront cost | ~0 | Retrieval infra | Data + training + evals | Everything, plus orchestration |
| Freshness | Instant | Instant (re-index) | Stale at training time | Instant |
| Auditability | Prompt is readable | **Citations** | Opaque weights | Traceable |
| Failure mode | Fragile, prompt-injectable | Retrieval misses; distractor context | Forgetting, template bugs | Compounding errors, cost blowup |
| Right first move | **Always** | When facts matter | When behaviour matters | When actions matter |
| Module | — | CS-04 | CS-01…CS-19 | CS-04 |

### 13.4 Base vs instruct — head to head

| Dimension | Base model | Instruct model |
|---|---|---|
| Training | Next-token prediction only | + SFT (+ preference alignment) |
| Ships a chat template | No (raw text completion) | Yes (`tokenizer.chat_template`) |
| Behaviour on a question | Continues the text pattern | Answers |
| Fine-tune data needed for a new task | 10k–100k+ (teaching format from scratch) | 1k–10k (refining an existing behaviour) |
| MMLU-format eval score | Near chance (format failure, not knowledge failure) | Meaningful |
| ICL / few-shot ability | Strong | Strong (slightly reduced by alignment) |
| Best for | Continued pretraining (CS-12), research, custom alignment | **Almost every applied SFT project** |
| Safety | None installed | Installed, and fragile under fine-tuning |
| Failure if you pick wrong | A chat app that writes more questions | A model that will not learn your new format as fast |

---

## 14. Debugging Playbook

The rule for every entry: **change one variable, rerun 50–200 steps, compare.** Most "training is broken" reports are one of the twelve rows below.

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| **Loss flat, ~ln(V)** (e.g. 10.4 for a 32k vocab) | No learning signal at all: LR ≈ 0, labels all `-100`, frozen weights, or data not reaching the model | Print `sum(l != -100 for l in labels)`; print 5 decoded training examples; check `requires_grad` on the LM head | Unmask the response span; unfreeze; verify the LR the scheduler actually emits |
| **Loss decreases then plateaus high** (e.g. stuck at 2.5) | Over-masking (loss computed on too few tokens) or LR too high and the model is in a bad basin | Compare the number of supervised tokens per example against expectations | Lower LR 5x; check the label mask |
| **Loss → 0.1 in one epoch** | Memorisation: too many epochs on too little data, or duplicated examples | Eval loss diverges from train loss; sample generations reproduce training rows verbatim | Dedup; cut epochs to 1–2; add `lora_dropout`; get more data |
| **Eval loss rises while train loss falls** | Textbook overfitting | The gap widens monotonically after some step | Early-stop at the minimum; regularise; reduce epochs |
| **Loss spikes upward mid-run** | LR too high, bad batch, or missing gradient clipping | Grad-norm spikes at the same step | Clip at 1.0; lower LR 2–5x; add warmup; consider skipping the batch |
| **Loss NaN / inf** | fp16 overflow (no loss scaling), LR too high, or a division by zero in a custom loss | Switch to bf16 and see if it disappears | Use `bf16=True` on Ampere+; else enable loss scaling; lower LR |
| **Loss NaN immediately at step 0** | Corrupt data: empty sequences, all-masked labels, or a tokenizer id ≥ `vocab_size` | Print `min/max(input_ids)` vs `model.config.vocab_size` | Filter empty examples; fix the tokenizer/model pairing; check `resize_token_embeddings` |
| **Model answers in the app but not after training** | Adapter not merged *and* not loaded at serve time | Diff the served model's output against a raw base-model call on one probe prompt | Load the adapter at serve time, or merge it; verify the served checkpoint hash |
| **Model ignores your system prompt after fine-tuning** | The system role is absent from your training data (or was dropped by the template) | Print the rendered training string — is `<\|start_header_id\|>system` present? | Include system turns in at least 20% of examples |
| **Model never stops generating** | EOS token masked out of the labels, or the template's stop token is not in `stop_strings` | Decode the last supervised token — is it `<\|eot_id\|>` / `</s>`? | Include the EOS token in the supervised span; set `stop_strings` correctly |
| **Model repeats the question before answering** | Loss was computed on prompt tokens, so it learned to generate prompts | Decode the supervised span (reconstruct it from labels ≠ -100) | Mask prompt tokens to `-100` |
| **Generations are incoherent but eval loss is good** | Position/template mismatch, or you are generating with a different template than you trained with | Compare the exact string sent at inference with the training format | Use `apply_chat_template` on both sides, with identical roles and order |
| **Loss goes down, benchmark does not move** | The benchmark is contaminated, out-of-distribution, or measured with a strict metric that hides smooth gains | Compute a *continuous* metric (token-level F1) on the same outputs | Switch metric; re-split; re-check contamination (§4.6) |
| **OOM at a batch size that fit yesterday** | Sequence-length outliers, fragmentation, or another process on the GPU | Log the max sequence length in the batch; check `nvidia-smi` | Filter long examples; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; reduce micro-batch and raise accumulation |
| **Throughput collapses after enabling gradient checkpointing** | Expected: ~20–33% slower, more if the implementation is reentrant | Compare steps/sec with it on/off at the same batch | Use `gradient_checkpointing_kwargs={"use_reentrant": False}`; use selective checkpointing |
| **Loss is lower than it should be / suspiciously good** | Duplicates between train and eval, or eval data present in training | n-gram overlap between splits | Dedup across splits (§12.2) |
| **Model became worse at everything else** | Catastrophic forgetting from a too-high LR, too many epochs, or narrow data | Run the regression suite against the base | Lower LR; fewer epochs; mix 5–20% general instruction data into the SFT set |
| **Refusals appeared on benign queries** | Safety over-generalisation from narrow SFT data | Try 20 benign prompts in the affected domain | Add benign in-domain examples; reduce epochs; re-run the safety suite |
| **Fine-tune made hallucination worse** | You tried to teach facts via SFT — the model learned the *style* of confident assertion | Check whether wrong answers are stylistically identical to right ones | Move the knowledge to RAG; keep SFT for format |
| **Results do not reproduce** | Unseeded shuffling, non-deterministic kernels, or a regenerated dataset | Re-run the identical config twice and compare the first 20 losses | Pin the seed, `torch.use_deterministic_algorithms(True)`, hash and version the data |
| **Loss is exactly the same every run** | A cached dataset with stale tokenisation, or `train_on_inputs=False` mis-set so nothing is supervised | Print dataset fingerprints | Clear the HF cache; verify the tokenisation actually ran |

---

## 15. Applied Case Studies

### 15.1 Fintech: extracting structured JSON from 40 years of scanned loan documents

**Situation.** A mid-size lender has 2.1M scanned pages (1985–2025) of loan agreements. They need `{borrower, principal, rate, term, covenants[]}` extracted as JSON. A frontier model with a 30-page few-shot prompt costs ~$0.11/page → $231k for the corpus, and p95 latency is 9 s.

**Why fine-tuning.** The task is *format + vocabulary*, not knowledge: the schema is fixed, the domain jargon is narrow (LIBOR/SOFR transition language, covenants), and the document distribution is highly repetitive. This is the canonical fine-tuning case. RAG alone cannot produce a schema reliably; prompting alone is too expensive and too slow.

**Plan.**
- OCR 60k pages, hand-label 3,000 with the target JSON (2 reviewers, ±2% disagreement).
- Continued pretraining is *not* needed — the base already handles legal English. Go straight to SFT.
- Base: `Llama-3.1-8B-Instruct`. Method: **QLoRA r=32, α=64, dropout 0.05, target all linear layers**, `lr=1e-4`, cosine, 3% warmup, 2 epochs, `max_seq_length=4096`, global batch 32, `bf16`.

**Config and result.**
```
Trainable params : 84M / 8.03B  (1.05%)
Hardware         : 1 x A100-80GB (rented)
Wall clock       : 11.4 h  ->  ~$20
Schema-valid JSON: 71% (base + few-shot) -> 99.2% (fine-tuned)
Field-level F1   : 0.68 -> 0.94
Serving          : merged bf16 on 1 x L40S, 5.1 s p95 for 30 pages, ~$0.0009/page
Annual saving    : ~$217k in inference cost at full backfill
```

**What went wrong first.**
1. Truncation ate the JSON for 6% of examples (`max_seq_length=2048`). Raising to 4096 and truncating the *input* from the left fixed it. Loss had looked fine — the eval F1 was 0.71 and nobody understood why.
2. The first run masked nothing, so the model generated the *prompt* before the JSON. Schema-validity fell to 44%. Masking prompt tokens to `-100` fixed it.
3. First regression check found the model had become worse at general chat, so 8% general instruction data was mixed in and the regression suite returned to baseline.

### 15.2 SaaS: fine-tuning to kill a 6,000-token system prompt

**Situation.** A B2B analytics assistant uses a 6,000-token system prompt containing the schema, 14 glossary entries, tone rules and 22 few-shot examples. Cost: `6,000 × $3/1M input` on a mid-tier model = $0.018/request + output, at 900k requests/month = **$16,200/month**.

**Why fine-tuning.** Pure unit economics. The behaviour already works; the prompt is just expensive. This is the strongest financial case for SFT.

**Plan.**
- Mine 12,000 real (question, ideal-answer) pairs from production logs, filtered to thumbs-up or no-escalation, deduped by conversation.
- SFT a 7B with LoRA r=16 on the *distilled* behaviour, `lr=2e-4`, 1 epoch, `max_seq_length=1024`.
- System prompt after fine-tuning: 180 tokens.
- Evaluate with a 500-example held-out set of real production questions, judged pairwise against the incumbent (production responses as the reference), both orderings.

**Result.**
```
Cost before : 6,000 in + 400 out  ~ $0.018 + $0.0024 = $0.0204/req  ->  $18,360/mo
Cost after  :   180 in + 400 out  ~ $0.0005 + $0.0024 = $0.0029/req  ->   $2,610/mo
Saving      : ~$15,750/month (~$189k/year)
Training    : 1.6 GPU-hours on 1 x A100-80GB = ~$2.90
Payback     : under one hour of production traffic
Pairwise win vs incumbent : 54% (a modest but real improvement in tone consistency)
```

**What went wrong first.** The first model overfit the 22 few-shot *templates* rather than the behaviour, and produced formulaic answers that the pairwise judge penalised. Fix: 3x more data diversity, 1 epoch instead of 3, `lora_dropout=0.1`.

### 15.3 Healthcare: a domain SLM on-premises

**Situation.** A hospital network needs a clinical-notes summariser. Zero data egress is permitted (PHI). No GPU cluster exists; budget is one server.

**Why fine-tuning.** Privacy and cost. No API is legal. The domain has genuinely specialised vocabulary (ICD-10 phrasing, medication names, abbreviations).

**Plan.**
- Continued pretraining (CS-12) on 2.4B tokens of de-identified internal notes + public medical text, from `Llama-3.2-3B` **base** (not instruct — the alignment would have to be relearned anyway).
- Then SFT on 8,400 (note, summary) pairs written by clinicians, LoRA r=32.
- Then DPO (CS-14 §4.6.3) on 3,100 clinician preference pairs.
- Deploy 4-bit AWQ on 1 × L40S (48 GB), on-premises.

**Result.**
```
Continued pretraining : 2.4B tokens, 2 x A100-80GB, 31 h, ~$111
SFT                   : 8,400 examples, 3.1 h, ~$5.50
DPO                   : 3,100 pairs, 1.9 h, ~$3.40
Total training        : ~$120  +  ~110 clinician-hours (~$16,500 opportunity cost)
Clinician preference vs general 70B API : 61% wins
Latency               : 4.2 s p95 vs 8.9 s for the API
Marginal cost         : $0/request (on-prem)
```

**The real cost was the clinician time, not the GPUs** — 110 hours to write 8,400 summaries and 3,100 preferences dwarfed the $120 of compute. Budget your expert time, not your GPU time.

**What went wrong first.** Attempting SFT directly on the 3B *instruct* model on 1,200 examples produced a model that wrote summaries in fluent clinical prose *containing plausible but invented medication names*. The vocabulary was absent from the base's distribution. Continued pretraining on the domain corpus fixed it — the vocabulary had to exist before SFT could shape its use.

### 15.4 Support automation: the failed fine-tune that should have been RAG

**Situation.** A company wants a support bot that "knows our product." They fine-tune a 7B on 4,200 chunks of scraped documentation, 3 epochs, `lr=2e-4` LoRA.

**Result.** The model's answers are fluent and confidently wrong. Hallucination rate on a 300-question audit rises from 18% (base + RAG) to 41%. The fine-tuned model learned the *register* of the documentation and its topic vocabulary, but not the specific facts — and it now asserts them with the documentation's authority.

**Diagnosis.** The failure was category-level, not implementation-level: they used a form-changing tool for a knowledge problem.

**Fix.** Keep the fine-tune — but only for *format and tone* (answer in 3 bullets, always cite, never recommend a competitor, refuse out-of-scope questions), trained on 1,100 examples of the desired *interaction*, not on documentation text. Move the knowledge into RAG over the same corpus with a reranker.

**Post-fix numbers:** hallucination 6%, citation rate 97%, deflection 34% (up from 11%).

**The generalisable lesson:** if your training data is *documents*, you are doing RAG with extra steps and worse attribution. If your training data is *demonstrations of behaviour*, you are doing fine-tuning. Look at the shape of your data; it tells you which tool you are holding.

### 15.5 Startup: 3B QLoRA on a single consumer card

**Situation.** Two founders, one RTX 4090 24 GB, no cloud budget beyond ~$50/month. Target: a specialist code-review assistant for one language and one framework.

**Plan.** QLoRA (NF4, double quant) on `Qwen2.5-Coder-3B-Instruct`, r=64, α=128, all linear modules, `lr=2e-4`, 1 epoch, `max_seq_length=2048`, micro-batch 1, `grad_accum=16`, gradient checkpointing on, paged AdamW 8-bit, `packing=False`.

```
Dataset   : 6,300 (diff, review-comment) pairs, hand-curated
VRAM peak : 8.9 GB of 24 GB
Wall clock: 4 h 50 m on the 4090 (~$0.42 of electricity at 450W)
Eval      : 78% of reviews rated "useful" by 3 human raters, vs 41% for the
            same model with a 5-shot prompt
```

**What went wrong first.**
1. `packing=True` concatenated unrelated diffs and the model learned to review code that was not in the input. Turning packing off, and then using a *masked* packing strategy, fixed it.
2. r=16 was not enough for a genuinely new behaviour; r=64 with `lora_alpha=128` was. r=64 costs 0.9 GB extra VRAM and bought 11 points of usefulness.
3. The first deployment served the *base* model for four days because the adapter was saved but the serving config still pointed at the base repo. The bug was invisible: answers were plausible. Only a probe prompt diff caught it.

---

## 16. Production Considerations

### 16.1 Versioning and rollback

Version **five** things together, or you cannot roll back: base model revision (`@sha`), adapter/merged checkpoint hash, tokenizer + chat template (they can drift independently), dataset version (hash the file), and the training config. Log them into the model card. The instructor's instinct — *"if it is not working fine then again we can retrain it"* [4:22] — only works if you know exactly what the previous good state was.

> **Beyond the video:** pin the base by **commit SHA**, not by tag. `from_pretrained("meta-llama/Llama-3.1-8B-Instruct")` resolves to whatever the current `main` is, and model repos *do* get updated (tokenizer fixes, config corrections). A tag that silently moved has invalidated more fine-tunes than any other single cause.

### 16.2 Serving

| Decision | Options | When |
|---|---|---|
| Engine | vLLM (PagedAttention, high throughput), TGI (HF-native), Ollama/llama.cpp (local, GGUF), TensorRT-LLM (NVIDIA, fastest) | vLLM for GPU serving at scale; Ollama for on-device |
| Adapter strategy | Merge into the base (fastest, one model per deployment) or multi-LoRA (one base + N adapters, hot-swappable) | Multi-LoRA when you serve per-tenant variants |
| Precision | bf16 (best quality), FP8 (H100/B200, ~2x throughput), AWQ/GPTQ int4 (2x memory saving, ~1–3 point quality cost) | int4 for cost-sensitive, quality-tolerant tasks |
| Batching | Continuous batching (mandatory at scale) | Always, in a real deployment |
| Capacity | `VRAM = weights + KV_cache × ctx × concurrency + overhead` (§4.4e) | Recompute for every new context length |

**Latency has two halves and they behave differently.** Prefill (TTFT) is compute-bound and scales with prompt length — this is where fine-tuning *helps*, because it shortens the prompt. Decode (tokens/s) is bandwidth-bound and independent of prompt length — fine-tuning does not help here at all. If your latency complaint is TTFT, fine-tuning is a lever. If it is tokens/s, it is not.

### 16.3 Monitoring and drift

| Signal | What it detects | Alarm threshold (suggested) |
|---|---|---|
| Input length distribution | Prompt drift; upstream schema changes | p95 shifts > 30% |
| Output schema-validity rate | Format regression | < 99% for structured tasks |
| Refusal rate | Over-alignment or under-alignment | ±50% relative to baseline |
| Escalation / thumbs-down rate | Quality regression | +20% relative, 3-day moving average |
| Hallucination probe set (fixed 200 items, run nightly) | Factual drift | Any increase > 2 points |
| Latency p95, tokens/s | Capacity regression | Service-level objective |
| Cost per request | Prompt bloat creeping back | +25% |
| Base-model deprecation notices | Forced migration | Provider announcement |

**The hallucination probe set is the highest-value item on this list.** 200 fixed questions with known answers, run on every model release and nightly in production, is the cheapest possible early-warning system, and it is the *only* thing that catches the "we served the base model for four days" class of bug (case study 15.5).

### 16.4 Regression and safety testing

Every fine-tune of an aligned model must pass, before deploy:
1. **Capability regression** — a compact suite (50–200 items covering reasoning, code, math, multilingual, long-context). Alert on a drop beyond the noise band.
2. **Safety regression** — a refusal suite plus a jailbreak suite. Qi et al. (2023) showed 10 adversarial examples suffice to break alignment; assume your 5,000 benign examples also moved it.
3. **Task eval** — the metric you actually care about, with bootstrap CIs.
4. **Adversarial / red-team pass** — for anything user-facing.
5. **Cost and latency budget** — verify the prompt actually got shorter.

### 16.5 The compliance angle

Training data provenance must be documented: source, licence, PII handling, retention. Model cards should disclose the base, the data, the training method, and known limitations. Under the EU AI Act, general-purpose AI providers face training-data-summary disclosure obligations on a phased timeline through 2027, and high-risk deployments face conformity assessment. In regulated sectors (health, finance) the on-premises fine-tune is often chosen not for cost or quality but because **it is the only architecture that can pass a data-protection review** — which is a legitimate and frequently decisive reason to fine-tune.

---

## 17. Common Misconceptions

1. **"Fine-tuning is how you add knowledge."** Actually: fine-tuning adds *behaviour*; retrieval adds *knowledge*. Because knowledge in weights has no address, cannot be updated, and cannot be cited. Fix: RAG for facts, SFT for form.
2. **"A base model and an instruct model are basically the same model."** Actually: they differ in whether the model answers or continues, whether a chat template exists, and how many examples you need to teach a new format (10k–100k vs 1k–10k). Fix: check the model card for "Instruct"/"Chat" and a `chat_template`.
3. **"LoRA is faster than full fine-tuning because it trains fewer parameters."** Actually: it uses essentially the same FLOPs (6ND). It saves *memory*. Reported speedups come from kernel fusion and skipping frozen-gradient buffers, not from the rank decomposition.
4. **"Chinchilla says I need 20 tokens per parameter, so my 7B needs 140B tokens."** Actually: that is the *training*-compute optimum. Production models are deliberately overtrained 14–94x because inference cost dominates lifetime cost. Fix: compute total cost of ownership, not training cost.
5. **"If loss goes down, the fine-tune is working."** Actually: loss is computed over whatever tokens you did not mask, so a wrong mask produces a beautiful loss curve and a useless model. Fix: decode the supervised span.
6. **"More epochs means better results."** Actually: past 1–3 epochs on SFT data, eval loss rises while train loss falls; you are shipping a memoriser.
7. **"Bigger vocabulary is strictly better."** Actually: it adds embedding parameters (128k × 8192 ≈ 1B for Llama-3) and softmax cost, in exchange for shorter sequences. For a single-language deployment, it is a net loss.
8. **"Quantization is free."** Actually: int4 costs ~1–3 points on most tasks, and QLoRA training pays it in every forward pass. Fix: always re-run the eval after quantizing.
9. **"Emergent abilities prove you need scale to get capability X."** Actually: much of the apparent discontinuity is a metric artefact (Schaeffer et al. 2023). Fix: measure continuously before concluding incapability.
10. **"Fine-tuning an aligned model keeps its safety."** Actually: 10 adversarial examples can break it, and a few hundred benign ones can degrade it. Fix: run a safety regression suite on every fine-tune.
11. **"We need a 70B model to compete."** Actually: a fine-tuned 3B beats a generic 70B on a narrow, well-defined task — the hospital case study won 61% of pairwise comparisons against a 70B API. Fix: measure on *your* distribution.
12. **"The GPU bill is the cost of fine-tuning."** Actually: the GPU bill is ~$2–200. The costs are the data, the eval set, and the expert hours (case study 15.3: ~$120 of compute, ~$16,500 of clinician time).
13. **"RAG and fine-tuning are alternatives."** Actually: they compose. The instructor's own architecture is a fine-tuned model *under* a RAG and an agent [3:51–4:00].
14. **"A pretrained model will tell you when it does not know."** Actually: the tomato experiment [1:01:40–1:01:47] shows it returns the nearest label in its closed set. Fluency is not calibration.
15. **"Transfer learning and fine-tuning are different techniques."** The instructor's answer, which is correct: *"Transfer learning is nothing, it is just a way to perform a fine tuning"* [41:44–41:46]. Transfer learning is the goal; fine-tuning is the mechanism.

---

## 18. Key Takeaways

1. **Pretraining teaches the language; fine-tuning teaches the job.** Everything else in this handbook is downstream of that sentence.
2. **Training compute is `6 × N × D` FLOPs. Inference is `2 × N` FLOPs per token.** Memorise both; they turn every plan into a time estimate.
3. **Full fine-tuning costs ~16 bytes per parameter** (bf16 weights + grads + fp32 master + Adam m,v). 7B → 112 GB before activations. This single number explains the existence of the entire PEFT field.
4. **Causal LM supervises 100% of positions; MLM supervises ~15%.** That asymmetry, plus generation and ICL, is why decoder-only won the LLM era — and why BERT-family encoders still win classification, NER, and embeddings.
5. **Chinchilla: ~20 tokens per parameter for compute-optimal training.** Modern models ignore it deliberately, because serving cost recurs and training cost does not.
6. **Inference is usually bandwidth-bound, not compute-bound.** `tok/s ≈ bandwidth ÷ weight_bytes`. Batch size, not FLOPs, is how you saturate a GPU.
7. **`> **Correction:**` vocabulary discipline:** the objective is *autoregressive*, the algorithm is *Direct Preference Optimization*, and the Transformer is not an RNN variant. Say these right in an interview.
8. **The base-vs-instruct distinction is a bug-or-feature decision, not a preference.** Evaluating a base model with a chat benchmark is the field's most common self-inflicted error.
9. **Tokenizers produce more silent failures than any other component.** Inspect the rendered chat template string before you spend a GPU-hour.
10. **Fine-tuning is the fourth step, not the first:** prompt → few-shot → RAG → fine-tune → agent. Each step costs more to reverse.
11. **Fine-tuning teaches form. RAG supplies substance.** If your data is documents, you want RAG; if it is demonstrations of behaviour, you want SFT.
12. **Quality beats quantity at the SFT stage (LIMA: 1,000 examples; Phi: curated synthetic data), and volume still wins at the pretraining stage.** Do not cross the streams.
13. **Alignment is fragile in both directions.** Narrow SFT breaks safety; a few adversarial examples break it faster. Run a safety regression suite every time.
14. **Quantization trades quality for memory, and the trade is not free.** Re-evaluate after every quantization step.
15. **The GPU bill is the smallest line item.** Data, evals, expert time and the serving endpoint dominate. Budget accordingly.

---

## 19. Self-Check Questions

Answer these before moving to CS-02. Answers are in the collapsed block at the end.

1. A colleague says "we will fine-tune our 7B on our product documentation so it stops hallucinating." What is wrong with the plan, and what do you propose instead?
2. Compute the VRAM needed for full fine-tuning a 13B model with AdamW in bf16 with fp32 master weights. Show the arithmetic.
3. Your 7B fine-tune has a training loss that falls smoothly from 2.9 to 0.35 and an eval loss of 2.8. What are the three most likely causes, in order of probability?
4. Explain why causal language modelling supervises more tokens per sequence than masked language modelling, and state one task where MLM is still the better choice.
5. A 7B model must be trained on 2 billion tokens. How many A100-80GB hours will it take, and what will it cost at $1.79/GPU-hour? State your MFU assumption.
6. Why is decode bandwidth-bound rather than compute-bound, and what practical lever does that give you for increasing throughput?
7. What is the difference between a base model and an instruct model, mechanically, and what breaks if you fine-tune the wrong one?
8. Your fine-tuned model answers correctly in a chat UI but produces garbage through the batch inference endpoint. Name the most likely cause.
9. State the Chinchilla rule and compute the compute-optimal token count for a 3B model. Then explain why a production team might deliberately train it on 10x that.
10. A model scores 0% exact match and 0.82 token-level F1 on your eval. What does that pattern mean and what should you fix first?

---

## 20. Cross-References

| Relationship | Module |
|---|---|
| **Builds on** | — (this is the entry module) |
| **Needed by** | CS-02 (transfer learning), CS-05 (RNN→Transformer), CS-06 (Hugging Face), CS-07 (BERT), CS-10/11 (quantization), CS-12 (continued pretraining), CS-13 (SFT), CS-14 (alignment), CS-13 §6.8 + CS-11 §4.11 (LoRA/QLoRA), `code/10_embedding_finetune.py` (embedding FT). (CS-20/SLMs is planned, unwritten) |
| **Contrasts with** | CS-04 (fine-tuning vs RAG vs agents — the decision this module's vocabulary enables) |
| **Deepened by** | CS-05 (why the Transformer unlocked fine-tuning), CS-11 (quantization internals), CH-01 (the formulas, as a lookup card) |
| **Interview prep** | IQ-01 (100 questions across 5 levels on exactly this material) |
| **Capstone** | CS-28 — *planned, not yet written*. CS-13 §15 and CS-09 §6 are the nearest assembled pipelines |

---

## Appendix A — Instructor's Verbatim Key Claims

Verbatim quotes from the two transcripts, with timestamps. Where a quote is a misstatement, the correction is given inline — see also the `Correction` callouts in §4.2, §4.6, §4.8, §4.10.

**On why the topic matters (video 01):**

> *"This fine-tuning seems very underrated — most of the people don't discuss about it."* [3:07–3:12]

> *"No one talks about the fundamental. Believe me, this LLM fine-tuning, it's a fundamental topic of the generative AI."* [3:21–3:29]

> *"Companies are spending their money, their resources for fine-tuning their models. They are [using] the model from various APIs like Claude and OpenAI but along with that they are fine-tuning according to their requirement."* [3:37–3:52]

> *"People are fine-tuning open-source models, then on top of it they are creating a RAG, they are creating agents."* [3:51–4:00]

> *"Fine-tuning is just not a fine-tuning, just not a training the model — it is more than that."* [4:09–4:14]

> *"If it is not working fine then again we can retrain it."* [4:22–4:25]

> *"This could be expensive, I'm not saying it would be free... but how we can train it at a minimal cost, what frameworks, what new research — so with minimal cost, with no cost, we can fine-tune our model."* [4:32–4:54]

> *"I'm not going to discuss like one or two video. I never do that, because I believe to give you the entire content with a particular sequence."* [5:46–5:55]

> *"LangChain does not provide any support to the fine-tuning. It only supports the RAG-based application and the agentic application."* [2:33–2:41]

> *"This is the requirement of the industry. They might ask you the question from the fine-tuning."* [18:53–19:00]

**On training, pretraining and transfer learning (video 02):**

> *"Whenever we talk about model building: first we collect the data, then we analyse the data, then we pre-process the data, and then only we perform the model building. Model building — the simple meaning is model training."* [12:31–12:45]

> *"I'm doing a task-specific training. If I want to do text classification then I'll train a model on top of it. Then question answering — then I will train my model again from scratch."* [17:23–17:41]

> *"The era of the fine-tuning has started in 2011 itself. And you won't believe, the fine-tuning first time was introduced inside this CNN."* [18:15–18:31]

> *"Using this RNN and LSTM model even we cannot achieve the fine-tuning perpetually."* [18:39–18:43]

> *"This model is trained on a huge amount of data. This model is called the pre-trained model."* [20:07–20:28]

> *"If someone is going to ask you what is a foundation model — this pre-trained model only it is called the foundation model."* [36:04–36:08]

> *"ChatGPT is an application. GPT is a model."* [35:47–35:50]

> *"Transfer learning is nothing. It is just a way to perform a fine-tuning."* [41:44–41:46]

> *"Either we can only change the last layer of the model, or else we can change a couple of last layers and freeze most of the initial layers."* [41:51–42:05]

> *"Why use this pre-trained model? Deep learning models require a massive amount of data, and because of that we cannot train it again and again from scratch... it will take a very huge time, effort."* [42:28–42:57]

> *"Pre-train means: first teach the model on some basic knowledge before asking it to do a specific task."* [38:55–39:05]

**On the objectives and the pretraining loop:**

> *"In BERT they published about masked language modelling, and in the GPT research paper they published about causal language modelling."* [29:43–29:53]

> *"This causal language modelling — it is also called a regressive modelling."* [30:25–30:31]
> **Correction:** the term is **autoregressive**. "Regression" refers to continuous-target prediction.

> *"In the GPT model we use the causal modelling, in BERT we use the mask modelling, in T5 they were using the span masking, and in PaLM, LLaMA and all the state-of-the-art models they are using the causal pre-training model."* [54:22–54:38]

> *"We collect the data, we tokenize the data, we pass it to the transformer, we predict the next token, we calculate the loss, we do the back propagation, and then repeat. This is the entire training process of the transformer-based model."* [53:32–53:50]

> *"Why is it called unsupervised pre-training? There is no label... but this is not the correct one. In many blogs and research papers they have said you can use one better word: self-supervised learning."* [55:05–55:37]

> *"Instead of saying this is unsupervised pre-training, the better word will be self-supervised training... automatically the label is going to be created from the data."* [56:47–57:01]

> *"It's a self-taught student — first it is teaching itself, and then it's time to predict."* [57:05–57:17]

> *"Because of this pre-training it is able to understand the grammar inside the data, the word semantic relationship... it's trying to understand the crux of the language, and because of that it's going to work like magic."* [57:22–57:40]

> *"Evaluation of the model using perplexity, loss, token accuracy."* [35:16–35:20]

**On the origin story and the papers:**

> *"Two research papers you should go through. The first is encoder-decoder, published in 2014 — even though they used RNN and LSTM, lots of concepts have been discussed inside."* [22:53–23:08]

> *"The next research paper was published in 2018: Universal Language Model Fine-tuning for Text Classification. This paper is the real starting of the fine-tuning in the NLP domain."* [23:10–23:30]

> *"There is one more: Neural Machine Translation by Jointly Learning to Align and Translate. If you read all three, your understanding will be much more complete."* [23:31–23:50]

> *"The self-attention concept was inspired from this research paper — neural machine translation by jointly learning to align and translate. They defined the concept of attention."* [27:44–28:03]
> **Correction:** the transcript's account of *why* the 2018 paper (ULMFiT) matters is muddled. ULMFiT pre-trained with a **causal language-model objective on Wikitext-103**, not machine translation, and introduced *discriminative fine-tuning*, *slanted triangular learning rates* and *gradual unfreezing*. The machine-translation pretraining critique [25:46–27:12] belongs to the 2014 seq2seq work, not to ULMFiT.

> *"The limitation was: for the pre-training task they were using machine translation, and using this machine translation they were not able to teach the model in a very good way... the model was just learning something."* [25:46–26:18]

> *"This RNN and LSTM model is not a very good model for long-term dependency — we cannot handle very long sequences — and it was also not computationally efficient. Because of that this research paper introduced the fine-tuning but couldn't scale the entire idea."* [26:42–27:12]

> *"Transformer enters — and now the entire thing got changed, because this transformer architecture was not using the LSTM and the RNN at all."* [27:18–27:36]

**On the CNN prehistory:**

> *"Pre-training is when a convolution neural network is trained on a large or general data set, like the ImageNet data set."* [43:47–43:54]

> *"This data set was created by Fei-Fei [Li], and she was creating this data set from 2006 itself."* [44:08–44:20]

> *"There was one guy, Jeffrey — he participated with his own model. The model name was AlexNet."* [44:37–44:48]

> *"The primitive feature of Sunny and Rahul will be the same — nose, eyes, ear, eyebrows, forehead — but the sophisticated feature of Sunny and Rahul is going to be different."* [49:02–49:32]

> *"For the primitive layer the feature will be the same, so we don't need to fine-tune that. We'll keep it as it is, we'll freeze this layer, and we'll retrain a couple of last layers — the deep layers."* [51:02–51:21]

> *"They pre-trained the model on the ImageNet data set and then they were fine-tuning for any specific task."* [52:50–52:58]
> **Correction:** the transcript places hand-engineered features at "2012" [52:16–52:21] and pre-trained models "2012 onwards." The actual demarcation is **ILSVRC-2012**: AlexNet won with ~15.3% top-5 error against ~26.2% for the runner-up (which used hand-engineered SIFT/Fisher-vector features). Hand-engineered features dominated 2010–2011.

**On the modern lifecycle:**

> *"ChatGPT introduced one more level of training: reinforcement learning through human feedback."* [34:45–34:53]

> *"This RLHF is being replaced by this DPO technique. This is called direct reference optimization, DPO."* [36:59–37:06]
> **Correction:** **Direct Preference Optimization**. There is no "reference optimization" in the name; "reference" appears only because DPO uses a frozen *reference model* as the anchor for its implicit reward.

> *"Then we can perform the instruction fine-tuning, then we can perform the RLHF — and this continuous learning is going right."* [37:31–37:42]

> *"The number of tokens and all — how many tokens they are using, at which scale, and with which technique."* [54:41–54:50]

> *"They are collecting data from Common Crawl, from Wikipedia, news articles, social media posts, books, GitHub and all — from every place they are taking the data."* [54:52–55:03]

---

## Appendix B — Reference Links & Papers

**Named in the transcripts (video 02):**

| Paper | Why it matters | Reference |
|---|---|---|
| Encoder–decoder sequence learning (2014) | The instructor's "first paper"; showed seq2seq learning with RNN/LSTM and established the pretrain-then-transfer idea | Sutskever, Vinyals & Le, *Sequence to Sequence Learning with Neural Networks*, arXiv:1409.3215 |
| Attention origin (2014) | Where the concept the Transformer industrialised came from; the instructor names it as the source of self-attention | Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, arXiv:1409.0473 |
| ULMFiT (2018) | "The real starting of fine-tuning in the NLP domain": discriminative fine-tuning, slanted triangular LR, gradual unfreezing, causal-LM pretraining on Wikitext-103 | Howard & Ruder, *Universal Language Model Fine-tuning for Text Classification*, arXiv:1801.06146 |
| Transformer (2017) | The architecture that made everything in this handbook possible | Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762 |
| BERT (2018) | Masked language modelling; the encoder-only branch | Devlin et al., arXiv:1810.04805 |
| GPT (2018) | Causal language modelling; the decoder-only branch | Radford et al., *Improving Language Understanding by Generative Pre-Training* |
| T5 (2019) | Span corruption; the encoder-decoder branch | Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, arXiv:1910.10683 |
| AlexNet (2012) | The ImageNet result that started the fine-tuning era | Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep Convolutional Neural Networks* |
| ResNet (2015) | The model in the instructor's demo | He et al., arXiv:1512.03385 |
| ImageNet / ILSVRC | The dataset and competition behind the CNN prehistory | Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database* (2009) |

**Beyond the video — the papers that decide budgets and architecture:**

| Paper | Why it matters | Reference |
|---|---|---|
| Kaplan scaling laws (2020) | The "buy parameters" era; the source of GPT-3's shape | arXiv:2001.08361 |
| **Chinchilla (2022)** | The 20-tokens-per-parameter correction; 70B beat 280B | Hoffmann et al., arXiv:2203.15556 |
| LLaMA (2023) | Inference-optimal overtraining; the open-weights inflection point | Touvron et al., arXiv:2302.13971 |
| LLaMA-2 (2023) | The base/instruct split as a product decision; 2T tokens on 7B | arXiv:2307.09288 |
| LLaMA-3 herd of models (2024) | 15T tokens, 128k vocabulary, GQA — the modern defaults | arXiv:2407.21783 |
| Emergent abilities (2022) | The claim | Wei et al., arXiv:2206.07682 |
| **Emergent abilities are a mirage (2023)** | The counter-argument you must be able to state | Schaeffer, Miranda & Koyejo, arXiv:2304.15004 |
| LoRA (2021) | The method that made fine-tuning affordable | Hu et al., arXiv:2106.09685 |
| QLoRA (2023) | 4-bit NF4 + paged optimisers; 65B on one 48 GB card | Dettmers et al., arXiv:2305.14314 |
| LIMA (2023) | 1,000 curated examples; quality beats quantity at the SFT stage | Zhou et al., arXiv:2305.11206 |
| Textbooks Are All You Need (2023) | The Phi thesis: curated synthetic data substitutes for scale | Gunasekar et al., arXiv:2306.11644 |
| Deduplicating training data (2021) | Why dedup is not optional | Lee et al., arXiv:2107.06499 |
| FlashAttention (2022) | Why attention no longer dominates activation memory | Dao et al., arXiv:2205.14135 |
| Reducing activation recomputation (2022) | The activation-memory algebra behind gradient checkpointing | Korthikanti et al., arXiv:2205.05198 |
| GPQA (2023) | A contamination-controlled benchmark — the right shape for eval design | Rein et al., arXiv:2311.12022 |
| Fine-tuning compromises safety (2023) | 10 adversarial examples break alignment | Qi et al., arXiv:2310.03693 |
| LoRA fine-tuning degrades safety (2023) | A few hundred benign examples suffice | Lermen et al., arXiv:2311.16153 |
| Safety is a removable low-rank direction (2023) | Why alignment is fragile by construction | Wei et al., arXiv:2312.06681 |
| DPO (2023) | The preference objective that replaced PPO for most teams | Rafailov et al., arXiv:2305.18290 |
| InstructGPT (2022) | The canonical RLHF pipeline | Ouyang et al., arXiv:2203.02155 |
| MMLU (2020) | The benchmark everyone quotes and nobody should trust blindly | Hendrycks et al., arXiv:2009.03300 |

**Documentation and tooling:**

| Resource | Use |
|---|---|
| Hugging Face `transformers` docs — chat templating | The authoritative reference for `apply_chat_template` and per-model templates |
| Hugging Face Open LLM Leaderboard / model cards | Licence, tokenizer and template per model — read the card before training |
| FineWeb / Dolma / SlimPajama technical reports | Public, reproducible recipes for the filtering and dedup pipeline in §4.9 |
| vLLM, TGI, llama.cpp, TensorRT-LLM docs | Serving engines; the PagedAttention paper explains continuous batching |
| EleutherAI `lm-evaluation-harness` | The standard harness; note its base-model vs instruct-model task variants |

---

## Self-Check Answers

<details>
<summary>Expand for the answers to all ten §19 questions</summary>

**1. "Fine-tune on product docs so it stops hallucinating."**
Wrong in category: documentation is *knowledge*, and SFT teaches *form*. Training on documents teaches the model the documentation's register and vocabulary, which makes hallucination more fluent and thus worse (case study 15.4: hallucination rose from 18% to 41%). Instead: (a) build RAG over the same corpus with citations; (b) if the interaction still needs shaping — always cite, never recommend competitors, refuse out-of-scope — fine-tune on 1,000+ demonstrations of the *desired interaction*, not on the documents; (c) evaluate hallucination with a fixed 200-item probe set before and after.

**2. VRAM for full FT of 13B, AdamW bf16 + fp32 master.**
```
weights (bf16)        13e9 × 2 B =  26 GB
gradients (bf16)      13e9 × 2 B =  26 GB
Adam m (fp32)         13e9 × 4 B =  52 GB
Adam v (fp32)         13e9 × 4 B =  52 GB
fp32 master weights   13e9 × 4 B =  52 GB
                              ─────────────
static total                      208 GB   (= 16 bytes/param)
activations (b=1,s=2048, ckpt)   ~ 2-8 GB
fragmentation + comm             ~ 5-10 GB
                              ─────────────
realistic                         220-230 GB  ->  4 x A100-80GB minimum
```

**3. Train loss 2.9 → 0.35, eval loss 2.8 — three causes in order of probability.**
(1) **Overfitting / memorisation** — too many epochs on too little data, or duplicates. Diagnostic: does eval loss rise monotonically after a minimum? Fix: fewer epochs, dedup, more data. (2) **Train/eval split leakage** — near-duplicate rows in both. Diagnostic: run MinHash across splits. (3) **Eval data formatted differently from training data** — a template or truncation mismatch. Diagnostic: decode 5 eval examples and 5 train examples and diff the raw strings. If eval loss is high *from step 0* and never improves, it is (3); if it dips then rises, it is (1).

**4. Causal LM vs MLM supervision.**
CLM predicts a token at *every* position, so a T-token sequence yields T supervised predictions — 100% supervision. MLM masks ~15% of positions and predicts only those, so a T-token sequence yields ~0.15T supervised predictions — 15% supervision. Additionally, CLM's objective at inference time *is* the generation procedure, and it enables ICL/few-shot prompting. MLM is still better when you need a **bidirectional contextual representation for a discriminative task**: classification (CS-07), NER, extractive QA, and sentence embeddings (`code/10_embedding_finetune.py`) — a token's meaning depends on what follows it as well as what precedes it, and a causal mask throws that away.

**5. 7B on 2B tokens — A100-hours and cost.**
```
FLOPs = 6 × 7e9 × 2e9 = 8.4e19
Assume 40% MFU on A100-80GB bf16 (312 TFLOP/s peak) -> 125 TFLOP/s effective
Hours = 8.4e19 / 1.25e14 = 6.72e5 s = 186.7 A100-80GB hours
Cost  = 186.7 × $1.79 = $334   (Lambda on-demand ballpark)
Wall clock on 8 x A100-80GB: 23.3 h
```
Sanity check against a known data point: Llama-2-7B used 184,320 A100-hours for 2T tokens, i.e. 184.3 A100-hours per billion tokens. This estimate gives 93.3 A100-hours per billion tokens — about 2x more efficient, because Llama-2's figure includes 3 epochs' worth of development runs, failed restarts and eval overhead. **Real training runs burn 1.5–3x the theoretical FLOPs.** Budget accordingly.

**6. Why decode is bandwidth-bound, and the lever.**
Each generated token requires reading every weight once (`2N` FLOPs, `2N` bytes in bf16) — an arithmetic intensity of roughly `2N / 2N = 1` FLOP per byte, far below the A100's ridge point of ~153 FLOP/byte. So the GPU is idle waiting on memory, and time per token ≈ weight bytes ÷ bandwidth. **The lever is batch size:** with batch B, you read the weights once and use them B times, raising intensity to ~B FLOP/byte. At B ≈ 150–300 you enter the compute-bound regime and the same GPU serves many users at once. Secondary levers: quantization (fewer bytes per weight → proportionally faster decode), GQA/smaller KV cache, speculative decoding.

**7. Base vs instruct, mechanically, and what breaks.**
A base model is a transformer + tokenizer + LM head trained only on next-token prediction; it has no chat template and no learned notion of "assistant". An instruct model is that plus SFT on (instruction, response) pairs (plus, usually, preference alignment), shipped with a `chat_template` and stop tokens. If you fine-tune the **wrong** one: (a) fine-tuning a *base* model on 500 instruction pairs will not produce an assistant — it needs 10k–100k examples to learn the format from zero; (b) fine-tuning an *instruct* model with a **different** family's chat template produces a smooth loss curve and incoherent outputs; (c) evaluating a base model on a chat benchmark yields near-chance scores that reflect the format, not the knowledge.

**8. Chat UI works, batch endpoint produces garbage.**
Almost certainly **padding side**. Decoder-only models require **left padding** for batched generation, because generation continues from the last token — with right padding, the model continues from a pad token. Most batch-serving code defaults to right padding (the Trainer's default). Fix: `tokenizer.padding_side = "left"` in the batch path. Second candidate: the batch path applies a different chat template (or none) than the chat UI.

**9. Chinchilla rule and the 3B case.**
`D_opt ≈ 20 × N`. For 3B: `20 × 3e9 = 6e10 = 60B tokens` (≈45 GB as uint16 token ids; `6 × 3e9 × 6e10 = 1.08e21` FLOPs ≈ 2,400 A100-hours ≈ $4,300). A production team may deliberately train 10x longer (600B tokens) because **inference cost recurs and training cost does not**. If the model serves 100M requests/month at 500 output tokens, the training cost is amortised across years of serving, while a larger model that was cheaper to train would cost more every single month. This is exactly why Llama-3-8B is trained to 1,875 tokens/param.

**10. EM = 0%, token-F1 = 0.82.**
The model has the knowledge and the content but is not producing the expected *surface form* — a format, template, or normalisation mismatch. This is the practical signature of the emergence-as-metric-artefact argument (§4.6). Fix, in order: (1) print the model's raw output and the reference side by side for 10 failures; (2) check the chat template and any stop strings; (3) check whether the reference expects a normalisation you are not applying (case, whitespace, units, a JSON wrapper); (4) only then consider more training data. **Do not spend GPU-hours on a formatting problem.**

</details>

---

*End of CS-01. Next: CS-02 — Transfer Learning & Model Fine-Tuning.*






