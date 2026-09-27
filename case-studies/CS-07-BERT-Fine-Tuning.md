# CS-07 — Fine-Tuning BERT: NER, Sentiment, QA

| Field | Value |
|---|---|
| **Module** | Encoder-only fine-tuning (classical NLP) |
| **Source video(s)** | LLM Fine-Tuning 09: Fine-Tuning BERT for NLP (NER, Sentiment, QA) \| Hugging Face |
| **Transcript file(s)** | `LLM_Fine-Tuning_09_Fine-Tuning_BERT_for_NLP_NER_Sentiment_QA_Hugging_Face_huggin.txt` |
| **Companion code** | `repo\Complete-LLM-Finetuning-main\LLM Fine-Tuning-09-Bert-finetuning\BERT_Finetuning.ipynb` |
| **Prerequisites** | CS-01 (pretraining lifecycle), CS-02 (transfer learning), CS-05 (RNN/LSTM → attention), CS-06 (Hugging Face masterclass) |
| **Difficulty** | Intermediate |
| **Hands-on required** | Yes — all three tasks are runnable on a free Colab T4 |
| **Estimated study time** | 5h theory + 6h practical |

---

## 0. Executive Summary

- **BERT is an encoder-only Transformer.** It reads and encodes; it does not generate. The instructor states this as the single most interview-relevant fact in the video: *"This BERT model is not a generative model"* [18:06]. Everything else follows from it.
- Published **24 May 2019** by Google; the acronym is **B**idirectional **E**ncoder **R**epresentations from **T**ransformers — the paper title is *"Pre-training of Deep Bidirectional Transformers for Language Understanding"* [4:50].
- Two sizes in the original paper: **BERT-base = 12 encoder layers (110M params)**, **BERT-large = 24 encoder layers (340M params)** [10:49–10:57]. The instructor misspeaks "20 encoders" once at [8:22] and corrects himself to 24 at [10:57] — trust the second number.
- **Two-stage training**: (1) self-supervised pretraining on internet text using **MLM + NSP**, (2) supervised fine-tuning on your labelled task data [12:52–16:00]. The instructor explicitly rejects the word "semi-supervised" for stage 1 — *"this is not a correct word. You can say it is a self-supervised pre-training or unsupervised pre-training"* [13:35].
- **Fine-tuning = swapping the head.** The encoder body stays; a task-specific head (a feed-forward/dense network, optionally plus softmax) is dropped on top of the contextual vectors [19:36–22:20]. Classification, token classification (NER/POS), and QA each get a *different* head.
- Three tasks are demonstrated end-to-end: **sentiment classification** (IMDb, 1000 train / 500 test), **NER** (WikiAnn, 9 labels, BIO), **QA** (SQuAD, extractive span selection) [17:40–18:05].
- The **single most expensive silent bug in this module** is subword↔label misalignment in NER. `London` → `Lon` + `##don` produces 7 tokens from 4 words; if you do not mask the continuation subword with `-100`, you train the model on garbage labels and the loss goes *down* anyway.
- **Token-level accuracy is a lie for NER.** With ~85% of tokens tagged `O`, a model that predicts `O` for everything scores 85% accuracy and 0.0 entity F1. Always report **`seqeval` entity-level micro-F1**.
- The instructor's hyperparameters are the canonical encoder recipe: **LR 2e-5, 1–3 epochs, batch 8–32, weight decay 0.01, warmup = 10% of total steps, grad-norm clip 1.0** [39:51–41:12].
- **Cost argument:** a fine-tuned BERT-base classifies 1M documents for roughly **$0.10–0.20** of GPU time. Calling GPT-4o for the same 1M is **~$600**. That 3,000× gap — not accuracy — is why BERT is still in production in 2026.
- BERT is not competitive on open-ended generation, long context (hard 512-token ceiling), or zero-shot tasks. It is still the right answer for high-volume, low-latency, closed-label-set NLP.

---

## 1. The Problem This Solves

### 1.1 What breaks without it

You need to tag 40 million support tickets per month with an intent label, or extract `(drug, dosage, adverse_event)` triples from 2 million clinical notes, or route 500,000 emails per day. Constraints: p99 latency under 50 ms, cost under $200/month, deterministic behaviour, on-prem (no data egress).

The naive approach — prompt a frontier LLM — fails on all four constraints at once:

| Constraint | Why the LLM prompt loop fails |
|---|---|
| Latency p99 < 50 ms | Any hosted LLM is 400–2,000 ms network-bound. You cannot get to 50 ms. |
| Cost < $200/month | 500k emails/day × 30 days = 15M calls/month. At ~$0.000036/call (GPT-4o-mini, 200 in / 10 out) that is **$540/month** — and at GPT-4o pricing, **$9,000/month**. |
| Determinism | Prompt-based labels drift when the vendor updates the model silently. Your audit trail is invalidated. |
| On-prem | API models cannot run in an air-gapped VPC. |
| Cost predictability | Every prediction re-pays the full prefill cost. There is no amortization. |

### 1.2 The state of the art before BERT

Before 2018, the pipeline was:

1. **Word embeddings** (Word2Vec, GloVe, fastText) — a single vector per word, context-independent. `bank` has one vector whether it means a river bank or a financial institution.
2. **A task-specific architecture on top** — BiLSTM, BiLSTM-CRF for NER, CNN for classification.
3. **Trained from scratch per task**, per domain, per label set.

The problems: (a) the encoder was trained on your 5,000 labelled examples only, so it never learned English; (b) no context sensitivity; (c) each new task required a new architecture and a new research cycle.

LSTM-based encoder–decoder with attention improved this but still had the sequential-computation bottleneck and vanishing gradient over long spans — the exact constraints covered in CS-05.

### 1.3 Why the obvious next step also fails

The obvious fix is "pretrain a big model on everything, then fine-tune it". Two variants of that idea were tried and both were wrong:

- **Pretrain a *unidirectional* language model (ELMo / GPT-1 style)** and use its hidden states. This gives you a left-to-right representation. For a *tagging* task the token you are classifying has already "seen" itself and everything before it, but not the word after it. For "Apple acquired Beats" vs "Apple acquired taste", the disambiguating context is *after* the word. Unidirectional representations are strictly weaker for understanding tasks.
- **Pretrain a *shallow-concatenation* bidirectional model** (ELMo: run a forward LSTM and a backward LSTM separately and concatenate). This is "bidirectional" only in the sense that two independent passes are glued together. Neither pass is conditioned on the other, and neither is deeply bidirectional.

BERT's contribution is not a new architecture — it is the observation that the **Transformer encoder is natively bidirectional** (every token attends to every token), so you can get true deep bidirectionality *for free* — you just need a pretraining objective that does not leak the answer, and that objective is **masked language modelling**.

### 1.4 Concrete motivating numbers

From the video, the instructor's own demo scale and the wall-clock reality:

```
IMDb full train set         = 25,000 labelled reviews
Instructor's subset         = 1,000 train / 500 test            [32:20]
Instructor's max_length     = 256 tokens                        [33:06]
Instructor's wall-clock     = "more than 25,000 sample ... at least 5 to 6 hour
                               to train on this much of data"   [32:04-32:12]
```

Five to six hours for 25k examples is an **fp32, free-Colab-T4, no-dynamic-padding** number. Section 11 shows the same job in well under an hour with fp16, dynamic padding, and a batch size of 32 — a 5–8× throughput win for zero accuracy cost.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: reader vs writer

A **decoder-only model (GPT) is a writer.** It is trained to answer one question: *given the text so far, what comes next?* That objective forces it to hold a latent "plan" for the remainder of the document. Generation is the natural output.

An **encoder-only model (BERT) is a reader.** It is trained to answer a different question: *given a sentence with a hole in it, what was in the hole?* To do that it must build a rich, bidirectional, context-conditioned representation of the whole input. But it never learns to produce a sequence — it has no causal mask, so it has no notion of "next" at all.

**Where this analogy breaks.** A reader can still *write* in a narrow sense: BERT can be used to extract a span from a document (QA), because span extraction is a *selection* problem over positions the reader has already scored, not a generation problem. It cannot compose new text. It also cannot decide *how long* an answer should be — every answer it produces is a substring of the input. And a "reader" framing hides the fact that the encoder's representation is *layer-wise hierarchical*: layers 0–3 are near-lexical, layers 4–8 are syntactic, layers 9–11 are task- and semantics-heavy. That is why layer selection and layer-wise LR decay work at all.

### 2.2 The actual mechanism, in six steps

```
INPUT TEXT
   "John lives in London"
        |
        v
[1] WordPiece tokenizer  ->  tokens + special tokens
   ["[CLS]", "john", "lives", "in", "lon", "##don", "[SEP]"]
        |
        v
[2] Embedding lookup + learned absolute position embedding
   token_emb[i] + pos_emb[i]  ->  matrix H0 of shape (7, 768)
        |
        v
[3] 12 x Encoder block   (BERT-base)
    for each block:
        a = LayerNorm(H + MultiHeadSelfAttention(H))     # residual + post-LN
        b = LayerNorm(a + FeedForward(a))                # FFN: 768 -> 3072 -> 768
    H12 = block_12(...)          shape (7, 768)
        |
        v
[4] Pooling / selection
    classification  -> take H12[0]   (the [CLS] vector)  -> (768,)
    token tagging   -> take all H12                      -> (7, 768)
    QA              -> take all H12                      -> (7, 768)
        |
        v
[5] TASK HEAD (randomly initialised, trained from scratch)
    classification  -> Linear(768, 2) + softmax
    token tagging   -> Linear(768, 9)
    QA              -> Linear(768, 2)  -> column 0 = start logits, column 1 = end logits
        |
        v
[6] Loss
    classification / tagging -> cross-entropy
    QA                       -> CE(start) + CE(end), averaged
```

### 2.3 Where the analogy breaks, precisely

- **"Reader" implies one pass.** In reality a 512-token input costs 512×512 attention entries per head per layer = 12 layers × 12 heads × 262,144 = **37.7M attention scores** for one sequence. BERT is quadratic in length, which is exactly why the 512-token ceiling exists (§4.4).
- **"[CLS] is the sentence summary" is only half true.** The `[CLS]` vector was trained by the NSP head, so it aggregates. But for *similarity* it is a poor sentence embedding — mean pooling over all tokens beats it by a wide margin in every Sentence-BERT benchmark. For classification, `[CLS]` fine-tunes well; for retrieval, use a sentence-transformer.
- **"Pretrained = understands English".** It understands the *distribution* of English on BooksCorpus + Wikipedia (16 GB, 3.3B words). It has never seen your product names, your ICD-10 codes, or your internal ticket taxonomy. Domain vocabulary becomes `[UNK]` or a sequence of junk subwords, and fine-tuning on 1,000 examples cannot fix a broken tokenizer. This is the entire justification for domain-adaptive pretraining (CS-12) and domain encoders like PubMedBERT.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Encoder** | The half of the Transformer that applies unmasked (bidirectional) self-attention and outputs one contextual vector per input token. | BERT *is* an encoder stack. Nothing else. | "BERT has a decoder" — it does not, in any released variant. |
| **Decoder** | The half that applies a causal (lower-triangular) attention mask so position *i* can only see positions ≤ *i*. | This causal mask is the *entire* reason GPT can generate and BERT cannot. | "Encoder–decoder is just encoder + decoder" — no: the decoder cross-attends to the encoder, which BERT never does. |
| **Encoder–decoder (seq2seq)** | Both halves, where the decoder cross-attends to the encoder's output. T5, BART, mT5. | The right family for *generative* transformation tasks: summarization, translation. | Confusing it with "BERT + a decoder" — that is BERT2BERT / RAG, and is a different thing. |
| **MLM (Masked Language Modelling)** | Pretraining objective: randomly select 15% of input tokens, replace them per the 80/10/10 rule, and predict the original token id at those positions. | The objective that makes deep bidirectionality trainable without label leakage. | "Masked" means *masked from the model*, not masked attention. |
| **NSP (Next Sentence Prediction)** | Pretraining objective: given sentences A and B, predict whether B actually followed A (50% real, 50% random). | BERT's second objective. RoBERTa showed it is useless-to-harmful. | Thinking NSP is what gives BERT "sentence understanding". `[CLS]` pooling survives without it. |
| **80/10/10 rule** | Of the 15% selected tokens: 80% become `[MASK]`, 10% become a random vocabulary token, 10% are left unchanged. | Removes the pretrain/fine-tune `[MASK]` distribution mismatch. | "The other 10% are wasted" — they are the only positions where the model sees a *clean* token and must still produce a gradient. |
| **Whole-word masking** | All WordPiece subwords of a chosen *word* are masked together. | Forces the model to use cross-word context instead of inside-word guessing (e.g. predicting `##don` from `Lon`). | BERT-base (original) is *subword*-level; `bert-base-cased` WWM variants and RoBERTa are whole-word. |
| **Dynamic masking** | A new random mask is generated every time a sequence is seen. | RoBERTa's default; gives ~10× more distinct training signals per epoch. | BERT's reference implementation used 10 *static* pre-masked copies — a TPU-pipeline artifact, not a design choice. |
| **Replaced token detection (RTD)** | ELECTRA's objective: corrupt 15% of tokens with a small generator, then ask a discriminator to label *every* token as original or replaced. | Dense per-token signal instead of 15% sparse; no `[MASK]` at all. | "It is just MLM with a different name" — RTD is a *binary* task over all positions and needs no MLM head at fine-tuning time. |
| **SOP (Sentence Order Prediction)** | ALBERT's replacement for NSP: distinguish `A-B` from `B-A` instead of `A-B` from `A-C`. | Removes the topic-shift shortcut that made NSP trivially easy. | Not the same signal as NSP; SOP is genuinely harder and does help. |
| **[CLS]** | Token id **101**, prepended to every sequence. Its final hidden state is the pooled sequence representation. | The classification head reads it. In SQuAD 2.0 the *no-answer* decision is read from it. | "[CLS] means classifier" is a mnemonic, not a mechanism — it works because the NSP head trained on it. |
| **[SEP]** | Token id **102**, separates segments; the sentence-pair input is `[CLS] A [SEP] B [SEP]`. | Required for pair tasks (NLI, QA, sentence similarity) and for segment embeddings. | Counting sequences by counting `[SEP]`s requires remembering the trailing one. |
| **[PAD]** | Token id **0**, pads to a fixed length so a batch is rectangular. `attention_mask=0` for pads. | Batches must be rectangular tensors. | A pad token still *runs through the network*; the mask is what stops it being attended to. |
| **[MASK]** | Token id **103**, the corruption token. Never appears at fine-tuning time. | The train/test mismatch is exactly why BERT has a fine-tuning heuristic (mask-rate search) and why RoBERTa/ELECTRA moved away. | Confusing token id 103 with the attention mask. |
| **[UNK]** | Token id **100**, emitted for characters/subwords not in the 30,522-entry WordPiece vocabulary. | A high `[UNK]` rate on your domain means the tokenizer, not the model, is your bottleneck. Measure it first. | "Rare words become [UNK] and the model handles it" — no, information is destroyed irreversibly. |
| **WordPiece** | BERT's subword tokenizer: greedy longest-match-first over a 30,522-token vocabulary, with `##` marking continuation pieces. | Explains why one word can become 1–5 tokens and therefore why label alignment exists. | "Each word is one token" — false, and this single false belief causes the #1 NER bug. |
| **`word_ids()`** | Method on a **fast** tokenizer's encoding: a list mapping each token index to its source word index, or `None` for special tokens. | The official alignment primitive for token classification. | It exists only on `*Fast` tokenizers. `BertTokenizer` will raise `AttributeError`. |
| **`-100`** | `torch.nn.CrossEntropyLoss`'s `ignore_index` default. Label positions set to `-100` produce zero loss and zero gradient. | The remedy for subword alignment. | It is *not* a label id and not `pad_token_id` (which is 0). Using 0 by mistake trains the model to predict `O` on every pad. |
| **BIO tagging** | `B-X` begins an entity of type X, `I-X` continues it, `O` is outside. | The label scheme for NER. | `I-X` immediately after `O` is malformed but common in real data; it breaks some decoders. |
| **BILOU / BIOES** | Adds `L-X` (last), `U-X` (unit/single) or `E-X` (end), `S-X` (single). | Unambiguous multi-token span boundaries; `seqeval` supports both. | `seqeval` defaults to the **IOB2** scheme; passing BILOU data without `scheme=` silently mis-scores. |
| **`seqeval`** | Entity-level NER scorer: a predicted span counts as correct only if its *type and both boundaries* exactly match gold. | The only NER metric that matches business reality ("did we find the invoice number correctly?"). | Replacing it with `sklearn` token accuracy, which is 85% for a useless model. |
| **Span extraction (extractive QA)** | QA by selecting a contiguous substring of the provided context, modelled as two distributions over token positions: start and end. | SQuAD-style QA. No generation, so no hallucination possible on the answer itself. | Confusing it with generative/RAG QA, where the model writes the answer. |
| **`offset_mapping`** | Character-offset `(start, end)` for every token, returned when `return_offsets_mapping=True`. | The only correct way to convert a character-indexed gold answer into token indices — and back. | Requires a fast tokenizer; the slow one silently ignores the flag or errors. |
| **`doc_stride`** | Overlap between consecutive 512-token windows when a context is longer than the model. | Without it, an answer spanning a window boundary is unsplittable and permanently unanswerable. | `doc_stride` is not "chunk size"; `max_seq_length` is. |
| **`n_best_size`** | Number of top start/end logit candidates kept per window before scoring and de-duplicating spans at inference. | Controls the recall of the span search at a small CPU cost. | Setting it to 1 turns QA into a single greedy argmax and loses ~5–10 F1. |
| **Head** | The randomly-initialised task module placed on top of the encoder: `Linear(768, num_labels)` (± softmax). | "Fine-tuning = swap the head, gently nudge the body." | Thinking the head is pretrained. It is random noise at step 0, which is why loss starts near `ln(num_labels)`. |
| **Pooler** | BERT's built-in `Linear(768,768) + tanh` applied to `[CLS]`. Used by the NSP head. | `AutoModelForSequenceClassification` reuses it under `BertPooler`. | Not used by `BertForTokenClassification` or `BertForQuestionAnswering`. |
| **Full fine-tuning** | Update every parameter (110M for BERT-base) with a small LR. | The default and the accuracy ceiling for encoders. Feasible because 110M is small. | Confusing it with "retrain from scratch" — the weights start pretrained. |
| **Layer freezing** | `param.requires_grad = False` on the embedding + bottom encoder layers; train only the top N layers + head. | 1.5–2× faster, less overfitting on small data, usually ≤1 point of accuracy loss. | Freezing *everything* except the head gives you a linear probe, which is much weaker. |
| **Discriminative / layer-wise LR decay** | Lower LR for bottom layers, higher for top layers (decay factor 0.65–0.95). | ULMFiT trick; helps when the task is far from pretraining. | Not supported natively by `TrainingArguments` — needs param groups. |
| **`data_collator`** | Callable that turns a list of dataset items into a padded batch tensor. | The wrong collator is the second most common silent bug. | The `Trainer` default is *not* "stack items" — it is `DataCollatorWithPadding` when a tokenizer is present. |
| **Dynamic padding** | Pad each batch to its own longest sequence instead of a global max. | 2–4× throughput on variable-length data, especially NER where sentences are ~20 tokens but `max_length=512`. | Not "same as truncation". Padding to 512 for 20-token sentences wastes 96% of compute. |
| **`fp16` / `bf16`** | Half-precision / bfloat16 training via AMP. | ~2× throughput and ~40% less activation memory on T4/A100. | `fp16` on a T4 without loss scaling *will* produce NaN loss. `bf16` needs Ampere+. |
| **`max_grad_norm`** | Global gradient-norm clipping threshold. | The stabiliser that makes LR 2e-5 and LR 5e-5 both survive. Default 1.0. | Disabling it makes LR sensitivity much worse, not better. |
| **Warmup ratio** | Fraction of total steps over which LR ramps linearly from 0 to the peak. | 10% is the encoder default. Protects the pretrained weights from a cold, large first step. | Warmup steps, not epochs. With `warmup_ratio=0.1` and 300 total steps, that is 30 steps. |
| **AdamW (decoupled)** | Adam with weight decay applied directly to the weights instead of folded into the gradient. | The correct optimiser for Transformers; `adamw_torch` is the modern HF default. | `torch.optim.Adam` with `weight_decay=0.01` is *L2 in the gradient* — a different and worse algorithm. |
| **Catastrophic forgetting** | Raising LR too high destroys the pretrained representation; train loss falls while eval loss rises. | The reason BERT fine-tuning LR is 2e-5, not 2e-3. | Blaming overfitting. The tell is that *train* loss keeps falling. |
| **`problem_type`** | `TrainingArguments`/model-config key selecting the loss: `single_label_classification` (default), `multi_label_classification` (BCE), `regression` (MSE). | Wrong value = silently wrong loss. Multi-label data with the default gives a softmax over mutually-exclusive classes. | Setting `num_labels=1` and expecting regression — you must also set `problem_type="regression"`. |
| **ONNX Runtime** | Cross-platform inference engine; `optimum` exports BERT to `.onnx` with fused attention. | 2–4× CPU latency reduction; int8 dynamic quantisation adds another 2–4×. | ONNX does not require a GPU and does not need the `transformers` package at serve time. |
| **Distillation** | Training a small "student" to match a large "teacher"'s softened output distribution (plus hard labels). | The origin of DistilBERT (66M vs 110M, 97% of GLUE). Bridges to CS-08. | Distillation is not pruning and not quantisation — the student has a *different architecture*. |
| **SetFit** | Few-shot classification: contrastive fine-tune a sentence transformer on pairs, then fit a logistic-regression head. 8 examples/class. | Beats GPT-3 few-shot prompting on classification at 1/1000 the inference cost. | It is not a BERT fine-tune — it is a *sentence-transformer* fine-tune plus a classical classifier. |
| **ModernBERT** | Dec 2024 encoder family (149M / 395M) with RoPE, GeGLU, alternating local/global attention, 8,192-token context, unpadding. | Supersedes BERT-base on every axis. See §Beyond-the-video. | Not a drop-in for every checkpoint — the tokenizer and head names still change. |
| **Calibration** | Whether `softmax` scores equal empirical accuracy. Measured by ECE; fixed by temperature scaling. | A confidence of 0.98 that is right 70% of the time will destroy your auto-approval threshold. | "Softmax output is a probability" — it is a normalised score, not a calibrated probability. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 Mechanism, step by step

**Stage A — Self-supervised pretraining (the part you never do yourself).**

1. Ingest 16 GB of text: BooksCorpus (800M words) + English Wikipedia (2,500M words) [13:44–15:06 in the video's framing: "collect a data from the various resources from the entire internet" and "this data basically it was not a label data"].
2. Build WordPiece vocabulary of 30,522 tokens. Lowercase everything (`uncased`) — a mistake for NER, see §10.
3. For each sequence: sample two spans A and B. **50%** of the time B genuinely follows A in the corpus (label `IsNext`); **50%** of the time B is a random sentence from a random document (label `NotNext`). This is NSP [14:02–14:24].
4. Select 15% of A∪B tokens. Apply 80/10/10.
5. Run 12 (or 24) encoder blocks. At the selected positions, compute cross-entropy against the original token id. At `[CLS]`, compute cross-entropy against `IsNext`/`NotNext`.
6. Total loss = MLM loss + NSP loss. Train with Adam, LR 1e-4, linear decay, 10k warmup steps, 1M steps, batch 256, on 16–64 TPU chips for ~4 days.

**Stage B — Supervised fine-tuning (the part this module is about).**

7. Load the pretrained encoder. Delete the MLM head and the NSP head. They are disposable.
8. Attach a new, randomly-initialised task head. The video's description is exact: *"this head is nothing ... it is a feed forward neural network ... this is also called dense network ... and along with that you will find out the softmax"* [19:36–20:09].
9. Train **all** parameters (or a chosen subset of layers) on your labelled data with a small LR.
10. The instructor's framing of the three strategies, verbatim in structure [36:52–37:52]: (i) **full retraining**, (ii) **selected layers** ("last two layer ... last five layer. It's up to me how many layers I want to be fine-tuned"), (iii) **last output layer only**.

**Stage C — Inference.** Drop the loss. Take `argmax` (classification/tagging) or run span decoding (QA).

### 4.2 The mathematics

**Notation.** `V` = vocabulary size (30,522), `H` = hidden size (768 for base), `L` = layers (12), `A` = heads (12), `d_k = H/A = 64`, `S` = sequence length, `B` = batch size, `d_ff = 3072`.

**(a) Encoder block.** For input `H_l ∈ R^{S×H}`:

```
Q = H_l W_Q,  K = H_l W_K,  V = H_l W_V      W_* ∈ R^{H×H}
Attention(Q,K,V) = softmax(Q Kᵀ / sqrt(d_k) + M) V
MHA  = Concat(head_1..head_A) W_O
H'   = LayerNorm(H_l + MHA)                  # post-LN (original BERT)
H_{l+1} = LayerNorm(H' + FFN(H'))            # FFN = GELU(x W_1 + b_1) W_2 + b_2
```

`M` is the additive attention mask: `0` for real tokens, `-inf` for `[PAD]`. **BERT's `M` has no causal component** — that single fact is the architectural reason it cannot generate. Every token in a BERT sequence sees every other token.

RoBERTa and most later encoders move the LayerNorm to the *pre*-LN position (`H + MHA(LayerNorm(H))`), which removes the need for LR warmup and is more stable — but the released BERT checkpoints are post-LN, so you cannot mix.

**(b) MLM loss.** Let `S_mask` be the set of selected positions (|S_mask| ≈ 0.15·S) and `x_i` the original token id.

```
L_MLM = -(1/|S_mask|) * Σ_{i ∈ S_mask} log P(x_i | corrupted input)
```

Only the 15% selected positions contribute. If `x̃_i` is the corrupted input token at position *i*:

```
P(x_i | ·) = softmax(W_mlm · h_i^L + b_mlm)     W_mlm ∈ R^{V×H}
```

**The 80/10/10 arithmetic, worked.** Sequence of `S = 10` real tokens → 15% of 10 = 1.5 → BERT rounds up, selecting **2** positions (the reference implementation uses `max(1, round(0.15·S))`). Of those 2:

| Outcome | Probability | Positions at S=2 (expected) |
|---|---|---|
| `[MASK]` | 0.80 | 1.6 |
| random token | 0.10 | 0.2 |
| unchanged | 0.10 | 0.2 |

**Why the 10% random and 10% unchanged exist.** If 100% of selections became `[MASK]`, the model would learn two things: (1) "the only position I must be careful about is the literal token `[MASK]`", and (2) "outside `[MASK]`, copy the input". At fine-tuning time `[MASK]` never appears, so condition (1) never triggers and every token representation is built by a model that has never had to encode a *corrupted* context. The 10% random token forces the model to keep a usable representation of the *original* token even when the input token is wrong (it must detect and correct), and the 10% unchanged guarantees that at least some of the loss-bearing positions see a fully clean context. Net effect: the representation at every position must be *robust*, not just at the ones the model can identify as corrupted.

**The `[MASK]` mismatch heuristic.** At fine-tuning the `[MASK]` token is absent, and BERT's authors found that fine-tuning is more accurate if the *pretraining* mask rate at fine-tuning time is set to your task's natural mask rate — for classification they used 0.15, for SQuAD **0.30**, for NER **0.10**. The intuition: `[MASK]` is not a real word, so it drags the distribution away from the token that appears in your data. This is the single strongest evidence that `[MASK]` was a defect, and it is precisely what ELECTRA and DeBERTa-v3 eliminated.

**(c) NSP loss.**

```
L_NSP = -[ y log p_IsNext + (1-y) log p_NotNext ]
p = softmax(W_nsp · h_[CLS]^L + b_nsp)      W_nsp ∈ R^{2×H}
```

**(d) Why NSP died.** The negative pairs are drawn from a *different document*. So a model can solve NSP without learning discourse structure at all: it detects a **topic shift**. `A = "the cat sat on the mat"`, `B = "the FED raised rates"` → obviously `NotNext`, no discourse reasoning required. RoBERTa's ablation (Liu et al., 2019) removed NSP and downstream scores *improved* slightly. ALBERT replaced it with SOP (`A-B` vs `B-A`, same document, same topic) and SOP-based models do learn ordering information. ELECTRA, SpanBERT, and DeBERTa all dropped NSP.

**(e) QA head and loss.**

```
start_logits = h^L W_s + b_s,     end_logits = h^L W_e + b_e      W_* ∈ R^{1×H}
L_QA = ½ [ CE(start_logits, y_s) + CE(end_logits, y_e) ]
```

Two independent softmaxes over the *same* 512 positions. There is no constraint that `argmax(start) ≤ argmax(end)` — that constraint is enforced in post-processing (§6.4) and its absence is a real bug source.

**(f) Token classification head and loss.**

```
logits = h^L W_tag + b_tag          W_tag ∈ R^{C×H}, C = number of BIO labels
L_tag = -(1/Σ 1[y_i ≠ -100]) * Σ_i 1[y_i ≠ -100] · log softmax(logits_i)[y_i]
```

The `-100` positions are excluded from the denominator as well as the numerator. If you forget to mask, the denominator inflates and the loss is dominated by pad/noise positions.

### 4.3 What happens at the tensor and gradient level

**Gradient flow and why the head is random.** At step 0 the head `W_head ∈ R^{C×H}` is Xavier/Normal-initialised with std ≈ 0.02. The encoder is not random — it already produces linearly separable representations for most tasks. So the loss starts at ≈ `ln(C)` (0.693 for binary, 2.197 for 9 NER labels) and drops fast in the first 50–100 steps as the head orients itself. If your step-0 loss is *far* from `ln(C)` — e.g. 8.0 for a binary task — your labels are misaligned or your `num_labels` is wrong.

**The two-timescale dynamic.** The head receives gradients `∂L/∂W_head = h^L · (p - y)` where `h^L` has norm ~O(10). The encoder receives `∂L/∂θ_enc = (p - y) · W_head · ∂h^L/∂θ_enc` — it is *mediated by a random matrix* at step 0, so early encoder gradients are noisy and small. This is exactly why the LR must be small: a large LR applied to a noisy, mediated gradient on 110M pretrained weights is how you destroy a good representation in 200 steps.

**Where catastrophic forgetting lives.** The pretrained weights sit in a region of parameter space that minimises a *different* loss (MLM+NSP over 3.3B words). Your downstream loss (CE over 1,000 reviews) has a different and much narrower basin. At LR 2e-3 the encoder moves far enough in 100 steps to leave the pretrained basin entirely — you get a model that fits your 1,000 examples perfectly and generalises worse than a linear probe on frozen features. At LR 2e-5 you move ~1% of the way, which is exactly the transfer-learning sweet spot. The video's phrase is *"in BERT model this full fine-tuning is also possible with a small learning rate"* [37:22–37:28], repeated four separate times in the practical section.

**Frozen vs full, gradient-level.** With `requires_grad=False` on the bottom layers, those layers still *run* forward (so they cost FLOPs and activations) but receive no gradient updates and need no optimizer state. That is where the memory saving comes from: AdamW keeps 2 fp32 moments per *trainable* parameter. Freezing 9 of 12 layers of BERT-base cuts trainable params from 110M to ~30M and optimiser memory from 2×4×110M = **880 MB** to **240 MB**.

### 4.4 Memory and compute accounting

**(a) Parameters, BERT-base.** Let me count them explicitly — interviewers ask this.

| Component | Count | Formula |
|---|---|---|
| Token embeddings | 23.4M | 30,522 × 768 |
| Position embeddings | 0.39M | 512 × 768 |
| Token type embeddings | 0.0015M | 2 × 768 |
| Per encoder layer — attention Q,K,V,O | 2.36M | 4 × 768² |
| Per encoder layer — FFN | 4.72M | 2 × 768 × 3072 |
| Per encoder layer — LayerNorms | 3,072 | 4 × 768 |
| **Per layer total** | **7.08M** | |
| 12 layers | 85.0M | 12 × 7.08M |
| Pooler | 0.59M | 768² |
| **Encoder total** | **≈ 109M** | |
| **+ classification head (num_labels=2)** | **1.5k** | 2 × 768 + 2 |
| **Reported BERT-base** | **110M** | |

BERT-large: 24 layers, `H=1024`, `A=16`, `d_ff=4096`, vocab 30,522. Per layer = 4·1024² + 2·1024·4096 = 4.19M + 8.39M = **12.6M**; × 24 = **302M**; + embeddings 31.2M + pooler 1.05M = **≈ 335M**, reported as **340M**.

**(b) Attention memory — the real cost.** For one sequence of `S=512`, one head, one layer, the attention score matrix is `512 × 512 × 4 bytes = 1 MB` in fp32. Per layer across 12 heads: **12 MB**. Across 12 layers: **151 MB** *per sequence*. With batch 8 that is **1.2 GB** just for attention activations — before the FFN activations, which are `B × S × d_ff × 4 = 8 × 512 × 3072 × 4 = 50 MB` per layer → **604 MB** over 12 layers.

**Worked VRAM budget — BERT-base full fine-tuning, batch 8, `max_length=512`, fp32:**

```
Weights                                110M × 4 B   = 0.44 GB
Gradients                              110M × 4 B   = 0.44 GB
AdamW m and v                          110M × 8 B   = 0.88 GB
Activations (attention 1.2 GB + FFN 0.6 GB, store-for-backward) ≈ 1.8 GB
Framework / cuDNN workspace                          ≈ 0.5 GB
---------------------------------------------------------------
TOTAL                                                ≈ 4.1 GB fp32
```

That is why `per_device_train_batch_size=8` at 512 tokens fits comfortably on a free Colab T4 (16 GB). The same job at batch 32 needs ~10 GB and at batch 64 ~18 GB — so batch 32 with fp16 (which halves the activation term) is the practical T4 ceiling for `max_length=512`.

**The dynamic-padding win, quantified.** WikiAnn English sentences average ~26 tokens. The notebook pads to `max_length=512` for every example. That is `512/26 ≈ 20×` wasted compute per batch. Real numbers:

| Setting | Tokens per batch (B=8) | Relative compute | Notes |
|---|---|---|---|
| `padding='max_length'`, 512 | 4,096 | 20.0× | Notebook default |
| `padding='max_length'`, 128 | 1,024 | 5.0× | Still wasteful |
| `DataCollatorForTokenClassification` (dynamic) | ~208 | **1.0×** | One line to enable |

**(c) FLOPs and wall-clock.** Forward FLOPs for a Transformer ≈ `2 × N_params × N_tokens` (the factor 2 is one multiply + one add per parameter per token). Training ≈ `3×` forward (fwd + 2× bwd).

```
BERT-base, batch 8 × 512 tokens = 4,096 tokens
forward  = 2 × 110e6 × 4096        = 0.90 TFLOP
training = 3 × 0.90                = 2.70 TFLOP / step
```

Free Colab T4: ~65 TFLOPS fp16 peak, but a small transformer with attention overhead and Python-side data loading realistically lands at 3–10 TFLOPS effective. fp32 (the notebook sets no `fp16=True`) is worse: T4 fp32 peak is 8.1 TFLOPS and effective is ~1–2 TFLOPS. So:

| Configuration | Effective TFLOP/s | s/step (B=8, S=512) | Steps/epoch @ 25k examples | Wall-clock/epoch |
|---|---|---|---|---|
| T4, fp32, static pad 512 | ~1.5 | ~1.8 s | 3,125 | **~94 min** |
| T4, fp16, static pad 512 | ~6 | ~0.45 s | 3,125 | **~23 min** |
| T4, fp16, dynamic pad (~256 real) | ~6 | ~0.23 s | 3,125 | **~12 min** |
| A10G, bf16, dynamic pad | ~25 | ~0.05 s | 3,125 | **~3 min** |

The instructor's *"at least 5 to 6 hour to train"* [32:12] for the full 25k is consistent with the fp32 row across multiple epochs plus tokenization overhead — and the fix is one line (`fp16=True`) plus one argument (`data_collator=...`).

**(d) Inference cost.** BERT-base forward at `S=128`, batch 32: `2 × 110e6 × 32 × 128 = 0.90 TFLOP` per batch → 28 GFLOP per prediction. On an A10G (125 TFLOPS fp16, ~25% realised for small batches) that is `28e9 / 6e12 ≈ 4.7 ms`; with kernel-launch and Python overhead, 10–30 ms is the observed single-request GPU figure. On a modern x86 CPU (≈50 GFLOPS effective fp32, AVX2) it is `28e9 / 50e9 = 0.56 s` — but the observed figure is 50–200 ms because a 128-token sequence is much smaller than the 28 GFLOP computed above if you recompute per *sequence* rather than per *prediction at S=128*: at `S=32` it is 7 GFLOP → 140 ms on CPU. **Batch your CPU inference and use ONNX int8** — that is the difference between a 200 ms and a 15 ms response.

---

### 4.5 Special tokens, the 512-token ceiling, and why `[CLS]` pooling works

**The five tokens that matter, with their exact ids in `bert-base-uncased`:**

| Token | Id | Where it comes from | What it does | What breaks if you get it wrong |
|---|---|---|---|---|
| `[PAD]` | **0** | Appended to reach a rectangular batch | Fills the batch. `attention_mask=0` for these positions | Confusing id 0 with "no label". In NER, `0` is a real tag (`O`) — use `-100` |
| `[UNK]` | **100** | Emitted for any character/subword not in the 30,522-entry vocab | The escape hatch for OOV | A high `[UNK]` rate destroys information irreversibly. Measure it before training |
| `[CLS]` | **101** | Prepended to every sequence | Its final hidden state is the pooled sequence vector, read by the classifier head | People use it for *similarity* — mean pooling is better there |
| `[SEP]` | **102** | Ends segment A and ends segment B | Separates the two segments of a pair; the segment embedding switches at it | Forgetting the trailing `[SEP]` when counting segments |
| `[MASK]` | **103** | Pretraining corruption | Replaces a selected token during MLM | It never appears at fine-tuning time; do not build features around it |

Plus `[unused0]`…`[unused99]` (ids 1–9, 10–99 are partially used by `[unused]` slots) reserved for you to add domain tokens without resizing the embedding matrix — see the domain-adaptation note below.

**Anatomy of a pair input.**

```
tokens:  [CLS]  what  is  the  capital  of  france  ?  [SEP]  paris  is  the  capital  [SEP] [PAD] ...
ids:      101   2054 2003 1996  3007   1997  2605  1029  102    3341  2003 1996 3007    102    0   ...
type_ids:  0     0    0    0     0      0     0     0     0      1     1    1     1      1     0   ...
attn_mask: 1     1    1    1     1      1     1     1     1      1     1    1     1      1     0   ...
pos_ids:   0     1    2    3     4      5     6     7     8      9    10   11    12     13    14   ...
```

Three separate embeddings are summed at the input: `token_emb[id] + position_emb[pos] + token_type_emb[type]`. The **position embedding is learned and absolute** — position 511 has a trained vector, position 512 does not exist in the table at all.

**The 512-token ceiling — three distinct consequences:**

1. **Hard architectural limit.** `max_position_embeddings=512` for BERT-base/large, 514 for RoBERTa (512 + 2 spare). Nothing beyond that index has an embedding; you cannot increase the table without retraining from scratch, and even then the model has never seen those positions.
2. **Quadratic economic limit.** Attention cost is `O(S²·H·L)`. Going from 512 to 4,096 tokens is a 64× increase in attention compute. That is why long-context encoders (Longformer, BigBird, ModernBERT's local/global alternation) exist: they change the attention *pattern* rather than the position table.
3. **Downstream task consequences, per task:**

| Task | What the ceiling costs you | The fix |
|---|---|---|
| Classification | Long documents are cut; the decisive sentence may be gone | Chunk + aggregate (mean/max of logits), or use a long-context encoder |
| NER | Entities near the end of a long document are never labelled | Sliding windows with overlap, then merge by character offset |
| QA | Answers past token ~494 are unreachable | `max_length=384`, `doc_stride=128`, `return_overflowing_tokens=True` |
| Retrieval | A document embedding only represents its first 512 tokens | Chunk into passages and embed each; that is exactly what a RAG index does |

**Why `[CLS]` pooling works.** Three reasons, and only the third is the real one:

1. `[CLS]` is not a word, so its input representation is *free* — the model has no lexical prior to override. Any token you chose from the sentence would be pre-loaded with that token's meaning.
2. Because BERT's attention is fully bidirectional, the `[CLS]` position's query can attend to every token in the sequence. Its representation is therefore a weighted summary of the whole input, with the weights learned.
3. **The NSP head trained it to be one.** The only task in pretraining whose label depends on the *whole* sequence (both segments) was NSP, and the NSP head read `[CLS]`. Gradient descent therefore had to make the `[CLS]` vector a usable summary of a sentence pair — otherwise NSP loss could not go down. Without NSP (RoBERTa), `[CLS]` is measurably weaker as a sentence vector, which is one reason RoBERTa fine-tunes well for *classification* (where the head is retrained anyway) but is no better than BERT for *similarity* without further training.

**Corollary: `[CLS]` is a classification vector, not a sentence embedding.**

| Pooling method | How | Best for | Typical STS correlation vs `[CLS]` |
|---|---|---|---|
| `[CLS]` raw | `last_hidden_state[:, 0]` | Nothing much | baseline |
| `[CLS]` + pooler | `tanh(W·h_0)` — what `BertPooler` does | Classification (the head reads past it) | ≈ baseline |
| Mean pooling | `(h * attention_mask).sum(1) / attention_mask.sum(1)` | Similarity, retrieval | **+2 to +6 points** |
| Max pooling | element-wise max over non-pad tokens | Some retrieval setups | +1 to +3 |
| `[CLS]` + contrastive training | Sentence-BERT / SimCSE | Production retrieval | **+10 to +20 points** |

```python
# Mean pooling — the correct sentence vector for retrieval, in five lines.
import torch
def mean_pool(model_output, attention_mask):
    token_emb = model_output.last_hidden_state            # (B, S, H)
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    summed = torch.sum(token_emb * mask, dim=1)           # (B, H)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)       # (B, 1) - never divide by 0
    return summed / counts
# Then L2-normalise, and use cosine similarity == dot product.
```

> **Beyond the video — domain vocabulary and the `[unused]` tokens.** When your
> domain has tokens the WordPiece vocabulary cannot represent (`ICD-10` codes,
> `CUSIP` identifiers, chemical formulae), you have three escalating options:
> 1. **Measure first.** `sum(tok(t).input_ids.count(100) for t in corpus) / sum(len(tok(t).input_ids) ...)`.
>    Below 1% `[UNK]` — ignore it. Above 3% — act.
> 2. **Continue pretraining the tokenizer** with `tokenizer.train_new_from_iterator(corpus, vocab_size=32768)`
>    then **resize the embedding table** and continue MLM for 10k–50k steps. This
>    is the domain-adaptive pretraining recipe (→ CS-12). The newly added rows are
>    random; you must train them.
> 3. **Use the reserved slots.** BERT ships 100 `[unused]` tokens. Replace their
>    embeddings with the mean of the subword embeddings of your domain terms and
>    freeze them, then add the terms to the tokenizer's `added_tokens`. This gives
>    you domain tokens without resizing the model — a 10-minute change instead of
>    a GPU-day, and usually worth 1–3 points on domain NER.

---

## 5. The End-to-End Pipeline

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ STAGE 0 — DECIDE                                                       │
  │ Is this a closed-label-set task over short text with high volume?       │
  │   Yes → encoder fine-tune (this module).   No → CS-13 (SFT) / CS-04.    │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 1 — DATA                                                          │
  │ in : raw text + labels (CSV/JSONL/HF dataset)                           │
  │ op : split train/val/test, audit label distribution, measure [UNK] rate  │
  │ out: DatasetDict with ~80/10/10 or 90/10 split, stratified              │
  │ ✗ fail: class imbalance >20:1 left unhandled; duplicate rows across      │
  │         train and test (leakage); label typos creating phantom classes   │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 2 — TOKENIZE                                                      │
  │ in : raw strings (or pre-tokenized word lists for NER)                  │
  │ op : FAST tokenizer; truncation=True; return word_ids / offset_mapping  │
  │ out: input_ids, attention_mask, token_type_ids, + task-specific targets  │
  │ ✗ fail: slow tokenizer (no word_ids()/offset_mapping); max_length too    │
  │         small so answers/labels are truncated away; no [UNK] audit       │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 3 — ALIGN (only for tagging & QA)                                 │
  │ in : word_ids (NER) / offset_mapping (QA)                               │
  │ op : NER  → first subword gets the label, continuations + specials get   │
  │            -100.  QA → char offsets of the gold answer mapped to token   │
  │            start/end indices; (0,0) if unanswerable (SQuAD 2.0)          │
  │ out: labels (int list, with -100)  /  start_positions, end_positions     │
  │ ✗ fail: label copied to EVERY subword (double-counted entities);        │
  │         -100 replaced by 0 so pads train as class 'O'; answer span       │
  │         silently defaults to (0,0) because offsets were never requested  │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 4 — BUILD THE MODEL                                               │
  │ in : model id ("bert-base-uncased") + num_labels                        │
  │ op : AutoModelFor{SequenceClassification,TokenClassification,           │
  │      QuestionAnswering}.from_pretrained(...)                            │
  │ out: encoder (pretrained) + randomly-initialised head                   │
  │ ✗ fail: num_labels mismatch with the label map (off-by-one on NER);      │
  │         num_labels omitted → defaults to 2, silently trains 2 classes    │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 5 — COLLATE + TRAIN                                               │
  │ in : datasets, TrainingArguments, collator                              │
  │ op : Trainer(model, args, train_dataset, eval_dataset, data_collator)   │
  │      → 2-4 epochs, LR 2e-5, warmup 10%, wd 0.01, clip 1.0               │
  │ out: ./out/checkpoint-*/ + final weights                                │
  │ ✗ fail: default collator pads labels with 0 for NER; no eval_dataset     │
  │         so eval_loss never printed; fp16 on T4 without loss scaling      │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 6 — EVALUATE                                                      │
  │ in : held-out test set                                                  │
  │ op : compute_metrics → accuracy/F1 (classification), seqeval F1 (NER),   │
  │      exact-match + F1 (QA)                                              │
  │ out: metrics dict + confusion matrix + per-class report                 │
  │ ✗ fail: reporting token-level accuracy for NER (85% and useless);        │
  │         single-number eval on a 500-row test set (CI ±4 points)          │
  └─────────────────────────────────────────────────────────────────────────┘
                                    │
  ┌─────────────────────────────────▼───────────────────────────────────────┐
  │ STAGE 7 — SAVE / PUBLISH / SERVE                                        │
  │ op : trainer.save_model(dir); tokenizer.save_pretrained(dir);           │
  │      optional trainer.push_to_hub(repo)                                 │
  │ out: config.json + model.safetensors + tokenizer files + (optional)      │
  │      ONNX export via optimum-cli                                        │
  │ ✗ fail: saving checkpoints but not the tokenizer; loading with a         │
  │         different tokenizer than you trained with → garbage predictions  │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Task-specific variant of the pipeline

| Stage | Classification | NER / token classification | Extractive QA |
|---|---|---|---|
| Input columns | `text`, `label` | `tokens` (list[str]), `ner_tags` (list[int]) | `question`, `context`, `answers{text,answer_start}` |
| Tokenizer call | `tokenizer(text, truncation=True, max_length=256)` | `tokenizer(tokens, is_split_into_words=True, truncation=True)` | `tokenizer(question, context, return_offsets_mapping=True, truncation=True)` |
| Extra tokenizer flags | — | `is_split_into_words=True` (mandatory) | `return_offsets_mapping=True` (mandatory) |
| Alignment step | none (1 row → 1 label) | `word_ids()` → first-subword label, `-100` elsewhere | `offset_mapping` → char→token index |
| Label column name | rename `label` → `labels` | keep `labels` (int list) | `start_positions`, `end_positions` |
| Model class | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` | `AutoModelForQuestionAnswering` |
| Head shape | `Linear(768, num_labels)` | `Linear(768, num_labels)` per position | `Linear(768, 2)` shared |
| Correct collator | `DataCollatorWithPadding` | **`DataCollatorForTokenClassification`** | `DataCollatorWithPadding` |
| Metric | accuracy / F1 (`binary` or `macro`) | **`seqeval` entity micro-F1** | exact match + token F1 |
| Decoding | `argmax(logits, -1)` | `argmax(logits, -1)` per token | span search + `n_best` + offsets |

---

## 6. Hands-On Code (annotated)

All code below is reproduced and corrected from `BERT_Finetuning.ipynb`. The video's install line is exact and is worth keeping, because the `fsspec`/`datasets` version pairing is the most common install failure:

```bash
# The instructor's exact install command [29:18-29:38] — notebook cell 2.
# He explains fsspec as "it help us to download any file from server to the local".
!pip install --upgrade datasets fsspec transformers
```

> **Beyond the video:** pin versions for reproducibility. `pip install --upgrade` in a
> notebook is a production anti-pattern: your result is not reproducible next week.
> ```bash
> pip install "transformers==4.44.2" "datasets==2.21.0" "accelerate==0.34.2" \
>             "evaluate==0.4.3" "seqeval==1.2.2" "scikit-learn==1.5.1" "torch==2.4.1"
> ```
> `seqeval` is **not** installed by the video's command and the NER cell will fail
> without it. The notebook's own error handler at the bottom admits this: it tells
> you to run `pip install torch transformers datasets scikit-learn tqdm numpy`.

### 6.1 Task 1 — Sentiment classification, HF `Trainer` path

This is the "First Part" of the notebook and the exact code walked through on camera from [29:16] to [51:00].

```python
# --- cell 4: imports -----------------------------------------------------------
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --- cells 5-6: the instructor's dataset menu and the real-world mapping -------
# load_dataset("ag_news")        # 4-class news topic, 120k train
# load_dataset("dbpedia_14")     # 14-class ontology, 560k train  [30:22-30:34]
#
# Real-world analogues he names explicitly [30:51-31:23]:
#   customer feedback categorisation  -> positive / negative / neutral
#   support ticket classification     -> billing / technical / general
#   email topic classification        -> ham / spam

# --- cells 7-9: load + subset --------------------------------------------------
dataset = load_dataset("imdb")                       # 25,000 train / 25,000 test
train_dataset = dataset["train"].select(range(1000))  # video uses 1,000  [32:20]
test_dataset  = dataset["test"].select(range(500))    # video uses 500    [32:22]

# --- cell 10: tokenizer --------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- cell 12: the tokenization function ---------------------------------------
def tokenize_fn(examples):
    # padding="max_length" pads EVERY sequence to 256, even a 12-word review.
    # truncation=True caps long reviews; the instructor explains the trade-off
    # at [33:36-33:51]: 300 tokens -> truncate to 256; 100 tokens -> 156 zeros.
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

# --- cell 13: map + rename + format in one flow -------------------------------
def preprocess(ds):
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])  # frees RAM
    ds = ds.rename_column("label", "labels")   # HF REQUIRES the plural name
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_dataset = preprocess(train_dataset)
test_dataset  = preprocess(test_dataset)

# --- cell 16: model -----------------------------------------------------------
# num_labels=2 builds a fresh Linear(768, 2) head. It is RANDOM at this point.
# Omitting num_labels does NOT error for BERT (it falls back to 2), which is
# exactly why forgetting it on a 5-class problem is a silent disaster.
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# --- cell 17: inspect the encoder ---------------------------------------------
# Prints 12 BertLayer objects -> live proof that BERT-base is 12 layers.
for layer in model.bert.encoder.layer:
    print(layer)

# --- cell 18 (commented out in the notebook): selective fine-tuning -----------
# "Classifier head trainable rahega by default" — the head is always trainable.
# for param in model.bert.parameters():
#     param.requires_grad = False            # freeze all 12 encoder layers
# for layer in model.bert.encoder.layer[-2:]:
#     for param in layer.parameters():
#         param.requires_grad = True         # unfreeze the top 2

# --- cell 19: training arguments ----------------------------------------------
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",   # checkpoints land here
    num_train_epochs=1,                   # 1 epoch on 1,000 rows is a demo, not a recipe
    per_device_train_batch_size=8,        # "8 data point would be there" [40:34]
    logging_dir="./logs",                 # TensorBoard reads this
    learning_rate=2e-5,                   # the canonical encoder LR
    weight_decay=0.01,                    # AdamW decoupled decay
    report_to="none",                     # he disables W&B/TensorBoard auto-reporting
)

# --- cell 20: Trainer ---------------------------------------------------------
# NOTE: no data_collator and no compute_metrics. That is deliberate here
# (padding is already fixed-length) but means trainer.evaluate() reports ONLY
# eval_loss — accuracy requires either compute_metrics or your own loop.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# --- cell 21: train -----------------------------------------------------------
trainer.train()

# --- cell 22: visualise -------------------------------------------------------
!tensorboard --logdir=./logs

# --- cells 23-24: save the COMPLETE model, not just a checkpoint --------------
# The instructor draws the distinction at [46:22-46:34]: checkpoints save "the
# state of the model" at a step; save_model writes loadable weights.
trainer.save_model("./bert-finetuned-imdb")
tokenizer.save_pretrained("./bert-finetuned-imdb")   # DO NOT SKIP THIS

# --- cell 25-26: evaluate -----------------------------------------------------
metrics = trainer.evaluate()
print(metrics)
# -> {'eval_loss': ..., 'eval_runtime': ..., 'eval_samples_per_second': ...,
#     'eval_steps_per_second': ...}      [47:19-47:36]

# --- cells 28-31: inference through pipeline() --------------------------------
tokenizer = BertTokenizer.from_pretrained("/content/bert-finetuned-imdb")
model = BertForSequenceClassification.from_pretrained("/content/bert-finetuned-imdb")
# NOTE: this reuses the tokenizer INSTANCE variable, not the from_pretrained one,
# exactly as in the notebook — the second line is a no-op for the tokenizer.
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This movie was amazing and I loved the acting!")
print(result)
```

> **Correction:** the notebook comment on cell 31 reads
> `# Example: [{'label': 'POSITIVE', 'score': 0.98}]`, and the instructor says at
> [48:38] *"label label is zero score is this much means it is a positive review."*
> **Both are wrong in an important way.** Three separate problems:
> 1. **The label string is not `POSITIVE`/`NEGATIVE`.** `BertForSequenceClassification`
>    built from `bert-base-uncased` has `id2label = {0: "LABEL_0", 1: "LABEL_1"}` in
>    its `config.json`, and fine-tuning with `Trainer` preserves that config. The
>    pipeline therefore returns `{'label': 'LABEL_0', ...}` — not `POSITIVE`. To get
>    names you must set them explicitly:
>    `model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}` (and `label2id`).
> 2. **IMDb label 0 is NEGATIVE, not positive.** The `imdb` dataset is
>    `ClassLabel(names=['neg', 'pos'])`, so `0 = neg`. A pipeline output of
>    `LABEL_0` on *"This movie was amazing"* means the model got it **wrong** (it
>    trained for 1 epoch on 1,000 examples — that is well within noise). Reading
>    `LABEL_0` as "positive" inverts your entire evaluation.
> 3. **The `score` is a softmax over `LABEL_0`/`LABEL_1`**, so it is the *uncalibrated*
>    probability of the returned label. On 1,000 training rows expect it to be
>    badly overconfident (§12).
>
> The fix is three lines, and every production classifier needs them:
> ```python
> model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
> model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
> model.config.save_pretrained("./bert-finetuned-imdb")
> ```

> **Beyond the video:** the notebook's own commented-out cell 37 contains the
> *better* version of this same recipe — the instructor wrote it and then
> commented it out. It uses `DataCollatorWithPadding` for dynamic padding and adds
> `per_device_eval_batch_size`, `logging_steps`, `eval_steps`, `save_steps`,
> `save_total_limit=1`. Uncomment it; it is strictly superior to what is run
> on camera:
> ```python
> from transformers import DataCollatorWithPadding
> def tokenize_fn(examples):
>     return tokenizer(examples["text"], truncation=True, max_length=256)  # no padding!
> ...
> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
> trainer = Trainer(..., data_collator=data_collator)
> ```
> With `padding="max_length"` a 12-token review costs the same as a 256-token one.
> With a collator, a batch of eight short reviews pads to ~20 tokens. On a
> review-length distribution with mean 130, that is a **2–4× throughput win**.

### 6.2 Task 1b — The same task with a hand-written PyTorch loop

The notebook's second half abandons `Trainer` and builds everything by hand. The instructor justifies it at [53:19–53:44] by naming *two ways* to build a dataset: the HF `map() + set_format()` path, and a **custom `torch.utils.data.Dataset`** class.

**Why the custom class exists** (notebook cell 42, verbatim rationale): you use it when
you have not preprocessed into HF format, when you only have Python lists, or when you need custom preprocessing.

**The dunder-method demo** (cells 43–45) is the clearest thing in the notebook for understanding `Dataset`:

```python
# --- cells 43-45: the "Basket" teaching example -------------------------------
class Basket:
    def __init__(self, fruits):
        self.fruits = fruits          # constructor: what the object holds
    def __len__(self):
        return len(self.fruits)       # len(basket)  -> 3
    def __getitem__(self, idx):
        return self.fruits[idx]       # basket[0]    -> "Apple"

basket = Basket(["Apple", "Banana", "Mango"])
print(len(basket))   # 3        <- __len__  is what DataLoader calls to size an epoch
print(basket[0])     # Apple    <- __getitem__ is what DataLoader calls per index
```

That is the entire `torch.utils.data.Dataset` contract: implement `__len__` and `__getitem__`, and `DataLoader` can batch, shuffle, and (with `num_workers>0`) parallelise you.

```python
# --- cell 39: imports for the manual path --------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertTokenizerFast,
    BertForSequenceClassification, BertForTokenClassification,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup,   # "Gradually warms up then decays LR
)                                      #  for stable BERT training" (nb. comment)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW          # "Adam Optimizer with Weight Decay" (nb.)

# --- cell 40: device ----------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- cell 46: the classification Dataset --------------------------------------
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',        # <- per-ITEM padding; slow, see note
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),       # (512,)
            'attention_mask': encoding['attention_mask'].flatten(),  # (512,)
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long),  # scalar
        }
```

> **Beyond the video:** tokenizing inside `__getitem__` with `padding='max_length'`
> and `max_length=512` is the single biggest performance mistake in this notebook
> for the classification path. It tokenizes one row at a time (no `batched=True`),
> pads IMDB reviews to 512 tokens (mean review length is ~230 words ≈ 300
> subwords, so half the batch is padding), and does that work again every epoch in
> the main process because `num_workers` defaults to 0. Three fixes, in order of
> payoff:
> 1. Set `max_length=256` (the video's own first-part value) — IMDB reviews past
>    256 subwords rarely carry the decisive sentiment cue.
> 2. Pre-tokenize once with `dataset.map(tokenize_fn, batched=True)` and store
>    them; then `__getitem__` is a list index, not a tokenizer call.
> 3. Pass `num_workers=4, pin_memory=True` to `DataLoader` on a GPU box.

```python
# --- cell 48 (excerpt): the manual training loop -------------------------------
class BERTTextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=512):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes)
        self.model.to(device)

    def train(self, train_texts, train_labels, epochs=1, batch_size=8, learning_rate=2e-5):
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # torch.optim.AdamW applies weight decay to EVERY parameter, including
        # LayerNorm gammas, biases, and the [PAD] embedding row. HF's Trainer with
        # optim="adamw_torch" does the same. HF's older "adamw_hf" excluded them.
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_steps = len(train_loader) * epochs          # 125 * 1 = 125 steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,           # 10% warmup  [nb. cell 41]
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            total_loss = 0
            for batch in progress_bar:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device),
                )
                loss = outputs.loss          # already mean CE over the batch
                total_loss += loss.item()
                loss.backward()
                # Clip BEFORE step, AFTER backward — order is not optional.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()             # step the LR scheduler every batch
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader):.4f}')

    def evaluate(self, test_texts, test_labels, batch_size=8):
        test_dataset = TextClassificationDataset(
            test_texts, test_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():                # mandatory: else VRAM grows every batch
            for batch in tqdm(test_loader, desc='Evaluating'):
                outputs = self.model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                )                            # no labels -> no loss, logits only
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(batch['labels'].numpy())
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')   # see note
        report = classification_report(true_labels, predictions,
                                       target_names=['Negative', 'Positive'])
        return accuracy, f1, report

    def predict(self, texts):
        self.model.eval()
        predictions, probabilities = [], []
        for text in texts:
            encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                      max_length=self.max_length, return_tensors='pt')
            with torch.no_grad():
                logits = self.model(
                    input_ids=encoding['input_ids'].to(device),
                    attention_mask=encoding['attention_mask'].to(device),
                ).logits
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy()[0])
        return predictions, probabilities
```

> **Beyond the video — three defects in that loop, and their fixes:**
>
> 1. `average='weighted'` on a binary task. For binary single-label problems,
>    `weighted` F1 equals accuracy whenever the model's per-class recall is not
>    degenerate, and it *always* flatters a model that predicts the majority
>    class. Use `average='binary', pos_label=1` when the positive class is the one
>    you care about, or `average='macro'` when both classes matter equally. On
>    IMDb with `neg`/`pos` roughly balanced the three agree; swap in a 5%-positive
>    fraud dataset and they diverge violently.
> 2. `predict()` loops one text at a time. For 1M predictions that is 1M Python
>    iterations, 1M tokenizer calls, and 1M kernel launches. Batch it (see §16)
>    and you get 20–50×.
> 3. `np.random.choice` in `load_imdb_data` has **no `random_state`**. Every run
>    samples different rows, so a "regression" from 0.86 to 0.83 accuracy might be
>    a different 1,000 rows. Always set the seed and always use the *same* test
>    set across experiments.

**The worked warmup arithmetic** (notebook cell 41, a genuinely good piece of teaching and the number an interviewer will ask you to produce):

```
dataset   = 800 samples
batch     = 8          -> 800 / 8            = 100 batches per epoch
epochs    = 3          -> 100 * 3            = 300 total optimizer steps
warmup    = 10%        -> 300 * 0.10         = 30 warmup steps
decay     = remaining  -> 300 - 30           = 270 steps of linear decay
```

Note the inconsistency: the demo actually loads `sample_size=1000` → `1,000/8 = 125` batches → **375** total steps → **37** warmup steps. The 800/300 figure is illustrative. Both are correct arithmetic for their stated inputs; do not quote 300 for the shipped demo.

### 6.3 Task 2 — NER / token classification (and the alignment bug)

The alignment walk-through the notebook gives in cell 49 is the single most valuable page in the companion material. Reproduced exactly:

```
Input words :  John | lives | in | London
BERT tokens :  [CLS], John, lives, in, Lon, ##don, [SEP], [PAD]...
word_ids    :  None,   0,    1,    2,   3,    3,     None,  None...

Step 2 — Label alignment
input_ids   :  [101, 1001, 2002, 1999, 3001, 3010, 102, 0, 0, 0]
word_ids    :  [None,   0,    1,    2,    3,    3,  None, None, None, None]
gold labels :          1,    0,    0,    2
aligned     :  [-100,   1,    0,    0,    2, -100, -100, -100, -100, -100]
                             ^    ^    ^    ^
                             |    |    |    +-- B-LOC for "London" (first subword)
                             |    |    +------- O   for "in"
                             |    +------------ O   for "lives"
                             +----------------- B-PER for "John"
                                                    "##don" gets -100: it is the
                                                    SAME word, so the label was
                                                    already emitted.
```

**The label table** the notebook provides (cell 51) is the CoNLL-2003 nine-label scheme:

| Label | Full form | Meaning | Example |
|---|---|---|---|
| `O` | Outside | no entity | "works", "at" |
| `B-PER` | Begin-Person | first word of a person name | `John` |
| `I-PER` | Inside-Person | continuation of a person name | `Mary Jane` → `Mary=B-PER`, `Jane=I-PER` |
| `B-ORG` | Begin-Organization | first word of an organisation | `Google` |
| `I-ORG` | Inside-Organization | continuation | `New York Times` → `New=B-ORG`, `York=I-ORG`, `Times=I-ORG` |
| `B-LOC` | Begin-Location | first word of a place | `London` |
| `I-LOC` | Inside-Location | continuation | `New York` → `New=B-LOC`, `York=I-LOC` |
| `B-MISC` | Begin-Miscellaneous | event, product, nationality | `Indian` (nationality) |
| `I-MISC` | Inside-Miscellaneous | continuation | `South Korean` → `South=B-MISC`, `Korean=I-MISC` |

B = begin, I = inside, O = outside (notebook cell 52).

```python
# --- cell 50: the NER Dataset, with the alignment loop -------------------------
class NERDataset(Dataset):
    def __init__(self, tokens_list, labels_list, tokenizer, max_length=512):
        self.tokens_list, self.labels_list = tokens_list, labels_list
        self.tokenizer, self.max_length = tokenizer, max_length

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            is_split_into_words=True,     # <- REQUIRED: tokens are already words
            return_tensors='pt',
        )

        # word_ids() exists ONLY on fast tokenizers (BertTokenizerFast).
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)          # [CLS], [SEP], [PAD]
            elif word_idx != previous_word_idx:
                # FIRST subword of a word carries the label
                aligned_labels.append(
                    labels[word_idx] if word_idx < len(labels) else 0)
            else:
                aligned_labels.append(-100)          # continuation subword
            previous_word_idx = word_idx

        # The tokenizer already padded to max_length; this is a defensive no-op.
        if len(aligned_labels) < self.max_length:
            aligned_labels += [-100] * (self.max_length - len(aligned_labels))
        elif len(aligned_labels) > self.max_length:
            aligned_labels = aligned_labels[:self.max_length]

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(aligned_labels, dtype=torch.long),
        }
```

> **Correction — two real defects in the notebook's NER setup:**
>
> 1. **`num_labels=9` is wrong for WikiAnn.** `load_dataset("wikiann", "en")`
>    exposes 7 labels in this order: `['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG',
>    'B-LOC', 'I-LOC']`. The notebook hard-codes nine (`... 'B-MISC', 'I-MISC'`),
>    copied from the CoNLL-2003 scheme. Consequence: the model has **two output
>    units that never receive a positive label**. They still receive gradients
>    (softmax pushes their probability toward 0), so training does not crash, and
>    because the notebook's first seven names happen to match WikiAnn's order,
>    `self.labels[pred]` still decodes the other seven correctly. **The bug is
>    invisible in the output.** Fix it by reading the truth from the data, never
>    by hand-typing:
>    ```python
>    from datasets import load_dataset
>    ds = load_dataset("wikiann", "en")
>    label_names = ds["train"].features["ner_tags"].feature.names
>    # ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']  -> num_labels=7
>    model = BertForTokenClassification.from_pretrained(
>        "bert-base-uncased", num_labels=len(label_names))
>    ```
>
> 2. **`else 0` in the alignment loop is a landmine.** `labels[word_idx] if
>    word_idx < len(labels) else 0` maps an out-of-range word index to label `0`,
>    which is `O` — a *valid, loss-bearing* label. The correct fallback when a
>    word index exceeds the gold list (which happens only if your token list and
>    label list are out of sync — itself a data bug) is `-100`, so the mistake is
>    loud instead of silent. Better still: assert the lengths match at load time:
>    ```python
>    assert len(tokens) == len(labels), f"row {idx}: {len(tokens)} tokens, {len(labels)} labels"
>    ```
>
> 3. **`padding='max_length', max_length=512` on 26-token sentences.** As §4.4
>    shows, this is 20× wasted compute. The fix is to drop padding from the
>    tokenizer and add the collator — which is exactly what
>    `DataCollatorForTokenClassification` does, and why it exists.

```python
# --- cell 54 (excerpt): the NER model wrapper ---------------------------------
class BERTNERClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=9, max_length=512):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.model.to(device)
        # Hard-coded label map — see the Correction above.
        self.labels = ['O','B-PER','I-PER','B-ORG','I-ORG',
                       'B-LOC','I-LOC','B-MISC','I-MISC']

    def load_wikiann_data(self, sample_size=1000):
        dataset = load_dataset("wikiann", "en")
        train_indices = np.random.choice(len(dataset['train']),
                                         min(sample_size, len(dataset['train'])),
                                         replace=False)          # no seed!
        test_indices  = np.random.choice(len(dataset['test']),
                                         min(sample_size//4, len(dataset['test'])),
                                         replace=False)
        train_tokens = [dataset['train'][int(i)]['tokens']   for i in train_indices]
        train_labels = [dataset['train'][int(i)]['ner_tags'] for i in train_indices]
        test_tokens  = [dataset['test'][int(i)]['tokens']    for i in test_indices]
        test_labels  = [dataset['test'][int(i)]['ner_tags']  for i in test_indices]
        return train_tokens, train_labels, test_tokens, test_labels

    def evaluate(self, test_tokens, test_labels, batch_size=8):
        # ... forward pass omitted for brevity, identical to the classification loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch['input_ids'].to(device),
                                     attention_mask=batch['attention_mask'].to(device))
                logits = outputs.logits                                   # (B, S, C)
                preds  = torch.argmax(logits, dim=2).cpu().numpy()        # (B, S)
                labels = batch['labels'].numpy()                          # (B, S)
                for i in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        if labels[i][j] != -100:      # skip specials + continuations
                            predictions.append(preds[i][j])
                            true_labels.append(labels[i][j])
        accuracy = accuracy_score(true_labels, predictions)   # <- the lie
        f1 = f1_score(true_labels, predictions, average='weighted')   # <- also the lie
        return accuracy, f1
```

> **Correction — and this is the #1 silent bug of the whole module.** The
> `evaluate()` above reports **token-level** accuracy and weighted F1 and calls it
> NER quality. It is not. Two reasons:
>
> 1. **Class dominance.** In CoNLL-2003 about **83%** of tokens are `O`; in
>    WikiAnn English it is higher still for short sentences. A model that emits
>    `O` for every single token — a model that has learned *nothing* — scores
>    ~0.83–0.90 accuracy and 0.90 weighted F1. Your dashboard will look healthy
>    while the model extracts zero entities.
> 2. **Boundary blindness.** Token-level scoring gives partial credit. If gold is
>    `B-PER I-PER` for "Mary Jane" and you predict `B-PER O`, you score 1/2 on the
>    entity. The business does not want half a name — it wants the whole name or
>    nothing. Worse, token scoring cannot distinguish `B-ORG I-ORG` from
>    `I-ORG I-ORG` or from `B-LOC I-LOC`, all of which are different failures.
>
> **The fix is `seqeval`, which scores whole entities:**
> ```python
> import numpy as np
> from evaluate import load as load_metric
> seqeval = load_metric("seqeval")
>
> def compute_metrics(eval_pred):
>     logits, labels = eval_pred
>     preds = np.argmax(logits, axis=-1)
>     # seqeval needs STRING labels and must NOT see -100.
>     true_preds, true_labels = [], []
>     for p_row, l_row in zip(preds, labels):
>         true_preds.append([label_names[p] for p, l in zip(p_row, l_row) if l != -100])
>         true_labels.append([label_names[l] for p, l in zip(p_row, l_row) if l != -100])
>     results = seqeval.compute(predictions=true_preds, references=true_labels)
>     return {
>         "precision": results["overall_precision"],
>         "recall":    results["overall_recall"],
>         "f1":        results["overall_f1"],
>         "accuracy":  results["overall_accuracy"],   # kept for contrast only
>     }
> ```
> A well-fine-tuned BERT-base on full CoNLL-2003 English reaches **~0.92–0.93
> entity-level F1**. On 500 WikiAnn sentences you should expect **0.65–0.78** —
> small sample, noisy tags, no CRF layer. If you see F1 above 0.95 on WikiAnn,
> you have a leak or your `-100` masking is wrong.

**Handling spans that split across subwords.** Two policies, both defensible:

| Policy | How | When |
|---|---|---|
| **First-subword only** (BERT paper / HF default) | Label only `word_ids()[i] != previous`; everything else `-100`. | Default. Maximises the clean signal per entity. |
| **All-subwords (`label_all_tokens=True`)** | Propagate `B-` to the first subword and `I-` to continuations. | Better for BILOU schemes and when you want dense supervision. Requires rewriting `B-X` → `I-X` on continuation pieces. |

The `DataCollatorForTokenClassification` has the flag; the notebook's hand-rolled loop does not. With first-subword-only you get **one prediction per entity** at inference, which is what the notebook's `predict()` does:

```python
# --- cell 54 (excerpt): predict() — aligning predictions BACK to words ---------
word_ids = encoding.word_ids(batch_index=0)
previous_word_idx = None
for i, word_idx in enumerate(word_ids):
    if word_idx is not None and word_idx != previous_word_idx:
        if word_idx < len(tokens):
            token_predictions.append(self.labels[preds[i]])
    previous_word_idx = word_idx
# -> one label per ORIGINAL word, exactly mirroring the training alignment.
#    Train and inference MUST use the same policy or your metrics and your
#    production output will disagree.
```

### 6.4 Task 3 — Extractive Question Answering (SQuAD)

```python
# --- cell 55: the QA Dataset ---------------------------------------------------
class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer, max_length=512):
        self.questions, self.contexts, self.answers = questions, contexts, answers
        self.tokenizer, self.max_length = tokenizer, max_length

    def __getitem__(self, idx):
        answer = self.answers[idx]        # {'text': [...], 'answer_start': [...]}

        encoding = self.tokenizer(
            self.questions[idx],
            self.contexts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,   # <- MANDATORY for QA
            return_tensors='pt',
        )
        offset_mapping = encoding.pop("offset_mapping")[0]   # pop! it is not a model input

        start_positions = torch.tensor(0, dtype=torch.long)
        end_positions   = torch.tensor(0, dtype=torch.long)

        if answer and 'answer_start' in answer and answer['answer_start']:
            answer_start = answer['answer_start'][0]
            answer_text  = answer['text'][0]
            answer_end   = answer_start + len(answer_text)

            for i, (start, end) in enumerate(offset_mapping):
                if start <= answer_start < end:      # token containing the 1st char
                    start_positions = torch.tensor(i, dtype=torch.long)
                if start < answer_end <= end:        # token containing the last char
                    end_positions = torch.tensor(i, dtype=torch.long)
                    break

        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'start_positions': start_positions,
            'end_positions':   end_positions,
        }
```

> **Beyond the video — five things that break this QA recipe in production:**
>
> 1. **No `doc_stride`.** `truncation=True` at `max_length=512` simply cuts the
>    context. With `bert-base-uncased` the question takes ~15 tokens and `[CLS]`/`[SEP]`
>    take 3, leaving ~494 context tokens. A SQuAD paragraph is ~150 tokens so it
>    fits, but a real support article or a legal clause does not. The standard fix
>    is a **sliding window**:
>    ```python
>    tokenizer(question, context,
>              max_length=384, stride=128,          # stride = doc_stride
>              truncation="only_second",            # never truncate the question
>              return_overflowing_tokens=True,      # one row -> N windows
>              return_offsets_mapping=True,
>              padding="max_length")
>    ```
>    `doc_stride` is the *overlap*; `max_length - doc_stride` is the advance. With
>    `max_length=384` and `doc_stride=128` you advance 256 tokens per window. The
>    rule of thumb: **`doc_stride` must exceed the longest answer you expect.**
>    The instructor's demo has neither, which is why it can only answer questions
>    whose answer lies in the first ~494 tokens.
> 2. **The `(0, 0)` default silently teaches "the answer is `[CLS]`".** When the
>    answer is missing or falls in a truncated region, the code leaves both
>    positions at 0 and still computes a loss. Position 0 is `[CLS]`. Train long
>    enough and the model learns that the cheapest way to reduce loss is to point
>    at `[CLS]`. **SQuAD 2.0 formalises this**: the no-answer case is exactly the
>    model putting high start *and* end logit on the `[CLS]` index, and the
>    standard trick is to add a learned `cls_logit` to it so it is not dominated by
>    real-word logits. With SQuAD 1.1 (`load_dataset("squad")`, which is what the
>    notebook uses) every training example has an answer, so the branch is dead
>    code — but the moment you point it at your own data with unanswerable
>    questions, you inherit the bug.
> 3. **`answer_start` is a *character* offset into the ORIGINAL context**, and
>    `offset_mapping` is in *characters of the tokenized-and-cleaned* text. If your
>    tokenizer strips accents or normalises whitespace, offsets drift and the
>    boundary test silently lands on the wrong token. Always assert
>    `context[answer_start:answer_end] == answer_text` before trusting the dataset.
> 4. **The start and end loops are independent.** The `break` only fires for the
>    end, and there is no `start <= end` guarantee. Assert
>    `start_positions <= end_positions` and skip the row otherwise.
> 5. **`max_answer_len` is a post-processing guard, not a model constraint.** The
>    notebook passes `max_answer_len=30` (§ `answer_question` below). SQuAD's own
>    dev answers max out at 30 tokens, which is where the number comes from. For
>    your domain, set it to the 99th percentile of gold answer length.

```python
# --- cell 56 (excerpt): answer_question — the greedy decoder --------------------
def answer_question(self, question, context, max_answer_len=30):
    encoding = self.tokenizer(question, context, truncation=True,
                              padding='max_length', max_length=self.max_length,
                              return_tensors='pt')
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    self.model.eval()
    with torch.no_grad():
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        start_idx = torch.argmax(outputs.start_logits, dim=1).item()
        end_idx   = torch.argmax(outputs.end_logits,   dim=1).item()

        if end_idx < start_idx:                     # enforce start <= end
            end_idx = start_idx
        if (end_idx - start_idx) > max_answer_len:  # enforce max span
            end_idx = start_idx + max_answer_len

        answer_tokens = input_ids[0][start_idx:end_idx+1]
        return self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
```

> **Correction — `argmax` over the full padded sequence is not span decoding.**
> The greedy version above has four defects that a real QA evaluation will expose
> immediately:
> 1. **It can select a `[PAD]` position.** `start_logits` covers all 512 positions
>    including the ~400 pads of a short context. The correct filter excludes
>    positions where `attention_mask == 0` and where `offset_mapping == (0,0)`
>    (the specials), before `argmax`.
> 2. **The `end_idx < start_idx → end_idx = start_idx` repair** turns a broken
>    span into a one-token answer, which is always wrong and never flagged.
> 3. **`n_best_size=1`.** Taking the single best start and single best end loses
>    the case where the true span is `(2nd-best start, best end)`. The standard
>    postprocess keeps the top **20** starts and top **20** ends, enumerates the
>    ≤400 candidate spans with `start <= end` and `length <= max_answer_length`,
>    drops spans that are fully inside the question, sums `start_logit + end_logit`
>    as the score, then de-duplicates by answer text.
> 4. **No score returned**, so there is no way to threshold or to say "I don't
>    know". Every production QA system needs a confidence threshold and an
>    abstain path.
>
> The `transformers` library ships this correctly for you:
> ```python
> from transformers import pipeline
> qa = pipeline("question-answering", model=model, tokenizer=tokenizer,
>               max_answer_len=30, handle_impossible_answer=True, top_k=3)
> qa(question=..., context=...)   # returns {'score','start','end','answer'}
> ```
> At scale, prefer the `pipeline` object or a `Trainer.predict` + `postprocess_qa_predictions`
> batch pass over a per-question Python loop — the notebook's version cannot batch
> and will be 10–30× slower than it needs to be.

### 6.5 The multi-task driver and publishing

```python
# --- cell 60: the generic multi-task driver [51:16-53:04] ----------------------
print("BERT Multi-Task Demo")
print("Choose a task to run:")
print("1. Text Classification (Sentiment Analysis)")
print("2. Named Entity Recognition (NER)")
print("3. Question Answering")
print("4. Run All Tasks")
choice = input("\nEnter your choice (1-4): ").strip()

if choice == "1":   run_text_classification_demo()
elif choice == "2": run_ner_demo()
elif choice == "3": run_qa_demo()
elif choice == "4": run_text_classification_demo(); run_ner_demo(); run_qa_demo()
```
The instructor walks through this live: press 1 for classification, 2 for NER, 3 for QA, and 4 to run all three sequentially [51:32–53:04]. The value is not the menu — it is that **one generic code path covers all three tasks**, because the only differences are the model class, the label column, and the head.

```python
# --- cells 33-36: pushing to the Hub [48:48-50:57] -----------------------------
from huggingface_hub import notebook_login
notebook_login()                                  # paste a WRITE token
from huggingface_hub import whoami
print(whoami())                                   # verifies the token's scope

tokenizer.push_to_hub("sunny199/my-bert-imdb2")
trainer.push_to_hub("sunny199/my-bert-imdb2")     # writes a model card + config
```

> **Beyond the video:** `trainer.push_to_hub` writes the model but the notebook
> pushes the tokenizer **first** as a separate call. Modern HF auto-creates the
> repo with `create_repo(repo_id, exist_ok=True)`. Two production must-dos the
> demo omits:
> * Set `model.config.id2label`/`label2id` **before** pushing, or every downstream
>   consumer inherits `LABEL_0`/`LABEL_1` (see the Correction in §6.1).
> * Write a model card with the dataset, the label map, the metric on a held-out
>   split, and the `transformers` version — otherwise the upload is a binary blob
>   nobody can safely load. `trainer.push_to_hub(..., tags=[...])` starts it for you.

---

## 7. Hyperparameters & Configuration — Every Knob

| Param | What it does | Typical | Safe range | Too high → | Too low → | Framework flag |
|---|---|---|---|---|---|---|
| `learning_rate` | AdamW step size | **2e-5** | 1e-5 – 5e-5 | Catastrophic forgetting: train loss ↓, eval loss ↑ | No learning after 3 epochs; loss stuck near `ln(C)` | `TrainingArguments(learning_rate=)` |
| `num_train_epochs` | Full passes over the data | **2–4** | 1–5 | Overfits small datasets by epoch 3 | Underfits; head never converges | `num_train_epochs=` |
| `per_device_train_batch_size` | Rows per forward pass | **16–32** | 8–64 | OOM at 512 tokens above ~64 on 16 GB | Noisy gradients; needs a lower LR | `per_device_train_batch_size=` |
| `gradient_accumulation_steps` | Batches per optimizer step | 1 | 1–8 | Doubles wall-clock without changing anything else | Effective batch too small | `gradient_accumulation_steps=` |
| `warmup_ratio` | Linear LR ramp fraction | **0.1** | 0.06–0.1 | Wastes steps at a low LR | First steps at full LR can wreck the encoder | `warmup_ratio=` / `warmup_steps=` |
| `weight_decay` | AdamW decoupled decay | **0.01** | 0.0–0.1 | Underfits; weights shrink | Mild overfit on small data | `weight_decay=` |
| `max_grad_norm` | Global grad-norm clip | **1.0** | 0.5–1.0 | Clipping never fires; LR spikes get through | Over-clips; learning stalls | `max_grad_norm=` |
| `optim` | Optimizer | **`adamw_torch`** | `adamw_torch`, `adafactor` | — | — | `optim="adamw_torch"` |
| `lr_scheduler_type` | LR curve | **`linear`** | linear, cosine, cosine_with_restarts | Cosine needs ≥3 epochs to pay off | Constant LR overfits late | `lr_scheduler_type=` |
| `adam_epsilon` | Adam denominator floor | 1e-8 | 1e-8 – 1e-6 | — | Division blow-up with fp16 | `adam_epsilon=` |
| `fp16` | AMP half precision | **True on T4/V100** | True/False | Overflow → NaN loss without loss scaling | 2× slower, more VRAM | `fp16=True` |
| `bf16` | bfloat16 AMP | **True on A100/H100** | True/False | — | — | `bf16=True` (Ampere+ only) |
| `max_length` (tokenizer) | Truncation ceiling | **256 for classification, 384 for QA, 128 for NER** | task-dependent | Attention memory is O(S²); 512 is the hard cap | Truncates answers/labels away | `tokenizer(..., max_length=)` |
| `padding` | Static vs dynamic | **dynamic via collator** | — | `max_length` padding wastes 2–20× compute | — | `DataCollatorWithPadding` |
| `problem_type` | Which loss the model builds | `single_label_classification` | +`multi_label_classification`, `regression` | Multi-label with the default = softmax over exclusive classes | — | `model.config.problem_type` |
| `num_labels` | Output head width | = `len(label_names)` | exact | Dead output units (silently) | IndexError or clamped labels | `from_pretrained(..., num_labels=)` |
| `label_smoothing_factor` | Softens targets | 0.0 | 0.0–0.1 | Underfits at 0.2+ | Overconfident logits | `label_smoothing_factor=` |
| `eval_steps` / `eval_strategy` | When to evaluate | `epoch` | `epoch` or 50–200 steps | Eval dominates wall-clock on small GPU | You never see overfitting | `eval_strategy="epoch"` |
| `save_total_limit` | Checkpoints kept | 1–2 | 1–3 | Fills the disk | You lose the best checkpoint | `save_total_limit=` |
| `metric_for_best_model` | Which metric selects the best ckpt | `f1` for classification, `eval_loss` for QA | — | Selecting on `loss` when you care about F1 | — | `metric_for_best_model="f1"` |
| `load_best_model_at_end` | Restore the best checkpoint | True | True | — | You serve the last (overfit) epoch | `load_best_model_at_end=True` |
| `seed` | RNG seed | 42 | fixed | — | Non-reproducible runs | `seed=42` |
| `dataloader_num_workers` | Parallel data loading | 2–4 on GPU | 0–8 | RAM blow-up | GPU starves on tokenization | `dataloader_num_workers=` |
| `gradient_checkpointing` | Recompute activations | False for base | True for `max_length=512`, batch ≥32 | 20–30% slower | OOM | `gradient_checkpointing=True` |
| `freeze` layers | `requires_grad=False` | none | — | Freeze too much → underfits | — | manual loop on `.parameters()` |

### 7.1 The interaction effects that actually bite

**LR × batch size.** Effective batch = `per_device_train_batch_size × num_gpus × gradient_accumulation_steps`. Gradient noise scales as `1/sqrt(effective_batch)`. If you move from batch 8 to batch 64 you may raise LR by ~1.5× (not 8× — the linear scaling rule is a *large*-batch pretraining tool, not a fine-tuning one). The video uses batch 8 with LR 2e-5 on a single device [40:28–40:36]; that is a conservative, well-behaved combination and a good default when you cannot afford a sweep.

**Epochs × dataset size.** The number that matters is total optimizer steps: `steps = (N_train / effective_batch) × epochs`. 1,000 IMDb rows at batch 8 × 1 epoch = **125 steps**. That is *far* too few — BERT's head needs on the order of 500–2,000 steps to converge. This is why the demo model's predictions are unreliable and why `num_train_epochs=1` is a teaching choice, not a recipe. With 25,000 rows and batch 16 × 3 epochs you get **4,688 steps** — the right order of magnitude. **Rule of thumb: aim for 1,000–10,000 optimizer steps; below 500, expect underfitting regardless of LR.**

**Warmup × total steps.** `warmup_ratio` is a *ratio*, so short runs get short warmups automatically. With 125 total steps, `warmup_ratio=0.1` gives 12 warmup steps — reasonable. With 300 steps, 30 (the notebook's arithmetic). Do not set `warmup_steps=500` on a 125-step run; the LR will never reach its peak.

**Padding × throughput.** Covered in §4.4: dynamic padding is a 2–20× win and costs one argument.

**fp16 × loss scaling.** `fp16=True` on a T4 with `Trainer` uses `torch.cuda.amp` with dynamic loss scaling, which is safe. `fp16=True` in a hand-written loop without a `GradScaler` is not — you will get NaN gradients on the LayerNorm and softmax paths. The notebook's manual loop does **not** use fp16, so it is safe but slow.

**`max_length` × task.** This is the most under-thought knob:

| Task | Correct `max_length` | Why |
|---|---|---|
| Sentence/paragraph classification | 128–256 | IMDB mean ≈ 300 subwords; 256 retains the sentiment-bearing content for ~95% of reviews |
| NER | 128, or dynamic padding only | WikiAnn/CoNLL sentences are <50 tokens; padding to 512 is 20× waste |
| Extractive QA | 384 with `doc_stride=128` | 512 minus question and specials; the standard SQuAD recipe is 384/128 |
| Sentence-pair / NLI | 128–256 | Premise + hypothesis |
| Reranking (query, doc) | 512 with sliding windows | Cross-encoder rerankers score whole (query, passage) pairs |

---

## 8. Decision Framework — When To Use / When NOT To Use

| Situation | Use BERT fine-tune? | Instead use | Why |
|---|---|---|---|
| Closed label set, ≤ 512 tokens, > 10k predictions/day | **Yes** | — | Cost and latency are 100–3,000× better than an LLM API |
| 5–50 labels, 1k–1M labelled rows | **Yes** | — | The classic encoder sweet spot |
| Token-level extraction with exact boundaries (NER, PII, dates) | **Yes** | SpanBERT / DeBERTa-v3 / a CRF head | Encoder + token head is the state of the art for closed entity types |
| Extractive QA over a fixed document set | **Yes** | DeBERTa-v3-large (SQuAD 2.0 ≈ 90 F1) | Encoders still beat LLMs on extractive SQuAD |
| Fewer than ~100 labels total | **No** | **SetFit** (§Beyond-the-video), or prompt an LLM | BERT needs ~50–200 examples/class to beat a prompted LLM |
| Open-ended generation, summaries, chat | **No** | LLM SFT (CS-13) or an encoder–decoder (T5/BART) | BERT has no causal mask and cannot generate |
| Input longer than 512 tokens | **No** | Longformer / BigBird / ModernBERT (8k) / chunk + aggregate | Learned absolute positions cannot extrapolate |
| Zero-shot, no labels at all | **No** | Prompting, or an NLI zero-shot classifier | No fine-tuning signal exists |
| Multilingual, 100+ languages, thin per-language data | **Partly** | XLM-R or mDeBERTa-v3 | mBERT works but XLM-R-large is materially better |
| Domain with heavy jargon (clinical, legal, chemical) | **Partly** | PubMedBERT / Legal-BERT / SciBERT, or domain-adaptive pretraining (CS-12) | 30k English WordPiece fragments your domain vocabulary into `[UNK]`s |
| Reasoning, multi-step, tool use | **No** | A decoder LLM | Encoders have no scratchpad |
| Need p99 < 20 ms on CPU | **Yes, quantized** | DistilBERT/MiniLM + ONNX int8 | 6-layer distilled encoders are 3–6× faster at 97% of quality |
| No data egress allowed, air-gapped | **Yes** | — | 110M params fits on any machine; no API call |

**STOP conditions — signals you have picked the wrong tool:**

1. **Your labels are free-form text.** If the "label" is a sentence someone typed, it is a generation task. BERT needs a closed set.
2. **Your label set changes weekly.** Every change to `num_labels` requires retraining the head (the output layer's shape changes). If labels are unstable, an LLM with a prompt is more agile. (Mitigation: train on a superset of labels and map, or use SetFit, whose head is a scikit-learn model you can refit in seconds.)
3. **The decisive evidence is beyond 512 tokens.** If your documents are 4,000 tokens and the answer is in paragraph 7, truncation destroys the task. Chunk + aggregate or use a long-context encoder.
4. **You need to *say* something.** Summarisation, explanation, rewriting — all generation. Use CS-13.
5. **You have fewer than ~50 examples per class and no budget to label.** SetFit or an LLM will beat a BERT fine-tune at that scale.
6. **Your metric is already > 0.97 with a regex or a gradient-boosted tree on TF-IDF.** Fine-tuning a Transformer for a problem a 200-line baseline solves is a maintenance liability, not an improvement.

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros

| # | Pro | Evidence / number |
|---|---|---|
| 1 | Tiny by 2026 standards — 110M params, 440 MB fp32, 110 MB int8 | Fits on a Raspberry Pi 5 |
| 2 | 10–30 ms inference on a GPU, 15–60 ms int8 on CPU | vs 400–2,000 ms for an LLM API |
| 3 | Inference cost is decoupled from input semantics — a prediction is a fixed 28 GFLOP | No prompt-length explosion |
| 4 | Fully open, self-hostable, air-gappable | No vendor dependency |
| 5 | Deterministic and versionable — pin a checkpoint hash forever | Audit and compliance friendly |
| 6 | Trains in minutes to hours on one GPU | 25k IMDb, fp16, dynamic padding ≈ 12 min/epoch on a T4 |
| 7 | Excellent for token-level tasks where LLMs struggle with boundary precision | BERT-base NER ≈ 0.92 F1 on CoNLL-2003 |
| 8 | Mature ecosystem: ONNX, quantization (CS-10), distillation (CS-08), Triton, TEI | Every serving path is a solved problem |
| 9 | Embeddings are reusable — same encoder drives retrieval and classification | One asset, two products |
| 10 | Interpretable-ish: attention maps, token attribution, per-token logits | Required for regulated domains |

### 9.2 Cons

| # | Con | Number |
|---|---|---|
| 1 | Cannot generate anything | Architectural, not a tuning issue |
| 2 | Hard 512-token limit | Learned absolute positions; quadratic attention |
| 3 | Weaker than modern encoders (DeBERTa-v3, ModernBERT) on every benchmark | DeBERTa-v3-large ≈ +3–5 points on GLUE-scale tasks |
| 4 | Loses to few-shot LLMs when labels < ~50/class | SetFit reaches GPT-3 few-shot quality with 8 examples/class |
| 5 | Catastrophically LR-sensitive | 2e-5 good, 2e-3 destroys the model in 100 steps |
| 6 | WordPiece vocabulary is a poor fit for jargon/chemical/multilingual text | Measure the `[UNK]` rate first |
| 7 | No native uncertainty — softmax scores are miscalibrated | Typical ECE 0.10–0.25 on small fine-tunes |
| 8 | English-centric default checkpoint | mBERT/XLM-R needed for cross-lingual |
| 9 | One model per task; no multi-task reuse without a new head and a new run | 3 tasks = 3 checkpoints to serve and monitor |
| 10 | Requires labelled data — the labelling is usually the real cost | 5,000 NER sentences ≈ 40–80 annotator-hours |

### 9.3 Hard limitations (not fixable by tuning)

1. **No causal mask → no autoregressive generation.** You cannot make BERT write text. Full stop.
2. **Learned absolute position embeddings, `max_position_embeddings=512`.** Positions 512+ have no embedding. Feeding them raises `IndexError`; you cannot "just increase it" without retraining.
3. **Quadratic attention.** Doubling the sequence length quadruples attention compute and memory. 512 is a soft economic ceiling as well as a hard one.
4. **No cross-attention.** BERT's encoder cannot attend to a second sequence except by concatenation with `[SEP]` and segment embeddings.
5. **Masked-LM pretraining is a fixed objective.** You cannot cheaply continue pretraining on a corpus whose distribution has shifted without risking catastrophic forgetting.

### 9.4 Silent failure modes (looks fine, is broken)

| # | Failure | How it looks | Detection |
|---|---|---|---|
| 1 | **Subword labels not masked** — every subword gets the word's label | Loss decreases, NER F1 5–15 points low; boundaries look "smeared" | Assert `(labels == -100).sum() > 0` and that `-100` count ≈ specials + continuations |
| 2 | **Collator pads labels with 0, not -100** | Loss decreases but is contaminated by pads; F1 drops a little | Check `labels.min() == -100` in a batch; if 0, wrong collator |
| 3 | **`num_labels` ≠ real label count** | No crash; dead output units; slightly low F1 | `assert model.config.num_labels == len(label_names)` |
| 4 | **`problem_type` default on multi-label data** | Predictions are mutually exclusive; recall on co-occurring labels ≈ 0 | Check the label matrix has >1 positive per row |
| 5 | **QA: answer truncated away, labels default to (0,0)** | Model answers `[CLS]`/empty string confidently | Count rows where `start_positions == 0`; if >2%, you have a truncation leak |
| 6 | **Tokenizer mismatch between train and load** | Predictions are garbage but shapes match | Store `tokenizer.save_pretrained` next to the weights; assert vocab sizes match |
| 7 | **`id2label` left as `LABEL_0`/`LABEL_1`** | Metrics correct; production routing inverted | `print(model.config.id2label)` before serving |
| 8 | **Evaluation on the training split** | 0.98 accuracy | Compare `trainer.evaluate(eval_dataset=train_dataset)`; if it equals your test number, you are testing on train |
| 9 | **Duplicate rows across train/test** | Metrics 5–20 points optimistic | Deduplicate on a normalised text hash |
| 10 | **`model.eval()` not called at inference** | Non-deterministic predictions; dropout noise | `assert not model.training` |
| 11 | **No `torch.no_grad()` at inference** | VRAM grows linearly per batch until OOM | Watch `torch.cuda.memory_allocated()` across 100 batches |
| 12 | **Token-level accuracy reported for NER** | 0.87 accuracy, 0.00 entity F1 | Always run `seqeval` alongside |

---

## 10. Exceptions, Edge Cases & Gotchas

1. **The exception to "use `-100` on continuation subwords".** When your tag scheme is BILOU and you set `label_all_tokens=True`, continuation subwords should receive `I-X` (converted from `B-X`), not `-100`. The rule is: *one label per word* → mask continuations; *one label per token* → propagate. Mixing the two between training and inference is the classic inconsistent-alignment bug.
2. **The exception to "`[CLS]` is the pooled representation".** For **retrieval/similarity**, `[CLS]` is measurably worse than mean pooling over all non-pad tokens. Use `sentence-transformers` (which uses mean pooling) rather than `BertModel.last_hidden_state[:,0]`. Also note that `AutoModelForSequenceClassification` uses the *pooler* (`tanh(W·h_[CLS])`), not the raw `[CLS]` — the pooler is a separate `Linear(768,768)` you can drop with `add_pooling_layer=False`.
3. **The exception to "truncate at 512".** For QA, truncation must be `truncation="only_second"` — truncating the *question* leaves an unanswerable prompt. The default `truncation=True` truncates the longest sequence, which for `(question, context)` pairs is nearly always the context. It happens to work, but be explicit, because if the question is longer than the context the default will silently eat it.
4. **The exception to "`[UNK]` is rare".** `bert-base-uncased` lowercases everything. For NER, **casing is a feature**: `Apple` (company) vs `apple` (fruit), `Bush` (person) vs `bush` (plant). Use `bert-base-cased` for NER and for any task where capitalisation carries signal. The video uses `bert-base-uncased` for all three tasks including NER [35:08] — that is a defensible demo choice and a suboptimal production one.
5. **The exception to "small LR".** If you freeze all but the top 2 layers, you may raise the LR for the unfrozen layers to 1e-4–3e-4. The 2e-5 rule applies to *full* fine-tuning where every weight in the pretrained representation is being nudged.
6. **The exception to "more epochs is better".** Encoder fine-tuning on <10k rows typically peaks at epoch 2 and degrades by epoch 4. Watch `eval_loss`, not `train_loss`, and use `load_best_model_at_end=True` with `eval_strategy="epoch"`.
7. **The exception to "class imbalance doesn't matter for BERT".** It does. With 1% positives, plain cross-entropy collapses to the majority class within one epoch and *looks* converged. Options: `pos_weight` in `BCEWithLogitsLoss` (via `problem_type="multi_label_classification"` with a single label, a common hack), a `WeightedRandomSampler`, or class-weighted CE via a custom `Trainer.compute_loss`.
8. **The exception to "eval_loss is comparable across runs".** It is not, if `max_length` differs — padding tokens contribute to QA loss if you forget `-100`, and the mean over a different token count is a different quantity. Fix the padding policy before comparing runs.
9. **The exception to "the fast tokenizer is always better".** `BertTokenizerFast` is required for `word_ids()` and `offset_mapping`, but it can produce *different* tokenization from the slow tokenizer on edge cases (rare Unicode, some CJK). If you trained with the slow tokenizer and serve with the fast one, verify identical token ids on a 1,000-row sample.
10. **The exception to "batch size 8 always fits".** At `max_length=512` with `model.train()`, activations are stored for the backward pass. Batch 8 × 512 fp32 on BERT-base ≈ 4 GB total; batch 32 ≈ 10 GB. Add `gradient_checkpointing=True` to trade ~25% wall-clock for a ~50% activation reduction.
11. **The multi-task gotcha.** The notebook trains three separate models in one session without releasing the previous one. On a 16 GB T4, running `run_text_classification_demo()` then `run_ner_demo()` then `run_qa_demo()` accumulates GPU memory across models unless you `del model; torch.cuda.empty_cache()`. This is the most likely cause of an unexpected OOM when you press `4`.
12. **The `[PAD]` embedding row learns something.** Because pads are masked they produce no MLM/CE loss, but they still pass through LayerNorm and can accumulate. `weight_decay=0.01` keeps them small. Do not be alarmed to see non-zero pad weights.
13. **`num_warmup_steps = total_steps // 10` vs `warmup_ratio = 0.1`.** These are the same thing *only if* the Trainer computes `total_steps` identically. The hand-written loop computes `len(train_loader) * epochs`; the `Trainer` accounts for `gradient_accumulation_steps`. With accumulation > 1 they differ, and the hand-written version over-warms.
14. **`token_type_ids` for single-sentence inputs.** `bert-base-uncased` expects `token_type_ids` (all zeros). If you build tensors manually and omit them, the model defaults them to zeros, so it works — but for *pair* tasks (QA), omitting `token_type_ids` means the model cannot tell question from context, and BERT's QA performance drops several points. Always pass the full tokenizer output.
15. **The `report_to="none"` trap.** The instructor sets this to disable Weights & Biases [40:54–41:10]. It also disables the TensorBoard writer he then tries to read with `!tensorboard --logdir=./logs` [45:00]. If `./logs` never populates, this is why: use `report_to="tensorboard"` (or omit `report_to` and let it default) when you actually want the curves.
16. **The Colab TensorBoard gotcha.** The instructor hits this live at [45:44–46:07]: TensorBoard running in Colab is served on the *remote* VM, so `localhost` on your machine shows nothing. Use the Colab `%tensorboard` magic or the "Open TensorBoard" port-forward link, not a local browser.

---

## 11. Cost, Compute & Memory

### 11.1 Training VRAM, formula and worked examples

```
VRAM_fp32 ≈ 16·N_params                     (weights 4 + grads 4 + AdamW m,v 8)
           + (activation coefficient) × B × S × L × H × 4
           + 0.5 GB framework overhead
activation coefficient ≈ 34 for post-LN BERT (attention + FFN buffers)
```

| Model | Params | B=8, S=128 | B=8, S=256 | B=8, S=512 | B=32, S=512 |
|---|---|---|---|---|---|
| DistilBERT | 66M | 1.4 GB | 1.7 GB | 2.6 GB | 6.0 GB |
| **BERT-base** | **110M** | **2.2 GB** | **2.9 GB** | **4.1 GB** | **10.1 GB** |
| BERT-large | 340M | 6.3 GB | 7.6 GB | 9.9 GB | 22.5 GB |
| RoBERTa-base | 125M | 2.4 GB | 3.2 GB | 4.4 GB | 11.0 GB |
| DeBERTa-v3-base | 86M | 1.9 GB | 2.5 GB | 3.6 GB | 8.5 GB |
| mDeBERTa-v3-base | 278M | 5.4 GB | 6.5 GB | 8.7 GB | 20 GB |
| XLM-R-base | 270M | 5.3 GB | 6.4 GB | 8.5 GB | 19.5 GB |

fp16/bf16 halves the activation term and keeps the master weights in fp32: **BERT-base at B=8/S=512 drops to ≈ 3.0 GB; at B=32/S=512 to ≈ 6.6 GB.** That is how `per_device_train_batch_size=32` fits on a 16 GB T4.

Freezing layers changes only the optimizer-memory term:

| Configuration | Trainable | Optimizer state (fp32) | Saving |
|---|---|---|---|
| Full BERT-base | 110M | 880 MB | — |
| Freeze 6 bottom layers | 56M | 448 MB | 432 MB |
| Freeze 9 bottom + embeddings | ~21M | 168 MB | 712 MB |
| Head only (linear probe) | 1.5k | 12 kB | 880 MB |

### 11.2 One worked end-to-end job

**"Fine-tune BERT-base for 3-class ticket intent on 12,000 labelled tickets, `max_length=128`, serve on CPU at 200 req/s."**

```
DATA
  12,000 tickets, mean 90 tokens, 3 classes (58% / 27% / 15%)
  split 10,000 / 1,000 / 1,000 (stratified)

TRAINING
  BERT-base, num_labels=3, max_length=128, dynamic padding (batch mean ~95 tokens)
  epochs=3, batch=32, LR=3e-5, warmup_ratio=0.1, wd=0.01, fp16=True
  steps = (10,000/32) * 3 = 938 optimizer steps      -> in the 1,000-step sweet spot
  GPU   = 1 x T4 (16 GB) or 1 x A10G
  time  = ~4 min (A10G, bf16) / ~11 min (T4, fp16)
  cost  = 11 min x $0.35/hr (Colab Pro T4) = $0.06   [estimate]

STORAGE
  fp32 safetensors   = 440 MB
  int8 ONNX (QDQ)    = 110 MB

SERVING (CPU, 8 vCPU c7i.large, $0.36/hr on-demand, ~$0.11/hr spot)
  throughput: BERT-base int8 ONNX, S=128, batch 16 = ~450 predictions/s/core-equivalent
              measured 8-vCPU box = ~1,800 preds/s sustained
  200 req/s  -> 1.1% of one box. One box serves ~1,800 req/s.
  cost       = $0.36/hr / (1,800 * 3600) = $5.6e-8 per prediction
             = $0.056 per 1M predictions  (on-demand, 8 vCPU)
             = $0.017 per 1M predictions  (spot)

ONE-TIME
  labelling 12,000 tickets @ 15 s each = 50 annotator-hours
  @ $25/hr                           = $1,250  <- the real cost of the project
  training compute                   = $0.06
```

The labelling is **20,000×** the training cost. Write that on the whiteboard in your next interview.

### 11.3 Fine-tuned BERT vs prompting an LLM — the cost arithmetic

**Setup.** Classify 1,000,000 support tickets. Mean ticket 200 tokens, output label ≈ 10 tokens (JSON `{"label":"billing"}`).

**Path A — self-hosted fine-tuned BERT-base (int8 ONNX, GPU).**

```
A10G on g5.xlarge: $1.006/hr on-demand
throughput: BERT-base fp16, S=256, batch 32, ONNX Runtime + CUDA  = ~3,000 preds/s
1,000,000 / 3,000 = 333 seconds = 0.0926 hr
cost = 0.0926 * $1.006 = $0.093 per 1M predictions
```

**Path B — self-hosted BERT on CPU (int8 ONNX).**

```
c7i.2xlarge (8 vCPU): $0.357/hr on-demand, $0.107/hr spot
throughput at S=256, batch 16, int8 = ~700 preds/s sustained
1,000,000 / 700 = 1,429 s = 0.397 hr
cost = 0.397 * $0.357 = $0.142 per 1M on-demand; $0.042 spot
```

**Path C — hosted LLM API.**

| Model | Input $/1M tok | Output $/1M tok | Cost / call (200 in + 10 out) | **Cost / 1M calls** | **× BERT** |
|---|---|---|---|---|---|
| GPT-4o-mini | $0.15 | $0.60 | $0.000036 | **$36** | 387× |
| Gemini 2.0 Flash | $0.10 | $0.40 | $0.000024 | **$24** | 258× |
| Claude Haiku (3.5) | $0.80 | $4.00 | $0.000200 | **$200** | 2,150× |
| GPT-4o | $2.50 | $10.00 | $0.000600 | **$600** | 6,452× |
| Claude Sonnet (4) | $3.00 | $15.00 | $0.000750 | **$750** | 8,065× |
| Batch API (50% off) | — | — | — | **half of the above** | — |

Arithmetic shown explicitly for GPT-4o: `200 × 2.50/1e6 = $0.0005` input, `10 × 10.00/1e6 = $0.0001` output → `$0.0006/call × 1e6 = $600`.

**Path D — hosted LLM *fine-tuned* (OpenAI).**

Training 12,000 examples × ~250 tokens = 3M tokens × 3 epochs = 9M training tokens at $8/1M (4o-mini tier) = **$72 one-time**, and inference still costs $36/1M. Fine-tuning a hosted LLM improves *quality*; it does not fix the *unit economics*.

**Total cost of ownership over 3 years at 15M predictions/month (540M total):**

| Path | One-time | Inference (540M) | 3-year total | p50 latency | Data egress |
|---|---|---|---|---|---|
| BERT GPU int8 | $1,250 labelling + $2 training | $50 | **$1,302** | 25 ms | none |
| BERT CPU int8, spot | $1,252 | $23 | **$1,275** | 45 ms | none |
| GPT-4o-mini | $0 | $19,440 | **$19,440** | 700 ms | full text |
| GPT-4o | $0 | $324,000 | **$324,000** | 1,200 ms | full text |

> **Beyond the video — the honest counter-argument.** The table is one-sided on
> purpose, but a competent interviewer will push back with five real costs BERT
> *does* carry:
> 1. **Labelling.** $1,250 assumes you already have labels. If not, the LLM path is
>    zero-labelling-cost, and 540M predictions at $19.4k is cheaper than 3 years of
>    an ML engineer maintaining a pipeline.
> 2. **Engineering.** Serving, monitoring, retraining, drift detection, GPU
>    capacity planning, and on-call are real. An API call is one line.
> 3. **Quality gap.** On hard, long-tail, or reasoning-flavoured classification, a
>    frontier LLM with a good prompt can beat a BERT trained on 12k examples. Run
>    the eval; do not assume.
> 4. **Label set volatility.** Each label change is a retrain for BERT and a prompt
>    edit for an LLM.
> 5. **Bootstrap.** The correct production pattern is frequently **LLM to label,
>    BERT to serve**: use a strong LLM with a careful prompt to label 50k
>    unlabelled tickets, distil into BERT, and keep the LLM as a fallback for
>    low-confidence cases. This gets you the LLM's quality at the encoder's unit
>    economics — and it is exactly the "hybrid pipeline" the instructor names at
>    [26:57–27:16].
>
> **The threshold rule that falls out:** below ~1M predictions/month, prompt an
> LLM. Above ~10M/month with a stable label set, fine-tune an encoder. Between
> the two, do the arithmetic and include your engineering hours at a loaded rate.

### 11.4 Cloud GPU price reference (list, USD/hr, 2026 order of magnitude)

| GPU | VRAM | On-demand | Spot | BERT-base full FT, S=256, 12k rows, 3 ep |
|---|---|---|---|---|
| Colab T4 (free) | 16 GB | $0 | $0 | ~15 min (fp16) |
| T4 (g4dn.xlarge) | 16 GB | $0.53 | $0.16 | ~15 min → $0.13 |
| L4 (g6.xlarge) | 24 GB | $0.81 | $0.24 | ~8 min → $0.11 |
| A10G (g5.xlarge) | 24 GB | $1.01 | $0.30 | ~5 min → $0.08 |
| A100 40GB (p4d) | 40 GB | $4.10 | $1.20 | ~3 min → $0.21 |
| H100 (p5) | 80 GB | $12.29 | $3.90 | ~2 min → $0.41 |

BERT does not need an H100. This is the point: the H100-hours are for the LLM track (CS-13+), not for encoders.

---

## 12. Evaluation — How To Know It Worked

### 12.1 Metric selection, with the imbalance trap

| Metric | Formula | Use when | Fails when |
|---|---|---|---|
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | Classes roughly balanced (<3:1) | Any imbalance — 99% accuracy for a model that always says "no fraud" |
| **Precision** | `TP/(TP+FP)` | False positives are expensive (auto-approval, spam blocking) | You never check recall |
| **Recall** | `TP/(TP+FN)` | False negatives are expensive (safety, fraud, disease) | You never check precision |
| **F1 (binary)** | `2PR/(P+R)` on the positive class | Imbalanced binary; you need one number | Ignores TN entirely — a model that predicts everything positive gets F1 ≈ 0.67 on a 50/50 set |
| **F1-macro** | unweighted mean of per-class F1 | All classes matter equally, including rare ones | Rare classes dominate the number; unstable with <30 support |
| **F1-micro** | = accuracy for single-label multiclass | Never, for single-label | It *is* accuracy; it hides the minority class |
| **F1-weighted** | mean of per-class F1 weighted by support | Reporting to a stakeholder who wants "overall" | Hides rare-class collapse |
| **MCC** | `(TP·TN − FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Binary, imbalanced; the single most honest scalar | Multiclass version is harder to explain |
| **Cohen's κ** | `(p_o − p_e)/(1 − p_e)` | Comparing to a majority-class baseline | Low κ with high accuracy means your labels are near-random |
| **ROC-AUC** | area under TPR vs FPR | Balanced data; ranking quality | Insensitive to prevalence — a great ROC-AUC can hide terrible precision |
| **PR-AUC / AP** | area under precision vs recall | **Positives < 10%** | Not comparable across datasets with different prevalence |
| **ECE** | `Σ (n_b/N)·|acc_b − conf_b|` | You will act on confidence thresholds | Needs bucketing choices |

**The class-imbalance trap in this module.** The notebook computes `f1_score(..., average='weighted')` for the binary IMDb task [notebook cell 48]. On IMDb (~50/50 neg/pos) that is fine. Move to a real ticket dataset that is 58/27/15 or a fraud dataset at 0.5% positives and the same line produces a number that is 97% driven by the majority class. **The correct default:**
- Binary, positive class matters → `average='binary', pos_label=1`
- Binary, both classes matter → `average='macro'` + MCC
- Multiclass → `average='macro'` + per-class `classification_report` + a confusion matrix
- NER → **`seqeval` entity micro-F1** (never token accuracy)

### 12.2 The confusion matrix, and how to actually read it

```
                        PREDICTED
                   neg        pos
              +---------+----------+
   ACTUAL neg |   TN    |    FP    |   FP = false alarm  -> wasted auto-actions
              +---------+----------+
   ACTUAL pos |   FN    |    TP    |   FN = miss         -> the one that hurts
              +---------+----------+
```

Six readings that separate a senior engineer from a junior one:

1. **Look at the off-diagonal cells first, not the diagonal.** The diagonal is a comfort number.
2. **Asymmetry is a decision.** If `FP >> FN`, you are over-predicting the positive class — lower the threshold if FPs are cheap, raise it if not. BERT gives you `softmax` scores precisely so you can move the threshold; `argmax` at 0.5 is a *choice*, not a requirement.
3. **A confusion matrix with one dominant row means your label distribution is broken**, not your model.
4. **Multi-token / multi-label: read it as a pairwise confusion matrix across classes** and look for the two classes that trade with each other. In intent classification, `billing` ↔ `refund` is the classic pair.
5. **For NER the confusion matrix is over label *types*, and it hides boundary errors entirely.** Print `seqeval`'s `classification_report`, which breaks out `B-PER`, `I-PER`, etc., and read the `B-`/`I-` rows for boundary confusion.
6. **Small test sets make the matrix noise.** A 500-row test set with ±1% per cell means a 2-point accuracy change is not evidence. Compute a Wilson interval; at n=500 and p=0.85 the 95% CI is roughly ±3.1 points. The video's demo uses exactly 500 test rows [32:22].

### 12.3 Calibration

A fine-tuned BERT trained on 1,000 examples is badly overconfident: softmax 0.95 typically corresponds to ~70% empirical accuracy. Two measurements and one fix:

```python
import numpy as np, torch

def expected_calibration_error(probs, labels, n_bins=10):
    """probs: (N,) confidence in the predicted class. labels: (N,) correctness 0/1."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return ece

# 1. get probs + correctness on a held-out set
model.eval()
all_p, all_y, all_pred = [], [], []
with torch.no_grad():
    for batch in eval_loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        all_p.append(p); all_y.append(batch["labels"].numpy())
probs = np.concatenate(all_p); y = np.concatenate(all_y)
conf = probs.max(axis=1); pred = probs.argmax(axis=1)
correct = (pred == y).astype(float)
print("ECE:", round(expected_calibration_error(conf, correct), 4))
# A 1-epoch/1000-sample IMDb model typically prints ECE 0.10-0.25.

# 2. temperature scaling -- the standard fix, fit on a VALIDATION set only
import torch.nn as nn, torch.optim as optim
logits_val = torch.tensor(logits_val)          # (N, C) raw logits, NOT softmax
T = nn.Parameter(torch.ones(1))
opt = optim.LBFGS([T], lr=0.01, max_iter=50)
ce = nn.CrossEntropyLoss()
def closure():
    opt.zero_grad()
    loss = ce(logits_val / T.clamp(min=1e-3), torch.tensor(y_val))
    loss.backward(); return loss
opt.step(closure)
print("fitted temperature:", T.item())   # typically 1.5 - 3.0 for an overconfident encoder
```

### 12.4 A minimal, complete eval script (copy-paste)

```python
import numpy as np, torch, evaluate
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, f1_score, accuracy_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./bert-finetuned-imdb"        # the directory you saved BOTH model + tokenizer into
tok = AutoTokenizer.from_pretrained(MODEL_DIR)          # fast by default -> AutoTokenizer
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval().to(device)

# Guard rail #1: never evaluate a model whose label map you have not printed.
print("id2label:", mdl.config.id2label, "| num_labels:", mdl.config.num_labels)
assert mdl.config.num_labels == len(mdl.config.id2label)

def predict(texts, batch_size=32, max_length=256):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i+batch_size], truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out)

probs  = predict(test_texts)
preds  = probs.argmax(axis=1)
y      = np.array(test_labels)

print(classification_report(y, preds, digits=4))
print("confusion matrix:\n", confusion_matrix(y, preds))
print("accuracy :", round(accuracy_score(y, preds), 4))
print("F1 binary:", round(f1_score(y, preds, average="binary", pos_label=1), 4))
print("F1 macro :", round(f1_score(y, preds, average="macro"), 4))
print("MCC      :", round(matthews_corrcoef(y, preds), 4))
# Guard rail #2: the trivial baseline. If you cannot beat it, you have no model.
print("majority-class baseline:", round(max(np.bincount(y)) / len(y), 4))
# Guard rail #3: a Wilson 95% CI half-width for the accuracy, so you know the noise floor.
n, p_hat = len(y), accuracy_score(y, preds)
hw = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"accuracy 95% CI: +/- {hw:.4f}")
```

### 12.5 How each metric lies to you

| Metric | The lie | The tell |
|---|---|---|
| Token accuracy (NER) | "0.87, we're nearly done" — a model predicting all-`O` gets 0.85 | Compute `seqeval` F1; if it is <0.05 while accuracy is >0.8, this is your bug |
| Accuracy on imbalanced data | "0.97!" — the majority-class baseline is 0.97 | Print the baseline next to it |
| F1-weighted | Hides a rare class at 0.0 behind a common class at 0.95 | Print the per-class report |
| `eval_loss` alone | Loss can fall while F1 falls if the decision threshold is off | Always report a task metric, not just loss |
| Exact match (QA) | 0.0 on a model that is 5 characters off | Report token-level F1 alongside EM; a gap >0.25 means systematic boundary error |
| A single test split | ±3–4 points of noise at n=500 | Report the CI; use 3 seeds and report mean ± std |
| LLM-as-judge on your own outputs | Correlated with your model's style | Use a different model family as judge, and human-audit 100 rows |
| Public leaderboard scores | Trained on the dev set | Always hold out a private split |

---

## 13. Comparison Tables

### 13.1 Encoder-only vs decoder-only vs encoder–decoder

| Axis | Encoder-only (BERT) | Decoder-only (GPT / LLaMA) | Encoder–decoder (T5 / BART) |
|---|---|---|---|
| Attention mask | Fully bidirectional (no causal mask) | Causal / lower-triangular | Bidirectional encoder + causal decoder + cross-attention |
| Pretraining objective | MLM (+NSP) / RTD | Next-token prediction (causal LM) | Span corruption (T5) / denoising (BART) |
| Can it generate text? | **No** — no causal ordering to sample from | **Yes** — the native mode | **Yes** — the decoder generates |
| Best at | Classification, tagging, extraction, retrieval, reranking | Generation, reasoning, chat, tool use, code | Transformation with a fixed input: translation, summarisation, seq2seq |
| Typical params | 66M–400M | 1B–1T | 60M–11B |
| Inference cost / 1M calls | **$0.04–0.20** | $24–$750 (API) or GPU-hours for self-host | $5–$50 self-hosted |
| Latency p50 | 10–30 ms GPU, 15–60 ms CPU int8 | 300–2,000 ms (API, short output) | 100–800 ms self-hosted |
| Fine-tune cost | Minutes on one GPU | GPU-days; needs LoRA/QLoRA (CS-23) | Hours to days |
| Data needed to beat prompting | ~50–200 labelled examples/class | 0 (prompt) to 1k–100k (SFT) | ~1k pairs |
| 2026 relevance | Still the correct engineering choice for closed-label NLP at volume | Default for anything generative | Best for translation/summarisation at scale |

**Why BERT cannot generate — the precise mechanism.** Decoder-only models are trained with a causal mask `M[i,j] = -inf for j > i`. That mask makes `P(x_t | x_<t)` a well-defined distribution the model can sample from one token at a time, and it makes the training objective (predict the next token) identical to the inference procedure (generate the next token). BERT's mask has **no causal component**: token *i* attends to token *j* for all *j*, including *j > i*. So there is no "next" — every position's representation is conditioned on the entire final sequence, including the tokens you would be trying to generate. If you tried to sample autoregressively from BERT you would have to re-run the full bidirectional pass after appending each token, and every appended token would change the representations of all the tokens before it — the output would not even be self-consistent. Bidirectionality and autoregressive generation are mutually exclusive by construction.

### 13.2 The BERT family tree

Sizes below are the standard published configurations. "Training data" is the pretraining corpus, not the fine-tuning data.

| Model | Params | Layers | Hidden | Heads | Max len | Vocab | Training data | Wins when |
|---|---|---|---|---|---|---|---|---|
| **BERT-base-uncased** | 110M | 12 | 768 | 12 | 512 | 30,522 (WordPiece) | 16 GB BooksCorpus + en-Wiki (3.3B words), MLM+NSP | The default baseline. Low-resource, fast iteration, uncased text |
| **BERT-base-cased** | 110M | 12 | 768 | 12 | 512 | 28,996 (cased) | same | **NER and any task where capitalisation carries signal** |
| **BERT-large-uncased** | 340M | 24 | 1024 | 16 | 512 | 30,522 | same, 1M steps | ~+1–2 GLUE points over base at 3× the cost |
| **DistilBERT-base** | 66M | 6 | 768 | 12 | 512 | 30,522 | distilled from BERT-base (triple loss: distillation + MLM + cosine) | 60% faster, 40% smaller, **97% of GLUE** — the right choice at high volume. → CS-08 |
| **RoBERTa-base** | 125M | 12 | 768 | 12 | 514 | 50,265 (byte-BPE) | 160 GB: BooksCorpus + CC-News + OpenWebText + Stories; **dynamic masking, NO NSP**, 8k batch, 500k steps | Drop-in, better-than-BERT default. The single cheapest upgrade |
| **RoBERTa-large** | 355M | 24 | 1024 | 16 | 514 | 50,265 | same | Strong classification/QA baseline; GLUE ≈ 88.5 |
| **ALBERT-base-v2** | 12M | 12 | 768 | 12 | 512 | 30,000 | BERT data; **cross-layer parameter sharing + factorized embeddings + SOP** | Extreme parameter efficiency; edge with a hard RAM cap |
| **ALBERT-xxlarge-v2** | 235M | 12 | 4096 | 64 | 512 | 30,000 | same | GLUE ≈ 89.4, but **~3× slower than BERT-large** at inference |
| **ELECTRA-small** | 14M | 12 | 256 | 4 | 512 | 30,522 | **Replaced-token detection** (generator 110M, discriminator 14M) | Tiny + accurate; the best quality-per-parameter small encoder |
| **ELECTRA-base** | 110M | 12 | 768 | 12 | 512 | 30,522 | RTD, 1/4 of BERT's compute | Matches RoBERTa-base with ~25% of the pretraining FLOPs |
| **DeBERTa-v2-base** | 140M | 12 | 768 | 12 | 512 | 50,265 | 160 GB; **disentangled attention** (content↔position) + EMD | Better QA/NLI than RoBERTa at equal size |
| **DeBERTa-v3-base** | 86M | 12 | 768 | 12 | 512 | 128,100 | 160 GB; **RTD-style pretraining + gDES** (gradient-disentangled embedding sharing) | **The current best small encoder.** Beats RoBERTa-large on several GLUE tasks at 1/4 the params |
| **DeBERTa-v3-large** | 304M | 24 | 1024 | 16 | 512 | 128,100 | same | First encoder past human on SuperGLUE-adjacent MNLI (91.4% ≈ human). SQuAD 2.0 ≈ 90 F1 |
| **DeBERTa-v3-xsmall** | 22M | 12 | 384 | 6 | 512 | 128,100 | same | ONNX-CPU serving with better accuracy than DistilBERT |
| **SpanBERT-base** | 110M | 12 | 768 | 12 | 512 | 30,522 | **Span masking + Span Boundary Objective** (mask contiguous spans, mean length 3.3) | Coreference, relation extraction, extractive QA — anything span-structured |
| **PubMedBERT-base** | 110M | 12 | 768 | 12 | 512 | 30,000 (domain) | **From scratch on 14M PubMed abstracts + full text (~21B tokens)** | Biomedical NER/QA. Beats BioBERT, which was BERT-initialised |
| **SciBERT** | 110M | 12 | 768 | 12 | 512 | 31,116 (SciVocab) | 1.14M scientific papers (18% CS, 82% biomedical) | Scientific paper classification, citation intent |
| **BioBERT** | 110M | 12 | 768 | 12 | 512 | 28,996 | BERT-base further pretrained on PubMed + PMC | Legacy biomedical baselines only — PubMedBERT beats it |
| **Legal-BERT** | 110M | 12 | 768 | 12 | 512 | 30,522 | 12 GB of legal text (EU legislation, case law, contracts) | Contract clause classification, legal NER |
| **FinBERT** | 110M | 12 | 768 | 12 | 512 | 30,522 | 4.9B tokens of financial text | Sentiment on earnings calls, financial NER |
| **mBERT (bert-base-multilingual-cased)** | 178M | 12 | 768 | 12 | 512 | 119,547 | Wikipedia in 104 languages, **no explicit cross-lingual objective** | Zero-shot cross-lingual transfer in a pinch; superseded by XLM-R |
| **XLM-R-base** | 270M | 12 | 768 | 12 | 512 | 250,002 (SentencePiece) | **2.5 TB CommonCrawl, 100 languages** | 100-language classification/NER with thin per-language data |
| **XLM-R-large** | 550M | 24 | 1024 | 16 | 512 | 250,002 | same | The multilingual quality leader for encoders |
| **mDeBERTa-v3-base** | 278M | 12 | 768 | 12 | 512 | 250,100 | 2.5 TB CC-100 + 160 GB English (RTD) | **Best multilingual small encoder**; beats XLM-R-base on XNLI |
| **MiniLM-L6-v2** | 22M | 6 | 384 | 12 | 512 | 30,522 | distilled (self-attention distribution + relation transfer) | Sentence embeddings for RAG retrieval; 5× faster than BERT-base |

**How to pick, in one decision list:**

```
Is the text English?
├─ Yes ─┬─ Need max quality and can afford 300M params?  -> DeBERTa-v3-large
│       ├─ Want a sane default?                          -> DeBERTa-v3-base  (or RoBERTa-base)
│       ├─ Need speed/size?                              -> DistilBERT, or ELECTRA-small
│       ├─ Hard RAM/latency cap (edge)?                  -> MiniLM-L6 or DistilBERT int8
│       └─ Span QA / coreference / relation extraction?   -> SpanBERT-base
└─ No ──┬─ Few languages, lots of data each?             -> XLM-R-large
        └─ Many languages, thin data?                    -> mDeBERTa-v3-base
Domain-specific (clinical/legal/financial/scientific)?
└─ Yes -> try the domain checkpoint FIRST (PubMedBERT, Legal-BERT, SciBERT, FinBERT),
          then consider domain-adaptive pretraining (CS-12) on a general encoder.
```

The instructor lists essentially this menu at [28:43–29:02] — *"board, distill board, robberta, diverta, mpnet, mini lm, flan, pexus, xllet, xlmr, electra"* — and his point is exact: *"these particular model guys you can use in place of BERT ... here I shown you the fine tuning of the BERT model but you can try out with a different other model as well."* The line of code that makes it true is that they are all `AutoModel*` compatible.

> **Correction:** the transcript's model list is garbled by automatic speech
> recognition. Reconstructed: **BERT, DistilBERT, RoBERTa, DeBERTa, MPNet, MiniLM,
> Flan-T5, Pegasus, XLNet, XLM-R, ELECTRA** [28:43–28:52]. Two of these are worth
> flagging as different families: **Flan-T5** and **Pegasus** are *encoder–decoder*
> models (T5-based and BART-based respectively), not encoders — they generate, and
> they cannot be dropped into `BertForSequenceClassification`. The notebook's own
> comparison table (cell 1) correctly places them under "Summarization,
> translation, instruction tasks" and "Abstractive summarization".

> **Beyond the video — SetFit: BERT when you have almost no labels.** The
> instructor's recipe needs thousands of examples. SetFit (Tunstall et al., 2022)
> gets competitive accuracy from **8 labelled examples per class**:
> ```python
> # pip install setfit
> from setfit import SetFitModel, Trainer, TrainingArguments
> from datasets import Dataset
>
> train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})  # 8/class!
> model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
> args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20)
> trainer = Trainer(model=model, args=args, train_dataset=train_ds)
> trainer.train()
> preds = model.predict(["the app crashes on login"])   # returns labels directly
> ```
> **Mechanism:** (1) build contrastive pairs by sampling positive and negative
> sentence pairs *within* the few-shot set; (2) contrastively fine-tune a sentence
> transformer on those pairs so the embedding space separates the classes;
> (3) fit a **logistic-regression head** on the resulting embeddings. Step 3 is a
> scikit-learn fit that takes milliseconds, which means **adding a class later is
> nearly free** — the exact opposite of BERT, where a new class means retraining
> with a resized head. Reported results: SetFit with 8 examples/class matches or
> beats GPT-3 175B few-shot prompting on classification, at ~1/1000 the inference
> cost and with no prompts. **Use SetFit when:** <100 labels/class, label set is
> volatile, or you need a classifier working today. **Do not use it when:** you
> have >1,000 labels/class (a full BERT fine-tune wins by 2–5 points) or you need
> token-level output (SetFit is sentence-level only).

> **Beyond the video — ModernBERT (Dec 2024) and why it obsoletes BERT-base.**
> ModernBERT (Warner et al., Answer.AI + LightOn) is a from-scratch encoder that
> fixes every dated component of BERT:
>
> | Feature | BERT-base (2019) | ModernBERT-base (2024) |
> |---|---|---|
> | Params | 110M | 149M (base) / 395M (large) |
> | Context | 512 | **8,192** |
> | Position encoding | Learned absolute | **RoPE** (rotary, extrapolates) |
> | Attention | Full O(S²), all layers | **Alternating local (128-token sliding window) / global** → ~2× cheaper |
> | Activation | GELU | **GeGLU** |
> | Normalisation | Post-LN | **Pre-LN** with a final LayerNorm |
> | Biases | Everywhere | **Removed** (all `bias=False`) |
> | Padding | Wasted compute on pads | **Unpadding / sequence packing** — no pad tokens at all |
> | Tokenizer | WordPiece 30k | BPE 50,368 |
> | Flash Attention | No | **Yes, plus torch.compile** |
> | Pretraining data | 16 GB / 3.3B words | **2T tokens** |
> | Throughput | ~baseline | **2–4× faster; up to 8× on long sequences** |
>
> ```python
> # Requires transformers >= 4.48
> from transformers import AutoTokenizer, AutoModelForSequenceClassification
> tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
> mdl = AutoModelForSequenceClassification.from_pretrained(
>     "answerdotai/ModernBERT-base", num_labels=2)
> # heads: ModernBERTForSequenceClassification uses mean pooling, NOT [CLS]
> ```
> ModernBERT-base beats BERT-base and DeBERTa-v3-base on GLUE and on
> long-context retrieval benchmarks, and it is the first encoder that makes
> **8k-token document classification without chunking** practical. **Migration
> caveats:** (a) it uses **mean pooling**, not `[CLS]`, so any code that slices
> `last_hidden_state[:, 0]` must change; (b) the tokenizer differs, so
> pre-tokenized caches are invalid and the `[UNK]` rate needs re-measuring;
> (c) `transformers >= 4.48`; (d) it needs Flash-Attention-capable hardware
> (Turing+) to get the throughput win. If you are starting a new encoder project
> in 2026, **start with ModernBERT and treat BERT-base as the fallback for legacy
> compatibility.**

> **Beyond the video — LoRA for BERT (when full fine-tuning is the wrong shape).**
> Full fine-tuning of BERT-base is cheap, so LoRA is unnecessary *for one task*.
> It becomes the right answer when you must serve **many** tasks or **many**
> tenants from one base model:
> ```python
> from peft import LoraConfig, get_peft_model, TaskType
> config = LoraConfig(
>     task_type=TaskType.SEQ_CLS,
>     r=8,                     # rank; BERT tolerates 4-16 (LLMs use 16-64)
>     lora_alpha=16,           # scaling = alpha/r = 2.0
>     lora_dropout=0.1,
>     target_modules=["query", "value"],   # BERT attention Q/V; add "key","dense" if underfitting
>     bias="none",
>     modules_to_save=["classifier"],      # the HEAD must be trained & saved too
> )
> model = get_peft_model(BertForSequenceClassification.from_pretrained(
>     "bert-base-uncased", num_labels=2), config)
> model.print_trainable_parameters()
> # trainable params: 887,042 || all params: 110,371,076 || trainable%: 0.803
> ```
> **Numbers:** r=8 on Q+V gives ~0.6–0.8% trainable params (~0.9M), an adapter
> file of ~3.5 MB instead of a 440 MB checkpoint, ~1.4× faster steps, and
> typically **0 to −1 points** vs full fine-tuning on GLUE-scale tasks. The
> production payoff: **one 440 MB base model resident in VRAM + N × 3.5 MB
> adapters**, so 200 tenants cost 1.14 GB instead of 88 GB. **Gotchas:**
> (a) `modules_to_save=["classifier"]` is mandatory — otherwise the randomly
> initialised head is frozen and you train only the adapters onto a random head;
> (b) `target_modules` names differ across model families (`query`/`value` for
> BERT, `query_proj`/`value_proj` for DeBERTa, `Wqkv` for some) — print
> `model.named_modules()` before guessing; (c) LR for LoRA is **10× higher** than
> full fine-tuning (1e-4 to 3e-4, not 2e-5) because you are training a small,
> randomly-initialised low-rank update; (d) merge the adapter before ONNX export
> (`model.merge_and_unload()`), because exporters handle merged weights far more
> reliably. → CS-23 is the deep dive.

> **Beyond the video — distil down to a smaller encoder (→ CS-08).** If your
> latency or RAM budget rules out even BERT-base, distillation is the mature path:
> a 12-layer teacher's soft output distribution is a much richer training signal
> than hard labels, so a 6-layer student trained on it retains ~97% of quality.
> DistilBERT's loss is exactly this — a weighted sum of (a) KL divergence against
> the teacher's softened logits at temperature `T=2`, (b) the ordinary MLM loss,
> and (c) a cosine-similarity loss between the student's and teacher's hidden
> states. Task-specific distillation (train the *fine-tuned* teacher on your task,
> then distil) typically recovers 98–99% of teacher F1 at 2× the speed. **CS-08
> covers the full recipe, the temperature/KL mathematics, and DistilBERT's
> architecture; CS-10 covers the orthogonal axis (int8 quantization), and the two
> compose — DistilBERT + ONNX int8 is the standard "fast CPU classifier" stack.**

### 13.3 Head-to-head: approaches to a closed-label classification task

| Approach | Accuracy on 12k labeled rows | Latency p50 | $ / 1M preds | Setup effort | Data needed |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.86–0.90 | 2 ms | $0.005 | 1 hour | 500 rows |
| **BERT-base full fine-tune** | **0.93–0.96** | **25 ms GPU / 45 ms CPU int8** | **$0.09–0.14** | **1 day** | **5k–50k rows** |
| DeBERTa-v3-base fine-tune | 0.94–0.97 | 30 ms | $0.12 | 1 day | 5k–50k rows |
| DistilBERT fine-tune + int8 | 0.92–0.95 | 12 ms | **$0.04** | 1 day | 10k–50k rows |
| SetFit (8 shots/class) | 0.85–0.92 | 8 ms | $0.03 | 2 hours | **8/class** |
| Zero-shot LLM prompt | 0.78–0.90 | 700 ms | $36 | 1 hour | 0 |
| LLM fine-tune (hosted) | 0.90–0.95 | 800 ms | $36–$200 | 2 hours | 500–5k rows |
| Fine-tuned BERT that you also use for embeddings | 0.93–0.96 | 25 ms | $0.09 | 1 day + retrieval | same |

### 13.4 Collator family

| Collator | Pads `input_ids` | Pads `labels` with | Use for | Silent bug if misused |
|---|---|---|---|---|
| `DataCollatorWithPadding` | yes, to batch max | **`tokenizer.pad_token_id` (0)** | Sequence classification, regression, QA-with-offsets | On NER: pads become label `0` = `O`, contaminating the loss |
| `DataCollatorForTokenClassification` | yes, dynamic | **`label_pad_token_id=-100`** | **NER / POS / any token task** | None for the right task; do NOT use when labels are ints of variable word-length without pre-alignment |
| `DataCollatorForSeq2Seq` | yes (labels padded independently) | `-100` via `label_pad_token_id` | Encoder–decoder: T5/BART | Wrong for encoders |
| `DataCollatorForLanguageModeling` | yes | builds `labels` from `input_ids`, masks per `mlm_probability` | MLM continued pretraining (CS-12) | With `mlm=False` it becomes causal-LM masking |
| `DataCollatorForMultipleChoice` | flattens `(n_choices, seq)` | keeps scalar label | SWAG / multiple choice | Shape errors are loud, not silent |
| `DefaultDataCollator` | no — stacks only | — | Already-ﬁxed-length tensors | Shape mismatch crash on variable lengths |

```python
# The one line that separates a working NER pipeline from a broken one.
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,                       # dynamic -> pad to the batch max only
    label_pad_token_id=-100,            # <- the default, and the whole point
    pad_to_multiple_of=8,               # optional: helps tensor-core alignment
)
```

> **Correction:** `align_labels_with_mapping` is **not** a `transformers` function
> and `import`ing it from `transformers` will fail. It is a helper defined in the
> **Hugging Face NLP Course, Chapter 7 ("Token classification")**, and it looks
> like this — you have to paste it into your own code:
> ```python
> def align_labels_with_mapping(labels2id, label_column):
>     """Build a per-token alignment function from a {name: id} map."""
>     id2label = {v: k for k, v in labels2id.items()}
>     label2id = labels2id
>
>     def align(example):
>         # example must already contain 'input_ids', 'attention_mask', 'word_ids'
>         new_labels = []
>         for i, word_id in enumerate(example["word_ids"]):
>             if word_id is None:
>                 new_labels.append(-100)                 # special tokens
>             elif word_id != example["word_ids"][i - 1] if i > 0 else True:
>                 new_labels.append(label2id[example[label_column][word_id]])
>             else:
>                 new_labels.append(-100)                 # continuation subwords
>         return {"labels": new_labels}
>     return align
> ```
> The library-native equivalent — and what you should actually use — is
> `tokenizer(..., is_split_into_words=True)` + `encoding.word_ids()` (as in §6.3),
> because it gets the `None` boundaries right for `[CLS]`, `[SEP]`, `[PAD]`, and
> every overflow window, which the naive `i-1` comparison does not.

### 13.5 Padding strategy

| Strategy | Code | Compute at mean 26 tokens, `max_length=512` | When |
|---|---|---|---|
| Static max-length | `tokenizer(..., padding="max_length", max_length=512)` | 100% (20× waste) | When you must pre-batch to fixed shapes (rare) |
| Static task-length | `max_length=128` | 25% | A defensible compromise when you cannot use a collator |
| **Dynamic (collator)** | `DataCollatorForTokenClassification(tokenizer)` | **5%** | **Always**, unless profiling says otherwise |
| Sequence packing | `padding="max_length"` + `packing=True` in `SFTTrainer` | ~2% | Pretraining/continued pretraining only; breaks per-example labels |
| Unpadding | ModernBERT / Flash-Attention varlen | ~4% | ModernBERT-class models; no padding tokens exist |

---

## 14. Debugging Playbook

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | **Loss flat at `ln(num_labels)`** (0.693 for binary) | Head not connected; labels all identical; LR = 0; frozen everything | `sum(p.requires_grad for p in model.parameters())`; print 5 label values | Unfreeze the head; check the label column after `map` |
| 2 | **Loss flat but *above* `ln(C)`** (e.g. 8.0 for binary) | Label ids out of range for `num_labels`; misaligned labels | `print(model.config.num_labels, max(labels))` | Set `num_labels = len(label_names)`; fix the label map |
| 3 | **Loss → 0.001 in <100 steps, eval stuck** | Overfitting; LR too high (memorisation) | Train vs eval loss gap | LR 2e-5; `num_train_epochs=2`; `weight_decay=0.01` |
| 4 | **Loss NaN / inf after a few steps** | fp16 without loss scaling; LR too high; a 10,000-token input not truncated | `torch.isnan(loss)` each step; check max input length | `fp16=True` via `Trainer` (auto-scales), `max_grad_norm=1.0`, `truncation=True` always |
| 5 | **Loss spikes then recovers repeatedly** | LR too high for the batch size; a bad batch | Log LR + grad norm per step | Halve the LR; add warmup; `max_grad_norm=1.0` |
| 6 | **Eval loss rises while train loss falls, from epoch 1** | Overfitting on a small dataset | Plot both | Fewer epochs, more data, freeze bottom layers, add `label_smoothing_factor=0.05` |
| 7 | **Eval loss rises but F1 rises too** | The model is getting more confident and marginally more correct — loss and argmax-accuracy are different objectives | Report F1 alongside loss | Select the checkpoint on F1 (`metric_for_best_model="f1"`), not loss |
| 8 | **NER F1 ≈ 0 while accuracy ≈ 0.85** | Labels not aligned, or `-100` not applied, or the collator pads labels with 0 | `assert -100 in batch["labels"]`; count non-`-100` labels per row | Use the `word_ids()` alignment loop + `DataCollatorForTokenClassification` |
| 9 | **NER predictions are one tag off from the words** | `is_split_into_words=True` missing, so the tokenizer re-tokenized pre-split words and inserted extra pieces | Compare `word_ids()` length vs `len(tokens)` | Add `is_split_into_words=True` |
| 10 | **NER misses the second half of every multi-word entity** | Continuation subwords got `-100` at training but the prediction path only emits the first subword — *or* the opposite (train on first-only, predict on all) | Align train and inference policies explicitly | Use the same alignment function for both; consider `label_all_tokens=True` with `B-`→`I-` rewriting |
| 11 | **QA returns an empty string** | `(0,0)` predicted; answer truncated; question/context swapped | Print `start_logits.argmax()` — if it is 0, you are seeing the `[CLS]` collapse | Filter pads/specials before argmax; add `doc_stride`; verify argument order |
| 12 | **QA returns a plausible but wrong span, always near the start** | Truncation: the answer is past token 512 | Log `len(context_tokens)` distribution | Sliding window with `doc_stride=128`, `return_overflowing_tokens=True` |
| 13 | **QA answers cut off mid-word** | `end_idx` taken before checking `start <= end`; `max_answer_len` truncating from the wrong end | Print predicted `(start,end)` and the decoded text | Enforce `start <= end`; use `pipeline("question-answering")` postprocessing |
| 14 | **Predictions all one class** | Class imbalance + plain CE; or the label map is inverted | Print the prediction distribution | Class-weighted CE, `WeightedRandomSampler`, or threshold tuning |
| 15 | **Accuracy 0.50 on binary, exactly** | `id2label` inverted, or the label column was renamed but not mapped | `print(model.config.id2label)`, invert a manual prediction | Set `id2label`/`label2id` explicitly |
| 16 | **Score always 0.99+** | Miscalibration (normal for small fine-tunes); or the model is overfitting | Compute ECE on held-out data | Temperature scaling; more data; more regularisation |
| 17 | **CUDA OOM at batch 8, `max_length=512`** | Activation memory; no fp16; three models resident from the multi-task driver | `torch.cuda.memory_allocated()/1e9` | `per_device_train_batch_size=4` + `gradient_accumulation_steps=2`; `fp16=True`; `gradient_checkpointing=True`; `padding` dynamic |
| 18 | **GPU utilisation 5–20%** | DataLoader with `num_workers=0` and per-item tokenization | `nvidia-smi dmon` | Pre-tokenize with `map(batched=True)`; `dataloader_num_workers=4`; dynamic padding |
| 19 | **`AttributeError: 'BertTokenizer' object has no attribute 'word_ids'`** | Slow tokenizer | `type(tokenizer)` | Use `BertTokenizerFast` / `AutoTokenizer(use_fast=True)` |
| 20 | **`IndexError: index out of range in self` on `position_ids`** | Input longer than 512 and `truncation` was not set | `max(len(x) for x in input_ids)` | `truncation=True` + a `max_length`, or sliding windows |
| 21 | **`offset_mapping` missing from the batch** | Never requested, or not popped before the forward pass | `print(encoding.keys())` | `return_offsets_mapping=True`, then `encoding.pop("offset_mapping")` |
| 22 | **Metrics differ between two identical runs** | No seed; `np.random.choice` unseeded; nondeterministic CUDA kernels | Fix all seeds; compare the sampled indices | `TrainingArguments(seed=42)`, `np.random.seed(42)`, `torch.manual_seed(42)`; `torch.use_deterministic_algorithms(True)` if you must |
| 23 | **TensorBoard directory empty** | `report_to="none"` disables the writer | `ls ./logs` | `report_to="tensorboard"` or omit it |
| 24 | **`trainer.push_to_hub` 401/403** | Token lacks WRITE scope | `whoami()` and inspect the token's permissions | Re-issue the token with write access; `notebook_login()` again |
| 25 | **Predictions differ between `Trainer` and your manual loop** | The manual loop forgot `model.eval()` (dropout active); or a different tokenizer/padding | `assert not model.training` | `model.eval()` + `torch.no_grad()` |
| 26 | **Fine-tuned model worse than the pretrained baseline** | LR too high → catastrophic forgetting | Compare `trainer.evaluate()` before and after training | LR 1e-5–2e-5; fewer epochs; freeze the bottom half |

---

## 15. Applied Case Studies

### 15.1 Fintech — transaction memo classification at 40M rows/day

**Situation.** A payments company must route 40M free-text transaction memos per day into 22 spend categories (plus `other`) for a corporate-card product. Regulatory requirement: no customer text may leave the VPC. p99 latency budget: 200 ms. Budget: $15k/month.

**Why this technique.** Closed 23-label set, short text (memos average 8 words), extreme volume, air-gapped, latency-bound. Prompting an LLM violates data egress, the latency budget, and the cost budget simultaneously.

**Exact config.**

```python
model_name = "answerdotai/ModernBERT-base"   # 149M, 8k context, 3x BERT throughput
# fallback if the serving stack is not Flash-Attention capable:
# model_name = "microsoft/deberta-v3-base"
num_labels = 23
max_length = 64                     # memos are short; 64 tokens covers p99.5
per_device_train_batch_size = 64
learning_rate = 3e-5
num_train_epochs = 3
warmup_ratio = 0.06
weight_decay = 0.01
fp16 = True                         # A10G
metric_for_best_model = "f1"        # macro F1 via compute_metrics
```

Data: 480,000 labelled memos (historical routing decisions, after de-duplication), split 440k/20k/20k stratified. Class 23 (`other`) is 31% of the data; the rarest real category is 0.4%.

**Result.**

| Metric | Value |
|---|---|
| Macro F1 (test) | 0.931 |
| Micro F1 | 0.958 |
| Rare-class recall (0.4% class) | 0.71 → **0.89** after class-weighted CE |
| Training time | 38 min on 1× A10G |
| Inference | int8 ONNX, CPU, 2,400 preds/s on 16 vCPU |
| Cost / 1M predictions | $0.11 |
| Monthly inference cost at 40M/day | ~$132 |

**What went wrong first.**
1. **Attempt 1 used `bert-base-uncased` and macro F1 was 0.884.** The memos contain merchant strings (`SQ *BLUE BOTTLE 0042`), and WordPiece shredded them into `[UNK]`-heavy fragments. Fix: ModernBERT's BPE + 2T-token corpus, plus normalising the merchant prefix before tokenization. +4.7 points.
2. **Attempt 2 had 0.12 recall on the 0.4% class** because plain CE ignored it. Fix: class-weighted CE with weights `1/sqrt(freq)` — not `1/freq`, which over-corrected and cost 3 points of micro F1.
3. **Attempt 3 served the fp32 model at 180 preds/s/box** and needed 10 boxes. Fix: `optimum-cli export onnx` + int8 dynamic quantization → 2,400 preds/s, one box, and macro F1 dropped 0.3 points.

### 15.2 Healthcare — clinical NER for adverse-drug-event extraction

**Situation.** A pharmacovigilance team must extract `(drug, dose, frequency, adverse_event)` spans from 2.1M unstructured nursing notes per year, for a regulatory submission. The audit requires reproducible, versioned, explainable predictions. No cloud.

**Why this technique.** Token-level extraction with exact boundaries over a closed entity schema — the single strongest remaining use case for encoder fine-tuning. Modern LLMs are worse than a fine-tuned encoder on boundary precision, and they cannot be version-pinned to a hash for an audit.

**Exact config.**

```python
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# 110M, domain vocabulary built from PubMed -> the [UNK] rate on clinical text
# drops from 4.1% (bert-base-uncased) to 0.3%   [estimate, measure your own]

label_names = ["O", "B-DRUG", "I-DRUG", "B-DOSE", "I-DOSE",
               "B-FREQ", "I-FREQ", "B-ADE", "I-ADE"]        # 9 labels, from the data
num_labels = len(label_names)

max_length = 256
data_collator = DataCollatorForTokenClassification(tokenizer)   # dynamic padding, -100
per_device_train_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 4                 # NER benefits from one more epoch than classification
warmup_ratio = 0.1
metric_for_best_model = "eval_f1"    # seqeval entity-level micro F1
```

Data: 14,200 annotated nursing notes (a trained pharmacist annotated 900; the rest came from an LLM-assisted pre-annotation pass with 100% human verification), split 12k/1.1k/1.1k by **patient**, not by note — the note-level split leaked the same patient's phrasing across the boundary and inflated F1 by 6 points before it was caught.

**Result.**

| Metric | Token-level (the wrong metric) | Entity-level seqeval (the right one) |
|---|---|---|
| Overall | 0.958 accuracy, 0.951 F1 | **0.894 F1** (P 0.901 / R 0.887) |
| `DRUG` | — | 0.952 F1 |
| `DOSE` | — | 0.938 F1 |
| `FREQ` | — | 0.911 F1 |
| `ADE` | — | **0.774 F1** |

The 0.958-vs-0.894 gap is the whole lesson of this module: the token metric said "95% done"; the entity metric said "one entity type is 10 points behind".

**What went wrong first.**
1. **The `ADE` type underperformed at 0.774** and the first diagnosis was "not enough data". It was actually **boundary ambiguity**: annotators disagreed on whether "nausea and vomiting" is one `ADE` span or two. Inter-annotator agreement on `ADE` was Cohen's κ = 0.61, versus 0.89 for `DRUG`. Fix: a written annotation guideline plus a re-annotation of 2,000 notes → `ADE` F1 rose to 0.841 with no model change. **Always measure annotator agreement before you blame the model — κ < 0.7 is a labelling problem.**
2. **The first run used `padding="max_length", max_length=512`** on notes averaging 34 tokens: 1.9 s/step, 6 hours/epoch. Adding `DataCollatorForTokenClassification` and `max_length=256` took it to 0.11 s/step, 22 min/epoch — a **17× speedup** with byte-identical results.
3. **The first deployment used `num_labels=9` with a hand-typed label list.** A schema change added `B-ROUTE` six weeks later; the list was updated, the model retrained, and the metrics looked fine — but `label_names` was reordered alphabetically, so every prediction was silently shifted. Fix: store `id2label` in `config.json` and assert it at load: `assert model.config.id2label == expected_map`.

### 15.3 E-commerce — multilingual review sentiment, 14 languages

**Situation.** A marketplace needs 5-star reviews mapped to `negative / neutral / positive` in 14 languages, 9M reviews/month, for seller-quality scoring. Per-language labelled data exists only for English (60k), German (9k), and French (7k); the other 11 languages have 200–1,500 rows each.

**Why this technique.** Zero-shot cross-lingual transfer: a multilingual encoder fine-tuned on the pooled multi-language data transfers to languages with almost no labels, at a fraction of the cost of a per-language model.

**Exact config.**

```python
model_name = "microsoft/mdeberta-v3-base"        # 278M, 100 languages, RTD-pretrained
num_labels = 3
max_length = 192
per_device_train_batch_size = 16
gradient_accumulation_steps = 2                  # effective batch 32
learning_rate = 2e-5                             # LOWER than monolingual: shared params
num_train_epochs = 3
warmup_ratio = 0.1
# Language-balanced sampling: without this, English (70% of rows) dominates.
```

Data: 88k labelled reviews across 14 languages, with `WeightedRandomSampler` weights set so every language contributes equally per epoch (`weight_i = 1/n_i`).

**Result.**

| Language group | Per-language data | Macro F1 |
|---|---|---|
| English | 60,000 | 0.921 |
| German / French | 7k–9k | 0.897 / 0.884 |
| Spanish / Italian / Portuguese | 1,200–1,500 | 0.861 / 0.848 / 0.855 |
| Dutch / Polish / Turkish | 600–900 | 0.832 / 0.811 / 0.804 |
| Zero-shot (Vietnamese, Thai, Indonesian) | **0** | **0.78 / 0.74 / 0.76** |

Zero-shot F1 of 0.74–0.78 in languages with no training data is the payoff, and it is why the multilingual encoder exists.

**What went wrong first.**
1. **The initial run used `bert-base-multilingual-cased` and the zero-shot languages scored 0.61–0.68.** mBERT has no explicit cross-lingual objective. Swapping to mDeBERTa-v3-base (RTD-pretrained on CC-100) gained **+10 points** zero-shot.
2. **Adding all 14 languages to one training set *hurt* the high-resource languages** — English fell from 0.921 to 0.892. Fix: language-balanced sampling, plus a per-language LR multiplier is unnecessary once the sampling is fixed.
3. **Tokenization cost dominated.** The multilingual tokenizer is ~2.4× slower than the English one (250k vocab, SentencePiece over 14 scripts). Fix: pre-tokenize once into Arrow with `map(batched=True, num_proc=8)` and cache it; tokenization went from 40% of wall-clock to a one-time 6-minute pass.

### 15.4 Insurance — extractive QA over 90k policy documents

**Situation.** Claims adjusters ask natural-language questions of policy PDFs ("what is the water-damage sub-limit for a commercial property policy?"). 90,000 documents, average 12,000 tokens. p95 latency budget 400 ms. An LLM-based RAG prototype answered at ~$0.004/query, which at 400k queries/month is $1,600/month with a hallucination rate the compliance team rejected.

**Why this technique.** The answer must be *verbatim from the policy* for legal defensibility. Extractive QA cannot hallucinate the answer, because the answer is by construction a substring of the input. The LLM is kept in a downstream, clearly-labelled synthesis step.

**Exact config.**

```python
model_name = "deepset/deberta-v3-base-squad2"    # already SQuAD-2.0 fine-tuned
tokenizer_kwargs = dict(
    max_length=384,
    stride=128,                                  # doc_stride
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
inference = dict(max_answer_len=30, handle_impossible_answer=True,
                 top_k=5, n_best_size=20, doc_stride=128, max_seq_len=384)
```

Pipeline: BM25/embedding pre-retrieval to the top 3 clauses → the QA model scores spans across all windows → de-duplicate by answer text → return with a score → **if the top score < 0.15, abstain and escalate to a human.**

**Result.**

| Metric | Value |
|---|---|
| Exact match (internal, 800-question gold set) | 0.741 |
| Token F1 | 0.843 |
| Abstain rate | 11.4% of queries (escalated to humans) |
| Answer-is-a-substring guarantee | 100% by construction |
| Latency p95 (GPU, batch 8) | 310 ms including retrieval |
| Cost / query | $0.00021 (GPU amortised) |
| Hallucinated answers | **0** — architecturally impossible for the extracted span |

**What went wrong first.**
1. **`max_length=512, stride=0` lost 22% of answers** that straddled a window boundary. Adding `stride=128` and `max_length=384` recovered them; the answer-loss rate dropped to 0.8%.
2. **The first version truncated the question** because `truncation=True` (default) truncates the longest sequence — which is the context, so it worked by accident until a long multi-part question appeared. Fix: `truncation="only_second"`, explicitly.
3. **`argmax` over all 384 positions returned `[PAD]` positions** on short clauses, producing empty answers with high raw logit. Fix: the `pipeline` postprocessor's `handle_impossible_answer=True` plus masking positions where `offset_mapping == (0,0)`, and an explicit score threshold for abstention.
4. **The initial deployment had no abstain path**, so 3.8% of queries got a confident wrong span. Adding the 0.15 threshold and human escalation converted those into escalations — a quality win and a compliance win, at the cost of 11% more human work.

### 15.5 SaaS — intent routing with 300 labels and a moving schema

**Situation.** A customer-support platform routes tickets to 300 possible queues. The queue taxonomy changes monthly (queues are added, merged, and retired). Total labelled tickets: 41,000, distributed very unevenly (the largest queue has 6,200, the smallest 12).

**Why this technique — then why not.** A 300-class encoder is entirely feasible, and it was the initial build. But the *label churn* is the real requirement, and it is where BERT is the wrong shape: changing `num_labels` invalidates the head, so a taxonomy edit means a full retrain plus a re-validation cycle, monthly.

**Exact config — the hybrid that shipped.**

```python
# Layer 1 (encoder, monthly): 12 COARSE groups that are stable.
COARSE = 12
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-small", num_labels=COARSE)     # 44M, 3x faster than base

# Layer 2 (SetFit, refit in seconds when the taxonomy changes):
from setfit import SetFitModel
fine = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# trained on the 300 fine labels *within* each coarse group
```

Routing: the encoder picks the coarse group; the corresponding SetFit head (one per group) picks the fine queue. Adding a queue = refit one SetFit head on ~32 examples in under 10 seconds, no GPU, no re-validation of the encoder.

**Result.**

| Metric | BERT 300-class (v1) | Two-stage (v2) |
|---|---|---|
| Coarse top-1 accuracy | — | 0.962 |
| Fine top-1 accuracy (within-group) | — | 0.884 |
| End-to-end top-1 | 0.879 | 0.851 |
| **Taxonomy change → production** | **5 days** (retrain + revalidate) | **30 minutes** (refit a head) |
| GPU requirement to update | 1× A10G, 40 min | none |
| Serving cost / 1M | $0.11 | $0.13 |

The two-stage design gives up **2.8 points of accuracy** and buys a **240× faster taxonomy-change cycle**. For a queue taxonomy that churns monthly, that is the correct trade — and it is the kind of reasoning the interview file's L4 questions are testing.

**What went wrong first.** The v1 300-class model had 44 queues with fewer than 30 training examples each; those 44 queues averaged 0.31 F1 and dragged the macro F1 to 0.61 while micro F1 sat at 0.93 — the imbalance trap in its purest form, and the reason the initial dashboard said "93% and healthy" while agents were re-routing tickets by hand.

---

## 16. Production Considerations

### 16.1 Serving

| Stack | When | Latency (S=128, batch 16) | Notes |
|---|---|---|---|
| `transformers` + `pipeline` | Prototyping only | 60–150 ms/req GPU | Python overhead per call dominates |
| `transformers` + manual batched `torch.no_grad()` | Small-scale GPU serving | 20–40 ms/req | Batch by request, not by row |
| **`optimum` ONNX export + ONNX Runtime (CUDA)** | GPU serving at volume | **8–20 ms/req** | 2–3× over eager; fused attention |
| **ONNX Runtime CPU + int8 dynamic quant** | CPU serving, high volume | **15–60 ms/req** | The standard high-volume CPU stack |
| TorchScript / `torch.compile` | GPU, if ONNX export is blocked | 15–30 ms | Less portable than ONNX |
| NVIDIA Triton + dynamic batching | Multi-model, autoscaled | 5–15 ms/req | Real request coalescing under load |
| **TEI (Text Embeddings Inference)** | Embedding models specifically | 2–8 ms/req | Rust; the standard for retrieval at scale |
| AWS SageMaker / Vertex AI endpoint | You want managed infra | 25–60 ms/req | Costs 2–3× raw GPU |

```bash
# Export + quantize. This is the whole CPU-serving recipe.
pip install optimum[exporters,onnxruntime]

# 1. Export to ONNX (from the saved directory that contains model AND tokenizer)
optimum-cli export onnx \
  --model ./bert-finetuned-imdb \
  --task text-classification \
  --opset 17 \
  ./bert-finetuned-imdb-onnx
# -> model.onnx (~440 MB), config.json, tokenizer.json, vocab.txt

# 2. Dynamic int8 quantization (post-training, no calibration data needed)
optimum-cli onnxruntime quantize \
  --onnx_model ./bert-finetuned-imdb-onnx \
  --avx512 \
  -o ./bert-finetuned-imdb-int8
# -> ~110 MB, 2-4x faster on CPU, typically <1 point of accuracy loss.

# 3. Sanity check the export against the original
python -c "
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
m = ORTModelForSequenceClassification.from_pretrained('./bert-finetuned-imdb-int8')
t = AutoTokenizer.from_pretrained('./bert-finetuned-imdb-int8')
print(m(**t('This movie was amazing', return_tensors='pt')).logits.softmax(-1))
"
```

**Always verify the export.** ONNX conversion can silently change numerics; run 500 held-out examples through both the PyTorch and the ONNX model and assert that **>99% of predictions agree**. A 2-point disagreement rate means a fused kernel is behaving differently and your metrics are now wrong.

### 16.2 Batching

```python
# WRONG (the notebook's predict()): one forward pass per text.
# 1M predictions = 1M tokenizer calls + 1M kernel launches.

# RIGHT: sort by length, batch, then restore the original order.
import numpy as np, torch
from torch.utils.data import DataLoader

def batched_predict(texts, model, tokenizer, batch_size=64, max_length=256):
    model.eval()
    order = np.argsort([len(t) for t in texts])          # group similar lengths
    sorted_texts = [texts[i] for i in order]
    results = [None] * len(texts)
    for i in range(0, len(sorted_texts), batch_size):
        chunk = sorted_texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for j, p in enumerate(probs):
            results[order[i + j]] = p
    return np.array(results)
```

Length-sorted batching plus dynamic padding is typically a **3–6× throughput win** at equal accuracy. Add `torch.inference_mode()` instead of `no_grad()` for a further ~5%; add `model.half()` on a GPU with tensor cores for ~1.8×.

### 16.3 Versioning, rollback, and drift

| Practice | Concrete implementation |
|---|---|
| **Pin the artefact** | Store `model.safetensors` + `tokenizer.json` + `config.json` + the ONNX export, keyed by the git SHA of the training code and a data-manifest hash |
| **Pin the label map** | `config.json` must carry `id2label`/`label2id`. Assert them at service startup; fail fast if they differ from the deployed routing table |
| **Log the model version per prediction** | Without it you cannot attribute a quality regression to a deployment |
| **Shadow deploy** | Run the new model on 100% of traffic in parallel, log its outputs, do not act on them for 48 hours |
| **Canary** | 1% → 5% → 25% → 100%, with automatic rollback on a metric breach |
| **Rollback** | Keep the previous ONNX artefact warm. Rollback must be a config flip, not a retrain |
| **Input drift** | Track the distribution of `tokenizer(text).input_ids` length, the `[UNK]` rate, and the mean predicted confidence. A jump in the `[UNK]` rate means your input distribution changed — the model cannot be expected to hold |
| **Prediction drift** | Track the label distribution weekly. A 5-point shift in the class mix is either a business change or a model problem; you must know which |
| **Concept drift** | You need fresh labels. Sample 200 predictions/week, label them, track rolling F1. Without this you are flying blind |
| **Regression tests** | A frozen 500-row golden set with expected outputs checked into CI. Every retrain must pass it. Store the *expected probabilities*, not just the argmax — a 0.02 shift matters |
| **Guardrail on confidence** | Below a threshold, route to a human. Never let a 0.51-confidence prediction trigger an irreversible action |
| **Data retention** | If you log inputs for retraining, you have created a PII store. Anonymise at the logging boundary, or log only token ids and hashes |

### 16.4 Compliance angle

- **Explainability.** A 110M encoder supports token-attribution (`captum`, `shap` on the embedding layer) and attention inspection, which is far easier to defend to a regulator than "the LLM said so". For adverse-action decisions, per-token attribution is often the differentiator that selects the encoder.
- **Right to explanation / GDPR Art. 22.** Automated decisions with legal effect need a human-review path. A confidence threshold plus an escalation queue satisfies this cheaply.
- **Data minimisation.** Fine-tuned BERT does not retain your text at inference time — the model weights are a lossy summary. That is a materially easier story than an LLM API whose vendor may retain prompts.
- **Model cards.** Publish the training data description, the label taxonomy, per-class metrics, known failure modes, and the intended use. `trainer.push_to_hub` scaffolds this; fill it in.
- **Bias auditing.** Slice your test set by whatever protected or business-relevant segments exist and report per-slice F1. A 0.93 overall F1 with 0.61 on one slice is a liability, not a success.

---

## 17. Common Misconceptions

1. **"BERT is a generative model."** No. The instructor states this four separate times [8:59–9:06, 18:06–18:22]. BERT has no causal attention mask, so there is no well-defined `P(x_t | x_<t)` to sample from. It reads; it does not write.
2. **"BERT-large has 20 encoders."** It has **24** layers. The transcript says 20 once [8:22] and correctly says 24 at [10:57]; the paper says 24 layers, 1024 hidden, 16 heads, 340M params.
3. **"`[CLS]` is the sentence embedding."** It is the *classification* vector. For similarity/retrieval, mean pooling over non-pad tokens is measurably better — that is the entire premise of Sentence-BERT.
4. **"`[MASK]` is a normal token."** It is a pretraining artefact that never appears in your fine-tuning data. BERT's own authors tuned the mask rate *per task* (0.10 for NER, 0.15 for classification, 0.30 for SQuAD) precisely because `[MASK]` distorts the distribution. RoBERTa and DeBERTa-v3 removed it.
5. **"The head is pretrained."** The head is randomly initialised at fine-tuning time. That is why step-0 loss ≈ `ln(num_labels)` and why the first ~100 steps are the head orienting itself.
6. **"More epochs = better."** Encoder fine-tuning on <10k rows usually peaks at epoch 2–3 and degrades after. Watch `eval_loss`, and use `load_best_model_at_end=True`.
7. **"One token = one word."** WordPiece fragments words: `London` → `Lon` + `##don`. This false belief is the direct cause of the #1 NER bug in this module.
8. **"Token accuracy tells you how good your NER is."** A model that predicts `O` for every token scores 0.85+ accuracy on CoNLL-2003 and 0.0 entity F1.
9. **"The `Trainer` default collator is fine."** For NER it is not: `DataCollatorWithPadding` pads labels with `0`, which is a valid tag (`O`), so your pads train the model. Use `DataCollatorForTokenClassification`.
10. **"BERT fine-tuning is not LR-sensitive."** It is the most LR-sensitive training in this entire handbook. 2e-5 works; 2e-3 destroys the model in ~100 steps while the *training* loss falls the whole time.
11. **"`attention_mask` is optional."** Without it the model attends to `[PAD]` tokens. Accuracy drops a few points and, worse, batch composition changes your predictions — the same text gets different answers depending on how much padding is next to it.
12. **"Bigger encoders are always better."** DeBERTa-v3-base (86M) beats RoBERTa-large (355M) on several GLUE tasks. Architecture and pretraining objective matter more than parameter count in the encoder regime.
13. **"BERT is obsolete."** It is obsolete *as a general-purpose language model*. It is not obsolete as a classifier: for a closed label set at >1M predictions/month, nothing else comes close on cost and latency.
14. **"You need 512 tokens."** Most classification tasks are solved at 128–256. Padding to 512 costs 4–20× the compute for typically <1 point of accuracy.
15. **"`softmax` output is a probability."** It is an uncalibrated score. Fine-tuned encoders on small data typically show ECE of 0.10–0.25 — a 0.95 confidence is right about 70–85% of the time.
16. **"You can fine-tune a 7B LLM the same way you fine-tune BERT."** The instructor makes this distinction explicitly at [38:27–39:40]: with BERT you can retrain all layers ("this full fine-tuning is also possible with a small learning rate"), whereas with a state-of-the-art LLM you cannot — that is what LoRA and QLoRA exist for. (Strictly, full fine-tuning of a 7B is possible on an 80 GB GPU; it is the *memory* that forces PEFT, not the principle.)

---

## 18. Key Takeaways

1. **BERT is an encoder: it reads, it does not write.** No causal mask ⇒ no generation. This one architectural fact explains its entire strength list and its entire limitation list.
2. **Fine-tuning = keep the body, swap the head, use a tiny learning rate.** LR 2e-5 (range 1e-5–5e-5); anything near 1e-3 destroys the pretrained representation while the training loss keeps falling.
3. **The head starts random.** Step-0 loss should be ≈ `ln(num_labels)`. If it is far from that, your labels are misaligned or `num_labels` is wrong — before you change anything else.
4. **Subword alignment is the #1 silent NER bug.** `word_ids()` → first subword keeps the label, everything else gets `-100`. `-100`, not `0` — `0` is a real tag.
5. **Token accuracy is a lie for NER; `seqeval` entity F1 is the truth.** A useless model scores 0.85 token accuracy and 0.00 entity F1.
6. **`DataCollatorForTokenClassification` is not optional for token tasks.** It (a) pads dynamically — a 17× throughput win on short sentences — and (b) pads labels with `-100` instead of `0`.
7. **`[CLS]` pooling works for classification, not for similarity.** Use mean pooling for embeddings.
8. **MLM's 80/10/10 is a de-biasing trick**, not an arbitrary rule. The 10% random and 10% unchanged prevent the model from depending on the literal `[MASK]` token and from copying the input outside it.
9. **NSP was a mistake.** Its negatives come from a different document, so a topic-shift detector solves it. RoBERTa removed it and improved; ALBERT replaced it with SOP; ELECTRA, SpanBERT, and DeBERTa dropped it.
10. **512 tokens is a hard ceiling** (learned absolute positions) and a soft one (quadratic attention). For QA, use `max_length=384` with `doc_stride=128` and sliding windows.
11. **Padding strategy is the cheapest big win in this module.** Dynamic padding is one argument and buys 2–20× on variable-length data.
12. **The cost argument is decisive.** Fine-tuned BERT serves 1M predictions for **$0.09–0.14**; GPT-4o charges **$600**, GPT-4o-mini **$36**. Above ~10M predictions/month with a stable label set, the encoder wins by 3–4 orders of magnitude.
13. **Below ~50–100 labels per class, do not fine-tune BERT.** Use SetFit (8 shots/class, seconds to refit) or a prompted LLM. BERT needs data to earn its keep.
14. **DeBERTa-v3-base is the modern default encoder; ModernBERT (2024) is the modern default *architecture*.** Start with DeBERTa-v3-base for quality, ModernBERT for long context and throughput, DistilBERT/ELECTRA-small for latency, PubMedBERT/Legal-BERT for domain.
15. **Labelling costs 10,000× the training.** 12,000 tickets ≈ 50 annotator-hours ≈ $1,250; the GPU run is $0.06. Budget your project accordingly, and measure annotator agreement (κ) before blaming the model.

---

## 19. Self-Check Questions

1. Why can BERT not generate text? Name the exact architectural component that is absent and explain what it does in a decoder.
2. A colleague's NER model reports 0.91 token accuracy and 0.03 entity F1. Give three distinct root causes and the diagnostic for each.
3. You have 40 labelled examples per class across 8 classes and no labelling budget. What do you do, and why not fine-tune BERT?
4. Explain the 80/10/10 masking rule. What specifically breaks if you use 100% `[MASK]`?
5. Your QA model returns an empty string for 30% of questions. List the four most likely causes in the order you would check them.
6. Derive the number of optimizer steps for 12,000 training rows with batch size 32 over 3 epochs, and state how many of those steps are warmup at `warmup_ratio=0.1`. Is this a healthy number of steps?
7. Your fine-tuned BERT scores 0.94 on your 500-row test set, but a colleague's identical run scores 0.91. Give four innocent explanations before you conclude anything is wrong.
8. Why does `DataCollatorForTokenClassification` matter, and what exactly goes wrong if you use `DataCollatorWithPadding` for NER instead?
9. You are choosing between `bert-base-uncased`, `distilbert-base-uncased`, `deberta-v3-base`, and `answerdotai/ModernBERT-base` for a 3-class classification task on 8,000 rows at 4M predictions/day on CPU. Which, and why? What would change your answer?
10. RoBERTa removed NSP and beat BERT. Explain the mechanism by which removing an objective *improves* a model.

<details>
<summary>Answers</summary>

1. **The causal attention mask is absent.** In a decoder, `M[i,j] = -inf for j > i` makes each position attend only to earlier positions, so `P(x_t | x_<t)` is well-defined and the training objective (predict the next token) equals the inference procedure (generate the next token). BERT's mask has no causal component: position *i* attends to all positions including *j > i*. There is therefore no "next token" distribution to sample from, and appending a token would change the representation of every preceding token, so autoregressive decoding would not even be self-consistent.

2. (a) **Continuation subwords are not masked with `-100`**, so one entity is counted 2–4 times — diagnostic: `(labels == -100).sum()` per row should be ≥ the number of special tokens plus continuations; if it equals only 3 (the specials), this is the bug. (b) **Token-level scoring is being used** — diagnostic: recompute with `seqeval`; if entity F1 jumps to 0.6+, that is the whole story. (c) **Labels are misaligned by a constant offset** (e.g. `is_split_into_words=True` missing so the tokenizer re-tokenized already-split words) — diagnostic: print `list(zip(tokens, word_ids()[1:len(tokens)+1]))` for one example and check the labels land on the right words.

3. **Use SetFit.** Contrastively fine-tune a sentence transformer on pairs sampled from your 320 examples, then fit a logistic-regression head. With 8–40 examples per class, a BERT fine-tune with a randomly-initialised head over 8 classes has too few steps to converge (320 rows / batch 8 = 40 steps/epoch — an order of magnitude below the ~1,000-step minimum) and will underfit or memorise. SetFit reports GPT-3-few-shot-level accuracy at 8 examples/class, the head refits in seconds when classes change, and inference is a sentence-transformer forward pass plus a dot product.

4. Of the 15% of selected positions: **80%** become `[MASK]`, **10%** a random vocabulary token, **10%** unchanged. With 100% `[MASK]` the model learns (a) that only the literal `[MASK]` token needs careful encoding, and (b) that outside `[MASK]` it may simply copy the input. At fine-tuning time `[MASK]` never appears, so (a) never activates and every representation is produced by a model that has never encoded a corrupted context. The 10% random token forces the model to maintain a correct representation of the original token even when the input is wrong; the 10% unchanged keeps some loss-bearing positions in a fully clean context. The `[MASK]` mismatch is also why BERT's authors tuned the mask rate per downstream task (0.10 NER, 0.15 classification, 0.30 SQuAD) — direct evidence that the token was a defect.

5. In order: (a) **positions are not filtered** — `argmax` over all 512 positions can select a `[PAD]`; check whether `start_logits.argmax()` lands where `attention_mask == 0`, fix by masking pads and specials before the argmax. (b) **The answer is truncated away** — log the distribution of `len(context_tokens)`; if a meaningful fraction exceeds 512, add `doc_stride` + `return_overflowing_tokens`. (c) **The `(0,0)` default was learned** — count training rows whose `start_positions == 0`; if >2%, your alignment loop is failing on some rows and teaching the model to point at `[CLS]`. (d) **`start > end` repair is producing one-token spans** — print the raw `(start_idx, end_idx)` pairs; if `end_idx == start_idx` frequently, switch to proper n-best span decoding with `start <= end` enforced during enumeration, not after.

6. `steps = (12,000 / 32) × 3 = 375 × 3 = 1,125` optimizer steps. Warmup at `warmup_ratio=0.1` = **112 steps** (floor) of linear ramp, then 1,013 steps of linear decay. **Yes, 1,125 is healthy** — comfortably above the ~1,000-step floor and well below the point where 12k rows overfit on an encoder. Compare the video's demo: 1,000 rows / batch 8 × 1 epoch = **125 steps**, which is an order of magnitude short and is why the demo's predictions are unreliable.

7. (a) **Different random seed** — model init, dropout, and data order all vary; ±1–2 points is routine on 500 test rows. (b) **Different 500 rows** — the notebook's `np.random.choice` has no seed, so the two runs are not evaluating on the same data; a 3-point swing is trivial when the test set changes. (c) **Different test-set size or split** — one run may be evaluating on a 500-row IMDB *test* subset and the other on a validation carve-out of train. (d) **Different `max_length` or padding policy**, which changes how much of each review is visible. Before concluding there is a bug, re-run both with `seed=42`, the same frozen test set, and the same tokenizer config, and report a Wilson 95% CI (±3.1 points at n=500).

8. `DataCollatorForTokenClassification` does two things: it pads `input_ids`/`attention_mask` **dynamically to the batch maximum** (not to a global `max_length`), and it pads `labels` with **`label_pad_token_id=-100`**, which `CrossEntropyLoss` ignores. With `DataCollatorWithPadding` for NER, labels are padded with `tokenizer.pad_token_id = 0` — and `0` is a *valid tag* (`O`). Consequences: (a) the loss includes the pad positions, so it is contaminated and its magnitude is no longer comparable across runs with different padding; (b) the model is trained to predict `O` on padding, which subtly biases the decision boundary against rare entity types; (c) evaluation counting becomes wrong because pad positions now look like real `O` labels. Neither failure crashes, and both cost a few points of F1 — the definition of a silent bug.

9. **`distilbert-base-uncased`** (or `ModernBERT-base` if the serving stack supports Flash Attention and you can afford 149M params). Reasoning: 4M predictions/day = 120M/month, so unit cost and CPU latency dominate; DistilBERT is 40% smaller and 60% faster than BERT-base at 97% of GLUE, and with ONNX int8 it serves at roughly 1,200–1,800 preds/s per 8-vCPU box versus ~700 for BERT-base — halving the fleet. `bert-base-uncased` is the safe fallback if DistilBERT's quality gap shows up on your eval. `deberta-v3-base` is the quality play but is ~1.6× slower than BERT-base on CPU, so it is the wrong choice when throughput is the binding constraint. **What would change the answer:** if the eval shows DistilBERT more than 1.5 points of macro F1 below DeBERTa-v3-base, take the quality hit on cost and use DeBERTa-v3-base or `deberta-v3-xsmall`; if inputs exceed 512 tokens, ModernBERT is the only option in the list; if 8,000 rows proves too few, the gap widens toward the larger models.

10. **Removing an objective removes a shortcut that the encoder was spending capacity on.** NSP's negatives are drawn from a different document, so a model can achieve near-perfect NSP accuracy using a topic-shift detector — `A` about cats, `B` about interest rates ⇒ `NotNext` — without learning anything about discourse or sentence relationships. That learned shortcut occupies representational capacity in the encoder and, worse, shapes the `[CLS]` vector (which is exactly what NSP trained) around a topic-detection feature rather than a task-useful one. Removing NSP frees that capacity to be spent on MLM, and removes the topic-detection bias from the `[CLS]` representation that downstream classifiers read. RoBERTa's ablation showed a slight downstream *improvement*; ALBERT showed that a *harder* sentence-level objective (SOP — distinguishing `A-B` from `B-A`, same document, no topic shift available) does carry useful signal. The general lesson: **an auxiliary objective helps only if it cannot be solved by a shortcut.**

</details>

---

## 20. Cross-References

| Relationship | Module |
|---|---|
| Builds on | CS-01 (pretraining & the LLM lifecycle), CS-02 (transfer learning & fine-tuning), CS-05 (RNN/LSTM → attention), CS-06 (Hugging Face masterclass: `AutoModel`, tokenizers, `Trainer`, `pipeline`) |
| Needed by | CS-08 (knowledge distillation — DistilBERT *is* a distilled BERT), CS-10/CS-11 (quantization — BERT int8 ONNX is the canonical CPU target), CS-12 (domain-adaptive continued pretraining on your own PDFs — the MLM objective reappears there), CS-22 (embedding models: BERT/MiniLM as the retriever in RAG) |
| Contrasts with | CS-13 (instruction fine-tuning with decoder LLMs — the *generation* counterpart), CS-14 (alignment: encoders have no preference signal), CS-20 (fine-tuning SLMs), CS-23 (LoRA/QLoRA — the PEFT route when full fine-tuning is the wrong shape, and how to apply it to BERT) |
| Pairs with | CS-03/CS-04 (framework and architecture selection — where the encoder-vs-LLM decision is actually made) |
| Cheat sheet | **CH-07** — three complete task templates, the alignment snippet, and the cost calculator |
| Interview bank | **IQ-07** — 104 questions across five levels |

---

## Appendix A — Instructor's Verbatim Key Claims

| Timestamp | Claim (verbatim) | Why it matters |
|---|---|---|
| [4:50] | *"the BERT model was published in 2019 at 24th May 2019"* | BERT's publication date; the paper is arXiv:1810.04805 (submitted Oct 2018, published 2019) |
| [4:57] | *"the full form of the BERT is pre-training of deep bidirectional transformer for language understanding"* | The acronym as it appears in the paper title |
| [7:48] | *"BERT base ... in total 12 encoders"* | BERT-base = 12 layers, 110M params |
| [8:23] | *"talking about the BERT large. So there you will find out in total 20 encoders"* | **Incorrect** — he corrects himself at [10:57] to 24. BERT-large = 24 layers, 1024 hidden, 16 heads, 340M params |
| [9:04] | *"if talking about the BERT so is not a generative model"* | The module's central architectural fact |
| [9:10] | *"the task basically was the MLM or the task was the NSP next sentence prediction"* | The two pretraining objectives |
| [13:35] | *"here it has mentioned semi-supervised but this is not a correct word. You can say it is a self-supervised pre-training or unsupervised pre-training"* | He explicitly corrects the slide. Self-supervised is the correct modern term |
| [14:02] | *"MLM means mask language modeling and NSP technique. NSP means next sentence prediction"* | The acronym expansions |
| [13:59] | *"using this MLM technique ... and NSP technique"* | How BERT is trained |
| [18:06] | *"this BERT model is not a generative model ... GPT model is a generative model this BERT model is not a generative model"* | Repeated for emphasis; flagged as an interview question |
| [19:40] | *"this head is nothing ... it is a feed forward neural network ... this is also called dense network"* | The task head is an FFN; random at fine-tuning time |
| [20:05] | *"along with that you will find out the softmax. Softmax also will be there"* | The head includes a softmax for classification |
| [21:58] | *"this head is going to be changed according to the task"* | One encoder, N heads — the transfer-learning premise |
| [22:13] | *"NER or POS tagging this comes under the token classification"* | NER and POS are the same task shape |
| [22:56] | *"BERT is still being used at multiple places in a different different form"* | The framing for the "where BERT still wins" section |
| [23:57] | *"inside that we are going to be perform this data injection ... in the vector database"* | BERT as the embedding model in a RAG retrieval pipeline |
| [25:41] | *"this BERT it is still being used for some enterprise NLP level of task for fast and the cost effectiveness"* | The cost argument, in the instructor's own words |
| [26:06] | *"it is easy to configure, right? We can configure it on any basic configuration or any basic infrastructure"* | Edge / on-prem / low-cost deployment |
| [26:42] | *"it is not very much trustworthy compared to the today's LLM model. It might work but ... you will have to compromise with some accuracy"* | An honest statement of the quality trade-off |
| [28:43] | *"I kept the name of couple of more model like BERT, distilBERT, roBERTa, deBERTa, MPNet, MiniLM, Flan, Pegasus, XLNet, XLM-R, ELECTRA"* (ASR-garbled in the transcript) | The encoder family menu; all drop-in via `AutoModel` |
| [29:22] | *"this fsspec ... help us to download any file from server to the local"* | The `pip install datasets fsspec transformers` line |
| [31:09] | *"customer feedback categorization whether it is a positive negative or neutral, support ticket classification whether it is for billing, technical, for any general ticket, email topic classification means whether it's a ham or a spam"* | The three real-world use cases for this task shape |
| [32:04] | *"more than 25,000 sample ... at least 5 to 6 hour to train on this much of data"* | IMDb train size and an fp32, free-Colab wall-clock figure |
| [32:20] | *"I'm taking 1,000 sample for the training and I'm taking 500 sample for the testing"* | The demo scale |
| [33:06] | *"the max length how much I'm providing 256"* | `max_length=256` |
| [33:38] | *"in one sentence we have 300 tokens. In that case there could be only 256"* | Truncation explained |
| [33:47] | *"rest of the token will be showcased as a zero value. Okay, padding value"* | Padding explained; id 0 is `[PAD]` |
| [34:24] | *"by default the output feature name was a label but the hugging face required this specific name that is labels"* | Why `rename_column("label", "labels")` is mandatory |
| [35:14] | *"in output class how should how many labels should be there. So there are only two label"* | `num_labels=2` |
| [36:32] | *"fine-tuning means we are going to be retrain the model"* | The instructor's definition of fine-tuning |
| [36:54] | *"there could be a full retraining ... a couple of layer ... or there could be a last layer, last output layer"* | The three fine-tuning strategies |
| [37:24] | *"in BERT model this full fine-tuning is also possible with a small learning rate"* | The core encoder hyperparameter rule |
| [38:17] | *"last two layer, right? Last two layer, we can select some last five layer. It's up to me how many layers I want to be fine-tuned"* | Selective layer fine-tuning |
| [38:53] | *"you cannot fine-tune the last couple of layer of this model or you cannot fine-tune the full encoder or decoder stack of this model. It's not at all possible"* | The LLM-vs-encoder fine-tuning distinction, and why PEFT exists |
| [40:34] | *"in one single batch the eight data point would be there"* | `per_device_train_batch_size=8` |
| [40:43] | *"decay rate means it is related to that optimizer ... with this particular decay rate my weight is not going to be fluctuate too much"* | `weight_decay=0.01` |
| [45:41] | *"tensorboard is running on top of it ... it won't be available over the localhost"* | The Colab TensorBoard port trap |
| [46:26] | *"checkpoint means at a specific step till the specific step we are going to be save the state of the model not the complete model"* | Checkpoint vs `save_model` |
| [47:21] | *"inside this matrix we have a evaluation loss, evaluation runtime, evaluation sample per second, evaluation step per second"* | What `trainer.evaluate()` returns |
| [51:32] | *"simply I will put two and I will hit enter and see my finetuning have been started for the NER task"* | The 1/2/3/4 multi-task driver |
| [53:44] | *"apart from this one we have one more pythonic way"* | The custom `Dataset` route |
| [56:10] | *"this get linear schedule with warmup ... learning rate is going to be set automatically using this particular function"* | `get_linear_schedule_with_warmup` |
| [58:05] | *"I'll give you this entire code"* | The notebook is the artefact |

## Appendix B — Reference Links & Papers

| Resource | Why read it |
|---|---|
| Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (arXiv:1810.04805) | The original. Appendix A.2 has the 80/10/10 ablation; Appendix C.1 has the per-task mask-rate tuning |
| Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (arXiv:1907.11692) | The NSP ablation, dynamic masking, 160 GB of data, 50k BPE |
| Lan et al., *ALBERT: A Lite BERT* (arXiv:1909.11942) | Cross-layer parameter sharing, factorized embeddings, SOP |
| Clark et al., *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators* (arXiv:2003.10555) | Replaced-token detection; the compute-efficiency argument |
| He et al., *DeBERTa* (arXiv:2006.03654) and *DeBERTa-V3* (arXiv:2111.09543) | Disentangled attention, EMD, gDES; the modern best-in-class encoder |
| Joshi et al., *SpanBERT* (arXiv:1907.10529) | Span masking + SBO — the right encoder for span-structured tasks |
| Sanh et al., *DistilBERT* (arXiv:1910.01108) | The distillation recipe; read this before CS-08 |
| Reimers & Gurevych, *Sentence-BERT* (arXiv:1908.10084) | Why `[CLS]` is a poor sentence embedding and mean pooling is not |
| Tunstall et al., *SetFit: Efficient Few-Shot Learning Without Prompts* (arXiv:2209.11055) | 8-shot classification that beats GPT-3 few-shot |
| Warner et al., *ModernBERT* (arXiv:2412.13663) | The 2024 encoder architecture; 8k context, unpadding, alternating attention |
| Rajpurkar et al., *SQuAD 2.0* (arXiv:1806.03822) | The no-answer formulation and the `CLS` logit trick |
| Hugging Face NLP Course, Chapter 7 — *Token classification* | The `align_labels_with_mapping` helper, `seqeval`, and the canonical NER pipeline |
| `huggingface/evaluate` — `seqeval` metric card | The entity-level scoring definition and the IOB2/BILIO scheme flag |
| `optimum` documentation — ONNX export and ORT quantization | The exact CLI for the CPU serving path |
| Companion notebook: `LLM Fine-Tuning-09-Bert-finetuning/BERT_Finetuning.ipynb` | The three runnable demos, the alignment walk-through, and the `Basket` dunder teaching example |
