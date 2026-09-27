# IQ-07 — Interview Questions: Fine-Tuning BERT (NER, Sentiment, QA)

| Field | Value |
|---|---|
| **Module** | Encoder-only fine-tuning (classical NLP) |
| **Pairs with** | CS-07, CH-07 |
| **Total questions** | 104 (32 L1 + 32 L2 + 22 L3 + 8 L4 + 10 L5) |
| **Levels covered** | Screen / Intermediate / Advanced / System Design / Debug |
| **Plus** | Rapid-fire T/F table, 6 whiteboard tasks, a numbers-to-memorise sheet, and full answers to the CS-07 self-check |

---

## How To Use This File

- **L1 = phone screen / recruiter filter** — 30-second answers. If you cannot answer these cold, you will not reach L2.
- **L2 = working engineer** — 2–3 minute answers. The interviewer expects implementation detail: library flags, exact column names, real numbers.
- **L3 = senior / specialist** — 5 minutes. Trade-offs, internals, and *when this is the wrong tool*.
- **L4 = staff / system design** — 15-minute whiteboard prompts. Structure every answer as: requirements → constraints → design → trade-offs → failure modes.
- **L5 = debugging & incident response** — "your metric does X, what do you check and in what order". The order is graded as much as the content.
- `[Company style: ...]` tags flag questions typical of a particular loop.

**The five facts that carry this entire module.** If you memorise nothing else:
1. BERT is encoder-only — no causal mask → cannot generate.
2. Fine-tune LR is **2e-5** (1e-5–5e-5). Nothing else in this module is as load-bearing.
3. Subword alignment: `word_ids()` → first subword gets the label, everything else gets **`-100`**.
4. Token accuracy is a lie for NER. Use **`seqeval` entity F1**.
5. A fine-tuned BERT serves 1M predictions for **~$0.10**; GPT-4o charges **~$600**.

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. What does BERT stand for, and when was it published?**
- **Answer:** **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. The paper is *"Pre-training of Deep Bidirectional Transformers for Language Understanding"* (Devlin et al.), Google, arXiv October 2018, published May 2019. The course states the publication date as **24 May 2019**.
- **Why asked:** Instant credibility check. Anyone who has genuinely worked with BERT knows the acronym expansion.
- **Trap:** Saying "Bidirectional Encoder Representations from Transformers" but then claiming it has a decoder. The acronym itself contains the answer.

**Q2. Is BERT an encoder, a decoder, or both?**
- **Answer:** **Encoder only.** It is a stack of Transformer encoder blocks with fully bidirectional self-attention. It has no causal mask, no cross-attention, and no decoder in any released variant. BERT-base has 12 encoder layers; BERT-large has 24.
- **Why asked:** This is the single most important architectural fact. It determines every capability and every limitation.
- **Trap:** "It's an encoder-decoder like T5" or "it has a decoder you can enable". Neither exists.

**Q3. Can BERT generate text? Why not?**
- **Answer:** **No.** Generation requires a well-defined next-token distribution `P(x_t | x_<t)`. Decoders get that from a **causal attention mask** (`M[i,j] = -inf for j > i`). BERT's attention mask has no causal component, so every position attends to every other position including later ones. There is no "next", and appending a token would change the representation of all preceding tokens.
- **Why asked:** Tests whether you understand *why*, not just *that*. The follow-up is always "what specifically is missing".
- **Trap:** "It can generate, it's just not good at it." No — the mechanism does not exist.

**Q4. What are BERT's two sizes, and their dimensions?**
- **Answer:** **BERT-base**: 12 layers, hidden 768, 12 attention heads, FFN 3072, 110M params. **BERT-large**: 24 layers, hidden 1024, 16 heads, FFN 4096, 340M params. Both: vocabulary 30,522 (WordPiece), max sequence 512.
- **Why asked:** Basic spec recall. Interviewers use it to check whether you have ever read the config.
- **Trap:** Saying BERT-large has 20 layers (a slip the course itself makes once before correcting to 24).

**Q5. What are the two pretraining objectives of BERT?**
- **Answer:** **MLM (Masked Language Modelling)** — corrupt 15% of tokens and predict the originals — and **NSP (Next Sentence Prediction)** — given sentences A and B, predict whether B followed A. Total loss is `L_MLM + L_NSP`.
- **Why asked:** Foundational. The follow-up is always "which one turned out to be useless?"
- **Trap:** Forgetting that MLM operates only on the 15% selected positions — the other 85% contribute no loss.

**Q6. What is the 80/10/10 rule?**
- **Answer:** Of the 15% of tokens selected for MLM: **80%** are replaced with `[MASK]`, **10%** with a random vocabulary token, **10%** are left unchanged. Loss is computed on all 15% regardless of which corruption was applied.
- **Why asked:** Tests depth beyond "BERT masks tokens". The follow-up is "why bother with the 10%/10%?"
- **Trap:** Saying the 10% random tokens are noise/regularisation only. They exist to force the model to *detect and correct* a corrupted input rather than rely on a `[MASK]` marker.

**Q7. What is `[CLS]`, and why is it used for classification?**
- **Answer:** `[CLS]` is a special token (id **101** in `bert-base-uncased`) prepended to every sequence. Its final hidden state is the pooled sequence representation, and the classification head reads it. It works because (a) it carries no lexical prior, (b) bidirectional attention lets it summarise every token, and (c) the **NSP head was trained on it**, which forced it to become a usable whole-sequence summary.
- **Why asked:** Tests whether you know pooling is not arbitrary.
- **Trap:** "It's called CLS so it's for classification." The naming is a mnemonic; `[CLS]` exists because of NSP.

**Q8. What is `[SEP]` for, and what does a pair input look like?**
- **Answer:** `[SEP]` (id **102**) marks segment boundaries. A pair input is `[CLS] A [SEP] B [SEP]`. Token type ids are 0 for segment A (including `[CLS]`) and 1 for segment B (including the trailing `[SEP]`).
- **Why asked:** Pair tasks (NLI, QA, reranking) are the majority of production encoder work.
- **Trap:** Omitting `token_type_ids` for QA. The model then cannot distinguish question from context and QA F1 drops several points.

**Q9. What is `[PAD]`'s token id, and why is it dangerous in NER?**
- **Answer:** `[PAD]` is id **0**. It is dangerous because in a BIO tag scheme, label id **0 is `O`** — a real, loss-bearing tag. If you pad your *labels* with 0, every pad position trains the model. The label pad value must be **`-100`**.
- **Why asked:** This is the exact bug `DataCollatorForTokenClassification` exists to prevent.
- **Trap:** "Pads are masked so they don't matter." The attention mask stops pads being *attended to*; it does not stop the *loss* on pad positions unless the label is `-100`.

**Q10. What is `[UNK]`, and when should it worry you?**
- **Answer:** `[UNK]` (id **100**) is emitted for any character or subword not in the 30,522-entry WordPiece vocabulary. It should worry you when its **rate exceeds ~1–3%** on your corpus: information is destroyed irreversibly and fine-tuning on a few thousand rows cannot recover it. Measure it before training.
- **Why asked:** Separates people who have shipped domain NLP from people who have only run IMDB.
- **Trap:** "The model handles rare words." It does not — the vocabulary is fixed and `[UNK]` carries no information.

**Q11. What is `[MASK]` and when do you see it?**
- **Answer:** `[MASK]` (id **103**) is the MLM corruption token. **You never see it at fine-tuning time** — your data has no masks. That train/inference mismatch is a known defect, and BERT's own authors worked around it by tuning the pretraining mask rate per downstream task (0.10 for NER, 0.15 for classification, 0.30 for SQuAD).
- **Why asked:** Tests whether you know MLM's weakness, not just its definition.
- **Trap:** Adding `[MASK]` to your own training data "to match pretraining". That is exactly backwards.

**Q12. What is the hard maximum sequence length for BERT, and why?**
- **Answer:** **512 tokens.** It comes from `max_position_embeddings=512`: BERT uses **learned absolute** position embeddings, so index 512 simply does not exist in the table. RoBERTa extends this slightly to 514. ModernBERT (2024) uses RoPE and supports 8,192.
- **Why asked:** A practical constraint that shapes every long-document pipeline.
- **Trap:** "You can just raise `max_position_embeddings` in the config." You can change the number; the new rows are random and the model has never seen those positions.

**Q13. Define fine-tuning as it applies to BERT.**
- **Answer:** Load the pretrained encoder, discard the MLM and NSP heads, attach a **new, randomly-initialised task head**, and train the whole network (or a chosen subset of layers) on labelled data with a small learning rate. The course's phrasing: *"fine-tuning means we are going to be retrain the model ... on our own data"*.
- **Why asked:** Vocabulary check for the module's core operation.
- **Trap:** "Fine-tuning means retraining from scratch on my data." The pretrained weights are the starting point; that is the whole point.

**Q14. What are the three strategies for choosing how much of BERT to fine-tune?**
- **Answer:** (1) **Full fine-tuning** — update every parameter (110M for base) with a small LR. (2) **Selective layer fine-tuning** — freeze the bottom N layers and train the top layers plus the head ("last two layer ... last five layer"). (3) **Head only** — freeze the whole encoder and train just the classification head, which is a linear probe.
- **Why asked:** Tests practical awareness of the compute/quality trade-off.
- **Trap:** Claiming head-only is "basically the same". A linear probe is materially weaker — 3–8 points on most tasks.

**Q15. What is a "head" in this context, mechanically?**
- **Answer:** A `torch.nn.Linear` layer mapping the encoder's hidden size to the number of outputs — `Linear(768, num_labels)` for classification and token classification, `Linear(768, 2)` for QA (one column of start logits, one of end logits). For sequence classification BERT also applies a `tanh` **pooler** to `[CLS]` before the linear layer. It is randomly initialised at fine-tuning time.
- **Why asked:** Tests whether you understand what is actually being trained.
- **Trap:** "The head is pretrained." It is not; that is why training loss starts near `ln(num_labels)`.

**Q16. What loss does `BertForSequenceClassification` compute, and what is `num_labels`?**
- **Answer:** Cross-entropy over the pooled `[CLS]` representation. `num_labels` sets the head's output width and is passed to `from_pretrained(..., num_labels=N)`. Getting it wrong is a silent failure: too few labels truncates or wraps, and omitting it falls back to **2**.
- **Why asked:** The most common configuration error in the whole module.
- **Trap:** "If I omit `num_labels` it will infer the count from my data." It will not — it defaults to 2 for BERT.

**Q17. What is `problem_type` and what are its values?**
- **Answer:** A model-config key that selects the loss function. `single_label_classification` (default) → softmax + cross-entropy over mutually exclusive classes. `multi_label_classification` → **BCEWithLogitsLoss** over independent sigmoids. `regression` → MSE on a single-logit output.
- **Why asked:** Multi-label classification is extremely common in production and is silently wrong by default.
- **Trap:** Using the default for multi-label data. The model then learns a softmax over labels that can co-occur, and recall on co-occurring labels collapses.

**Q18. What is the BIO tagging scheme?**
- **Answer:** **B-X** begins an entity of type X, **I-X** continues an entity of type X, **O** is outside any entity. So "John lives in London" → `B-PER O O B-LOC`. BILOU/BIOES adds `L-X` (last), `U-X`/`S-X` (unit/single), `E-X` (end) to make boundaries unambiguous.
- **Why asked:** The prerequisite for any NER conversation.
- **Trap:** Not knowing that `seqeval` defaults to the **IOB2** scheme, so feeding it BILOU data without `scheme="BILOU"` silently mis-scores.

**Q19. What does `word_ids()` do, and what class of tokenizer provides it?**
- **Answer:** It returns a list mapping each **token** index to its source **word** index, with `None` for special tokens. It exists **only on fast tokenizers** (`BertTokenizerFast`, `AutoTokenizer(use_fast=True)`); calling it on the slow `BertTokenizer` raises `AttributeError`.
- **Why asked:** It is the alignment primitive for all token-classification work.
- **Trap:** Using the slow tokenizer in a tutorial that uses `word_ids()`, then spending an hour on an `AttributeError`.

**Q20. What is `-100` in a label tensor?**
- **Answer:** `torch.nn.CrossEntropyLoss`'s default `ignore_index`. Positions labelled `-100` contribute **zero loss and zero gradient**, and are also excluded from the loss denominator. It is the mechanism for masking special tokens and continuation subwords.
- **Why asked:** The remedy for the #1 silent bug in this module.
- **Trap:** Confusing `-100` with the pad token id `0`. Using 0 trains the model on pads and on specials.

**Q21. What is `seqeval` and why is it preferred over `sklearn` F1 for NER?**
- **Answer:** `seqeval` computes **entity-level** scores: a predicted span counts as correct only if its type **and both boundaries** match gold exactly. `sklearn`'s F1 operates on individual token labels, which gives partial credit for half-correct entities and makes a model that predicts all `O` score ~0.85–0.95.
- **Why asked:** Directly tests whether the candidate has ever evaluated an NER model honestly.
- **Trap:** "F1 is F1." Token-level and entity-level F1 differ by 5–15 points on real NER data.

**Q22. What is an "extractive" QA model?**
- **Answer:** A model that answers by selecting a contiguous **substring of the provided context**, modelled as two distributions over token positions — a **start** and an **end**. It cannot produce text that is not in the input, which makes hallucinating the answer span architecturally impossible.
- **Why asked:** The distinction from generative/RAG QA is a common trip-up.
- **Trap:** Saying "extractive QA generates the answer from the context". It selects, it does not generate.

**Q23. What is `offset_mapping` and why does QA need it?**
- **Answer:** A `(start, end)` **character-offset** pair per token, returned by a fast tokenizer when you pass `return_offsets_mapping=True`. Your gold answer is a character span (`answer_start`); the model needs **token indices**. Offsets are the only correct bridge between the two. You must `pop()` it out before the forward pass because it is not a model input.
- **Why asked:** The QA equivalent of `word_ids()` — the alignment primitive.
- **Trap:** Forgetting to `pop` and getting a forward-pass error, or never requesting it and silently training on `(0, 0)` labels.

**Q24. What is `doc_stride`?**
- **Answer:** The **overlap** between consecutive windows when a context exceeds `max_length`. With `max_length=384, doc_stride=128`, each window advances 256 tokens. It exists so that an answer straddling a window boundary is still fully contained in at least one window. Rule of thumb: **`doc_stride` must exceed the longest answer you expect.**
- **Why asked:** Reveals whether the candidate has run QA on real documents or only on SQuAD paragraphs that happen to fit.
- **Trap:** Confusing `doc_stride` with `max_length`. `max_length - doc_stride` is the advance.

**Q25. What is `max_answer_length`?**
- **Answer:** A post-processing cap on the number of tokens a predicted span may contain. SQuAD's convention is **30**; for your domain, set it to the 99th percentile of gold answer length. It is a guard against the model producing an end index far from the start index.
- **Why asked:** Small detail that shows hands-on QA experience.
- **Trap:** Believing it constrains the model. It only constrains decoding.

**Q26. What is `n_best_size`?**
- **Answer:** The number of top-scoring start and end candidates kept per window before enumerating and scoring spans. The standard is **20** starts × 20 ends = up to 400 candidate spans, filtered by `start <= end`, `length <= max_answer_length`, and "not entirely inside the question". Setting it to 1 is a pure greedy argmax and loses roughly 5–10 F1.
- **Why asked:** Distinguishes `pipeline("question-answering")` users from people who wrote the decoder.
- **Trap:** "It's the number of answers returned." That is `top_k`.

**Q27. What is a data collator, and what is the `Trainer` default?**
- **Answer:** A callable that turns a list of dataset items into a padded batch of tensors. `Trainer`'s default is `DataCollatorWithPadding` when a tokenizer is supplied (or `DefaultDataCollator`, which only stacks, when it is not).
- **Why asked:** Collator choice is the second-most-common silent bug in this module.
- **Trap:** "DefaultDataCollator is the default." It is only used when no tokenizer is available.

**Q28. Name the correct collator for each of the three tasks.**
- **Answer:** Sequence classification → `DataCollatorWithPadding`. Token classification (NER) → **`DataCollatorForTokenClassification`** (pads labels with `-100`). QA → `DataCollatorWithPadding` (the labels are scalars, not per-token). Seq2seq → `DataCollatorForSeq2Seq`.
- **Why asked:** Fast filter for whether the candidate has built an NER pipeline.
- **Trap:** Using `DataCollatorWithPadding` for NER because "it works for classification". It runs, and it silently trains on pads.

**Q29. What learning rate do you use to fine-tune BERT, and why so small?**
- **Answer:** **2e-5**, with a usable range of 1e-5 to 5e-5. Pretrained BERT weights sit in a basin that minimises MLM+NSP over billions of tokens; a downstream dataset of a few thousand rows defines a much narrower basin. A large LR (1e-3 and above) moves the weights out of the pretrained basin within ~100 steps — **catastrophic forgetting** — and you will see the *training* loss fall while eval loss rises.
- **Why asked:** The highest-value hyperparameter in the module.
- **Trap:** "The default 5e-5 in `TrainingArguments` is fine." It is a *higher* default than most encoder recipes want; 2e-5 is the safer starting point.

**Q30. How many epochs do you fine-tune a BERT encoder for?**
- **Answer:** **2–4.** On small datasets (<10k rows) performance typically peaks at epoch 2–3 and degrades after. Use `eval_strategy="epoch"`, `load_best_model_at_end=True`, and `metric_for_best_model` set to your task metric so you serve the best checkpoint, not the last one.
- **Why asked:** Tests whether the candidate watches eval loss or just runs a fixed number of epochs.
- **Trap:** "More epochs, more better." The classic encoder overfitting signature is F1 peaking at epoch 2 and falling at epoch 5.

**Q31. Why is BERT still used in 2026 rather than an LLM?**
- **Answer:** Cost and latency, at volume. A fine-tuned BERT-base serves **1M predictions for roughly $0.09–0.14** on GPU or CPU-int8 hosting; GPT-4o charges **~$600** and GPT-4o-mini **~$36** for the same 1M. Latency is 10–30 ms on GPU or 15–60 ms CPU-int8 versus 400–2,000 ms for an API. It is also self-hostable, deterministic, and version-pinnable. The course states the argument directly: *"this BERT is still being used for some enterprise NLP level of task for fast and the cost effectiveness"*.
- **Why asked:** The most likely "why does this module exist" question. A candidate who cannot do the arithmetic looks like they have only used notebooks.
- **Trap:** "Nobody uses BERT any more." Wrong — it is the standard production choice for closed-label NLP at volume.

**Q32. Name four places BERT is still used today.**
- **Answer:** (1) **Retrieval/embeddings** — dense vectors for a RAG index (Sentence-Transformers, MiniLM, E5). (2) **Enterprise NLP at scale** — NER, sentiment, intent classification, spam detection. (3) **Hybrid RAG pipelines** — BERT retriever feeding an LLM generator. (4) **On-device / edge inference** — quantized DistilBERT/MiniLM in mobile and offline apps. A fifth: **reranking** — a cross-encoder scoring `(query, passage)` pairs.
- **Why asked:** Tests whether the candidate can place an old architecture in a modern stack.
- **Trap:** Listing only "classification". The retriever role in RAG is now the single largest deployment of BERT-family models.

---

## Level 2 — Applied & Implementation

**Q33. Walk me through fine-tuning BERT for sentiment classification. Give me the actual code shape.**
- **Answer:**
```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=256)   # NO padding here
ds = ds.map(tokenize, batched=True, remove_columns=["text"])
ds = ds.rename_column("label", "labels")        # mandatory: HF wants 'labels'
ds = ds.select_columns(["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}      # do this BEFORE saving

args = TrainingArguments(
    output_dir="./out", num_train_epochs=3, per_device_train_batch_size=16,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1", fp16=True, seed=42, report_to="none")

Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics).train()
```
- **Why asked:** The single most likely "write it for me" question. The graded details are `rename_column`, no static padding, `num_labels`, and `compute_metrics`.
- **Trap:** Using `padding="max_length"` in `tokenize()` and then also passing a collator. Padding twice wastes 2–4× compute.

**Q34. Why must you `rename_column("label", "labels")`?**
- **Answer:** Hugging Face models expect the target column to be named **`labels`** (plural). `Trainer` looks for that key in each batch and passes it to the model, which computes the loss internally. Leaving it as `label` means the model gets **no labels**, returns no loss, and `Trainer` silently trains on... nothing useful — you get an error or a zero-loss run. The course flags this explicitly: *"the hugging face required this specific name that is labels"*.
- **Why asked:** A one-line detail that separates people who have run the code from people who have read about it.
- **Trap:** Renaming to `label` (singular) "for consistency with my CSV". It must be `labels`.

**Q35. What does `trainer.evaluate()` return for a classification model with no `compute_metrics`?**
- **Answer:** Only `eval_loss`, `eval_runtime`, `eval_samples_per_second`, `eval_steps_per_second`, and `eval_jit_compilation_time`. **No accuracy, no F1** — you must supply `compute_metrics`. The course demonstrates exactly this: it prints the metrics dict and gets loss and throughput only.
- **Why asked:** Catches candidates who report "accuracy" without ever having computed it.
- **Trap:** Assuming `Trainer` reports accuracy by default. It does not.

**Q36. Write `compute_metrics` for a binary sentiment task with class imbalance.**
- **Answer:**
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1)          # rare class = positive
    return {
        "accuracy":   accuracy_score(labels, preds),
        "precision":  p,
        "recall":     r,
        "f1_binary":  f1,                                       # the metric to select on
        "f1_macro":   f1_score(labels, preds, average="macro"),
        "mcc":        matthews_corrcoef(labels, preds),
    }
```
- **Why asked:** Tests three things at once: `argmax` over logits, the right averaging mode, and awareness of imbalance.
- **Trap:** `average="weighted"`. On a 2%-positive dataset, weighted F1 tracks accuracy and will read ~0.98 while the model has 0.0 recall on the class you care about.

**Q37. Walk me through the NER data path, including alignment.**
- **Answer:**
```python
def tokenize_and_align(batch):
    enc = tok(batch["tokens"], truncation=True, max_length=128,
              is_split_into_words=True)              # tokens are ALREADY words
    all_labels = []
    for i, labels in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)       # fast tokenizer only
        prev, out = None, []
        for w in word_ids:
            if w is None:
                out.append(-100)                     # [CLS], [SEP], [PAD]
            elif w != prev:
                out.append(labels[w])                # FIRST subword gets the tag
            else:
                out.append(-100)                     # continuation subword
            prev = w
        all_labels.append(out)
    enc["labels"] = all_labels
    return enc
```
plus `DataCollatorForTokenClassification(tok)` at the `Trainer`. Note the label list has the **same length as `input_ids`**, because `-100` entries are inserted for specials and continuations.
- **Why asked:** The core technical skill of the module. Interviewers grade the `is_split_into_words`, the `None` branch, and the `w != prev` branch.
- **Trap:** Omitting `is_split_into_words=True`. The tokenizer then treats each pre-split word as a *string* to tokenize and returns a batch of `(num_words, subwords)` — shape explosion and total misalignment.

**Q38. Why does the first subword get the label and not all of them?**
- **Answer:** Two reasons. (1) **Avoid double-counting**: `London` is one entity; if `Lon` and `##don` both carry `B-LOC`, the loss counts the entity twice and entity-level F1 is computed on a span that does not exist. (2) **Avoid teaching garbage transitions**: a continuation piece has no meaningful BIO state of its own — `##don` cannot "begin" anything. The alternative policy, `label_all_tokens=True`, propagates `B-X` → `I-X` on continuations; that is valid for BILOU schemes but must be applied identically at inference.
- **Why asked:** Tests whether you understand *why* the convention exists, not just that it does.
- **Trap:** "Continuation subwords get the same label — they're part of the same word." That is the bug.

**Q39. What breaks if you use `DataCollatorWithPadding` for NER instead of `DataCollatorForTokenClassification`?**
- **Answer:** `DataCollatorWithPadding` pads the `labels` tensor with `tokenizer.pad_token_id`, which is **0**. In a BIO scheme, `0` is `O` — a valid, loss-bearing label. So (a) the loss includes every pad position and is no longer comparable across batches with different padding; (b) the model is trained to emit `O` on padding, biasing the boundary against rare entity types; (c) any evaluation that counts label occurrences is wrong. `DataCollatorForTokenClassification` pads with `label_pad_token_id=-100` by default, which `CrossEntropyLoss` ignores entirely.
- **Why asked:** The precise failure mode of a mistake that does not crash.
- **Trap:** "Padding is masked by `attention_mask`, so it's fine." The attention mask stops pads being *attended to*; it does not touch the loss on pad positions.

**Q40. How do you build a QA training example from a `(question, context, answer_start, answer_text)` tuple?**
- **Answer:**
```python
enc = tok(question, context, truncation="only_second", max_length=384,
          stride=128, return_overflowing_tokens=True,
          return_offsets_mapping=True, padding="max_length")

offset_mapping = enc.pop("offset_mapping")
start_positions, end_positions = [], []
for i, offsets in enumerate(offset_mapping):
    # map each window back to its source example to find the right answer_start
    seq_ids = enc["overflow_to_sample_mapping"][i]
    a_start = answers[seq_ids]["answer_start"][0]
    a_end   = a_start + len(answers[seq_ids]["text"][0])
    # positions for the current window, defaulting to the CLS index (0)
    cls_idx = enc["input_ids"][i].index(tok.cls_token_id)
    s = cls_idx
    if not (offsets[cls_idx][0] <= a_start < offsets[cls_idx][1]):
        s = next((j for j, (o1, o2) in enumerate(offsets)
                  if o1 <= a_start < o2), cls_idx)
    e = next((j for j, (o1, o2) in enumerate(offsets)
              if o1 < a_end <= o2), cls_idx)
    start_positions.append(s)
    end_positions.append(e)
```
The key points: `truncation="only_second"` so the question is never cut; `return_overflowing_tokens` produces **more rows than inputs**; and the default when the answer is not in this window is the **`[CLS]` index**, not `0` blindly (they coincide, but the intent matters).
- **Why asked:** This is the hardest data-engineering step in the module. Most candidates have only used `pipeline`.
- **Trap:** Setting `start_positions`/`end_positions` to `0` for out-of-window answers by accident rather than by design — that teaches the model to point at `[CLS]` for everything it cannot find.

**Q41. What does the QA model output, and what is the loss?**
- **Answer:** `start_logits` and `end_logits`, each of shape `(batch, seq_len)` — two independent softmax distributions over the **same** token positions. The loss is `½[CE(start_logits, start_positions) + CE(end_logits, end_positions)]`. There is **no constraint** that `argmax(start) ≤ argmax(end)`; that is enforced in post-processing.
- **Why asked:** Tests the difference between what the model produces and what the decoder must do.
- **Trap:** Assuming a single joint distribution over spans. It is two marginal distributions, which is exactly why `n_best` span enumeration exists.

**Q42. How do you handle unanswerable questions (SQuAD 2.0)?**
- **Answer:** The `[CLS]` position is the no-answer token. In training, unanswerable examples are labelled with `start = end = cls_index`. At inference, the model's score for `[CLS]` is compared against the best span score; if `best_span_score < best_cls_score + threshold`, return "no answer". The important detail is that `[CLS]`'s logit is not directly comparable to word positions, so implementations add a learned or tuned `cls_logit` offset. `pipeline("question-answering", handle_impossible_answer=True)` does this for you.
- **Why asked:** Tests whether the candidate has actually handled abstention — a production requirement.
- **Trap:** "Just check if the confidence is low." Without the `[CLS]` baseline, a model trained on SQuAD 1.1 has no notion of "no answer" at all.

**Q43. Your QA inference returns an empty string. Diagnose.**
- **Answer:** In order: (1) **`argmax` selected a `[PAD]` position** — you did not filter `attention_mask == 0`. (2) **`end_idx < start_idx`** and your repair collapsed it to one token that decoded to whitespace. (3) **The `(0,0)` collapse** — the model learned to point at `[CLS]` because your training labels defaulted to 0 on truncated examples. (4) **The answer is genuinely truncated away** — the context exceeded 512 tokens and you set no `doc_stride`. Check by printing `(start, end)` and the raw logits around them.
- **Why asked:** A real debugging question with a specific answer ordering.
- **Trap:** Jumping straight to "the model needs more training". All four causes are data/pipeline bugs, not model bugs.

**Q44. What is `is_split_into_words=True` and when do you need it?**
- **Answer:** It tells the tokenizer that the input is a **pre-tokenized list of words** rather than a raw string, so it tokenizes each word independently and records the word boundaries for `word_ids()`. You need it for every token-classification task where your labels are per-word (NER, POS, chunking).
- **Why asked:** The single most common shape error in NER code.
- **Trap:** Passing the pre-split list as a list *without* the flag. Hugging Face then treats it as a batch of strings and returns a nested shape.

**Q45. What is the difference between `input_ids`, `attention_mask`, and `token_type_ids`?**
- **Answer:** `input_ids` are vocabulary indices. `attention_mask` is 1 for real tokens and 0 for `[PAD]` — it is added to the attention scores as `-inf` at masked positions. `token_type_ids` (segment ids) are 0 for the first segment (`[CLS] A [SEP]`) and 1 for the second (`B [SEP]`). All three are summed into the input representation as embeddings.
- **Why asked:** Basic but load-bearing, especially for pair tasks.
- **Trap:** Dropping `token_type_ids` for QA. The model then cannot tell question tokens from context tokens.

**Q46. What is the `Trainer` and what does it handle for you?**
- **Answer:** A training loop wrapper: batching via the collator, the optimizer and LR scheduler, gradient clipping (`max_grad_norm`), mixed precision, gradient accumulation, evaluation and checkpointing on a schedule, logging, distributed/`accelerate` launch, best-model restoration, and `push_to_hub`. You supply model, args, datasets, an optional collator, and an optional `compute_metrics`.
- **Why asked:** Tests whether the candidate can also write the loop by hand when `Trainer` does not fit.
- **Trap:** Not knowing that `Trainer` applies `max_grad_norm=1.0` and `weight_decay=0.01` by default, so the defaults are already reasonable.

**Q47. Write the equivalent training loop by hand. What must you not forget?**
- **Answer:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # BEFORE step
        optimizer.step()
        scheduler.step()                                          # EVERY batch
```
Must-not-forgets: `optimizer.zero_grad()`; `clip_grad_norm_` **after** backward and **before** step; `scheduler.step()` per batch (not per epoch); `model.eval()` + `torch.no_grad()` in evaluation; and moving tensors to `device`. Use a `GradScaler` if you enable fp16 manually.
- **Why asked:** `Trainer` hides these, so this question finds out whether you know what it is hiding.
- **Trap:** Stepping the scheduler per epoch. With `get_linear_schedule_with_warmup` built for `total_steps`, per-epoch stepping gives a wildly wrong LR curve.

**Q48. Why `AdamW` and not `Adam` with `weight_decay`?**
- **Answer:** `AdamW` applies **decoupled** weight decay — it shrinks the weights directly (`w ← w − lr·λ·w`) instead of adding `λw` to the gradient. Folding decay into the gradient (classic Adam + L2) interacts badly with Adam's per-parameter learning-rate scaling, so parameters with large gradient variance get effectively less regularisation. `adamw_torch` is the modern `Trainer` default.
- **Why asked:** Tests whether "weight decay" is a magic number or a mechanism.
- **Trap:** "They're the same thing." `torch.optim.Adam(weight_decay=0.01)` is **L2-in-gradient**, a different algorithm with different behaviour.

**Q49. Why is gradient clipping at 1.0 important for BERT fine-tuning?**
- **Answer:** Fine-tuning has occasional batches with very large gradients (a rare-but-confident wrong prediction, an outlier token). Without clipping, one such batch can move all 110M weights far enough to disrupt the pretrained representation. Clipping the global norm to 1.0 bounds the step size, which is what makes the training robust enough that both LR 2e-5 and LR 5e-5 work. It is the reason a single bad batch does not ruin the run.
- **Why asked:** A cheap check for whether the candidate understands training stability.
- **Trap:** Disabling it "because the loss looks fine". The failure is intermittent and looks like a random bad epoch.

**Q50. What is warmup and why 10%?**
- **Answer:** A linear ramp of the learning rate from 0 (or a small value) to the peak over the first N steps. It protects the pretrained weights from a cold, full-size first step, when the randomly-initialised head produces large, noisy gradients that propagate back through the whole encoder. 10% of total steps is the standard encoder value — with 1,000 total steps that is 100 warmup steps. BERT's original pretraining used 10,000 warmup steps out of 1,000,000 (1%).
- **Why asked:** Tests the interaction between warmup and total steps — a common misconfiguration.
- **Trap:** Setting a fixed `warmup_steps=500` on a 125-step run. The LR never reaches its peak and the model underfits.

**Q51. Why dynamic padding, and how much does it buy?**
- **Answer:** Static `padding="max_length"` pads every sequence to the global maximum. Dynamic padding (a collator) pads each batch only to its own longest sequence. On WikiAnn English (mean ~26 tokens) padded to 512, that is a **~20× compute reduction** — 4,096 padded tokens per batch versus ~208. At IMDB's mean of ~130 subwords with `max_length=256` it is roughly **2×**. Cost: one argument (`data_collator=...`) and removing `padding` from the tokenizer call.
- **Why asked:** The cheapest large win in the module, and a marker of practical experience.
- **Trap:** Leaving `padding="max_length"` in the tokenizer *and* passing a collator. The collator then has nothing to do.

**Q52. Name the `TrainingArguments` you would change from the defaults for a BERT fine-tune.**
- **Answer:** `learning_rate=2e-5` (default is 5e-5), `num_train_epochs=3` (default 3 is fine), `per_device_train_batch_size=16` (default 8), `warmup_ratio=0.1`, `weight_decay=0.01` (already default), `max_grad_norm=1.0` (already default), `optim="adamw_torch"`, `fp16=True` or `bf16=True`, `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1"`, `save_total_limit=2`, `seed=42`, `report_to="tensorboard"`, `dataloader_num_workers=4`.
- **Why asked:** A "do you have a recipe" question. The graded items are the LR, warmup, best-model restoration, and fp16.
- **Trap:** Leaving `report_to="none"` and then wondering why `tensorboard --logdir` shows nothing. The course hits exactly this.

**Q53. How do you save a fine-tuned model, and what is the difference from a checkpoint?**
- **Answer:** `trainer.save_model("./dir")` writes the **complete, loadable** artefact: `model.safetensors` (or `pytorch_model.bin`), `config.json`, and generation config. A **checkpoint** (`./dir/checkpoint-500`) is Trainer-managed state — model plus optimizer, scheduler, and RNG state — used to resume training, and it is not a distribution artefact. Critically, `save_model` does **not** save the tokenizer: call `tokenizer.save_pretrained("./dir")` too, or you have weights nobody can correctly load.
- **Why asked:** The most common packaging mistake, and it fails loudly at load time but silently if you load *a different* tokenizer.
- **Trap:** Shipping the checkpoint directory as the model.

**Q54. How do you push a fine-tuned model to the Hugging Face Hub?**
- **Answer:**
```python
from huggingface_hub import notebook_login, whoami
notebook_login()                                  # needs a WRITE-scoped token
print(whoami())                                   # verify the scope before pushing
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}   # BEFORE pushing
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
tokenizer.push_to_hub("user/my-bert-imdb")
model.push_to_hub("user/my-bert-imdb")            # or trainer.push_to_hub(...)
```
Both model and tokenizer must go into the **same repo**; a common failure is pushing the model to one repo and the tokenizer to another.
- **Why asked:** Tests an end-to-end workflow, and whether the candidate sets `id2label` before shipping.
- **Trap:** Pushing without `id2label`. Every downstream consumer then sees `LABEL_0`/`LABEL_1` and has to guess the class order.

**Q55. Why use `BertTokenizerFast` over `BertTokenizer`?**
- **Answer:** The fast tokenizers are Rust-backed (`tokenizers` library) and are typically **5–20× faster** on large corpora; more importantly they expose `word_ids()`, `offset_mapping`, and `overflow_to_sample_mapping`, which are **required** for token classification and QA alignment. The slow tokenizer has none of these.
- **Why asked:** Small detail, large consequence.
- **Trap:** Using the slow tokenizer because the tutorial imported it, then hitting `AttributeError: 'BertTokenizer' object has no attribute 'word_ids'`.

**Q56. How do you handle a document longer than 512 tokens for classification?**
- **Answer:** Four options, in order of preference: (1) **Chunk and aggregate** — split into 512-token windows with 64–128 token overlap, run the classifier on each, then aggregate (mean of probabilities, max, or a learned attention pool). (2) **Truncate intelligently** — take the head plus the tail; for support tickets the decisive content is usually early. (3) **Use a long-context encoder** — ModernBERT (8,192), Longformer, or BigBird. (4) **Two-stage** — retrieve the relevant passage with a retriever, then classify only the passage.
- **Why asked:** Tests whether the 512 limit is an abstraction or a constraint you have actually hit.
- **Trap:** "Increase `max_length` to 2048." It will error or silently ignore it, because the position-embedding table has 512 rows.

**Q57. How do you decide `max_length` for a new dataset?**
- **Answer:** Look at the **token-length distribution** of the training data, not the word count. Compute `len(tokenizer(text)["input_ids"])` for every row, take the 95th–99th percentile, and round up to a multiple of 8. IMDB: p95 ≈ 500+ subwords, so 256 truncates a real fraction — choose 256 for speed and accept it, or 512 for accuracy. WikiAnn: p99 ≈ 50, so 512 wastes 10–20× compute. SQuAD QA: 384 with `doc_stride=128` is the community standard.
- **Why asked:** Tests whether the number is chosen or copied.
- **Trap:** Setting `max_length` to the model's maximum "to be safe". That is the single most common cause of needlessly slow encoder training.

**Q58. How do you count the layers of a loaded BERT and confirm it matches the paper?**
- **Answer:**
```python
from transformers import AutoModel
m = AutoModel.from_pretrained("bert-base-uncased")
print(m.config.num_hidden_layers,   # 12
      m.config.hidden_size,         # 768
      m.config.num_attention_heads, # 12
      m.config.max_position_embeddings)  # 512
print(len(m.encoder.layer))         # 12
print(sum(p.numel() for p in m.parameters()) / 1e6)   # ~109.5 (110M incl. head/pooler)
```
The course does exactly this — iterating `model.bert.encoder.layer` and printing each `BertLayer` to demonstrate the 12-layer stack.
- **Why asked:** Fast way to see if someone has explored the object model or only called `.train()`.
- **Trap:** Expecting exactly 110,000,000 parameters. The encoder alone is ~109.5M; the reported 110M includes pooler and head.

**Q59. What does the `[CLS]` pooler actually do, and does `BertForTokenClassification` use it?**
- **Answer:** `BertPooler` is `Linear(768, 768) + tanh` applied to the `[CLS]` hidden state. `BertForSequenceClassification` uses it (then a dropout and the classification linear layer). `BertForTokenClassification` and `BertForQuestionAnswering` **do not** — they apply their heads to every token position. You can also disable it with `add_pooling_layer=False`.
- **Why asked:** Tests the object model at a level that only comes from debugging a head.
- **Trap:** Assuming every BERT head reads `[CLS]`. Only sequence classification does.

**Q60. How would you fine-tune BERT on a 5-class problem with 400 training examples?**
- **Answer:** Honestly, probably do not. 400 rows / batch 16 = 25 steps/epoch; even 10 epochs is 250 steps, far below the ~1,000-step floor, and the head will not converge. Options in order: (1) **SetFit** with a sentence transformer — 80 examples per class is plenty, the head refits in seconds, and it is designed for this regime. (2) A **linear probe on frozen embeddings** plus logistic regression. (3) If you must fine-tune BERT: freeze the bottom 8 layers, LR 1e-4 for the top layers and head, `num_train_epochs=10–20`, strong weight decay, and heavy stratification with cross-validation. (4) Prompt an LLM.
- **Why asked:** Tests whether the candidate will actually push back on a bad setup — a strong senior signal.
- **Trap:** Just running BERT fine-tuning with defaults and reporting whatever comes out. At 400 rows you will get a number that is mostly noise.

**Q61. How do you freeze layers in BERT via Hugging Face, and what exactly does it save?**
- **Answer:**
```python
for p in model.bert.embeddings.parameters():
    p.requires_grad = False
for layer in model.bert.encoder.layer[:-2]:    # freeze all but the top 2
    for p in layer.parameters():
        p.requires_grad = False
# the classifier head is trainable by default
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
```
Frozen layers still run **forward** (so FLOPs and activations are unchanged), but they need no gradients and no AdamW state — so you save `8 bytes × frozen params` (2 fp32 moments) plus their gradient buffers. Freezing 10 of 12 layers of BERT-base cuts trainable params from 110M to ~15M and optimizer memory from ~880 MB to ~120 MB.
- **Why asked:** Tests the distinction between compute savings (none) and memory savings (large).
- **Trap:** "Freezing makes training faster." It makes it *use less memory*, which may let you raise the batch size or sequence length — that is where the speed comes from.

**Q62. What is layer-wise (discriminative) learning rate decay, and why use it?**
- **Answer:** Assign a smaller LR to the bottom layers and a larger one to the top layers, typically `lr × decay^(L - l)` with `decay` in 0.65–0.95. The intuition (from ULMFiT, popularised for BERT in the original fine-tuning literature): bottom layers encode general, pretrained linguistic features that should barely change; top layers encode task-specific features that should adapt. It helps when your task is far from pretraining, and it is a common Kaggle-winning trick. Hugging Face does not support it natively — you build parameter groups manually.
- **Why asked:** Tests depth beyond the default recipe.
- **Trap:** Applying it when you have plenty of data and a task close to pretraining. There it usually does nothing or slightly hurts.

**Q63. How do you make a BERT fine-tune reproducible?**
- **Answer:** Fix every source of randomness: `TrainingArguments(seed=42, data_seed=42, full_determinism=True)`, `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.cuda.manual_seed_all`; set `np.random.choice(..., random_state=42)` in your own sampling; pin library versions (`transformers`, `torch`, `datasets`); freeze the exact train/test split to disk rather than regenerating it; and record the git SHA of the training script plus a hash of the dataset. Expect a residual ±0.3–0.8 point variance on GPU from nondeterministic kernels even with all seeds fixed.
- **Why asked:** A production-readiness question. The course's own notebook fails this: `np.random.choice` is unseeded.
- **Trap:** "Setting `seed=42` is enough." GPU atomics and cuDNN benchmarking remain nondeterministic unless you set `torch.use_deterministic_algorithms(True)` — which costs throughput.

**Q64. What is `label_smoothing_factor` and when do you use it?**
- **Answer:** A `TrainingArguments` flag that replaces one-hot targets with `(1-ε)` on the true class and `ε/(C-1)` elsewhere. It prevents the model from driving logits to ±infinity on small datasets, and it improves calibration. Typical values 0.0–0.1; the default is 0.0. For encoder fine-tuning on <10k rows it is a cheap 0.2–0.5 point win and reduces overconfidence. Do not use it for token classification (the `-100` masking and per-token targets make it less clean) or when you need exactly-matching probabilities.
- **Why asked:** A regularisation detail that separates recipe-followers from practitioners.
- **Trap:** Setting 0.2 on a small dataset. It starts to underfit because the target signal is genuinely weaker.

---

## Level 3 — Advanced, Internals & Theory

**Q65. Why is NSP considered a failed objective? Be specific about the mechanism.**
- **Answer:** NSP's negatives are drawn from a **different document**, so a topic-shift detector solves it without learning anything about discourse. Given `A = "the cat sat on the mat"` and `B = "the FED raised rates"`, `NotNext` is obvious from topic alone. Consequences: (a) the objective is nearly free, so it provides little gradient signal; (b) worse, it *shapes* the `[CLS]` representation toward a topic-detection feature, which is not what downstream classifiers want. RoBERTa's ablation removed NSP and downstream scores improved slightly; ALBERT replaced it with **SOP** (distinguish `A-B` from `B-A`, same document, no topic shift available), which does carry useful ordering signal; ELECTRA, SpanBERT, and DeBERTa all dropped NSP.
- **Why asked:** Tests the general principle "an auxiliary objective only helps if it cannot be solved by a shortcut" — a research-maturity signal.
- **Trap:** "NSP was removed because it was too hard." It was too *easy*, and for the wrong reason.

**Q66. Why does MLM use a random token 10% of the time? It seems to add noise.**
- **Answer:** The purpose is to force the model to **detect and correct** a corrupted input, rather than to rely on a marker. Concretely: without the random-token case, the model learns "the positions I must reason about are exactly the positions containing the literal `[MASK]` token, and everywhere else I may copy my input." At fine-tuning time `[MASK]` never appears, so the first rule never fires and the model's representations are built by a network that has never had to encode a *wrong* token. The random 10% makes "the input at this position may be wrong" a property of the task rather than of a special vocabulary item, so the representation at every position must be robust. The 10% *unchanged* serves the complementary purpose: it guarantees some loss-bearing positions see a fully clean context, which keeps the model from learning to distrust everything.
- **Why asked:** The follow-up to the 80/10/10 question. Most candidates know the ratio and not the reason.
- **Trap:** "It's regularisation." It is a *distribution-matching* fix, not a noise-injection regulariser, and the distinction shows up in the fact that BERT's authors tuned mask rates per task.

**Q67. ELECTRA's replaced-token detection — how does it differ from MLM, and what does it buy?**
- **Answer:** A small **generator** MLM corrupts 15% of tokens with *plausible* replacements (not `[MASK]`). A **discriminator** then classifies **every** token as "original" or "replaced" — a binary task over 100% of positions rather than a 30k-way task over 15%. Three consequences: (1) **dense signal** — every token contributes loss, so the same compute yields far more gradient; (2) **no `[MASK]` mismatch** — the discriminator never sees a special token at pretraining, so the train/test discrepancy vanishes; (3) **discriminator-only inference** — you throw away the generator, so the deployed model is small. ELECTRA-base matches RoBERTa-base with roughly **1/4 of the pretraining compute**; ELECTRA-small (14M discriminator) is one of the best quality-per-parameter encoders available.
- **Why asked:** Tests knowledge of the generation of encoders after BERT, and the compute-efficiency argument.
- **Trap:** "It's MLM with a different loss." The task is binary classification over all positions, and the objective change is what removes the `[MASK]` artifact.

**Q68. Dynamic vs static masking — what is the difference and which wins?**
- **Answer:** **Static masking** (BERT's reference implementation) pre-processes the corpus into a fixed number of masked copies — typically 10 — so the same sequence sees the same mask on every epoch. **Dynamic masking** (RoBERTa) generates a fresh mask every time the sequence is fed, so over 40 epochs a sequence sees 40 distinct masks. Dynamic masking gives ~4× more distinct training signals per epoch for the same data, and RoBERTa's ablation found it comparable-to-slightly-better. The reason BERT used static masking is an implementation artifact of Google's TPU data pipeline (the masks had to be materialised before training), not a design decision.
- **Why asked:** Distinguishes people who have read the RoBERTa paper from people who have only read the BERT paper.
- **Trap:** "Static is better because the masks are consistent." Consistency is exactly the problem — it means the model sees the same corrupted sequence repeatedly.

**Q69. What is the difference between whole-word masking and subword masking, and which should you use?**
- **Answer:** **Subword masking** selects individual WordPiece pieces, so `London` can be masked as `Lon` + `[MASK]`, letting the model predict `##don` from its own prefix — a shortcut that teaches morphology, not context. **Whole-word masking** selects a *word* and masks all of its pieces together, so the model must use cross-word context. BERT's original paper and released checkpoints are subword-level (Google later published `*-wwm` variants); RoBERTa makes whole-word masking the default; SpanBERT goes further and masks **contiguous spans** with a geometric length distribution (mean 3.3) plus a span-boundary objective. For NER and QA, span/whole-word masking pretraining transfers better.
- **Why asked:** Tests familiarity with the pretraining-detail layer of the family tree.
- **Trap:** Assuming `bert-base-uncased` uses whole-word masking. It does not.

**Q70. Derive BERT-base's parameter count.**
- **Answer:**
```
Token embeddings     30,522 × 768          = 23.44M
Position embeddings     512 × 768          =  0.39M
Token-type embeddings     2 × 768          =  0.0015M
Per encoder layer:
  attention Q,K,V,O   4 × 768 × 768        =  2.36M
  FFN                 2 × 768 × 3072       =  4.72M
  LayerNorms          4 × 768 × 2 (g,b)    =  0.003M
  ------------------------------------------------
  per layer                                  7.08M
12 layers                                  = 84.93M
Pooler                  768 × 768          =  0.59M
Embedding LayerNorm                        =  0.0015M
-----------------------------------------------
encoder total                              ≈ 109.4M
+ classification head (num_labels=2)       =  0.0015M
-----------------------------------------------
TOTAL                                      ≈ 110M   (rounded)
```
The FFN is the dominant term because `d_ff = 4H`. For BERT-large: `4·1024² + 2·1024·4096 = 12.58M` per layer × 24 = 302M, plus 31.3M of embeddings and a 1.05M pooler = **≈ 335M**, reported as 340M.
- **Why asked:** A whiteboard derivation that reveals whether the candidate knows the Transformer block's actual shape. `[Company style: big-tech research, ML-platform]`
- **Trap:** Forgetting that the FFN holds `2/3` of each layer's parameters, or omitting the embedding matrix (which is 21% of the total).

**Q71. Why is BERT fine-tuning more learning-rate-sensitive than LLM SFT?**
- **Answer:** Five reasons. (1) **You update 100% of the parameters.** With LoRA you update a low-rank adapter, so the effective displacement in weight space is tiny even at LR 1e-4. (2) **The datasets are smaller.** 5,000 rows at batch 16 is 312 steps; a 7B SFT run is often 2,000–20,000 steps, so a large early step is amortised and corrected. (3) **BERT is post-LayerNorm.** Post-LN Transformers are known to be unstable without warmup; pre-LN models (most modern LLMs, and RoBERTa-class encoders) tolerate much higher LRs. (4) **Scale of representation.** 110M parameters must encode `P(next token)` over a 30k vocabulary; a single bad step can move the whole representation, since there is no redundancy. (5) **No output-embedding tie.** The MLM head is a full `768 × 30,522` matrix; nothing anchors the hidden scale. Empirically: 2e-5 works, 5e-4 gives a model that scores below a linear probe, and 2e-3 destroys it within ~100 steps while the training loss falls monotonically.
- **Why asked:** A favourite for candidates who claim both encoder and LLM experience. It tests whether the "small learning rate" rule is understood or memorised. `[Company style: big-tech research]`
- **Trap:** "Because BERT is smaller." Size is not the mechanism — post-LN placement and full-parameter updates are.

**Q72. Explain catastrophic forgetting in encoder fine-tuning with the loss curves.**
- **Answer:** At LR ≥ ~5e-4 the encoder's weights leave the region that minimises the pretraining objective and enter a narrow basin that fits your few thousand labelled rows. The signature is unmistakable: **training loss falls smoothly and monotonically, while eval loss falls for the first ~100–300 steps and then rises sharply**, and eval accuracy plateaus *below* the linear-probe baseline. The gap between the two curves is forgetting, not overfitting — the distinction matters because the remedies differ. Overfitting is fixed with more data and more regularisation; forgetting is fixed with a **lower learning rate**, fewer epochs, or **freezing the bottom layers**. The practical diagnostic: compare `trainer.evaluate()` before training (on the pretrained encoder plus a random head, train the head only for 100 steps) against after full fine-tuning. If full fine-tuning is worse, you have forgetting.
- **Why asked:** Tests the loss-curve reading skill and the distinction between two things that look identical on a plot.
- **Trap:** Diagnosing it as overfitting and adding dropout/weight decay. That does not fix forgetting and makes the underfitting worse.

**Q73. Why does `[CLS]` pooling work at all, and when does it break?**
- **Answer:** It works because (a) `[CLS]` carries no lexical prior, (b) bidirectional attention lets it attend to every token, and (c) **the NSP head was trained on it**, forcing gradient descent to make it a usable whole-sequence summary. It breaks in three cases: (1) **similarity/retrieval** — mean pooling over non-pad tokens beats raw `[CLS]` by 2–6 STS points, which is the entire premise of Sentence-BERT; (2) **models trained without NSP** — RoBERTa's `[CLS]` is measurably weaker as a sentence vector, though it fine-tunes fine for classification because the head is retrained anyway; (3) **long or multi-topic documents**, where a single vector cannot hold all the content and chunked mean pooling over passages is required.
- **Why asked:** A subtle point that reveals whether the candidate has done retrieval work or only classification.
- **Trap:** "`[CLS]` is the sentence embedding" — stated as universal. It is task- and pooling-dependent.

**Q74. Explain the QA span-selection head and its failure to enforce `start <= end`.**
- **Answer:** The head is a single `Linear(768, 2)` applied to every token position: column 0 produces `start_logits`, column 1 produces `end_logits`. Each gets an **independent** softmax over the sequence, and the loss is the sum of the two cross-entropies. Nothing in the architecture couples them, so `argmax(start) > argmax(end)` is entirely possible. Post-processing must therefore enforce `start <= end` **during span enumeration** — the standard decoder keeps the top 20 starts and top 20 ends, iterates all ≤400 pairs, keeps only those with `start <= end` and `length <= max_answer_length` and not fully inside the question, and scores each as `start_logit + end_logit`. The naive "take both argmaxes then repair" approach is wrong because the repair (`end = start`) manufactures a one-token answer that is guaranteed incorrect rather than falling back to the true second-best span.
- **Why asked:** Tests whether the candidate has written the decoder or only called the pipeline.
- **Trap:** "The model learns to output valid spans." It learns nothing of the sort — the constraint is not in the loss.

**Q75. What is the SQuAD 2.0 no-answer mechanism, and why does it need a special logit?**
- **Answer:** The `[CLS]` token is treated as the no-answer candidate: unanswerable training examples are labelled `start = end = 0`, and at inference you compare the best span score against the `[CLS]` score. The problem is that `[CLS]`'s logit is produced by the same linear layer but its hidden state comes from a different distribution (it summarises the whole sequence, whereas word positions encode local content), so its logit is systematically smaller. The standard fix is to add a **learned or tuned `cls_logit` offset** before comparing, or to calibrate a threshold on the dev set. Without it, the model almost never abstains and you get confident wrong answers.
- **Why asked:** Abstention is a production requirement that most tutorials skip entirely. `[Company style: applied ML at a product company]`
- **Trap:** "Use the softmax probability and threshold at 0.5." The two logit distributions are not on the same scale; a fixed 0.5 threshold is meaningless.

**Q76. Why is token-level accuracy for NER misleading? Give the numbers.**
- **Answer:** Two reasons. (1) **Class dominance** — in CoNLL-2003 English, **~83%** of tokens are `O`; in short WikiAnn sentences it is higher. A model that emits `O` everywhere therefore scores 0.83–0.90 accuracy and **0.00** entity-level F1. (2) **Boundary blindness** — token scoring gives partial credit, so `B-PER I-PER` predicted as `B-PER O` scores 1/2 on the entity, and `B-ORG I-ORG` vs `B-LOC I-LOC` confusion is not distinguished from a boundary error. Entity-level `seqeval` requires the **type and both boundaries** to match exactly, which is what the business actually asks ("did we find the invoice number?"). Typical gap on real data: token F1 0.95 vs entity F1 0.89 — six points of self-deception. In the CS-07 healthcare case study the gap was 0.958 versus 0.894.
- **Why asked:** The highest-signal NER question in this file. Anyone who has shipped NER will answer instantly.
- **Trap:** "Both metrics are useful." Token accuracy is useful only as a *sanity* signal alongside entity F1 — never as the selection metric.

**Q77. How does `seqeval` handle a `B-X I-Y` sequence (a malformed transition)?**
- **Answer:** In the **IOB2** scheme, a `B-X` immediately followed by `I-Y` for `Y ≠ X` is a type change without a `B-`, which is malformed. `seqeval` treats the `I-Y` as the start of a new entity of type `Y` — it is forgiving, which matters because gold data is frequently malformed in exactly this way in CoNLL-2003 (documented annotation errors). The practical implication: your reported F1 is partly determined by how malformed gold is handled, so pin the `seqeval` version and the `scheme` argument (`IOB1`, `IOB2`, `IOE1`, `IOE2`, `BILOU`) in your evaluation harness. If your data is BILOU and you do not pass `scheme="BILOU"`, the scores are silently wrong.
- **Why asked:** Separates people who have read the `seqeval` source from people who have called `seqeval.compute()`.
- **Trap:** Assuming any BIO-like scheme is scored identically.

**Q78. You have 300 classes and a taxonomy that changes monthly. How does that change your architecture choice?**
- **Answer:** The taxonomy churn, not the class count, is the binding constraint. Every change to `num_labels` invalidates the classification head — it is a `Linear(768, 300)` whose shape changes — so a taxonomy edit means a full retrain plus a re-validation cycle. Better designs: (1) **two-stage** — a stable coarse encoder (12–20 groups) plus a per-group **SetFit** or logistic-regression head, so adding a queue refits one head in seconds; (2) **nearest-centroid / k-NN over embeddings** — encode the label descriptions once, classify by similarity, and a new label needs only an embedding, not a retrain; (3) **hierarchical classification** — encoder for the coarse level, cheap heads below. In the CS-07 case study the two-stage design cost **2.8 points of accuracy** and cut the taxonomy-change cycle from **5 days to 30 minutes** — the correct trade for monthly churn. `[Company style: applied ML at a product company, ML-platform]`
- **Why asked:** Tests whether architecture choice is driven by the deployment lifecycle rather than by benchmark scores.
- **Trap:** Reaching for a bigger model. A 300-class DeBERTa does not solve a label-churn problem.

**Q79. You need extractive QA where the answer must be verbatim for legal reasons. Why prefer an encoder to an LLM?**
- **Answer:** Because an extractive encoder **cannot produce the answer** — it can only select a contiguous substring of the input, so hallucinated content is architecturally impossible for the span itself. An LLM, even with `"answer only from the context"`, can paraphrase, splice, or invent. Additional advantages: deterministic and version-pinnable (pin a checkpoint hash for an audit), 10–30 ms versus 400–2,000 ms, self-hostable in a VPC, and 3–4 orders of magnitude cheaper per query. The costs: SQuAD 2.0 exact match on a domain set is 0.70–0.80 rather than 0.90+, you need an abstain threshold, and you must handle the 512-token windowing. The standard architecture is encoder-extracts-then-LLM-synthesises, where the LLM's input is a verbatim, labelled quote. `[Company style: regulated industry, big-tech applied ML]`
- **Why asked:** Tests whether the candidate understands that "cannot hallucinate the span" is a *structural* guarantee, not a prompt instruction.
- **Trap:** "With a good prompt the LLM is just as safe." A prompt is a request, not a constraint.

**Q80. A cross-encoder reranker uses BERT in a way that is not classification or QA. Explain.**
- **Answer:** A cross-encoder scores a `(query, passage)` pair by concatenating them as `[CLS] query [SEP] passage [SEP]` and reading the `[CLS]` logit as a relevance score (a regression or binary head). This is the same architecture as sentence-pair classification. It is far more accurate than a bi-encoder (which embeds query and passage separately and uses cosine similarity) because every query token can attend to every passage token — full cross-attention over the pair. The cost: it cannot pre-compute passage embeddings, so it must run a forward pass **per (query, passage) pair**. The production pattern is therefore **retrieve-then-rerank**: a bi-encoder or BM25 retrieves the top 100–1,000 candidates cheaply, and the cross-encoder rescores only those, typically 20–50 pairs per query in ~10–30 ms on GPU. Gains of 5–15 nDCG points over bi-encoder-only retrieval are routine. `[Company style: search/ranking, big-tech applied ML]`
- **Why asked:** Tests whether BERT knowledge connects to modern retrieval stacks — the largest remaining encoder deployment.
- **Trap:** "Just use the cross-encoder for retrieval." It is O(N) forward passes per query and cannot scale to a million-document index.

**Q81. What is the `Trainer`'s `compute_metrics` contract, and what is the most common mistake in it?**
- **Answer:** It receives an `EvalPrediction` with `.predictions` (usually raw **logits**, not probabilities) and `.label_ids`, and returns a dict of `str → float`. The most common mistake is calling `argmax` on logits without realising they are logits — that is actually correct; the real mistakes are (a) forgetting `axis=-1` so `argmax` flattens the batch, (b) applying `softmax` and then `argmax` (harmless but wasteful), (c) **using `average="weighted"`** on imbalanced data, and (d) for NER, passing raw label ids to `seqeval`, which requires **strings** and must not see `-100`. Also: `compute_metrics` runs on the full accumulated prediction set, so memory on a large eval set can spike — set `eval_accumulation_steps` if so.
- **Why asked:** A concrete implementation question that a candidate with real hours in `Trainer` answers from memory.
- **Trap:** Assuming `compute_metrics` gets probabilities.

**Q82. How would you fine-tune BERT for a regression task (predicting a 1–5 star rating) instead of classification?**
- **Answer:** Set `num_labels=1` **and** `problem_type="regression"` on the model config, which switches the loss to MSELoss and the output to a single logit. Evaluate with MAE, RMSE, and Spearman correlation rather than accuracy or F1. Practical notes: (1) `num_labels=1` alone without `problem_type` gives you a 1-class softmax, which is degenerate; (2) regression targets should usually be standardised or at least centered, and the logit is unbounded so you must clamp at inference; (3) a common alternative is **ordinal classification** — treat the 5 ratings as 5 classes with a cross-entropy loss and take the expected value `Σ k·p_k` as the prediction, which often beats MSE because it respects the ordering and gives you a distribution; (4) `metric_for_best_model="mae"` with `greater_is_better=False`.
- **Why asked:** Tests whether the candidate knows `problem_type` is a real, load-bearing flag.
- **Trap:** `num_labels=1` with the default `single_label_classification`. Shapes match, and the loss is meaningless.

**Q83. What is the relationship between BERT and DistilBERT, and when do you distil instead of quantizing?**
- **Answer:** DistilBERT is a **6-layer, 66M-parameter student** distilled from BERT-base by Sanh et al., trained with a triple loss: KL divergence against the teacher's softened logits (temperature 2), the ordinary MLM loss, and a cosine-similarity loss between student and teacher hidden states. Result: **40% smaller, 60% faster, 97% of GLUE**. Distillation and quantization are **orthogonal axes** and compose: distillation changes the *architecture* (fewer layers), quantization changes the *numeric precision* (int8) — DistilBERT + int8 ONNX is the standard fast-CPU classifier. Choose distillation when you control the model architecture and can afford a training run (it needs the teacher and a data pass); choose quantization when you have a trained model and just need it faster with no retraining. A useful asymmetry: quantization is a ~30-minute, calibration-only operation with <1 point loss; distillation is a GPU-day with 3 points of loss — but distillation gives a *permanently* faster model that also quantizes well afterwards. → CS-08 and CS-10.
- **Why asked:** Tests whether the two efficiency techniques are understood as a pipeline rather than as competitors.
- **Trap:** Treating them as alternatives. Applying int8 to DistilBERT gives you both wins.

**Q84. BERT-large is 3× the parameters of BERT-base but only ~1–2 GLUE points better. Why the poor scaling?**
- **Answer:** Three reasons. (1) **Pretraining data is the bottleneck, not capacity.** BERT was trained on 16 GB / 3.3B words for 1M steps — Chinchilla-style scaling analysis says that is far too few tokens for 340M parameters, so BERT-large is *data-starved relative to its capacity*. RoBERTa-large (355M, 160 GB, 500k steps, batch 8k, ~2T tokens) gains far more over RoBERTa-base than BERT-large does over BERT-base — that is the empirical proof. (2) **Downstream datasets are tiny.** GLUE tasks have 2.5k–400k training rows; a bigger model overfits them faster, and the capacity advantage cannot be expressed. (3) **Architecture was fixed.** Both sizes use the same block, so the only lever is depth/width — and RoBERTa/DeBERTa/ELECTRA showed that better objectives and attention (RTD, disentangled attention) buy more than width does. The consequence for you: **more data and a better objective beat a bigger encoder.** The corollary is DeBERTa-v3-base (86M) beating RoBERTa-large (355M) on several GLUE tasks. `[Company style: big-tech research]`
- **Why asked:** Tests scaling intuition specifically for encoders, where the usual "bigger is better" instinct is wrong.
- **Trap:** "Encoder scaling saturates." It does not — it was *data*-limited. RoBERTa-large is the counterexample.

**Q85. What is the argument that pretraining data beats architecture?**
- **Answer:** RoBERTa keeps BERT's *architecture* almost unchanged and changes only the training recipe: 160 GB of data instead of 16 GB, batch 8k instead of 256, 500k steps, dynamic instead of static masking, whole-word masking, a 50k byte-level BPE vocabulary instead of 30k WordPiece, and **no NSP**. It gains **2–4 GLUE points** and beats BERT-large while using a base-sized model. The token arithmetic: RoBERTa sees roughly `8,000 × 512 × 500,000 ≈ 2×10¹²` tokens versus BERT's `256 × 128 × 1,000,000 ≈ 3.3×10¹⁰` — a **~60× increase in tokens seen**. No architectural change in the same era produced a comparable gain. The practical takeaway for a practitioner: if you are choosing where to spend effort on a domain model, **domain-adaptive continued pretraining (CS-12) usually beats switching to a bigger architecture.**
- **Why asked:** Tests whether the candidate reasons about the pretraining budget or only about model size. `[Company style: big-tech research, startup ML eng]`
- **Trap:** "RoBERTa is just BERT with more data" — true and the point, but omitting the objective changes (no NSP, dynamic masking) misses half the answer.

**Q86. Why does SpanBERT beat BERT on QA and coreference but not on classification?**
- **Answer:** SpanBERT's pretraining masks **contiguous spans** (geometric length distribution, mean 3.3) instead of individual tokens, and adds a **Span Boundary Objective (SBO)**: the representation of the two tokens flanking a masked span must predict every token inside it. That objective forces the model to encode *span-level* structure and boundary information — exactly what coreference resolution, relation extraction, and extractive QA need, since all three are about relating or locating spans. Sentence-level classification does not use boundary information at all: it needs a single pooled vector summarising the whole input, and the SBO objective spends capacity on a skill the classification task never queries. So SpanBERT gives roughly break-even on GLUE-scale classification and +2–5 points on span-structure tasks.
- **Why asked:** Tests whether "objective alignment to task structure" is understood as a design principle.
- **Trap:** Assuming a better-pretrained model is better everywhere.

---

## Level 4 — System Design & Scenario

*Each prompt is a 15-minute whiteboard. Structure every answer as: requirements → constraints → design → trade-offs → failure modes. The interviewer is grading the structure and the numbers, not the model name.*

**Q87. Design a document-classification system for a legal team: 500,000 contracts, 90 label types, must run on-premise, p95 latency under 200 ms, $3,000/month budget.**
- **Answer — Requirements:** 90 mutually-exclusive clause-presence labels over contracts averaging 40 pages (~55,000 tokens). On-prem (no egress). p95 < 200 ms. Throughput: assume 5,000 documents/day and 50,000 ad-hoc queries/day = ~55k/day ≈ 0.64 QPS average, with maybe 10× peak.
- **Constraints:** 512-token hard limit vs 55,000-token documents → chunking is mandatory. 90 labels → hierarchical or multi-label structure. On-prem CPU likely (GPU budget would exceed $3,000/month for a dedicated instance).
- **Design:**
  1. **Segmentation** — split each contract into 512-token windows with 128-token stride at *clause* boundaries where possible (a clause splitter is more accurate than a fixed stride).
  2. **Encoder** — `microsoft/deberta-v3-base` fine-tuned as **multi-label** (`problem_type="multi_label_classification"`, BCE loss), because a contract can contain several clause types simultaneously. Domain adaptation: continue pretraining on 20M tokens of the firm's unlabelled contracts for 20k steps (CS-12) — this typically gives +3–5 F1 on legal jargon.
  3. **Aggregation** — max-pool the per-chunk sigmoid scores per label, then apply a per-label threshold calibrated on the dev set (not a global 0.5).
  4. **Serving** — ONNX export + int8 dynamic quantization, served from ONNX Runtime on 4 × c7i.2xlarge behind a queue. Chunk-level batching gives real throughput: 55k documents × ~110 chunks = 6M chunk-predictions/day; at 700 preds/s/box that is 2.4 hours of compute across 4 boxes, comfortably inside a day.
  5. **Abstain path** — any label with max score in [t_low, t_high] goes to a paralegal review queue.
- **Costs:** 4 × c7i.2xlarge on-demand = $1,428/month; spot ≈ $428/month. Training: one A10G for 6 hours = $6. Labelling: 8,000 contracts × 90 labels — use a **two-stage labelling** approach (LLM pre-annotation with a human verifying only the positive labels) to keep it to ~300 hours.
- **Trade-offs:** Multi-label BCE versus 90 independent binary classifiers — BCE shares the encoder and is 90× cheaper at inference; independent classifiers allow per-label models but multiply serving cost by 90. Chunk-level max pooling versus a hierarchical attention pool — max pooling is free and within 1 point; a learned pool needs a second training stage.
- **Failure modes:** (a) label leakage across chunk boundaries — a clause split mid-sentence; mitigate with stride overlap and merging adjacent positive chunks. (b) 90 labels with <20 examples each — those labels will sit at 0.3–0.5 F1; report per-label metrics and route low-support labels to review. (c) drift when the firm's contract template changes; monitor the per-label positive rate weekly. (d) int8 export changing numerics — assert >99% prediction agreement between PyTorch and ONNX on 500 held-out documents.

**Q88. Design NER for PII redaction on 10M support tickets/day, with a hard requirement of <0.5% false-negative rate and a p99 of 50 ms.**
- **Answer — Requirements:** Nine entity types (PERSON, EMAIL, PHONE, ADDRESS, SSN, CARD, IBAN, DOB, IP). 10M tickets/day ≈ 116 QPS average, maybe 400 QPS peak. **Recall is the binding constraint** — a missed SSN is a breach.
- **Design:**
  1. **Model** — BERT-base-cased (casing matters: `MacKenzie` vs a common noun) or DeBERTa-v3-base, `num_labels = 2× len(types) + 1`, fine-tuned with `DataCollatorForTokenClassification` and **`seqeval` entity F1** as the selection metric. Train with `label_all_tokens=True` so partial-word entities are fully covered.
  2. **Threshold for recall** — do not use `argmax`. Compute, per entity type, a **B- tag threshold** calibrated so that dev-set recall ≥ 0.995, accepting precision loss. This is the single most important design decision: the operating point, not the model.
  3. **Deterministic backstop** — run regex/checksum validators (Luhn for cards, IBAN mod-97, SSN format, email/phone patterns) *in parallel* with the model and **union** the results. Recall-critical systems never rely on a single component. This typically lifts SSN/card/email recall from 0.97 to 0.999 for a few milliseconds.
  4. **Windowing** — tickets average 90 tokens but support email threads can be 5,000. Slide with 128-token stride, merge spans across windows by character offset, and deduplicate overlapping spans by choosing the highest-scoring one.
  5. **Serving** — ONNX int8. At 116 QPS average and ~30 ms per ticket on CPU, that is ~4 cores; a 2-box autoscaling group handles 400 QPS peak. Cost ≈ $500/month.
- **Trade-offs:** Recall-tuned thresholds cost precision, which means more redaction tokens — acceptable for PII, unacceptable for most other tasks. A CRF layer on top of the token classifier improves boundary consistency by 1–2 F1 but complicates ONNX export — measure before adopting.
- **Failure modes:** (a) **`-100` masking errors** — a mis-aligned NER model will report 0.95 token accuracy and 0.4 entity F1, and in a PII system that is a breach, so assert entity F1 ≥ 0.90 in CI before any deploy. (b) **Tokenization mismatch at serve time** — if you serve a fast tokenizer and trained on a slow one, entity offsets shift; pin the tokenizer directory with the model. (c) **Offsets** — you must return *character* spans for redaction, so store `offset_mapping` and convert back; subword-to-char conversion is a classic off-by-one source. (d) **New entity types** (a new national ID format) require a retrain, so monitor for high-entropy out-of-vocabulary numerics.

**Q89. Design a two-stage retrieval + reranking system for 5M support articles. Where does BERT appear?**
- **Answer — Requirements:** 5M articles, 50k queries/day, p95 < 300 ms, quality target nDCG@10 ≥ 0.55.
- **Design — BERT appears in three places:**
  1. **Bi-encoder retrieval** — `all-MiniLM-L6-v2` (22M, 6 layers) or `multi-qa-mpnet-base` embeds chunks offline into a vector index (FAISS HNSW or Qdrant). 5M documents × ~3 chunks = 15M vectors at 384 dims fp16 = 11 GB — fits in RAM on one box. This is the BERT-family model that does the heavy lifting at scale, and **offline embedding is why it scales**: the 15M vectors are computed once and rebuilt only when the corpus changes.
  2. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores the top 100 retrieved `(query, chunk)` pairs with full `[CLS] query [SEP] chunk [SEP]` cross-attention. 100 pairs × ~2 ms = 200 ms on GPU, or ~40 ms batched on an A10G. This is where the nDCG gain comes from: **+5–15 points over bi-encoder alone.**
  3. *(Optional)* **Answer extraction** — a SQuAD-style extractive QA model picks the answer span from the top reranked chunk, giving a verbatim, citable answer.
- **Latency budget:** embed the query (10 ms) + ANN search over 15M vectors (5–15 ms) + rerank 100 pairs (40 ms) + optional span extraction (15 ms) = **~80 ms p50**, comfortable inside 300 ms p95.
- **Cost:** one A10G for reranking ($1/hr = $730/month) plus one memory-optimised CPU box for the index ($250/month). Versus an LLM reranker at 100 pairs × 2,000 tokens × 50k queries/day = 10M tokens/day = $30/day = **$900/month for the reranking step alone**, with 10× the latency.
- **Trade-offs:** rerank depth — 100 pairs costs 40 ms and gains ~80% of the achievable nDCG; 500 pairs costs 200 ms and gains another 1–2 points. ANN recall — HNSW with `ef_search=64` recalls ~0.95 of true top-100; raising it costs latency linearly.
- **Failure modes:** (a) **embedding/query distribution mismatch** — a bi-encoder trained on MS MARCO underperforms on your support jargon; fine-tune it on 5k `(query, clicked-article)` pairs (`code/10_embedding_finetune.py`) for +5–10 points. (b) **chunking that splits answers** — use 256-token chunks with 64-token overlap, not 512/0. (c) **stale index** — rebuild on a nightly job and version the index alongside the model; a query against a v1 index with a v2 encoder returns garbage. (d) **reranker latency blow-up** under bursty load — batch requests with a 20 ms coalescing window rather than serving one query at a time.

**Q90. Design an experimentation and rollout plan for replacing a rules-based ticket router with a fine-tuned BERT, with a rollback path.**
- **Answer — Requirements:** 40 queues, 200k tickets/day, agents currently correct 100% of routes manually with ~15% override rate. The router must not make things worse.
- **Phase 0 — offline (2 weeks).** Build a frozen 5,000-ticket golden set by sampling stratified by queue, labelled by two agents independently (measure Cohen's κ; if κ < 0.75, fix the queue definitions before continuing). Fine-tune DeBERTa-v3-base, 3 epochs, LR 2e-5, per-class thresholds calibrated on dev. Gate: macro F1 ≥ 0.85 **and** no queue below 0.60 **and** the confusion matrix shows no dangerous pair (e.g. `fraud` predicted as `billing`).
- **Phase 1 — shadow (1 week).** Run the model on 100% of live traffic, log predictions, take no action. Compare against the human route. This surfaces *distribution* problems offline sets never show: new product launches, seasonal spikes, an upstream form change.
- **Phase 2 — canary (2 weeks).** Route 1% of traffic through the model as a *suggestion* to agents (not an automatic route), then 5%, then 25%. Track per-queue override rate, agent handle time, and reopen rate. Gate: override rate ≤ the 15% baseline and handle time not worse.
- **Phase 3 — auto-route with a confidence guard.** Auto-route only when the top-1 score exceeds a per-queue threshold calibrated so that precision ≥ 0.95; send the rest to the existing rules path. This usually covers 70–85% of volume. Gate: reopen rate ≤ baseline.
- **Phase 4 — full rollout** with a 10% holdback for ongoing measurement.
- **Rollback:** the ONNX artefact for the previous version stays resident; the routing decision is a config flag with a feature-flag override, so rollback is a config flip measured in seconds, **not** a retrain. Never make rollback require a deploy.
- **Monitoring:** (a) input drift — token-length distribution and `[UNK]` rate weekly; (b) prediction drift — queue distribution, alert on a >5-point shift; (c) quality drift — sample 200 predictions/week, label them, track rolling macro F1; (d) calibration is not a control input unless you use scores for auto-routing, in which case it is.
- **Failure modes:** (a) **label leakage from the golden set** — if the golden set was sampled from tickets the model trained on, the offline gate is meaningless; hold out by time, not at random. (b) **queue definition drift** — a queue is added mid-experiment, invalidating the model's head; the two-stage design (encoder for the coarse group, refittable head for the queue) is the mitigation. (c) **feedback loops** — once the model routes, the human corrections you would use as future labels change distribution; keep a random 5% routed by the rules engine as a control. (d) **silent degradation** — the override rate creeps from 12% to 22% over three months without any alert; the weekly sampled audit is the only thing that catches it.

**Q91. You must serve 300 tenant-specific classifiers from one GPU. Design it.**
- **Answer — Requirements:** 300 customers, each with 5–20 labels and 500–20,000 of their own labelled rows. One A10G (24 GB). 2,000 predictions/s aggregate.
- **Naive design and why it fails:** 300 separate full fine-tunes = 300 × 440 MB = **132 GB**. Does not fit; each tenant retrain is a separate GPU job.
- **Design — one shared base + LoRA adapters:**
  1. **Base model** — `deberta-v3-base` or `bert-base-uncased`, frozen, resident once (440 MB fp32 / 220 MB fp16).
  2. **Per-tenant LoRA** — `LoraConfig(r=8, target_modules=["query","value"], modules_to_save=["classifier"], task_type=SEQ_CLS)`. Roughly 0.6–0.9M trainable params per tenant → **~3.5 MB per adapter**. 300 adapters = **1.05 GB**.
  3. **Serving** — a multi-adapter server: load the base once, load/serve adapters keyed by tenant id. `peft` supports `model.set_adapter("tenant_42")` and vLLM-style batched multi-LoRA; for encoders the simplest reliable pattern is a small pool of processes each holding the base plus a hot set of adapters, with an LRU eviction policy (most tenants are cold — 300 tenants will not all be active within a minute).
  4. **Quality** — LoRA at r=8 costs 0 to −1 point versus full fine-tuning on GLUE-scale tasks, and tenant datasets are small enough that the gap is usually zero.
- **Cost/benefit:** VRAM goes from an infeasible 132 GB to a base of 0.44 GB + 1.05 GB of adapters + 2–4 GB of activations. Cold-start latency for a non-resident adapter is ~200 ms (a 3.5 MB file plus a merge); warm it on a nightly schedule or on first request with a warmup call.
- **Trade-offs:** r=8 vs r=16 — r=16 doubles the adapter size for typically +0.2 points; not worth it at this tenant count. Shared base vs per-tenant base — sharing means a base-model update improves all tenants at once, and also means a base-model bug breaks all tenants at once. Adapter isolation must be enforced at the request-routing layer, or tenant A's data leaks into tenant B's predictions.
- **Failure modes:** (a) **adapter crossover** — a routing bug serves the wrong adapter; assert the adapter name equals the tenant id and log it per prediction. (b) **forgetting `modules_to_save=["classifier"]`** — the head stays random and frozen while you train adapters onto a random head; the loss plateaus and looks like an LR problem. (c) **LoRA LR too low** — LoRA needs 1e-4 to 3e-4, ten times the full-fine-tuning LR, because you are training a small randomly-initialised update. (d) **ONNX export with an adapter** — merge first with `model.merge_and_unload()`, or export per tenant and lose the sharing benefit.

**Q92. You have one 8-GB GPU and a 200,000-row NER dataset. Design the training run.**
- **Answer — Requirements:** 200k rows, 9 labels, `max_length` ~128, one 8 GB GPU (an RTX 3070-class or a Colab T4 with the batch squeezed).
- **Design:**
  - **Sequence length** — measure the p99 token length; NER sentences are typically <64 tokens. Set `max_length=128` (headroom) and use `DataCollatorForTokenClassification` for dynamic padding. This alone is a 4–20× compute reduction versus 512.
  - **Batch size** — BERT-base at 8 GB, S=128, fp16: `per_device_train_batch_size=32` fits (~2.5 GB). Add `gradient_accumulation_steps=2` for an effective batch of 64.
  - **Epochs** — 200k rows / 64 = 3,125 steps per epoch. **2 epochs = 6,250 steps**, comfortably in the healthy range. Three epochs is likely to overfit; select the checkpoint on entity F1.
  - **Optimizer memory** — 8 bytes/param for AdamW. Consider `optim="adafactor"` (no second moment, ~0.5 bytes/param) if VRAM is tight, at a small quality cost.
  - **Throughput** — ~0.10 s/step at fp16/S=128/batch 32 on a T4 → 3,125 steps ≈ 5 min/epoch, 10 minutes total. On a 3070, ~2× faster.
  - **fp16 caution** — enable it via `TrainingArguments(fp16=True)` so the scaler is handled; never hand-roll it on this GPU without a `GradScaler`.
  - **Evaluation** — `compute_metrics` with `seqeval`, `eval_strategy="steps", eval_steps=500`.
- **Trade-offs:** `max_length=128` truncates the 1% of long sentences. For NER that loses labels, not just context. Check whether *any* entity spans the truncation point; if so, use sliding windows at inference and merge by character offset.
- **Failure modes:** (a) OOM mid-epoch — drop batch to 16 and raise accumulation to 4; the effective batch is what matters. (b) fp16 NaN on LayerNorm — switch to `bf16` if the GPU is Ampere+, else keep fp16 with the built-in scaler. (c) `num_workers=0` starving the GPU — set `dataloader_num_workers=4`. (d) silent label misalignment — assert `(batch["labels"] == -100).sum() > 0` on the first batch.

**Q93. You are asked to cut inference cost by 60% for an existing BERT-base classifier serving 200M predictions/month. Where do you look, in what order?**
- **Answer — Baseline:** 200M/month ≈ 77 predictions/s average. On a GPU at 3,000 preds/s that is ~2.6% utilisation — you are paying for an idle GPU.
- **Order of attack, highest payoff first:**
  1. **Right-size the hardware.** At 77 preds/s average, an 8-vCPU CPU box at 700 preds/s is plenty. Move off the GPU: **$1,006/month → $260/month**, an immediate 74% cut with zero model change. Check the peak-to-average ratio first; autoscale for peaks.
  2. **Dynamic padding + batching.** The classic hidden waste: if inference pads to 256 and real inputs are 90 tokens, you are doing 2.8× extra work. Batch with length-sorted inputs and pad per batch. **20–60% throughput gain, free.**
  3. **ONNX + int8 dynamic quantization.** `optimum-cli export onnx` then `optimum-cli onnxruntime quantize --avx512`. **2–4× CPU throughput**, typically <1 point of accuracy. Verify with a 500-row PyTorch-vs-ONNX agreement check. → CH-07 §6.
  4. **Distil or switch architecture.** DistilBERT (66M) is 60% faster at 97% of GLUE. DeBERTa-v3-xsmall (22M) or MiniLM-L6 are another 2–3× on top. This is the 3–5× tier and needs a retrain. → CS-08.
  5. **Cache.** Many high-volume classifiers see heavy repetition (the same template ticket, the same product review). An exact-match cache on a normalised input hash plus a semantic cache on the embedding can absorb **20–50%** of traffic for the cost of a dictionary.
  6. **Cascade.** Route the easy cases (score > 0.95) with a tiny model (MiniLM or even TF-IDF) and only run BERT on the uncertain tail. If 80% of traffic is easy, you cut BERT's load by 5×.
- **Expected combined result:** steps 1+3 get you past 60% alone; 1–6 typically yields **85–95% cost reduction** with ≤1 point of quality loss.
- **Trade-offs:** int8 costs a small accuracy delta and requires numerics verification; a cascade costs pipeline complexity and an extra threshold to tune; a semantic cache risks returning stale answers if the underlying data changes.
- **What NOT to do first:** switch to a smaller model. It is the most expensive, slowest option and the one people reach for first.

**Q94. Design the evaluation harness for an NER model that will be retrained monthly on new labelled data.**
- **Answer — Requirements:** Monthly retrains, 9 entity types, an annotator team producing ~5k new sentences/month, a need to prove the model did not regress.
- **Design:**
  1. **Frozen golden set + rolling fresh set.** A 3,000-sentence golden set that never changes (for cross-version comparability) *plus* a 1,000-sentence set drawn from the newest month (for distribution relevance). Report both. A model that improves on the fresh set and regresses on the golden set has drifted, not improved.
  2. **Metrics — `seqeval` entity-level micro F1 as the headline, plus per-type P/R/F1.** Micro-F1 hides a collapsing entity type; the per-type table is what catches it. Alert on any type dropping >3 points.
  3. **Annotator-agreement guard.** Before any new data is used, compute Cohen's κ between annotators on a 200-sentence overlap. κ < 0.7 on a type means that type's labels are noise; report the model's ceiling as roughly κ, not 1.0.
  4. **Statistical significance.** At n=3,000, the 95% CI on F1 is roughly ±1.5 points. Do not ship a "2-point improvement" without a paired bootstrap over the same examples; the paired test is far more sensitive than comparing two independent CIs.
  5. **Regression suite in CI.** 500 sentences with expected outputs (label sequences, not probabilities — probabilities shift every run). Gate the deploy on 100% pass. Store the *predicted entity spans*, so a golden-set change is explicit.
  6. **Slice evaluation.** Report F1 by sentence length bucket (<32, 32–128, >128 tokens) and by source (which upstream system the text came from). Most NER regressions hide in one slice.
  7. **Error taxonomy.** Every month, hand-classify 100 errors into: boundary error, type confusion, missed entity, spurious entity, tokenization/`[UNK]` failure. Track the mix over time. A rising `[UNK]` share is a vocabulary problem, not a model problem.
  8. **Calibration of the operating point.** Because NER thresholds are tuned for the business's P/R preference, re-tune the thresholds on the fresh dev set each month and record them with the model version.
- **Trade-offs:** A frozen golden set drifts out of distribution over ~12 months; refresh it annually but version it so historical comparisons remain valid. Per-type thresholds are more accurate than one global threshold but add 18 more config values to version.
- **Failure modes:** (a) the golden set leaks into training via a shared annotation pool — keep it physically separate and hash-check training rows against it. (b) Metric drift from a `seqeval` version bump — pin it. (c) Evaluating on data the annotators produced after seeing model outputs (anchoring) — always annotate from scratch, blind to predictions.

---

## Level 5 — Debugging & Incident Response

**Q95. Your NER model reports 0.93 token accuracy and 0.02 entity F1. Walk me through your diagnosis in order.**
- **Answer — In this order:**
  1. **Confirm the metric is the problem, not the model.** Compute `seqeval` F1 properly and print the per-type table. If entity F1 is genuinely 0.02, the model has learned nothing about entities while learning the `O` class perfectly. If entity F1 is actually 0.85 and you mis-computed, you have an evaluation bug, not a model bug — and that is common enough to check first.
  2. **Check the collision rate of `[UNK]` and subword splits.** Print `tokenizer.tokenize()` for 20 gold entities. If a gold entity's first token is not the token your alignment thinks it is, you have an alignment bug.
  3. **Assert the `-100` masking.** `print((batch["labels"] == -100).sum(), batch["labels"].numel())`. For 9 labels with `[CLS]`/`[SEP]` and continuations, `-100` should be a substantial fraction. If it is only 3 per row, you are not masking continuations — or you are using `DataCollatorWithPadding`, which pads labels with `0` and trains on pads.
  4. **Count label frequencies in the processed dataset.** `Counter(l for row in ds["train"]["labels"] for l in row if l != -100)`. If you see only one value, your alignment loop is emitting a constant.
  5. **Verify the label map ordering.** `assert model.config.num_labels == len(label_names)` and check that `id2label` matches the scheme the data uses (`O`, `B-PER`, `I-PER`, …). A permuted label list gives exactly this symptom: the model is learning something, but the ids it predicts decode to the wrong names.
  6. **Only then** suspect the model: LR too high, too few steps, or the encoder was accidentally frozen.
- **Why the order matters:** steps 1–5 are free and account for the large majority of real occurrences. Step 6 costs a retrain. Candidates who jump to "more epochs" get marked down.

**Q96. Loss is 0.693 and completely flat for 500 steps on a binary classification task. What do you check?**
- **Answer:** 0.693 is exactly `ln(2)`, the loss of a model outputting 0.5 for both classes — i.e. **the head is not learning at all**. In order:
  1. **Is the head trainable?** `sum(p.requires_grad for p in model.parameters())`. A previous cell that froze `model.bert.parameters()` without re-enabling the classifier is the classic cause.
  2. **Is the optimizer seeing the head's parameters?** If you built `optimizer = AdamW(model.parameters(), ...)` *before* replacing the head, or constructed parameter groups manually and omitted the classifier, the head never updates. Print `len(optimizer.param_groups[0]['params'])`.
  3. **Is the LR effectively zero?** Print the LR from `scheduler.get_last_lr()` during training. A scheduler created with `num_training_steps=0`, or `warmup_steps` exceeding the total, pins the LR at ~0.
  4. **Are the labels all one value?** `Counter(train_dataset["labels"])`. A single-class dataset gives exactly `ln(2)` for binary and cannot be improved.
  5. **Are the labels reaching the model?** If you renamed the column wrongly, the model gets no `labels` and returns `loss=None` — but in a hand-written loop that raises, whereas in some setups you may be computing loss against a zero tensor. Print the batch's labels.
  6. **Is the input actually the input?** A tokenizer padding/attention bug that produces near-identical inputs for every row also yields a constant 0.5 output. Print the first three rows' `input_ids`.
  7. **Is the loss being computed with `model.eval()` on?** Dropout off is fine, but if you accidentally wrapped training in `torch.no_grad()` the model cannot learn and the loss stays exactly flat.
- **Why the order matters:** Flat-at-`ln(C)` is a *pipeline* symptom, not an optimisation symptom. Checking the LR and adding epochs is the wrong first move.

**Q97. NER F1 was 0.91 last week and is 0.62 today. Nothing changed in the model. What do you check?**
- **Answer — In order, cheapest and most likely first:**
  1. **The evaluation code, not the model.** Diff the eval harness against the last known-good commit. The most common cause is a change in the *scoring* path: a `seqeval` version bump, a label-order change, `scheme` argument removed, or a `-100` filter that now includes pad positions.
  2. **The test data.** Was the eval set regenerated? A re-split, a re-shuffle, or a new `np.random.choice` without a seed changes the test set. Compare the hash of the eval file to the recorded hash. **This is the #1 real cause of "the metric dropped and nothing changed".**
  3. **The collator or the alignment function.** If the dataset was re-processed (e.g. `map()` re-run on a fresh session), check that `labels` still contain `-100`. A dataset that lost its `-100` values scores catastrophically on entity F1 while token accuracy barely moves.
  4. **The tokenizer.** If the model was loaded with a different tokenizer (a different vocab file, fast vs slow), token ids shift and everything breaks while all shapes remain valid. `assert model.config.vocab_size == len(tokenizer)`.
  5. **The label map.** Someone edited `label_names` (a sort, an alphabetisation, a new type added). Print `model.config.id2label` and compare to the harness's list.
  6. **The input data distribution.** If the *serving* data changed (a new upstream system, a new locale, HTML tags now present) but the gold data did not, the model genuinely is worse on this input. Compare `[UNK]` rates and length histograms between the two weeks.
  7. **The artefact.** Confirm which checkpoint hash is actually loaded. A staging pointer that flipped to a half-trained checkpoint is embarrassing but common.
- **Why the order matters:** A 29-point drop with no model change is almost never a model problem. Steps 1–3 account for most cases and cost minutes.

**Q98. Your fine-tuned BERT beats the baseline on validation but is worse in production. Where do you look?**
- **Answer:** The gap between offline and online is almost always one of six things:
  1. **Distribution shift.** Production text differs from your training distribution — different source systems, HTML, new product names, non-English rows. Measure: compare token-length histograms, `[UNK]` rates, and vocabulary overlap between the two. If `[UNK]` tripled, the tokenizer is the problem.
  2. **Leakage in the offline split.** Duplicate or near-duplicate rows across train and test inflate offline scores. Deduplicate on a normalised text hash (lowercase, strip whitespace and punctuation, hash the first 200 chars) and re-evaluate. Leakage of 5–15 points is common in scraped datasets.
  3. **A preprocessing mismatch.** The training pipeline lowercased/stripped/truncated and the serving path does not (or vice versa). Assert that the exact same `tokenize_fn` is used in both, imported from a shared module — a copy-pasted function is how this happens.
  4. **Wrong artefact loaded.** Tokenizer from one checkpoint, weights from another, or a stale ONNX file. Assert vocab size and a known input's logits.
  5. **The offline metric is not the business metric.** Offline accuracy weights all classes equally; production cost is dominated by one class's errors. Re-evaluate with a cost-weighted metric derived from real escalation costs, and report per-class.
  6. **Calibration.** Production applies a threshold (auto-route above 0.9) that you never evaluated offline. An overconfident model with ECE 0.20 auto-routes far too much. Measure ECE and re-tune the threshold on held-out data.
  7. **Feedback/latency effects.** In production the model may be fed truncated input because of a timeout, or fed the wrong field because of an upstream schema change. Log the exact string the model received, not the string you think it received.
- **Why the order matters:** 1 and 2 are the most common and are falsifiable in an hour. 7 is the one people forget entirely.

**Q99. Your BERT fine-tune produces NaN loss at step ~40. Diagnose and fix.**
- **Answer:** NaN at a specific step (rather than step 0) means a large gradient or an overflow, not a shape bug. In order:
  1. **fp16 without loss scaling.** If you hand-rolled the loop with `model.half()` or `torch.autocast` without a `GradScaler`, an overflow in the softmax or LayerNorm produces `inf` gradients. Fix: use `TrainingArguments(fp16=True)` which manages the scaler, or add `torch.cuda.amp.GradScaler` and `scaler.scale(loss).backward()` / `scaler.step(optimizer)` / `scaler.update()`.
  2. **Learning rate too high.** 2e-3 on a full fine-tune produces exploding gradients within tens of steps. Confirm by logging `grad_norm` per step — a spike from ~1 to 1e4 before the NaN is the signature. Fix: LR 2e-5, and set `max_grad_norm=1.0` (which should have clipped it — check it is actually applied).
  3. **A bad batch.** An input with a huge sequence length (a row you forgot to truncate), empty strings, or `None` values. Print `max(input_ids.shape)` and scan for empties. Fix: `truncation=True` always, and assert every input is a non-empty string.
  4. **A division by zero in a custom loss or metric.** If you added class weights or a custom head, `log(0)` and `0/0` produce NaN. Wrap in `float('-inf')`-safe forms (`log_softmax` instead of `log(softmax)`) and `torch.clamp` denominators.
  5. **Corrupt labels.** A label id outside `[0, num_labels)` makes `CrossEntropyLoss` produce NaN rather than raising. `assert 0 <= max_label < num_labels`.
  6. **AdamW epsilon with fp16.** `adam_epsilon=1e-8` underflows in fp16; HF sets 1e-8 by default and the optimizer state is kept in fp32, but a hand-rolled fp16 optimizer can blow up. Raise to 1e-6.
- **Detect it early:** `torch.autograd.set_detect_anomaly(True)` on a single short run pinpoints the operation. It is slow — use it for diagnosis, then turn it off.

**Q100. Your model's predictions are correct but the confidence scores are wrong — 0.99 on everything. What is happening and what do you do?**
- **Answer:** This is **miscalibration**, and it is the expected outcome of fine-tuning on a small dataset, not a bug. A model trained to minimise cross-entropy on 1,000 rows will push logits toward ±∞ because that reduces loss on the training set. Typical ECE for a 1-epoch/1,000-sample IMDb fine-tune is **0.10–0.25** — a stated 0.95 confidence is empirically right about 70–85% of the time. Diagnose and fix:
  1. **Measure it.** Bucket held-out predictions into deciles and plot accuracy against mean confidence; compute ECE. If ECE > 0.05, act.
  2. **Temperature scaling (the fix).** Fit a single scalar `T` on a **validation** set (never the test set) by minimising NLL on `logits / T`. Typical fitted `T` for an overconfident encoder is 1.5–3.0. It changes nothing about the argmax, so accuracy is preserved exactly.
  3. **Label smoothing during training** (`label_smoothing_factor=0.05–0.1`) prevents the logits from running away in the first place.
  4. **More data or fewer epochs.** Both reduce overconfidence; the model is overfitting confidence even when accuracy has plateaued.
  5. **Do not use the score as a probability until it is calibrated.** If a business rule says "auto-approve above 0.9", that threshold was tuned on uncalibrated scores and will behave completely differently after calibration. **Re-tune the threshold after calibrating.**
- **Why it matters:** A confidence-driven auto-approval pipeline built on miscalibrated scores either approves far too much (dangerous) or far too little (useless), and the model's accuracy metrics will never reveal it.

**Q101. Your NER model detects multi-word entities but always misses the last word. What is the bug?**
- **Answer:** This is a **BIO boundary/tag-transition bug**, and there are four distinct causes to separate:
  1. **Training labels missing `I-` tags.** Count the label distribution in the processed dataset: `Counter(l for row in ds['train']['labels'] for l in row if l != -100)`. If `I-PER`/`I-ORG` counts are near zero, your source annotation used `B-` for every token of an entity (a common convention in some tools) or your alignment overwrote them. Fix: convert `B-X B-X B-X` to `B-X I-X I-X` in the data, or train with `label_all_tokens=True` and do the conversion in the collator.
  2. **`-100` applied to continuation subwords *and* to continuation words.** If your alignment compares word indices incorrectly (e.g. `word_idx != previous_word_idx` using the wrong variable, or resetting `previous_word_idx` in the wrong scope), the second word of a two-word entity can be masked out. Fix: print `list(zip(tokens, word_ids, aligned_labels))` for one multi-word entity and check by hand.
  3. **Truncation cutting the entity.** If the entity straddles `max_length`, the last word is never seen. Check the length distribution against `max_length`.
  4. **CRF/decode post-processing dropping the tail.** If you added a CRF or a BIO-repair step, a malformed transition (`I-PER` after `O`) may be silently discarded along with everything after it. Fix: fix the transitions rather than deleting them.
  5. **The model is genuinely confused at boundaries** — the least likely cause, and only after 1–4 are excluded. A correctly-trained BERT-base on CoNLL-2003 makes boundary errors, but not systematically on the *last* word only.
- **Why the order matters:** A *systematic* miss of the last word is a data or alignment property, not a model-capacity property. Retraining with more epochs will not fix it.

**Q102. You switch from `padding="max_length"` to `DataCollatorWithPadding` and your F1 drops by a point. Why might that be, and what do you do?**
- **Answer:** Dynamic padding is a pure efficiency change and should not affect quality — so something else changed, and there are five candidates:
  1. **The collator changed the label padding for token tasks.** You probably also switched a NER pipeline to a collator and lost the `-100` behaviour. Check: `DataCollatorWithPadding` pads labels with `0`; `DataCollatorForTokenClassification` pads with `-100`. This is the most likely cause, and it is exactly the trap this module is built around.
  2. **A different `pad_token` or `pad_token_id`.** If the tokenizer's pad token was changed or `pad_token_id` was set manually, the model's embedding of pads changes. For BERT, id 0 is `[PAD]`; a tokenizer whose pad token is `[SEP]` or `eos` will break BERT's attention-mask semantics. Assert `tokenizer.pad_token_id == 0` for BERT.
  3. **Different effective batch composition.** Dynamic padding means each batch has a different real token count, so gradient noise changes; with a borderline LR this can cost a point. Re-run with `seed=42` and both settings to distinguish signal from noise — a 1-point difference at n=500 rows is inside the ±3-point CI.
  4. **Evaluation-time padding affects the QA path.** For SQuAD-style QA, padding changes the number of positions over which `argmax` searches. With dynamic padding the padded positions differ per batch and can (correctly) be filtered; if your decoder does not filter `attention_mask == 0`, the *evaluation* results change. Fix the decoder, not the collator.
  5. **`pad_to_multiple_of` or `return_tensors` differences.** A collator that returns `list` instead of tensors, or pads to a multiple of 8, changes nothing mathematically but can expose a downstream dtype bug.
- **What to do:** Re-run both configurations with identical seeds and identical data. If the gap disappears, it was noise. If it persists, it is cause 1 or 2 — both are configuration, not the collator.

**Q103. Your QA model answers correctly on SQuAD but returns empty strings on your own documents, which are 8,000 tokens. What is happening?**
- **Answer:** This is the **512-token truncation** failure, and the empty string is the `[CLS]` collapse presenting itself. Diagnosis:
  1. **Confirm truncation.** `len(tokenizer(context)["input_ids"])` for a typical document. At 8,000 tokens you are using 4% of the context, and the answer is almost certainly in the discarded 96%.
  2. **Confirm the `(0,0)` collapse.** Print `start_logits.argmax()` and `end_logits.argmax()` on a failing example. If both are 0, `[CLS]`, the model is not "unsure" — it is doing what it was trained to do for out-of-window content, because your training labels defaulted to 0 for truncation-affected rows.
  3. **The fix has three parts:**
     - **Sliding windows** — `max_length=384, stride=128, return_overflowing_tokens=True, truncation="only_second"`. One document becomes N rows; you score spans in each and merge by character offset.
     - **Correct out-of-window labelling** — a window that does not contain the answer must be labelled `(cls_index, cls_index)` **by design**, and you need a threshold for the no-answer decision (SQuAD 2.0 style) or the model will still answer from a window that does not contain it.
     - **Span decoding across windows** — collect the top `n_best_size` spans per window with their scores, de-duplicate by answer text, and pick the global best. Do not take the argmax of one window.
  4. **Then re-evaluate.** SQuAD-trained QA models degrade on out-of-domain long documents even with correct windowing — expect a large drop, and consider fine-tuning on 500–2,000 of your own labelled question-answer spans.
  5. **Also check the offset mapping.** For long contexts, `offset_mapping` values are in the coordinates of the *cleaned, tokenized* text. If the document had HTML or unusual whitespace, offsets drift and answers decode at the wrong place. Assert `context[offsets[i][0]:offsets[i][1]] == tokenizer.convert_ids_to_tokens(ids[i])` semantically on a sample.
- **Why the order matters:** "Empty string" sounds like a decoding bug but is usually a truncation-plus-label-design bug, and the fix is data-side, not decode-side.

**Q104. You deploy an int8 ONNX version of your BERT classifier and accuracy drops 4 points, but the PyTorch model is fine. What went wrong?**
- **Answer:** A 4-point drop from int8 is well outside the expected <1 point, so this is not normal quantization error. In order:
  1. **The export itself, not the quantization.** Run the *fp32 ONNX* model (before quantization) and compare to PyTorch on 500 held-out rows. If fp32 ONNX already loses points, the export is the problem: a wrong `task` passed to `optimum-cli export onnx` (e.g. `feature-extraction` instead of `text-classification`), a missing pooling layer, or the wrong output tensor being read (`logits` vs `last_hidden_state`).
  2. **The head is not in the graph.** Exporting `AutoModel` instead of `AutoModelForSequenceClassification` produces a graph with no classifier, so you are classifying from pooled embeddings. The predictions look plausible and are wrong.
  3. **Tokenizer mismatch at serve time.** The ONNX directory may have a different `tokenizer.json` (or none, falling back to a default). Compare `len(tokenizer)` between the PyTorch and ONNX directories.
  4. **`id2label` lost in the export.** The `config.json` in the ONNX directory may have reverted to `LABEL_0`/`LABEL_1`, so the *labels* are permuted even if the logits are identical. Compare raw logits, not label strings.
  5. **Dynamic quantization applied to the wrong ops.** Quantizing `MatMul`/`Gemm` in the attention path is standard; quantizing the LayerNorm or softmax path is not. `optimum-cli onnxruntime quantize` defaults are sane — but the `--avx512` / `--avx2` / `--arm64` flag matters: using an instruction set the host does not support silently falls back, and using arm64 flags on x86 produces a model that runs but with wrong kernels in some runtimes. Match the flag to the host.
  6. **Truly aggressive quantization.** If you used static quantization with a bad calibration set (calibrating on padded or truncated inputs), the activation ranges are wrong. Recalibrate on 200–500 *representative* real inputs.
  7. **Comparison methodology.** Verify you are comparing on the *same* 500 rows with the *same* preprocessing. A "4-point drop" is frequently an evaluation bug.
- **The discipline:** always run a **PyTorch vs ONNX agreement check** as part of the export pipeline and fail the build if agreement is below 99%. That check makes this whole class of incident a non-event. → CH-07 §11.

---

## Rapid Fire — True / False / One-Liner

| Statement | Verdict + why |
|---|---|
| BERT can generate text if you prompt it correctly. | **False.** No causal mask ⇒ no `P(x_t \| x_<t)`. |
| BERT-large has 24 encoder layers. | **True.** 24 layers, 1024 hidden, 16 heads, 340M params. |
| `[MASK]` appears in your fine-tuning data. | **False.** It is a pretraining artifact and never appears downstream. |
| The classification head is pretrained. | **False.** Randomly initialised; that is why step-0 loss ≈ `ln(C)`. |
| Fine-tuning LR should be ~2e-5. | **True.** Range 1e-5–5e-5; near 1e-3 destroys the model. |
| `-100` means "class index 100 is invalid". | **False.** It is `ignore_index` — a sentinel with no class meaning. |
| `[PAD]` id is 0 and that is also the `O` tag. | **True** — and that collision is the reason `-100` exists. |
| Continuation subwords should get the same label as the first subword. | **False.** They get `-100` (or `I-X` under `label_all_tokens`). |
| Token accuracy is a fine NER metric. | **False.** An all-`O` model scores ~0.85 and 0.00 entity F1. |
| `seqeval` scores whole entities. | **True.** Type and both boundaries must match exactly. |
| NSP is still used in modern encoders. | **False.** RoBERTa removed it; ALBERT replaced it with SOP; ELECTRA/DeBERTa dropped it. |
| The `[CLS]` vector is the best sentence embedding. | **False.** Mean pooling beats it for similarity; `[CLS]` is for classification. |
| BERT's position embeddings extrapolate past 512. | **False.** Learned absolute embeddings; index 512 does not exist. |
| `DataCollatorWithPadding` is correct for NER. | **False.** It pads labels with 0 = `O`. Use `DataCollatorForTokenClassification`. |
| Dynamic padding is a quality technique. | **False.** It is a throughput technique (2–20×) with no quality change. |
| Warmup should be ~10% of total steps. | **True.** Protects the pretrained weights from a cold first step. |
| `AdamW` and `Adam(weight_decay=...)` are the same. | **False.** AdamW is decoupled; the other is L2-in-gradient. |
| Gradient clipping at 1.0 is standard for BERT. | **True.** `max_grad_norm=1.0`, applied after backward and before step. |
| You must save the tokenizer alongside the model. | **True.** Weights without the matching tokenizer are unusable. |
| `id2label` defaults to `POSITIVE`/`NEGATIVE`. | **False.** It is `LABEL_0`/`LABEL_1` unless you set it. |
| Fine-tuned BERT is cheaper per prediction than an LLM API. | **True.** ~$0.10 vs ~$36–$600 per 1M. |
| BERT is obsolete for closed-label classification. | **False.** It remains the standard production choice at volume. |
| SetFit works with ~8 examples per class. | **True.** Contrastive sentence-transformer fine-tune + a logistic-regression head. |
| DeBERTa-v3-base is smaller *and* better than RoBERTa-large on some tasks. | **True.** 86M vs 355M; architecture and objective beat size. |
| ModernBERT supports 8,192 tokens. | **True.** RoPE + alternating local/global attention, Dec 2024. |
| LoRA on BERT should use LR 2e-5 like full fine-tuning. | **False.** LoRA needs 1e-4–3e-4 — 10× higher. |
| Multi-label classification works with the default `problem_type`. | **False.** You need `multi_label_classification` (BCE) or recall collapses. |
| `num_labels` is inferred from the data if omitted. | **False.** It defaults to 2 for BERT. |
| A BERT fine-tune needs ~1,000+ optimizer steps to converge the head. | **True.** Below ~500 steps, expect underfitting. |
| Softmax scores are calibrated probabilities. | **False.** Typical ECE 0.10–0.25 on a small fine-tune. |
| ONNX int8 typically costs <1 point of accuracy. | **True** — but a 4-point drop means the export is broken, not the quantization. |
| Extractive QA can hallucinate the answer text. | **False.** The answer is a substring of the input by construction. |
| `doc_stride` should exceed the longest expected answer. | **True.** Otherwise answers straddling a window boundary are lost. |
| `n_best_size=1` is the recommended QA setting. | **False.** 20 is standard; 1 costs 5–10 F1. |
| Encoder–decoder models like T5 can be dropped into `BertForSequenceClassification`. | **False.** Different architecture and head names. |
| `word_ids()` works on `BertTokenizer`. | **False.** Fast tokenizers only. |

---

## Coding / Whiteboard Tasks

**T1. Write the NER label-alignment function, from scratch.**
```python
def align_labels(batch_tokens, batch_tags, tokenizer, max_length=128):
    """Return an encoding whose 'labels' are aligned to subwords, with -100."""
    enc = tokenizer(batch_tokens, truncation=True, max_length=max_length,
                    is_split_into_words=True)          # (1) flag required
    aligned = []
    for i, tags in enumerate(batch_tags):
        word_ids = enc.word_ids(batch_index=i)         # (2) fast tokenizer only
        prev, row = None, []
        for w in word_ids:
            if w is None:                              # (3) [CLS]/[SEP]/[PAD]
                row.append(-100)
            elif w != prev:                            # (4) first subword of a word
                row.append(tags[w])
            else:                                      # (5) continuation subword
                row.append(-100)
            prev = w
        aligned.append(row)
    enc["labels"] = aligned                            # (6) same length as input_ids
    return enc
```
**Graded:** item (1) the `is_split_into_words` flag; item (3) the `None` branch; item (4) the `w != prev` comparison with `prev` updated *outside* the conditional; item (6) length equality with `input_ids`. Missing any one of these produces a model that trains and scores badly.

**T2. Write a batched inference function for a classification model.**
```python
def predict(texts, model, tok, batch_size=64, max_length=256):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sort
    out = [None] * len(texts)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]                                 # (1) batch, not row
        enc = tok([texts[j] for j in idx], truncation=True,
                  padding=True, max_length=max_length,
                  return_tensors="pt")                                 # (2) dynamic padding
        with torch.inference_mode():                                  # (3) cheaper than no_grad
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for k, j in enumerate(idx):
            out[j] = probs[k]                                          # (4) restore order
    return np.array(out)
```
**Graded:** batching rather than looping; length-sorting for padding efficiency; restoring the original order; `inference_mode`/`no_grad`. The notebook's per-text loop is 20–50× slower — that is the point of the question.

**T3. Write `compute_metrics` for token classification using `seqeval`.**
```python
import numpy as np
from evaluate import load as load_metric
seqeval = load_metric("seqeval")

def make_compute_metrics(label_names):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred                       # (B, S, C) and (B, S)
        preds = np.argmax(logits, axis=-1)
        true_preds, true_labels = [], []
        for p_row, l_row in zip(preds, labels):
            # seqeval needs STRINGS and must never see -100
            keep = [k for k, l in enumerate(l_row) if l != -100]
            true_preds.append([label_names[p_row[k]] for k in keep])
            true_labels.append([label_names[l_row[k]] for k in keep])
        r = seqeval.compute(predictions=true_preds, references=true_labels)
        return {"precision": r["overall_precision"],
                "recall":    r["overall_recall"],
                "f1":        r["overall_f1"],
                "accuracy":  r["overall_accuracy"]}
    return compute_metrics
```
**Graded:** the `-100` filter applied to **both** predictions and labels with the **same index set**; conversion to strings; `axis=-1`. A candidate who filters only one side, or passes ints, silently mis-scores.

**T4. Write the QA post-processing that converts token indices back to a character span.**
```python
def tokens_to_char_span(offset_mapping, start_idx, end_idx):
    """offset_mapping: list of (char_start, char_end) from the tokenizer."""
    if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
        return None
    cs = offset_mapping[start_idx][0]
    ce = offset_mapping[end_idx][1]
    if cs is None or ce is None or ce <= cs:
        return None
    return cs, ce
```
**Graded:** bounds checking, the `ce <= cs` guard (a zero-length span means a special token or a bad pair), and awareness that special tokens carry `(0,0)` offsets which must be excluded before this call. The follow-up is "how do you handle a span that crosses a window boundary?" — answer: with `doc_stride` overlap, score the span in whichever window fully contains it, and de-duplicate by answer text.

**T5. Compute the VRAM for a BERT-base fine-tune and say whether it fits in 8 GB.**
```
Given: BERT-base 110M params, batch 16, max_length 256, fp32, full fine-tuning.

Weights            110e6 * 4 B                    = 0.44 GB
Gradients          110e6 * 4 B                    = 0.44 GB
AdamW m, v         110e6 * 8 B                    = 0.88 GB
Activations        B * S * L * H * ~34 * 4 B
                 = 16 * 256 * 12 * 768 * 34 * 4   = 5.13 GB
Framework / cuDNN                                 = 0.50 GB
------------------------------------------------------------
TOTAL                                             = 7.39 GB   -> fits in 8 GB, barely
```
**Graded:** the 16-bytes-per-parameter rule (4 weights + 4 grads + 8 AdamW), and the activation term which dominates. **Follow-ups:** with `fp16=True` the activation term halves → **4.8 GB**, comfortable. With `per_device_train_batch_size=32` at fp32 it is 11.5 GB → OOM. With `gradient_checkpointing=True` the activation term drops ~5× (at ~25% wall-clock cost) → ~2.4 GB total.

**T6. Estimate the cost of 50M predictions: fine-tuned BERT on CPU int8 vs GPT-4o-mini.**
```
Assume: 200 input tokens, 10 output tokens per prediction.

BERT-base int8 on 8 vCPU (c7i.2xlarge @ $0.357/hr on-demand):
  throughput ~700 preds/s
  50e6 / 700 = 71,429 s = 19.8 hr
  cost = 19.8 * $0.357 = $7.08
  => $0.14 per 1M

GPT-4o-mini ($0.15/1M in, $0.60/1M out):
  input  200 * 0.15/1e6 = $0.00003
  output  10 * 0.60/1e6 = $0.000006
  per call = $0.000036
  50e6 * 0.000036 = $1,800
  => $36 per 1M

Ratio: 1,800 / 7.08 = 254x
```
**Graded:** showing the arithmetic, using real prices, and stating the annualised figure ($21,600/yr vs $85/yr at 50M/month). The bonus answer notes the one-time costs BERT carries (labelling, engineering) and the breakeven at roughly **1M predictions/month**.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value |
|---|---|
| BERT-base | 110M params, 12 layers, 768 hidden, 12 heads, 3072 FFN, 512 max |
| BERT-large | 340M params, 24 layers, 1024 hidden, 16 heads, 4096 FFN, 512 max |
| WordPiece vocabulary | 30,522 (uncased) / 28,996 (cased) |
| Special token ids | `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103 |
| `ignore_index` | **-100** |
| MLM mask rate | 15% selected; **80/10/10** mask/random/unchanged |
| Per-task mask rates (BERT paper) | NER 0.10, classification 0.15, SQuAD 0.30 |
| NSP | 50% IsNext / 50% NotNext; **removed in RoBERTa** |
| VRAM per parameter (full FT fp32) | **16 bytes** (4 weights + 4 grads + 8 AdamW) |
| Optimizer state, frozen params | 0 bytes — freezing saves 8 B/param |
| Fine-tuning LR | **2e-5** (range 1e-5 – 5e-5) |
| LoRA LR on BERT | 1e-4 – 3e-4 |
| Full-SFT LLM LR | 1e-5 – 2e-5 (often 2e-5) |
| Epochs | 2–4 (encoder); 1–3 (LLM SFT) |
| Batch size | 16–32 (encoder); 8–64 with accumulation |
| Warmup | 10% of total steps |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Optimizer | `adamw_torch` |
| Healthy optimizer steps | 1,000 – 10,000 |
| Minimum viable steps | ~500 (below this, expect underfitting) |
| CoNLL-2003 `O` token share | ~83% |
| Good NER F1 (CoNLL-2003) | 0.92–0.93 entity F1 (BERT-base) |
| WikiAnn-en labels | **7** (`O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`) |
| CoNLL-2003 labels | 9 (adds `B-MISC`, `I-MISC`) |
| QA standard config | `max_length=384`, `doc_stride=128`, `n_best_size=20`, `max_answer_length=30` |
| SQuAD 2.0 good F1 | 0.88–0.92 (DeBERTa-v3-large ~0.90) |
| DistilBERT | 66M, 6 layers, 60% faster, 40% smaller, 97% of GLUE |
| RoBERTa-base / large | 125M / 355M, 160 GB text, ~2T tokens, 50,265 vocab |
| DeBERTa-v3-base / large | 86M / 304M, 128,100 vocab |
| ModernBERT | 149M / 395M, 8,192 context, Dec 2024 |
| mBERT / XLM-R | 178M / 270M(base) / 550M(large), 119k / 250k vocab |
| BERT inference latency | 10–30 ms GPU, 50–200 ms CPU fp32, **15–60 ms CPU int8** |
| BERT throughput | ~3,000 preds/s GPU (fp16, batch 32); ~700 preds/s CPU int8 (8 vCPU) |
| Cost per 1M predictions | BERT **$0.09–0.14**; GPT-4o-mini $36; GPT-4o $600 |
| Cost per call | 200 input + 10 output: GPT-4o-mini $0.000036; GPT-4o $0.0006 |
| Breakeven volume | ~1M predictions/month |
| Latency p50 | encoder 25 ms; LLM API 700–1,200 ms |
| 95% CI on accuracy at n=500 | ±3.1 points |
| 95% CI on F1 at n=3,000 | ±1.5 points |
| Typical ECE, small fine-tune | 0.10–0.25 |
| Typical fitted temperature | 1.5–3.0 |
| Labelling cost, 12k examples | ~50 hours ≈ $1,250 |
| Training cost, same run | **$0.06** |

---

## Answers To The Self-Check Questions From CS-07

*(Mirrors §19 of CS-07.)*

**S1. Why can BERT not generate text? Name the exact architectural component that is absent and explain what it does in a decoder.**
The **causal attention mask** is absent. In a decoder, `M[i,j] = -inf for j > i`, so position *i* attends only to positions ≤ *i*. That makes `P(x_t | x_<t)` well-defined and makes the training objective (predict the next token) identical to the inference procedure (generate the next token). BERT's mask has no causal component — token *i* attends to all *j*, including *j > i*. There is therefore no next-token distribution to sample from, and appending a generated token would change the representation of every token before it, so autoregressive decoding would not even be self-consistent.

**S2. A colleague's NER model reports 0.91 token accuracy and 0.03 entity F1. Give three distinct root causes and the diagnostic for each.**
(a) **Continuation subwords not masked with `-100`** — the entity is counted 2–4 times and the model learns smeared boundaries. Diagnostic: print the per-row count of `-100`; it should cover specials *and* continuations, not just the 3 specials. (b) **The wrong collator** — `DataCollatorWithPadding` pads labels with `0` (= `O`), so pads are trained and the loss is contaminated. Diagnostic: check that the minimum label value in a batch is `-100`. (c) **`num_labels` or the label order is wrong** — e.g. a hand-typed 9-name list against WikiAnn's 7 labels, or a reordered list. Diagnostic: `assert model.config.num_labels == len(label_names)` and compare `id2label` to the dataset's `ClassLabel.names`. Bonus fourth: **token-level scoring is the wrong metric** — recompute with `seqeval`; if entity F1 jumps to 0.6+, the "bug" was the metric.

**S3. You have 40 labelled examples per class across 8 classes and no labelling budget. What do you do, and why not fine-tune BERT?**
Use **SetFit**: contrastively fine-tune a sentence transformer on pairs sampled from the 320 examples, then fit a logistic-regression head on the resulting embeddings. Do not fine-tune BERT because 320 rows at batch 8 is **40 optimizer steps per epoch** — an order of magnitude below the ~500-step minimum for the head to converge — so you will either underfit or memorise. SetFit reports GPT-3-few-shot-level accuracy at 8 examples/class, the head refits in seconds when the label set changes (unlike BERT, where a new class changes `num_labels` and forces a retrain), and inference is a sentence-transformer forward pass plus a dot product.

**S4. Explain the 80/10/10 masking rule. What specifically breaks if you use 100% `[MASK]`?**
Of the 15% of tokens selected for MLM, 80% become `[MASK]`, 10% become a random vocabulary token, and 10% are left unchanged; the loss is computed on all 15%. With 100% `[MASK]` the model learns (a) that the only positions requiring careful encoding are those containing the literal `[MASK]` token, and (b) that everywhere else it may copy the input. Since `[MASK]` never appears at fine-tuning time, (a) never activates, so every downstream representation is produced by a network that has never had to encode a corrupted context. The 10% random token forces the model to detect and correct a wrong input; the 10% unchanged keeps some loss-bearing positions in a clean context. The strongest evidence that `[MASK]` was a defect is that BERT's authors tuned the mask rate per downstream task (0.10 NER, 0.15 classification, 0.30 SQuAD).

**S5. Your QA model returns an empty string for 30% of questions. List the four most likely causes in the order you would check them.**
(a) **`argmax` selected a `[PAD]` position** — you did not filter `attention_mask == 0`. Diagnostic: check whether the argmax indices land in the padded region. (b) **The answer was truncated away** — the context exceeded 512 tokens and you set no `doc_stride`. Diagnostic: log the token-length distribution of your contexts. (c) **The `(0,0)` collapse** — training labels defaulted to 0 for unanswerable/truncated rows, teaching the model to point at `[CLS]`. Diagnostic: count training rows with `start_positions == 0`; if >2%, this is it. (d) **The `start > end` repair manufactured a whitespace span** — print the raw `(start_idx, end_idx)` pairs and switch to proper `n_best` span enumeration with `start <= end` enforced during enumeration.

**S6. Derive the number of optimizer steps for 12,000 training rows with batch size 32 over 3 epochs, and state how many of those steps are warmup at `warmup_ratio=0.1`. Is this a healthy number of steps?**
`steps = (12,000 / 32) × 3 = 375 × 3 = 1,125`. Warmup = `floor(0.1 × 1,125) = 112` steps of linear ramp, then 1,013 steps of linear decay. **Yes, 1,125 is healthy** — above the ~1,000-step floor and well below the point where 12k rows overfit an encoder. Compare the video's demo: 1,000 rows at batch 8 for 1 epoch = **125 steps**, which is an order of magnitude short and is why its predictions are unreliable.

**S7. Your fine-tuned BERT scores 0.94 on your 500-row test set, but a colleague's identical run scores 0.91. Give four innocent explanations before you conclude anything is wrong.**
(a) **Different random seed** — model init, dropout, and data order all vary; ±1–2 points is routine. (b) **Different test rows** — the notebook's `np.random.choice` has no `random_state`, so a 3-point swing from a different 500 rows is entirely expected. (c) **Different split** — one run may be evaluating on an IMDB *test* subset and the other on a validation carve-out of train. (d) **Different `max_length` or padding policy**, which changes how much of each review the model sees. Before concluding there is a bug, fix `seed=42`, freeze the same test set to disk, unify the tokenizer config, and report a Wilson 95% CI — **±3.1 points at n=500**, which covers the whole gap.

**S8. Why does `DataCollatorForTokenClassification` matter, and what exactly goes wrong if you use `DataCollatorWithPadding` for NER instead?**
`DataCollatorForTokenClassification` pads `input_ids`/`attention_mask` **dynamically to the batch maximum** — a 17–20× compute reduction on short sentences — and pads `labels` with **`label_pad_token_id=-100`**, which `CrossEntropyLoss` ignores entirely. With `DataCollatorWithPadding`, labels are padded with `tokenizer.pad_token_id = 0`, and **0 is a valid tag (`O`)**. Three consequences: the loss includes every pad position and is no longer comparable across batches with different padding; the model is trained to emit `O` on padding, biasing the decision boundary against rare entity types; and any label-count-based evaluation is wrong. None of these crash, and all cost a few points of F1.

**S9. You are choosing between `bert-base-uncased`, `distilbert-base-uncased`, `deberta-v3-base`, and `answerdotai/ModernBERT-base` for a 3-class classification task on 8,000 rows at 4M predictions/day on CPU. Which, and why? What would change your answer?**
**`distilbert-base-uncased`** (or ModernBERT-base if the serving stack supports Flash Attention and the extra 149M/83M params are affordable). Reasoning: 4M/day = 120M/month, so unit cost and CPU latency dominate the decision. DistilBERT is 40% smaller and 60% faster than BERT-base at ~97% of GLUE quality; with ONNX int8 it serves roughly 1,200–1,800 preds/s on an 8-vCPU box versus ~700 for BERT-base, halving the fleet. `bert-base-uncased` is the safe fallback if DistilBERT's gap shows on your eval. `deberta-v3-base` is the quality play but ~1.6× slower than BERT-base on CPU — wrong when throughput binds. **What would change the answer:** if the eval shows DistilBERT more than ~1.5 points of macro F1 below DeBERTa-v3-base, take the cost hit and use DeBERTa-v3-base or `deberta-v3-xsmall`; if inputs exceed 512 tokens, ModernBERT is the only option in the list; if 8,000 rows proves too few for the head to converge, the balance shifts toward the larger pretrained models.

**S10. RoBERTa removed NSP and beat BERT. Explain the mechanism by which removing an objective *improves* a model.**
Removing an objective removes a **shortcut** the encoder was spending capacity on. NSP's negatives come from a different document, so a topic-shift detector achieves near-perfect NSP accuracy without learning anything about discourse — `A` about cats, `B` about interest rates ⇒ `NotNext`. That shortcut occupies representational capacity and, worse, **shapes the `[CLS]` vector**, since the NSP head read exactly that vector. Removing NSP frees the capacity for MLM and removes the topic-detection bias from `[CLS]`, which is the representation downstream classifiers read. RoBERTa's ablation showed a slight downstream *improvement*. ALBERT then showed that a *harder* sentence-level objective — SOP, distinguishing `A-B` from `B-A` within the same document, where no topic shift is available — does carry useful signal. The general principle: **an auxiliary objective helps only if it cannot be solved by a shortcut.**

