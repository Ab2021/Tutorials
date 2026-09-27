# CH-07 — Fine-Tuning BERT: NER, Sentiment, QA Cheat Sheet

**One-line purpose:** Everything you need to fine-tune, evaluate, debug, and serve an encoder-only model for classification, token classification, and extractive QA — with the numbers.
**Use when:** the task is a **closed-label** prediction over text (classify, tag spans, extract a span) and you have ≥ 1,000 labelled examples and ≥ 1M predictions/month.
**Do NOT use when:** you need generation, multi-step reasoning, or free-form output (→ CS-13, CS-14); you have < 100 labels total (→ SetFit, §3); or the label set changes weekly (→ a refittable head, §8).

---

## 1. The 10-Second Summary

1. **BERT is encoder-only.** No causal mask ⇒ token *i* sees token *j > i* ⇒ no `P(x_t | x_<t)` exists. It cannot generate. Ever. → §2.1
2. **Three heads cover 95% of encoder work:** sequence classification (`[CLS]` → Linear), token classification (every token → Linear), span QA (every token → two Linear heads for start/end).
3. **The #1 silent bug is subword alignment.** `word_ids()`, first-subword-only, everything else `-100`. It trains fine and scores 0.03 entity F1.
4. **`-100` is `ignore_index`, not a class.** `[PAD]`=0 collides with the `O` tag — that collision is why `-100` exists.
5. **512 tokens is a hard ceiling** (learned absolute position embeddings). Long docs ⇒ sliding windows with stride.
6. **Fine-tuning LR is 2e-5**, not 2e-4. The head is random; the encoder is precious.
7. **The metric is usually the bug.** `seqeval` for NER, F1-macro for imbalanced classification, EM/F1 for QA — never accuracy alone.
8. **Cost: ~$0.10 per 1M predictions.** The same work via GPT-4o-mini is ~$36, via GPT-4o ~$600. That 250–6,000× gap is the entire business case.
9. **Deploy with ONNX int8.** 2–4× CPU throughput, < 1 point of accuracy loss, one CLI command.
10. **Always verify.** PyTorch-vs-ONNX agreement ≥ 99%, entity F1 (not token accuracy), and a calibrated threshold before any auto-action.

---

## 2. Core Formulas

### 2.1 The architectural fact

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Self-attention | $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ | $d_k$ = head dim | BERT-base: $d_k = 768/12 = 64$, so $\sqrt{d_k}=8$ |
| Decoder causal mask | $M_{ij} = -\infty$ for $j>i$, else $0$ | $M$ added to scores pre-softmax | BERT has **no such mask** ⇒ cannot generate |
| Attention cost | $O(n^2 d)$ | $n$ = seq len, $d$ = hidden | $n$=512, $d$=768: $512^2 \times 768 = 201\text{M}$ MACs **per head per layer** |
| Params per layer | $4d^2 + 8d^2 + 2d^2 \cdot 4$ (approx $12d^2$) | $d$ = hidden | $12 \times 768^2 = 7.08\text{M}$; $\times 12$ layers ≈ 85M + emb |

### 2.2 Parameter count, derived (memorize the shape of this)

```
BERT-base, d=768, L=12, V=30522, S=512:
  Emb   = V*d + S*d + 2d          = 23.44M + 0.39M + 0.002M  = 23.83M
  Layer = 4d^2 (attn) + 2*d*4d (ffn) + 8d (2 LayerNorms)
        = 2.36M + 4.72M + 0.006M                              =  7.09M
  x12 layers                                                  = 85.05M
  Pooler = d^2 + d                                            =  0.59M
  TOTAL                                                       ~= 110M
Rule: params scale as d^2 per layer, so 2x hidden = 4x params per layer.
```

### 2.3 MLM corruption arithmetic

| Quantity | Formula | Value at 15%, S=512 |
|---|---|---|
| Tokens selected | $0.15 \times S$ | 76.8 → 77 |
| Replaced with `[MASK]` | $0.80 \times 77$ | ~62 |
| Replaced with random token | $0.10 \times 77$ | ~8 |
| Left unchanged | $0.10 \times 77$ | ~8 |
| Loss-bearing positions | 77 of 512 | **15.0%** |

Per-task mask rates from the BERT paper: **NER 0.10 · classification 0.15 · SQuAD 0.30**. The fact that they differ per task is itself the evidence that `[MASK]` was a defect, not a feature.

### 2.4 Loss functions

| Task | Loss | Head | Notes |
|---|---|---|---|
| Sequence classification | `CrossEntropyLoss` | `Linear(d, C)` on `[CLS]` | `num_labels=C`; step-0 loss ≈ $\ln C$ |
| Multi-label classification | `BCEWithLogitsLoss` | `Linear(d, C)` + sigmoid | `problem_type="multi_label_classification"` |
| Regression | `MSELoss` | `Linear(d, 1)` | `problem_type="regression"` |
| Token classification | `CrossEntropyLoss(ignore_index=-100)` | `Linear(d, 2T+1)` | `T` entity types |
| Extractive QA | `CE(start) + CE(end)` | 2 × `Linear(d, 1)` | Two **independent** softmaxes |

### 2.5 Optimizer-step arithmetic (memorize this)

```
steps/epoch = ceil(N_train / (per_device_batch * grad_accum * n_gpu))
total_steps = steps/epoch * epochs
warmup      = int(warmup_ratio * total_steps)

Example: N=12,000, batch=32, 1 GPU, 3 epochs
  steps/epoch = 12000/32 = 375
  total_steps = 375 * 3 = 1,125          <- healthy (target 1,000-10,000)
  warmup      = 0.1 * 1125 = 112 steps linear ramp, then 1,013 linear decay

Counter-example from the video: N=1,000, batch=8, 1 epoch
  total_steps = 125                       <- 8x below the floor; underfits
```

### 2.6 VRAM

```
Weights+grads+AdamW = 16 bytes/param (fp32 full FT); frozen params cost 0
Activations ~ B * S * L * H * C      (C ~ 34 for BERT-base with dropout)
  -> halve S and activations halve AND attention MACs drop 4x

BERT-base, B=16, S=256, fp32: 0.44 + 0.44 + 0.88 + 5.13 + 0.50 = 7.39 GB (barely 8 GB)
  bf16/fp16: 4.8 GB  |  grad_checkpointing: ~2.4 GB (costs ~25% wall-clock)
```

### 2.7 Cost

| | Fine-tuned BERT (CPU int8) | GPT-4o-mini | GPT-4o |
|---|---|---|---|
| Per 1M predictions | **$0.09–0.14** | ~$36 | ~$600 |
| Per call (200 in / 10 out) | $0.00000014 | $0.000036 | $0.0006 |
| 50M predictions | $7 | $1,800 | $30,000 |
| Latency p50 | 25 ms | ~700 ms | ~1,100 ms |
| Breakeven vs API | — | ~1M/month | ~1M/month |

```
BERT-base int8, 8 vCPU c7i.2xlarge @ $0.357/hr, ~700 preds/s:
  50e6 / 700 = 71,429 s = 19.8 hr  ->  19.8 * 0.357 = $7.08
```

---

## 3. Decision Tree

```
What is the output shape?
│
├─ ONE label for the whole text (or a few independent labels)
│  │
│  ├─ Labels are mutually exclusive?
│  │   ├─ YES, balanced classes        -> AutoModelForSequenceClassification, num_labels=C
│  │   └─ YES, imbalanced (<5% minority) -> same, but PROBLEM_TYPE default,
│  │                                        metric = F1-macro, class weights, threshold tune
│  └─ NO, multiple labels can co-occur -> problem_type="multi_label_classification" (BCE)
│
├─ A LABEL FOR EACH TOKEN (names, spans, PII)
│   -> AutoModelForTokenClassification, num_labels = 2T+1 (BIO) or 4T+1 (BILOU)
│      collator = DataCollatorForTokenClassification
│      metric   = seqeval entity micro-F1  (NEVER token accuracy)
│
└─ A SPAN OF TEXT FROM THE INPUT (extractive answer)
    -> AutoModelForQuestionAnswering, 2 logits per token
       max_length=384, doc_stride=128, n_best_size=20, max_answer_length=30

Then, orthogonal:
│
├─ Labelled rows < 100 total, or < 8 per class?
│   -> STOP. Use SetFit (contrastive + logistic head). Fine-tuning BERT here underfits.
│
├─ Text > 512 tokens?
│   ├─ Classification -> chunk with stride, train on chunks, infer by max-pooling
│   │                    (or use ModernBERT at 8,192 if serving supports it)
│   └─ QA -> sliding window with doc_stride, merge spans by char offset
│
├─ Predictions > 1M/month?
│   -> Yes: fine-tuned encoder is 250x-6,000x cheaper. This is the whole point.
│   -> No (< 10k/month): an LLM API may be cheaper once you price the labelling
│      and engineering. Do the arithmetic before you commit.
│
└─ Label set changes every few weeks?
    -> Train the encoder ONCE, cache embeddings, refit a logistic-regression head
       (sklearn) on new labels. New labels cost seconds, not a retrain.
```

---

## 4. Hyperparameter Quick Reference

| Param | Default / typical | Sweep range | Effect & why |
|---|---|---|---|
| `learning_rate` | **2e-5** | 1e-5 – 5e-5 | The single most important value. 1e-3 destroys pretrained weights (catastrophic forgetting); 1e-6 underfits. |
| `num_train_epochs` | 3 | 2 – 4 | 1 underfits; >5 overfits on <50k rows. Watch eval loss, not train loss. |
| `per_device_train_batch_size` | 16 | 8 – 32 | Bigger batch = smoother gradients, more VRAM. Use grad accumulation to decouple memory from effective batch. |
| `gradient_accumulation_steps` | 1 | 1 – 8 | Effective batch = per_device × accum × n_gpu. Target 32–64 effective. |
| `max_length` / `max_seq_length` | 128 (class.), 384 (QA) | measure p99 | Quadratic cost. The most common 3× waste is padding to 512 when p99 is 90. |
| `warmup_ratio` | **0.1** | 0.06 – 0.1 | 10% of steps linear ramp. Protects the encoder from a cold, high-variance first step. |
| `weight_decay` | **0.01** | 0 – 0.1 | Applied to weights only (not bias/LayerNorm). 0.1 on small datasets. |
| `max_grad_norm` | **1.0** | 0.5 – 1.0 | Gradient clipping. Prevents a single bad batch from spiking the loss to NaN. |
| `optim` | `adamw_torch` | `adafactor` if VRAM-tight | AdamW = decoupled decay. Adafactor drops the second moment (~0.5 B/param). |
| `lr_scheduler_type` | `linear` | `cosine` | Linear is the BERT-paper default; cosine gives ~+0.2 on long runs. |
| `fp16` / `bf16` | `bf16` on Ampere+ | — | ~1.8× throughput. `bf16` has no scaler and no NaN-on-LayerNorm risk. |
| `eval_strategy` / `eval_steps` | `steps` / 100–500 | — | You need 5–10 eval points to see the turn. |
| `save_total_limit` | 2 | — | Keeps disk sane; `load_best_model_at_end=True` picks the best. |
| `metric_for_best_model` | `f1` / `seqeval` F1 | — | Set this explicitly, or you select on `loss` and get a worse model. |
| `early_stopping_patience` | 2 | 1 – 3 | Stops at the turn instead of overfitting past it. |
| `label_smoothing_factor` | 0.0 | 0 – 0.1 | Reduces overconfidence/ECE; small accuracy cost. |
| `seed` | 42 | — | **Always set it.** Otherwise your A/B comparison is noise. |
| `dataloader_num_workers` | 4 | 2 – 8 | 0 starves the GPU on Windows/small machines. |
| `gradient_checkpointing` | False | True if OOM | Activations ÷ ~5, wall-clock × ~1.25. |
| `report_to` | `"none"` for scripts | `"tensorboard"` | `"none"` disables the writer — do not then try to read TB logs. |
| LoRA `r` / `alpha` / LR | 8 / 16 / **2e-4** | 4–16 | LoRA LR is **10× the full-FT LR** — the adapter is randomly initialised. |
| LoRA `target_modules` | `["query","value"]` | + `key`,`dense` | q/v gives ~90% of the benefit. |
| LoRA `modules_to_save` | `["classifier"]` | required | Omit this and you train adapters onto a random frozen head. |

---

## 5. Copy-Paste Code Snippets

### 5.1 Task 1 — Sentiment / sequence classification (complete, runnable)

```python
# pip install "transformers>=4.44" "datasets>=2.20" "evaluate>=0.4" accelerate scikit-learn
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from evaluate import load as load_metric

MODEL = "bert-base-uncased"
ds = load_dataset("imdb")                       # 25k train / 25k test, label 0=neg 1=pos
tok = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=256)

ds = ds.map(preprocess, batched=True)
# small stratified carve-outs so this runs in minutes
train = ds["train"].shuffle(seed=42).select(range(4000))
eval_ = ds["test"].shuffle(seed=42).select(range(1000))

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label=id2label, label2id=label2id)   # head is RANDOM

acc = load_metric("accuracy")
f1  = load_metric("f1")

def compute_metrics(ep):
    logits, labels = ep
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_pos":   f1.compute(predictions=preds, references=labels,
                               average="binary", pos_label=1)["f1"],
    }

args = TrainingArguments(
    output_dir="out/sentiment",
    learning_rate=2e-5, num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,          # effective batch 32
    warmup_ratio=0.1, weight_decay=0.01, max_grad_norm=1.0,
    optim="adamw_torch", lr_scheduler_type="linear",
    fp16=False, bf16=True,                  # bf16 on Ampere+; else fp16=True
    eval_strategy="steps", eval_steps=100, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="f1_macro",
    logging_steps=50, report_to="none", seed=42,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train, eval_dataset=eval_,
    data_collator=DataCollatorWithPadding(tok),   # dynamic padding
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("out/sentiment/final")
tok.save_pretrained("out/sentiment/final")        # ALWAYS save the tokenizer too

# Inference
from transformers import pipeline
clf = pipeline("text-classification", model="out/sentiment/final",
               tokenizer="out/sentiment/final", device=-1, top_k=None)
print(clf("This film was an absolute waste of two hours."))
# -> [{'label': 'NEGATIVE', 'score': 0.99}]   (labels come from id2label, set above)
```

### 5.2 Task 2 — NER / token classification (complete, runnable)

```python
# pip install seqeval
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
from evaluate import load as load_metric

MODEL = "bert-base-cased"                        # cased matters for names
ds = load_dataset("wikiann", "en")               # labels: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC
tok = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)

# --- 1. num_labels must come from the DATA, never hand-typed ---
label_names = ds["train"].features["ner_tags"].feature.names     # exactly 7 for WikiAnn
id2label = dict(enumerate(label_names))
label2id = {v: k for k, v in id2label.items()}
print(len(label_names), label_names)   # 7  ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']

# --- 2. THE alignment function. This is the whole task. ---
def tokenize_and_align(batch):
    enc = tok(batch["tokens"], truncation=True, max_length=128,
              is_split_into_words=True)              # REQUIRED: words, not strings
    all_labels = []
    for i, tags in enumerate(batch["ner_tags"]):
        word_ids = enc.word_ids(batch_index=i)        # fast tokenizers only
        prev, row = None, []
        for w in word_ids:
            if w is None:                             # [CLS] [SEP] [PAD]
                row.append(-100)
            elif w != prev:                           # FIRST subword of a word
                row.append(tags[w])
            else:                                     # continuation subword
                row.append(-100)
            prev = w
        all_labels.append(row)
    enc["labels"] = all_labels                        # same length as input_ids
    return enc

ds_enc = ds.map(tokenize_and_align, batched=True,
                remove_columns=ds["train"].column_names)

# --- 3. Sanity assertion: catch the silent bug before the GPU does ---
row = ds_enc["train"][0]
pairs = list(zip(tok.convert_ids_to_tokens(row["input_ids"]), row["labels"]))
print(pairs[:12])
assert len(row["labels"]) == len(row["input_ids"]), "ALIGNMENT BROKEN"
assert sum(1 for l in row["labels"] if l != -100) > 0, "no supervised positions"

# --- 4. seqeval, entity-level. Token accuracy is the lie. ---
seqeval = load_metric("seqeval")
def compute_metrics(ep):
    logits, labels = ep
    preds = np.argmax(logits, axis=-1)
    true_p, true_l = [], []
    for p_row, l_row in zip(preds, labels):
        keep = [k for k, l in enumerate(l_row) if l != -100]   # filter BOTH sides, same idx
        true_p.append([label_names[p_row[k]] for k in keep])
        true_l.append([label_names[l_row[k]] for k in keep])
    r = seqeval.compute(predictions=true_p, references=true_l)
    return {"precision": r["overall_precision"], "recall": r["overall_recall"],
            "f1": r["overall_f1"], "accuracy": r["overall_accuracy"]}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(label_names), id2label=id2label, label2id=label2id)

args = TrainingArguments(
    output_dir="out/ner", learning_rate=3e-5, num_train_epochs=3,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    warmup_ratio=0.1, weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
    eval_strategy="steps", eval_steps=200, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="f1", logging_steps=50,
    report_to="none", seed=42, bf16=True,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds_enc["train"].select(range(5000)),
    eval_dataset=ds_enc["validation"].select(range(1000)),
    data_collator=DataCollatorForTokenClassification(tok),   # pads labels with -100
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_pretrained("out/ner/final"); tok.save_pretrained("out/ner/final")

# --- 5. Inference with entity grouping (aggregation_strategy) ---
from transformers import pipeline
ner = pipeline("ner", model="out/ner/final", tokenizer="out/ner/final",
               aggregation_strategy="simple")    # merges B-/I- subwords into whole spans
print(ner("Sunny Savita works at Hugging Face in New York."))
# [{'entity_group':'PER','word':'Sunny Savita','start':0,'end':12,'score':0.99}, ...]
```

### 5.3 Task 3 — Extractive QA (complete, runnable)

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, default_data_collator)

MODEL = "deepset/bert-base-cased-squad2"      # fine-tuned further, or "bert-base-uncased" from scratch
ds = load_dataset("squad_v2")                  # SQuAD 2.0 includes unanswerable questions
tok = AutoTokenizer.from_pretrained(MODEL)

MAX_LEN, STRIDE = 384, 128                     # stride MUST exceed the longest answer

def prepare(batch):
    q = [x.lstrip() for x in batch["question"]]
    enc = tok(q, batch["context"], truncation="only_second", max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")
    enc["start_positions"], enc["end_positions"] = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        seq_ids = enc.sequence_ids(i)                  # 0=question 1=context None=special
        start, end = 0, 0                              # 0 == [CLS] == "no answer"
        if batch["answers"][sample_map[i]]["answer_start"]:
            a_start = batch["answers"][sample_map[i]]["answer_start"][0]
            a_text  = batch["answers"][sample_map[i]]["text"][0]
            a_end   = a_start + len(a_text)
            ci = [k for k, s in enumerate(seq_ids) if s == 1]
            if ci and offsets[ci[0]][0] <= a_start and offsets[ci[-1]][1] >= a_end:
                # answer fully inside THIS window -> label it
                while ci and offsets[ci[0]][1] <= a_start: ci.pop(0)
                start = ci[0]
                ci2 = [k for k, s in enumerate(seq_ids) if s == 1]
                while ci2 and offsets[ci2[-1]][0] >= a_end: ci2.pop()
                end = ci2[-1]
            # else: leave (0,0) -> window is "unanswerable", which is CORRECT, not a bug
        enc["start_positions"].append(start)
        enc["end_positions"].append(end)
    return enc

train_enc = ds["train"].select(range(8000)).map(
    prepare, batched=True, remove_columns=ds["train"].column_names)
eval_enc = ds["validation"].select(range(1000)).map(
    prepare, batched=True, remove_columns=ds["validation"].column_names)

model = AutoModelForQuestionAnswering.from_pretrained(MODEL)

args = TrainingArguments(
    output_dir="out/qa", learning_rate=3e-5, num_train_epochs=2,
    per_device_train_batch_size=8, gradient_accumulation_steps=2,   # 384 len needs less batch
    per_device_eval_batch_size=16, warmup_ratio=0.1, weight_decay=0.01,
    max_grad_norm=1.0, optim="adamw_torch", eval_strategy="steps", eval_steps=500,
    save_total_limit=2, report_to="none", seed=42, bf16=True,
)
Trainer(model=model, args=args, train_dataset=train_enc, eval_dataset=eval_enc,
        data_collator=default_data_collator).train()
model.save_pretrained("out/qa/final"); tok.save_pretrained("out/qa/final")

# --- Inference: sliding windows + n-best span search. NOT a single argmax. ---
def answer_question(question, context, model, tok, n_best=20, max_answer_len=30):
    enc = tok(question, context, truncation="only_second", max_length=MAX_LEN,
              stride=STRIDE, return_overflowing_tokens=True, return_offsets_mapping=True,
              padding=True, return_tensors="pt")
    offsets_all = enc.pop("offset_mapping")
    seq_ids_all = enc.pop("sequence_ids") if "sequence_ids" in enc else None
    import torch
    with torch.inference_mode():
        out = model(**enc)
    start_logits, end_logits = out.start_logits, out.end_logits
    candidates = []
    for w in range(start_logits.shape[0]):
        sl, el = start_logits[w], end_logits[w]
        s_idx = torch.topk(sl, n_best).indices.tolist()
        e_idx = torch.topk(el, n_best).indices.tolist()
        for s in s_idx:
            for e in e_idx:
                if e < s or (e - s) + 1 > max_answer_len:   # enforce the two rules
                    continue
                cs, ce = offsets_all[w][s][0], offsets_all[w][e][1]
                if cs is None or ce is None or ce <= cs:
                    continue
                candidates.append((float(sl[s] + el[e]), context[cs:ce]))
        # CLS score = the model's "no answer" vote (SQuAD 2.0)
        candidates.append((float(sl[0] + el[0]), ""))
    best = max(candidates, key=lambda t: t[0])
    return best[1]

print(answer_question("Where does Sunny work?", "Sunny Savita works at Hugging Face.", model, tok))
```

### 5.4 Common variations

```python
# (a) Multi-label classification (labels co-occur) -- recall collapses without this
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=6, problem_type="multi_label_classification")
# labels must be float multi-hot: torch.tensor([1., 0., 1., 0., 0., 1.])
# inference: torch.sigmoid(logits) then PER-LABEL thresholds (never one global 0.5)

# (b) Regression (e.g. a 1-5 quality score)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=1, problem_type="regression")
# loss becomes MSELoss automatically; labels = torch.tensor([4.0])

# (c) Freeze the encoder, train only the head (fast, weak, good baseline)
for p in model.bert.parameters():
    p.requires_grad = False
# sanity: the head must still be trainable
assert sum(p.requires_grad for p in model.parameters()) > 0

# (d) LoRA instead of full fine-tuning
from peft import LoraConfig, get_peft_model, TaskType
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                 target_modules=["query", "value"],
                 modules_to_save=["classifier"],          # DO NOT OMIT
                 task_type=TaskType.SEQ_CLS)
model = get_peft_model(model, cfg)
# then use LR 2e-4 (10x the full-FT LR), NOT 2e-5

# (e) Chunk a long document for classification, then max-pool
def chunk(text, tok, size=512, stride=128):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [ids[i:i + size - 2] for i in range(0, len(ids), size - stride)] or [[]]
# score each chunk as a separate row; at inference take the MAX sigmoid per label

# (f) Cache embeddings, refit a sklearn head -- new labels cost seconds
from sklearn.linear_model import LogisticRegression
with torch.inference_mode():
    emb = model.bert(**enc).last_hidden_state[:, 0]      # [CLS]
clf = LogisticRegression(max_iter=1000).fit(emb.numpy(), y)
```

### 5.5 Framework-specific

```python
# ONNX export + int8 quantization (the 2-4x CPU win)
# optimum-cli export onnx --model out/sentiment/final --task text-classification onnx/sentiment
# optimum-cli onnxruntime quantize --avx512 --onnx_model onnx/sentiment -o onnx/sentiment-int8

from optimum.onnxruntime import ORTModelForSequenceClassification
ort = ORTModelForSequenceClassification.from_pretrained("onnx/sentiment-int8")
from transformers import pipeline
clf = pipeline("text-classification", model=ort, tokenizer="onnx/sentiment-int8")

# HF Trainer alternative: use accelerate/DeepSpeed ZeRO-2 for multi-GPU
# accelerate launch --num_processes 4 train.py
# TrainingArguments(deepspeed="ds_config_zero2.json")   # optimizer state sharded
```

---

## 6. CLI Commands

```bash
# --- environment (the video's stack) ---
pip install "transformers>=4.44" "datasets>=2.20" "evaluate>=0.4" \
            accelerate seqeval scikit-learn optimum[onnxruntime] peft

# --- run a fine-tune from a script ---
python train.py --model bert-base-uncased --dataset imdb --epochs 3 --lr 2e-5

# --- multi-GPU / mixed precision ---
accelerate config                      # interactive; pick bf16, multi-GPU, no DeepSpeed
accelerate launch --num_processes 4 train.py

# --- ONNX export + quantize (the production path) ---
optimum-cli export onnx --model out/sentiment/final \
  --task text-classification onnx/sentiment
optimum-cli onnxruntime quantize --avx512 --onnx_model onnx/sentiment -o onnx/sentiment-int8
#   --avx512 | --avx2 | --arm64   MUST match the target host, or you get wrong kernels

# --- verify the export before you ship it (fail the build below 99%) ---
python - <<'PY'
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
m = "out/sentiment/final"; o = "onnx/sentiment-int8"
pt  = AutoModelForSequenceClassification.from_pretrained(m).eval()
ort = ORTModelForSequenceClassification.from_pretrained(o)
tok = AutoTokenizer.from_pretrained(m)
texts = ["great film", "terrible film", "it was fine", "loved every minute"]
enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.inference_mode():
    a = torch.argmax(pt(**enc).logits, -1).numpy()
    b = np.argmax(ort(**enc).logits, -1).numpy()
print("agreement:", (a == b).mean(), a, b)
assert (a == b).mean() == 1.0
PY

# --- serve ---
pip install text-embeddings-inference    # TEI, for embeddings/rerankers
# or: triton / ONNX Runtime Server / a FastAPI + ORT process pool

# --- push to the Hub (with a model card) ---
huggingface-cli login
huggingface-cli upload <user>/bert-sentiment-imdb out/sentiment/final --repo-type model
```

---

## 7. VRAM / Cost Calculator

### Training VRAM (full fine-tuning, fp32, batch 16, seq 256, 16 B/param)

| Model | Params | Weights+grads+AdamW | Activations | Total | Fits 8 GB? |
|---|---|---|---|---|---|
| MiniLM-L6-v2 | 22M | 0.35 GB | 2.6 GB | **3.0 GB** | Yes, comfortably |
| DistilBERT-base | 66M | 1.06 GB | 2.6 GB | **4.2 GB** | Yes |
| BERT-base | 110M | 1.76 GB | 5.13 GB | **7.4 GB** | Barely — use fp16 |
| BERT-base (fp16) | 110M | 1.76 GB | 2.6 GB | **4.8 GB** | Yes |
| BERT-base (+grad ckpt) | 110M | 1.76 GB | 1.0 GB | **3.3 GB** | Yes, at ~25% slower |
| DeBERTa-v3-base | 86M | 1.38 GB | ~4.0 GB | **5.9 GB** | Yes |
| RoBERTa-base | 125M | 2.00 GB | 5.13 GB | **7.6 GB** | Tight |
| BERT-large | 340M | 5.44 GB | 13.7 GB | **19.6 GB** | No — 24 GB card |
| ModernBERT-base | 149M | 2.38 GB | ~5.5 GB | **8.4 GB** | No at fp32; yes at bf16 (~5.9 GB) |

**Rules of thumb:** activations dominate and scale linearly with batch × seq × layers, so **halve `max_length` before you halve the batch** (quadratic attention + linear activations). Freeze the encoder and total VRAM drops by the 12 B/param of grads+AdamW (BERT-base: 1.76 GB → 0.44 GB of weights only).

### Training cost (the worked example from the video's 12,000-row scenario)

| Item | Detail | Cost |
|---|---|---|
| Rows | 12,000 labelled examples | — |
| `steps/epoch` | 12,000 / 32 = 375 | — |
| Total steps | 375 × 3 epochs = 1,125 | healthy range |
| Wall clock (A10G, batch 32, seq 256) | ~1.2 s/step × 1,125 = 22 min | — |
| GPU cost | 0.37 hr × $1.00/hr | **$0.37** |
| Labelling (human, 2 min/row) | 12,000 × 2 min = 400 hr | $10,000 |
| Labelling (LLM pre-annotate + 30 s human verify) | 100 hr | $2,500 |
| **Total one-time** | | **~$2,500–$10,000** |
| **Marginal training cost** | | **$0.37** |

> **The point:** training is essentially free. **Labelling is the cost.** Budget accordingly, and use the model you trained to pre-label the next batch (human verifies only the positives) — the classic label-efficiency loop.

### Inference cost at scale

| Volume/month | BERT int8 CPU | GPT-4o-mini | GPT-4o | Winner |
|---|---|---|---|---|
| 100k | $0.014 (idle box: $260) | $3.60 | $60 | API (box is idle) |
| 1M | $0.14 (idle box: $260) | $36 | $600 | BERT at ~$260 all-in, or API |
| 10M | $1.40 (box: $260) | $360 | $6,000 | **BERT** |
| 100M | $14 (2 boxes: $520) | $3,600 | $60,000 | **BERT by 7×** |
| 1B | $140 (20 boxes: $5,200) | $36,000 | $600,000 | **BERT by 7–115×** |

**Read the table this way:** BERT's cost is *step-wise* (you rent a box; it serves up to its throughput for the same money), the API's is *linear* (every call bills). The crossover is where the box's monthly rent equals the API bill — **~7M calls/month for GPT-4o-mini, ~450k for GPT-4o** on a single $260 box. Below that, the API wins on total cost because the box is idle; above it, BERT wins and the gap widens without limit.

### Latency reference (single sequence, batch 1)

| Setup | p50 | p99 | Throughput (batched) |
|---|---|---|---|
| BERT-base GPU fp16 (A10G) | 8 ms | 15 ms | ~3,000 preds/s |
| BERT-base CPU fp32 (8 vCPU) | 90 ms | 180 ms | ~120 preds/s |
| BERT-base CPU **int8 ONNX** | **25 ms** | 55 ms | **~700 preds/s** |
| DistilBERT CPU int8 | 15 ms | 30 ms | ~1,500 preds/s |
| MiniLM-L6 CPU int8 | 6 ms | 12 ms | ~3,000 preds/s |
| GPT-4o-mini API | 700 ms | 2,500 ms | — (rate-limited) |

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Diagnostic | Fix |
|---|---|---|---|
| NER token accuracy 0.93, entity F1 0.02 | Model predicts all-`O`; alignment broke, or metric is wrong | `Counter(all labels)` — if one value, alignment. Else re-score with `seqeval` | Fix the alignment (`word_ids()` loop); score with `seqeval` |
| NER F1 fine in training, garbage at inference | Missing `aggregation_strategy`; raw subword tags returned | Print the pipeline output shape | `pipeline(..., aggregation_strategy="simple")` |
| NER misses the last word of every multi-word entity | `B-`-only training data, or truncation cutting the entity | Count `I-` tags in the processed dataset | Convert `B B B` → `B I I`; raise `max_length` |
| Loss stuck at exactly 0.693 (binary) / `ln(C)` | Head is random and not learning; frozen, missing from optimizer, or labels are single-class | `sum(p.requires_grad ...)`, `len(optimizer.param_groups[0]['params'])`, `Counter(labels)` | Unfreeze the head; rebuild the optimizer *after* swapping the head; drop constant labels |
| Loss flat at any value from step 0 | LR ≈ 0 (scheduler misconfigured), or `torch.no_grad()` wrapping training | `scheduler.get_last_lr()` | `warmup_steps < total_steps`; remove the `no_grad` |
| NaN at step ~40 | fp16 without a `GradScaler`, or LR too high | Log `grad_norm` — a spike to 1e4 precedes it | `TrainingArguments(fp16=True)` (manages the scaler); LR 2e-5; `max_grad_norm=1.0` |
| Eval loss falling then rising after 100–300 steps | **Catastrophic forgetting** — the encoder is being overwritten | Compare a frozen-encoder run's curve | LR 2e-5 (not 2e-4), 2–3 epochs, warmup 0.1, early stopping |
| Train loss → 0.01, eval loss rising from step 0 | Overfitting on a small dataset | `steps × batch` vs `N` | More data, fewer epochs, weight_decay 0.1, freeze the encoder |
| Accuracy 0.95 but F1-macro 0.40 | Class imbalance — the model predicts the majority class | Confusion matrix: all mass in one column | F1-macro as the selection metric, class weights, threshold tuning |
| QA returns `""` for 30% of questions | Argmax over pads; truncation; `(0,0)` collapse | Print `start_logits.argmax()` — if 0, it is `[CLS]` | Filter `attention_mask`; add `doc_stride`; label out-of-window rows `(0,0)` *by design* |
| QA returns a span that is a substring of a longer gold answer | Window boundary split the answer | Check whether the answer crosses `max_length` | `doc_stride` ≥ max answer length (128 for 30-token answers) |
| QA confidence 0.99 on everything | Miscalibration from a small fine-tune (ECE 0.10–0.25) | Decile reliability plot, compute ECE | Temperature scaling on a val set (fit `T` ≈ 1.5–3.0); label smoothing 0.05 |
| ONNX accuracy drops 4 points | Broken export, not quantization | Compare **fp32 ONNX** vs PyTorch first | Export `--task text-classification` from `AutoModelForSequenceClassification` |
| ONNX logits match but labels are permuted | `id2label` lost in the export | Print `config.json` | Re-set `id2label` in the ONNX dir, or map in the serving layer |
| Predictions correct offline, worse in production | Distribution shift, leakage in the split, or a preprocessing mismatch | `[UNK]` rate + length histogram, offline vs online; dedup on a text hash | Share one `tokenize_fn`; dedup train/test; recalibrate the threshold |
| A metric dropped 29 points overnight, model unchanged | The eval set changed, or the scoring path changed | Hash the eval file; `git diff` the harness | Pin `seqeval`, seed the split, version the golden set |
| `num_labels=9` but the dataset has 7 | Hand-typed label list copied from the wrong scheme | `ds['train'].features['ner_tags'].feature.names` | Always derive `label_names` from the data |
| Model ignores multi-label semantics, one label always wins | Default `problem_type` (softmax CE) on multi-label data | Check `model.config.problem_type` | `problem_type="multi_label_classification"`; per-label thresholds |
| New label requires a full retrain | Labels are baked into the head | — | Cache embeddings; refit a sklearn head on top |
| LoRA loss plateaus at high value | `modules_to_save=["classifier"]` omitted → training onto a frozen random head | Check `sum(p.requires_grad ...)` | Add `modules_to_save`; LoRA LR 2e-4 |
| OOM at batch 16, seq 256 | Activations dominate | Use the §7 table | fp16/bf16 first, then `max_length` 128, then grad checkpointing, then batch 8 + accum 4 |
| GPU at 3% utilisation in production | You are paying for an idle GPU at low QPS | Measure preds/s vs provisioned | Move to CPU int8; autoscale for peaks |

---

## 9. Comparison Matrix

### 9.1 Architecture families — why BERT cannot generate

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder–decoder (T5, BART) |
|---|---|---|---|
| Attention | **Bidirectional** — no causal mask | **Causal** — `M_ij = -inf` for `j > i` | Bidirectional encoder + causal decoder + cross-attention |
| Objective | MLM (fill a blank) | Next-token prediction | Span corruption → reconstruct |
| `P(x_t \| x_<t)` exists? | **No** | Yes | Yes (in the decoder) |
| Can generate? | **No** | Yes | Yes |
| Best at | Classification, NER, QA, embeddings, reranking | Open-ended generation, chat, reasoning | Seq2seq: summarise, translate, rewrite |
| Params (base) | 110M | 124M (GPT-2) | 220M (T5-base) |
| Inference cost | **1 forward pass** | n forward passes for n tokens | n decoder passes (+1 encoder) |
| Deterministic output? | **Yes** (argmax) | No (sampling) | No |
| Can hallucinate? | **No** — QA answers are substrings of the input | Yes | Yes |
| Use for | closed-label prediction at volume | generation, agents | transformation with a fixed output schema |

### 9.2 The BERT family — 2025 decision table

| Model | Params | Layers / Hidden | Vocab | Pretraining data / objective | Pick it when |
|---|---|---|---|---|---|
| `bert-base-uncased` | 110M | 12 / 768 | 30,522 WordPiece | 16 GB BooksCorpus+Wiki, MLM+NSP | Baseline; English; needs no surprises |
| `bert-base-cased` | 110M | 12 / 768 | 28,996 | same | **NER** — casing is signal (`MacKenzie`) |
| `bert-large-uncased` | 340M | 24 / 1024 | 30,522 | same | +1–2 F1 worth a 3× latency bill |
| `distilbert-base-uncased` | 66M | 6 / 768 | 30,522 | Distilled from BERT-base (KL T=2 + MLM + cosine) | 60% faster / 40% smaller at ~97% of GLUE. **The default production choice.** |
| `roberta-base` / `large` | 125M / 355M | 12 / 768, 24 / 1024 | 50,265 BPE | 160 GB, **no NSP**, dynamic masking | Better than BERT at equal size; +2–3 GLUE |
| `albert-base-v2` | 12M | 12 / 768 (shared) | 30,000 | **Cross-layer parameter sharing**, SOP instead of NSP | Embedding-heavy / memory-tight |
| `albert-xxlarge-v2` | 235M | 12 / 4096 | 30,000 | same | Rarely worth it — slow, superseded |
| `google/electra-base-discriminator` | 110M | 12 / 768 | 30,522 | **Replaced-token detection** (dense per-token signal, no `[MASK]`) | Same compute, better quality than BERT-base |
| `google/electra-small-discriminator` | **14M** | 12 / 256 | 30,522 | same | Extreme-latency CPU serving |
| `microsoft/deberta-v3-base` | 86M | 12 / 768 | 128,100 SentencePiece | **Disentangled attention** (content + position) + ELECTRA-style RTD | **The 2025 quality default** — beats RoBERTa-large on several tasks at 1/4 the size |
| `microsoft/deberta-v3-large` | 304M | 24 / 1024 | 128,100 | same | SOTA-class on GLUE/SQuAD; needs a 24 GB card |
| `microsoft/deberta-v3-xsmall` | 22M | 12 / 384 | 128,100 | same | Tiny and strong; good distilled-teacher target |
| `SpanBERT/spanbert-base-cased` | 110M | 12 / 768 | 30,522 | **Span masking + span-boundary objective** | Span-labelled tasks (NER, coref, relation extraction) |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | 110M | 12 / 768 | 30,522 (domain) | PubMed abstracts + full text, from scratch | Biomedical / clinical NER — +5–10 F1 over BERT-base |
| `dmis-lab/biobert-base-cased-v1.2` | 110M | 12 / 768 | 28,996 | PubMed + PMC, BERT init | Same niche; PubMedBERT usually wins |
| `bert-base-multilingual-cased` (mBERT) | 178M | 12 / 768 | 119,547 | 104 languages, Wiki, no translation | Many languages, one model, low-resource languages |
| `xlm-roberta-base` / `large` | 270M / 550M | 12 / 768, 24 / 1024 | 250,002 | 100 languages, 2.5 TB CommonCrawl | **Multilingual default**; beats mBERT clearly |
| `microsoft/mdeberta-v3-base` | 278M | 12 / 768 | 251,000 | DeBERTa-v3 recipe, multilingual | Best multilingual quality at base size |
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | 6 / 384 | 30,522 | Distilled sentence transformer | Embeddings / retrieval / semantic cache |
| `answerdotai/ModernBERT-base` / `-large` | 149M / 395M | 22 / 768, 28 / 1024 | 50,368 | 2T tokens, **RoPE**, GeGLU, alternating local/global, unpadding | **8,192-token context**, 2024-era speed. Needs Flash Attention 2 |
| `google-bert/bert-base-chinese`, `-japanese`, etc. | 110M | 12 / 768 | varies | Language-specific | Monolingual beats multilingual by 2–5 F1 when you have the data |

**Rule:** go down the table in order of *size you can afford*, and only up in *quality* when your eval demands it. **Measure; do not assume.** On a small dataset (< 5,000 rows), `distilbert` frequently matches `deberta-v3-base` because the head is the bottleneck, not the encoder.

### 9.3 Collators — pick the wrong one and it trains, silently wrong

| Collator | Pads `input_ids` with | Pads `labels` with | Use for |
|---|---|---|---|
| `DataCollatorWithPadding` | `pad_token_id` (0) | **`0`** ← the trap | Classification, regression (no labels to pad) |
| `DataCollatorForTokenClassification` | 0 | **-100** | **NER / POS** |
| `DataCollatorForSeq2Seq` | 0 | -100 (and shifts decoder ids right) | Encoder–decoder |
| `DataCollatorForLanguageModeling` | 0 | -100 (builds MLM labels) | Continued pretraining (CS-12) |
| `DefaultDataCollator` | nothing (assumes pre-padded) | nothing | QA — the data is already `padding="max_length"` |

### 9.4 Metrics

| Metric | Formula | Use when | Trap |
|---|---|---|---|
| Accuracy | `(TP+TN)/N` | Balanced classes only | 95% accuracy with a 5% minority = 0.0 minority recall |
| Precision | `TP/(TP+FP)` | False positives are expensive (spam filter) | — |
| Recall | `TP/(TP+FN)` | False negatives are expensive (**PII redaction**) | — |
| F1-binary | `2PR/(P+R)` for the positive class | Imbalanced binary | Ignores the negative class entirely |
| F1-macro | mean of per-class F1 | **Imbalanced multi-class — the default** | Rare classes swing the number; report per-class too |
| F1-micro | pooling all TP/FP/FN | Token-level; equals accuracy in single-label | Dominated by the majority class |
| F1-weighted | F1 per class, weighted by support | You want a "business-mix" number | **Hides minority collapse** — never alone |
| MCC | balanced correlation, −1…+1 | Imbalanced binary, single number | Hard to explain to stakeholders |
| Cohen's κ | agreement above chance | **Annotator agreement**, or vs a human baseline | κ < 0.7 on labels ⇒ your ceiling is κ, not 1.0 |
| ROC-AUC | TPR vs FPR across thresholds | Balanced | **Optimistic under imbalance** — FPR is diluted by the majority |
| **PR-AUC** | Precision vs recall | **Imbalanced — use this instead** | Noisy with few positives |
| seqeval entity F1 | whole-entity exact match | **NER, always** | Token accuracy is the lie |
| EM / F1 (QA) | exact span / token overlap | QA | EM of 0 with F1 of 0.8 is normal |
| ECE | `Σ (n_b/M) · \|acc_b − conf_b\|` | Any confidence-gated pipeline | Typical 0.10–0.25 post-fine-tune |

---

## 10. Numbers To Memorize

| Quantity | Value |
|---|---|
| BERT-base | **110M params · 12 layers · 768 hidden · 12 heads · 3072 FFN · 512 max** |
| BERT-large | 340M · 24 · 1024 · 16 heads · 4096 FFN · 512 max |
| WordPiece vocab | 30,522 (uncased) / 28,996 (cased) |
| Special ids | `[PAD]`=**0** · `[UNK]`=100 · `[CLS]`=**101** · `[SEP]`=102 · `[MASK]`=103 |
| `ignore_index` | **-100** |
| MLM | 15% selected · **80/10/10** mask/random/keep |
| Per-task mask rates | NER 0.10 · classification 0.15 · SQuAD 0.30 |
| NSP | 50/50 IsNext/NotNext · **removed in RoBERTa** |
| VRAM per param (full FT fp32) | **16 B** (4 w + 4 g + 8 AdamW); frozen = 0 |
| **Fine-tuning LR** | **2e-5** (1e-5 – 5e-5) |
| LoRA LR | 2e-4 (1e-4 – 3e-4) — **10× the full-FT LR** |
| Epochs | **2–4** |
| Batch | 16–32 (effective 32–64) |
| Warmup | **10%** of steps |
| Weight decay / max_grad_norm | 0.01 / 1.0 |
| Healthy steps | **1,000 – 10,000** (below ~500 ⇒ underfit) |
| CoNLL-2003 `O` share | ~83% → an all-`O` model scores 0.83 token accuracy, 0.00 entity F1 |
| CoNLL-2003 labels / WikiAnn labels | **9** (adds MISC) / **7** |
| Good NER F1 | 0.92–0.93 (BERT-base, CoNLL-2003) |
| QA config | `max_length=384 · doc_stride=128 · n_best_size=20 · max_answer_length=30` |
| Good SQuAD 2.0 F1 | 0.88–0.92 |
| DistilBERT | 66M · 6 layers · 60% faster · 40% smaller · 97% GLUE |
| RoBERTa-base / large | 125M / 355M · 160 GB · 2T tokens · 50,265 BPE · no NSP |
| DeBERTa-v3 base / large / xsmall | 86M / 304M / 22M · 128,100 vocab |
| ELECTRA-small | **14M** |
| ModernBERT base / large | 149M / 395M · **8,192** context · Dec 2024 |
| mBERT / XLM-R | 178M · 104 langs · 119,547 vocab / 270M · 100 langs · 250k vocab |
| PubMedBERT | 110M · domain vocab · +5–10 F1 on biomedical NER |
| Inference latency | GPU 8 ms · CPU fp32 90 ms · **CPU int8 25 ms** |
| Throughput | GPU ~3,000/s · CPU int8 **~700/s** (8 vCPU) |
| Cost per 1M | BERT **$0.09–0.14** · GPT-4o-mini $36 · GPT-4o $600 |
| Breakeven vs API | ~1M/month (GPT-4o-mini) · ~450k (GPT-4o) |
| CI on accuracy, n=500 | **±3.1 points** (Wilson 95%) |
| Typical ECE after fine-tune | 0.10–0.25 · fitted temperature **1.5–3.0** |
| Training cost, 12k rows | **$0.37** GPU + $2,500–$10,000 labelling |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: too many values to unpack (expected 2)` in `compute_metrics` | `EvalPrediction` unpacked wrongly, or logits are 3-D | `logits, labels = eval_pred` then `np.argmax(logits, axis=-1)` |
| `AssertionError: TextEncodeInput must be...` | Passing word lists without `is_split_into_words=True` | Add the flag |
| `word_ids` `AttributeError` | Using a **slow** tokenizer | `AutoTokenizer.from_pretrained(name, use_fast=True)` |
| `The size of tensor a (…) must match the size of tensor b (…)` on the loss | `labels` length ≠ `input_ids` length | Your alignment loop appended the wrong number of items |
| `Target … is out of bounds` | A label id ≥ `num_labels`, or padding used `0` where `-100` was needed | Derive `num_labels` from the data; use `label_pad_token_id=-100` |
| CUDA OOM | Activations dominate | bf16 → `max_length` halves → grad checkpointing → batch 8 + accum 4 |
| `RuntimeError: expected scalar type Half but found Float` | Hand-rolled `.half()` without a scaler | Use `TrainingArguments(fp16=True)` |
| Loss exactly `nan` from step 1 | Bad labels, or `log(0)` in a custom loss | Assert `0 <= label < num_labels`; use `log_softmax` |
| Loss `nan` at step ~40 | fp16 overflow, or LR too high | GradScaler via `fp16=True`; LR 2e-5; `max_grad_norm=1.0` |
| `The following columns in the training set don't have a corresponding argument in ... forward()` | A string column left in the dataset (e.g. `text`, `tokens`) | `remove_columns=ds["train"].column_names` in `map` |
| `KeyError: 'label'` after `map()` | `remove_columns` stripped the label column | Add the label to the map's output, or use `DatasetDict.rename_column` |
| `TypeError: Object of type int64 is not JSON serializable` | `compute_metrics` returned numpy types | Cast to `float()`/`int()` |
| `Some weights of BertForSequenceClassification were not initialized` | **Expected** — the head is random | This is correct, not a bug. It is why step-0 loss ≈ `ln C` |
| `max_length` warning / `Token indices sequence length is longer than ...` | A sequence exceeds the model maximum; it will be **truncated without error** | `truncation=True` + `max_length` explicitly; check the length distribution |
| `You should probably TRAIN this model ...` | Loading a base model without a fine-tuned head | Expected; fine-tune it |
| `[MASK]` shows up in your inference output | You fed `[MASK]` in, or you are running the MLM head | `[MASK]` never belongs in downstream data |
| `seqeval` returns `None` for `overall_f1` | Empty prediction/reference lists, or ints instead of strings | Filter `-100` with the **same index set** on both sides; convert to label strings |
| `KeyError: 'eval_loss'` with `metric_for_best_model="eval_f1"` | `compute_metrics` did not return `f1`, or eval never ran | Return the exact key; set `eval_strategy` |
| ONNX `Unsupported operator` / `Invalid graph` | Wrong `--task`, or a custom layer | Re-export with the right task; `--opset 17` |
| `id2label` reverts to `LABEL_0` after export | Not serialized into the ONNX `config.json` | Write `config.json` with `id2label` post-export, or map in the serving layer |
| Model serves but predictions are all one class | Missing `attention_mask`, or the tokenizer's `pad_token` changed | Assert `tokenizer.pad_token_id == 0`; pass the mask |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` after LoRA | `modules_to_save` omitted, head shape mismatch | `modules_to_save=["classifier"]`; re-instantiate with `num_labels` |
| Inference 20× slower than expected | Per-row loop instead of batched | Batch, length-sort, `padding=True`, `torch.inference_mode()` |

---

## 12. Copy-Paste Starter Config

**A genuinely runnable end-to-end file.** Save as `bert_quickstart.py`, then `python bert_quickstart.py`. Runs in ~5 minutes on a CPU-only box (uses 2,000 rows) or ~1 minute on any GPU.

```python
"""
CH-07 starter: fine-tune BERT on IMDb sentiment, evaluate properly, save, and
serve. Swap the DATASET / PREPROCESS / COLLATOR / MODEL blocks for NER or QA
using sections 5.2 and 5.3.
"""
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          pipeline)
from evaluate import load as load_metric

# ---------------- 1. CONFIG ----------------
MODEL_NAME = "bert-base-uncased"
OUT_DIR    = "out/quickstart"
MAX_LEN    = 256
N_TRAIN    = 2000      # raise to 20000+ for a real run
N_EVAL     = 500
SEED       = 42

# ---------------- 2. DATA ----------------
ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

ds = ds.map(preprocess, batched=True)
train_ds = ds["train"].shuffle(seed=SEED).select(range(N_TRAIN))
eval_ds  = ds["test"].shuffle(seed=SEED).select(range(N_EVAL))

# ---------------- 3. MODEL ----------------
# IMDb: label 0 = neg, 1 = pos  (set id2label or the pipeline returns LABEL_0)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label,
    label2id={v: k for k, v in id2label.items()})

# ---------------- 4. METRICS ----------------
acc = load_metric("accuracy")
f1  = load_metric("f1")

def compute_metrics(ep):
    logits, labels = ep
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(acc.compute(predictions=preds, references=labels)["accuracy"]),
        "f1_macro": float(f1.compute(predictions=preds, references=labels,
                                     average="macro")["f1"]),
        "f1_pos": float(f1.compute(predictions=preds, references=labels,
                                   average="binary", pos_label=1)["f1"]),
    }

# ---------------- 5. TRAIN ----------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=2e-5,               # 1e-5 .. 5e-5  (never 1e-3)
    num_train_epochs=3,               # 2 .. 4
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,    # effective batch 32
    warmup_ratio=0.1,                 # 10% linear ramp
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="linear",
    fp16=False, bf16=False,           # set bf16=True on Ampere+ for ~1.8x
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="f1_macro",
    logging_steps=25, report_to="none",
    seed=SEED, dataloader_num_workers=0 if __import__("os").name == "nt" else 4,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=DataCollatorWithPadding(tok),   # dynamic padding = free speedup
    compute_metrics=compute_metrics,
)
trainer.train()

# ---------------- 6. SAVE (weights AND tokenizer) ----------------
trainer.save_model(f"{OUT_DIR}/final")
tok.save_pretrained(f"{OUT_DIR}/final")
print("saved ->", f"{OUT_DIR}/final")

# ---------------- 7. EVALUATE ON HELD-OUT ----------------
metrics = trainer.evaluate()
print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
print(f"optimizer steps = {trainer.state.global_step}  (healthy: 1,000-10,000)")

# ---------------- 8. INFERENCE ----------------
clf = pipeline("text-classification", model=f"{OUT_DIR}/final",
               tokenizer=f"{OUT_DIR}/final", device=-1, top_k=None)

samples = [
    "An absolute masterpiece — I have watched it four times.",
    "Two hours of my life I will never get back.",
    "It was fine. Nothing special, nothing terrible.",
]
for s, p in zip(samples, clf(samples)):
    print(f"{p[0]['label']:9s} {p[0]['score']:.3f}  |  {s}")

# ---------------- 9. SANITY CHECKS (do not skip) ----------------
assert trainer.state.global_step >= 100, "too few steps: the head cannot have converged"
assert metrics.get("eval_f1_macro", 0) > 0.70, \
    "below 0.70 at 2k rows/max_len 256 -> check LR, epochs, and label mapping"
print("checks passed")

# ---------------- 10. PRODUCTION NEXT STEPS ----------------
#   optimum-cli export onnx --model out/quickstart/final --task text-classification onnx/sent
#   optimum-cli onnxruntime quantize --avx512 --onnx_model onnx/sent -o onnx/sent-int8
#   -> 2-4x CPU throughput, <1 point accuracy loss. Then verify PyTorch-vs-ONNX
#      agreement >= 99% on 500 held-out rows before shipping (section 6).
```

**Expected output on 2,000 IMDb rows (CPU, ~5 min):** `eval_accuracy ≈ 0.88–0.91`, `eval_f1_macro ≈ 0.88–0.91`, `global_step = 188`. If you see `global_step = 188` and think it is fine, re-read §2.5 — 188 steps is **below the ~500 floor**, which is exactly why the video's own demo gives unreliable predictions. Raise `N_TRAIN` to 20,000 for a real run (≈1,875 steps).

---

## 13. What To Read Next

| Module | Why |
|---|---|
| **CS-07 — the case study** | The full derivation: tensor-level mechanics, 26-row debugging playbook, 5 applied case studies, production/versioning/compliance. Read alongside this sheet. |
| **IQ-07 — BERT interview bank** | 104 questions across 5 levels, with the traps an interviewer is actually testing. |
| **CS-08 — Knowledge Distillation I: DistilBERT** | How BERT-base → DistilBERT actually works (KL T=2 + MLM + cosine), and how to distil your own fine-tune into a 3× faster student. **The highest-leverage follow-up to this sheet.** |
| **CS-12 — Domain-Adaptive Continued Pretraining** | MLM on your own unlabelled corpus *before* fine-tuning: the +3–5 F1 that costs almost nothing. |
| **CS-10 / CS-11 — Quantization I & II** | PTQ vs QAT vs GPTQ/AWQ — the full story behind the int8 line in §6. |
| **Embeddings — `code/10_embedding_finetune.py`** | The `[CLS]` vs mean-pooling debate, bi-encoders, and fine-tuning a retriever. (**CS-22** planned, not yet written) |
| **CS-13 / CH-04 / CH-12** | The decoder-side counterpart, and when fine-tuning is the wrong tool at all (vs RAG, vs prompting). |
| **CS-05 / CS-06 / CS-13 §6.8** | Attention math and why the bidirectional encoder exists; the HF API surface this sheet assumes; the LoRA config in full (IQ-07 Q91's multi-tenant design). (**CS-23** planned, not yet written) |

---

*CH-07 — Fine-Tuning BERT: NER, Sentiment, QA. Companion to CS-07 and IQ-07. Source: `LLM_Fine-Tuning_09_*` [00:00–1:05:00] and the companion notebook `BERT_Finetuning.ipynb`.*
