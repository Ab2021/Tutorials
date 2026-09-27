# CH-04 — Fine-Tuning vs RAG vs Agents Cheat Sheet

**One-line purpose:** Pick the right architecture in ten minutes instead of six weeks.
**Use when:** A stakeholder says "the model doesn't know our stuff", or you are choosing between training, retrieving, and acting.
**Do NOT use when:** You have not built a prompt-only baseline. Go do that first — it is rung zero and it takes an afternoon.

---

## 1. The 10-Second Summary

**Three levers. Each fixes exactly one failure class. Nothing else.**

| Lever | Approach | Fixes | Cannot fix |
|---|---|---|---|
| **Weights** | Fine-tuning | Behaviour: format, tone, skill, refusal, unit cost | Facts, freshness, citations, ACLs |
| **Context** | Prompting, RAG | Knowledge: facts, freshness, provenance, access control | Behaviour, format, latency, cost |
| **Control flow** | Agents | Action: side effects, multi-step tasks | Determinism, latency, cost, correctness guarantees |

**The attribution test** — ask this before naming any technology:
> *"If the model had the perfect information, perfectly formatted, would it produce the right answer?"*
> No → **RAG**. Yes but wrong shape → **fine-tune**. Yes and right → **prompt better**. Right but nothing happens → **agent**. Both wrong → **hybrid**.

**The ladder** — climb in this order, never skip: `prompt → RAG → fine-tune → agent`. Costs go ~$0 → $2–15k → $5–60k → $20–200k; time goes hours → days → weeks → months.

**The one heuristic:** RAG for facts, FT for form. **The one exception:** fine-tuning works for knowledge when the fact set is <10k, changes quarterly-or-slower, is needed by nearly every query, AND needs no citations — all four.

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Fine-tune update | `θ' = θ − η∇L(θ; D)` | θ weights, η LR, D task data | 8B, 8k rows, 3 epochs, η=2e-4 |
| LoRA | `W' = W₀ + BA`, `B∈R^{d×r}`, `A∈R^{r×k}` | r ≪ min(d,k) | r=16 → ~42M trainable on 8B (0.5%) |
| Full-FT memory | `16 bytes/param` (fp16 w+g, fp32 master, Adam m+v) | +activations +temp | 8B → ~112–128 GB |
| QLoRA memory | `~1.2 bytes/param` base (NF4) + adapters + activations | | 8B → ~9–12 GB |
| Cosine similarity | `sim(q,c) = q·c / (‖q‖‖c‖)` | q query vec, c chunk vec | used to rank top-k |
| Agent end-to-end | `P = p^n` | p per-step success, n steps | 0.95^10 = **59.9%** |
| Required per-step | `p = target^(1/n)` | | 0.90 over 10 → **98.95%** |
| Prefill FLOPs | `2 × N_params × N_prompt_tokens` | | 7B × 4k = 56 TFLOPs |
| Decode ceiling | `memory_BW / weight_bytes_per_token` | | 8B fp16 on H100 → ~209 tok/s |
| KV cache | `2 × n_layers × n_kv_heads × head_dim × seq × batch × dtype_bytes` | | 8B, 4k, bs1, fp16 = 524 MB |
| Two-hop recall | `r₁ × r₂` | | 0.92² = 0.85; 0.78² = 0.61 |
| Cost per query | `tok_in×P_in + tok_out×P_out` (+ 0.1× for cached reads) | | 2.65k in, 250 out @ $0.15/$0.60 = $0.00055 |
| Cost per success | `total_cost / successes` | | $0.066 / 0.663 = $0.10 |
| Index size | `N_chunks × dim × bytes_per_dim` ×1.4 (HNSW) | | 500k × 1536 × 4B = 3.1 GB → ~4.3 GB |

---

## 3. Decision Tree

```
Is there a labelled failure taxonomy with counts?
├─ No  → build 200-item golden set from production failures. STOP. (S1)
└─ Yes ↓

Would a perfect-information, perfectly-formatted answer be correct?
├─ No, it lacks information
│   ├─ Does the content change faster than your retrain cadence?
│   │   ├─ Yes → RAG. (S2)
│   │   └─ No  → <10k facts AND every query needs them AND no citations?
│   │            ├─ Yes → FT for knowledge is allowed (rare)
│   │            └─ No  → RAG
│   └─ Do users have different document permissions? → RAG (ACL filter). (S4)
│
├─ Yes, but wrong format / tone / skill
│   ├─ Did few-shot prompting get you to SLA?
│   │   ├─ Yes → ship the prompt
│   │   └─ No  → fine-tune (small model, LoRA/QLoRA). Format is a weights problem.
│   └─ Do you have ≥2k labelled examples or a cheap teacher to distil from?
│       ├─ No  → S6: label first, or accept the prompt's quality
│       └─ Yes → fine-tune
│
├─ Yes, and it's right — I just asked badly
│   └─ Prompt engineering. Do not build anything.
│
└─ Right answer, but nothing happens in the world
    ├─ Is the action irreversible?
    │   ├─ Yes → agent WITH a confirmation gate + code-enforced caps. (S9)
    │   └─ No  → agent
    └─ Is the median task >15 steps?
        ├─ Yes → decompose into sub-agents, or narrow scope, or fixed pipeline. (S11)
        └─ No  → agent with max_steps, $ budget, wall-clock budget, loop detection

Latency override (applies to everything above):
  p95 budget < 300 ms?
  ├─ Yes → no retrieval, no agent. Small fine-tuned model, structured output. (S5)
  └─ No  → proceed
```

---

## 4. Scoring Rubric — Score Each, Apply The Overrides

Score 1–5 per dimension. Weights in the column header. Two dimensions are **vetoes**, not scores.

| Dimension | W | Score 1 | Score 3 | Score 5 |
|---|---|---|---|---|
| **Data volatility** | 3 | Monthly or rarer | Quarterly, scheduled | Daily / real-time / per-user |
| **Output-format rigidity** | 3 | Prose fine | Structured, humans read it | Strict schema, machine-parsed, 99.9% |
| **Latency budget (p95)** | 3 | Minutes (async) | 3–10 s | <300 ms |
| **Cost ceiling ($/query)** | 2 | >$0.05 | $0.001–0.05 | <$0.0005 |
| **Corpus volume** | 2 | <100k tokens | 100k–10M | >10M |
| **Label availability** | 3 | None obtainable | 2k rows with effort | 10k+ labelled rows |
| **Action requirement** | **VETO** | Read-only | Drafts for humans | Irreversible autonomy |
| **Auditability requirement** | **VETO** | Not needed | Nice to have | Regulated/contractual |

**Read the profile off this table:**

| Profile | Winner |
|---|---|
| High volatility + high volume + loose format | **RAG** |
| Low volatility + rigid format + labels available | **Fine-tune** |
| Low volatility + rigid format + <300 ms + low cost | **Fine-tune a small model, serve locally** |
| High volatility + rigid format + latency-tolerant | **Hybrid: FT generator + RAG** |
| Action requirement = 5 | **Agent**, with a confirmation gate unless reversible |
| Auditability = 5 | **RAG** — veto regardless of every other score |
| All scores 1–2 | **Prompt engineering / long context** — build nothing |

**STOP conditions (do not build):**

| # | Signal |
|---|---|
| S1 | You cannot name the failure mode in one sentence |
| S2 | Content changes faster than your retrain cadence |
| S3 | Every claim must be traceable to a source (weights cannot cite) |
| S4 | Users have different document permissions |
| S5 | p95 < 500 ms and the task needs retrieved context |
| S6 | <200 labelled examples and no budget to make more |
| S7 | No prompt-only baseline measured |
| S8 | Task value < FT build cost amortised over 12 months |
| S9 | Irreversible action, unattended |
| S10 | Retrieval recall@20 < 0.75 (the generator is not the problem) |
| S11 | Median task needs >15 agent steps (`p^n` kills you) |
| S12 | No agreed success metric |

---

## 5. Latency, Cost And Quality — The Three Regimes

| | Fine-tune (SLM) | RAG | Agent |
|---|---|---|---|
| p50 end-to-end | **50–200 ms** (short structured output) | **1.2–3.5 s** | **6–90 s** |
| p50, 200-token answer | 0.8–2.0 s | included above | — |
| Dominated by | prefill + few decode steps | retrieval + rerank + prefill of 2–3k extra tokens | n × (prefill of growing context + decode + tool I/O) |
| Build time | 2–6 weeks | 3–5 days | 4–10 weeks |
| Build cost | $5k–60k | $2k–15k | $20k–200k |
| Unit cost | **~$0.000003–0.000008/req** (4-bit 3B on L4) | ~$0.00055–0.008/req | ~$0.03–0.19/call |
| Cost per success | ≈ unit cost | ≈ unit cost | **unit ÷ success rate** (1.5–2.5×) |
| Breaks at | >10k facts, or weekly changes | <300 ms budgets; strict formats | high volume × long trajectories |

**Cost per 1M requests at 2.65k in / 250 out:**
- GPT-4o ($2.50/$10.00): **$8,125** → GPT-4o-mini ($0.15/$0.60): **$550** → FT 3B on an L4: **~$8**
- Retrieved context (2,000 tok) is 75% of the prompt. **Cache the static prefix at ~0.1×** and this drops hard.

---

## 6. Hyperparameter Quick Reference

| Param | Default | Sweep | Effect |
|---|---|---|---|
| `lora_r` | 16 | 8 / 16 / 32 / 64 | Capacity for behaviour. Down if memorising, up if underfitting |
| `lora_alpha` | 32 (ratio 2) | 16–64 | Adapter contribution scale |
| `lora_dropout` | 0.05 | 0–0.1 | Regularisation |
| `target_modules` | q,k,v,o,gate,up,down | — | **Verify `print_trainable_parameters()` ≠ 0** |
| `learning_rate` | 2e-4 (LoRA) | 1e-4 – 3e-4 | Higher → spikes/forgetting; lower → frozen loss |
| `num_train_epochs` | 3 | 1–3 | Form is learned in 1–3. More → memorisation |
| `per_device_batch` × `grad_accum` | 4 × 4 | — | Effective batch 16 is fine |
| `max_seq_length` | 1024–4096 | — | **Must exceed p99 of prompt+target** or targets truncate |
| `packing` | False | — | Only if rows ≪ max_seq_length |
| `bf16` | True | — | Use bf16, never fp16, on modern GPUs |
| `gradient_checkpointing` | True | — | −60% memory, +20–30% time |
| `temperature` (inference) | **0** | — | Anything parsed must be greedy |
| `chunk_size` | 400–600 tok | 200–1000 | Granularity of retrieval |
| `chunk_overlap` | 10–20% | 50–150 tok | Boundary coverage |
| `k` (retrieve) | 20 dense + 20 sparse | 10–50 | Recall protection |
| `k_final` (prompt) | 5 | 3–10 | Cost + latency + lost-in-the-middle |
| reranker | **on** | on/off | +10–25 nDCG for +50–300 ms |
| `max_steps` | 8 | 4–15 | Agent stop condition (mandatory) |
| `$ budget` / `wall-clock` | task-specific | — | Agent stop conditions (mandatory) |
| tool count | ≤10 | 5–15 | Selection accuracy degrades past ~15 |

---

## 7. Copy-Paste Code Snippets

### 7.1 Fine-tune — QLoRA for a format task (full, runnable)

```python
# pip install "transformers>=4.44" "peft>=0.12" "trl>=0.9" "bitsandbytes>=0.43" datasets accelerate
import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SCHEMA   = '{"intent": <str>, "priority": "P0|P1|P2|P3", "entities": [<str>]}'
SYS      = f"Reply with a single JSON object and nothing else. Schema: {SCHEMA}"

def to_row(x, y):                       # y must be the FULL assistant turn incl. stop point
    return {"messages": [
        {"role": "system",  "content": SYS},
        {"role": "user",    "content": x},
        {"role": "assistant","content": json.dumps(y, separators=(",", ":"))},
    ]}

train = Dataset.from_list([to_row(x, y) for x, y in load_train()])
evals = Dataset.from_list([to_row(x, y) for x, y in load_eval()])   # held out, split by doc/time

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="auto", attn_implementation="sdpa")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()      # sanity: must be ~0.5%, not 0

SFTTrainer(model=model, processing_class=tok, train_dataset=train, eval_dataset=evals,
    args=SFTConfig(output_dir="./out/fmt", num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03, max_seq_length=1024, packing=False,
        bf16=True, gradient_checkpointing=True, logging_steps=10,
        eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
        save_total_limit=2, report_to="none")).train()
```

### 7.2 RAG — hybrid + rerank + cite-or-refuse (full, runnable)

```python
# pip install sentence-transformers rank_bm25 openai numpy
import hashlib, numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI

EMB, RERANK, client = (SentenceTransformer("BAAI/bge-base-en-v1.5"),
                       CrossEncoder("BAAI/bge-reranker-base"), OpenAI())

def chunk(text, size=500, overlap=80):
    w, out, i = text.split(), [], 0
    while i < len(w):
        out.append(" ".join(w[i:i+size])); i += size - overlap
    return out

def cid(doc_id, seq): return hashlib.sha1(f"{doc_id}:{seq}".encode()).hexdigest()[:16]

def index(docs):                       # docs = [(doc_id, title, text)]
    chunks = [{"doc_id": d, "seq": i, "cid": cid(d, i), "text": t,
               # contextual retrieval: situate the chunk before embedding (biggest recall win)
               "embed": f"{title} — chunk {i} of {d}\n\n{t}"}
              for d, title, text in docs for i, t in enumerate(chunk(text))]
    vecs = EMB.encode([c["embed"] for c in chunks], normalize_embeddings=True, batch_size=64)
    for c, v in zip(chunks, vecs): c["vec"] = v
    return chunks, BM25Okapi([c["text"].lower().split() for c in chunks])

def retrieve(q, chunks, bm25, k=20, k_final=5):
    qv = EMB.encode([q], normalize_embeddings=True)[0]
    d  = np.argsort([-float(qv @ c["vec"]) for c in chunks])[:k].tolist()
    s  = np.argsort(-bm25.get_scores(q.lower().split()))[:k].tolist()
    cand = list(dict.fromkeys(d + s))
    rr = RERANK.predict([(q, chunks[i]["text"]) for i in cand])
    return [chunks[cand[i]] for i in np.argsort(-rr)[:k_final]]

SYS_RAG = ("Answer using ONLY the context. Cite each claim as [doc:cid]. "
           "If the context does not contain the answer, reply exactly: "
           "'I don't have that information.' Do not use prior knowledge.")

def answer(q, chunks, bm25, acl=None):
    ctx = retrieve(q, chunks, bm25)
    if acl: ctx = [c for c in ctx if acl(c["doc_id"])]        # filter BEFORE the prompt
    body = "\n\n".join(f'[{c["doc_id"]}:{c["cid"]}] {c["text"]}' for c in ctx)
    msgs = [{"role": "system", "content": SYS_RAG},           # cache breakpoint goes HERE
            {"role": "user",   "content": f"Context:\n{body}\n\nQuestion: {q}"}]
    r = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0)
    return r.choices[0].message.content, [c["cid"] for c in ctx], r.usage.model_dump()

def recall_at_k(gold, chunks, bm25, k=10):   # gold = [(query, gold_cid)]
    return sum(any(c["cid"] == cid_g for c in retrieve(q, chunks, bm25, k_final=k))
               for q, cid_g in gold) / len(gold)
```

### 7.3 Agent — three stop conditions + loop detection (full, runnable)

```python
# pip install openai
import json, time, hashlib
from openai import OpenAI
client = OpenAI()

TOOLS = [{"type": "function", "function": {
    "name": "search_kb",
    "description": ("Search the internal knowledge base. USE FOR: policy, pricing, product "
                    "behaviour. DO NOT USE FOR: order status (use get_order)."),
    "parameters": {"type": "object", "required": ["query"],
                   "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
    "name": "issue_refund",
    "description": ("Issue a refund. IRREVERSIBLE. Only call after the user has explicitly "
                    "confirmed the order id and amount in their own message."),
    "parameters": {"type": "object", "required": ["order_id", "amount_usd"],
                   "properties": {"order_id": {"type": "string"},
                                  "amount_usd": {"type": "number"}}}}}]

def run_agent(goal, max_steps=8, budget_usd=0.25, max_seconds=30.0):
    msgs = [{"role": "system", "content": "Use tools when you need facts. Never call an "
             "irreversible tool without explicit user confirmation. If you cannot finish, say so."},
            {"role": "user", "content": goal}]
    trace, spent, t0, seen = [], 0.0, time.perf_counter(), {}

    for step in range(max_steps):
        if spent > budget_usd or time.perf_counter() - t0 > max_seconds:
            return {"status": "budget_exhausted", "trace": trace, "spent_usd": spent}
        r = client.chat.completions.create(model="gpt-4o", messages=msgs,
                                           tools=TOOLS, temperature=0)
        spent += (r.usage.prompt_tokens * 2.50 + r.usage.completion_tokens * 10.0) / 1e6
        m = r.choices[0].message
        if not m.tool_calls:                                  # STOP 1: final answer
            return {"status": "done", "answer": m.content, "trace": trace, "spent_usd": spent}
        msgs.append(m)
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            key = hashlib.sha1(f"{tc.function.name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:12]
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:                                # STOP 4: loop detection
                return {"status": "loop", "trace": trace, "spent_usd": spent}
            try:    obs, err = run_tool(tc.function.name, args), None
            except Exception as e: obs, err = None, repr(e)   # feed errors back, do not raise
            trace.append({"step": step, "tool": tc.function.name, "args": args, "error": err})
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps({"result": obs, "error": err})[:4000]})
    return {"status": "max_steps", "trace": trace, "spent_usd": spent}   # STOP 2 + 3
```

### 7.4 The hybrid — FT generator + RAG, cache-safe prompt order

```python
def hybrid(q, acl, retriever, local_ft_model):
    ctx = [c for c in retriever(q, k_final=5) if acl(c.doc_id)]
    body = "\n\n".join(f"[{c.doc_id}:{c.cid}] {c.text}" for c in ctx)
    msgs = [{"role": "system", "content": STATIC_SYSTEM},   # static → cached at 0.1x
            {"role": "system", "content": STATIC_RULES},    # static → cached
            {"role": "user",   "content": f"Context:\n{body}\n\nQuestion: {q}"}]  # volatile
    return local_ft_model.chat(msgs, temperature=0)         # 3B LoRA, ~$0.000005/query
```

---

## 8. CLI Commands

```bash
# ---------- FINE-TUNING ----------
pip install "transformers>=4.44" "peft>=0.12" "trl>=0.9" "bitsandbytes>=0.43" datasets accelerate

# TRL CLI: SFT with LoRA, config-driven (same as the Python above, no script needed)
trl sft --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
        --dataset_name ./data/tickets_train.jsonl --dataset_text_field messages \
        --output_dir ./out/fmt --num_train_epochs 3 \
        --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 --lr_scheduler_type cosine --warmup_ratio 0.03 \
        --max_seq_length 1024 --bf16 --gradient_checkpointing \
        --use_peft --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --eval_strategy steps --eval_steps 50 --save_steps 50 --save_total_limit 2

# Merge the adapter for serving
python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM as M, \
AutoTokenizer as T; b=M.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'); \
m=PeftModel.from_pretrained(b,'./out/fmt').merge_and_unload(); m.save_pretrained('./merged'); \
T.from_pretrained('./out/fmt').save_pretrained('./merged')"

# ---------- SERVING ----------
pip install vllm
vllm serve ./merged --quantization awq --max-model-len 4096 \
     --gpu-memory-utilization 0.90 --port 8000       # 4-bit 3B fits an L4 24 GB
# OpenAI-compatible: point your client at http://localhost:8000/v1

# vLLM with hot-swappable LoRA adapters (no merge, no restart)
vllm serve meta-llama/Llama-3.2-3B-Instruct --enable-lora \
     --lora-modules fmt=./out/fmt --max-lora-rank 16

# ---------- RAG INFRASTRUCTURE ----------
docker run -d -p 6333:6333 -v $(pwd)/qdrant:/qdrant/storage qdrant/qdrant     # vector DB
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=x pgvector/pgvector:pg16      # pgvector

pip install qdrant-client sentence-transformers rank_bm25
python -c "from sentence_transformers import SentenceTransformer as S; \
print(S('BAAI/bge-base-en-v1.5').encode(['test']).shape)"   # smoke-test the embedder

# ---------- EVALUATION ----------
pip install ragas datasets
python -m ragas.evals --help     # faithfulness, answer_relevancy, context_precision, context_recall

# Contamination check: 13-gram overlap between train and eval
python - <<'PY'
import re
def grams(t, n=13): w = re.findall(r"\w+", t.lower()); return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
tr = set().union(*[grams(l) for l in open("data/train.txt", encoding="utf-8")])
ev = set().union(*[grams(l) for l in open("data/eval.txt",  encoding="utf-8")])
print("overlap:", len(tr & ev), "— must be 0")
PY

# ---------- COST ESTIMATOR ----------
python - <<'PY'
IN, OUT, CACHE = 2.50, 10.00, 0.25   # $/1M tokens; CACHE = cached-read multiplier
def cost(tin, tout, qpm, cache_frac=0.0):
    eff = tin * (1 - cache_frac) + tin * cache_frac * CACHE
    return (eff * IN + tout * OUT) / 1e6 * qpm
print("frontier  $/mo:", round(cost(2650, 250, 1_000_000), 2))
print("mini      $/mo:", round(cost(2650, 250, 1_000_000) * 0.06, 2))
print("cached 80%$/mo:", round(cost(2650, 250, 1_000_000, cache_frac=0.8), 2))
PY
```

---

## 9. VRAM / Cost Calculator

**Fine-tuning (per model, mixed precision):**

| Model | Full FT (fp16+Adam) | LoRA fp16 | QLoRA 4-bit | Min practical GPU |
|---|---|---|---|---|
| 0.5B | ~8 GB | ~3 GB | ~2 GB | T4 16 GB |
| 1B | ~16 GB | ~6 GB | ~3 GB | T4 16 GB |
| 3B | ~48 GB | ~10 GB | ~5 GB | L4 24 GB |
| 7–8B | ~112–128 GB | ~18–22 GB | ~9–12 GB | A100 40 GB (QLoRA: 16 GB) |
| 13B | ~208 GB | ~30 GB | ~16 GB | A100 40 GB (QLoRA) |
| 70B | ~1.1 TB | ~160 GB | ~40–48 GB | 2× A100 80 GB (QLoRA) |

**Serving (4-bit, vLLM, short outputs):**

| Model | Weights | GPU | Rough req/s | $/request (amortised) |
|---|---|---|---|---|
| 1.5B 4-bit | ~1 GB | L4 24 GB | 40–60 | $0.0000025 |
| 3B 4-bit | ~2 GB | L4 24 GB | 25–40 | $0.0000056 |
| 8B 4-bit | ~4.5 GB | L4 24 GB | 10–18 | $0.000012 |
| 70B 4-bit | ~35 GB | 2× A100 80 GB | 2–5 | $0.0003 |

**RAG index:**

| Corpus | Chunks (500 tok) | Dim | Raw fp32 | With HNSW | Hosting/mo |
|---|---|---|---|---|---|
| 10k docs | 50k | 1536 | 307 MB | ~430 MB | <$1 (pgvector) |
| 100k docs | 500k | 1536 | 3.1 GB | ~4.3 GB | ~$4–10 |
| 1M docs | 5M | 1536 | 31 GB | ~43 GB | $15–40 |

**One-time embedding cost:** corpus tokens × embedder price. 25M tokens × $0.02/1M = **$0.50** (text-embedding-3-small). Contextualisation adds **~$1.02 per 1M document tokens**.

**Per-query cost at 1M queries/month (2,650 in / 250 out):**

| Configuration | $/query | $/month |
|---|---|---|
| Frontier (gpt-4o) | $0.008125 | $8,125 |
| Mini (gpt-4o-mini) | $0.000550 | $550 |
| Mini + 80% cache hit | ~$0.000145 | $145 |
| FT 3B on an L4, no retrieval | ~$0.000006 | ~$6 |
| 8-step agent on gpt-4o | $0.066/call, **$0.10/success** | $6,600/call... × 100k tasks = $6,600 |

---

## 10. Symptom → Fix Lookup Table

| Symptom | Most likely cause | First diagnostic | Fix |
|---|---|---|---|
| Loss flat, never moves | `target_modules` matched nothing, or loss over prompt tokens | `print_trainable_parameters()`; decode 3 tokenised examples | Fix target modules; enable completion-only loss |
| Loss → 0 in 200 steps | Memorisation (data too small/templated) | Plot eval loss vs train loss | Fewer epochs, more diverse data |
| Loss NaN | fp16 overflow | Check first 20 steps; grad norms | bf16, lower LR, clip grad at 1.0 |
| Format improved, task accuracy dropped | Learned the shape, not the content | Confusion matrix on the label field | More target diversity; check label distribution |
| Model never refuses | No refusal examples | Refusal rate on out-of-scope inputs | Add 5–15% out-of-scope rows with refusal targets |
| Model refuses in-scope questions | Too many refusals, or serve/train prompt mismatch | Refusal rate on in-scope held-out | Rebalance; make serving prompt byte-identical |
| Verbatim leakage of training rows | Small repetitive dataset, too many epochs | 13-gram overlap of outputs vs training data | Dedupe, cut epochs, paraphrase |
| General capability regression | Catastrophic forgetting | Capability suite vs base | Lower LR, fewer epochs, LoRA, +10% general data |
| FT model ignores retrieved context | Trained without context in the prompt format | Log the assembled prompt | Retrain with context + hard negatives + refusal |
| RAG ignores the context | Context placed after the question, or too long | Prompt order; prompt token count | Context first; cut `k_final` to 3–5 |
| The right doc is never retrieved | Decontextualised chunk, boilerplate, or embedding mismatch | Find the gold chunk's rank position by ID | Contextual retrieval, BM25 hybrid, reranker, FT embedder |
| Retrieval works but answers are wrong | Faithfulness failure | RAGAS faithfulness with high context recall | Cite-or-refuse prompt, temperature 0, stronger generator |
| Top-5 chunks are near-duplicates | Overlap too high or boilerplate headers | Hash chunk texts | Strip boilerplate, dedupe/MMR before assembly |
| Stale answers, index timestamp current | Ingestion no-op, version conflict, or embedding-version mismatch | Canary doc probe | Fix the parser; version + precedence metadata; re-embed |
| Latency p95 up, p50 flat | Cache misses on a tail, or long-input prefill | Stage-level p95 (retrieve/rerank/prefill/decode); cache hit rate | Restore prompt ordering; cap input length |
| Cost up 3–10× with flat volume | `k_final`/chunk size changed, or prompt prefix became volatile | Tokens per request histogram | Restore static-first ordering; retune `k_final` |
| Agent loops forever | Tool returns nothing useful; errors raised not returned | Hash `(tool, args)`; read the observation | Return structured errors; add a clarification exit; hard stop |
| Agent says success, nothing happened | Silent tool failure; `except: return None` | Read the trace at the write step | Canonical result objects with success flags; verify-by-read-back |
| Agent cost tripled in a week | Context growth, retries, or cache breakage | Steps histogram + prompt tokens/step + cache hit rate | Summarise observations; fix the failing tool; reorder the prompt |
| Agent fails after step 3 | Error compounding or context poisoning | Trace at the failing step | Fine-tune the tool-caller; validate tool outputs |
| Router cuts cost but hard-query quality collapsed | Regret hidden by aggregate metrics | Quality *conditional on route* | Raise the confidence threshold; retrain on outcome labels |
| Eval metric rises forever | Train/eval contamination | 13-gram overlap check | Re-split by document and time |

---

## 11. Comparison Matrix

| | Prompt/long ctx | RAG | Fine-tune | Agent |
|---|---|---|---|---|
| Changes | context | context + evidence | weights | control flow |
| Fixes | under-specification | knowledge, freshness | behaviour, format, skill | action |
| Cannot fix | anything structural | behaviour, latency | provenance, ACLs, freshness | latency, determinism, cost |
| Time to v1 | hours | 3–5 days | 2–6 weeks | 4–10 weeks |
| Build $ | ~0 | $2–15k | $5–60k | $20–200k |
| Unit cost | per-token | per-query context, forever | amortised (cheapest) | per-step, multiplied |
| Citations | no | **yes** | no | yes (via tools) |
| Per-user ACL | no | **yes** | no | yes (via tools) |
| Per-fact rollback | n/a | **yes** (upsert) | no | n/a |
| p50 latency | 0.8–2 s | 1.2–3.5 s | 50 ms–2 s | 6–90 s |
| Determinism | temp 0 ≈ | ≈ | ≈ | no |
| Best metric | task pass rate | recall@k + faithfulness | format compliance + task metric | task success + cost/success |
| Failure looks like | wrong tone/format | confidently wrong, no source | confidently wrong, well-formatted | fluent narrative over a failed action |
| Hardest part | brittleness | retrieval recall | data labelling | evaluation + safety |

---

## 12. Numbers To Memorize

| Number | Value |
|---|---|
| p^n | 0.95^10 = **59.9%**; 0.95^5 = 77.4% |
| Per-step for 90% over 10 steps | **98.95%** |
| Prefill FLOPs | `2 × N_params × N_tokens`; 7B × 4k = 56 TFLOPs |
| Decode ceiling | 8B fp16 on H100 ≈ 209 tok/s; 4-bit ≈ 838 tok/s |
| Full FT memory | ~16 bytes/param → 8B = 112–128 GB |
| QLoRA memory | 8B = ~9–12 GB; 70B = ~40–48 GB |
| LoRA params | r=16 on 8B ≈ 42M ≈ 0.5% |
| SFT LR / epochs | 1e-4–3e-4 (LoRA) / 1–3 epochs |
| Data for a format task | 2k–10k rows |
| Chunk size / overlap | 300–800 tok / 10–20% |
| k / k_final | 20–50 / 3–10 |
| Reranker | +10–25 nDCG, +50–300 ms |
| Contextual retrieval | −35% / −49% / −67% failure; ~$1.02 per 1M doc tokens |
| Cache read price | ~0.1× input; write ~1.25× |
| Retrieved-context tax | 2k tok × 1M queries ≈ $5,000/mo at $2.50/1M |
| Latency | FT SLM 50–200 ms · FT 8B 0.8–2 s · RAG 1.2–3.5 s · agent 6–90 s |
| Two-hop recall | 0.92² = 0.85; 0.78² = 0.61 |
| Agent cost | $0.066/call at 8 steps; **$0.10/success** at 66% |
| SLM serving | ~$0.000003–0.000008/request |
| Fact-injection limit | <10k facts, quarterly-or-slower, all 4 conditions |
| Golden set / win-rate study | 150–300 items / 200+ position-swapped pairs |
| Ladder cost/time | $0 → 2–15k → 5–60k → 20–200k; hours → days → weeks → months |

---

## 13. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: Target modules ['q_proj'] not found in the base model` | Wrong module names for the architecture | Print `[n for n,_ in model.named_modules()]`; use the arch's names (Llama: `q_proj`; Falcon: `query_key_value`) |
| `print_trainable_parameters()` shows `trainable params: 0` | No module matched | Fix `target_modules` before doing anything else |
| `torch.cuda.OutOfMemoryError` during training | Batch × seq too large, or no gradient checkpointing | Halve batch, enable `gradient_checkpointing=True`, lower `max_seq_length` |
| `RuntimeError: element 0 of tensors does not require grad` | Base model frozen incorrectly / adapters not attached | Call `get_peft_model` before the trainer; `prepare_model_for_kbit_training` for QLoRA |
| `json.decoder.JSONDecodeError: Expecting value: line 1 column 1` | Model emitted a preamble or a markdown fence | Fine-tune for format; add grammar-constrained decoding; greedy decode |
| `AssertionError: pad token id must be set` | Tokenizer has no pad token | `tok.pad_token = tok.eos_token` |
| `UserWarning: max_seq_length is smaller than the longest sample` | Targets are being truncated | Raise `max_seq_length` above the p99 of prompt+target |
| `qdrant_client.http.exceptions.UnexpectedResponse: 400 Wrong vector dimension` | Embedder changed but the collection was created with the old dim | Recreate the collection with the new dim and re-embed everything |
| `openai.BadRequestError: This model's maximum context length is 128000 tokens` | `k_final` × chunk size exceeded the window | Lower `k_final`, lower chunk size, or truncate history |
| `openai.RateLimitError: 429` | Concurrency, not quota | Add exponential backoff with jitter; cap concurrency |
| `KeyError: 'tool_call_id'` | Malformed tool-result message | Every `role: "tool"` message needs the matching `tool_call_id` |
| `TypeError: Object of type float32 is not JSON serializable` | Non-serialisable tool result | Coerce numpy/torch types before `json.dumps` |
| `ragas` returns `nan` for faithfulness | The judge model failed to parse, or no context was supplied | Pin the judge model version; assert contexts are non-empty |
| `AssertionError: recall@k below threshold` in CI | A retrieval regression | Do not merge. Check the chunker, embedder version, and index snapshot |

---

## 14. Copy-Paste Starter Config

`starters/arch_select.py` — one file that (a) scores your situation against the rubric, (b) prints the recommended architecture with the STOP conditions, and (c) emits the starter config for the winning approach.

```python
#!/usr/bin/env python3
"""Architecture selection starter. Fill in SITUATION, run, get a recommendation + a config."""

SITUATION = {
    "volatility":          "daily",     # monthly | quarterly | daily | realtime
    "format_rigidity":     "strict",    # prose  | structured | strict
    "latency_p95_ms":      1500,        # your product's real budget
    "cost_ceiling_per_q":  0.002,       # $ you can afford per query
    "corpus_tokens":       40_000_000,  # total tokens you must make available
    "labelled_rows":       0,           # labelled behaviour examples you HAVE
    "needs_action":        False,       # must the system change the world?
    "needs_citations":     True,        # must every claim be traceable?
    "queries_per_month":   500_000,
}

V = {"monthly": 1, "quarterly": 3, "daily": 5, "realtime": 5}
F = {"prose": 1, "structured": 3, "strict": 5}
S = SITUATION

def score():
    out = {}
    out["volatility"]   = V[S["volatility"]]
    out["format"]       = F[S["format_rigidity"]]
    out["latency"]      = 5 if S["latency_p95_ms"] < 300 else 3 if S["latency_p95_ms"] < 10_000 else 1
    out["cost"]         = 5 if S["cost_ceiling_per_q"] < 0.0005 else 3 if S["cost_ceiling_per_q"] < 0.05 else 1
    out["volume"]       = 5 if S["corpus_tokens"] > 10_000_000 else 3 if S["corpus_tokens"] > 100_000 else 1
    out["labels"]       = 5 if S["labelled_rows"] >= 10_000 else 3 if S["labelled_rows"] >= 2_000 else 1
    return out

def recommend(sc):
    """Always returns (choice: str, stops: list[str])."""
    stops = []
    if S["needs_citations"] and sc["volatility"] >= 3:
        stops.append("S3/S2: citations required + changing content -> weights are disqualified")
    if S["needs_action"]:
        stops.append("S9: action required -> agent, and gate irreversible tools if unattended")
    if sc["latency"] == 5 and sc["volume"] >= 3:
        stops.append("S5: <300 ms with a large corpus -> no retrieval; fine-tune facts in or change the SLO")
    if sc["labels"] <= 1 and sc["format"] == 5:
        stops.append("S6: strict format but <2k labels -> label first (or distil), then fine-tune")

    if S["corpus_tokens"] > 10_000_000 and sc["labels"] <= 1:
        return "RAG", stops
    if sc["format"] >= 5 and sc["volatility"] <= 1 and sc["labels"] >= 3:
        return "FINE-TUNE", stops
    if sc["volatility"] >= 5:
        return "RAG", stops
    if sc["format"] >= 5 and sc["volatility"] >= 3:
        return "HYBRID (FT generator + RAG)", stops
    if sc["latency"] == 5 and sc["cost"] == 5:
        return "FINE-TUNE (small model, local)", stops
    return "PROMPT ENGINEERING / LONG CONTEXT", stops

def config_for(choice):
    if choice.startswith("FINE-TUNE"):
        return {"approach": "qlora_sft", "base": "meta-llama/Llama-3.2-3B-Instruct",
                "r": 16, "alpha": 32, "lr": 2e-4, "epochs": 3, "max_seq_length": 1024,
                "targets": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                "serve": "vllm --quantization awq on 1x L4 24GB",
                "metrics": ["format_compliance", "task_exact_match", "capability_suite", "p95_ms"]}
    if choice.startswith("RAG") or choice.startswith("HYBRID"):
        return {"approach": "hybrid_rag", "chunk": 500, "overlap": 80,
                "embed": "BAAI/bge-base-en-v1.5", "hybrid": "dense + BM25",
                "rerank": "BAAI/bge-reranker-base", "k": 20, "k_final": 5,
                "contextual_retrieval": True,
                "generator": ("gpt-4o-mini" if S["volatility"] in ("quarterly", "daily", "realtime")
                              else "local 3B LoRA"),
                "metrics": ["recall@10", "nDCG@10", "faithfulness", "citation_correctness",
                            "abstention_correctness"]}
    return {"approach": "prompt_only", "system": "<your system prompt>",
            "few_shot": 5, "long_context": S["corpus_tokens"] < 100_000,
            "metrics": ["task_pass_rate", "p95_ms", "cost_per_query"]}

if __name__ == "__main__":
    import json
    sc = score()
    print("scores:", sc, "\nweighted:",
          round(3*sc["volatility"] + 3*sc["format"] + 3*sc["latency"] + 2*sc["cost"] +
                2*sc["volume"] + 3*sc["labels"], 1))
    choice, stops = recommend(sc)
    print("\nRECOMMENDATION:", choice)
    for s in stops:
        print("  STOP ->", s)
    print("\nstarter config:", json.dumps(config_for(choice), indent=2))
    print("\nBEFORE YOU BUILD: measure the prompt-only baseline on 200 items. No baseline, no project.")
```

```bash
# The three starter commands, whichever the script recommends
python starters/arch_select.py                                    # decide
python train_qlora.py --config out/config.json                    # fine-tune path
python rag_baseline.py --corpus ./docs --golden ./golden.jsonl    # RAG path
python agent_min.py --task "issue refund for order 1234"          # agent path
```

---

## 15. What To Read Next

| If you need to go deeper on | Go to |
|---|---|
| The full reasoning, cost models and worked case studies | **CS-04** |
| Interview practice at all five levels | **IQ-04** |
| Fine-tuning BERT-style encoders for classification/NER | CS-07 |
| Instruction fine-tuning (SFT) in depth | CS-13 |
| LoRA/QLoRA mechanics, rank selection, merging | CS-23 |
| Embedding models and fine-tuning the retriever | CS-22 |
| Domain-adaptive continued pretraining on your own PDFs | CS-12 |
| Preference training when SFT plateaus (DPO/ORPO/RLHF) | CS-14, CS-25, CS-27 |
| GRPO for tool-calling and verifiable agent tasks | CS-26 |
| Quantization for cheap SLM serving | CS-10, CS-11 |
| No-code / YAML training frameworks | CS-15, CS-16, CS-17 |
| The end-to-end capstone pipeline | CS-28 |

**The three sentences to remember:** *Fine-tuning changes behaviour. RAG changes knowledge. Agents change what the system can do.* Climb the ladder — prompt, RAG, fine-tune, agent — and do not skip a rung, because the rung you skip is the baseline that would have told you not to build the one above it.
