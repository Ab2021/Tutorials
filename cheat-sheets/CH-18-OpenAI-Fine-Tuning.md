# CH-18 — OpenAI GPT Fine-Tuning Cheat Sheet (Hosted SFT)

**One-line purpose:** supervised fine-tune a hosted GPT model over an HTTP API — you upload a
JSONL file, set three hyperparameters, and get back a model ID string you call forever.
**Use when:** you want a behaviour baked in with no GPU, no serving stack and no MLOps, and your
task fits in tens of thousands of examples.
**Do NOT use when:** you need the weights (you never get them), you need to quantise, merge, serve
locally or distill from the result, your problem is *knowledge* (that is RAG — CH-04), your problem
is JSON *shape* (that is structured outputs), or you are starting a multi-year project on this
specific platform without reading the Status warning below.

> ## ⛔ STATUS WARNING — READ BEFORE ANYTHING ELSE
>
> **This platform is winding self-serve fine-tuning down, with dated milestones. CS-18 §16.8 is
> load-bearing, not historical colour.**
>
> | Date | Change |
> |---|---|
> | **2026-05-07** | An organisation that has **never** run a fine-tuning job can no longer create one. |
> | **2026-07-02** | No new job for an org with no fine-tuned-model **inference** traffic in the trailing 60 days — a dormant fine-tune cannot be revived. |
> | **2027-01-06** | Existing active customers can no longer create **new** fine-tuning jobs at all. |
> | **Until the base model is deprecated** | Already-trained fine-tuned models **keep serving**. |
>
> The stated reason is the important part: newer base models follow instructions and formats well
> enough that **prompting is cheaper and faster than fine-tuning**. That is the same conclusion
> CH-18 §4.12 and §9.1 reach from first principles, arriving from the vendor's side.
>
> **Treat every date above as a prompt to go and re-check the live status page, not as a fact** —
> wind-downs get extended and reversed. The *shape* to plan around is: hosted fine-tuning is being
> de-emphasised for new users while existing models serve until their base retires.
>
> Everything else on this card is the **canonical anatomy of a hosted fine-tuning API** — a shape
> Azure OpenAI, Vertex AI (CH-19), Bedrock, Together, Predibase and Fireworks all share to a first
> approximation. Learn it here; spend your production bet where it survives.

> **The one sentence that matters.** You are buying a **behaviour**, not a model — and the training
> bill is a rounding error while the **inference markup is forever**.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **The whole API is four calls.** upload → create → poll → chat. | Everything else is validation and arithmetic you do *before* spending money. |
| 2 | **`{"messages": [...]}` JSONL, one object per line — nothing else.** | The only accepted format. `function`/`assistant`/`system`/`user`/`tool`. |
| 3 | **10 examples is the API minimum, not the useful minimum.** | 50–100 distinct rows is the smallest thing worth a job (CS-18 §10.1). |
| 4 | **Training is billed per 1M TRAINING TOKENS, not per hour.** | Only **reinforcement** fine-tuning is hourly (CS-18 §11.1). |
| 5 | **The billed count is bigger than your visible count.** | ~3 tokens per message + 3 reply priming. **+19.6%** measured on short examples (CS-18 §4.5.2). |
| 6 | **`n_epochs` is resolved for you if you leave it `"auto"`.** | `min(25, 100 // N)` below 100 rows — a 10-row dataset trains for **10 epochs** (CS-18 §4.8). |
| 7 | **Read `job.hyperparameters` back, every time.** | What you sent is not what ran. `batch_size=16` on 10 rows is accepted and ignored. |
| 8 | **Fine-tuned inference carries a ~2× markup on both directions, forever.** | This is the bill. 95–99.9% of lifetime cost (CS-18 §11.2). |
| 9 | **Break-even is `P_f < P_b/2 − 2O`.** | Fine-tuning wins on cost only if you can *delete prompt tokens* and beat the markup. |
| 10 | **No `validation_file` means no overfitting signal at all.** | The only automatic generalisation detector the platform gives you. |
| 11 | **"It won't return valid JSON" is a schema problem.** | `response_format` with `strict: true` makes bad output unrepresentable. Fine-tune for tone, never for shape. |
| 12 | **Compute `identical_outputs` vs the base model first.** | Near 1.0 means the fine-tune did nothing and every other eval is wasted money. |

---

## 2. Core Formulas

### 2.1 Token accounting — the visible count is not the billed count

| Concept | Formula | Symbols | Worked example (CS-18 §4.5.2) |
|---|---|---|---|
| **Visible tokens** | `Σ len(encode(m["content"]))` | content only | 562 on the video's 10-row set |
| **Billed tokens** | `Σ (tokens_per_message + Σ encode(v)) + tokens_per_reply` | 3 / 3 in the corrected counter | **672** |
| **Overhead factor** | `billed / visible − 1` | — | `672/562 − 1` = **+19.6%** |
| **Overhead per row** | `n_messages × 3 + 3` | 2-message row → 9 | 60-row fixture → **+17.1%** |
| **Per-example cap** | `min(counted, MAX_TOKENS_PER_EXAMPLE)` | cap applied **after** counting | 16,385 in the reference impl. — **stale** for the `gpt-4.1` line |
| **Training tokens** | `billed_tokens_per_epoch × n_epochs` | — | `672 × 3` = 2,016 |
| **Words → tokens** | `tokens ≈ words × 1.33` | English prose, ±15% | the handbook scripts' estimator |
| **Chars → tokens** | `tokens ≈ chars / 4` | English | `"Hello, how are you?"` = 19 chars = 6 tokens |

> **Overhead is a constant per message, so it does not stay at 19.6%.** It is ~+16% on 56-token
> examples and under +0.5% on 2,000-token examples. It is worst on exactly the tiny demo datasets
> everyone builds first, which is the regime where a surprise bill is most annoying.

### 2.2 Cost — the four terms, and the one that dominates

```text
LIFETIME COST =  T  +  V  +  I  +  M

  T  training    = (train_tokens + val_tokens) × n_epochs × train_price_per_M   one-off
  V  validation  = ~20% of T                                                    one-off, often ignored
  I  inference   = calls × (prompt × in_price + completion × out_price)         RECURRING, monthly
  M  maintenance = engineer time to re-run, re-eval, migrate                    unbounded, never budgeted
```

| Term | Formula | Scaling | Share of a 12-month bill | Lever |
|---|---|---|---|---|
| `T` training | `billed_tokens × epochs × $/1M` | one-off, linear in data × epochs | **0.01–2%** | dataset size, `n_epochs`, model tier |
| `V` validation | `val_tokens × epochs × $/1M` | one-off | <0.5% | split size |
| `I` inference | `calls × (P_in·r_in + P_out·r_out + P_cache·r_cache)` | linear in **volume**, **2× markup** | **95–99.9%** | prompt length, output cap, caching |
| `M` maintenance | engineer-days | unbounded | unbounded | whether an eval harness existed *before* you needed it |

```
                ONE-OFF                  RECURRING (monthly, forever)
OpenAI FT     tokens × epochs      per-token usage ONLY — no idle charge, ~2× markup
Vertex CH-19  tuning cost          endpoint $/HOUR while deployed (idle!) + tokens
Self-hosted   GPU-hours to train   GPU-hours — but YOU stop the instance, marginal ≈ 0
```

### 2.3 Break-even — the inequality that decides whether cost is a reason

Derived in **CS-18 §4.6.5**, assuming a **2× markup on both directions** and the **4:1
output:input price ratio** every model in the family has:

$$P_f \;<\; \frac{P_b}{2} \;-\; 2O$$

| Symbol | Meaning |
|---|---|
| `P_b` | base-model prompt tokens (e.g. a few-shot prompt) |
| `P_f` | fine-tuned prompt tokens (what the behaviour replaced) |
| `O` | output tokens per call |

**General form (any markup `k` and price ratio):** fine-tuning wins on cost when
`P_b·r_in,base + O·r_out,base > P_f·k·r_in,base + O·k·r_out,base`, i.e. when the *deleted* prompt
value exceeds the *markup* paid on everything else. With `k = 2` and `r_out = 4·r_in` it collapses to
the boxed inequality above.

| Case (all `gpt-4.1-mini`) | `P_b` | `O` | Break-even `P_f` | Actual `P_f` | Verdict |
|---|---|---|---|---|---|
| 8-shot support prompt | 1,200 | 150 | < 300 | 400 | **FT loses** |
| 30-shot extraction prompt | 3,000 | 90 | < 1,320 | 700 | **FT wins** |
| 2-shot, short, long output | 450 | 200 | < −175 (impossible) | 300 | **FT loses at any prompt length** |
| §11.3 worked budget | 900 | 120 | < 210 | 500 | **FT loses — +$46.40/month, no payback** |
| §11.3 second version | 900 | 120 | < 210 | 180 | **FT wins — $4.80/month, 4.4-month payback** |

> **The strategic consequence.** The only hosted fine-tunes that pay for themselves are the ones
> that **replace a large block of few-shot examples**. If your prompt is short, fine-tuning will
> *never* pay back on hosted economics — you fine-tune for quality or format compliance, and you
> should say so out loud, because **"it will be cheaper" will be false**.

---

## 3. Decision Tree

```
Is the problem a SHAPE problem ("it won't return valid JSON")?
├─ Yes → response_format / structured outputs with strict:true. Stop.
│         Constrained decoding = 100%, free, one afternoon. (CS-18 §4.12, §15.3)
└─ No ↓
Is the problem KNOWLEDGE (facts, documents, a changing corpus)?
├─ Yes → RAG. Fine-tuning on facts produces a CONFIDENT hallucinator — it is
│         trained to answer in-domain, so it invents in your house style. (CH-04; CS-18 §4.12, §17 #2)
└─ No — it is a BEHAVIOUR (tone, format style, refusal boundary, vocabulary) ↓
Can a shorter/cleaner prompt + structured outputs get you there, and do you have
≥50 distinct, high-quality examples with a held-out set?
├─ No prompt → do that first and MEASURE. It costs nothing and it is your baseline.
├─ No data   → STOP. 10 rows makes a template, not a model (CS-18 §4.9.3).
│              Generate synthetic data (CH-09) or use prompting.
└─ Yes ↓
Do you need the WEIGHTS (on-prem, merge, quantise, self-host, distill a student)?
├─ Yes → hosted is disqualified. CH-13 → CH-15/16/17 (train) → CH-10/11 (quantise).
└─ No ↓
Is your volume high and steady (roughly >450k calls/month)?
├─ Yes → price open-weight self-hosting from day one (CS-18 §11.5). The hosted
│         column is a floor that scales with your success; self-hosting is a
│         step function you can grow by adding GPUs.
└─ No / bursty — and are you starting a NEW project on this platform after 2026?
   ├─ Yes → read the Status warning. Prefer a surviving platform; if you stay,
   │         script the pipeline AND start the distillation capture on day one.
   └─ No, or you accept that ↓
Compute the break-even BEFORE you create a job:
    P_f < P_b/2 - 2O  →  wins on cost
    P_f >= P_b/2 - 2O →  proceed only for QUALITY reasons, written down
Then: validate → count → budget → upload → create → poll → EVALUATE vs base.
```

---

## 4. Hyperparameter Quick Reference

### 4.1 The three knobs — that is the entire training surface

| Param | What it does | Default | Safe range | Too high → | Too low → | API path |
|---|---|---|---|---|---|---|
| `n_epochs` | Full passes; **linear multiplier on the bill** | `"auto"` → target-example policy (§4.2) | **1–3** (≥1k rows); 3–10 (<100 rows) | Memorisation; `valid_loss` rises while `train_loss` falls | Behaviour change never lands | `method.supervised.hyperparameters.n_epochs` |
| `batch_size` | Examples per optimiser step in the provider's trainer | `"auto"` → **~0.2% of rows, capped 256, floored 1** | `"auto"` unless you have a reason | Fewer, larger steps; LR mismatch | Noisier gradient; LR mismatch | `...hyperparameters.batch_size` |
| `learning_rate_multiplier` | A **multiplier on a hidden base LR** — not a learning rate | `"auto"` (observed resolving to **2.0** on tiny data) | 0.5–2 (chat-era); legacy 0.02–0.2 | Divergence, NaN, forgetting | **No learning at all** — job "succeeds", model unchanged | `...hyperparameters.learning_rate_multiplier` |
| `beta` (DPO only) | KL anchor strength to the reference model | `"auto"` | 0.01–0.5 | Model barely moves from SFT | Likelihood collapse on both chosen and rejected | `method.dpo.hyperparameters.beta` |

> **Correction (CS-18 §4.7.2):** `learning_rate_multiplier: 1.0` means **"use the provider's
> default base rate"** — not "zero learning rate" and not `1e-5`. The actual rate is
> `base_lr × multiplier` and `base_lr` is not exposed. Treat the knob as **ordinal** ("same as
> default" / "more" / "less"), never cardinal, and never compare its value across providers.
> If you need a real learning rate, that is an argument for open-weight training (CH-13 §4.1).

### 4.2 The auto-epoch policy, in full (CS-18 §4.8)

```python
TARGET_EPOCHS        = 3
MIN_TARGET_EXAMPLES  = 100
MAX_TARGET_EXAMPLES  = 25000
MIN_DEFAULT_EPOCHS   = 1
MAX_DEFAULT_EPOCHS   = 25

n_epochs = TARGET_EPOCHS
if len(data) * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // len(data))   # min(25, 100 // N)
elif len(data) * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // len(data))
```

| Training rows | `N × 3` | Regime | Resolved `n_epochs` | Billed passes |
|---|---|---|---|---|
| **10** | 30 | 1 — "too little data, crank the epochs" | **10** | 100 examples-worth |
| 25 | 75 | 1 | **4** | 100 |
| 33 | 99 | 1 | **3** | 99 |
| 50 | 150 | 2 — flat | **3** | 150 |
| 1,000 | 3,000 | 2 | **3** | 3,000 |
| 8,333 | 24,999 | 2 | **3** | 24,999 |
| 8,400 | 25,200 | 3 — "too much data, throttle" | **2** | 16,800 |
| 25,000 | 75,000 | 3 | **1** | 25,000 |
| 200,000 | 600,000 | 3 | **1** | 200,000 |

**This was verified against a real job object** (CS-18 §4.7.3): a 10-row job created with no
hyperparameters came back as
`Hyperparameters(batch_size=1, learning_rate_multiplier=2.0, n_epochs=10)` — exactly what
`min(25, 100 // 10)` predicts, with `batch_size` floored to 1 and the LR policy keyed to that batch
size.

> **Beyond the video:** the policy bounds **cost**, not quality, and its two regimes pull in opposite
> directions. With **50 good rows** it gives you 2 epochs, which is almost certainly *too few* —
> raise it explicitly. With **200,000 rows** it gives you 1 epoch, which is probably right, but at
> that bill an open-weight LoRA costs a fraction and you keep the weights. **Always set `n_epochs`
> explicitly and read the resolved value back** — the provider's policy can change without an API
> version bump.

### 4.3 Job-level configuration — the fields the video omits

| Param | Typical | Why it matters | Omit it and… |
|---|---|---|---|
| `model` | A **dated snapshot ID**, not the alias | The alias moves under you; the snapshot does not | You silently train a different base next month |
| `training_file` | `file-...` ID | Provenance; the same validated file can back multiple jobs | — |
| `validation_file` | **10–20% holdout, always set** | The only automatic overfitting detector | **You get no generalisation signal at all** |
| `suffix` | ≤18 chars, e.g. `support-v3` | The only human handle on the artefact | You cannot map models to data |
| `seed` | `42` | A/B comparisons need it | Two runs differ and you cannot attribute it |
| `metadata` | `{"data_version": ..., "git_sha": ...}` (≤16 pairs) | Answers "what trained this?" in six months | The honest answer is "I don't know" |
| `integrations` | W&B / similar | Loss curves without polling | You poll by hand |
| `method.type` | `"supervised"` / `"dpo"` / `"reinforcement"` | Selects the trainer | Defaults are version-dependent |

### 4.4 Inference-time knobs — where the money actually lives

| Param | Cost effect | Quality effect | For a fine-tuned model |
|---|---|---|---|
| `model` | The FT ID carries ~2× input and ~2× output price | — | The FT ID — but check §2.3 first |
| `messages` length | **Linear in input cost** | The system prompt you trained with must be present and **byte-identical** | Trim only prompt you did not train with |
| `max_tokens` | Caps the most expensive side | Too low truncates JSON mid-object | **Always set it** |
| `temperature` | None | 0–0.3 extraction; 0.7+ chat | **0.2** |
| `response_format` | Adds constrained decoding, no training cost | **Guarantees** the schema | Use it *in addition to* the fine-tune |
| `frequency_penalty` / `presence_penalty` | None | Can break a learned style | **0** |
| `stop` | Truncates output early = cheaper | Needed for some templates | Set if your format has a terminator |
| `n` | Linear in output cost | — | 1 in production |
| Prompt caching | Cached input at ~25% of the input price | None | **Put the stable system prompt first** — the largest inference lever you control |

### 4.5 Data sizing rules of thumb

| Goal | Examples | Evidence |
|---|---|---|
| API will accept the file | **10** | The platform's guard rail, not a target |
| Practical floor for a behaviour change | **50–100** | CS-18 §10.1; better than most tutorials |
| Format / style adoption | 100 – 1,000 | CH-13 §4.3 |
| A narrow task with real diversity | 500 – 5,000 | The cost-justified case in CS-18 §15.2 used 8,000 |
| Preference pairs (DPO) | 100+ before signal is meaningful | Pairs, not rows, are the scarce resource |

---

## 5. Copy-Paste Code Snippets

### 5.1 The JSONL schema, minimally and maximally

```json
{"messages":[{"role":"user","content":"How long is the warranty?"},{"role":"assistant","content":"One year, covering manufacturing defects."}]}
```

```json
{"messages":[{"role":"system","content":"You are a customer support assistant for a smartphone company. You are friendly, concise, and provide only factual answers related to smartphones."},
             {"role":"user","content":"How long is the warranty?"},
             {"role":"assistant","content":"Most smartphones include a one-year limited warranty that covers manufacturing defects."}]}
```

```json
{"messages":[{"role":"system","content":"You are a support agent with access to order lookup."},
             {"role":"user","content":"Where is order 88231?"},
             {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup_order","arguments":"{\"id\":\"88231\"}"}}]},
             {"role":"tool","tool_call_id":"call_1","content":"{\"status\":\"in_transit\",\"eta\":\"2026-10-02\"}"},
             {"role":"assistant","content":"Order 88231 is in transit and should arrive by 2 October."}]}
```

| Rule | Detail |
|---|---|
| Top-level key | `messages`, a **list**. Nothing else is required; `chosen`/`rejected` are added for DPO. |
| Roles | `system`, `user`, `assistant`, `tool`. (`function` is legacy-only.) |
| `content` | A non-empty string — or `null` on an assistant turn that only calls a tool. |
| Supervised span | The `assistant` turns. Everything else is context — **billed but not learned from**. |
| Last turn | Should be `assistant`; otherwise the row has no completion to learn from. |
| One bad line | Fails the **whole file** at `validating_files`. |
| Format | JSONL, one object per line — `.json` (an array) is rejected. |

### 5.2 The validator — seven checks plus the three the platform will never run

```python
from collections import defaultdict
import json

# Pin this beside your SDK version. The completion-era list is
# ("system", "user", "assistant", "function") and WILL false-positive on modern data.
VALID_ROLES    = ("system", "developer", "user", "assistant", "tool")
ALLOWED_KEYS   = ("role", "content", "name", "tool_calls", "tool_call_id",
                  "function_call", "refusal", "weight")   # `weight` is newer-schema
MAX_EXAMPLES, MIN_EXAMPLES = 50_000, 10

def validate(path):
    errors = defaultdict(int)
    rows = []
    # Parse PER LINE: json.loads raises on the first bad line, so a naive comprehension
    # makes a 10,000-row file with one bad row at 4,000 look like a 3,999-row file.
    for i, line in enumerate(open(path, encoding="utf-8-sig"), 1):   # -sig for the Excel BOM
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            errors[f"json_decode_error:line={i}"] += 1
            continue
        rows.append(ex)

    for i, ex in enumerate(rows):
        if not isinstance(ex, dict):                       # CHECK 1
            errors["data_type"] += 1
            continue
        messages = ex.get("messages")
        if not messages:                                   # CHECK 2
            errors["missing_messages_list"] += 1
            continue
        if not isinstance(messages, list):                 # the hole the cookbook leaves open
            errors["messages_not_a_list"] += 1
            continue
        for m in messages:
            if "role" not in m or "content" not in m:      # CHECK 3
                errors["message_missing_key"] += 1
            if any(k not in ALLOWED_KEYS for k in m):      # CHECK 4
                errors["message_unrecognized_key"] += 1
            if m.get("role") not in VALID_ROLES:           # CHECK 5
                errors["unrecognized_role"] += 1
            content = m.get("content")
            has_call = m.get("tool_calls") or m.get("function_call")
            if (not content and not has_call) or (content is not None and not isinstance(content, str)):
                errors["missing_content"] += 1             # CHECK 6
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                errors["name_on_wrong_role"] += 1 if "name" in m else 0
        if not any(m.get("role") == "assistant" for m in messages):   # CHECK 7
            errors["example_missing_assistant_message"] += 1
        roles = [m.get("role") for m in messages]
        if roles[-1] != "assistant":
            errors["last_message_not_assistant"] += 1

    # ---- the three checks the platform will never run for you --------------------
    n = max(len(rows), 1)
    uniq = len({tuple((m.get("role"), m.get("content")) for m in ex["messages"]) for ex in rows})
    if uniq / n < 0.95:
        errors["low_uniqueness_ratio"] += 1                # export ran twice / dedup key wrong
    sys_prompts = {next((m["content"] for m in ex["messages"]
                         if m.get("role") == "system"), None) for ex in rows}
    if len(sys_prompts) > 1:
        errors["inconsistent_system_prompt"] += 1          # the model averages two voices
    if len(rows) < MIN_EXAMPLES:
        errors["too_few_examples"] += 1
    if len(rows) > MAX_EXAMPLES:
        errors["too_many_examples"] += 1

    print(f"rows={len(rows)}  unique_convs={uniq}  ratio={uniq/n:.2f}")
    print(dict(errors) if errors else "no structural problems found")
    return len(errors) == 0                                 # exit non-zero in CI
```

| # | Error key | What it ensures | The trap |
|---|---|---|---|
| 1 | `data_type` | Each line is a JSON object | — |
| 2 | `missing_messages_list` | `messages` exists and is non-empty | — |
| 3 | `message_missing_key` | Every message has `role` and `content` | Counting is **per violation, not per example** — checks 3–6 do not `continue`, so one message can raise several |
| 4 | `message_unrecognized_key` | No unexpected keys | **The allow-list is a versioned artefact.** See the Correction below |
| 5 | `unrecognized_role` | Role is in the allow-list | Same |
| 6 | `missing_content` | Non-empty string, or a tool call | `(not content and not function_call)` treats `""`, `None`, `0`, `[]` identically |
| 7 | `example_missing_assistant_message` | A supervised turn exists | Fires only when there is **no** assistant turn at all — an assistant turn with `content: ""` trips `missing_content` instead |

> **Correction (CS-18 §6.3):** the role allow-list copied from the original cookbook —
> `("system", "user", "assistant", "function")` — is the **completion-era** list and produces
> **false positives** on modern data:
>
> | Your row | Copied validator says | Reality |
> |---|---|---|
> | `{"role":"tool","tool_call_id":"call_1","content":"..."}` | `unrecognized_role: 1` | Valid |
> | `{"role":"developer","content":"..."}` | `unrecognized_role: 1` | Accepted as a system-equivalent |
> | `{"role":"assistant","content":null,"tool_calls":[...]}` | `missing_content: 1` | **The correct modern shape** for a tool-only assistant turn |
>
> The last row is the dangerous one: the validator counts a **correct** row as broken and inflates
> the histogram until a data problem seems to exist. **A copied allow-list is a versioned artefact,
> and yours is now older than the API.** Pin it beside your SDK version and re-read it on upgrade.
> The handbook's own `code/13_openai_finetune.py` accepts `tool` but not `developer` — so it is
> *newer* than the cookbook and *older* than CS-18 §6.3's corrected tuple.

> **Beyond the video:** make the validator a **CI gate that exits non-zero**, not a notebook cell
> that prints "No errors found". Wire it to `pytest` with a fixture file of deliberately-broken rows
> and assert each check fires. Then add the checks the platform has no concept of: **uniqueness
> ratio** (refuse below ~0.9), **task entropy** (cluster assistant turns — ten clusters from 1,000
> rows means one behaviour repeated), and **label balance** (a dataset that is 95% answers produces
> a model that never refuses; one that is 40% refusals produces an over-refusing model).

### 5.3 Upload + create + poll (the four calls, production-shaped)

```python
import os, time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()

# 1. UPLOAD. Binary mode. `purpose` is mandatory and easy to omit.
train_file = client.files.create(file=open("train.jsonl", "rb"), purpose="fine-tune")
val_file   = client.files.create(file=open("val.jsonl",   "rb"), purpose="fine-tune")

# 2. INSPECT. `uploaded` is not ready; wait for `processed`.
for f, path in ((train_file, "train.jsonl"), (val_file, "val.jsonl")):
    assert client.files.retrieve(f.id).bytes == os.path.getsize(path)
    while client.files.retrieve(f.id).status != "processed":
        time.sleep(5)

# 3. CREATE. Snapshot ID, holdout, seed, metadata — the four fields tutorials omit.
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,                 # <- never None in a real run
    model="gpt-4.1-mini-2025-04-14",             # dated SNAPSHOT, not the alias
    suffix="support-v3",
    method={"type": "supervised",
            "supervised": {"hyperparameters": {"n_epochs": 3,
                                               "batch_size": "auto",
                                               "learning_rate_multiplier": "auto"}}},
    seed=42,
    metadata={"data_version": "2026-09-01", "git_sha": os.environ.get("GIT_SHA", "dev")},
)
assert job.training_file != job.validation_file, "train and val are the same file"

# 4. POLL TO A TERMINAL STATE — never sleep a fixed interval and hope.
TERMINAL = {"succeeded", "failed", "cancelled"}
while True:
    j = client.fine_tuning.jobs.retrieve(job.id)
    if j.status in TERMINAL:
        break
    time.sleep(30)
if j.status != "succeeded":
    raise RuntimeError(f"fine-tune {j.id} ended {j.status}: {j.error}")
model_id = j.fine_tuned_model                  # guaranteed non-null here
print(model_id, j.trained_tokens, j.hyperparameters)   # <- READ THE RESOLVED KNOBS BACK
```

```python
# The metrics CSV — the only place the loss curves live.
csv_id = client.fine_tuning.jobs.retrieve(job.id).result_files[0]
print(client.files.content(csv_id).text)
# columns: step, train_loss, valid_loss, full_valid_loss,
#          train_mean_token_accuracy, valid_mean_token_accuracy

# Clean up: retention is YOUR responsibility. Files persist until deleted.
client.files.delete(train_file.id)
client.files.delete(val_file.id)
```

### 5.4 The one evaluation to run first — `identical_outputs`

```python
import json
base_id, ft_id = "gpt-4.1-mini-2025-04-14", model_id

def run(m, prompts, temperature=0.2):
    out = []
    for p in prompts:
        r = client.chat.completions.create(
            model=m, temperature=temperature, seed=42, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": p}])
        out.append(r.choices[0].message.content)
    return out

prompts = [json.loads(l)["prompt"] for l in open("frozen_eval_prompts.jsonl", encoding="utf-8")]
base, ft = run(base_id, prompts), run(ft_id, prompts)
print("identical:", sum(a == b for a, b in zip(base, ft)), "/", len(prompts))
```

| `identical_outputs` | Reading |
|---|---|
| **~1.00** | The fine-tune **did nothing** — LR multiplier too low, rows truncated, or the wrong file was uploaded. Stop and investigate before spending on eval. |
| 0.5–0.9 | A limited, probably **format-related** change. Expected on a small dataset. |
| 0.1–0.5 | Substantive change. Now you need task metrics to know if it is the *right* change. |
| **< 0.1** | Near-total replacement. On a small dataset this is usually **memorisation**, not generalisation. |

---

## 6. CLI Commands

```bash
# ── The handbook's script (validate / estimate / upload / train) ─────────────────
python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --validate
python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --estimate --epochs 3
python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --estimate \
    --model gpt-4.1-mini-2025-04-14 --inference-volume 200000
python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --upload --suffix support-v3
python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --train \
    --model gpt-4.1-mini-2025-04-14 --suffix support-v3 --epochs 3
```

| Flag | Does | Note |
|---|---|---|
| `--validate` | Local structural pre-flight, no API calls | **Also runs `--estimate`** — the script runs the estimate whenever neither `--upload` nor `--train` was passed |
| `--estimate` | Cost arithmetic, no API calls | Same coupling in reverse: it also validates |
| `--upload` | `files.create(purpose="fine-tune")` | **Runs neither the validator nor the estimator.** Uploading an unvalidated file is one flag away |
| `--train` | Creates a job against the newest uploaded fine-tune file | Picks `files.list()[0]` — see §11 |
| `--inference-volume` | Monthly calls for the projection | Default 100,000 |
| `--avg-output-tokens` | Assumed completion length | Default 250 |

```bash
# ── Environment ─────────────────────────────────────────────────────────────────
pip install "openai>=1.40,<3" tiktoken      # PIN. `method={...}` nesting replaced flat hyperparameters in 2.x
export OPENAI_API_KEY="sk-..."              # never inline it in a script

# ── Count tokens locally before you are billed for them ─────────────────────────
python -c "
import tiktoken
enc = tiktoken.get_encoding('o200k_base')   # gpt-4o / gpt-4.1 families.
                                            # 'cl100k_base' is the gpt-4/gpt-3.5 era — a small
                                            # systematic error, worse on non-Latin and code.
print(len(enc.encode('Hello, how are you?')))   # 6
"

# ── Watch a job without the SDK ─────────────────────────────────────────────────
python -c "from openai import OpenAI; import sys; \
print(OpenAI().fine_tuning.jobs.retrieve(sys.argv[1]).status)" ftjob-XXXXXXXX

# ── Verify the current fine-tuning availability BEFORE budgeting ─────────────────
# Check the platform's live status and deprecations page. The dates in the Status
# warning at the top of this file are a prompt to go and look, not a fact.
```

---

## 7. Cost Calculator

### 7.1 What the script actually prints (verified run)

`python code/13_openai_finetune.py --data code/data/sample_sft.jsonl --estimate
--model gpt-4.1-mini-2025-04-14 --epochs 3 --inference-volume 200000`

| Line item | Script output |
|---|---|
| Examples / file size | 60 / 0.02 MB |
| est. tokens/epoch | **3,149** (in 1,075 / out 2,074) |
| est. train tokens (3 epochs) | **9,447** |
| **Training cost** | **$0.03** |
| Inference input (200k calls) | 3,583,333 → $1.43 |
| Inference output (200k calls, 250 tok) | 50,000,000 → $80.00 |
| **Monthly / annualised** | **$81.43 / $977.20** |
| Month 1 / Year 1 | $81.46 / $977.23 |
| Training share of year 1 | **0.0%** |
| "training pays back in" | 0.0 months |

> **Correction — four things this output is not telling you.** The script is a useful skeleton, but
> read it as a *lower bound on the training bill and a mislabel on the payback line*:
>
> | # | Issue | Effect |
> |---|---|---|
> | 1 | **The inference prices are the BASE model's, not the fine-tuned model's.** The `gpt-4.1-mini` row is `(train 3.00, in 0.40, cached 0.20, out 1.60)` — but `in 0.40`/`out 1.60` are the **base** rates; the fine-tuned rates are `0.80`/`3.20` (CS-18 §13.3). The cached figure **is** the FT one. | The recurring bill — the one that matters — is understated by up to **2×**. |
> | 2 | **The estimator omits per-message overhead.** It uses `words × 1.33` over content only. On this fixture that is 3,149 content tokens across 60 two-message rows; the platform also bills `2 × 3 + 3 = 9` overhead tokens per row = **540 more, +17.1%**. CS-18 measured **+19.6%** on the video's shorter examples. | Training bill understated by ~17%, and it is worst on short examples. |
> | 3 | **"Training pays back in X months of inference" is not a payback.** It computes `train_cost / monthly_inference`, which is finite and near zero for any dataset. A real payback is `T / (base_monthly − ft_monthly)` (CS-18 §11.3) — and it is **infinite when the fine-tune costs more to run**, which is the common case. | The single most misleading line in the output. |
> | 4 | **Inference prompt length is taken from the TRAINING rows' average input**, not your production prompt. | The projection is a shape, not an estimate. Substitute your real `P_f` and `P_b`. |

### 7.2 Prices on file in the script (VERIFY BEFORE BUDGETING)

USD per 1M tokens. The script's own header says: *"treat every number here as verify before you
commit budget."*

| Model family | Train | Input | Cached in | Output |
|---|---|---|---|---|
| `gpt-4o-mini` | $3.00 | $0.30 | $0.15 | $1.20 |
| `gpt-4o` | $25.00 | $3.75 | $1.875 | $15.00 |
| `gpt-4.1-mini` | $3.00 | $0.40 | $0.20 | $1.60 |
| `gpt-4.1` | $25.00 | $2.00 | $1.00 | $8.00 |
| `gpt-3.5-turbo` | $8.00 | $3.00 | $1.50 | $6.00 |

For comparison, CS-18 §13.3 records the **fine-tuned** `gpt-4.1` family rates as of Sept 2026:
`gpt-4.1-nano` **$1.50/M train**, $0.20 in, $0.05 cached, $0.80 out; `gpt-4.1-mini` ~**$3.00/M
train**, $0.80 in, $0.20 cached, $3.20 out; `gpt-4.1` ~**$25/M train**, ~$6 in, ~$1.50 cached,
~$24 out. Every one is a clean **2×** over its own base model.

> ⚠️ **These are snapshots of a price list, not a price list.** The fine-tuning line-up and its
> prices churn monthly (CS-18 §4.4.2), several of the models the video names are already retired,
> and the whole platform is on a wind-down track. **The structure of the arithmetic is the durable
> part — `billed_tokens × epochs × $/1M` for training, `calls × (P·r_in + O·r_out)` for inference.**
> Verify every number against the live pricing page before you attach it to a budget.

> **Correction — the price lookup mis-prices the model CS-18 names as the demo model.**
> The script matches by substring in insertion order: `next(k for k in PRICES if k in model)`.
> So `--model gpt-4.1-nano-2025-04-14` matches **`gpt-4.1`** at **$25.00/M** — a **16.7×
> overestimate** of nano's $1.50/M — and it does so **silently**, because the code only warns when
> *nothing* matches. A `gpt-4.1` prefix shadowing `gpt-4.1-mini` is one dict reorder away. Never
> trust a prefix match on a model ID; match the exact snapshot.

### 7.3 The corrected arithmetic for the same fixture

| Quantity | Script | Corrected | Why |
|---|---|---|---|
| Content tokens/epoch | 3,149 | 3,149 | `words × 1.33` |
| Per-message overhead | — | **+540** (60 rows × 9) | 3/msg + 3 reply priming |
| **Billed tokens/epoch** | 3,149 | **~3,689** | +17.1% |
| **Training tokens (3 ep)** | 9,447 | **~11,067** | × 3 |
| **Training cost** | $0.03 | **~$0.03** | at $3.00/M — the training bill really is a rounding error |
| Inference / month @200k | $81.43 | **$162.87** | FT rates: in $0.80, out $3.20 |
| **Year 1** | $977.23 | **$1,954.44** | the recurring 2× is the whole story |

### 7.4 Fixed vs usage — and who you are really competing with

| Calls/month | Hosted base (`gpt-4.1-mini`, 900/120) | Hosted FT (500/120) | Self-host a 3B LoRA, 1×24 GB, 24/7 | Cheapest |
|---|---|---|---|---|
| 20,000 | $11.04 | $15.68 | $252 | **Hosted base** |
| 100,000 | $55.20 | $78.40 | $252 | **Hosted base** |
| **500,000** | $276.00 | $392.00 | **$252** | *crossover ≈ 460k calls* |
| 1,000,000 | $552.00 | $784.00 | $252 | **Self-host** |
| 50,000,000 | $27,600.00 | $39,200.00 | $2,520 (10 GPUs) | **Self-host, 10×** |

*Assumptions are CS-18 §11.5's: ~2,000–4,000 aggregate tok/s on one 24 GB card at $0.35/GPU-hour,
730 h/month. A 7B roughly halves throughput and moves the crossover to ~230k calls. Serverless /
scale-to-zero GPU changes the low end entirely.*

> **Cross-reference — CH-19 §7.1 is the same arithmetic with the opposite shape.** On Vertex the
> **fixed** hourly endpoint charge dominates ($2,190/month at $3/hour) and the per-token usage is
> noise until ~28.7M calls/month. Here there is **no idle charge at all** and the **per-token
> premium is the whole bill**. Both cards therefore reach the same conclusion from opposite
> directions: **the tuning cost is a rounding error and the recurring term decides the project.**

### 7.5 The four-line budget function

```python
def total_cost(train_rows, val_rows, avg_tokens, epochs, train_price,
               calls, base_prompt, ft_prompt, out_tokens,
               base_in, base_out, ft_in, ft_out):
    billed = avg_tokens + 9                      # 3 per message + 3 reply, 2-message row
    train  = (train_rows + val_rows) * billed * epochs / 1e6 * train_price
    base_m = calls * (base_prompt/1e6*base_in + out_tokens/1e6*base_out)
    ft_m   = calls * (ft_prompt  /1e6*ft_in   + out_tokens/1e6*ft_out)
    return {"train_one_off": round(train, 2), "base_per_month": round(base_m, 2),
            "ft_per_month": round(ft_m, 2),
            "payback_months": round(train / (base_m - ft_m), 1) if ft_m < base_m else None}

# CS-18 §11.3's scenario: 5,000 rows, 3 epochs, 200k calls, 900 -> 500 prompt tokens
print(total_cost(5_000, 800, 400, 3, 3.00, 200_000, 900, 500, 120,
                 0.40, 1.60, 0.80, 3.20))
# {'train_one_off': 21.35, 'base_per_month': 110.4, 'ft_per_month': 156.8, 'payback_months': None}
```

**`payback_months: None` is the answer more often than anyone expects.** The fine-tune makes the
system **more** expensive to operate, permanently. If it is still worth doing, it is worth doing for
a reason that is not on this spreadsheet: brand voice, a liability the base model creates by
promising things it cannot deliver, or refusals that cost more in support tickets than the delta.

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Surprise bill, 3–10× the estimate** | Auto `n_epochs` resolved high on a small dataset | Read `job.hyperparameters`; set `n_epochs` explicitly (§4.2) |
| **Training bill ~17–20% above your own count** | You counted content tokens only | Add 3 per message + 3 reply priming (§2.1) |
| Job fails in `validating_files` | Any of the seven structural errors | Fix the file and **re-upload**; never re-run the same file ID |
| `unrecognized_role` on valid `tool` rows | A completion-era allow-list | Widen to `("system","developer","user","assistant","tool")` (§5.2) |
| `missing_content` on a correct tool-call row | `content: null` with `tool_calls` populated | Allow null content when a call is present |
| `KeyError: 'data_type'` | Dict key read before assignment | `format_errors = collections.defaultdict(int)` |
| Validator reports fewer rows than the file has | One bad line aborts the whole comprehension | Parse per line in a `try/except` and collect line numbers |
| `JSONDecodeError` on **line 1 only**, or a stray `\r` per row | A UTF-8 BOM, or Windows line endings | Read `encoding="utf-8-sig"`; write `newline="\n"` |
| **`404 model_not_found`** on an `ft:` ID | Job is not `succeeded` — running, failed, or cancelled | Poll to a terminal state; a cancelled job **never mints a model** |
| Model replies, but it is obviously the base model | You passed the base ID, or the fine-tune learned nothing | Check the ID starts with `ft:`; then compute `identical_outputs` (§5.4) |
| Chat demo looks great, production is wrong | You asked a question that was in the training set | Paraphrase test; frozen held-out set |
| Out-of-scope questions get answered confidently | All-positive training data erased the refusal boundary | **Add refusal examples and re-run — it is a data fix, not a prompt fix** |
| `valid_loss` **below** `train_loss` | Leak — duplicate rows across the split | Content-hash every row; dedupe **before** splitting |
| `valid_loss` noisy and unreadable | Validation file too small | ≥100 validation rows, or read `full_valid_loss` (end-of-epoch) |
| `train_loss → ~0` | Memorisation (typical below 100 rows) | Cut `n_epochs`; get more *distinct* data; you have a template |
| Both losses flat from step 1 | LR multiplier too low, wrong file, or no signal | Raise the multiplier 2× once; confirm the file; confirm the base model can do the task when prompted |
| Loss goes to NaN | LR multiplier far too high, or one pathological row | Reset to 1.0; trim the longest examples |
| `trained_tokens` far below your estimate | **Silent truncation** of over-long rows | Compare your longest rows to the per-example cap; pre-truncate yourself (§10) |
| `succeeded` but `fine_tuned_model` is `null` | Short poll window — the ID lags the status | Poll on `fine_tuned_model is not None`, not on `status` alone |
| Job stuck in `queued` for hours | Provider capacity | Wait. Do not cancel and restart — a restart rejoins the same queue |
| Output truncated mid-JSON | `max_tokens` too low | Raise it, and add `response_format` so the decoder cannot open an object it cannot close |
| Cost per call doubled overnight | Prompt grew, or cache hit rate collapsed | Alert on cost-per-1k-calls; pin the prompt; keep the stable prefix first |
| Quality drops in production but not in tests | The system prompt differs from training | Hash the system prompt and assert it at import time (§12) |
| Repeated phrases / degeneration | `temperature` too high for a fine-tuned model | Drop to ≤1.0; fine-tunes narrow the distribution |
| Fine-tune is worse than the base at everything | Overfit, or the training data corrected the base's *good* answers | **Roll back — it is a one-line config change** (the one real advantage here) |
| Two teams see different behaviour from one model | Two different system prompts, one trained identity | One system prompt per model ID, owned in one place |

---

## 9. Comparison Matrix

### 9.1 Hosted FT vs the four nearest alternatives

| Dimension | **Hosted FT** | **Open-weight FT** (CH-13) | **Prompt engineering** | **RAG** (CH-04) | **Structured outputs** |
|---|---|---|---|---|---|
| Deliverable | An endpoint **ID** | Weights you own | A prompt string | An index + a prompt | A schema |
| Weights | **Never** | **Yours** | n/a | n/a | n/a |
| One-off cost | token-billed; $0.003–$500 typical | GPU-hours; $0–$300 for a LoRA | $0 | index build | $0 |
| Per-call cost | base price **× 2** | ~$0 marginal on your GPU | base price (longer prompt) | base + retrieval | base price |
| Time to first result | **Minutes** | Hours to days | Minutes | Days | Minutes |
| Teaches **behaviour** | Yes | Yes | Fragilely | No | No |
| Teaches **facts** | Poorly, and dangerously | Poorly | No | **Yes** | No |
| Guarantees **shape** | No | No | No | No | **Yes, exactly** |
| Ops burden | **None** | Serving stack + on-call | None | Vector DB + freshness | None |
| Loss / LR control | **Three knobs, one hidden** | All of them | n/a | n/a | n/a |
| Merge / quantise | **Impossible** | Yes | n/a | n/a | n/a |
| Vendor lock-in | **Total** | Low | Low | Medium | Low |
| Use when | No GPUs, a supported base model, a behaviour change, small data | You need the artefact, the control, or the volume | The behaviour is already close | The problem is knowledge | The problem is shape |

### 9.2 The four hosted methods

| | **SFT** | **Vision FT** | **DPO** | **RFT** |
|---|---|---|---|---|
| Teaches | A behaviour from demonstrations | The same, over images | A **preference** between two behaviours | A **verifiable skill** |
| Data shape | `messages` | `messages` + `image_url` | `messages` + `chosen` + `rejected` | prompts + a **grader** |
| Human effort per row | 1 completion | 1 completion + an image | **2 completions, ranked** | A grader you must write |
| **Billing** | **Per 1M tokens** | **Per 1M tokens** | **Per 1M tokens** | **Per HOUR** |
| Typical `n_epochs` | 1–3 | 1–3 | **1** | n/a |
| Signature failure | Memorisation on small data | Same | Likelihood collapse | Reward hacking the grader |
| Prerequisite | — | — | **An SFT model** | Automatically verifiable answers |

> **Correction:** the fourth method is **reinforcement fine-tuning (RFT)**, not RLHF. Classical
> RLHF trains a separate **reward model** on human preference pairs and runs PPO against it with a
> KL anchor; RFT takes a **grader** — a Python function, a test harness, or an LLM judge — and
> optimises against its score directly. No reward model, no preference data, no PPO. Two practical
> consequences: RFT needs a task with an **automatically checkable** answer (so it is useless for
> "make the tone friendlier"), and it is billed **per hour**, so **your grader's latency is a line
> item** (CS-18 §11.6). Calling hosted RFT "RLHF" in an interview is an immediate tell.

### 9.3 Model tiering

| | `gpt-4.1-nano` | `gpt-4.1-mini` | `gpt-4.1` |
|---|---|---|---|
| Training (per 1M) | **$1.50** | **~$3.00** | **~$25.00** |
| Markup over its own base | 2× | 2× | 2× |
| Best for | Classification, routing, extraction, high volume | The default | Rarely worth a fine-tune |
| Risk of wasting money | Low | Low | **High** |

> **Fine-tune the smallest model that can do the task when prompted. Prompt the largest model you
> can afford when you cannot.** The gap between them is where money is wasted: a nano fine-tune at
> 94% costs ~2.5× less to train and ~4× less to serve than a mini fine-tune at 96%, and those two
> points only matter if you have a business justification for them.

### 9.4 Hosted OpenAI vs hosted Vertex — the two shapes of "managed"

| Dimension | **OpenAI** (this card) | **Vertex / Gemini** (CH-19) |
|---|---|---|
| Weights returned | ❌ | ❌ |
| **Idle cost** | **$0 — no idle charge** | **$3–5/hour while deployed** |
| The dominant recurring term | **per-token markup (~2×)** | **the hourly endpoint charge** |
| Schema | `messages`, role **`assistant`** | `messages`, role **`model`** |
| System message position | Any (first is conventional) | **First message only** |
| Hyperparameters | `n_epochs`, `batch_size`, `learning_rate_multiplier` | `epochs`, `learning_rate_multiplier`, `adapter_size` (the LoRA rank) |
| Billing unit for SFT | per 1M training tokens | per token, plus hourly while deployed |
| Setup prerequisite | An API key | A GCP project + bucket + IAM |
| Status | **Winding down (§Status warning)** | Available |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| API minimum examples | **10** | A guard rail, not a target |
| Practical minimum | **50–100** | Below this you ship a template |
| Per-message overhead | **3 tokens** | Plus **3** for reply priming |
| Measured overhead, short examples | **+19.6%** | 562 → 672 (CS-18 §4.5.2) |
| Auto epochs, N < 100 | **`min(25, 100 // N)`** | 10 rows → **10 epochs** |
| Auto epochs, N in [34, 8333] | **3** | The flat regime |
| Auto epochs, N > 8333 | **`max(1, 25000 // N)`** | 200,000 rows → 1 epoch |
| Auto `batch_size` | **~0.2% of rows, ≤256, ≥1** | 10 rows → 1 |
| Inference markup | **2×** input and output | The bill |
| Training share of a 12-month bill | **0.01–2%** | Inference is 95–99.9% |
| Break-even | **`P_f < P_b/2 − 2O`** | Assumes 2× markup and 4:1 output:input |
| Self-host crossover (3B LoRA, 1×24 GB) | **~230k–460k calls/month** | Below it, hosted wins |
| Hours per month | **730** | The multiplier for any hourly rate (CH-19 §10) |
| Tokens per word | **≈1.33** | English prose, ±15% |
| Chars per token | **≈4** | English |
| Per-example cap in the reference impl. | **16,385** | **Stale** for the `gpt-4.1` line — verify (§10) |
| Job status machine | `validating_files` → `queued` → `running` → `succeeded`/`failed`/`cancelled` | Poll to a terminal state |
| `identical_outputs` no-op threshold | **> 0.90** | The fine-tune did nothing |
| Suffix length cap | **≤18 chars** | Over-long suffixes are silently truncated |
| `metadata` pairs | **≤16** | `data_version`, `git_sha`, `owner` |
| Validation split | **10–20%** | Plus a frozen test set the job cannot read |

---

## 11. Common Errors And Their Exact Messages

> This table mixes **verbatim strings** (marked ▣) with **symptom descriptions** (unmarked). The
> verbatim ones are reproduced from CS-18 or from the handbook script's own output. **The API's
> error text churns; treat anything unmarked as "what you will see", not "what the SDK will print".**

| Error / symptom | Meaning | Fix |
|---|---|---|
| ▣ `KeyError: 'data_type'` | Dict key read before assignment — the validator bug the video hits live | `format_errors = collections.defaultdict(int)` |
| ▣ `NotFoundError: Error code: 404 - {'error': {'message': 'The model \`ft:...\` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'code': 'model_not_found'}}` | The job never reached `succeeded`; a cancelled job mints no model | Poll `jobs.retrieve()` for a terminal status. **It is never a credentials problem and retrying never helps.** |
| ▣ `data_type` | A JSONL line is not a JSON object | Fix the file |
| ▣ `missing_messages_list` | The `messages` key is absent or empty | Fix the file |
| ▣ `message_missing_key` | A message lacks `role` or `content` | Fix the message |
| ▣ `message_unrecognized_key` | A key is outside the allow-list | Widen a **stale** allow-list before assuming the data is broken |
| ▣ `unrecognized_role` | Role not in the allow-list | `assistant`/`tool`, not `human`/`gpt`/`instruction` |
| ▣ `missing_content` | Empty/falsy content and no tool call | Allow `content: null` when `tool_calls` is present |
| ▣ `example_missing_assistant_message` | No supervised turn in the row | Add the assistant turn |
| ▣ `too_few_examples` / `too_many_examples` | Below the API minimum or above the file maximum | The script's own keys — 10 / 50,000 |
| ▣ `malformed_shape` | The row is not one of the accepted shapes | `{"messages":[...]}`, `instruction`/`output`, or `prompt`/`completion`. **Note the script prints the row's keys, which reads as "the keys are wrong" when the real problem is the value's type** |
| ▣ `no_assistant_message` / `last_message_not_assistant` | The row has no completion to learn from | Reorder or drop the row |
| ▣ `unknown_role` | Role outside `system`/`user`/`assistant`/`tool` | The script rejects `developer`; a modern file may legitimately use it |
| ▣ `empty_content` / `very_short_completion` / `duplicate_examples` | Whitespace-only content, a completion under 3 words, near-duplicate rows | These are the script's quality *warnings* — fix the data, not the validator |
| Job fails in `validating_files` | One bad line fails the whole file | The local validator is a *convenience*, not a safety net — the server would have caught it too; run it locally because round trips are slow |
| `messages` is a string/dict/int | The type hole the cookbook validator leaves open | Guard with `isinstance(messages, list)` before iterating |
| `trained_tokens` well below your sum | Silent truncation of over-long rows | Re-budget with the correct per-example cap; pre-truncate yourself |
| Job `succeeded`, model is a no-op | LR multiplier too low, rows truncated, or the wrong file | `identical_outputs` (§5.4) |
| Fine-tuned model behaves exactly like the base | Same as above | Compare 50 completions; if byte-identical, nothing happened |

---

## 12. Copy-Paste Starter Config

There is no YAML — hosted fine-tuning is configured by arguments. The equivalent:

```bash
# ── The whole run, in order. Run each line deliberately. ─────────────────────────
export OPENAI_API_KEY="sk-..."                     # never inline it
export BASE_MODEL=gpt-4.1-mini-2025-04-14          # a dated SNAPSHOT, not the alias
export SUFFIX=support-v1                           # <=18 chars
export TRAIN=code/data/sample_sft.jsonl

# 0. Re-check the platform's fine-tuning status page and deprecation list FIRST.
#    The Status warning at the top of this file is a prompt to go and look.

# 1. Validate locally — costs nothing, catches everything structural
python code/13_openai_finetune.py --data $TRAIN --validate

# 2. Cost model — READ IT, then correct it by hand (§7.3)
python code/13_openai_finetune.py --data $TRAIN --estimate \
    --model $BASE_MODEL --epochs 3 --inference-volume 200000

# 3. Break-even, on the back of an envelope. If this is FALSE, write down the
#    QUALITY reason you are proceeding anyway — 'cheaper' is not available to you.
#       P_f < P_b/2 - 2O   ?   e.g.  180 < 900/2 - 2*120 = 210  -> TRUE

# 4. Split. Dedupe ACROSS the split before you write the files.
#    train.jsonl 70-80% | val.jsonl 10-15% | test.jsonl 10-15% (frozen)

# 5. Upload
python code/13_openai_finetune.py --data $TRAIN --upload --suffix $SUFFIX

# 6. Train — and read the resolved hyperparameters back off the job
python code/13_openai_finetune.py --data $TRAIN --train --model $BASE_MODEL \
    --suffix $SUFFIX --epochs 3

# 7. Evaluate BEFORE you ship: identical_outputs, structural pass rate,
#    task accuracy, out-of-scope refusal rate — base vs fine-tuned, same prompts.
```

`model_registry.py` — the one artefact that survives a platform migration:

```python
# The model ID is a versioned CONFIG VALUE, not a code constant.
REGISTRY = {
    "support-v1": {"id": "ft:gpt-4.1-mini-2025-04-14:org:support-v1:Aa11",
                   "base": "gpt-4.1-mini-2025-04-14", "prompt_sha": "3f9a1c2e",
                   "data_version": "2026-09-01", "job_id": "ftjob-XXXX", "trained_tokens": 11_067,
                   "eval": {"structural": 1.00, "task_acc": 0.94, "identical_outputs": 0.18}},
}
ACTIVE = os.environ.get("ACTIVE_MODEL", "support-v1")     # env-var rollback, no deploy
```

### The five checks before you commit

| # | Check | Pass condition |
|---|---|---|
| 1 | `--validate` is clean, and `unique_convs / rows > 0.95` | No structural problems, no redundancy |
| 2 | Overhead-corrected token count and `--estimate` both read | You know the training bill **and** the monthly one |
| 3 | `P_f < P_b/2 − 2O` computed, or a written non-cost justification | You are not shipping a cost story that is false |
| 4 | `validation_file`, `seed` and `metadata` are all set | The run is reproducible and the holdout is real |
| 5 | The base model's baseline numbers exist, on the same prompts | You can tell success from `succeeded` |

> **Check 5 is the one that cannot be done after the fact**, and it is the one the source video
> skips entirely: a fine-tuning run without a paired baseline is **a purchase, not an experiment.**

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full treated case study, with every timestamped claim audited | **CS-18 — OpenAI GPT Fine-Tuning Masterclass** (especially §4.2, §4.5, §4.6, §11, §16.8) |
| The other hosted platform, and why its cost shape is the opposite | **CH-19 / CS-19 — Gemini / Vertex AI Fine-Tuning** |
| To train the same data yourself, on your own GPU, with the weights | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To know when fine-tuning is the wrong tool at all | **CH-04 / CS-04 — Fine-Tuning vs RAG vs Agents** |
| To recover an artefact from an endpoint-only model | **CH-09 / CS-09 — Distillation: LLM → SLM** |
| To train on preferences instead of demonstrations | **CH-14 — The Alignment Map**; CS-25 for the DPO loss |
| To make the open-weight path cheap enough to be the migration target | **CH-10 — Quantization**; CS-23 — LoRA & QLoRA |
| To run the numbers on the handbook's fixture | `code/13_openai_finetune.py` (`--validate`, `--estimate`) |
| To practise being interviewed on this | **IQ-18 — Interview Questions: OpenAI GPT Fine-Tuning** |
