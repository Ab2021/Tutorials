# CH-19 — Gemini / Vertex AI Fine-Tuning Cheat Sheet

**One-line purpose:** supervised fine-tune a Gemini model as a managed service on Google
Cloud — no GPU, no weights, no merge step.
**Use when:** you want a tuned model without owning the training stack, your data is already
in GCP, or you need Google's serving infrastructure and compliance posture.
**Do NOT use when:** you need the weights (you never get them), you need to control the
serving stack or quantisation, or your request volume is high enough that a deployed
endpoint's **hourly** charge dominates. That last one is the trap this whole card is about.

> **The one sentence that matters.** On Vertex you pay for a tuned model **per hour while it
> is deployed**, whether or not anyone calls it. Everything else on this card is secondary to
> remembering to `undeploy`.

---

## 1. The 10-Second Summary

| # | Fact | Why |
|---|---|---|
| 1 | **A deployed endpoint bills per HOUR, idle or not.** | The single biggest cost surprise on Vertex. |
| 2 | **You must `undeploy` when done.** | Storing the model is cheap; *serving* it is not. |
| 3 | **The dataset must be in GCS.** | Vertex cannot read a local file. Bucket + IAM first. |
| 4 | The model turn uses role **`model`**, not `assistant`. | Vertex validates roles strictly and fails the whole job. |
| 5 | **`system` is allowed only as the FIRST message.** | A mid-conversation system turn fails the job. |
| 6 | **`adapter_size` is the LoRA rank.** | 1 is cheapest/most constrained, 32 the most capacity. |
| 7 | You get a **tuned model resource**, then must **deploy** it to an **endpoint**. | Two separate objects; only the endpoint bills hourly. |
| 8 | `adapter_size` also drives **artifact size and cost**. | Bigger adapter = more to store and serve. |
| 9 | Prices and model ids **change often**. | Verify before you budget. Every figure here is a placeholder until you check. |
| 10 | Compare against **self-hosting** before committing. | An open model has no idle-hour charge. |

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Tuning cost** | `tokens × epochs / 1e6 × $per_1M_train` | — | 2M tokens × 3 epochs = 6M → $18 at $3/1M |
| **Idle endpoint (monthly)** | `hourly × 730` | 730 h ≈ 1 month | $3/h → **$2,190/month** |
| **Per-token inference** | `in/1e6 × $in + out/1e6 × $out` | — | 100k calls → compute it, don't guess |
| **Crossover vs usage** | `calls where per_token = monthly_fixed` | — | Compare the **dollars**, not a call count |
| **Token estimate** | `words × 1.33` | English | The script's estimator |
| **Year-1 total** | `train + 12 × (fixed + per_token)` | — | |

### 2.1 The cost shape, in one picture

```
                    ONE-OFF            RECURRING
                 ┌──────────┐    ┌─────────────────────────┐
Vertex           │  tuning  │    │ endpoint $/HOUR  (idle!) │  ← the trap
                 │  cost    │    │ + per-token usage        │
                 └──────────┘    └─────────────────────────┘

                 ┌──────────┐    ┌─────────────────────────┐
OpenAI           │  tuning  │    │ per-token usage ONLY     │  ← no idle charge
                 │  cost    │    │                          │
                 └──────────┘    └─────────────────────────┘

                 ┌──────────┐    ┌─────────────────────────┐
Self-hosted      │  GPU-hrs │    │ GPU-hrs (whether idle or │  ← your own idle cost,
                 │  to train│    │ not — but you control it)│     and you can stop it
                 └──────────┘    └─────────────────────────┘
```

> The decisive difference from OpenAI is **not** the per-token price. It is that Vertex
> charges for the endpoint by the hour, so a low-traffic tuned model is dominated by
> fixed cost, while an OpenAI tuned model is not.

---

## 3. Decision Tree

```
Is your data already in GCP?
├─ No  → add a GCS bucket + IAM work to your estimate. It is real setup cost.
└─ Yes ↓

Do you need the model WEIGHTS (to merge, quantise, self-host, or distil from)?
├─ Yes → Vertex is the wrong tool. You never get weights. Use open-weight training.
└─ No ↓

Is your request volume HIGH and STEADY (roughly >1M calls/month)?
├─ Yes → a deployed endpoint's hourly charge is amortised.
│         Still compare against self-hosting an open model — you may win on price
│         AND on control.
└─ No (low or bursty volume) ↓

    ┌─ Bursty (batch jobs, occasional use)?
    │   → DEPLOY, use, UNDEPLOY. Script it so you cannot forget.
    └─ Steady but low?
        → The idle charge will dominate. Reconsider: an un-tuned Gemini call,
          or an OpenAI tuned model with no idle charge, may be cheaper.

Do you need to control the serving stack (quantisation, batching, custom kernels)?
├─ Yes → Vertex is the wrong tool.
└─ No  → Vertex is a reasonable choice.

Are you doing preference tuning, or SFT?
├─ SFT (supervised)  → supported on Vertex (this card)
└─ Preference (RLHF/DPO) → check current support; availability changes

Do you need the system instruction to change mid-conversation?
├─ Yes → restructure. Vertex allows `system` only as the first message.
└─ No  → proceed

Is your JSONL schema validated?
├─ No  → validate first. Vertex fails the ENTIRE job on one bad row.
└─ Yes → upload to GCS, tune, deploy, UNDEPLOY.
```

---

## 4. Hyperparameter Quick Reference

| Param | Script default | Typical | Effect |
|---|---|---|---|
| `source_model` | `gemini-1.5-flash-002` | the base model id | **Verify current tuned-model support** — not every Gemini model can be tuned |
| `epochs` | 3 | 1–5 | More epochs = more overfit; Vertex's default is often 3 |
| `learning_rate_multiplier` | 1.0 | 0.5–2.0 | Vertex tunes at a fixed base LR; this scales it. 1.0 = their default |
| `adapter_size` | 4 | 1, 4, 8, 16, 32 | **The LoRA rank.** Bigger = more capacity, more artifact, more cost |
| `tuned_model_display_name` | `handbook-sft` | descriptive | Your only handle on the job afterwards |

### 4.1 Choosing `adapter_size`

| Size | Use for | Risk |
|---|---|---|
| 1 | Tiny style tweaks; near-minimum change | May not have capacity to learn the task |
| **4** | **The safe default** | Start here |
| 8 | A clear behaviour change with a few thousand examples | — |
| 16 | A substantial change, larger dataset | Overfits small data |
| 32 | Near-full-capacity adaptation | Overfits unless the dataset is large |

> `adapter_size` is the knob people ignore. It is the LoRA rank of the tuning adapter: it
> caps *how much the model can change*, and it determines the artifact you deploy. For
> format or style adaptation, 4–8 is usually plenty; 32 on 500 examples will memorise them.

### 4.2 The dataset schema

```jsonl
{"messages": [{"role": "system",  "content": "You are a support agent."},
              {"role": "user",    "content": "Refund policy for digital goods?"},
              {"role": "model",   "content": "Digital goods are refundable within 14 days..."}]}
{"messages": [{"role": "user",    "content": "..."},
              {"role": "model",   "content": "..."}]}
```

| Rule | Detail |
|---|---|
| Key | `messages`, a list |
| Roles | `system` (first message only), `user`, `model` |
| `model`, not `assistant` | The script normalises it on upload — confirm the file if the job rejects it |
| Last turn | Must be a `model` turn |
| One bad row | **Fails the whole job** |

---

## 5. Copy-Paste Code Snippets

### 5.1 The full lifecycle, via the handbook's script

```bash
# 1. Validate the schema locally (no GCP needed)
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --validate

# 2. Cost model — see the idle-endpoint arithmetic before you spend anything
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --estimate

# 3. Normalise + upload to GCS
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --upload \
    --project my-proj --bucket my-bucket

# 4. Start the tuning job
python code/14_vertex_gemini_finetune.py --train --project my-proj --bucket my-bucket

# 5. See what exists — and what is BILLING RIGHT NOW
python code/14_vertex_gemini_finetune.py --list --project my-proj

# 6. Stop the hourly charge
python code/14_vertex_gemini_finetune.py --undeploy ENDPOINT_ID --project my-proj
```

> Step 5 is the one to run reflexively. The script's endpoint listing prints
> *"Any endpoint listed above is billing per hour right now, whether or not it is receiving
> traffic."* If you have several, that is several hourly charges.

### 5.2 The GCS prerequisite, exactly

```bash
# Create a bucket in a region near your Vertex location
gsutil mb -l us-central1 gs://my-bucket

# Grant the Vertex service account read access to it
gcloud projects add-iam-policy-binding my-proj \
  --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# The user running the job also needs Vertex + storage permissions
gcloud projects add-iam-policy-binding my-proj \
  --member="user:you@example.com" --role="roles/aiplatform.user"

# Confirm
gcloud auth application-default login
gcloud config set project my-proj
```

> **`service-PROJECT_NUMBER`, not `PROJECT_ID`.** A very common copy-paste error, and the
> symptom is a job that starts and then fails with a permissions error minutes later.

### 5.3 Normalising your data to Vertex's shape

```python
def to_vertex(row):
    """Normalise any reasonable on-disk shape to Vertex's messages list."""
    msgs = None
    if isinstance(row.get("messages"), list):
        msgs = row["messages"]
    elif "instruction" in row and "output" in row:            # alpaca
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        ins = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        msgs += [{"role": "user",  "content": ins},
                 {"role": "model", "content": row["output"]}]
    elif "prompt" in row and "completion" in row:             # openai-completion style
        msgs = [{"role": "user",  "content": row["prompt"]},
                {"role": "model", "content": row["completion"]}]
    if msgs is None:
        return None
    # 'assistant' -> 'model'. Vertex's tuned-model format wants 'model'.
    return [{"role": ("model" if m.get("role") == "assistant" else m.get("role")),
             "content": str(m.get("content", ""))} for m in msgs]
```

### 5.4 Validating before you spend

```python
def validate(rows):
    problems = {}
    def flag(k): problems[k] = problems.get(k, 0) + 1

    for i, r in enumerate(rows):
        msgs = to_vertex(r)
        if not msgs:
            flag("not_convertible"); continue
        roles = [m["role"] for m in msgs]

        if msgs[0]["role"] == "system" and "system" in roles[1:]:
            flag("system_message_not_first")
        if "model" not in roles:
            flag("no_model_turn")
        elif roles[-1] != "model":
            flag("last_turn_is_not_the_model")
        for m in msgs:
            if m["role"] not in ("system", "user", "model"):
                flag(f"bad_role:{m['role']}")
            if not m["content"].strip():
                flag("empty_content")

    if len(rows) < 10:
        flag("too_few_examples_for_a_tuning_job")     # Vertex enforces a minimum
    return problems
```

> The last check matters: Vertex enforces a **minimum example count** for a tuning job, and
> a 6-row smoke test will be rejected at the API. Test the pipeline with real-sized data.

### 5.5 The Python SDK, directly

```python
import vertexai
from vertexai.tuning import sft

vertexai.init(project="my-proj", location="us-central1")

job = sft.train(
    source_model="gemini-1.5-flash-002",
    train_dataset="gs://my-bucket/vertex/train.jsonl",
    tuned_model_display_name="handbook-sft",
    epochs=3,
    learning_rate_multiplier=1.0,
    adapter_size=4,
)
print(job.resource_name)
# job.state  -> JOB_STATE_RUNNING / SUCCEEDED / FAILED
```

```python
# Deploy the tuned model to an endpoint — THIS is what starts the hourly billing
from vertexai.preview import tuning
tuned = tuning.TunedModel(model=..., endpoint=...)
tuned.deploy()

# Call it
from vertexai.generative_models import GenerativeModel
m = GenerativeModel("projects/P/locations/L/endpoints/E")
print(m.generate_content("Test prompt").text)

# STOP THE BILLING
import vertexai
from vertexai.preview import tuning
tuning.TunedModel(...).undeploy()
```

<!-- CONTINUE -->
