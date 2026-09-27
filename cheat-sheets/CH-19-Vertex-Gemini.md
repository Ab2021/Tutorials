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
| `source_model` | `gemini-2.5-flash` | the base model id | **Verify current tuned-model support** — not every Gemini model can be tuned, and the 1.5 generation is superseded. Pro-tier models are usually *not* tunable |
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
    source_model="gemini-2.5-flash",
    train_dataset="gs://my-bucket/vertex/train.jsonl",
    tuned_model_display_name="handbook-sft",
    epochs=3,
    learning_rate_multiplier=1.0,
    adapter_size=4,
)
print(job.resource_name)
# job.state  -> JOB_STATE_RUNNING / SUCCEEDED / FAILED
```

> **Two SDK surfaces, and mixing them is a common 404.** The snippet above is the
> **Vertex AI SDK** (`vertexai.tuning.sft.train`, snake_case args, `job.resource_name`).
> The video and CS-19 §6.7 use the **GenAI SDK** (`client.tunings.tune(base_model=...,
> training_dataset=TuningDataset(gcs_uri=...), config=CreateTuningJobConfig(...))`,
> `tuned_model.endpoint`). They create the same kind of `TuningJob` but expose different
> fields — in particular teardown, which the GenAI SDK's `tunings` module has not
> historically exposed at all (CS-19 §10.3 Correction 1). Pick one and use it for the whole
> lifecycle, including the undeploy.

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

---

## 6. CLI Commands

```bash
# ── Auth & project ──────────────────────────────────────────────────────────
gcloud auth login                                    # your user
gcloud auth application-default login                # what the Python SDK uses
gcloud config set project my-proj
gcloud services enable aiplatform.googleapis.com

# ── The GCS prerequisite ────────────────────────────────────────────────────
gsutil mb -l us-central1 gs://my-bucket
gsutil cp code/data/sft.vertex.jsonl gs://my-bucket/vertex/
gsutil ls gs://my-bucket/vertex/
gsutil iam ch serviceAccount:service-NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com:objectViewer gs://my-bucket
# Or via gcloud:
gcloud projects add-iam-policy-binding my-proj \
  --member="serviceAccount:service-NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# ── The handbook's script (does validate / estimate / upload / train / list / undeploy) ──
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --validate
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --estimate
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --upload --project P --bucket B
python code/14_vertex_gemini_finetune.py --train --project P --bucket B --adapter-size 4
python code/14_vertex_gemini_finetune.py --list --project P
python code/14_vertex_gemini_finetune.py --undeploy ENDPOINT_ID --project P

# ── gcloud equivalents ──────────────────────────────────────────────────────
gcloud ai tuning-jobs list --region=us-central1
gcloud ai endpoints list --region=us-central1        # WHAT IS BILLING RIGHT NOW
gcloud ai models list --region=us-central1           # stored tuned models (cheap)

# ── Install ─────────────────────────────────────────────────────────────────
pip install google-cloud-aiplatform
pip install "google-cloud-aiplatform[tensorboard]"   # if you want the metrics
```

> `gcloud ai endpoints list` is the single most useful command in this file. Every endpoint it
> prints is accruing an hourly charge at that moment — whether or not it has ever served a
> request.

---

## 7. VRAM / Cost Calculator

### 7.1 The verified worked example

Using `code/14_vertex_gemini_finetune.py --estimate` on the 60-row sample SFT fixture, with
`gemini-2.5-flash` prices as recorded in the script ($5.00/1M train tokens, $0.30/1M input,
$2.50/1M output, **$2.00/hour** endpoint — CS-19 §4.6's rate) at 100,000 calls/month:

| Line item | Amount |
|---|---|
| Tuning (9,447 tokens × 3 epochs) | **$0.05** |
| Endpoint, 1 month ($2.00 × 730 h) | **$1,460.00** |
| Per-token usage, 100k calls | **$63.04** |
| **Month 1 total** | **$1,523.08** |
| **Year 1 total** | **$18,276.50** |
| Fixed endpoint share of recurring bill | **96%** |
| Fixed ÷ usage ratio | **23×** |
| Calls/month for usage to catch up | **~2,316,000** |

> **Read that table again.** Training is **five cents**. The idle endpoint is **$18,277 in
> year one**. The thing people agonise over — the tuning — is a rounding error; the thing
> people forget — the deployment — is the entire bill.
>
> The ~2.3M-calls/month crossover is a *usage* figure, not a business-plan figure: it is
> where per-token spend equals one always-on endpoint, so an app serving 2.3M calls/month
> **and no more** is still better off undeploying between sessions. For everyone below it,
> the endpoint is a fixed cost you must actively manage.
>
> **Do not compare this table to CS-19 §11.1 or §4.6 without checking which generation each
> one prices.** CS-19 §11.2 uses the video's 2.5-generation rates; this table does too, so
> they now agree — CS-19's $1,459/month is $2.00/hour × 729.6 h, this table's $1,460 is the
> same thing at 730 h. Earlier revisions of this sheet printed the 1.5-generation rates
> alongside CS-19's 2.5-generation rates, which made the training rate disagree by 1.7×
> ($3.00 vs $5.00 for a Flash-tier model) with no note saying which was current.

### 7.2 Prices on file in the script (VERIFY BEFORE BUDGETING)

| Model | Train /1M | Input /1M | Output /1M | Endpoint /hour |
|---|---|---|---|---|
| `gemini-2.5-flash` | $5.00 | $0.30 | $2.50 | $2.00 |
| `gemini-2.5-flash-lite` | $1.50 | $0.10 | $0.40 | $2.00 |
| `gemini-2.5-pro` | **not on file** — see below | $1.25 | $10.00 | $2.00 |
| `gemini-2.0-flash` | $3.00 | $0.10 | $0.40 | $3.00 |
| `gemini-1.5-flash` | $3.00 | $0.075 | $0.30 | $3.00 |
| `gemini-1.5-pro` | $8.00 | $1.25 | $5.00 | $5.00 |

> **Why `gemini-2.5-pro` has no training rate.** The video quotes $25/1M for it [22:29], but
> the Pro tier of the 2.5 and 2.0 generations has generally **not** been offered as a
> supervised-tuning base model — tuning has concentrated on Flash and Flash-Lite (CS-19 §4.1
> Correction 2). The script therefore omits the key and `--estimate` prints `≥ $X` for its
> totals rather than folding an invented number into a total and presenting it as fact. A
> 5×-wrong training rate is worse than a visible gap.
>
> The 1.5-generation rows are retained only because an existing job may still reference them
> by id; 1.5 Flash was deprecated as a tunable base in **May 2025**. `DEFAULT_MODEL` is
> `gemini-2.5-flash`, and `_resolve_price_key` matches the **longest** model id on a token
> boundary, so `gemini-2.5-flash-lite-001` resolves to the Flash-Lite rate rather than
> silently taking Flash's (3.3× more expensive) training rate.

> ⚠️ **These are the values the script ships with, not a live price list.** Google changes
> prices, model availability and tuned-model support regularly, and the 1.5-generation models
> are superseded. Treat every number here as a placeholder to be checked against the current
> pricing page before you commit a budget. The *structure* of the arithmetic — the hourly
> endpoint charge dominating — is the durable lesson.

### 7.3 Fixed vs usage, by volume

| Calls/month | Per-token | Fixed endpoint | Fixed share | Verdict |
|---|---|---|---|---|
| 1,000 | ~$0.08 | $2,190 | ~100% | Absurd — undeploy between uses |
| 100,000 | $7.63 | $2,190 | ~100% | Still absurd |
| 1,000,000 | ~$76 | $2,190 | 97% | Endpoint dominates |
| 10,000,000 | ~$763 | $2,190 | 74% | Getting reasonable |
| 28,686,000 | ~$2,190 | $2,190 | 50% | The crossover |
| 100,000,000 | ~$7,630 | $2,190 | 22% | Now self-hosting is worth pricing |

### 7.4 What a self-hosted open model would cost instead

| Item | Vertex tuned endpoint | Self-hosted open model |
|---|---|---|
| Idle cost | **$2,190/month** | $0 if you stop the instance |
| Weights | You do not get them | Yours |
| Control over quantisation | None | Full |
| Control over serving stack | None | Full (vLLM, llama.cpp) |
| Ops burden | None | Real |
| Minimum viable scale | Any | ~1 GPU-month of commitment |

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| **Surprise bill** | A deployed endpoint left running | `gcloud ai endpoints list`, then `--undeploy` |
| `Permission denied` on the dataset, minutes into the job | The Vertex service account lacks `objectViewer` on the bucket | §5.2; check you used `service-NUMBER`, not `PROJECT_ID` |
| Job fails immediately with a schema error | One bad row; Vertex fails the **whole job** | Run `--validate` first |
| `Invalid role` / unexpected role error | `assistant` instead of `model` | Normalise on upload (the script does) |
| Job rejects mid-conversation `system` turn | `system` allowed only as the first message | Restructure the conversation |
| `Too few examples` | Below Vertex's minimum | Test with realistic data volumes |
| Model id rejected / not tunable | That Gemini generation does not support tuning | Check current tuned-model support |
| Tuned model exists but calls 404 | Not **deployed** to an endpoint | Tuning and deployment are separate steps |
| Endpoint deployed but slow first response | Cold start | Batch or keep warm — both cost money |
| Cannot download the tuned weights | By design | Vertex never gives you weights; use open-weight training |
| `google.auth.exceptions.DefaultCredentialsError` | No ADC | `gcloud auth application-default login` |
| `403 ... aiplatform.googleapis.com has not been used` | API not enabled | `gcloud services enable aiplatform.googleapis.com` |
| Cost estimate differs wildly from the invoice | Stale prices in `PRICES` | Update the table; verify against current pricing |
| Billing continues after deleting the *model* | The **endpoint** is what bills | Undeploy the endpoint, then delete the model |

> **The last row is the classic confusion.** On Vertex there are two objects — a tuned
> **model** and an **endpoint**. Storing the model is cheap. The endpoint is what charges by
> the hour. Deleting the model without undeploying the endpoint leaves you paying for a
> deployment with nothing behind it.

---

## 9. Comparison Matrix

| Dimension | Vertex AI (Gemini) | OpenAI fine-tuning | Self-hosted open weights |
|---|---|---|---|
| Weights returned | ❌ | ❌ | ✅ |
| Idle cost | **$3–5/hour** | **$0** | $0 if you stop the instance |
| Data location | Your GCS bucket | OpenAI's cloud | Your own |
| Setup prerequisite | **GCP project + bucket + IAM** | An API key | A GPU + a stack |
| Schema | `messages` with role `model` | `messages` with role `assistant` | Whatever you write |
| Minimum data | Enforced minimum | Enforced minimum | None |
| Merge / quantise / export | ❌ | ❌ | ✅ |
| Distil from the tuned model | ⚠️ via API only | ⚠️ via API only | ✅ fully |
| Serving control | ❌ | ❌ | ✅ |
| Ops burden | None | None | Real |
| Best when | Data is in GCP; compliance needs it | Simple path, no idle cost | You need control or scale |

### 9.1 Choosing

| Situation | Pick |
|---|---|
| Data already in BigQuery/GCS, compliance demands GCP | **Vertex** |
| Bursty, low-volume use | **OpenAI** (no idle charge) or Vertex with scripted undeploy |
| Very high steady volume (>28M calls/month) | Price self-hosting an open model |
| You need the weights | **Self-hosted open weights** |
| You need a tuned model today with no ops | Either managed service |

---

## 10. Numbers To Memorize

| Number | Value | Context |
|---|---|---|
| Hours per month | **730** | The multiplier for any hourly rate |
| Idle endpoint, flash | **$3.00/hour** | → **$2,190/month** |
| Idle endpoint, pro | **$5.00/hour** | → **$3,650/month** |
| Crossover vs usage (worked example) | **~28.7M calls/month** | When usage equals the idle charge |
| Fixed ÷ usage at 100k calls | **287×** | The endpoint is everything |
| Tuning cost (worked example) | **$0.03** | Training is not the cost |
| Year-1 total (worked example) | **$26,371.64** | Almost entirely the endpoint |
| `adapter_size` options | **1, 4, 8, 16, 32** | Default 4 |
| Default epochs | **3** | |
| Roles | `system`, `user`, **`model`** | Not `assistant` |
| System message position | **first only** | |
| Tokens per word | **≈1.33** | |
| Default location | `us-central1` | Keep the bucket in the same region |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials` | No ADC | `gcloud auth application-default login` |
| `403 Permission 'aiplatform.tuningJobs.create' denied` | Missing `roles/aiplatform.user` | Grant the role to your principal |
| `403 ... does not have storage.objects.get access` | Service account lacks bucket read | §5.2 — use `service-NUMBER` |
| `400 Invalid JSON at line N` | Malformed row | Validate locally first |
| `400 ... role must be one of ['system','user','model']` | `assistant` used | Normalise to `model` |
| `400 ... system message must be the first` | Mid-conversation system turn | Restructure |
| `400 The training dataset must contain at least N examples` | Below the minimum | Use realistic volumes |
| `400 Model ... does not support tuning` | That generation is not tunable | Check current support |
| `404 ... endpoint not found` | Not deployed, or wrong region | Deploy; check `--location` |
| `429 Resource exhausted` | Quota | Request a quota increase |
| `FAILED_PRECONDITION: The service account ... does not exist` | Wrong project number | Re-derive it |
| `InvalidArgument: adapter_size must be one of ...` | Unsupported size | Use 1/4/8/16/32 |
| Silent: job succeeds, model is bad | Bad data, or too few examples | Validate + inspect the data (CH-13) |
| Silent: bill keeps growing | Endpoint still deployed | `gcloud ai endpoints list` |

---

## 12. Copy-Paste Starter Config

There is no YAML — Vertex is configured by arguments. The equivalent:

```bash
# ── The whole run, in order. Run each line deliberately. ────────────────────
export PROJECT=my-proj
export BUCKET=my-bucket
export LOCATION=us-central1

# 0. One-time setup
gcloud config set project $PROJECT
gcloud services enable aiplatform.googleapis.com
gsutil mb -l $LOCATION gs://$BUCKET || true

# 1. Validate locally — costs nothing, catches everything
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --validate

# 2. Cost model — READ IT before spending
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --estimate \
    --inference-volume 100000

# 3. Upload
python code/14_vertex_gemini_finetune.py --data data/sft.jsonl --upload \
    --project $PROJECT --bucket $BUCKET

# 4. Tune
python code/14_vertex_gemini_finetune.py --train \
    --project $PROJECT --bucket $BUCKET \
    --model gemini-2.5-flash --epochs 3 --lr-multiplier 1.0 --adapter-size 4

# 5. Deploy (starts the meter) → use → 6. UNDEPLOY (stops it)
python code/14_vertex_gemini_finetune.py --list --project $PROJECT
# ... then:
# python code/14_vertex_gemini_finetune.py --undeploy ENDPOINT_ID --project $PROJECT
```

### The five checks before you commit

| # | Check | Pass condition |
|---|---|---|
| 1 | `--validate` is clean | No schema problems |
| 2 | `--estimate` has been read | You know the monthly fixed charge |
| 3 | Prices in the script are current | Verified against the live pricing page |
| 4 | You have a written plan to undeploy | A reminder, a cron job, a checklist — something |
| 5 | You compared against self-hosting | You have seen the alternative's number |

> Check 4 is not a joke. The most common way this module produces a surprise invoice is a
> deployed endpoint someone forgot about for three months — $6,570 at the flash rate.

---

## 13. What To Read Next

| If you want to… | Read |
|---|---|
| The full treated case study | **CS-19 — Gemini / Vertex AI Fine-Tuning** |
| The OpenAI equivalent, and why it differs | **CH-18 / CS-18 — OpenAI GPT Fine-Tuning** |
| To decide managed vs self-hosted | **CH-03 / CS-03 — Framework Landscape** |
| To understand what SFT is doing | **CH-13 / CS-13 — Instruction Fine-Tuning** |
| To compare against distillation from a big model | **CH-09 / CS-09 — Distillation: LLM → SLM** |
| To serve your own model instead | `code/15_serve_vllm.py` |
| Practice being interviewed on this | **IQ-19 — Interview Questions: Gemini / Vertex AI** |

