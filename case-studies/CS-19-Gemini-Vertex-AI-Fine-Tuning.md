# CS-19 — Google Gemini Fine-Tuning on Vertex AI (Managed SFT)

| Field | Value |
|---|---|
| **Module** | Managed / API-only fine-tuning (closed-weight track) |
| **Source video(s)** | LLM Fine-Tuning 21: Google Gemini Fine-Tuning Masterclass using Vertex AI \| Supervised Finetuning |
| **Transcript file(s)** | `LLM_Fine-Tuning_21_Google_Gemini_Fine-Tuning_Masterclass_using_Vertex_AI_Supervi.txt` (1539 lines) |
| **Companion code** | `LLM Fine-Tuning-21-GEMINI-Finetuning/gemini_finetuning_clean.ipynb`, `data.jsonl` (10 rows, mobile-customer-support SFT), `handwritten-notes.pdf` |
| **Prerequisites** | CS-13 (instruction fine-tuning — the objective), CS-18 (OpenAI GPT fine-tuning — the sibling managed service), CS-03 (fine-tuning framework landscape), CS-23 (LoRA/QLoRA — what "adapter size" means) |
| **Neighbours** | CS-16 (Unsloth), CS-17 (Axolotl), CS-20 (SLMs), CS-21 (multimodal) — the self-hosted counterfactual |
| **Difficulty** | Beginner to run, Intermediate to run *profitably*. The API is 8 lines; the billing model is where people lose money. |
| **Hands-on required** | Yes — but not for the reason you expect. You need a GCP project, a billing account and a GCS bucket before a single line of Python runs. |
| **Estimated study time** | 4h theory + 3h practical (the practical is mostly IAM, billing and remembering to undeploy) |

---

## 0. Executive Summary

- **Vertex AI supervised fine-tuning (SFT) is a *rented* fine-tune, not a downloaded model.** The instructor states the constraint bluntly [3:35]: *"this is not a open-source model. This is a closed source... we don't have access like where we can download the model but yeah we can access the model via API."* You never see a weight tensor, you never set a learning rate schedule, you get back a **tuned model resource** and an **endpoint**. Everything that follows — cost, control, exit strategy — is a consequence of that one fact.
- **The single most important number in this module is the per-hour endpoint charge, not the per-token training charge.** The training run in the video costs roughly **$0.003** (564 tokens × $5 / 1M) [22:53], [40:10]. The endpoint that the job *automatically creates and deploys* [50:06] bills while it exists, whether or not a single request arrives. A trained-and-forgotten endpoint is the default outcome of following this video, and it is the single most expensive mistake a reader of this module can make.
- **The API surface is two calls.** `client.tunings.tune(base_model=..., training_dataset=..., config=...)` to start [45:16]; `client.models.generate_content(model=tuning_job.tuned_model.endpoint, contents=...)` to infer [54:13]. Everything else in the notebook is authentication, token counting, cost estimation, and cleanup.
- **The prerequisite chain is longer than the code.** GCP project ID, a region (`us-central1` by default [21:02]), a **billing account with a payment method actually attached** [29:40] (the free tier does *not* cover this — [28:27]), a **Cloud Storage bucket in the same region** holding the JSONL [43:53], and a Colab session authenticated as **the same Google identity that owns the project** [23:38]. The instructor's own warning [19:53]: *"this fine tuning is not very straightforward. Whatever code they have given you in the documentation if you're directly going to be executed it will not work until and unless you are not doing a proper setup."*
- **Vertex's JSONL schema is *not* OpenAI's.** OpenAI: `{"messages": [{"role": "system"\|"user"\|"assistant", "content": "..."}]}`. Vertex: `{"systemInstruction": {"role": "system", "parts": [{"text": "..."}]}, "contents": [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]}`. Note `parts` (an array), `text` (nested inside `parts`), and `role: "model"` where OpenAI says `assistant`. Full schema, plus the newer `messages`-shaped variant, in §4.2.
- **The video presents two tuning methods; the API enum exposes three.** The instructor covers supervised fine-tuning and mentions preference tuning [11:13], and the video's scope is SFT only. The `tuningMethod` field itself accepts three values — `SUPERVISED_TUNING`, `DISTILLATION` and `PREFERENCE_TUNING` — with availability varying by model family and generation (§7.1). Read the API as the superset: two of its three methods are outside this video's scope, and preference tuning maps to CS-14/CS-24/CS-25/CS-27.
- **Two weight-update regimes are exposed [13:14]:** parameter-efficient fine-tuning (PEFT — a LoRA-style adapter, selected via `adapterSize`) and full fine-tuning. On Vertex this is **not your choice of library**; it is a field in the tuning spec, and it materially changes quality, cost and whether the tuned model can be exported at all. §7.5.
- **You do not need a GPU, and this is the point of the whole design.** The instructor switches Colab to a CPU-only runtime [18:50]: *"I'm not going to train the model on my own server. The model is being trained over the Google server only."* There is no VRAM arithmetic in this module — there is a **token arithmetic** instead, and §11 does it.
- **The default validation split is automatic and easy to forget.** If you supply only a training file, Vertex holds out a slice for you. If you want control you must pass a *second* GCS URI. The video never mentions this; the automatic metrics it points at [51:36] are computed against that held-out slice, so if you don't know what the split was, you don't know what the metrics mean. §4.5 and §12.
- **Corrections you need before you run anything:** the video's model list includes models that almost certainly do not support tuning (§4.1); the transcription renders "Gemini" as "Jimny" and "Vertex" as "Vortex" throughout, so do not paste model IDs from the transcript text; the cleanup cell deletes the *model* and never touches the *endpoint* (§10.3); and there are three hyperparameters (`epochCount`, `learningRateMultiplier`, `batchSize`) that the video never names at all (§7).
- **When this beats self-hosting:** you have < ~100k training examples, no MLOps team, no GPU quota, and you want the newest base model without waiting for open weights. **When it loses:** sustained high QPS, sovereignty requirements, a need to merge weights, or anything where an idle endpoint's monthly bill exceeds a reserved GPU. Break-even table in §13.4.

---

## 1. The Problem This Solves

### 1.1 What "managed fine-tuning" actually is

A managed fine-tuning service is a **rental agreement with four clauses**:

1. **You supply a dataset in the vendor's schema, in the vendor's object store, in the vendor's region.**
2. **The vendor runs an undisclosed training procedure** on undisclosed hardware for an undisclosed number of steps, and charges you a published $/token rate for the tokens it read.
3. **You receive an opaque artefact** — a model ID and an endpoint URL. You cannot download it, inspect it, quantise it, merge it, or run it anywhere else.
4. **You pay for the artefact's existence**, not only for its use. This is the clause everyone reads past.

Contrast with CS-13, where the output of a fine-tune is a directory of safetensors on your disk that you can `scp` anywhere. Here, the output is a **resource name**:

```
projects/313902242452/locations/us-central1/models/2360465647369977856@1
projects/313902242452/locations/us-central1/endpoints/7655379395204349952
```

Those two strings are the entire deliverable. The second one is the one that bills.

### 1.2 Where the reader actually meets this

| Situation | Why someone reaches for Vertex SFT | What it costs them |
|---|---|---|
| **Enterprise already on GCP** | Procurement, IAM, VPC-SC and audit logging are solved. No new vendor. | Fine — this is the strongest case for the managed route. |
| **"We want Gemini but with our tone"** | The base model is good; the delivery is wrong. Format/style SFT is the cheapest possible win (CS-13 §4.1). | ~1k–10k examples and one tuning job. Genuinely cheap. |
| **"We cannot get GPU quota"** | A100/H100 quota requests take days-to-weeks and are often denied for new accounts. Vertex SFT needs no GPU. | You inherit Vertex's price floor. |
| **"Legal says the data cannot leave our tenancy"** | GCS in your project, Vertex training under your project's IAM. | You must verify Vertex's data-handling terms — "in your project" is not the same as "never leaves Google". |
| **"We need it cheaper than GPT-4"** | A tuned 2.5 Flash at $5/1M training tokens looks irresistible next to a frontier model. | The demo's training run cost $0.00282; the idle endpoint it created costs ~$1,459/month. **The endpoint is ~5×10⁵ times the training run** — the token rate tells you nothing about the run rate. §11.4, §16.6. |
| **"Our competitor fine-tuned on OpenAI, so we'll do Gemini"** | Vendor diversification. | Two divergent schemas, two eval harnesses, two cost models. §13. |

### 1.3 The naive approach, and exactly how it fails

**Naive plan:** "I have 10 rows of customer-support Q&A. I'll follow the docs, tune Gemini 2.5 Flash, and chat with it."

That is *literally* the video's workflow — a 564-token, 10-row dataset [40:10] — and it is a valid **learning** exercise and an invalid **production** one. Four separable failures:

1. **The dataset cannot teach anything.** Ten examples at 564 tokens total, with a *single identical system instruction repeated verbatim in all ten rows* (`You are a customer support assistant for a smartphone company...`). There is zero variance in the conditioning variable, which is exactly the CS-13 §1.3 failure mode: you are paying for a very small continued-pretraining run with a decorative prefix. Anything the model says after tuning it will say — the base model would have said too.
2. **The cost model you built is measuring the wrong thing.** The notebook's cost function computes `564 tokens × $5/1M = $0.0028` [42:25]. That number is correct and irrelevant. The endpoint bills by the hour.
3. **You will leave the endpoint up.** Vertex creates and deploys the endpoint *for you* when the job succeeds [50:06], so there is no moment at which you consciously "deploy" and therefore no moment at which you consciously decide to keep paying. The video's own cleanup cell deletes the model and not the endpoint (§10.3).
4. **You will not know whether it worked.** The dashboard gives you `evaluation total loss` and `fraction of correct next-step prediction` [51:36] on an automatic split you did not choose, computed on 2 examples if the split is 80/20 of 10 rows. §12 shows what those numbers can and cannot tell you.

### 1.4 What the alternative costs

For the same 10 rows on a self-hosted stack (CS-16/CS-17/CS-23): a Colab T4, QLoRA on a 1B–3B open model, ~2 minutes of training, $0.00 marginal, adapter file on disk, served from the same notebook. The managed path is objectively worse on every axis at this data scale. **The managed path only becomes rational when the data scale and the organisational constraints make "we cannot run this ourselves" true** — §8 formalises that.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: a managed fine-tune is a *dry-cleaning* contract, not a *sewing machine*

Self-hosted fine-tuning (CS-13, CS-16, CS-17, CS-23) is a sewing machine in your basement. You buy the fabric (the base model), you thread it yourself, you keep the machine, and you keep everything you sew. The machine costs money whether you use it or not, but you control every stitch.

Vertex SFT is the dry cleaner. You hand over the garment (your JSONL), they take the measurements (the hyperparameters you are allowed to set), they perform an operation you cannot see, and they hand you back **a ticket** — not a garment. The ticket says "your cleaned suit is at counter 7". Counter 7 is a *rented counter*. As long as the suit sits at counter 7, you pay rent on counter 7. You cannot take the suit home. You cannot take it to a different dry cleaner. You cannot inspect the stitching.

**Where this analogy breaks.** Two places, and both matter operationally:

1. **The dry cleaner will hold your suit *for free* if you don't ask for a counter.** In Vertex the artefact (the tuned model resource) exists independently of the endpoint. The *model* is cheap to keep. The *endpoint* is what bills. So there is a middle state — "trained but not deployed" — that the analogy has no word for, and it is the state you should be in 99% of the time. Getting to that state deliberately is the single highest-value operational skill in this module.
2. **The dry cleaner also keeps a copy of your garment, forever, and may improve their process using it.** Managed training means your data traverses someone else's infrastructure and your trained artefact lives in their control plane. That is a governance question, not a technical one, and it belongs in §16.7 rather than in a shrug.

### 2.2 The mechanism, stated precisely

You are not performing fine-tuning. You are **submitting a specification and receiving a resource identifier**. The pipeline that runs behind the API is, at the level of abstraction you can observe:

```
your JSONL in GCS
   │
   ├─ vendor reads N training tokens (you are billed for these)
   ├─ vendor holds out a validation slice (default split; assumed not billed — verify, §6.10)
   ├─ vendor applies a LoRA-style adapter to the base model's weights
   │     └─ rank = adapterSize  (1 / 4 / 8 / 16 / 32)
   ├─ vendor minimises the same next-token cross-entropy as CS-13,
   │     masked to assistant/model turns only
   ├─ vendor emits (a) a model resource, (b) an endpoint resource, (c) metric series
   └─ vendor starts billing you per hour for (b)
```

Three consequences fall straight out of this and they are the whole module:

| Observation | Consequence | Where |
|---|---|---|
| The training tokens are what you are billed for | Cost scales with *dataset size × epochs*, not with wall-clock. `epochCount = 3` triples the bill. | §7.1, §11.2 |
| The operation is LoRA-shaped and the rank is a dropdown | You can get near-full-FT quality on a narrow distribution but you cannot change model architecture or vocabulary, and you cannot merge the adapter out. | §7.5 |
| The endpoint is a separate billed resource | The correct lifecycle is `tune → wait → evaluate → (deploy) → predict → **undeploy**`, and the parentheses are the important part. | §5, §10.3 |

### 2.3 The mental model that prevents 90% of the mistakes

Think of the tuned model and the endpoint as **two separate line items with two separate lifetimes**:

| Resource | What it is | Billed | Can you delete it independently? |
|---|---|---|---|
| `TuningJob` | A record of a training run. Holds metrics, logs, the TensorBoard experiment, and the tuned-model pointer. | No (billed via training tokens) | Yes — but if you delete the job's artefacts you lose the metrics. |
| `Model` (`.../models/<id>@1`) | The stored tuned artefact. A versioned resource. | No published hourly storage charge for the Gemini case; verify against current pricing. | Yes — `aiplatform.Model(name).delete()` — this is what the video's cleanup does [56:39]. |
| `Endpoint` (`.../endpoints/<id>`) | A deployed serving replica behind a DNS name. | **Yes, per hour, whether or not it serves traffic.** | Yes — but you must `undeploy_all()` before `delete()`, or the deployment keeps its capacity reserved. |

> **Correction:** at [50:06] the instructor says *"automatically it will be deployed and it will give you the end point... automatically the endpoint will be created once the job will be completed"* — and then at [56:39] his cleanup cell is `aiplatform.Model(MODEL_ID).delete()`. Deleting the **Model** resource is not the same as releasing the **Endpoint**'s deployment. The correct teardown of a Vertex deployment is:
> ```python
> from google.cloud import aiplatform
> aiplatform.init(project=PROJECT_ID, location=LOCATION)
>
> ep = aiplatform.Endpoint(ENDPOINT_ID)
> ep.undeploy_all()          # releases the model from the endpoint's replicas
> ep.delete()                # deletes the endpoint resource itself
>
> aiplatform.Model(MODEL_ID).delete()   # then the tuned model
> ```
> Verify the exact method names against the Vertex AI SDK version you pin (`undeploy_all` has existed for a long time; the GenAI SDK's `client.tunings` module does not yet expose endpoint teardown at all as of the versions in the video). **Mark this as needing verification against the SDK you install** — see §6.9. The video's own lesson, though, is unambiguous and he says it twice [30:51], [57:00]: *"after fine-tune clean up everything that is required otherwise the cost would be there... the cost will be increasing by the time."*
>
> **Beyond the video:** put the teardown in a `finally` block or a scheduled Cloud Function from day one. Deployments that outlive their purpose are not an individual failure mode; they are the *default* failure mode of every managed fine-tuning service, and every experienced GCP engineer has a story about a $3k endpoint someone left up over a holiday weekend.

---

## 3. Core Concepts — Exhaustive Glossary

Every term the video introduces, plus every term you need to operate it. Read the "Common confusion" column; that is where the money is.

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **GCP** (Google Cloud Platform) | Google's cloud. The account, project, billing, IAM and networking layer. | Every prerequisite in §6.1 is a GCP object, not a Vertex object. | The instructor has to spell this out [25:29]: *"GCP is a cloud platform, over the GCP you will get the Vertex AI. This is the service."* People conflate the platform with the service. |
| **Vertex AI** | The managed ML service *inside* GCP. Model Garden, tuning, endpoints, pipelines, feature store, experiments. | This is where Gemini tuning lives. | **"Vertex"** is how the transcript writes it when the speaker says it; the transcript also renders it **"Vortex"** throughout — see the transcription-noise callout in §10.1. |
| **Google AI Studio / Gemini API** | The developer-facing surface at `aistudio.google.com` / `ai.google.dev`. API key auth, fast prototyping, free tier. | Where you *start*; not where you fine-tune. | The two surfaces share the `google-genai` SDK but have **different auth, different model IDs, different quotas and different tuning support**. The instructor makes this distinction the spine of the video [2:45]: *"both are the different thing."* |
| **`google-genai` SDK ("Google Gen AI SDK")** | The Python client. `pip install --upgrade google-genai`. Used with `vertexai=True` to talk to Vertex, or with an API key to talk to AI Studio. | The SDK the video uses [18:44]. | Not the same package as `google-cloud-aiplatform` (the "Vertex AI SDK"), which you still need alongside it for cleanup [27:00]. |
| **Vertex AI SDK for Python** | `google-cloud-aiplatform`. The older, heavier, MLOps-complete SDK: `aiplatform.Model`, `aiplatform.Endpoint`, `aiplatform.init`. | The only clean way to delete resources. The video imports it purely for cleanup [26:53]. | People assume `google-genai` supersedes it. It does not — resource lifecycle is still largely `aiplatform`-shaped. |
| **`HttpOptions(api_version=...)`** | Pins the REST API version the SDK talks to. The video uses `"v1beta1"` for the tuning client [27:22] and constructs a second client pinned to `"v1"` for inference/retrieval. | Version skew is the #1 cause of "`AttributeError: 'Client' object has no attribute 'tunings'`". Tuning features ship in `v1beta1` first. | `v1beta1` is not "old"; it is "ahead". Features graduate *from* `v1beta1` *to* `v1`. |
| **`genai.Client(vertexai=True, project=..., location=...)`** | The client construction for the Vertex backend. | Without `vertexai=True` you get the AI Studio backend and your `project`/`location` are ignored. | Passing `project`/`location` without `vertexai=True` silently does nothing. |
| **Project ID** | The GCP project identifier, e.g. `gen-lang-client-0486212375`. | Scopes every resource name and every IAM decision. | It is **not** the project *number* and **not** the project *name*. The video shows where to copy it [20:23]. |
| **Location / region** | Where the job runs and where the model is served, e.g. `us-central1`. | The endpoint's DNS is `{location}-aiplatform.googleapis.com`. **Your GCS bucket must be in a compatible location.** | `us-central1` is the default in the video [21:02] but is not the right default for an EU-resident dataset (see §10.5). |
| **Billing account** | The GCP object holding a payment method. | Vertex AI model access is completely blocked without it [17:42], [28:44]. Free-tier credits do not apply. | The instructor's "no credit card, just a GPay mandate" [29:40] is a region-specific UPI autopay flow, not a general truth. §10.2. |
| **Cloud Storage (GCS) bucket** | `gs://`-prefixed object store, analogous to S3. | The **only** supported location for the training dataset. Local Colab files and Drive are not accepted [43:20]. | The video's bucket is `gs://gemini-sft-custom` and the object is `gs://gemini-sft-custom/data.jsonl` [44:53]. Note the `gs://` scheme — a bare `bucket/path` string fails validation. |
| **JSONL / NDJSON** | One JSON object per line, newline-delimited. | The training-file format. A single well-formed JSON *array* is rejected. | The video's file is named `data.json` but is JSONL [34:47] — the extension is a lie, the parser is line-based. |
| **`systemInstruction`** | Vertex's top-level field for the system prompt, shaped `{"role":"system","parts":[{"text": ...}]}`. | Carries the persona. In the companion data it is **byte-identical across all 10 rows** [35:11]. | OpenAI puts the system prompt *inside* the `messages` array. Vertex hoists it out. Copy-pasting an OpenAI JSONL into Vertex produces a 400, not a warning. |
| **`contents`** | Vertex's array of conversation turns. | The conversation body. | The *singular* `content` is a different field (used in `CountTokens` requests). Off-by-one-s errors here produce confusing validation failures. |
| **`parts`** | A **list** of content fragments inside each turn: `[{"text": "..."}]`. | This is the multimodal seam — a `part` can be `text`, `inlineData` (bytes), or `fileData` (GCS URI). | Every non-multimodal tutorial writes `parts` as a one-element list and readers assume it is an object. It is not. It is a list because a turn can carry text + an image + an audio clip simultaneously. |
| **`role: "model"`** | Vertex's name for the assistant turn. | The turn the loss is computed on. | OpenAI says `assistant`; Vertex says `model`. `role: "assistant"` in a Vertex `contents` array is **invalid**. |
| **`TuningDataset`** | SDK type wrapping the dataset location: `TuningDataset(gcs_uri="gs://...")`. | The object passed to `tune()` [45:04]. | In some SDK versions the field is `gcs_uri`; in others it is `tuning_dataset_uri` or you pass a plain string. Version-sensitive — §6.7. |
| **`CreateTuningJobConfig`** | SDK type holding the job's configuration: display name, evaluation config, hyperparameters. | The only place you influence the run. | `tuned_model_display_name` is a *label*, not an ID. Two jobs with the same display name are legal and confusing. |
| **`tuned_model_display_name`** | Human-readable name for the resulting model. | Appears in the console and in `tuned_model.model`'s friendly name. | It does not determine the model resource name. |
| **`base_model`** | The model you are fine-tuning, e.g. `"gemini-2.5-flash"`. | Determines the price, the capabilities, and whether tuning is supported at all. | Tuning requires a **pinned version** (`gemini-2.5-flash`) rather than a floating alias for reproducibility, and the set of tunable IDs changes quarterly. §4.1. |
| **`tuningJob`** | The Vertex REST resource (`projects/{p}/locations/{l}/tuningJobs/{id}`) representing the run. | The thing whose `.state` you poll and whose `.tuned_model.model`/`.endpoint` you read [53:36]. | `client.tunings.tune()` returns it; `client.tunings.get(name=...)` re-fetches it; `client.tunings.list()` enumerates it. |
| **`JobState`** | Lifecycle enum: `JOB_STATE_PENDING`, `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_CANCELLED`, `JOB_STATE_EXPIRED`, `JOB_STATE_PARTIALLY_SUCCEEDED`. | The polling condition. The video's loop tests membership in `{"JOB_STATE_PENDING","JOB_STATE_RUNNING"}` [47:38]. | The video's spoken narration [47:28] describes the states as "pending" and "running" as if they were a closed set. They are not: a `FAILED` job exits the loop silently and the next line raises `AttributeError` on `tuned_model` being `None`. §14. |
| **`tuned_model.endpoint`** | The deployed endpoint resource name. | What you pass to `generate_content` for inference [54:13]. | `tuned_model.model` (the artefact) and `tuned_model.endpoint` (the deployment) are **different resources with different lifecycles**. The video prints both [53:36]. |
| **`tuned_model.model`** | The tuned artefact resource name, e.g. `.../models/2360465647369977856@1`. | What you delete in cleanup; what survives an undeploy. | The `@1` is a **version** suffix. A retrain produces `@2` on the same model ID, which is how Vertex does model versioning. |
| **Supervised fine-tuning (SFT)** | Next-token cross-entropy on prompt/response pairs, masked to response tokens. See CS-13 for the mechanism in full. | The method this module covers. | On Vertex, "SFT" also implies a **LoRA adapter** under the hood (`adapterSize`), not a full weight update, unless full tuning is selected. |
| **Preference tuning** | The second Vertex tuning method [11:15] — RLHF-style human-preference alignment. | Maps to CS-14/CS-24/CS-25/CS-27. Not covered here. | It consumes a *different* dataset schema (preference pairs with `score`), and you cannot feed an SFT JSONL into it. |
| **PEFT** (Parameter-Efficient Fine-Tuning) | Updating a small number of added parameters. LoRA is the canonical instance (CS-23). | Vertex exposes it as `adapterSize` [13:21]. | On Vertex you cannot choose the *target modules*, only the rank. |
| **Full fine-tuning** | Updating all base weights. | Vertex's other option [13:21]. | On a closed model this is not something you can verify happened. Treat "full tuning" on a managed service as a pricing tier, not an auditable claim. |
| **`adapterSize`** | Vertex's exposure of LoRA rank: `ADAPTER_SIZE_ONE`(1) … `ADAPTER_SIZE_THIRTY_TWO`(32). | Directly trades quality against training/inference cost and memory. §7.5. | Not the same scale as HF `r` in the obvious way, and not all sizes are available for all base models. **Needs verification against current docs** — the enum values have been stable but the *supported subset per model* has changed. |
| **`epochCount`** | Number of passes over the training data. | Multiplies your training bill linearly. §7.1. | Never mentioned in the video. Defaults to 1 for most Gemini tuning. |
| **`learningRateMultiplier`** | A scalar applied to Vertex's internal default LR (default `1.0`, range commonly 0.0–4.0). | The only LR control you get. §7.2. | You cannot set an absolute LR. If you reason in "2e-4" you are in the wrong mental model. |
| **`batchSize`** | Examples per gradient step. | Affects stability and step count; on Vertex it does **not** change your token bill. §7.3. | Not mentioned in the video. Vertex picks a default if unset. |
| **Validation dataset** | A second GCS URI of held-out examples. | The source of the automatic eval metrics. §4.5. | If omitted, Vertex splits automatically and you are not told the ratio — **verify the current default** (historically reported as 80/20 for some families). |
| **Automatic metrics** | The metrics Vertex computes without asking: training loss, evaluation total loss, fraction of correct next-step predictions, prediction counts [51:36]. | The only out-of-the-box evidence that training did anything. | They are **log-loss and next-token accuracy**, not task accuracy. A model with a superb eval loss can be useless. §12.2. |
| **LLM-based / autorater evaluation** | Optional post-training evaluation using Vertex's autorater metrics (pointwise and pairwise). | What you actually want for open-ended quality. §12.5. | Costs money, needs an output GCS URI, and the enum names and configuration shape have churned heavily. **Needs verification.** |
| **TensorBoard experiment** | Vertex logs tuning runs to a managed TensorBoard instance. | Where the loss curves live [49:33]. | The experiment is attached to the *job*; deleting the job or its backing resources loses the curves. |
| **Model Garden** | The catalogue of models in Vertex: Gemini, Imagen, Veo, Llama, plus third-party (Anthropic, Meta, Mistral, AI2, AI21) [16:51]. | Where you check what is tunable today. | Browsing Model Garden ≠ permission to tune. Tuning support is a per-model flag, and it changes without notice. |
| **Endpoint (Vertex)** | A deployed serving resource with an hourly price. | The thing that eats your budget. | "Endpoint" in Vertex means "a deployed model behind a DNS name", not "an HTTP route in your app". |
| **Undeploy** | Releasing a model from an endpoint's replicas so the endpoint stops reserving capacity. | The single most important operational action in this module. | Deleting the *model* and deleting the *endpoint* are separate operations, and the video's cleanup does only the first. §10.3. |
| **Token counting** | Estimating billable tokens before you train. | The video writes a bespoke counter [37:34] and gets 564 total tokens for 10 rows [40:10]. | Vertex bills a *rounded-up* token count and applies a floor in practice; a 564-token estimate does not become a 564-token bill. §11.2. |
| **Quota / QPS** | Per-project limits on tuning jobs and endpoint requests. | Your first tuning job may be rejected for quota, not for errors. | New projects frequently start with a tuning-jobs quota of a handful. |
| **VPC Service Controls / CMEK** | Perimeter and customer-managed encryption keys. | The enterprise reason to be on Vertex. | Neither is mentioned in the video; both change your setup materially. §16.7. |
| **`aiplatform.init()`** | Initialises the Vertex AI SDK with project/location. | Required before `aiplatform.Model(...)` works. | Must be called in the same session; the video does it only in the cleanup cell [57:00]. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 The Gemini generations, and which ones are tunable

The video's model list, quoted verbatim as the transcription renders it [10:46]:

> *"GPD 2.5 pro uh GPD 2.5 flash GPD 2.5 flash light uh 2.0 to flash and 2.0 flash light. So these many model is supporting for the finetuning."*

and again at [21:26]:

> *"Jimny 2.5 Pro flash and the flashlight 2.0 flash and flashlight."*

`GPD` and `Jimny` are speech-to-text artefacts for **Gemini**. Do not paste either string into code.

**The honest table — flagged for verification, because this is the single most churn-prone fact in the module:**

| Generation | Representative model IDs | Released | Tuning support (as the video presents it) | Status as of this writing |
|---|---|---|---|---|
| Gemini 1.0 | `gemini-1.0-pro`, `gemini-1.0-pro-001/002` | Dec 2023 | Tuning supported early, via both AI Studio and Vertex | **Retired.** The video confirms the family's retirement trajectory by citing the 1.5 Flash deprecation [7:18]. |
| Gemini 1.5 | `gemini-1.5-pro-001/002`, `gemini-1.5-flash-001/002`, `gemini-1.5-flash-8b` | 2024 | 1.5 Flash was the flagship tunable model for most of 2024; 1.5 Pro's tuning support was limited | **Superseded.** The video's own cited notice says 1.5 Flash was deprecated in **May 2025** and that *"we no longer have a model available which support fine-tuning in the Gemini API or AI Studio"* [7:18]–[7:29]. |
| Gemini 2.0 | `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.0-flash-001` | Dec 2024 – Feb 2025 | Tunable on Vertex (both Flash and Flash-Lite) | Present in the video's list and in the notebook's smoke test (`model="gemini-2.0-flash"` [notebook cell 12]). Ageing. |
| Gemini 2.5 | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | 2025 | The video lists **all three** as tunable [10:46], and the instructor fine-tunes `gemini-2.5-flash` [21:49] | **Verify before you plan.** See the Correction below. |
| Gemini 3 | `gemini-3-pro` (and successors) | Late 2025 | Not listed in the video at all; the video's *AI Studio* walkthrough shows Gemini 3 Pro as a **chat/generate** model [4:54], not as a tuning target | **Needs verification.** Do not assume the newest generation is tunable on day one. |
| Non-Gemini in Model Garden | Llama 4, Imagen 4, Veo 3, third-party (Anthropic, Meta, Mistral, AI2, AI21) [16:51] | — | Available to *call*; tuning support is per-model and largely absent for the third-party entries | Model Garden presence ≠ tuning support. |

> **Correction:** at [10:46] the instructor lists **Gemini 2.5 Pro** among the models that support fine-tuning, and at [22:29] he quotes a tuning price of *"around $25 for the 1 million token"* for 2.5 Pro. In my knowledge of the Vertex AI supervised-tuning documentation, the **Pro tier of the 2.5 and 2.0 generations has not generally been offered as a supervised-tuning base model** — tuning has been concentrated on the Flash and Flash-Lite tiers, with the Pro tier appearing on pricing pages mainly as a served-model rate. Treat the Pro row in the video's list as **suspect and needing verification against the live Vertex "Tune Gemini models" page**, and check *both* the model-support table and the pricing table, because they have disagreed with each other in the past. The practical consequence is small (Flash is the right tuning target on cost anyway) but the pricing claim is not: $25/1M is 5× the Flash rate he quotes, and if you budget for Pro you have mis-budgeted by 5×.

> **Beyond the video:** the reason to check the model list *first* and the docs *second* is that Vertex's tuning support is gated by **model version**, not model family. `gemini-2.5-flash` and `gemini-2.5-flash-001` are different tuning targets and one may be tunable while the other is not. Build your pipeline to read the base model ID from configuration and validate it against the supported list at start-up, so a Google-side change becomes a clear error at the top of the job rather than a confusing 400 in the middle of a CI run.

### 4.2 The JSONL schema — Vertex vs OpenAI, field by field

This is the section the video spends the most sustained effort on [34:11]–[36:26], and it is the section that breaks people's first attempt.

**The instructor's contrast, verbatim [34:19]–[35:11]:**

> *"You know how to format the data set for the GPT... we have the role, then content, then role, then content... Role will be system, then role will be the user, then role will be the assistant, and everything is going to be wrapped up into this single JSON object. And we have multiple JSON objects which is separated by the new line. So the file name was the data.json."*
>
> *"Now let me show you how to create a data set for the Gemini finetuning... The format is a little bit different over here. **We have systemInstruction instead of the message.** Now role will be the system. Then apart from this we have one more key that is **part**. Now under the part actually we will be having a **text**."*

He then notes the key asymmetry in `role` naming [36:07]:

> *"the role is the user then role again one more role is a **model** — so role user means user is asking a question, role model means model is generating an answer."*

**Side-by-side, the same conversation in both schemas:**

```json
// OpenAI / CS-18 schema — file: data.jsonl (JSONL, one object per line)
{"messages": [
  {"role": "system",    "content": "You are a customer support assistant for a smartphone company."},
  {"role": "user",      "content": "How long does the warranty last on a new smartphone?"},
  {"role": "assistant", "content": "Most smartphones include a one-year limited warranty."}
]}
```

```json
// Vertex / Gemini schema — same conversation, line 1 of the companion data.jsonl
{"systemInstruction": {"role": "system", "parts": [{"text": "You are a customer support assistant for a smartphone company. You are friendly, concise, and provide only factual answers related to smartphones."}]},
 "contents": [
   {"role": "user",  "parts": [{"text": "How long does the warranty last on a new smartphone?"}]},
   {"role": "model", "parts": [{"text": "Most smartphones include a one-year limited warranty that covers manufacturing defects. Exact details are available in the official documentation."}]}
]}
```

**The differences that will bite you, in order of how often they bite:**

| # | Difference | OpenAI | Vertex | What breaks if you get it wrong |
|---|---|---|---|---|
| 1 | Assistant role name | `"assistant"` | `"model"` | `400 INVALID_ARGUMENT`; the error text does not say "use model instead of assistant". |
| 2 | System prompt placement | Inside `messages[0]` | **Hoisted** to a sibling top-level key `systemInstruction` | A Vertex parser sees a `contents` array with a leading `system` turn and either rejects it or silently trains on it as a user turn. |
| 3 | Text field nesting | `content` is a **string** | `parts` is a **list**, and the text lives at `parts[0].text` | `KeyError: 'text'` or, worse, an empty turn that trains the model to emit nothing. |
| 4 | Top-level conversation key | `messages` | `contents` | Validation error. There is no aliasing. |
| 5 | Extra top-level fields | `tools`, `parallel_tool_calls`, `weight` (rare) | `contents`, `systemInstruction` | Vertex is stricter about unknown fields than OpenAI. |
| 6 | Line format | JSONL | JSONL | Both are line-delimited. Neither accepts a JSON array. |

> **Beyond the video:** there is a **second Vertex schema in the wild** — the `messages`-shaped variant, `{"messages": [{"role":"user","content":"..."}, {"role":"model","content":"..."}]}` — which Vertex accepted for some tuning flows to reduce exactly this copy-paste pain. Do not assume either shape works; **validate your file against the schema the console's "sample dataset" link gives you on the day you train**, because the two have coexisted and the accepted one has flipped. The companion notebook in this repo uses the `systemInstruction`/`contents`/`parts` shape, which is the one the video demonstrates; if your job 400s on a schema you copied from this module, that is the first thing to check.

**A real defect in the companion dataset worth noticing.** All ten rows carry a **byte-identical** `systemInstruction`:

```
You are a customer support assistant for a smartphone company.
You are friendly, concise, and provide only factual answers related to smartphones.
```

That is 26 tokens of the ~56-token average row [40:20] — **roughly 46% of your bill is a constant that carries zero gradient information about the task**, because `p(response | instruction)` has no variance in the conditioning variable (CS-13 §1.3). Row 9 is the only interesting one: it is a *refusal* — `"Can you recommend a good laptop for work?"` → `"I'm here to assist only with smartphone-related questions."` That is a hand-authored negative, and in a 10-row dataset it is 10% of the signal. In a 10,000-row dataset you would want roughly 5–10% of rows to be exactly this shape.

### 4.3 The REST surface under the SDK

The `google-genai` SDK is a thin wrapper over Vertex's REST API. Knowing the REST shape lets you debug the SDK, script it from any language, and understand what `tunings.tune()` actually did.

**Create a job:**

```http
POST https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/us-central1/tuningJobs
Authorization: Bearer $(gcloud auth print-access-token)
Content-Type: application/json

{
  "displayName": "youtube-sft-job",
  "baseModel": "gemini-2.5-flash",
  "supervisedTuningSpec": {
    "trainingDatasetUri": "gs://gemini-sft-custom/data.jsonl",
    "validationDatasetUri": "gs://gemini-sft-custom/validation.jsonl",
    "hyperParameters": {
      "epochCount": "3",
      "learningRateMultiplier": 1.0,
      "batchSize": 4,
      "adapterSize": "ADAPTER_SIZE_FOUR"
    }
  }
}
```

Note three things a reader always gets wrong:

- **`epochCount` is a JSON *string*** (`"3"`), because it is an `int64` in the protobuf and JSON cannot represent int64 exactly. The SDK accepts a Python `int` and serialises it correctly; hand-written REST does not.
- **`learningRateMultiplier` is a JSON *number*** (`1.0`), because it is a float.
- **The path prefix is the *region*.** `us-central1-aiplatform.googleapis.com`, not `aiplatform.googleapis.com`. This is the source of the "404 on a URL that looks right" class of bug.

**Poll it:**

```http
GET https://us-central1-aiplatform.googleapis.com/v1/{tuningJobName}
```

**List them:**

```http
GET https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/us-central1/tuningJobs
```

**The SDK equivalence, mapped line by line to the video:**

| REST concept | SDK expression in the notebook | Video timestamp |
|---|---|---|
| Build the job spec | `CreateTuningJobConfig(tuned_model_display_name="example-sft-job")` | [45:16] |
| Point at GCS | `TuningDataset(gcs_uri="gs://gemini-sft-custom/data.jsonl")` | [45:04] |
| POST + poll handle | `client.tunings.tune(base_model="gemini-2.5-flash", training_dataset=training_dataset, config=...)` | [45:18] |
| GET | `client.tunings.get(name=tuning_job.name)` | [47:49] |
| LIST | `client.tunings.list()` → iterable of jobs, **newest first** | [33:36], [46:53] |
| Inspect state | `tuning_job.state` | [47:49] |
| Read the artefacts | `tuning_job.tuned_model.model`, `tuning_job.tuned_model.endpoint` | [53:36] |
| Inference | `client.models.generate_content(model=..., contents=...)` | [54:13] |
| Delete | `aiplatform.Model(MODEL_ID).delete()` | [56:39] |

> **Beyond the video:** the SDK emits an explicit warning on `tunings.tune()` — visible in the companion notebook output: *"ExperimentalWarning: The SDK's tuning implementation is experimental, and may change in future versions."* Treat that as a contract term, not a courtesy. Pin `google-genai` to an exact version in your `requirements.txt` (the notebook resolved **1.62.0**, upgrading from 1.60.0 [notebook cell 2]), and treat a version bump as a code change requiring a test run. The `google-genai` tuning surface moved fast enough that a `pip install --upgrade` in a Monday deploy has broken Tuesday's pipeline for people.

### 4.4 What happens at the tensor level (as far as you can observe)

You cannot see the base weights, but the *shape* of the update is inferable from what Vertex exposes, and understanding it explains the limits.

Vertex exposes a single knob: `adapterSize ∈ {1, 4, 8, 16, 32}`. That is a **LoRA rank**, i.e. the inner dimension `r` of the decomposition `ΔW = BA` with `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}` (CS-23 §4.2). The parameter count of the update is therefore:

$$|\Delta W| = r \cdot (d + k) \quad \text{per adapted matrix}$$

For a transformer block with hidden size `d = 4096`, adapting the standard seven projections (`q, k, v, o, gate, up, down`) gives, per block:

| `adapterSize` (rank `r`) | Params per matrix (`r·(4096+4096)`) | × 7 matrices | × ~32 blocks | Total adapter params |
|---|---|---|---|---|
| 1 | 8,192 | 57,344 | 1,835,008 | **~1.8 M** |
| 4 | 32,768 | 229,376 | 7,340,032 | **~7.3 M** |
| 8 | 65,536 | 458,752 | 14,680,064 | **~14.7 M** |
| 16 | 131,072 | 917,504 | 29,360,128 | **~29.4 M** |
| 32 | 262,144 | 1,835,008 | 58,720,256 | **~58.7 M** |

*All figures are estimates from a nominal 7B-class, 32-block, d=4096 backbone. Google does not publish the exact target-module set or block count, so treat this as an order-of-magnitude model, not a specification.*

Four operational readings of that table:

1. **Even the largest adapter is ~0.5–0.8% of a 7B backbone.** This is why Vertex SFT is *cheap to store* and *cheap to serve at low traffic*: the served model is base + adapter, and the adapter is tens of megabytes.
2. **The whole model-plus-adapter is why you cannot have it.** A 60 MB artefact is trivially portable; Google does not give it to you because the *base* is the product, not the adapter. The adapter alone is useless without the weights it was trained against.
3. **Rank caps what you can learn.** Rank-1 adapters are for style and format; rank-16/32 are for genuinely new behaviour on a narrow distribution. `adapterSize` and `epochCount` interact: a rank-1 adapter trained for 10 epochs has more capacity to memorise than to generalise, which is how you get a tuned model that parrots the training set and fails on paraphrases (CS-13 §4.9).
4. **Adapter size affects your inference bill indirectly.** Bigger adapters consume more accelerator memory per replica, and in some Vertex configurations that changes how many concurrent requests fit — but it does **not** change the per-hour endpoint charge, which is what actually dominates (§11.4).

> **Beyond the video:** if you want to know *what* a rank-1 adapter can do before you spend money on Vertex, reproduce the experiment locally on an open model of similar size. Train rank-1 vs rank-16 LoRA on the same 500-example dataset with the same epochs (CS-23), and measure the eval delta. The transferable lesson is not the number; it is *the shape of the curve* — you will almost always find that rank matters far less than dataset quality, which is exactly the CS-13 thesis. Spending $200 of Vertex tuning before you have spent one afternoon on a free local rank sweep is the most common misallocation of effort in this module.

### 4.5 The tuning / validation split, and why you must know it

The video shows exactly one dataset object:

```python
training_dataset = TuningDataset(gcs_uri="gs://gemini-sft-custom/data.jsonl")
```

There is **no validation dataset anywhere in the notebook, and no mention of one in the transcript**. Yet at [51:36] the instructor points at the dashboard and lists:

> *"evaluation fraction of correct next step prediction, evaluation number of prediction, evaluation total loss, training loss — right, everything guys you will be able to get over here."*

The word `evaluation` in those metric names means those numbers were computed **on data that was not trained on**. Which data? Vertex held it out.

What you need to know, and what the video does not tell you:

| Question | Answer | Confidence |
|---|---|---|
| If I pass only `trainingDatasetUri`, does Vertex hold out a slice? | Yes. Vertex performs a default split of the supplied dataset. | High that a split happens; **the ratio needs verification** against the current docs (80/20 is the commonly reported figure for Gemini SFT). |
| Am I billed for the held-out slice? | Historically, the tuning bill is computed on training tokens. Assume the held-out slice is not billed, but **verify** — this is a 20% budget line if you are wrong. | Medium. |
| Can I control it? | Yes — pass `validationDatasetUri` (SDK: `validation_dataset=...`). | High. |
| How big should the held-out set be? | ≥ 100 examples, and ≥ 10% of the dataset, whichever is larger. | Industry practice (CS-13 §12). |
| What if my dataset is 10 rows? | You get 2 validation examples. Every "evaluation" number in that job is a statistic computed on 2 samples. | Arithmetic. |

**Why the default split is a trap in both directions:**

- **With 10 rows**, the eval metrics are noise. `evaluation total loss` on 2 examples moves by large amounts when one example changes. You will read a "good" number and believe it.
- **With 100,000 rows**, the default split wastes 20,000 rows of your best data and bills you for 80,000 — and if you later discover you wanted a *stratified* split (per-intent, per-language, per-difficulty), you cannot recover it because the split is not reproducible across jobs.

**The production rule:** always pass an explicit validation file, generated by your own deterministic, stratified, seeded splitter, and version-controlled alongside the training file:

```python
# split_dataset.py — run this BEFORE you touch Vertex. Deterministic and auditable.
import json, hashlib, random
from collections import defaultdict

def split(src_path: str, train_path: str, val_path: str, val_frac: float = 0.15,
          stratify_key: str = "systemInstruction", seed: int = 1337) -> None:
    """Stratified split with a content hash for reproducibility.

    Rows are bucketed by a stratification key (here, the system instruction),
    then shuffled with a fixed seed, so re-running produces byte-identical files.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    with open(src_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue                               # tolerate trailing newlines
            row = json.loads(line)
            key = json.dumps(row.get(stratify_key, ""), sort_keys=True)
            buckets[key].append(row)

    train, val = [], []
    for key in sorted(buckets):                        # sorted → stable iteration order
        rows = buckets[key]
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * val_frac))) if len(rows) > 1 else 0
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])

    for path, rows in ((train_path, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        print(f"{path}: {len(rows)} rows, sha256[:16]={digest}")

if __name__ == "__main__":
    split("data.jsonl", "train.jsonl", "validation.jsonl", val_frac=0.2)
```

Then upload **both** files and pass both URIs. The split is now a fact you own rather than a behaviour of the service, and the hash is what you pin into your model card.

### 4.6 Memory, compute and token accounting (the arithmetic that replaces VRAM)

There is no VRAM arithmetic in this module, and that is the point. But there is a **token** arithmetic, and it is the one that appears on your invoice.

**The billable quantity for tuning:**

$$T_{train} = N_{examples} \times \bar{t}_{row} \times \text{epochCount}$$

where $\bar{t}_{row}$ is the mean token count of a rendered training row. Note carefully what is *in* that count: `systemInstruction` text + every `user` turn + every `model` turn + whatever framing tokens Vertex adds. In the companion dataset:

| Quantity | Value | Source |
|---|---|---|
| Examples | 10 | [40:10] |
| Total tokens (all rows) | 564 | [40:10] |
| Minimum tokens in a row | 51 | [40:12] |
| Maximum tokens in a row | 62 | [40:18] |
| Mean tokens per row | 56 | [40:20] |
| Mean minus the constant system instruction (~26 tokens, estimate) | ~30 | computed |
| Tokens in the constant system instruction (estimate) | ~26 × 10 = 260 | computed — **46% of the dataset is one repeated string** |

**Worked cost, exactly as the notebook computes it:**

| Model | Tuning rate quoted [22:29] | Tokens | Epochs | Billable tokens | Tuning cost |
|---|---|---|---|---|---|
| `gemini-2.5-flash-lite` | $1.50 / 1M | 564 | 1 | 564 | **$0.00085** |
| `gemini-2.5-flash` (the video's choice) | $5.00 / 1M | 564 | 1 | 564 | **$0.00282** |
| `gemini-2.5-flash` | $5.00 / 1M | 564 | 3 | 1,692 | **$0.00846** |
| `gemini-2.5-pro` | $25.00 / 1M | 564 | 1 | 564 | **$0.0141** |
| `gemini-2.5-flash`, a realistic starter set | $5.00 / 1M | 2,000,000 | 1 | 2M | **$10.00** |
| `gemini-2.5-flash`, a realistic production set | $5.00 / 1M | 20,000,000 | 3 | 60M | **$300.00** |

**Read that table twice.** The video's entire training run costs **less than one US cent**. It is not the training that costs money. It never was.

**Serving cost, which is the actual budget:**

| Cost component | Basis | Worked example |
|---|---|---|
| Endpoint hourly charge | Per hour the deployment exists | **Not published in the video.** Order of magnitude for a small tuned Gemini deployment: **$1–$5 per hour**. *Needs verification against the live Vertex pricing page.* |
| | | $2.00/hr × 24 × 30.4 = **$1,459 / month, idle or busy** |
| Per-token inference on the tuned model | Per 1M input / output tokens, at a rate above the base model | At base-2.5-Flash-class * 2–3× premium, serving 5M input + 1M output tokens/month ≈ **$3–$12 / month** |

The ratio between the top row and the bottom row is the whole story: **the endpoint's existence costs ~100–500× more than its use at low traffic.** Everything in §10.3 and §16 exists because of this one number.

> **Beyond the video:** the reason to model this as "per hour" rather than "per request" is not a Vertex quirk; it is the standard economics of GPU serving. A deployed replica holds accelerator memory whether it is computing or not, and vendors pass that through as a reservation charge. AWS SageMaker endpoints, Azure ML managed online endpoints, and Vertex endpoints all bill this way. The *contrast* is the serverless model (Cloud Run, Modal, RunPod Serverless) which scales to zero — and the reason you cannot have that here is that Google will not let you run the weights anywhere but a Vertex endpoint. This is the single clearest example in the handbook of **a capability you are buying and a constraint you are accepting in the same transaction.**

---

## 5. The End-to-End Pipeline

### 5.1 The twelve stages (the video shows eleven of them)

```text
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STAGE 0 — ORGANISATIONAL (once per org, not per job)                        │
  │  GCP account → Project → Billing account + payment method → IAM              │
  │  Failure mode: you skip this and every later stage fails with a billing 403  │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 1 — AUTHOR THE DATA                                                   │
  │  Collect conversations → render to Vertex JSONL (systemInstruction/contents) │
  │  Output: data.jsonl, N rows                                                 │
  │  Failure mode: OpenAI schema pasted in; role:"assistant"; no "parts"          │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 2 — SPLIT (the stage the video omits)                                 │
  │  Deterministic stratified split → train.jsonl + validation.jsonl + hashes    │
  │  Failure mode: rely on the implicit split; you cannot reproduce or stratify   │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 3 — VALIDATE LOCALLY (cheap)                                          │
  │  Parse every line, assert schema, count tokens, estimate cost                │
  │  The video's two custom functions do a version of this [37:34], [41:42]      │
  │  Failure mode: skip it and discover a bad line after paying to train          │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 4 — UPLOAD TO GCS (same region as the job)                            │
  │  gsutil cp train.jsonl validation.jsonl gs://<bucket>/                       │
  │  Failure mode: local Colab path or Drive path — not accepted [43:20]          │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 5 — AUTHENTICATE                                                      │
  │  auth.authenticate_user() in Colab, as the SAME identity that owns the project│
  │  Failure mode: personal Gmail in Colab vs work account on GCP → 403 [23:38]   │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 6 — BUILD THE CLIENT                                                  │
  │  genai.Client(vertexai=True, project, location, HttpOptions("v1beta1"))       │
  │  Failure mode: vertexai=True omitted → AI Studio backend, no project scope    │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 7 — SMOKE-TEST THE BASE MODEL                                         │
  │  client.models.generate_content(model="gemini-2.5-flash", contents="...")    │
  │  This is the cheapest possible test of auth + billing + region + quota        │
  │  Failure mode: skip it and misattribute an auth bug to a training bug         │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 8 — START THE TUNING JOB                                              │
  │  client.tunings.tune(base_model, TuningDataset(gcs_uri), CreateTuningJobConfig)│
  │  Returns: tuningJob (state = PENDING)                                        │
  │  Failure mode: wrong base model string; dataset URI in another region         │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 9 — POLL TO TERMINAL STATE                                            │
  │  while state in {PENDING, RUNNING}: refetch; sleep(60)                       │
  │  Watch the loss curve in the TensorBoard experiment widget                    │
  │  Failure mode: FAILED exits the loop silently → AttributeError on tuned_model │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 10 — EVALUATE, THEN DECIDE                                            │
  │  Read automatic metrics → run YOUR OWN held-out suite against the endpoint    │
  │  Decision: promote, retrain, or abandon                                       │
  │  Failure mode: trusting eval loss; promoting on the automatic metrics alone   │
  └─────────────────────────────────────────────────────────────────────────────┘
                                        │
  ┌─────────────────────────────────────▼───────────────────────────────────────┐
  │  STAGE 11 — UNDEPLOY + DELETE (**the stage that saves the most money**)      │
  │  endpoint.undeploy_all() → endpoint.delete() → Model.delete()                 │
  │  Failure mode: delete the Model and leave the Endpoint deployed → $$$/hour    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Stage-by-stage contract table

| # | Stage | Input | Operation | Output | Failure mode | Video ref |
|---|---|---|---|---|---|---|
| 0 | Org setup | Google account | Create project; attach billing; grant IAM | Project ID, billing-enabled project | "Billing not enabled" 403 on the very first call | [20:08], [28:44] |
| 1 | Author data | Raw conversations | Render to Vertex JSONL | `data.jsonl` | OpenAI schema; `role: assistant` | [34:47] |
| 2 | Split | `data.jsonl` | Deterministic stratified split | `train.jsonl`, `validation.jsonl` + hashes | Implicit split, unreproducible | **absent** |
| 3 | Validate | JSONL | Parse, assert, count tokens, estimate $ | Token histogram, cost estimate | Malformed line discovered post-training | [37:34], [40:10], [41:42] |
| 4 | Upload | Local files | `gsutil` / `gcloud storage cp` | `gs://bucket/…` | Local path passed to `TuningDataset` | [43:53] |
| 5 | Auth | Colab session | `auth.authenticate_user()` | OAuth token bound to project | Identity mismatch | [23:12], [23:38] |
| 6 | Client | Project + location | `genai.Client(vertexai=True, …)` | Client object pinned to `v1beta1` | Missing `vertexai=True` | [27:07] |
| 7 | Smoke test | Client | `generate_content` on base model | Text response | Untested auth/billing | [27:39] |
| 8 | Start job | GCS URI + config | `tunings.tune(...)` | `TuningJob` in `PENDING` | Bad base model, cross-region dataset | [45:16] |
| 9 | Poll | Job name | Loop + `sleep(60)` | Terminal state + metrics | Silent exit on `FAILED` | [47:38] |
| 10 | Evaluate | Tuned model / endpoint | Metrics + your own suite | Go / no-go | Promoting on eval loss | [51:36] |
| 11 | Teardown | Deployment | `undeploy_all` → `delete` | Zero running cost | **Endpoint left deployed** | [56:39] |

### 5.3 The critical path, and where the wall-clock actually goes

The instructor timed his own run [50:56]:

> *"when I executed it first time guys, it took around — you won't believe it — took around guys 15 to 20 minutes."*

Fifteen to twenty minutes for a 564-token, 10-row dataset. That is the **fixed scheduling and provisioning overhead** of a managed tuning job, not training time. Consequences:

| Dataset size | Approximate wall clock | Practical implication |
|---|---|---|
| 10 rows / 564 tokens | 15–20 min [50:56] | 99.9% provisioning, 0.1% compute |
| 1,000 rows / ~56k tokens | 20–40 min (estimate) | Still provisioning-dominated |
| 50,000 rows / ~3M tokens | 1–3 h (estimate) | Compute starts to matter |
| 1M rows / ~60M tokens | 12–48 h (estimate) | Batch scheduling, queueing, potential preemption-and-retry |

**The corollary the video does not draw:** because the fixed overhead is ~20 minutes and the marginal cost of a small dataset is ~nothing, **the right move is always to train on your full, properly-split, properly-sized dataset in one job** rather than iterating with toy subsets. The opposite of the self-hosted intuition, where a 10-row smoke test is free and a full run costs GPU-hours. Here, the smoke test costs the same 20 minutes as the real run and tells you nothing.

> **Beyond the video:** budget for this asynchrony in your pipeline design. A tuning job is a **long-running background operation that can take hours**, so the job submission and the job result must be decoupled. Do not write a script that blocks for 20 minutes inside a web request, a Lambda/Cloud Function timeout, or a CI step with a 10-minute cap. Submit, persist the job name to a database or a state file, and have a separate poller (Cloud Scheduler + Cloud Run, or a cron on a small VM) resolve it. The video's `while … time.sleep(60)` loop is correct for a notebook and wrong for anything else.

---

## 6. Hands-On Code (annotated)

Everything below is reconstructed from `gemini_finetuning_clean.ipynb` and the video's spoken walkthrough. Where the notebook's cell is broken (several cells in the shipped notebook raise `NameError` because they were executed out of order), the code is presented in the order that actually works, and the breakage is called out.

### 6.0 Environment

```python
# The ONE dependency. The notebook's own log shows 1.60.0 upgraded to 1.62.0.
# Pin it:  google-genai==1.62.0
!pip install --upgrade google-genai
```

```
# Colab log, abbreviated [notebook cell 2]:
#   Requirement already satisfied: google-genai in /usr/local/lib/python3.12/dist-packages (1.60.0)
#   Downloading google_genai-1.62.0-py3-none-any.whl (724 kB)
#   Successfully installed google-genai-1.62.0
```

**Runtime selection** — this is not incidental, it is the architectural point [18:50]:

> *"I will select the CPU only because I don't require any GPU here because I'm not going to train the model on my own server. The model is being trained over the Google server only. I'm just hitting it through the Google API."*

Selecting a GPU runtime costs Colab credits and buys nothing. After installing, Colab demands a session restart [19:29] — an in-notebook `pip install` of a package you are about to import does not take effect in the running kernel.

### 6.1 Stage 0 — the prerequisites the docs do not mention

The instructor's framing [19:53]:

> *"This fine tuning is not very straightforward. Whatever code they have given you in the documentation, if you're directly going to execute it, it will not work until and unless you are not doing a proper setup."*

| Prerequisite | Where it comes from | Why it matters | Console path |
|---|---|---|---|
| **Project ID** | GCP console, top bar | Every resource name is scoped by it. Copy it verbatim [20:23]. | `console.cloud.google.com` → project selector |
| **Location** | Vertex AI dashboard → region selector | `us-central1` is the default [21:02]. Endpoint hostnames embed it. | Vertex AI → Dashboard |
| **Billing account, enabled** | Billing console | **Hard gate.** Without it, neither generation nor tuning works [31:16]: *"neither you can generate the output nor you can perform the finetuning."* | Billing → Link a billing account |
| **A budget alert** | Billing → Budgets & alerts | Not in the video. Non-negotiable. §11.5. | Billing → Budgets & alerts |
| **Cloud Storage bucket** | Cloud Storage console | The training file must live here [43:20]. Bucket **name is globally unique**. | Cloud Storage → Buckets → Create |
| **Region alignment** | — | Bucket location should match the tuning location (or at least be in a compatible multi-region). Cross-region dataset reads can be rejected or slow. | Bucket creation → Location type |
| **IAM role on the identity** | IAM & Admin | You need permission to create tuning jobs and to read the bucket. §6.2. | IAM & Admin → IAM |
| **Same Google identity in Colab and GCP** | — | [23:38]: *"please create a notebook with the same ID… otherwise you will face a issue… you will not be able to authenticate properly."* | — |
| **Tuning-job quota** | Quotas page | New projects have low limits; your first `tune()` may fail for quota, not for errors. | IAM & Admin → Quotas |

**Bucket upload, in the video's own naming [44:53]:**

```bash
# The video's bucket and object. Substitute your own globally-unique bucket name.
gcloud storage buckets create gs://gemini-sft-custom --location=us-central1

# Upload BOTH files. The video shows only the training file.
gcloud storage cp data.jsonl       gs://gemini-sft-custom/data.jsonl
gcloud storage cp validation.jsonl gs://gemini-sft-custom/validation.jsonl

gcloud storage ls gs://gemini-sft-custom/
```

> *"GS means Google storage, then 'Gemini SFT custom' — this is my bucket name — and under this bucket, this is my data set."* [44:53]

### 6.2 Which IAM roles actually matter

The video never names a single IAM role, and this is the gap that costs readers the most time. The `roles/aiplatform.user` role is the broad one that covers tuning and endpoints; it is also broader than most reviewers will approve. The practical decomposition:

| Need | Role | Scope | Notes |
|---|---|---|---|
| Run tuning jobs, create/invoke endpoints | `roles/aiplatform.user` | Project | The catch-all. Includes `tuningJobs.*`, `endpoints.*`, `models.*`. |
| Read the training data | `roles/storage.objectViewer` | **The bucket**, not the project | Grant on the bucket to narrow it. |
| Write the model/eval artefacts | `roles/storage.objectAdmin` or a bucket-scoped `objectCreator` | The output bucket | Only needed for evaluation output URIs. |
| Submit jobs from a service account | `roles/iam.serviceAccountUser` on that SA | SA | Required for "act as" flows, e.g. scheduled jobs. |
| View metrics / TensorBoard | `roles/aiplatform.user` covers it; `roles/monitoring.viewer` for Cloud Monitoring | Project | — |
| Manage Vertex "Experiments" | `roles/aiplatform.admin` if you hit permission errors on the experiment widget | Project | The narrow `aiplatform.user` role has historically not been sufficient for every Experiments operation. |
| Anything involving user-managed service accounts for Vertex | `roles/iam.serviceAccountUser` **and** the SA needs the Vertex roles | — | Classic two-hop mistake: you have the role, the SA does not. |

> **Beyond the video:** three IAM rules that will save you a day.
> 1. **Prefer a service account over a human identity** for anything automated. `auth.authenticate_user()` in Colab binds *your* OAuth identity; a Cloud Run job should use an attached SA. The two paths have different failure modes.
> 2. **The bucket-scoped grant is your security story.** `roles/aiplatform.user` at project level plus `roles/storage.objectViewer` at bucket level is a materially better posture than both at project level, and it is the version that survives a security review.
> 3. **`roles/aiplatform.admin` is not a shortcut.** It includes endpoint *deletion* and model registry changes; granting it to a data scientist to fix a metrics-permission error is how a demo endpoint gets deleted by a notebook re-run.

### 6.3 Stage 5 — authentication

```python
# Colab only. On a VM / Cloud Run / your laptop, use GOOGLE_APPLICATION_CREDENTIALS
# with a service-account key, or `gcloud auth application-default login`.
from google.colab import auth
auth.authenticate_user()     # [notebook cell 7], video [23:12]
```

The pop-up asks for permission to *"view and manage your data across Google Cloud Platform services"*. **Read that scope.** It is `cloud-platform` — full account scope, not a Vertex-scoped token. That is how Colab auth works, and it is why production pipelines should not use this path.

### 6.4 Stage 6 — configuration and client

```python
PROJECT_ID = "gen-lang-client-0486212375"     # your project ID, from the GCP console  [20:23]
LOCATION   = "us-central1"                    # the video's default region             [21:02]

# Optional — only if you enable autorater evaluation.
# OUTPUT_GCS_URI = "gs://gemini-sft-custom/eval-results/"
```

> **Correction:** the notebook ships with a **live project ID and project number** in it (`gen-lang-client-0486212375`, and `projects/313902242452/...` appears in the printed job names). Project IDs and numbers are not secrets in the cryptographic sense, but publishing them is unnecessary surface area and the notebook's author did it by accident. Read them from the environment (`os.environ["GOOGLE_CLOUD_PROJECT"]` resolves inside Vertex and Cloud Run automatically) rather than hard-coding them, and if you publish a notebook, replace them with placeholders.

```python
import time
from google import genai
from google.genai.types import HttpOptions, CreateTuningJobConfig, TuningDataset
from google.colab import auth
from google.cloud import aiplatform        # imported here, used only in cleanup
```

The `google-genai` import table, expanded from the notebook's own markdown cell:

| Import | Why it exists | Where it bites |
|---|---|---|
| `time` | The polling loop's `sleep(60)` — rate-limit hygiene. | Not needed if you use an event-driven poller. |
| `from google import genai` | The client for tuning, job listing, and inference. | — |
| `HttpOptions` | Pins `api_version`. `v1beta1` for tuning [27:26]; `v1` for inference and job retrieval in some SDK versions. | Mismatched version → `AttributeError` on `client.tunings`. |
| `CreateTuningJobConfig` | Display name + eval config + future hyperparameters. | The place `epochCount`/`adapterSize` go (§7). |
| `TuningDataset` | Wraps the GCS URI without downloading. | Field name is SDK-version-sensitive. |
| `google.colab.auth` | Colab OAuth. | Colab-only. |
| `google.cloud.aiplatform` | Resource lifecycle: `Model`, `Endpoint`, `Model.delete()`. | Requires `aiplatform.init()` first. |

```python
client = genai.Client(
    vertexai=True,                          # ← WITHOUT THIS YOU ARE ON AI STUDIO, NOT VERTEX
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1beta1"),   # tuning lives on the beta surface [27:26]
)
```

**Smoke test before you spend anything** [27:39]:

```python
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is a color of the water?",
)
print(resp.text)
```

The notebook's actual output for this cell was `OK` — a real, if unhelpful, answer. Note the notebook's smoke-test cell uses `model="gemini-2.0-flash"` while the tuning cell uses `base_model="gemini-2.5-flash"`; both work, and testing the *base* model is the point. **If this call raises a billing error, stop** — every subsequent stage will fail the same way, and you will waste an hour misreading it as a code bug.

### 6.5 Stage 3 — validate and count tokens (the video's custom function)

The instructor wrote two helpers that the Vertex docs do not give you. He says so plainly [39:50]: *"this is just a custom method which I have written to give you the detailed understanding of the data."* The code below is a **faithful reconstruction** of what he demonstrates on screen, described in his own words at [37:09]–[37:55]: *"this function, apart from the text we are checking part, then content from the role and text… we are going to check the complete data."*

```python
# Reconstructed from the video walkthrough — the notebook ships without these cells.
# Marked as a reconstruction: the exact formatting is the author's, the logic is the video's.
import json, os
from collections import Counter
from typing import Iterator

def iter_examples(json_path: str) -> Iterator[dict]:
    """Yield one parsed JSON object per non-empty line.

    JSONL, not JSON. The file is named data.json in the video but is line-delimited.
    """
    with open(json_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def example_to_text(example: dict) -> str:
    """Flatten a Vertex SFT row into the exact string the tuner will tokenise.

    Order matters: systemInstruction first, then every turn in `contents`.
    Every `parts` entry's `text` is concatenated — this is the multimodal seam.
    """
    chunks: list[str] = []

    sys_instr = example.get("systemInstruction") or {}
    for part in sys_instr.get("parts", []):
        if "text" in part:
            chunks.append(part["text"])

    for turn in example.get("contents", []):
        for part in turn.get("parts", []):
            if "text" in part:
                chunks.append(part["text"])

    return "\n".join(chunks)

def count_tokens_for_json(json_path: str, model: str = "gemini-2.5-flash",
                          show_top: int = 10) -> dict:
    """Per-row token accounting, then a cost estimate.

    Returns the summary dict AND prints it, matching the video's output:
      total examples / total tokens / min / max / average / per-line / top-N largest
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"dataset not found: {json_path}")   # video checks this [37:53]

    per_row: list[tuple[int, int]] = []          # (row_index, token_count)
    total = 0
    for idx, example in enumerate(iter_examples(json_path), start=1):
        text = example_to_text(example)
        # NOTE: the video counts tokens by re-tokenising through the Gemini model.
        # That is a network call per row here; the notebook uses the SDK's count_tokens.
        n = len(text.split()) * 1.3              # ← placeholder: replace with a real tokeniser
        per_row.append((idx, int(n)))
        total += int(n)

    counts = [c for _, c in per_row]
    summary = {
        "model": model,
        "total_examples": len(per_row),
        "total_tokens": total,
        "min_tokens": min(counts) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
        "avg_tokens": round(total / len(counts), 1) if counts else 0.0,
        "per_row": sorted(per_row, key=lambda t: t[1]),
        "top_largest": sorted(per_row, key=lambda t: -t[1])[:show_top],
    }
    return summary
```

> **Correction:** the code above deliberately contains a **placeholder token count** (`len(text.split()) * 1.3`) because the honest answer is that you cannot count Gemini tokens with a local tokeniser — Gemini's tokeniser is not published as an open artefact. The correct implementation is to call the API: `client.models.count_tokens(model=..., contents=...)`, which returns `total_tokens`. The video's function does exactly this, and that is why it is slow and why running it over a 100k-row dataset is itself a non-trivial API cost and a rate-limit risk. **For datasets larger than a few thousand rows, sample 200 rows, measure the mean tokens/row, multiply by N, and add a 20% safety margin.** Counting every row for a 1M-row dataset is a waste of an afternoon and quota.

**The video's actual output** [40:10]–[40:36], reproduced:

```
how many examples we have: 10 rows in total
total number of tokens:    564
minimum number of tokens in an example: 51
maximum:                                62
average:                                56
per line: line 1, line 2, line 4 … (ascending order)
top 10 largest examples: (shown)
```

#### 6.5.1 The gap this histogram exposes but the video never closes

The function above produces exactly the statistics you need to ask the next question — and then the video moves on to cost without asking it. **`max: 62` is the number that matters, and its meaning is: every one of these 10 examples fits, and nothing was ever truncated.** On a 62-token dataset you cannot observe the failure mode. On your dataset you can, and it is silent.

**What the video never mentions anywhere, in any section:** sequence length, padding, truncation, packing, or the per-example token limit.

| Question | Why it matters | What the video says | What to do |
|---|---|---|---|
| **Is there a per-example token limit?** | Vertex's supervised tuning rejects or truncates examples beyond a context bound, and the bound is not exposed as a knob you can set. | Nothing. | Find the documented limit for your model generation, then assert `max_len <= limit` locally and **refuse the upload** rather than discovering it at job time. `code/14_vertex_gemini_finetune.py` does exactly this: `--validate` prints a per-example length distribution and a `--max-example-tokens` guard, and `--upload` exits non-zero while any row is over it. |
| **Does a too-long example get truncated, or does the job fail?** | Truncation is the dangerous answer: the job succeeds, you are billed for the tokens, and the *target* turn may have been cut off — leaving an example whose assistant turn is half a sentence (§4.1 item 4). | Nothing. | Assume truncation until proven otherwise, and make the max-length assertion a hard stop. |
| **Is anything padded, and does the padding contribute loss?** | Not your concern on a managed service — you cannot see the collator. Unlike CS-13 §5, where you own the masking and the `IGNORE_INDEX`, here the vendor masks the prompt turns (§4.1 item 6) and pads as it sees fit. | Nothing. | Nothing to set. Just know that the loss number you see is not computed the way yours would be. |
| **Where does the length limit interact with the split?** | A length filter applied *after* the hold-out split changes the split's distribution; applied *before*, it changes your reported row counts and your cost estimate. | Nothing. | Filter first, then split, then report both counts (§4.5). |
| **Does packing exist here?** | CS-13 §7 and CH-12 §6 use packing to avoid wasting padding compute. On a per-token-billed managed service, packing would change your **bill**, not just your speed. | Nothing. | Do not assume it. If your examples average 60 tokens and the limit is thousands, you are being billed for sequences padded to some internal bucket — ask. |

**The honest summary:** every length decision that CS-13 asks you to make deliberately is made for you here, silently, by a collator you cannot inspect. That is the trade you accepted in the four clauses (§1). The one thing you *can* do is **assert on your side and refuse to upload** — which is exactly what `--max-example-tokens` and the `--upload` guard in `code/14_vertex_gemini_finetune.py` are for.

### 6.6 Stage 3b — the cost estimator (the video's second helper)

Described at [41:34]–[42:25]: *"I have a pricing. So Gemini price — Gemini 2.5 Flash pricing, Gemini 2.5 Pro and Gemini 2.5 Flash Lite… I took this pricing from the documentation itself. Then training cost in USD and training cost for the model."*

```python
# Reconstructed from the video walkthrough.
from dataclasses import dataclass

PRICING_USD_PER_1M_TUNING_TOKENS = {          # source: Vertex AI pricing page, via [22:29]
    "gemini-2.5-pro":        25.00,           # ← PRO TIER: see the Correction in §4.1
    "gemini-2.5-flash":       5.00,
    "gemini-2.5-flash-lite":  1.50,
}

@dataclass
class TuningCost:
    model: str
    total_tokens: int
    epochs: int = 1

    @property
    def price_per_1m(self) -> float:
        return PRICING_USD_PER_1M_TUNING_TOKENS[self.model]

    @property
    def billable_tokens(self) -> int:
        # epochs multiply the bill — the single most-forgotten line in this module.
        return self.total_tokens * self.epochs

    @property
    def training_cost_usd(self) -> float:
        return self.billable_tokens / 1_000_000 * self.price_per_1m

    def report(self) -> str:
        return (
            f"model={self.model}  tokens={self.total_tokens}  epochs={self.epochs}\n"
            f"rate=${self.price_per_1m:.2f}/1M  "
            f"billable={self.billable_tokens}  "
            f"training_cost=${self.training_cost_usd:.5f}"
        )

print(TuningCost("gemini-2.5-flash", total_tokens=564, epochs=1).report())
# model=gemini-2.5-flash  tokens=564  epochs=1
# rate=$5.00/1M  billable=564  training_cost=$0.00282
```

> **Beyond the video:** the estimator is the right *shape* and the wrong *scope*. Extend it into a full cost model before you run anything, because the number it prints is 0.02% of the real bill:
> ```python
> HOURLY_ENDPOINT_USD = 2.00        # ← VERIFY against the live Vertex pricing page
> HOURS_DEPLOYED      = 24 * 30.4   # a month of "I'll clean it up later"
> print(f"Training: ${0.00282:.5f}")
> print(f"Endpoint: ${HOURLY_ENDPOINT_USD * HOURS_DEPLOYED:,.2f}")   # $1,459.20
> print(f"Ratio:    {HOURLY_ENDPOINT_USD * HOURS_DEPLOYED / 0.00282:,.0f}x")
> # Ratio:    517,000x
> ```
> Any cost model for a managed fine-tune that does not have a *per-hour endpoint* line item is not a cost model; it is a marketing restatement.

### 6.7 Stage 8 — start the tuning job

```python
from google.genai.types import TuningDataset, CreateTuningJobConfig

training_dataset = TuningDataset(
    gcs_uri="gs://gemini-sft-custom/data.jsonl",
)

tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=training_dataset,
    config=CreateTuningJobConfig(
        tuned_model_display_name="example-sft-job",
        # ── NOT in the video; the knobs that matter live here. See §7. ──
        # hyper_parameters=...      # epoch_count, learning_rate_multiplier, batch_size, adapter_size
        # validation_dataset=...    # a second TuningDataset — set this. §4.5
    ),
)
print("Tuning Job:", tuning_job.name)
```

The notebook's own output for `client.tunings.tune(...)` carries the warning banner:

```
ExperimentalWarning: The SDK's tuning implementation is experimental,
and may change in future versions.
  tuning_job = client.tunings.tune(
```

**A real bug in the shipped notebook.** Several cells in `gemini_finetuning_clean.ipynb` raise:

```
NameError: name 'tuning_job' is not defined
```

because the `tune()` cell was re-run in a fresh session while the dependent cells (`print("Tuning Job:", tuning_job.name)`, the polling loop, and the inference cell) were executed before it. The notebook still ships the cells in that broken order. **Do not treat a notebook that ran once as a notebook that runs.** Re-execute top to bottom in a clean runtime before you trust it, and treat every `NameError` in a published notebook as a hint that the author's last session was not linear.

**Reconnecting to an existing job** — the pattern that makes the workflow resumable, and the one the video actually used for inference [52:20]:

```python
jobs = client.tunings.list()          # newest first [46:53]
for j in jobs:
    print(j.name)

tuning_job_name = jobs[0].name        # 0 = latest, 1 = second-most-recent …
tuning_job = client.tunings.get(name=tuning_job_name)
```

> *"latest job always you will get at a first place"* [47:03] — and the instructor relies on this ordering to reach his previously-trained model by index `[1]` [52:34]: *"I want the next job. So what I will do? I will put one over here."* **Indexing into `list()` is a bug waiting to happen**: a colleague's job, a retried job, or a test job moves your model to index 2. Store the job name returned by `tune()` in your own state and re-fetch by name.

### 6.8 Stage 9 — poll to a terminal state

The video's loop [47:38], with the states it tests [47:49]:

```python
import time

running_states = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}

while tuning_job.state in running_states:
    print("State:", tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)   # refetch — state is a snapshot
    time.sleep(60)                                          # 60s, not 1s: quota hygiene

print("Final State:", tuning_job.state)
print("MODEL:",    tuning_job.tuned_model.model)
print("ENDPOINT:", tuning_job.tuned_model.endpoint)
```

> *"It will print the state, then it will take that job from `client.tuning.get`, and then it will sleep for 60 seconds and then again it will check."* [48:16]

**Two defects in that loop, and the fix:**

| Defect | Symptom | Fix |
|---|---|---|
| **Non-terminal failure states exit the loop silently.** `JOB_STATE_FAILED`, `CANCELLED`, `EXPIRED`, `PARTIALLY_SUCCEEDED` are all "not in `running_states`", so the loop ends and the next line prints `tuning_job.tuned_model.model` — which is `None` for a failed job, raising `AttributeError: 'NoneType' object has no attribute 'model'`. | You see an `AttributeError` and spend twenty minutes debugging your SDK version instead of reading the job's error. | Use an explicit terminal set and branch on it. |
| **No timeout.** A job that hangs in `PENDING` for six hours polls forever. | A Colab cell that never returns. | Bound the loop and fail loudly. |

```python
# Production-shaped poller. Copy this instead of the video's loop.
TERMINAL_OK   = {"JOB_STATE_SUCCEEDED"}
TERMINAL_BAD  = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                 "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
RUNNING       = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
MAX_WAIT_S, POLL_S = 6 * 3600, 60

deadline = time.time() + MAX_WAIT_S
while True:
    tuning_job = client.tunings.get(name=tuning_job_name)     # always refetch the same name
    state = str(tuning_job.state).rsplit(".", 1)[-1]           # "JobState.JOB_STATE_RUNNING" → name
    print(f"[{time.strftime('%H:%M:%S')}] {state}")

    if state in TERMINAL_OK:
        break
    if state in TERMINAL_BAD:
        # Surface the vendor's own error before you raise: error is where the reason lives.
        raise RuntimeError(f"tuning job {state}: {getattr(tuning_job, 'error', None)}")
    if state not in RUNNING:
        raise RuntimeError(f"unexpected job state: {state}")
    if time.time() > deadline:
        raise TimeoutError(f"tuning job still {state} after {MAX_WAIT_S}s")
    time.sleep(POLL_S)

print("MODEL:   ", tuning_job.tuned_model.model)
print("ENDPOINT:", tuning_job.tuned_model.endpoint)
```

**What to watch while it runs** [48:40]–[49:33]:

> *"here they have given you this experiment. You can check out with this widget. See, the training is running… you can check out the complete loss evaluation. You can check everything over here through this particular dashboard."*

and on the TensorBoard option:

> *"even you can configure the TensorBoard also… you can open it inside the TensorBoard, that option also they are providing you."*

The metric names on that dashboard, as he reads them aloud [51:36]:

> *"evaluation fraction of correct next step prediction, evaluation number of prediction, evaluation total loss, training loss."*

**That is four series and they are all you get.** §12 tells you what they mean and what they cannot mean.

### 6.9 Stage 10 — inference against the tuned model

The video shows two client objects. The `v1beta1` client is for tuning; inference and job retrieval use a second client pinned to `v1` [notebook cell 24]:

```python
# The notebook's inference client. Note: api_version="v1", not "v1beta1".
client_v1 = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

response = client_v1.models.generate_content(
    model=tuning_job.tuned_model.endpoint,      # ← the ENDPOINT resource, per the video [54:13]
    contents="What happens if I factory reset my phone?",
)
print(response.text)
```

> **Correction — and this one is subtle enough that it deserves care.** At [54:13] the instructor says *"I'm calling `client.model.generate_content`, model `tuning_job.tuned_model.endpoint`"* and it works in his session. In the Vertex/GenAI SDK there are **two distinct inference paths**, and conflating them is a common source of "works in my notebook, 404s in prod":
>
> | Path | `model=` argument | Endpoint | Auth | Latency profile |
> |---|---|---|---|---|
> | **Tuned-model path** (Google-hosted serving) | `projects/{p}/locations/{l}/models/{id}@1` | none needed | standard `aiplatform.user` | shared serving, scales without you |
> | **Dedicated-endpoint path** | the endpoint resource name passed to a Vertex **`Endpoint.predict`** call | `projects/{p}/locations/{l}/endpoints/{id}` | `aiplatform.user` on the endpoint | your reserved replica, hourly billed |
>
> The notebook prints both resources [53:36] and then passes the *endpoint* to `generate_content`, which the SDK tolerates in the versions used. **Verify which path your SDK version takes**, and prefer being explicit: if you intend the dedicated endpoint, construct it with `aiplatform.Endpoint(...).predict(...)` so the billing and quota semantics are unambiguous; if you intend the managed tuned-model path, pass the `@version` model resource name. Whichever you pick, write it down in your runbook — the two paths have different cost profiles, and "which one am I on?" is not a question you want to answer during an incident.

**The demo question**, taken verbatim from the dataset [54:25]:

```python
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents="Most phones include a one-year limited warranty. Can you tell me why most of the phones give one year warranty?",
)
print(response.text)
```

The instructor's commentary [54:53]:

> *"this is not the pre-trained model, the Gemini model. This is the fine-tuned model actually. I given my data and I fine-tuned the model and I'm accessing it — I'm accessing through the endpoint."*

**Be adversarial about this demo.** The question is a paraphrase of training row 1 (`"How long does the warranty last on a new smartphone?"` → the one-year-warranty answer). A 10-row SFT job that has learned "when asked about warranty, say one year" is indistinguishable from a base model that already knew it — and Gemini 2.5 Flash absolutely already knew it, because the answer is a fact about the world, not a fact about the instructor's private business. **The demo does not evidence that fine-tuning worked.** §12.3 gives the test that would.

### 6.10 Stage 11 — teardown (the video's version, and the correct version)

The video [56:39]–[57:04]:

> *"if you want to do the cleanup guys, so for the cleanup, this is the code. So you can import this `aiplatform`. You can initiate this `aiplatform`. Then you can give the tuning job `tune_model.model`, whatever model you want to be removed. Then you can basically pass it to this `aiplatform.model` and then `model.delete`."*

```python
# ── The video's cleanup. Deletes the MODEL only. ──────────────────────────────
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)

MODEL_ID = tuning_job.tuned_model.model     # "projects/…/models/2360465647369977856@1"
model = aiplatform.Model(MODEL_ID)
model.delete()
print("Deleted model:", MODEL_ID)
```

```python
# ── The cleanup you should actually run. ──────────────────────────────────────
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)

MODEL_ID    = tuning_job.tuned_model.model
ENDPOINT_ID = tuning_job.tuned_model.endpoint

# 1. Release the deployment. Without this the endpoint keeps its capacity reserved
#    and, depending on the service, keeps billing.
endpoint = aiplatform.Endpoint(ENDPOINT_ID)
print("Deployed models:", endpoint.list_models())
endpoint.undeploy_all()

# 2. Delete the (now-empty) endpoint resource.
endpoint.delete()

# 3. Only then delete the tuned model artefact.
aiplatform.Model(MODEL_ID).delete()

print("Torn down:", MODEL_ID, ENDPOINT_ID)
```

**Order matters** and it is the reverse of intuition: undeploy → delete endpoint → delete model. Deleting the model first can leave the endpoint in a broken state that still bills, and some SDK paths will refuse to undeploy a model that no longer exists.

> **Beyond the video — the teardown guarantee.** A cleanup block at the bottom of a notebook is a *hope*, not a control. Make it structural:
> 1. **A budget alert with a hard cap.** Billing → Budgets & alerts → set a $50 (or ₹5,000) budget with alerts at 50/90/100% and, on supported account types, an automatic cap. This is the only control that survives you forgetting.
> 2. **A scheduled sweeper.** A Cloud Function on Cloud Scheduler that runs hourly, lists `tuningJobs` and `endpoints` older than N hours with a `ttl`/`purpose` label outside the allow-list, and undeploys them. Ten lines of Python and a Pub/Sub topic.
> 3. **A label convention.** Tag every endpoint with `owner`, `purpose`, `expires-at`. The sweeper reads the label; you sleep at night.
> 4. **`try/finally` in the notebook.** Even in a demo, wrap the inference block so a crash mid-cell still tears down.

---

## 7. Hyperparameters & Configuration — Every Knob

### 7.0 The headline finding about this section

**The video names zero hyperparameters.** Not one. It never says `epochCount`, never says `learningRateMultiplier`, never says `batchSize`, never says `adapterSize`. The `CreateTuningJobConfig` object it builds contains exactly one field — `tuned_model_display_name` — and the run proceeds on Vertex's defaults.

That is a defensible choice for a first tutorial (the defaults are sane) and an indefensible one for a practitioner, because **`epochCount` is a direct multiplier on your training bill** and `adapterSize` is a direct multiplier on what the model can learn. Everything in this section is therefore marked as coming from the Vertex tuning API rather than from the video.

> **Verification warning, stated once and applying to this entire section.** Vertex's supervised-tuning hyperparameter defaults, ranges and *names* have changed across the Gemini 1.0/1.5/2.0/2.5 generations, and the SDK field naming (`AdapterSize`, `ADAPTER_SIZE_ONE`) differs from the REST enum (`"ADAPTER_SIZE_ONE"`). **Every default and range below must be verified against the live "Tune Gemini models" documentation and the `TuningJob` REST reference before you rely on it.** What is durable is the *shape* of the argument — which knob does what, which direction is dangerous — and that is what this section teaches. Treat the numbers as a starting hypothesis, not a specification.

### 7.1 The master table

| Param | What it does | Vertex default | Safe range (practitioner) | Documented range | Too high → | Too low → | Where it goes |
|---|---|---|---|---|---|---|---|
| **`epochCount`** | Number of passes over the training rows. **Linearly multiplies your training token bill.** | `1` (verify) | **1–3.** Go to 5 only with a very large, very clean set. | Reported as 1–10; some families allow more. **Verify.** | Memorisation of the exact phrasings; loss → 0; the model parrots and fails on paraphrase (CS-13 §4.9). Bill scales linearly. | Underfitting: the new behaviour shows up inconsistently, or only on prompts near the training distribution. | `supervisedTuningSpec.hyperParameters.epochCount` (**string** in REST) / `hyper_parameters.epoch_count` in the SDK |
| **`learningRateMultiplier`** | Scalar on Vertex's internal default LR. **Not an absolute rate.** | `1.0` (verify) | 0.5–2.0 for most jobs; never touch it on a first run. | Reported as 0.0–4.0. **Verify.** | Loss spikes, then NaN, then a useless model; the job still "succeeds". | Training crawls; loss plateaus above where it should; you burn epochs instead. | `hyperParameters.learningRateMultiplier` (**number** in REST) |
| **`batchSize`** | Examples per gradient step. | Model-dependent; commonly reported as `4` for Flash-tier models. **Verify.** | 4–16. | Reported as 1–16 for some families. **Verify.** | With a small dataset, one step is a large fraction of your data; gradient noise per step up, effective step count down. May exceed memory on large `adapterSize`. | Noisy gradients; slower convergence per epoch; more steps for the same token budget. **Does not change your token bill.** | `hyperParameters.batchSize` |
| **`adapterSize`** | LoRA rank: how much capacity the update has. | `ADAPTER_SIZE_ONE` (rank 1) for several families. **Verify.** | **`ADAPTER_SIZE_FOUR` or `ADAPTER_SIZE_EIGHT`** for real tasks; rank 1 for pure style/format. | `1 / 4 / 8 / 16 / 32`, availability per model varies. **Verify.** | Overfits small datasets; higher serving memory; slower training. | Cannot represent the behaviour change; the tuned model looks like the base model with a slightly different tone. | `supervisedTuningSpec.hyperParameters.adapterSize` |
| **`validationDatasetUri`** | The held-out file. | Absent — Vertex splits automatically. | **Always set it explicitly.** ≥ 100 rows or ≥ 10%, whichever is larger. | n/a | Wasting good data on eval; eval not stratified. | Eval metrics are noise; you cannot detect overfitting. | `supervisedTuningSpec.validationDatasetUri` / `TuningDataset` |
| **`tuningMethod`** | Which objective to run. | `SUPERVISED_TUNING` | — | `SUPERVISED_TUNING`, `DISTILLATION`, `PREFERENCE_TUNING` (availability varies) | — | — | `tuningJob.tuningMethod` |
| **`baseModel`** | Which model to tune. | none — required. | Pin an explicit version, not a floating alias. | The supported list, which changes. | — | — | `tuningJob.baseModel` |
| **`tunedModelDisplayName`** | Human label. | none | A name encoding dataset version + date. | free text | Duplicate names make the console unusable. | — | `tunedModel.displayName` |
| **`outputUri`** (experimental eval) | Where autorater results are written. | none | A dedicated GCS prefix. | — | — | You lose the eval artefacts when the job record ages out. | `supervisedTuningSpec` / evaluation config |
| **`evaluationConfig.metrics`** | Autorater metrics to run post-training. | none (automatic metrics only) | One or two, for a first pass. | Enum names churn — **verify**. | Authoring cost per run; longer wall-clock; extra GCS output. | You ship on log loss alone (§12.2). | Evaluation config |
| **`exportLastCheckpointOnly`** | Whether intermediate checkpoints are exported. | Usually true | leave default | — | n/a | Intermediate checkpoints are rarely useful on Vertex because you cannot select one for serving in the common case — **verify**. | `supervisedTuningSpec` |

### 7.2 `epochCount` — the one knob that costs money

Three facts, in order of importance:

1. **`epochCount` multiplies the bill linearly.** `billable_tokens = total_tokens × epochCount`. If Vertex's defaults ever change from 1 to 3, your tuning bill triples with no code change on your side. Recompute your estimate from the job's own metric output (`totalBillableTokenCount`, if the job exposes it) rather than from your pre-flight estimate.
2. **The useful range for SFT is 1–3, and the video's own demo would be *harmed* by raising it.** With 10 rows and a constant system instruction, epoch 10 means the model has seen each row ten times — that is memorisation of 10 sentences, which is a lookup table, not a behaviour.
3. **You cannot early-stop.** There is no `earlyStoppingCallback` here; the job runs to completion and you pay for every epoch. The mitigation is to run epoch 1 first, evaluate, and only then decide whether epoch 3 is worth it — which costs the fixed ~20-minute overhead twice but saves the token bill, and the token bill is negligible, so in practice: **run 3 epochs, evaluate, and if it overfits, run 1.** Because training tokens are cheap and engineering time is not, the *cheap* knob to iterate on is epoch count, and the *expensive* knob is everything downstream of deployment.

### 7.3 `batchSize` and `learningRateMultiplier` — the knobs you should probably not touch

**`batchSize` does not change your token bill.** Every token is still read once per epoch. What it changes is:

| Effect | Mechanism |
|---|---|
| Step count | `steps = ceil(N_examples × epochs / batchSize)`. A batch of 4 on 1,000 rows × 3 epochs = 750 steps; a batch of 16 = 188 steps. |
| Gradient noise | Smaller batches → noisier gradients → sometimes better generalisation, always less stable. |
| Memory | Larger batches and larger adapters compete for the same memory. If a job OOMs during tuning, `batchSize` is the first thing to lower. |
| Interaction with LR | If you raise `batchSize` 4×, gradient variance drops and the effective step is larger; with a *fixed* multiplier you may want to raise the multiplier slightly. In practice Vertex's defaults assume its default batch size — change one or the other, not both. |

**`learningRateMultiplier` is a *multiplier*, and that has three consequences:**

1. **You cannot reason about it in absolute terms.** "Use 2e-4" is not a statement you can make about Vertex SFT. You can only say "1.0× of whatever Google chose", and you cannot read the underlying number.
2. **Its interaction with rank is the real story.** Small rank + high multiplier = the adapter's few parameters take large steps, which is the classic recipe for an adapter that memorises. Large rank + low multiplier = slow, stable, and often the best quality on a narrow task.
3. **A bad LR does not fail loudly.** The job reaches `JOB_STATE_SUCCEEDED`. The loss curve is the only warning, and you have to look at it (§12.4). **This is the "silent failure mode" of the whole platform.** The job state tells you the job ran; it does not tell you the job worked.

**The practical rule for a first Vertex job:**

```
epochCount            = 1        (then try 3, and compare)
learningRateMultiplier = 1.0     (default — do not touch)
batchSize              = default (do not touch)
adapterSize            = ADAPTER_SIZE_FOUR or ADAPTER_SIZE_EIGHT
validationDatasetUri   = your own explicit, stratified split
```

Change exactly one thing per job. The fixed overhead is 15–20 minutes [50:56], so a 2×2 experiment over two knobs is four jobs ≈ 80 minutes of wall-clock and pocket change in tokens. That is a cheap, well-designed experiment, and **not running it is the most common reason a team ships a tuned model that a rank sweep would have improved.**

### 7.4 Where hyperparameters go, in all three surfaces

**SDK (`google-genai`):**

```python
from google.genai.types import TuningDataset, CreateTuningJobConfig

tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=TuningDataset(gcs_uri="gs://gemini-sft-custom/train.jsonl"),
    config=CreateTuningJobConfig(
        tuned_model_display_name="cust-support-sft-v3-2026-09-27",
        # Field names below are SDK-version-sensitive — VERIFY against your installed version.
        # hyper_parameters=...            # dict or typed HyperParameters object
        # validation_dataset=TuningDataset(gcs_uri="gs://gemini-sft-custom/validation.jsonl"),
    ),
)
```

**REST** — the `supervisedTuningSpec` block from §4.3, repeated here because this is where it is used:

```json
{
  "displayName": "cust-support-sft-v3-2026-09-27",
  "baseModel": "gemini-2.5-flash",
  "supervisedTuningSpec": {
    "trainingDatasetUri": "gs://gemini-sft-custom/train.jsonl",
    "validationDatasetUri": "gs://gemini-sft-custom/validation.jsonl",
    "hyperParameters": {
      "epochCount": "3",
      "learningRateMultiplier": 1.0,
      "batchSize": 8,
      "adapterSize": "ADAPTER_SIZE_EIGHT"
    }
  }
}
```

**Console** — Vertex AI → Tuning → Create tuned model → Supervised fine-tuning. The same fields appear as form controls, and the console is the fastest way to *discover* the currently supported value set, because the dropdowns are generated from the live API. When the docs and the console disagree, **the console is right** — it is reading the same enum the API validates against.

### 7.5 Adapter size, tuning mode, and what "full fine-tuning" means here

The video mentions two tuning regimes [13:14] without naming a parameter:

> *"they are supporting two approaches. One they are first is the parameter efficient finetuning and second is the full fine tuning."*

| Mode | What it means mechanically | Cost profile | Quality profile | Can you get the weights out? |
|---|---|---|---|---|
| **PEFT / adapter tuning** (`adapterSize` 1–32) | `ΔW = BA` added to a fixed set of projection matrices. Base frozen. | Cheaper training (fewer trainable params), lower memory, faster. | Excellent for style, format, tone, and narrow distribution shift. Rank caps capacity. | No. Not exportable. |
| **Full tuning** | All base weights updated. | More expensive, slower, more memory. | Broader behavioural change; higher risk of catastrophic forgetting. | No. Not exportable — same as PEFT. |
| **Distillation** (separate `tuningMethod`) | Train a smaller/cheaper model on a larger model's outputs. | Different cost model entirely. | Task-specific compression, not behaviour alignment. | No. |

**The decision rule that actually matters here, and it is not the one people expect:**

```
Is your goal to change HOW the model says things?   → PEFT (adapterSize 4–8). Cheaper and sufficient.
Is your goal to change WHAT it says, broadly?       → full tuning, but re-read CS-13 §4.1 first:
                                                      SFT injects facts weakly and expensively.
                                                      RAG (CS-04) or continued pretraining (CS-12)
                                                      is usually the right answer.
Is your goal to get a portable artefact?            → Neither. Vertex cannot give you one.
                                                      Go to CS-16 / CS-17 / CS-23.
```

> **Beyond the video:** the reason "full tuning" on a closed model is closer to a *pricing tier* than a *technical claim* is that you have no way to verify it. You cannot inspect the delta between base and tuned weights, you cannot measure how many parameters moved, and the vendor has no incentive to update the full weight set if an adapter would do — it is strictly more expensive for them. Treat "full tuning" as "we will spend more compute and may make broader changes" and evaluate the *output*, not the label. If broad behavioural change is genuinely required, the honest path is an open-weight model where you can do it yourself and verify it (CS-23).

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 The two-question filter

Before anything else, answer these two. If either answer is "no", stop.

```
Q1. Do I need a Gemini-family model specifically?
      NO  → you have ~200 open-weight options and full control. CS-16/CS-17/CS-23.
      YES → go to Q2.

Q2. Can I run this myself, on hardware I control (owned, rented, or serverless)?
      YES → do that instead; you will get the weights, the eval harness and the
            exit strategy, and at small scale it is cheaper.
      NO  → Vertex SFT is a legitimate answer. Continue.
```

Q1 is asked first because it is the only question that makes the rest non-optional. If the requirement is "a good model with our tone", Gemini is one of six reasonable choices and the only one that takes away your artefact.

### 8.2 The full decision table

| Situation | Use Vertex SFT? | Instead use | Why |
|---|---|---|---|
| Enterprise already standardised on GCP; procurement and audit are the bottleneck | **Yes** | — | The managed service is the whole point: it fits the existing control plane. |
| You need Gemini's specific capabilities (long context, native multimodality, Google-integrated grounding) + your tone | **Yes** | — | You cannot get that combination by fine-tuning an open model. |
| No GPU quota, no GPU budget, no MLOps engineer | **Yes** | — | The alternative is not "self-host"; it is "don't do it". |
| Dataset < 1,000 examples and the goal is format/style | **Maybe** | First try prompt engineering + few-shot (CS-04) | 1k examples of style can often be achieved with a system prompt and 5 examples. Cheaper, instant, reversible. |
| Dataset < 1,000 examples and the goal is *knowledge* | **No** | RAG (CS-04) for retrievable facts; CS-12 for vocabulary | SFT injects facts weakly (CS-13 §4.1). You will spend money and get a confident wrong answer in a nice tone. |
| Sustained > 50 QPS to the tuned model | **No** | Self-host (CS-16/CS-17) or use the *base* model API | The endpoint's hourly cost is fixed regardless of throughput, but so is your inability to scale it economically past a certain point. §11.4. |
| You need the weights (merge, quantise, on-prem, air-gap) | **No** | CS-23 / CS-16 / CS-17 | Structural. Not negotiable, not workaroundable. |
| Data residency in a region Vertex tuning does not serve | **No** | A regional provider, or self-host in-region | Check the supported tuning locations for your generation **before** you build the pipeline. |
| You need a LoRA you can stack, swap, or serve with vLLM/SGLang/TGI | **No** | CS-23 | Vertex gives you an endpoint, not an adapter file. |
| You need per-request cost predictability with spiky traffic | **No** | Serverless self-host (scale-to-zero) | Hourly billing punishes spiky traffic; serverless billing rewards it. |
| You need to iterate on 20 dataset variants in a week | **Maybe** | Local first, Vertex for the final 2 | Each Vertex iteration costs ~20 minutes of fixed overhead **and** a deploy/undeploy cycle. Run the sweep locally on an open model of similar size, then confirm the winner on Vertex. |
| You are building a demo for a conference talk | **Yes** | — | It is genuinely the fastest path from JSONL to a live endpoint, and the cost for one day is a rounding error. **Undeploy afterwards.** |
| Regulated data with strict egress controls | **Verify first** | On-prem / VPC-SC-scoped self-host | "In my project" ≠ "never leaves the provider". Read the data-processing terms (§16.7). |

### 8.3 STOP conditions (hard signals this is the wrong tool)

1. **STOP if you cannot name the artefact you will hand to the next engineer.** If the deliverable is "endpoint `7655…`", you have no portable asset and no exit.
2. **STOP if the plan's cost model has no hourly line item.** You have not understood this service yet. §11.4.
3. **STOP if you cannot say which two examples were held out.** You do not know what your metrics mean. §4.5.
4. **STOP if the dataset has a constant system instruction repeated in every row.** 46% of the companion dataset's tokens are one frozen string [40:20, computed]. That is a data defect, not a tuning run.
5. **STOP if the goal is factual recall.** Wrong tool. CS-04 / CS-12.
6. **STOP if nobody owns the undeploy.** Assign it before the job starts, or it will not happen.
7. **STOP if the deployment will sit idle > 80% of the time.** The per-hour charge is indifferent to your traffic. §11.4.
8. **STOP if you have not checked the base model is still tunable this month.** The list changes; a plan written against a retired model fails at the last step.

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros (be specific — these are real)

| Pro | Why it is real | Evidence / magnitude |
|---|---|---|
| **No GPU, no VRAM arithmetic, no CUDA** | Training runs on Google's hardware. A CPU-only Colab runtime is sufficient [18:50]. | Removes the single largest barrier to entry in the whole handbook. |
| **No infrastructure to operate** | No vLLM, no TGI, no autoscaling group, no driver version matrix. | Hours-to-first-endpoint, not days. |
| **The newest base model, immediately** | You get Google's current model with no wait for open weights. | The video's model list [10:46] tracks the frontier. |
| **IAM, VPC-SC, CMEK, audit logging for free** | The tuned model inherits your project's controls. | This is the enterprise unlock — it is often the *only* reason the project is approved. |
| **Training cost is genuinely negligible at small scale** | 564 tokens × $5/1M = **$0.0028** [42:25]. | You can afford to run 100 experiments on the training side. |
| **Managed TensorBoard + experiment tracking included** | Loss curves and metrics with no MLflow/W&B to stand up [49:33]. | Real time saved on a small team. |
| **Automatic metrics without writing eval code** | `evaluation total loss`, `fraction of correct next step prediction` [51:36]. | Better than nothing, which is what many teams have. |
| **Two calls from JSONL to a live endpoint** | `tune()` then `generate_content()` [45:16], [54:13]. | Genuinely unbeatable on time-to-first-token. |
| **No data-engineering pipeline required** | A single `.jsonl` in a bucket is the whole input contract. | Fine for < ~100k rows. |

### 9.2 Cons

| Con | Why it hurts | Magnitude |
|---|---|---|
| **No artefact you can take with you** | Vendor lock-in is total and structural [3:35]. | Exit cost = re-doing the fine-tune on an open model. |
| **Idle endpoints bill per hour** | The default outcome of following this tutorial is a running endpoint you have forgotten about. | Order **$1–5/hour ≈ $1,000–$3,600/month**. Verify the rate. |
| **Hyperparameter surface is tiny** | Four knobs, **two of which you should not touch on a first run** (`learningRateMultiplier` and `batchSize` — §7.3). No schedulers, no warmup, no LoRA target-module selection. | You cannot reproduce a paper's recipe. |
| **No early stopping, no checkpoint selection** | The job runs to completion and bills for every epoch. | Costs money; risks overfitting with no automatic guard. |
| **Fixed ~15–20 minute overhead per job** [50:56] | Iteration is slow at small data scale, where compute is ~0. | 100 experiments = 33 hours of wall-clock. |
| **Metrics are log-loss and next-token accuracy** | Neither correlates well with task success on open-ended generation. | You will need your own eval harness anyway (§12). |
| **Not a full fine-tune in any verifiable sense** | You cannot audit what changed. | Quality claims are unfalsifiable from your side. |
| **Region and bucket constraints** | Bucket must be in a compatible location; not all regions serve tuning. | Blocks some EU/APAC deployments. |
| **Docs and SDK churn** | `v1beta1` vs `v1`; experimental warning on `tunings`; field renames between minor versions. | A pinned-version discipline is mandatory. |
| **Quota is invisible until it bites** | New projects have low tuning-job and endpoint quotas. | A CI pipeline that worked yesterday fails today. |
| **You cannot batch-tune, quantise, or distil onward** | No downstream compression path. | Cost-per-token stays at Google's price forever. |

### 9.3 Hard limitations (not fixable — plan around them)

| Limitation | Consequence | The only workaround |
|---|---|---|
| **Closed weights** | No export, ever. | Different model family (CS-16/17/23). |
| **No adapter export** | Cannot serve on your own infra. | — |
| **No tokeniser access** | Local token counting is impossible; you must call the API. | Sample-and-extrapolate (§6.5 correction). |
| **No loss curve control** | No `eval_steps`, no `save_steps`, no `logging_steps`. | Read the dashboard. |
| **No gradient/activation introspection** | Cannot debug *why* it failed. | Ablation on data, not on internals. |
| **No multi-adapter serving** | One model per endpoint. | Multiple endpoints = multiple hourly bills. |
| **No reproducible byte-identical rebuild** | Google's training run is not seeded or published. Two jobs on the same data differ. | Version the *data* and the *job ID*; accept non-determinism. |
| **No SLA on job completion time** | A job may queue for hours. | Do not put a tuning job on a critical path with a deadline. |

### 9.4 Silent failure modes (looks fine, is broken)

| Failure | What it looks like | Why it is silent | Detection |
|---|---|---|---|
| **Endpoint left deployed** | Everything works; the model answers. | Nothing in your application surfaces a billing signal. | Budget alerts; a scheduled sweeper listing `endpoints` (§6.10). |
| **Overfitting at `epochCount`=10** | Eval loss may even *fall* while the model memorises. | Nothing raises; the job reports `SUCCEEDED`. | Compare outputs on *paraphrases* of training prompts; measure verbosity and template rigidity (CS-13 §4.9). |
| **Automatic split put your best examples in validation** | Metrics look great. | You never see which rows were held out. | Pass an explicit validation file. §4.5. |
| **Constant system instruction dominates the gradient** | The model works and sounds right, but behaves identically to the base model. | The "it works" signal is a base-model capability, not a tuning gain. | **A/B against the untuned base model on the same prompts.** §12.3. |
| **`validationDatasetUri` points at a file that overlaps training** | Eval metrics are excellent, then production is mediocre. | No leakage check runs. | n-gram overlap check between train and validation (CS-13 §12). |
| **`generate_content` silently used against the base model** | The tuned model "works". | A typo in the model string falls back to a valid base model. | Log the resolved model ID on every request; assert it matches the tuned resource. |
| **Tuning job `FAILED` but loop exits "cleanly"** | You see `AttributeError: 'NoneType'` and blame the SDK. | The loop condition does not test for failure states. | The explicit terminal-state poller in §6.8. |
| **Data in the wrong region but the job runs from a cached copy** | Latency is fine in testing. | Cross-region reads may be permitted but slow/billed. | Assert the bucket location equals the job location in your pre-flight check. |
| **`tuned_model_display_name` reused across jobs** | The console shows three identical entries. | Names are not unique keys. | Encode dataset hash + date + model in the name. |
| **Deleting the model while the endpoint is still deployed** | Teardown "succeeded" but the bill continues. | `Model.delete()` returns success. | `endpoint.list_models()` after teardown; assert empty, then assert the endpoint is gone. |

> **Beyond the video:** the highest-value monitoring signal for this service is not latency or error rate — it is **`endpoint exists AND request_count == 0` over a 2-hour window**. Every managed-endpoint cost incident I am aware of in the wild has that shape: a perfectly healthy deployment that nobody called. Wire that one alert before you wire the p99 latency alert.

---

## 10. Exceptions, Edge Cases & Gotchas

### 10.1 Transcription noise — read this before copying any identifier

> **Correction:** the transcript of this video renders **every** occurrence of "Gemini" as **"Jimny"** (and occasionally "Gymni"), and **every** occurrence of "Vertex" as **"Vortex"**. It renders "GPD"/"GPT" where the speaker means "Gemini" [10:46], and "Sunonny" for "Sunny". If you search the transcript for `gemini-2.5-flash` you will find nothing; if you copy the string `"Jimny 2.5 flash"` into `base_model=` you will get a validation error.
>
> **The rule:** model IDs and resource names come from the console and the API reference, **never** from a transcript. The verified strings in this module (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash-lite`, `gs://gemini-sft-custom/data.jsonl`) come from the companion notebook's actual code cells, which are ground truth.

### 10.2 The billing setup is region-specific, and the video's version is India-only

> **Correction:** at [29:40] the instructor says *"you won't believe, you don't need to add any credit card and all, you just need to mandate it from your G Pay."* That describes a **UPI autopay mandate** on an India-billed Google Cloud account. It is not how GCP billing works in the US, EU, UK, or most of APAC, where a **credit card or bank account is mandatory** to create a billing account. The instructor's cost figures are also region-specific: *"so far I got 10 rupees of cost"* [29:17], *"just 5 to 10 rupees"* [30:35], *"10 to 15 rupees"* [30:38] — ₹5–15 ≈ **$0.06–$0.18**. That is plausible for a single small tuning job in a low-cost region and it is not a budget you can extrapolate to a production deployment, because the endpoint's hourly charge dominates everything he measures. **Check the pricing page for your billing region**, not the video's INR figures.

### 10.3 The cleanup that does not clean up

Covered in full in §2.3 and §6.10. The short version: deleting the `Model` does not release the `Endpoint`, and the video's teardown does only the former [56:39]. **This is gotcha #1 in this module.**

### 10.4 Your Colab identity and your GCP identity must match

> *"if you are logging to the GCP, please create a notebook with the same ID… otherwise you will face a issue… you will not be able to authenticate properly."* [23:38]

**Why this happens, which the video does not explain:** `auth.authenticate_user()` mints an OAuth token bound to *the Google account signed into Colab*. Every subsequent Vertex call is made as that identity, against the `project` you passed to `genai.Client()`. If the Colab account has no IAM role on that project, every call returns `403 PERMISSION_DENIED` — with a message that names the *wrong* principal relative to whatever you are signed into in the console tab beside it. This is the most confusing 20 minutes in the whole setup, and mismatched identities is the cause roughly 80% of the time.

**Also, the related trap the video misses:** even when the identities match, the OAuth token may not carry a **quota project** that is billing-enabled. If the smoke test fails with a quota-project error rather than a permission error, set it explicitly:

```python
# If you hit "quota project" errors with Colab user credentials:
import google.auth
from google.auth import default
creds, _ = default()
creds = creds.with_quota_project(PROJECT_ID)
```

### 10.5 Region alignment between bucket and job

The tuning job runs in `LOCATION`. The dataset lives in a bucket whose location is chosen at bucket-creation time and **cannot be changed**. Mismatches produce either a hard rejection or a silent cross-region read with egress cost and latency. **Verify before you create the bucket**, because recreating a bucket means re-uploading and re-pointing.

| Job location | Acceptable bucket location | Notes |
|---|---|---|
| `us-central1` | `us-central1`, or a US multi-region (`US`) | The video's configuration, and the safest default. |
| `europe-west4` | `europe-west4`, or `EU` | Needed for EU-resident data. |
| `asia-south1` | `asia-south1`, or `ASIA` | Verify that tuning is served in your chosen region at all. |

### 10.6 The `HttpOptions` version trap

The notebook constructs **two clients** with different `api_version`s: `v1beta1` for tuning and `v1` for inference [notebook cells 10 and 24]. Reasons this matters:

- **Tuning ships on `v1beta1` first.** If you pin `v1` and call `client.tunings.tune(...)`, you may get `AttributeError: 'Tunings' object has no attribute 'tune'` or a 404 on the endpoint — depending on how far the feature has graduated.
- **Inference and job retrieval are stable on `v1`.** If you pin `v1beta1` for everything, you may hit beta-only response shapes in your parsing code.
- **The graduation is one-directional.** A feature moves `v1beta1 → v1` and does not move back. Code that pinned `v1beta1` keeps working for a while and then breaks when the beta surface is retired.

**Practical discipline:** one module-level constant per client, a comment naming the feature that forced the version, and a smoke test in CI that runs both. When the `AttributeError` appears, the fix is a one-line version bump, and you want that to be obvious rather than mysterious.

### 10.7 The `ExperimentalWarning`

```
ExperimentalWarning: The SDK's tuning implementation is experimental, and may change in future versions.
```

The SDK says this about itself, in the notebook's captured output for the `tune()` call. Experimental means: **the API contract is not covered by a stability guarantee, and a minor version bump can change the signature.** Mitigations: pin the exact version (`google-genai==1.62.0`), keep the tuning call behind a thin wrapper of your own so the blast radius of a signature change is one function, and re-run a smoke test on every dependency bump.

### 10.8 `jobs[0]` is not your job

Covered in §6.7. The list is newest-first *for your project's visible jobs*, which includes every job any colleague created, every retried job, and every test job. **Persist the job name returned by `tune()`.** If you must re-find a job, match on `displayName` plus a creation-time window, not on index.

### 10.9 A 10-row dataset is a smoke test, not a model

The arithmetic, once, so it is unarguable:

| Quantity | Value |
|---|---|
| Rows | 10 |
| Total tokens | 564 |
| Tokens per parameter in a 7B model | 564 / 7,000,000,000 = **8.1 × 10⁻⁸** |
| Tokens per parameter in typical SFT (CS-13) | ~10⁻² to 10⁻¹ |
| Ratio | **~10⁵–10⁶× less data than a normal SFT run** |

The tuned model in the video is a **plumbing demonstration**. It proves the API path works. It cannot prove that fine-tuning improves anything, because there is not enough signal in the data to distinguish a tuned model from a prompted one. Anyone who shows you a 10-row managed fine-tune as evidence of quality is showing you a working API call.

### 10.10 Non-determinism you cannot control

Two jobs with identical data, identical hyperparameters, and the same base model will produce **different** tuned models. There is no seed you can set, no dataset-order control, no way to prove a regression came from your change rather than from Google's run. **The implication for A/B testing:** you cannot attribute a quality delta to a data change unless the delta is much larger than run-to-run variance, and you cannot measure run-to-run variance cheaply because each measurement is a full job. Run the same configuration twice, once, early, purely to calibrate your noise floor. Most teams never do this and then argue about 2-point eval differences that are entirely noise.

### 10.11 The evaluation-output bucket needs its own permissions

If you enable autorater evaluation with `OUTPUT_GCS_URI` (the commented-out line in the notebook's config cell), the Vertex *service agent* — not your user account — writes to that bucket. Granting `objectAdmin` to yourself is not sufficient; you need the service agent to have write access on the destination. Symptoms: the tuning job succeeds, the eval step fails, and the job state is `PARTIALLY_SUCCEEDED`. The usual grant is on the project's Vertex AI service agent (`service-<PROJECT_NUMBER>@gcp-sa-aiplatform.iam.gserviceaccount.com`).

### 10.12 Deleting the tuning job does not delete the model

A `TuningJob` is a *record*. Deleting it removes the metrics, the experiment linkage, and the audit trail — and **leaves the tuned `Model` and the `Endpoint` in place, billing**. If you are cleaning up, clean up the three resources in the order given in §6.10, and keep the job record: it is the only place your metrics live.

> **Beyond the video:** keep a JSON sidecar of every job you run — job name, display name, base model, dataset hashes, hyperparameters, final metrics, endpoint ID, deploy timestamp, undeploy timestamp. Twelve months later, when someone asks "which tuned model is in production and what data made it", this file is the answer, and reconstructing it from the console is hours of work.

---

## 11. Cost, Compute & Memory

> **Standing verification warning.** Every dollar figure in this section is either (a) quoted from the video, where it is dated 2025-era and region-specific, or (b) an **estimate** built from an order-of-magnitude model and labelled as such. Vertex pricing changed at least four times between the video and the publication of this module. **Re-derive every number against the live pricing page for your region before you commit budget.** What is durable here is the *structure* of the cost model — which term dominates — and that structure has not changed.

### 11.1 The four cost lines, and which one is real

| # | Cost line | Basis | Scale in the video's demo | Scale at production | Verdict |
|---|---|---|---|---|---|
| 1 | **Tuning (training tokens)** | $/1M tokens read × epochs | **$0.00282** [22:53] | $10–$300 per job | Trivial |
| 2 | **Endpoint deployment** | **$/hour the deployment exists** | not measured | **$700–$3,700 / month** | **Dominant** |
| 3 | **Tuned-model inference** | $/1M input + $/1M output tokens, at a premium over base | ~$0.00002 | **$3–$12 / month** at the §4.6 volume (5M input + 1M output); ~$50/month at 4–5× that traffic — still an order of magnitude below line 2 | Small |
| 4 | **Artefact storage** | No published hourly charge identified for the Gemini tuned-model artefact; **verify** | $0 | $0 (unpublished) | Negligible/none |

**The ratio between line 1 and line 2 at demo scale is roughly 500,000 : 1.** Every budgeting mistake in this module is a failure to notice that ratio.

### 11.2 Line 1 — training cost, derived properly

$$C_{train} = \frac{N \times \bar{t} \times E}{10^6} \times P_{tune}$$

with `N` = examples, `t̄` = mean tokens per rendered row, `E` = epochCount, `P_tune` = $/1M tuning tokens.

**Rates quoted in the video [22:29], verbatim:**

> *"Gemini 2.5 Pro around $25 for the 1 million token, 2.5 Flash $5, 2.5 Flash Lite $1.5."*

| Model | `P_tune` (video) | Verification status |
|---|---|---|
| `gemini-2.5-pro` | $25.00 / 1M | **Suspect** — the Pro tier's tuning support is itself in question (§4.1). |
| `gemini-2.5-flash` | $5.00 / 1M | Plausible order of magnitude. **Verify.** |
| `gemini-2.5-flash-lite` | $1.50 / 1M | Plausible order of magnitude. **Verify.** |

**Worked examples (arithmetic shown):**

| Scenario | N | t̄ | E | Tokens | Rate | Cost |
|---|---|---|---|---|---|---|
| The video's demo | 10 | 56 | 1 | 564 | $5/1M | **$0.00282** |
| The video's demo, 3 epochs | 10 | 56 | 3 | 1,692 | $5/1M | $0.00846 |
| Small production set | 5,000 | 400 | 1 | 2,000,000 | $5/1M | **$10.00** |
| Mid production set | 50,000 | 400 | 3 | 60,000,000 | $5/1M | **$300.00** |
| Large production set | 500,000 | 400 | 1 | 200,000,000 | $5/1M | **$1,000.00** |

`(5,000 × 400 × 1) / 1e6 × 5 = 10.00` — that is the whole formula.

**Three observations that change how you plan:**

1. **Even the "large" row is cheap next to a month of endpoint.** $1,000 of training is a *one-time* cost; $1,459/month of endpoint is *recurring*. The training line is the one you should be happy to spend on.
2. **`epochCount` is the only multiplier you have.** Going from 1 to 3 triples line 1 and changes line 2 not at all — so **epoch count should be chosen for quality, not for cost**, at every realistic dataset size. The cost argument against more epochs appears only above ~10⁹ tokens.
3. **The token count includes the system instruction in every row.** In the companion dataset that is ~46% of the bill [computed, §4.2]. If your system instruction is 2,000 tokens and your examples are 300, you are paying 87% of your tuning bill to re-read the same paragraph.

> **Beyond the video:** the *practical* floor on a tuning job is not the token cost, it is **the fixed ~20-minute overhead and any minimum-job billing the platform applies.** If Vertex enforces a minimum billable token count per job — and several managed tuning services do, in the 10⁵–10⁶ token range — then a 564-token dataset is billed as if it were much larger. **Verify whether a minimum applies**, because it changes the "it's basically free to experiment" conclusion at small scale. The instructor's own "10 to 15 rupees" [30:38] is consistent with a small minimum rather than with the strict token arithmetic, which would predict under one rupee.

### 11.3 Line 3 — the serving premium

A tuned model's per-token inference price is generally **higher** than the base model's, on both input and output. The reasons are mundane: the deployed artefact is larger (base + adapter), it is served from reserved rather than shared capacity in the dedicated-endpoint path, and the vendor prices the incremental value.

| Serving option | Basis | Approximate relationship to base rate | Verify? |
|---|---|---|---|
| Base model on the same endpoint infrastructure | $/1M in + $/1M out | 1× (reference) | — |
| **Tuned model, managed path** | $/1M in + $/1M out, premium | Commonly **2–10×** the base rate | **Yes — the multiplier has moved a lot.** |
| **Tuned model, dedicated endpoint** | Endpoint $/hour + $/1M in + $/1M out | Hourly charge dominates at low traffic | **Yes** |

**Worked monthly serving estimate** (assume a 3× premium over a base rate of $0.30/$2.50 per 1M in/out — *these are illustrative, verify*):

| Traffic | Input tokens/mo | Output tokens/mo | Per-token cost @3× | Endpoint cost | **Total** |
|---|---|---|---|---|---|
| Hobby (1k requests, 500 in / 300 out) | 0.5M | 0.3M | $1.35 | $1,459 | **$1,460** |
| Internal tool (100k requests, 800/400) | 80M | 40M | $312 | $1,459 | **$1,771** |
| Product (10M requests, 1,000/500) | 10,000M | 5,000M | $39,000+ | (multiple replicas) | **$40,000+** |

The shape is unmistakable: **at low traffic the hourly charge is ~100% of the cost; at very high traffic it becomes irrelevant.** §11.4 finds the crossover.

### 11.4 The break-even, worked

**The question:** at what utilisation does a dedicated tuned endpoint stop being obviously wasteful?

| Deployment | Cost basis (estimate, verify) | Monthly |
|---|---|---|
| Tuned Gemini endpoint (dedicated) | ~$2.00/hour | **$1,459** |
| Self-hosted open 7–8B on 1× A100-80GB, on-demand | ~$2.00/hour | $1,459 |
| Self-hosted open 7–8B on 1× L4 / A10G, on-demand | ~$0.80/hour | $584 |
| Self-hosted open 7–8B on serverless (scale-to-zero, e.g. Modal/RunPod Serverless) | ~$0.0002–$0.0006 per request or ~$0.50–$1.00 per GPU-hour while active | **$50–$300** at spiky low traffic |
| Reserved 1-year 1× A100 (self-host, open weights) | ~$1.00–$1.50/hour amortised | $730–$1,095 |

**Read the table as three regimes:**

| Regime | Monthly requests | Right answer |
|---|---|---|
| **< ~500k requests/month** | Low | **Do not deploy a dedicated endpoint.** Use the base model API, or the managed tuned path if it avoids a dedicated deployment, or self-host scale-to-zero. A dedicated endpoint costs 25–100× too much here. |
| **~500k – 20M requests/month** | Medium | The regimes are comparable on price. Decide on *control*: do you need the weights, the observability, the ability to quantise? If yes, self-host. If you need Gemini specifically, pay the endpoint. |
| **> ~20M requests/month** | High | **Self-host, if the task fits an open model.** At this volume you are paying for multiple replicas either way, and with open weights you control quantisation (CS-10/CS-11), batching and scheduling — which is where the 3–10× cost wins live. |

> **Beyond the video — the decision rule, stated for an interview:** *Managed fine-tuning wins on time-to-first-model and on organisational fit. It loses on unit economics the moment the endpoint is deployed, because you are paying for reserved capacity you may not use. The correct comparison is never "training cost vs training cost"; it is "monthly endpoint cost vs monthly self-hosted GPU cost", and that comparison is a traffic question, not a machine-learning question.*

### 11.5 The budget controls you must have before you run anything

```bash
# 1. A budget with alerts — create in the console: Billing → Budgets & alerts
#    Amount: $50 (or your local equivalent) with alerts at 50% / 90% / 100%.
#    Scope it to the project, filtered to the Vertex AI service if you want precision.

# 2. Verify the budget is actually scoped to the right project
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID

# 3. A monthly cost report you actually read
gcloud billing accounts list
# Then in BigQuery: enable the billing export and query by service + SKU.
# The SKU that matters is the endpoint/online-prediction node-hour line, not the tuning line.
```

**The one query to write** (billing export to BigQuery, standard schema — column names drift, verify):

```sql
-- Top Vertex AI cost drivers, last 7 days, by SKU.
-- This is how you find the endpoint you forgot about.
SELECT
  sku.description          AS sku,
  SUM(cost)                AS cost_usd,
  SUM(usage.amount)        AS usage_amount,
  usage.unit               AS unit
FROM `PROJECT.dataset.gcp_billing_export_v1_BILLING_ACCOUNT_ID`
WHERE invoice.month = FORMAT_DATE('%Y%m', CURRENT_DATE())
  AND service.description LIKE '%Vertex%'
GROUP BY sku, unit
ORDER BY cost_usd DESC
LIMIT 20;
```

### 11.6 Storage cost of the adapter — the honest answer

The task a reader will ask: "what does it cost to store the tuned artefact?"

**For Vertex Gemini tuned models, no separate published storage line item has been identified.** The tuned model is stored by Google as part of the managed service, and the video gives no storage figure. The line that shows up on your bill is the **endpoint**, not the artefact.

For comparison, and to show why the question is the wrong question:

| Platform | Storage/retention model | Order of magnitude |
|---|---|---|
| **Vertex Gemini** | Not separately published for the tuned artefact; the artefact persists with the `Model` resource. **Verify.** | ~$0 |
| **OpenAI fine-tuning** | Historically a per-hour hosting charge for a fine-tuned model *even when not serving*. **Verify the current model** — OpenAI restructured this at least twice. | Order $1–3 per model per hour when charged |
| **Self-hosted LoRA adapter (CS-23)** | 60 MB of object storage | 60 MB × $0.02/GB-month ≈ **$0.0012 / month** |

**The point of that table:** if you are worried about storage, you are worried about the wrong term by six orders of magnitude. A rank-32 adapter is ~60 MB [§4.4]. Object storage for a thousand of them costs less than a coffee. The cost is always the *deployment*, never the *artefact*.

### 11.7 A complete worked example: the reader's own project

**Scenario:** a 40-person SaaS company wants a support assistant that answers in house style, using their own product documentation. 8,000 curated support conversations. Mean 600 tokens per row (300 in, 300 out). They already have GCP.

| Stage | Quantity | Rate | Cost |
|---|---|---|---|
| Data preparation | 3 engineer-days | — | ~$2,400 (loaded) |
| Split + validation + upload | 1 engineer-hour | — | ~$100 |
| Tuning run 1 (`epochCount=1`, `adapterSize=4`, 4.8M tokens) | 4.8M tokens | $5/1M | **$24.00** |
| Tuning run 2 (`epochCount=3`, 14.4M tokens) | 14.4M tokens | $5/1M | **$72.00** |
| Tuning run 3 (data fix + `adapterSize=8`, `epochCount=3`) | 14.4M tokens | $5/1M | **$72.00** |
| Eval harness + LLM-as-judge runs | 5,000 judge calls | ~$0.001 each | **$5.00** |
| **Total training-side spend** | | | **~$173** |
| Endpoint, month 1 (left up while evaluating) | 720 hours | $2.00/hr | **$1,440** |
| Endpoint, month 2 (production, 60 QPS peak, ~300k req/mo) | 744 hours | $2.00/hr | **$1,488** |
| Inference tokens, month 2 | 180M in / 60M out | @3× base | ~$1,000 |
| **Total month 2 run rate** | | | **~$2,500/month** |

**The lesson:** the fine-tuning itself is a rounding error. Three full experiments and an eval harness cost less than one day of an engineer's time. The decision that determines whether this project is a $50/month project or a $2,500/month project is **the deployment architecture**, and it was decided — usually implicitly — in the first week.

**The counterfactual:** self-host the same behaviour on a 7B open model with vLLM on one L4, scale-to-zero serverless at night. ~$600–$900/month, plus the ops tax. Or: keep the base Gemini API and solve the style problem with a 20-example few-shot prompt and RAG over the docs, at ~$200/month. **Both are legitimate answers and neither requires a tuned endpoint.** The tuned endpoint is right only if the tuned model measurably beats both, at the same traffic, by more than the delta.

---

## 12. Evaluation — How To Know It Worked

### 12.1 What Vertex gives you out of the box

Read aloud by the instructor from the experiment dashboard [51:36]:

> *"evaluation fraction of correct next step prediction, evaluation number of prediction, evaluation total loss, training loss."*

| Metric (dashboard label) | What it actually measures | What it does NOT measure | API field (verify) |
|---|---|---|---|
| **Training loss** | Cross-entropy on the training rows, at the point the series was sampled. | Anything about generalisation. | `metrics` on the job / TensorBoard series |
| **Evaluation total loss** | Cross-entropy on the held-out slice. | Task success. Lower is better *up to a point*; past that it is overfitting. | likewise |
| **Fraction of correct next step prediction** | Teacher-forced token accuracy: given the true prefix, how often is the argmax the true next token? | Whether a *generated* full response is correct. Teacher forcing hides error compounding. | likewise |
| **Evaluation number of prediction(s)** | How many predictions the eval ran over — i.e. **the size of the held-out set**. | — | likewise |

> **Correction — read this before you trust any of those four numbers.** The instructor presents them as *"everything you will be able to get over here"* [51:43], which is true, and leaves the reader with the impression that they constitute an evaluation. They do not. Three specific problems:
> 1. **`fraction of correct next step prediction` is a teacher-forced metric.** It is computed with the ground-truth prefix available at every step. A model that produces a fluent, confident, *wrong* answer scores highly, because at each individual token the true continuation is usually the most likely one. Format-learned models score ~0.9+ on this while being wrong on content.
> 2. **The held-out set is a default split you did not choose and cannot see.** For the video's 10-row dataset that is 2 examples [§4.5]. "Evaluation total loss: 0.31" on 2 examples is not a measurement.
> 3. **No metric here compares against the base model.** A decrease in eval loss tells you the model got better at predicting *your* data. It does not tell you the model got better *at the task*, because the base model was already good at predicting your data — your data is ordinary English about smartphones.

### 12.2 The four ways the automatic metrics lie to you

| Lie | Mechanism | What you see | What is true |
|---|---|---|---|
| **"Low eval loss means it learned the task"** | Loss is next-token CE over your distribution. The base model's CE on your data is already low. | Eval loss 0.4 → 0.2 | Most of the drop is *style* (the constant system instruction's phrasing leaking into outputs), not task competence. |
| **"High fraction-correct means correct answers"** | Teacher forcing removes error compounding. | 0.94 | Generated answers may be 60% correct; the metric never sees the generation. |
| **"Eval loss fell, so 10 epochs was right"** | With a small dataset, eval loss can fall while the model memorises surface forms that happen to appear in both splits. | Monotonic improvement | Overfitting. Detect by prompt-mutation testing (§12.4). |
| **"The metrics are reproducible"** | There is no seed; each job is a different run. | Two jobs, two numbers | The delta may be noise. Calibrate the noise floor (§10.10). |

### 12.3 The evaluation you must write yourself

**Step 0 — the only comparison that matters.** Before you evaluate the tuned model, evaluate **the base model** on the same prompts. If the base model already does the thing, your fine-tune is decoration.

```python
"""
eval_ab.py — A/B the tuned model against the untuned base model on YOUR prompts.

This is the minimum viable evaluation for a managed fine-tune. It answers the one
question the Vertex dashboard cannot: did tuning change the behaviour you care about?
"""
# Marked as a FIXTURE: run this on any managed fine-tune, Vertex/OpenAI/otherwise.
import json, os, time
from statistics import mean, pstdev

import google.genai as genai
from google.genai.types import HttpOptions

PROJECT_ID  = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION    = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BASE_MODEL  = "gemini-2.5-flash"
TUNED_MODEL = os.environ["TUNED_MODEL_ENDPOINT"]     # .../endpoints/…

PROMPTS_PATH = "eval_prompts.jsonl"     # YOUR held-out set — never trained on
N_SAMPLES    = 1                        # raise to 3 for stability; costs 3× tokens

client = genai.Client(
    vertexai=True, project=PROJECT_ID, location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

def ask(model: str, prompt: str) -> str:
    """One call, no retries hidden — a failure must be visible in the report."""
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()

def load_prompts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def score(prompt: str, answer: str, expect: dict) -> float:
    """Return 0.0 / 0.5 / 1.0. Replace with a judge call for open-ended tasks (§12.5)."""
    a = answer.lower()
    if expect.get("must_contain_any") and not any(
        s.lower() in a for s in expect["must_contain_any"]
    ):
        return 0.0                                     # required content missing
    if expect.get("must_not_contain") and any(
        s.lower() in a for s in expect["must_not_contain"]
    ):
        return 0.0                                     # forbidden content present
    if expect.get("max_words") and len(a.split()) > expect["max_words"]:
        return 0.5                                     # violated a style constraint
    return 1.0

def main() -> None:
    prompts = load_prompts(PROMPTS_PATH)
    rows = []
    for item in prompts:
        p = item["prompt"]
        base_scores, tuned_scores = [], []
        for _ in range(N_SAMPLES):
            base_scores.append(score(p, ask(BASE_MODEL,  p), item.get("expect", {})))
            tuned_scores.append(score(p, ask(TUNED_MODEL, p), item.get("expect", {})))
            time.sleep(0.2)                            # be polite to the quota
        rows.append({
            "prompt":  p[:120],
            "base":    round(mean(base_scores), 3),
            "tuned":   round(mean(tuned_scores), 3),
            "delta":   round(mean(tuned_scores) - mean(base_scores), 3),
        })

    base_mean  = mean(r["base"]  for r in rows)
    tuned_mean = mean(r["tuned"] for r in rows)
    deltas     = [r["delta"] for r in rows]

    print(f"n_prompts={len(rows)}  base={base_mean:.3f}  tuned={tuned_mean:.3f}  "
          f"delta={tuned_mean - base_mean:+.3f}  sd(delta)={pstdev(deltas):.3f}")
    wins   = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    print(f"per-prompt: {wins} wins, {losses} losses, {len(rows)-wins-losses} ties")

    # The decision rule. A delta smaller than the run-to-run noise floor is not a win.
    if tuned_mean - base_mean > 0.05 and wins > 2 * max(losses, 1):
        print("VERDICT: ship it.")
    elif tuned_mean - base_mean < 0:
        print("VERDICT: the tuned model is WORSE. Do not ship. Check §14.")
    else:
        print("VERDICT: inconclusive — the base model already does this. "
              "Consider prompt engineering (CS-04) instead of a tuned endpoint.")

    with open("eval_report.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
```

**Why each design choice in that script exists:**

| Choice | Reason |
|---|---|
| Evaluates base **and** tuned in the same run | Eliminates any drift in your judge, your prompt file, or the API between two separate runs. |
| `expect` block per prompt | Forces you to write down what "correct" means per item, which is where most eval projects discover their task is underspecified. |
| Content checks *and* style checks | Format compliance and refusal rate are the two metrics that move first (CS-13 §12). A model that gets content right and style wrong is not shippable. |
| `sd(delta)` reported | Without a variance estimate you cannot tell a win from noise, and on managed services you cannot cheaply re-measure (§10.10). |
| Prints a verdict | An eval that ends in a table nobody reads is not an eval. |
| Writes `eval_report.jsonl` | The report is the artefact you attach to the model card, not the console screenshot. |

### 12.4 Prompt-mutation testing (the overfitting detector)

The dashboard can show a falling eval loss while the model has learned the *template* rather than the *task*. The test is to mutate the prompt along axes your training data never varied:

| Mutation | What it detects | Passing behaviour |
|---|---|---|
| Ask in a different language | Language overfitting | Answers in the asked language, or declines cleanly |
| Drop the system instruction | Dependence on the exact system prompt | Still answers in house style (the style is in the weights) — or, if it collapses, the style was never in the weights |
| Add leading/trailing whitespace and newlines | Tokenisation brittleness | No change |
| Rephrase the question in three ways | Surface-form memorisation | Consistent answers |
| Ask an out-of-scope question (the laptop question, row 9) | Refusal behaviour | Clean, brief refusal |
| Ask for a longer answer than any training example | Verbosity ceiling learned from data | Reasonable length, not truncation |
| Ask two questions in one turn | Multi-turn handling | Handles it or asks for clarification |

**If the model passes 1–4 and fails 5, you have trained a template-follower.** That is a real product defect, and it is invisible in every metric Vertex shows you.

### 12.5 LLM-as-judge, for open-ended outputs

When the correct answer is a paragraph rather than a string, use a judge — and use it correctly.

| Practice | Why | The failure it prevents |
|---|---|---|
| **Pairwise, not pointwise** | Absolute scores from a judge drift; comparisons are more stable. | Score inflation over a long run. |
| **Swap positions and average** | Judges have a measurable position bias. Published audits put it at 10–15 points of win rate. | A "win" that is an artefact of ordering. |
| **Blind the model identity** | A judge told "this is the fine-tuned model" is a different judge. | Expectation bias. |
| **Use a *different* family as judge** | Judging your own family's output with its own family is self-preference. | Self-preference bias. |
| **Hold out the judge's prompts from training** | Otherwise you are measuring memorisation. | Contaminated eval. |
| **Report effect size + CI, not just a win rate** | 54% with n=50 is noise. | Shipping on a coin flip. |

> **Beyond the video:** Vertex exposes autorater-style evaluation through an evaluation configuration (the commented-out `EvaluationConfig`/`OutputConfig`/`Metric` imports in the notebook's import cell). **The metric names and configuration shape for this feature have churned heavily and must be verified against the current docs.** The strategic point is that whether you use Vertex's autorater or your own judge, the judge is the *same* kind of instrument: a model scoring a model. Budget for it, version its prompt, and store its outputs. A judge prompt is a production artefact and belongs in git next to your training data.

### 12.6 The evaluation protocol, end to end

```
1. Freeze a held-out set BEFORE you train. 200–1,000 prompts. Never trained on.
2. Run the BASE model on it. Record scores. This is your null hypothesis.
3. Train with an explicit validationDatasetUri (§4.5).
4. Read the Vertex dashboard metrics AS A SMOKE TEST ONLY:
      training loss should fall; eval loss should fall then flatten.
      If eval loss rises while train loss falls → overfitting → reduce epochCount.
5. Run eval_ab.py against the tuned endpoint. Compare to step 2.
6. Run prompt-mutation testing (§12.4). Any failure disqualifies a promotion.
7. Run the regression suite: 50 frozen prompts asserting PROPERTIES
      (parses as JSON / contains no refusal phrase / ≤ N words / language matches).
8. Run the refusal suite: 200 benign in-domain prompts. Measure refusal rate.
      A tuned model that refuses 15% of in-domain questions is a product outage.
9. Only then promote. And then UNDEPLOY the evaluation endpoint.
```

**The two counter-metrics that catch what accuracy misses:** refusal rate on benign in-domain prompts (goes up with over-training) and format-compliance rate (goes down with under-training). Track both on every candidate. A model with 95% task accuracy and a 20% refusal rate is worse in production than one with 85% and 2%.

---

## 13. Comparison Tables

### 13.1 Three managed / semi-managed routes, head to head

| Dimension | **Vertex AI (Gemini)** | **OpenAI fine-tuning** (CS-18) | **Open-weight self-host** (CS-16/17/23) |
|---|---|---|---|
| **Data schema** | `systemInstruction` + `contents[].parts[].text`, `role: "model"` | `messages[].role/content`, `role: "assistant"` | Whatever your framework expects: Alpaca, ShareGPT, ChatML, OpenAI chat (CS-13 §4.2) |
| **File format** | JSONL, in **GCS**, in a compatible region | JSONL, uploaded via Files API | Any local file; you own the storage |
| **Split control** | Explicit `validationDatasetUri`, else implicit split | Explicit validation file, else implicit split | Fully yours; you decide every row |
| **Hyperparameters** | 4 knobs (`epochCount`, `learningRateMultiplier`, `batchSize`, `adapterSize`) | ~3 knobs (epochs, batch size, LR multiplier) | Every knob in the field: `r`, `alpha`, target modules, schedulers, warmup, packing, NEFTune (CS-13 §7) |
| **Base models** | Gemini Flash / Flash-Lite families | GPT-4o-mini, GPT-4.1-mini/nano, o-series (verify current list) | Any of ~100k Hub models, any size |
| **Output artefact** | Opaque model + endpoint resources | Opaque model ID | `.safetensors` on your disk — full ownership |
| **Can you export the weights?** | **No** | **No** | **Yes** |
| **Training cost** | ~$1.50–$25 / 1M tokens (video) | ~$3–$25 / 1M tokens depending on model (verify) | GPU-hours: ~$0.50–$3/hr |
| **Serving cost model** | Endpoint **per hour** + per-token premium | Per-token, **plus a hosting charge when idle** in some periods (verify — restructured repeatedly) | GPU-hours, or scale-to-zero serverless |
| **Idle cost** | **High and certain** | **Verify** | Zero if scale-to-zero |
| **Time to first endpoint** | Hours | Minutes | Hours to days (infrastructure) |
| **Determinism** | None | None | Full — you can seed everything |
| **Eval metrics out of the box** | 4 series (loss, eval loss, next-token accuracy, n predictions) | Similar loss/accuracy series | None — you build it (which you should anyway) |
| **Multi-turn support** | Yes (`contents` array) | Yes (`messages` array) | Yes, framework-dependent |
| **Multimodal tuning** | Text; images/audio/video stated as supported [12:36] (verify per model) | Image inputs for vision fine-tuning on some models | Full control (CS-21) |
| **Compliance story** | Strongest: IAM, VPC-SC, CMEK, audit logs, regional data residency | Vendor DPA; less control | Strongest of all: air-gapped possible |
| **Lock-in** | Total | Total | None |
| **Best for** | GCP-standardised enterprises needing Gemini | Teams already on OpenAI, wanting the fastest path | Anyone who needs the weights, the price, or the control |

### 13.2 Schema differences that break a migration in either direction

| Field | Vertex → OpenAI | OpenAI → Vertex |
|---|---|---|
| System prompt | Move `systemInstruction.parts[0].text` into `messages[0] = {"role":"system","content":...}` | Hoist `messages[0]` out into a top-level `systemInstruction` |
| Assistant role | `"model"` → `"assistant"` | `"assistant"` → `"model"` |
| Text location | `parts[0].text` → `content` (string) | `content` → `parts: [{"text": content}]` |
| Conversation key | `contents` → `messages` | `messages` → `contents` |
| Multi-part turns | Flatten or drop | Supported natively via multiple `parts` |

> **Beyond the video:** write a **schema adapter** rather than maintaining two datasets. One canonical internal representation (a list of `(role, text)` tuples), plus a renderer per target platform, is roughly 60 lines of Python and eliminates an entire category of "the dataset was regenerated and someone forgot the Vertex renderer" incident. It also lets you keep your split and your hashes stable across platforms, which is what makes a genuine cross-platform quality comparison possible at all.

### 13.3 Quality comparison — what the honest answer looks like

| Question | Vertex SFT | OpenAI FT | Self-hosted LoRA |
|---|---|---|---|
| Will it beat a good prompt on style/format? | Usually yes | Usually yes | Usually yes |
| Will it beat a good prompt on facts? | **No** (CS-13 §4.1) | **No** | **No** |
| Can you achieve ~100% format compliance? | Yes, with a few hundred good examples | Yes | Yes |
| Will it be reproducible? | No | No | Yes |
| Can you prove *why* it improved? | No | No | Yes (ablations) |
| Can you run the same recipe on a different base model next quarter? | No — the recipe is not portable | No | Yes |

**The pattern:** the three routes are close to equivalent on *quality* for the tasks fine-tuning is actually good at. They differ almost entirely on **cost structure, control, and exit**. Choose on those, not on quality claims you cannot verify.

### 13.4 Break-even, restated as a single table

| Monthly requests (600 in / 300 out tokens) | Vertex tuned endpoint (est.) | Self-host 7B on L4, always-on | Self-host 7B, scale-to-zero serverless | Base model API + RAG, no tuning |
|---|---|---|---|---|
| 10,000 | $1,459 + ~$13 = **$1,472** | $584 + ops ≈ **$700** | ≈ **$60** | ≈ **$10** |
| 500,000 | $1,459 + ~$650 = **$2,109** | $584 + ops ≈ **$900** | ≈ **$700** | ≈ **$400** |
| 5,000,000 | $1,459 + ~$6,500 + replicas = **$10,000+** | $1,168 (2×L4) ≈ **$1,700** | ≈ **$5,000** | ≈ **$3,500** |
| 50,000,000 | **$100,000+** | ~$6,000 (8×L4) | ≈ **$45,000** | ≈ **$30,000** |

*All figures are order-of-magnitude estimates with illustrative per-token rates. Re-derive with your own traffic profile and current prices. The shape is the finding: at every traffic level, the tuned dedicated endpoint is the most expensive option, and it only becomes the right one at very high volume combined with a hard requirement on the specific base model.*

> **Beyond the video — the sentence that should end every one of these discussions:** *A dedicated tuning endpoint is a fixed cost that buys you a capability. If you cannot name the capability in one sentence, and you cannot show a measurement where the tuned model beats the base model by more than the cost delta, you are paying rent on a demo.*

---

## 14. Debugging Playbook

### 14.1 By symptom

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `403 PERMISSION_DENIED` on the very first `generate_content` | Billing not enabled | Console → Billing; check the project is linked and the account is active | Enable billing [28:44]. Free-tier credits do not apply to Vertex model access. |
| `403` naming a principal you do not recognise | Colab identity ≠ GCP identity | Print the authenticated identity: `!gcloud auth list` / compare avatars | Sign into Colab with the project-owning account [23:38] |
| `403` from an automated pipeline | Service account lacks `aiplatform.user`, or lacks `serviceAccountUser` on the VM's SA | `gcloud projects get-iam-policy PROJECT --flatten=bindings` | Grant the role at project scope; grant `iam.serviceAccountUser` for two-hop flows |
| `AttributeError: 'Client' object has no attribute 'tunings'` | Wrong `api_version` (or a pinned SDK older than the feature) | Print `genai.__version__`; check `HttpOptions(api_version=...)` | Pin `v1beta1` for tuning (§10.6), and upgrade the SDK |
| `NameError: name 'tuning_job' is not defined` | Cells executed out of order in the notebook | Re-run top to bottom in a clean runtime | Restart & run all; persist the job name to a file so a crash is recoverable |
| `AttributeError: 'NoneType' object has no attribute 'model'` after the poll loop | The job reached `FAILED`/`CANCELLED`; the loop did not test for it | Print `tuning_job.state` and `tuning_job.error` | Use the explicit terminal-state poller (§6.8) |
| Job stuck in `JOB_STATE_PENDING` for hours | Quota, capacity, or region congestion | Quotas page; check the region's status | Request a quota increase; try a different region; wait — there is no SLA on queue time |
| `400 INVALID_ARGUMENT` on `tune()` | Schema error in the JSONL | Validate one line against the current sample schema from the console | Fix `role: "assistant"` → `"model"`, hoist `systemInstruction`, wrap text in `parts` (§4.2) |
| `400` naming the dataset URI | Bucket in a different region, or wrong `gs://` string | `gcloud storage buckets describe gs://BUCKET --format='value(location)'` | Recreate/repoint the bucket in the job's region (§10.5) |
| `404` on a URL that looks correct | Region omitted from the hostname, or the job belongs to a different location | Check the host is `{location}-aiplatform.googleapis.com` | Pass the same `location` everywhere; a job created in `us-central1` is invisible from `europe-west4` |
| `tuned_model.endpoint` is `None` after `SUCCEEDED` | Job completed but the deployment step failed or was disabled | `tuning_job.error`; check endpoint quota | Deploy explicitly, or file a support case — do not assume retraining will help |
| Inference returns base-model behaviour, not tuned | Wrong `model=` string resolving to a base model | Log the resolved model; compare a known training prompt's answer | Pass the correct tuned resource; assert in code that the model string contains `/models/` or `/endpoints/` |
| `401` mid-poll after hours of running | OAuth token expired (Colab tokens are short-lived) | The traceback will say `RefreshError`/`invalid_grant` | Re-authenticate and re-enter the poll loop; better, use a service account for long jobs |
| **Training loss flat, no decrease** | LR multiplier too low, or the adapter cannot represent the change (rank too small), or the data has no signal | Compare train loss at step 1 vs final; check `adapterSize` | Raise `adapterSize` first (cheap, structural); then try a higher multiplier |
| **Training loss goes to ~0 quickly** | Overfitting: too many epochs, too small a dataset, rank too high | Loss curve shape + prompt-mutation tests (§12.4) | Drop `epochCount` to 1; add data; reduce rank |
| **Loss spikes, then NaN** | LR multiplier too high | Loss curve has a characteristic spike-then-cliff | Halve the multiplier; the job will still report `SUCCEEDED` with a useless model |
| **Eval loss diverges from train loss (train ↓, eval ↑)** | Textbook overfitting | Two curves on the same axis in the experiment dashboard | Fewer epochs; more data; lower rank; add replay data (CS-13 §4.9.5) |
| **Eval loss ~= ln(vocab size)** | The model learned nothing — it is emitting a uniform distribution | `ln(100k) ≈ 11.5`; if your eval loss is near that, nothing happened | Check the mask: is the data actually in `model` turns? Is `contents` populated? |
| **Eval loss lower than train loss** | Not necessarily good: the eval slice may be easier, or it leaked into training | n-gram overlap check between train and validation files | Re-split with the deterministic splitter (§4.5) |
| Loss curves look fine but the model is bad | Log loss is the wrong metric for the task | Run `eval_ab.py` (§12.3) against the base model | The tuning may have done nothing useful. See §17. |
| Cost is 100× the estimate | **Endpoint left deployed** | Billing export query (§11.5); `aiplatform.Endpoint.list()` | `undeploy_all()` → `delete()` → `Model.delete()` (§6.10) |
| Tuning job rejected: "model not supported for tuning" | The base model ID is retired or is not tunable | Check the live tuning-support table and the console dropdown | Move to a currently-tunable base model — this is a plan-level problem, not a code bug |
| Job succeeds, evaluation output missing | The Vertex service agent lacks write access to the output bucket | Job state is `PARTIALLY_SUCCEEDED`; check the bucket IAM | Grant the correct permission to `service-<PROJECT_NUMBER>@gcp-sa-aiplatform…` (§10.11) |
| `gcloud storage cp` succeeds but the job cannot read the file | Object-level ACL rather than bucket-level; or a uniform-bucket-level-access mismatch | `gcloud storage objects describe gs://…` | Use uniform bucket-level access and a bucket-scoped `objectViewer` grant (§6.2) |

### 14.2 Distinguishing the four things that look like "the fine-tune didn't work"

This is the diagnostic that matters most, because all four produce the same complaint.

| Root cause | Signature | Test | Fix |
|---|---|---|---|
| **The data was bad** | Loss curve is *fine*; outputs are confidently wrong or generic | Read 20 random training rows aloud. Would a human produce these answers? | Re-author the data. No hyperparameter will save you. |
| **The hyperparameters were wrong** | Loss curve is visibly wrong (flat, spiking, or crashed to 0) | Look at the TensorBoard experiment | `epochCount` / `adapterSize` / `learningRateMultiplier` (§7) |
| **The schema was wrong** | Loss is near `ln(V)`; the model learned nothing | Print one tokenised row; check `role` and `parts` | Fix the JSONL (§4.2) |
| **The task was never a fine-tuning task** | Loss curve is *perfect*; the base model does just as well | Run the base model on the same prompts (§12.3) | Prompt engineering (CS-04) or RAG. Cancel the endpoint. |

**The order to test them in is the reverse of the order people test them.** Check the task first (cheapest, and rules out the whole exercise), then the schema, then the data, then the hyperparameters.

> **Beyond the video:** keep a `_debug/` directory in your repo containing (a) one pretty-printed tokenised row, (b) the loss curve PNG from the experiment dashboard, (c) the first 20 lines of the eval report, and (d) the resolved model/endpoint IDs from the run. When someone asks "why is the tuned model bad", that directory answers three of the four questions above in under a minute.

---

## 15. Applied Case Studies

### 15.1 Case A — The 10-row demo that should never have been deployed

**Situation.** A solutions engineer builds exactly the video's notebook for a customer demo: 10 rows of smartphone support Q&A, `gemini-2.5-flash`, one tuning job, ~20 minutes, a live endpoint. The demo lands well. The endpoint stays up for the quarter.

**Why this technique.** It genuinely is the fastest path from "a JSONL" to "a live tuned endpoint on a URL". For a demo, that is the requirement.

**Exact config.**

```python
tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=TuningDataset(gcs_uri="gs://demo-sft/data.jsonl"),   # 10 rows, 564 tokens
    config=CreateTuningJobConfig(tuned_model_display_name="demo-sft-job"),
)
```

**Result.** Training cost **$0.00282**. Wall clock 15–20 minutes. Wall clock of the deployment's life: one quarter.

**What went wrong first.** Nobody owned the undeploy. The endpoint billed through a quarter at an estimated ~$2/hour — **~$4,400** — to serve perhaps a few dozen demo requests. The engineering cost of the training run was three orders of magnitude smaller than the cost of the deployment it produced.

**The fix, in order of how much it would have saved.** A budget alert at $50 would have caught it in week one. A `expires-at` label plus a scheduled sweeper (§6.10) would have prevented it. A one-line note in the repo README — "undeploy before you close the ticket" — would have prevented it. **None of these is a technical change.**

---

### 15.2 Case B — The enterprise that could not get GPU quota

**Situation.** A 200-person insurance company on GCP wants a claims-triage assistant that writes in the company's regulated style. They have 6,000 curated claims notes with the "correct" triage summary written by senior adjusters. Their GPU quota request for an A100 was denied twice (new account, no ML history), and procurement will not approve a second cloud.

**Why this technique.** Vertex SFT requires no GPU quota. The data already lives in GCS. IAM, CMEK and audit logging are already configured at the org level, which means the security review — the usually fatal step — is a formality.

**Exact config.**

```python
tuning_job = client.tunings.tune(
    base_model="gemini-2.5-flash",
    training_dataset=TuningDataset(gcs_uri="gs://claims-sft/train.jsonl"),        # 5,100 rows
    config=CreateTuningJobConfig(
        tuned_model_display_name="claims-triage-sft-v2-2026-09-27",
        # validation_dataset=TuningDataset(gcs_uri="gs://claims-sft/validation.jsonl"),  # 900 rows
        # hyper_parameters: epoch_count=3, adapter_size="ADAPTER_SIZE_EIGHT"
    ),
)
```

**Numbers.** 5,100 rows × ~700 tokens × 3 epochs = **10.7M tokens** → at $5/1M ≈ **$54**. Plus ~$30 for the epoch-1 variant. Total training spend **under $100**. Held-out: 900 rows, stratified by claim type.

**Result.** Format compliance on the 900-row held-out set went from 71% (base, with a detailed system prompt and 5 few-shot examples) to 96%. Style-guideline adherence (a rubric scored by a judge) went from 3.1/5 to 4.4/5. **Task accuracy barely moved** — consistent with CS-13's thesis that SFT buys format and style, not knowledge.

**What went wrong first.** Attempt 1 used `epochCount=10` on 6,000 rows because "more epochs is better". The resulting model produced summaries that were *verbatim rearrangements of training claims* — it had memorised the dataset. Detected by prompt-mutation testing: it failed on rephrased questions. Fixed by dropping to `epochCount=3` and reducing `adapterSize` from 16 to 8.

**The economics.** ~$1,400/month endpoint + ~$600/month inference = **~$2,000/month** to serve 400 adjusters, versus an estimated $180,000/year fully loaded for the manual triage of the same volume. **The tuned endpoint is unambiguously the right call here** — but note that the endpoint is 95% of the run rate, and a self-hosted open 8B would have been ~$900/month.

---

### 15.3 Case C — The team that should have used RAG

**Situation.** A legal-tech startup fine-tunes Gemini 2.5 Flash on 12,000 Q&A pairs extracted from their case-law database. Goal: "the model should know our case law."

**Why this technique (as they framed it).** They had the data in a clean JSONL. SFT was the obvious next step.

**Exact config.** `epochCount=3`, `adapterSize=16`, 12,000 rows × 900 tokens ≈ 32M tokens → **~$162**.

**Result.** The tuned model answered questions about cases it had seen in training *beautifully* — in the house citation format, with the right tone. It **failed completely** on cases added to the database after the training cut-off, and hallucinated citations for them with high confidence. Measured: 94% accuracy on training-distribution cases, 31% on post-cut-off cases (worse than the base model's 44%, because it had learned to always produce a citation).

**Why it failed.** Textbook CS-13 §4.1: SFT does not inject retrievable facts reliably; it teaches format and behaviour. They had taught the model to *shape* a legal answer without teaching it *which* facts were true.

**The fix.** Retrieval over the case-law index (CS-04), with the tuned model kept **only for the formatting behaviour** — retrieve passages, then generate in house style. Accuracy on post-cut-off cases recovered to 78%, and they now update by re-indexing documents rather than re-training.

**What went wrong first.** They deployed, measured on in-distribution questions, saw 94%, and shipped. **The eval set was the bug**: it had been sampled from the same corpus as the training data. A post-cut-off set built by date-filtering would have caught it before deployment for the cost of one engineer-hour.

---

### 15.4 Case D — The high-volume serving decision

**Situation.** A consumer app with 25M requests/month wants a domain-tuned assistant. They evaluated Vertex SFT and self-hosted Llama-3.1-8B with LoRA + vLLM.

**Why both were on the table.** Quality was close. The decision was entirely economic.

| | Vertex tuned endpoint | Self-hosted 8B LoRA + vLLM |
|---|---|---|
| Training | ~$300 (60M tokens) | ~$60 (2× A100-hours + engineer time) |
| Serving, 25M req/mo (600 in / 300 out) | ~$100k+/month (multiple replicas, tuned premium) | ~$6k/month (8× L4 reserved, 1-yr commit) |
| Engineering to build | 1 week | 6 weeks |
| Ongoing ops | ~0 | ~0.2 FTE |
| Weights exportable | No | Yes |
| Time to production | 2 weeks | 8 weeks |

**Result.** They went self-hosted. The 6-week difference in time-to-production was paid back by **month 1** of the serving cost delta. The 0.2 FTE of ops was less than one-twentieth of the monthly saving.

**What went wrong first.** They spent three weeks on the Vertex path before doing this arithmetic, because the prototype had been so fast that nobody questioned the run rate. **The lesson: do the serving arithmetic on day one, before you build anything.**

> **Beyond the video:** the generalisable rule. **Prototype speed and production economics are different questions, and managed services optimise only the first.** Adopt the managed path to *learn whether the task is worth doing*; migrate to the self-hosted path when the traffic justifies it. The mistake is treating the prototype's architecture as the production architecture by default, which is what a fast prototype tempts you into.

---

### 15.5 Case E — The regulated deployment that could not use it

**Situation.** A European healthcare provider wants a patient-communication assistant. Their data is subject to residency requirements and their regulator requires the ability to demonstrate exactly what data trained what model.

**Why they considered Vertex.** Google's certifications and the GCP control plane are strong, and Vertex tuning is served in EU regions.

**Why it failed the review.** Three findings:
1. **The tuned artefact cannot be inspected.** The team could not produce a description of the model's parameters, size, or training procedure for the technical-file requirements. "A Google-managed resource" was not an acceptable answer to the auditor's question about the trained weights.
2. **No reproducible reconstruction.** `epochCount` and `adapterSize` were recorded, but there is no seed and no way to demonstrate that a given model corresponds to a given dataset beyond the job record. The regulator wanted stronger provenance.
3. **Data handling.** The training data traversed Google infrastructure under the standard terms. Whether that satisfied the specific residency requirement depended on a legal reading of the data-processing addendum that the team could not get confirmed in time.

**What they did.** Ran the fine-tune on an open model self-hosted in their own EU region (CS-23), which produced a downloadable adapter, a reproducible training command, a dataset hash, and a signed model card.

**What went wrong first.** They built the Vertex prototype before involving legal, then had to rebuild. **The lesson: in a regulated environment, the artefact-export and provenance questions decide the architecture, and they are not answerable after the fact.**

---

## 16. Production Considerations

### 16.1 The lifecycle, as five explicit states

The most common production failure in this module is treating "trained" and "deployed" as one state. They are five:

```
  ┌──────────┐  tune()   ┌──────────┐  deploy   ┌────────────┐
  │  DRAFT   │ ────────► │  TRAINED │ ────────► │  DEPLOYED  │
  │ (JSONL   │           │ (Model   │           │ (Endpoint  │
  │  + split)│           │  @N      │           │  billing   │
  └──────────┘           └──────────┘           │  per hour) │
                              ▲                 └────────────┘
                              │                       │
                              │  promote              │ undeploy_all()
                              │                       ▼
                         ┌──────────┐           ┌────────────┐
                         │  SHADOW  │ ◄──────── │  UNDEPLOYED│
                         │ (deployed│           │ (Model     │
                         │  but not │           │  retained) │
                         │  routed) │           └────────────┘
                         └──────────┘
```

| State | Resources alive | Billed | Duration in production | Entry criteria |
|---|---|---|---|---|
| **DRAFT** | GCS objects | storage pennies | days–weeks | A frozen, split, validated dataset |
| **TRAINED** | GCS + `Model@N` | ~nothing | hours–days | `eval_ab.py` shows a positive delta over base |
| **SHADOW** | + `Endpoint` | **per hour** | 1–14 days | Passing regression + refusal suites |
| **DEPLOYED** | + `Endpoint` serving traffic | per hour + tokens | as long as the product needs it | Shadow traffic agreement within tolerance |
| **UNDEPLOYED** | `Model@N` only | ~nothing | indefinitely | Traffic pattern no longer justifies it, or the model is superseded |

**Write these five states into your runbook with an explicit owner for each transition.** The transition nobody owns is DEPLOYED → UNDEPLOYED, and it is the expensive one.

### 16.2 Serving architecture

| Pattern | When | Notes |
|---|---|---|
| **Direct endpoint call from the app** | Small apps, low traffic | Simplest. The endpoint URL becomes an app config value. |
| **Backend proxy** | Almost always | Never let the browser talk to the endpoint. The proxy holds the credentials, applies per-user rate limits, logs the resolved model ID, and is the place you swap endpoints during a rollback. |
| **Shadow deployment** | Before promotion | Send a *copy* of production traffic to the candidate endpoint, compare outputs offline. Costs the endpoint hours; buys the confidence that a promotion is safe. The instructor does not mention this and it is the single highest-value production practice here. |
| **A/B with a routing rule** | After shadow | Route 5% → 25% → 50% → 100%. Requires the proxy to key on a stable user ID. |
| **Multi-endpoint fan-out** | Never, at first | Two endpoints = two hourly bills. Do not run a tuned + base endpoint "just in case" unless you have budgeted for the pair. |

### 16.3 Versioning and rollback

**What is versioned on Vertex:** the model resource carries `@N`. Every retrain creates `@N+1` under the same model ID. That is real and useful — you can see the lineage.

**What is not versioned:** the endpoint's routing. There is no "30% to @1, 70% to @2" primitive in the naive setup; you either deploy `@N` or `@N+1`.

| Practice | Why it matters here |
|---|---|
| **Pin the endpoint to an explicit `@N`** | A retrain that silently appends `@2` must not become production by accident. |
| **Keep the previous `Model@N-1` undeployed but alive** | Rollback is then "deploy the old one", minutes not hours. |
| **Record the dataset hash with the model version** | The `Model@N` resource does not carry your data lineage. Your sidecar JSON does (§10.12). |
| **Never delete the previous model until the new one has served a full traffic cycle** | Including a weekend and any batch job that runs monthly. |
| **Rollback procedure, written down and rehearsed once** | "Deploy `@N-1`, flip the proxy config, verify the eval suite." Four steps; the team should have run them once before they need to. |

### 16.4 Monitoring

| Signal | Source | Alert threshold | Why |
|---|---|---|---|
| **Endpoint exists AND 0 requests in 2h** | Cloud Monitoring / your proxy logs | Any occurrence (business hours) | The forgotten-deployment detector (§9.4) |
| **Endpoint monthly cost** | Billing export (§11.5) | > 120% of forecast | Catches a replica-count change or a second endpoint |
| **Resolved model ID per request** | Your proxy log | Any change without a deploy event | Catches a silent fallback to the base model |
| **Refusal rate** | Classifier over responses | > 5% on a 200-prompt benign suite | Over-refusal is the most common product regression |
| **Format-compliance rate** | Parser over responses | < 99% on a schema-constrained endpoint | Under-training or a prompt change upstream |
| **Output length distribution** | Histogram | Shift > 2σ from the baseline | Verbosity inflation is the first symptom of drift or of a training-set change |
| **p95 latency** | Proxy | > 2× baseline | Your endpoint is cold, throttled, or sharing capacity |
| **4xx/5xx rate** | Proxy + Vertex metrics | > 1% | Quota, auth expiry, or a base-model deprecation |
| **Data-drift proxy: input embedding centroid shift** | Embeddings of live inputs vs training inputs (CS-22) | Drift beyond a set threshold | The best leading indicator that your model is now off-distribution |
| **Eval-suite pass rate, run weekly** | `eval_ab.py` on a cron | Any regression | Detects changes in the *served* model you did not make |

> **Beyond the video:** the last row is the one people omit and the one that matters most on a managed service. **Because you do not control the serving stack, the model behind your endpoint can change without your involvement** — a base-model update, a serving-stack change, a capacity reshuffle. The only way to detect that is a **weekly scheduled eval** that runs the same frozen prompts against the same endpoint and diffs the outputs. Ten lines of code, one cron entry, and it is the difference between discovering a regression from a dashboard and discovering it from a customer.

### 16.5 Regression testing

Properties, not exact strings (CS-13 §16.5). A frozen suite of 50–200 prompts asserting:

```python
# regression_suite.py — asserts PROPERTIES, not golden strings.
# Golden strings break on every model update and teach the team to ignore failures.
CHECKS = [
    ("parses as JSON",              lambda r: is_json(r)),
    ("no refusal phrase",           lambda r: not any(p in r.lower() for p in REFUSAL_PHRASES)),
    ("<= 120 words",                lambda r: len(r.split()) <= 120),
    ("contains a citation",         lambda r: bool(CITATION_RE.search(r))),
    ("responds in the asked language", lambda r: detect_lang(r) == expected_lang),
    ("no PII-like patterns",        lambda r: not PII_RE.search(r)),
    ("does not mention the base model", lambda r: "gemini" not in r.lower()),
    ("ends with a complete sentence",   lambda r: r.rstrip().endswith((".", "!", "?"))),
]
```

Run this against **both** the current production endpoint and the candidate before every promotion, and store the pass/fail matrix. The diff between two matrices is your promotion decision.

### 16.6 Cost governance

| Control | Mechanism | Catches |
|---|---|---|
| Budget + alerts | Console → Billing | Everything, eventually |
| **Hard cap** (where supported) | Budget → automatic disable | The runaway case |
| Billing export → BigQuery | Daily query (§11.5) | Attribution: which SKU, which project |
| Label convention | `owner`, `purpose`, `expires-at` on endpoints | Ownership and sweeper automation |
| Scheduled sweeper | Cloud Function + Cloud Scheduler | Forgotten endpoints, mechanically |
| A "who owns this endpoint" dashboard | BigQuery + Looker Studio | The human failure |
| **A monthly 15-minute cost review** | Calendar invite | The organisational failure |

**The reason to write all of that down:** every one of these is cheaper than the first incident. The endpoint is a fixed cost with no natural feedback loop — nobody complains when it is running, because it is working fine.

### 16.7 The compliance angle

| Question | What to establish before you train |
|---|---|
| **Where does the data go?** | Your GCS bucket, in your project, in a region you chose. Vertex training reads from there. |
| **Where does the artefact live?** | In Google's control plane, associated with your project. You cannot export it. |
| **Is the data used to improve Google's models?** | Read the specific terms for the specific service. Do not assume from the consumer-product terms (which are different). |
| **What is your retention/erasure story?** | If a data subject asks for erasure, can you remove their influence from a trained model? (Answer: you can delete the dataset and retrain. You cannot un-learn.) |
| **Can you demonstrate provenance?** | Dataset hash + job ID + hyperparameters + metrics. Build the sidecar (§10.12). |
| **CMEK?** | Vertex supports customer-managed encryption keys for many resources; verify which apply to tuning artefacts and to the endpoint. |
| **VPC-SC?** | A perimeter around your project restricts data movement; verify it is compatible with Vertex tuning in your region. |
| **Audit logging?** | Cloud Audit Logs record the API calls. Enable Data Access logs for Vertex if your review requires request-level evidence. |
| **Who can call the endpoint?** | `aiplatform.user` at project scope means anyone with that role can call *every* endpoint. Consider a dedicated project or a stricter custom role. |

> **Beyond the video:** the compliance question this service makes *harder* than self-hosting is **provenance of the trained artefact**. With an open-weight fine-tune you can hash the adapter, sign the model card, and produce the exact training command. With a managed tune you have a job record and a metric series. That is a *weaker* evidentiary position, and it is the reason Case E (§15.5) went self-hosted despite the otherwise-attractive GCP fit. Say this out loud in your design review, because it is not obvious until the auditor asks.

---

## 17. Common Misconceptions

**1. "Fine-tuning costs money, so I should be careful about dataset size."**
It costs *almost nothing*. 564 tokens of Gemini 2.5 Flash training is **$0.00282** [42:25]. The median production fine-tune in this module is under **$100**. What costs money is the deployed endpoint, at an estimated ~$1,400/month. Be generous with data and ruthless about deployments.

**2. "Vertex and AI Studio are the same thing."**
Different products with different auth, quotas, model IDs, and — critically — **different tuning support**. The whole reason this module exists is that AI Studio lost fine-tuning [7:18]. They share a Python SDK, which is exactly what makes the confusion so easy.

**3. "You need a GPU to fine-tune Gemini."**
No. The instructor runs the entire practical on a **CPU-only Colab runtime** [18:50]. That is the value proposition. Select the cheapest runtime you can get.

**4. "The fine-tuned model will know my private business data."**
Only weakly, and only if the behaviour you want is *formatting* rather than *facts*. The companion dataset is 10 rows of smartphone support Q&A; the demo question ("why do phones have a one-year warranty?") is a **fact Gemini already knew**. Nothing in the video's demo distinguishes a tuned model from a prompted one. CS-13 §4.1 has the full argument.

**5. "Once the job succeeds, I'm done."**
The job succeeding *creates new billable resources*. `JOB_STATE_SUCCEEDED` is the moment your monthly cost starts, not the moment it ends.

**6. "Deleting the model cleans up."**
The video's cleanup deletes the `Model` [56:39]. The `Endpoint` is a separate resource with a separate lifecycle and a separate bill. §6.10, §10.3.

**7. "Lower evaluation loss means a better model."**
Eval loss is next-token cross-entropy on a held-out slice of *your own data*. It measures how well the model predicts your distribution, not how well it does your task. A model can have excellent eval loss and be worse in production (Case C, §15.3).

**8. "The validation split is handled for me."**
It is — and that is the problem. You did not choose the ratio, cannot see which examples were held out, and cannot stratify it. For a 10-row dataset it is two examples. **Pass an explicit validation file.** §4.5.

**9. "More epochs = better."**
`epochCount` is a linear multiplier on your training bill and the fastest route to memorisation. The useful SFT range is **1–3** (CS-13 §7.3). Case B (§15.2) lost a training cycle to `epochCount=10`.

**10. "Fine-tuning gives me a model I can run anywhere."**
It gives you a **resource name**. No download, no export, no quantisation, no merge, no on-prem. [3:35]. This is structural, not a limitation that will be lifted.

**11. "Vertex SFT must be expensive because it's enterprise."**
The training is nearly free; the *deployment* is expensive. The service is priced like a server (per hour), not like a batch job (per token). Misreading that single fact produces every budgeting error in this module.

**12. "I can compare two Vertex runs and conclude which dataset is better."**
There is no seed and no determinism (§10.10). Two identical jobs produce different models. You must calibrate your noise floor — run the same config twice, once — before attributing a delta to a change.

**13. "The console will tell me if something is wrong."**
The console tells you the job state. A model that trained perfectly and learned nothing reports `SUCCEEDED` with a falling loss curve. The console has no opinion about whether your model is *good*.

**14. "Preference tuning is just SFT with pairs."**
They are separate tuning methods on Vertex [11:15] with different schemas, different objectives and different failure modes. See CS-14/CS-24/CS-25/CS-27.

**15. "I'll just use the model string from the tutorial."**
The transcript renders Gemini as "Jimny" and Vertex as "Vortex" throughout, and the model line-up changes quarterly. Model IDs come from the console and the API reference. §10.1.

**16. "The 15–20 minute training time means the dataset was small."**
It means the *overhead* is fixed. A 564-token job and a 5,000-row job are both dominated by provisioning at the low end [50:56]. Budget the wall-clock accordingly and do not use it as a proxy for job size.

**17. "Full fine-tuning on Vertex means the weights all moved."**
You cannot verify that. Treat it as a pricing tier and evaluate the output. §7.5.

**18. "If it works in the notebook, it works in production."**
The shipped notebook in this repo contains cells that raise `NameError` and an inference cell that may resolve to the wrong resource. A notebook that ran once is not a pipeline.

**19. "The endpoint is cheap because I only pay per token."**
No. You pay per hour the deployment exists, plus tokens. At hobby traffic the hourly charge is ~100% of the cost (§11.3).

**20. "I should deploy immediately after training so I can test it."**
Test it, yes — then **undeploy**. The correct default state for a tuned model is TRAINED, not DEPLOYED (§16.1). Deploy for evaluation windows measured in hours, not months.

---

## 18. Key Takeaways

1. **Vertex SFT is a rental: no weights, no export, no exit.** [3:35] Everything else follows from that one design fact.
2. **Training is free; deployment is expensive.** 564 tokens costs $0.00282; an idle endpoint costs an estimated ~$1,400/month. The ratio is ~500,000:1.
3. **`tuned_model.model` (the artefact) and `tuned_model.endpoint` (the deployment) are two resources with two bills and two lifecycles.** Confusing them is the module's most expensive mistake.
4. **Deleting the model does not release the endpoint.** `undeploy_all()` → `endpoint.delete()` → `Model.delete()`, in that order.
5. **The Vertex JSONL schema is not OpenAI's.** `systemInstruction` + `contents[].parts[].text`, `role: "model"`, JSONL in GCS in the same region. §4.2.
6. **There are five states, not two: DRAFT → TRAINED → SHADOW → DEPLOYED → UNDEPLOYED.** Name an owner for each transition.
7. **The automatic metrics are log loss and teacher-forced token accuracy.** Neither measures task success. Build `eval_ab.py` and compare against the *base* model. §12.3.
8. **Always pass an explicit validation file.** The implicit split is unseen, unstratified and, at 10 rows, two examples. §4.5.
9. **Four hyperparameters exist; the video names none of them.** `epochCount` (1–3, multiplies the bill), `learningRateMultiplier` (leave at 1.0), `batchSize` (leave at default), `adapterSize` (4 or 8). §7.
10. **The setup is harder than the code.** Project, billing, region, bucket, IAM, matching Colab identity. The instructor says so himself [19:53].
11. **`v1beta1` for tuning, `v1` for inference.** Keep both clients, pin the SDK version, and read the `ExperimentalWarning` as a contract term. §10.6–10.7.
12. **The 15–20 minute fixed overhead makes small experiments nearly free and iteration slow.** Run the sweep locally on an open model; confirm the winner here.
13. **A 10-row managed fine-tune is a plumbing demo, not a model.** It cannot distinguish a tuned model from a prompted base model.
14. **The managed path wins on time-to-first-model and organisational fit, and loses on unit economics the moment you deploy.** Break-even is a traffic question. §13.4.
15. **Undeploy is a control, not a habit.** A budget alert plus a scheduled sweeper is the only teardown that survives a busy week.

---

## 19. Self-Check Questions

1. Why can't you download a fine-tuned Gemini model, and what are the two downstream consequences for your architecture?
2. Write the Vertex JSONL schema from memory, and name the three fields that differ from OpenAI's.
3. Your colleague's tuning job reports `JOB_STATE_SUCCEEDED`. What resources now exist, and which one bills?
4. You ran a job yesterday and forgot to note its name. What does `client.tunings.list()[1]` return, and why is that dangerous?
5. A tuning job on 10,000 rows × 500 tokens with `epochCount=3` at $5/1M tokens — what is the training cost?
6. The same job's endpoint runs for 45 days at an estimated $2/hour. What is the endpoint cost, and what is the ratio to question 5?
7. What is `learningRateMultiplier` a multiplier *of*, and why can't you set an absolute learning rate?
8. Your eval loss is 11.4 on a model with a 100k vocabulary. What does that tell you?
9. You must roll back a production tuned model to the previous version. What are the four steps, and which resource do you touch?
10. Give the two conditions under which a dedicated tuned endpoint beats self-hosting an open-weight model of comparable quality.

<details>
<summary>Answers</summary>

1. Because the base weights are closed-source and the tuned artefact is only ever exposed as a Vertex resource [3:35]. Consequences: (a) no exit strategy — migrating means re-doing the fine-tune elsewhere; (b) no downstream optimisation — you cannot quantise, merge, batch, or run the model outside a Vertex endpoint, so your serving cost is permanently Google's price.

2. Because the system prompt is hoisted out of the message list, the row looks like this:

```json
{"systemInstruction": {"role":"system","parts":[{"text":"..."}]},
 "contents":[{"role":"user","parts":[{"text":"..."}]},
             {"role":"model","parts":[{"text":"..."}]}]}
```

Three differences from OpenAI: the system prompt is **hoisted** out of the message list into a top-level `systemInstruction`; the assistant role is `"model"`, not `"assistant"`; the text is nested in a **list** of `parts` objects rather than being a flat `content` string.

3. A tuned `Model` resource (`.../models/<id>@1`) and, by default, a deployed `Endpoint` (`.../endpoints/<id>`). The **Endpoint** bills — per hour, whether or not it receives traffic. The Model itself carries no published hourly charge.

4. The second-most-recent job in the project, which may be a colleague's job, a retried job, or a test job. Dangerous because you would then read *its* `tuned_model.endpoint` and deploy or query the wrong model while believing it is yours. Persist the job name returned by `tune()`.

5. `10,000 × 500 × 3 = 15,000,000` tokens. `15 / 1e6 × $5 = ` **$75.00**.

6. `45 × 24 = 1,080 hours × $2 = ` **$2,160**. Ratio: `2,160 / 75 = ` **28.8×** — the deployment cost nearly thirty times the training cost, and it was entirely avoidable by undeploying.

7. It multiplies **Vertex's internal default learning rate**, which you cannot read. You can only express "more" or "less" relative to whatever Google chose, which means LR values you know from other frameworks (e.g. `2e-4`) are meaningless here, and a recipe from a paper cannot be reproduced exactly.

8. `ln(100,000) ≈ 11.51`. An eval loss of 11.4 is essentially the uniform-distribution entropy, meaning the model has learned **nothing** — it is spreading probability evenly across the vocabulary. Check the schema first (`role: "model"`, populated `parts`), then the data.

9. (1) `undeploy_all()` on the current endpoint, or deploy `@N-1` to a fresh endpoint; (2) deploy the previous `Model@N-1` to an endpoint; (3) flip your proxy's model config to the previous endpoint; (4) verify with the regression suite. The resources you touch are the **Endpoint** (which endpoint the proxy points at) and the **Model** (`@N-1`, which must still exist — so never delete the previous model until the new one has served a full traffic cycle).

10. (a) Traffic is high enough that the fixed hourly endpoint charge is a small fraction of the total, and the per-token comparison favours the self-hosted deployment; (b) the task cannot be served by an open-weight model of comparable quality — i.e. you specifically need Gemini's capabilities, and you have measured that the tuned Gemini beats the best open alternative by more than the cost delta.

</details>

---

## 20. Cross-References

| Relationship | Module | Why |
|---|---|---|
| **Builds on** | CS-13 (Instruction Fine-Tuning) | The objective, the masking, the data-format landscape and the "elicitation vs injection" thesis are all assumed here. |
| **Builds on** | CS-02 (Transfer Learning & Fine-Tuning) | What a fine-tune is, before it becomes a managed API call. |
| **Builds on** | CS-03 (Framework Landscape) | Where managed services sit relative to the open-source stacks. |
| **Pairs with** | CS-18 (OpenAI GPT Fine-Tuning) | The sibling managed service. Read them together; the schema contrast in §4.2 and the comparison table in §13.1 are the payoff. |
| **Contrasts with** | CS-16 (Unsloth) | The self-hosted counterfactual: 2–4× faster, low VRAM, and **you keep the adapter**. |
| **Contrasts with** | CS-17 (Axolotl) | YAML-driven self-hosted training at scale. Every knob Vertex hides, Axolotl exposes. |
| **Contrasts with** | CS-23 (LoRA & QLoRA) | What `adapterSize` means at the tensor level, and what you can do with an adapter you actually own. |
| **Needed by** | CS-20 (SLMs), CS-21 (Multimodal) | The "should this be managed or self-hosted" decision recurs in both. |
| **Related** | CS-04 (Fine-Tuning vs RAG vs Agents) | The decision that precedes this one. §15.3 is a RAG case in disguise. |
| **Related** | CS-14 (The Alignment Map) | Vertex's second tuning method (preference tuning) [11:15] maps onto this module's taxonomy. |
| **Related** | CS-10 / CS-11 (Quantization) | Only relevant on the self-hosted path — quantization is one of the capabilities the managed service removes. |
| **Related** | CS-22 (Embeddings) | The data-drift detector in §16.4 uses embedding centroids over live inputs. |

---

## Appendix A — Instructor's Verbatim Key Claims

Quotes reproduced as the transcript renders them, including the speech-to-text artefacts (`Jimny` = Gemini, `Vortex` = Vertex, `GPD` = Gemini). Timestamps are the video's.

**On the closed-source constraint [3:35]:**
> *"Again guys this is not a open-source model. This is a closed source. It is being provided by the Google. So we don't have access like where we can download the model but yeah we can access the model via API."*

**On the two Google platforms [2:45] and [25:29]:**
> *"Both are the different thing. I will discuss about both thing — what is the Jimny API and what is the Vortex AI or Vortex API."*
>
> *"If I'm saying GCP or Vortex AI, then both are same, right? And we have two way to access the Vortex AI. The first using the Google geni SDK. The second is Vortex AI SDK. Understood guys? Don't be confused here… GCP is a cloud platform, over the GCP you will get the Vortex AI. Okay, this is the service."*

**On where fine-tuning is available [7:18]:**
> *"they have clearly written with a depreation of the Jimny 1.5 flash in May 25, we no longer have a model available which support fine-tuning in the Jimny API or AI studio — uh but it support in the Vortex AI."*

**On the models that support tuning [10:46]:**
> *"GPD 2.5 pro, uh GPD 2.5 flash, GPD 2.5 flash light, uh 2.0 to flash and 2.0 flash light. So these many model is supporting for the finetuning. You can only fine-tune these mentioned model."*

**On the two tuning methods [11:13]:**
> *"So as of now they are just supporting two method. One is the supervised finetuning and second is the preference tuning."*

**On the two weight-update regimes [13:14]:**
> *"They are supporting two approaches. One they are first is the parameter efficient finetuning and second is the full fine tuning."*

**On the difficulty of the setup [19:53]:**
> *"That's why I was saying this fine tuning is not very straightforward. Whatever code they have given you in the documentation, if you're directly going to be executed, it will not work until and unless you are not doing a proper setup."*

**On why no GPU is needed [18:50]:**
> *"I will select the CPU only because I don't require any GPU here because I'm not going to train the model on my own server. Uh the model is being trained over the Google server only. I'm just hitting it through the Google API."*

**On the identity requirement [23:38]:**
> *"If you are logging to the GCP, right, so please create a notebook with the same ID… otherwise guys you will face a issue, you will face a trouble and you will not be able to authenticate properly."*

**On billing as a hard gate [28:16] and [31:16]:**
> *"You might get the error with respect to the billing."*
>
> *"Neither you can generate the output nor you can perform the finetuning. Nothing you can do until and unless you are not setting up the billing account."*

**On the schema difference [34:53]:**
> *"The format is a little bit different over here… We have system instruction in instead of the message. Now role will be the system. Then apart from this we have one more key that is part. Now under the part actually we will be having a text."*

**On the role naming [36:10]:**
> *"Role user means user is asking a question, role model means model is generating an answer."*

**On the dataset and token count [40:10] and [40:50]:**
> *"How many example we have? 10 rows in total. 10 rows we have. This is the total number of tokens — 5 64… minimum number of token in the example 51, 62 is a maximum one and 56 is the average token."*
>
> *"This data set actually it is all about the mobile… mobile customer support."*

**On the tuning prices [22:29]:**
> *"Jimny 2.5 Pro around $25 for the 1 million token, 2.5 flesh $5, 2.5 flesh light $1.5."*

**On where the data must live [43:20]:**
> *"Google cloud platform is not recommending this one… this Vortex AI is not recommending this one. So what they are saying: if you wanted to use the data then you will have to keep it inside the Google cloud storage."*

**On starting the job [45:16]:**
> *"I'm going to create my job — `client.tuning.tune`. I will pass my model, which model I want to be fine-tuned. Now here is my data set… and then this is my other configuration."*

**On the automatic endpoint [50:06]:**
> *"If you will check with the endpoint, so you will be getting the endpoint. So automatically it will be deployed and it will give you the end point… automatically the endpoint will be created once the job will be completed."*

**On the wall-clock [50:56]:**
> *"When I executed it first time guys, it took around — you won't believe it — took around guys 15 to 20 minutes. So you will have to wait maybe for 15 to 20 minutes sometime."*

**On the automatic metrics [51:36]:**
> *"Evaluation fraction of correct next step prediction, evaluation number of prediction, evaluation total loss, training loss — right, everything guys you will be able to get over here."*

**On inference [54:53]:**
> *"This is not the pre-trained model, the Jimny model. This is the fine-tuned model actually. I given my data and I fine-tuned the model and I'm accessing it — I'm accessing through the endpoint."*

**On the cost of a demo [30:35] and [30:51]:**
> *"You won't get much cost guys, just 5 to 10 rupees… even in the finetuning also I'll show you very minimum cost, 10 to 15 rupees, that's it."*
>
> *"After fine-tune, clean up everything that is required, otherwise the cost would be there… the cost will be increasing by the time."*

---

## Appendix B — Reference Links & Papers

**Primary sources for this module**

| Source | What it is |
|---|---|
| `D:\Finetuning\_source\transcripts\LLM_Fine-Tuning_21_Google_Gemini_Fine-Tuning_Masterclass_using_Vertex_AI_Supervi.txt` | The transcript (1539 lines). Ground truth for every timestamp above. |
| `D:\Finetuning\_source\repo\Complete-LLM-Finetuning-main\LLM Fine-Tuning-21-GEMINI-Finetuning\gemini_finetuning_clean.ipynb` | The companion notebook. 760 lines of JSON. Contains the runnable cells, several of which raise `NameError` as shipped. |
| `...\LLM Fine-Tuning-21-GEMINI-Finetuning\data.jsonl` | The 10-row mobile-customer-support SFT dataset. 564 tokens, constant system instruction. |
| `...\LLM Fine-Tuning-21-GEMINI-Finetuning\handwritten-notes.pdf` | Scanned notes (not machine-readable without a PDF renderer; not used for the technical content above). |

**Google documentation to verify against (all change frequently — check the date on the page)**

| Topic | What to look for |
|---|---|
| Tune Gemini models (Vertex AI) | The current tunable-model list, the hyperparameter defaults and ranges, and the JSONL sample schema. |
| Vertex AI pricing | The tuning $/1M-token rate **and** the endpoint per-hour / per-node-hour rate for the model generation you are using. Both lines. |
| Vertex AI `TuningJob` REST reference | The exact `supervisedTuningSpec` field names, the `JobState` enum, and the string-vs-number typing of `epochCount`. |
| Vertex AI quotas | Tuning jobs per project, endpoints per region. |
| Gemini API (`ai.google.dev`) model and deprecation pages | The AI Studio deprecation timeline the instructor cites at [7:18]. |
| `google-genai` SDK changelog | Breaking changes in the `tunings` module between the version in the notebook (1.62.0) and current. |
| Google Cloud billing export schema | Column names for the §11.5 cost query. |

**Background reading, and where it lives in this handbook**

| Topic | Module |
|---|---|
| The SFT objective, masking, and the elicitation-vs-injection thesis | CS-13 |
| The sibling managed service and its schema | CS-18 |
| LoRA mechanics, rank selection, adapter mathematics | CS-23 |
| Self-hosted alternatives that keep the artefact | CS-16, CS-17 |
| When to fine-tune at all — the RAG/agent decision | CS-04 |
| Preference tuning (Vertex's second method) | CS-14, CS-24, CS-25, CS-27 |

---

*End of CS-19.*

