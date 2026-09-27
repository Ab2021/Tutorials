# CS-18 — OpenAI GPT Fine-Tuning Masterclass (Hosted SFT)

| Field | Value |
|---|---|
| **Module** | Hosted / API Fine-Tuning (Managed SFT) |
| **Source video(s)** | `LLM Fine-Tuning 20: OpenAI(GPTs) Fine-Tuning Masterclass \| Supervised FT \| Token & Cost Analysis` |
| **Transcript file(s)** | `LLM_Fine-Tuning_20_OpenAIGPTs_Fine-Tuning_Masterclass_Supervised_FT_Token_Cost_A.txt` |
| **Companion code** | `LLM Fine-Tuning-20-GPT-Finetuning/openai_api_and_finetuning_of_gpt_model.ipynb` (the run that matches the video: `gpt-4.1-nano`, `data.jsonl`), `openai_api_and_finetuning_of_gpt_model (1).ipynb` (the later re-run: `gpt-4o-2024-08-06`, same data), `data.jsonl` (10 rows), `data2.jsonl` (10 rows, row 1 deliberately malformed), `hand-wriiten-notes.pdf` |
| **Prerequisites** | CS-01 (lifecycle), CS-02 (transfer learning), CS-13 (SFT — the loss and the masking story), CS-14 (the alignment map), CS-17 (Axolotl — the last open-weight trainer before this module) |
| **Neighbours** | CS-03 (framework landscape — this is the "hosted API" column of Axis 1), CS-19 (Gemini/Vertex — the sibling hosted API), CS-23 (LoRA/QLoRA), CS-09 (distillation) |
| **Difficulty** | Beginner to execute, Intermediate to cost-model, Expert to know when *not* to |
| **Hands-on required** | Yes for the mechanics (~40 lines of Python, no GPU); No for the economics, which is arithmetic on the back of an envelope |
| **Estimated study time** | 4h theory + 2h practical + 2h on the cost model (the cost model is the part that transfers) |

> **Status warning — read this before anything else.** The transcript is a 2025-era walkthrough of a
> platform that is, as of this writing (September 2026), **being shut down**. OpenAI began winding
> self-serve fine-tuning down on **2026-05-07**: organizations that have never run a fine-tuning job
> can no longer create one, the door closed further on **2026-07-02** for organizations that have not
> run *inference* on a fine-tuned model in the trailing 60 days, and **on 2027-01-06 existing active
> customers will no longer be able to create new fine-tuning jobs at all**. Inference on already
> trained fine-tuned models continues until the underlying base model is deprecated. OpenAI's stated
> reason is that newer base models follow instructions and formats well enough that prompting is
> "cheaper and faster". This module therefore teaches the transcript's material as **the canonical
> anatomy of a hosted fine-tuning API** — a shape that Azure OpenAI, Google Vertex AI, Amazon
> Bedrock, Together, Predibase and Fireworks all share to a first approximation — and flags every
> number in it that is now historical. Do not start a new project on this specific API without
> reading §16.8 first.

---

## 0. Executive Summary

- **Hosted fine-tuning gives you an endpoint, not a model.** You upload a JSONL file, you set three
  hyperparameters, you get back a string like
  `ft:gpt-4.1-nano-2025-04-14:personal:second-finetune-model:AbCdEfGh` [55:14]–[55:20]. You never see
  a tensor. There is no `merge_and_unload()`, no `save_pretrained()`, no GGUF export, no LoRA
  adapter you can carry to `vLLM`, no `bitsandbytes` knob, no `device_map`. Every downstream skill in
  this handbook — merging (CS-23), quantising (CS-10, CS-11), serving on your own hardware,
  distilling the result into something smaller (CS-09) — is **structurally impossible** against a
  hosted fine-tune. That single fact decides the use case more often than quality does.
- **The whole API is four calls.** `client.files.create(..., purpose="fine-tune")` →
  `client.fine_tuning.jobs.create(...)` → poll `client.fine_tuning.jobs.list()` → chat against
  `job.fine_tuned_model` [50:37]–[55:07]. Everything else in the video is validation and arithmetic
  you do *before* spending money.
- **The single most important number in this module is the training-token price, and it is not an
  hourly rate.** `gpt-4.1-nano` fine-tuning training is **$1.50 per 1,000,000 training tokens** — not
  "$1.5 per hour of GPU" as the instructor reads it at [47:31]. The instructor's own retraction at
  [38:09]–[38:14] of his earlier inference reading is also wrong (see §4.6). Getting this
  distinction right is the difference between a $0.003 job and a $9 job on the same data.
- **The bill is `billed_tokens × epochs × price_per_token`, where `billed_tokens` is computed for you
  server-side.** You do not get to argue with it. The API counts every token in every message of
  every example, adds per-message overhead tokens, caps each example at a model-specific maximum, and
  multiplies by the number of epochs. The cookbook constants that the video's notebook reproduces
  verbatim are `TARGET_EPOCHS=3`, `MIN_TARGET_EXAMPLES=100`, `MAX_TARGET_EXAMPLES=25000`,
  `MIN_DEFAULT_EPOCHS=1`, `MAX_DEFAULT_EPOCHS=25`, `MAX_TOKENS_PER_EXAMPLE=16385` (notebook cells 50,
  54, 55, 56). Those constants are the *entire* automatic-hyperparameter policy.
- **The instructor's cost arithmetic does not reconcile, and the correct arithmetic is 250× smaller
  than he implies.** His 10-row smartphone dataset bills **672 tokens per epoch** (cookbook token
  counter) or **562 tokens** (his own `tiktoken` counter, notebook cell 44) → **2,016 tokens at 3
  epochs** → **$0.0030** at `gpt-4.1-nano`'s $1.50/M. He reports `$0.702396` total and separately
  `$0.75` of "training cost" for "half an hour" [47:38], [48:43]. The $0.75 comes from reading a
  per-1M-token price as an hourly GPU rate. §4.6 reconstructs both numbers and shows where they
  diverge.
- **At production scale inference dominates, and the hosted inference markup is brutal.** Fine-tuned
  `gpt-4.1-mini` inference is **$0.80/M input, $3.20/M output** versus the base model's
  **$0.40/M input, $1.60/M output** — a clean 2×. A 20-shot base prompt at 1,500 input tokens
  (≈$0.00084/call) beats a fine-tune at 600 input tokens (≈$0.00096/call) **even though the
  fine-tune's prompt is 2.5× shorter**. You need to shrink the prompt below ~450 tokens to break even
  per call. §11.5 does the algebra.
- **The API will overwrite your epochs and refuse your batch size, and it tells you so only in the
  job object.** Ask for `n_epochs` and the server returns whatever its target-example policy decides;
  ask for `batch_size=16` on a 10-row dataset and it silently clamps. The returned
  `job.hyperparameters` is the ground truth, not what you sent [55:09]–[55:47], [1:00:52].
- **`purpose="fine-tune"` plus a validation pass is not the same as a held-out set.** The video's
  `data.jsonl` **repeats three unique Q/A pairs across 10 rows** (§4.2.4) — the dataset has three
  distinct examples, not ten, and there is no validation file at all in either notebook. Every
  "it works!" observation in the video is therefore in-sample. §12 shows what an honest protocol looks
  like.
- **The platform supports four training methods and the video demoes one.** Supervised (SFT), vision
  fine-tuning, Direct Preference Optimization (DPO), and reinforcement fine-tuning (RFT) [1:03]–[1:12].
  The video performs SFT only, and its data is a 10-row customer-support chatbot corpus.
- **STOP conditions that actually matter here:** (1) you need the weights — for on-prem, for a
  distilled student, for quantisation, for a merge — hosted is the wrong answer, full stop; (2) your
  task is "return valid JSON of this shape" and nothing else — that is `response_format` /
  structured outputs, not fine-tuning (§4.12); (3) the knowledge you need changes weekly — that is
  retrieval (CS-04); (4) you are starting a *new* project on this specific API in late 2026 — read
  §16.8 and pick a surviving platform.

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world without this

The failure this module addresses is not "the model is not smart enough". It is **"the model is smart
enough and we cannot ship it"**, along four separable axes:

| Failure | What it looks like | Root cause |
|---|---|---|
| **The prompt tax** | A 2,000-token system prompt with 20 worked examples, re-paid on every single API call, forever. | Few-shot prompting is a runtime cost that never amortises. You are renting the behaviour instead of buying it. |
| **The format violation** | 91% of responses parse. The other 9% page an on-call engineer at 3 a.m. | A general model was never constrained to your schema. |
| **The tone mismatch** | The answer is correct and unusable — it reads like a forum post, not like your support desk. | Style is a distribution over formats, which is exactly what SFT moves (CS-13 §4.1). |
| **The ops load** | You want the behaviour change and you do **not** want to own a GPU, a training framework, a checkpoint registry, or a serving stack. | Open-weight fine-tuning (CS-15, CS-16, CS-17) hands you all four. |

The hosted API is the answer to the fourth problem specifically. The instructor frames the whole
video around it: *"I'm not going to perform any training on my server. So the training will be
performed over the OpenAI server. I just need to configure the training using the Python API"*
[22:04]–[22:14]. He then selects a **CPU-only Colab runtime** [22:17] and never touches a GPU.

That is a genuine, not a cosmetic, advantage. It is also the source of every limitation in §9.

### 1.2 The state of the art before hosted fine-tuning

| Era | How you changed a model's behaviour | Cost of entry |
|---|---|---|
| 2020–2021 | Few-shot prompting GPT-3 in a giant context window | $0 fixed cost, ~$0.02–0.20 per call in prompt tokens |
| 2021–2022 | Fine-tune a task-specific BERT/RoBERTa (CS-07) | A labelled dataset, a GPU, a `Trainer`, a checkpoint |
| 2022–2023 | Fine-tune an open Llama with `peft`/LoRA on one GPU (CS-23) | 24 GB VRAM, ~300 lines of Python, a serving stack |
| **2023-08 →** | **`POST /v1/fine_tuning/jobs`** — hosted SFT, then DPO, then RFT, then vision | **~40 lines of Python, no GPU, no serving stack, a credit card** |
| 2024–2026 | Open-weight tooling closes the gap: `trl`, Unsloth, LLaMA-Factory, Axolotl, vLLM | A rented 24–80 GB GPU for an afternoon, ~$2–40 |
| **2026-05 →** | **Hosted self-serve fine-tuning winds down** | The whole category re-fragments across cloud vendors (§16.8) |

The hosted API's real historical contribution was **removing the last excuse**. It made the answer to
"should we fine-tune?" a question about data and economics rather than about infrastructure. That
contribution outlives the specific endpoint.

### 1.3 The naive approach, and precisely why it fails

**Naive approach: "I have 10 rows of examples and I'll fine-tune a small model so it becomes my
domain expert."**

This is essentially the video's demo, and it fails for five reasons that the demo itself illustrates:

1. **Ten rows is below the floor of useful learning and above the floor of the platform's minimum.**
   The docs require 10 examples to accept the file and *"we see improvements from fine-tuning on
   50–100 examples"* to change behaviour. The instructor gets the second half of this right
   [14:18]–[14:27] and then does 10 rows anyway *"just to demonstrate you"* [16:57]–[17:01]. A model
   trained for 3 epochs on 10 examples has roughly 30 gradient steps of signal. That is a formatting
   nudge, not a knowledge injection.
2. **The three unique examples are the entire learning signal.** `data.jsonl` contains three distinct
   Q/A pairs repeated as 10 rows (§4.2.4). The effective dataset is **3 conversations**. Fine-tuning
   on it teaches the model the *shape* of a smartphone-support reply — which is exactly, and only,
   what the video claims it does when it later says *"you want to tune your GPT model on top of this
   data so that whenever someone is visiting your website, model can reply in a well manner"*
   [1:03:51]–[1:03:58].
3. **There is no held-out evaluation, so the success claim is unfalsifiable.** The chat test at
   [1:03:02]–[1:03:16] asks the model a question whose answer is *in the training set verbatim*
   (the warranty question, row 1). The model answers fluently. That is a memorisation check, not an
   eval (CS-13 §12).
4. **The cost model in the video is wrong by two orders of magnitude in the direction that makes
   fine-tuning look expensive.** $0.0024 is a rounding error you can spend 400 times for a dollar;
   $0.75 is a real decision. §4.6.
5. **The one thing hosted fine-tuning cannot do for you is the thing you will want next.** After the
   model works, you will want it cheaper, or on-prem, or in a batch job on your own GPU, or
   distilled into something 30× smaller. The hosted endpoint can be none of those.

The correct version of the naive plan is: **collect 100–1,000 real support interactions**, hold out
20%, validate the file *before* uploading, run one epoch to a validation file, read the returned
metrics, and *then* decide whether the per-call economics beat a shorter prompt against the base
model (§11.5).

---

## 2. First-Principles Mental Model

### 2.1 The analogy: hosted fine-tuning is a laundry service, not a washing machine

Open-weight fine-tuning (CS-16, CS-17, CS-23) is **buying a washing machine**. You get the machine in
your basement, you control the water temperature, the spin cycle and the detergent, you can open the
door mid-cycle and look at the clothes, and when you move house you take it with you. You also have to
wire it in, fix it when it breaks, and pay for the electricity.

Hosted fine-tuning is a **laundry service**. You drop a bag at the counter (the JSONL upload), tick
three boxes on a form (`n_epochs`, `batch_size`, `learning_rate_multiplier`), and pick up a claim
ticket (`ft:gpt-4.1-nano-2025-04-14:personal:second-finetune-model:AbCdEfGh`). Your clothes come back
clean and pressed. You cannot see inside the machine, you cannot change the detergent, and when you
move to a country the service does not operate in, your clothes stay dirty.

**Where this analogy breaks.** Three places, and they matter:

1. **A laundry has a price list; a hosted fine-tune has a price list *and* a token counter you cannot
   audit.** The counter is deterministic once you know the overhead constants (§4.5), but you are
   billed on the service's count, not yours. A 2% disagreement across 100M tokens is $3 you cannot
   dispute.
2. **You do not get the washed clothes back — you get a *key* to a room where clean clothes appear on
   demand.** The fine-tuned model identifier is a pointer. Delete the job, lose access, and there is
   no `.safetensors` on your disk that survives. This is the difference between an *artefact* and a
   *capability*.
3. **The service can close.** Which, for this specific service, it did (§16.8). A washing machine you
   own does not get a deprecation email.

### 2.2 The mechanism, stated precisely

Hosted SFT runs **exactly the same loss** as open-weight SFT (CS-13 §2.2):

$$L_{SFT} = -\sum_{t \in \mathcal{R}} \log p_\theta(x_t \mid x_{<t}), \qquad x \sim \mathcal{D}_{instruct},\ \ \mathcal{R} = \text{assistant token positions only}$$

What differs is everything *around* the loss:

| Dimension | Open-weight (CS-13/15/16/17/23) | Hosted (this module) |
|---|---|---|
| Who owns $\theta$ | You | The provider |
| What you receive | `adapter_model.safetensors` (MB) or a merged checkpoint (GB) | A model ID string (`ft:...:`), a `FineTuningJob` object |
| Loss masking | Your code decides (`-100` on prompt spans) | The provider decides, and documents it only obliquely — the SFT schema's `assistant` content is the supervised target |
| Data format | Any (`alpaca`, `sharegpt`, `messages`, `prompt/completion`) | **`messages` only**, JSONL, one object per line |
| Hyperparameters | ~40 knobs (LR, scheduler, warmup, optimiser, rank, alpha, target modules, …) | **3** (`n_epochs`, `batch_size`, `learning_rate_multiplier`) |
| Cost unit | GPU-hours | **training tokens** (SFT) or wall-clock hours (RFT) |
| Iteration latency | Minutes (relaunch a script) | Minutes to ~an hour (queue + train), and you re-upload per dataset change |
| Deployment | You build it (`vLLM`, TGI, llama.cpp, Ollama) | `model="ft:..."` — already deployed |
| Reproducibility | Seed + config + code + data hash, all yours | Seed + config + data, plus a provider-side training stack you cannot pin |
| Determinism | Reasonably deterministic on one node | **Not guaranteed**; `seed` is accepted and is documented as best-effort |
| Exit path | Merge, quantise, convert to GGUF, serve anywhere | **None** |

### 2.3 The three-layer abstraction that actually matters

Every hosted fine-tuning API in the industry (OpenAI, Azure OpenAI, Vertex AI, Bedrock, Together,
Predibase, Fireworks) is the same three layers. Learn the shape once and you can move between
vendors in an afternoon.

```
┌────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — INFERENCE SURFACE                                        │
│   model = "ft:gpt-4.1-nano-2025-04-14:personal:my-suffix:AbCdEfGh" │
│   POST /v1/chat/completions   (same endpoint as the base model)    │
│   billed at the FINE-TUNED inference rate (§4.6)                   │
└────────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — TRAINING JOB (asynchronous, server-side)                 │
│   fine_tuning.jobs.create(training_file, model, suffix,            │
│                           method={"type":"supervised",             │
│                                   "supervised":{"hyperparameters": │
│                                       {"n_epochs":…,               │
│                                        "batch_size":…,             │
│                                        "learning_rate_multiplier" │
│                                       }}})                         │
│   → FineTuningJob{ id, status, trained_tokens, hyperparameters }   │
└────────────────────────────────────────────────────────────────────┘
                              ▲
┌────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — DATA CONTRACT                                            │
│   JSONL, one {"messages":[…]} object per line                      │
│   roles: system | user | assistant | function (+ tool_calls, name, │
│          weight in newer schemas)                                  │
│   validated at upload; a malformed line fails the WHOLE file       │
└────────────────────────────────────────────────────────────────────┘
```

> **Beyond the video:** the layer that vendors compete on is Layer 2, and the layer that decides
> whether your project succeeds is Layer 1. Layer 1 is also the only layer that is *portable*: a
> `{"messages":[…]}` JSONL file that trains a `gpt-4.1-mini` will train a Llama-3.1-8B with
> `trl`'s `SFTTrainer` after you apply that model's chat template (CS-13 §4.3). Build your pipeline
> around Layer 1 and the vendor becomes a config value. Build it around Layer 3 and you are locked
> in forever.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Hosted fine-tuning** | Fine-tuning performed on a provider's infrastructure, reached over an HTTP API, where the weights never leave the provider. | The subject of this module; the "hosted API" cell in CS-03's Axis 1. | People say "API fine-tuning" and mean "no training happens". Training happens; you just cannot see it. |
| **JSONL** (JSON Lines, `.jsonl`) | A file where each *line* is one complete, self-contained JSON object, no commas, no enclosing array. | The only accepted training-data format. One bad line fails the file. | The video conflates JSON and JSONL for a full minute [25:55]–[26:10] — he is *describing* JSONL while saying "JSON". `.json` (an array) is rejected. |
| **Example / row / training example** | One JSONL line = one conversation = one training sample. The instructor defines it correctly at [31:01]–[31:24]. | The unit of dataset size. "10 examples" means 10 lines. | Confusing examples with *unique* examples. The video's 10 rows contain 3 unique conversations (§4.2.4). |
| **`messages`** | The single required top-level key: an array of `{role, content}` objects. | The whole schema. | Not `messages[]` with different names, not `conversations`, not `prompt`/`completion` — those are other vendors' schemas. |
| **`role`** | One of `system`, `user`, `assistant`, `function` (legacy) / `tool` (current). Validated at upload. | Determines which turn the tokens belong to and, for `assistant`, whether they are supervised. | Roles are **not** `human`/`gpt` (ShareGPT) or `instruction`/`output` (Alpaca). CS-13's format table covers all five. |
| **`content`** | The message text. Must be a non-empty string (or absent if `function_call`/`tool_calls` carries the payload). | Empty content is a validation error (`missing_content`), not a warning. | `""` fails; a single space passes. |
| **`system` message** | The first message; sets persona and behavioural constraints. Not a training target. | The instructor's 10 rows all share one identical system prompt [54:00]–[54:05]. | It is a *conditioning* message, not a *supervised* one — but the whole system prompt is still billed as training tokens. |
| **`assistant` message** | The model's turn; this is the supervised target. | The rule that actually matters: **an example with no `assistant` message contributes no learning signal.** The validator enforces `example_missing_assistant_message`. | The video's own `data2.jsonl` fails exactly this check [36:53]–[37:08]. |
| **`function` / `tool` role** | Legacy way to return a function result to the model. Superseded by the `tool` role + `tool_calls`. | Only relevant if you fine-tune tool use. The instruction explicitly scopes this out: *"in our finetuning we are not going to fine-tune our model on a specific tool data… I'm just going to be fine tuned on a simple chatting"* [15:52]–[16:00]. | The validator accepts `function` (video era) but a modern file should use `tool`. |
| **`name`** | Optional per-message speaker name; costs 1 extra token in the counter. | Distinguishes multiple participants in a multi-user conversation. | Rarely needed; adds tokens to the bill for nothing. |
| **`weight`** | Optional per-message supervision weight in newer schemas. | Lets you up/down-weight individual turns without dropping them. | Not present in the video-era validator; if you include it, the video's validator flags `message_unrecognized_key`. |
| **Format validation** | A client-side pre-flight pass over the JSONL that counts schema violations by category before you upload. | Cheap, local, and catches the failures that would otherwise cost you an upload and a round trip. §4.2.3 has the seven checks verbatim. | Validation checks *shape*, never *quality*. A dataset that passes 100% can still be worthless. |
| **`defaultdict(int)`** | A `dict` subclass whose missing keys default to `0`, so `counts["x"] += 1` never raises `KeyError`. | The idiomatic way to build the error histogram. The notebook demonstrates the `KeyError` first (cell 14) then the fix (cells 17–23). | Not a "type". It is a factory-backed dict; `defaultdict(list)` gives `[]`, `defaultdict(int)` gives `0`. |
| **`tiktoken`** | OpenAI's open-source BPE tokeniser library. `encoding = tiktoken.get_encoding("cl100k_base")` [39:47]–[40:03]. | The only way to count tokens locally before you are billed for them. | The instructor calls it a "transformer based model" [40:14]–[40:17]. It is **not** a neural network — see the Correction in §4.5.1. |
| **`cl100k_base`** | The ~100k-merge byte-pair vocabulary used by `gpt-4`/`gpt-3.5-turbo`/embeddings-v2 era models. | The tokeniser the video's counter uses. | `o200k_base` is the vocabulary for the `gpt-4o`/`gpt-4.1` families. **Using `cl100k_base` to estimate `gpt-4.1-*` costs is a small systematic error** (§4.5.4). |
| **Training tokens** | All tokens in the dataset, **× epochs**, capped per example at the model limit. | The unit you are billed in for SFT. | Not the same as *dataset* tokens. `n_epochs` multiplies the bill linearly. |
| **Inference tokens** | Input + cached-input + output tokens on every call to the fine-tuned model. | The recurring cost, and at production volume the larger one. | The pricing page lists these in the *same table* as training price, which is exactly what confused the instructor. |
| **`MAX_TOKENS_PER_EXAMPLE`** | `16385` in the cookbook code the video reproduces. Examples longer than this are **truncated**, not rejected. | Silent data loss on long examples. | 16385 is a 2024 constant for a 16k-context model family. For the `gpt-4.1` line-up it is stale (§4.5.5). |
| **`TARGET_EPOCHS` / `MIN_TARGET_EXAMPLES` / `MAX_TARGET_EXAMPLES` / `MIN_DEFAULT_EPOCHS` / `MAX_DEFAULT_EPOCHS`** | The five constants of the automatic epoch policy: 3 / 100 / 25000 / 1 / 25. | They are the *entire* automated hyperparameter machinery. §4.8 works through the arithmetic. | These are **cookbook reference code**, not API behaviour you can rely on being identical today. Always read back `job.hyperparameters`. |
| **`n_epochs`** | Number of full passes over the dataset. Also the linear multiplier on your training bill. | The single highest-leverage knob for cost. | "auto" is resolved by the server; the resolved value is what you pay for. |
| **`batch_size`** | Examples per forward/backward step in the provider's trainer. | Interacts with the learning rate and with dataset size. | The instructor calls it "how many batches we are passing the data in, like in one batch we are passing 16 rows" [55:11]–[55:19] — the *direction* is right, the vocabulary is muddled (batch vs micro-batch). |
| **`learning_rate_multiplier`** | A **multiplier** on the provider's internally chosen base learning rate — not a learning rate. | 1.0 means "use the provider's default"; 2.0 means "twice whatever that is". | You cannot express an absolute LR here. If you do not know the base LR, you do not know your actual LR. |
| **`suffix`** | A human-readable tag appended to the fine-tuned model name so you can tell jobs apart. | The video uses `"first finetune model"` and `"second-finetune-model"` [53:52]–[54:16], [cell 76]. | It is a *label*, not the model ID. The ID also contains your org name and a random hash. |
| **`finetuned_model` string** | `ft:<base>:<org>:<suffix>:<hash>`. E.g. `ft:gpt-3.5-turbo-0125:personal:first-finetune-model:B48axNSg` [cell 20]. | This is your deployment handle. | The video's chat test 404s because the job is not finished — a genuinely instructive failure (§14, row 1). |
| **`FineTuningJob`** | The job object: `id`, `status`, `trained_tokens`, `hyperparameters`, `fine_tuned_model`, `error`, `seed`. | `trained_tokens` is the number you are billed for. Poll it. | `fine_tuned_model` is `null` until the job succeeds. `null` is not an error. |
| **Job status machine** | `validating_files` → `queued` → `running` → `succeeded` / `failed` / `cancelled`. | Drives your polling loop and your alerting. | The video observes `validating_files` [cell 78] and a prior `cancelled` [cell 79] without naming the machine. |
| **Validation file** | An optional second upload used to compute holdout loss during training. | The only automatic overfitting signal the hosted API gives you. | Absent from both of the video's notebooks. Without it, `metrics` are train-set only. |
| **DPO (Direct Preference Optimization)** | Preference fine-tuning from `chosen`/`rejected` pairs on the same hosted surface [5:53]–[6:00]. | The hosted answer to CS-14's preference stage. | Not the same data format as SFT — different key names, and the video does not demonstrate it. |
| **RFT (reinforcement fine-tuning)** | Grader-based RL fine-tuning on the hosted API (`o4-mini` class), *"a grader… you can manually grade the output whether it is a good or bad… otherwise you can use LLM as a grader"* [18:46]–[18:56]. | The hosted answer to CS-26's reasoning-RL stage. | The instructor calls it "RLHF" [1:08] and "reinforcement learning". It is **not** classical RLHF — there is no reward model, there is a *grader*. |
| **Structured outputs / `response_format`** | Server-enforced JSON-schema-constrained decoding. | The correct tool for "always emit JSON of shape X" — **not** fine-tuning. §4.12. | Fine-tuning *biases* format compliance; structured outputs *guarantees* it. Different guarantees. |
| **Data retention (API)** | By default, API and fine-tuning data is **not** used to train OpenAI's models; fine-tune data is retained until you delete the file; API data is retained ≤30 days for abuse monitoring. | The compliance answer for a regulated customer. | Consumer ChatGPT data *is* used for training unless you opt out. Business/API data is not. §4.13. |
| **Data-sharing opt-in** | An explicit, org-owner-enabled flag that lets the provider train on your data in exchange for discounted or complimentary tokens. | The one case where "we do not train on your data" stops being true. | Zero-Data-Retention organisations cannot opt in at all. |
| **Hosted-to-open distillation** | Using the hosted fine-tuned model as a *teacher* to label data, then training a small open student on those labels (CS-09). | The escape hatch that recovers an artefact from an endpoint-only model. §4.11. | It is a legal/ToS question before it is a technical one: many providers forbid using outputs to train competing models. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 Hosted vs open-weight: the difference is not performance, it is the artefact

The instructor's own framing of the video's value proposition is the cleanest statement of the trade:
*"in the previous video I showed you how you can fine-tune any LLM using Axolotl, which is low code
and no code. And now from this video onwards we are going to start API based fine-tuning"*
[0:16]–[0:30]. He positions hosted as the *next* rung of convenience after low-code open-weight
tooling. That ordering is right. What he never states is what you give up to climb it.

#### 4.1.1 The six capabilities you lose

| Capability | What it means in open-weight | Why it is impossible hosted |
|---|---|---|
| **Weight access** | `model.state_dict()`, `peft` adapters, `safetensors` on your disk. You own the bytes. | The weights never leave the provider's cluster. You receive a string. |
| **Merging** | `model.merge_and_unload()` folds a LoRA into the base for a single deployable artefact (CS-23). | There is no adapter to merge. There is nothing to fold into. |
| **Quantisation control** | Your choice of NF4 during training (QLoRA), GPTQ/AWQ for serving, GGUF for CPU/edge (CS-10, CS-11). | The provider serves at its own precision, on its own hardware. You cannot produce a 4-bit version. |
| **Local / on-prem serving** | `vLLM`, `SGLang`, `TGI`, `llama.cpp`, Ollama — any of them, on any GPU you own, at zero marginal cost per token. | Hard requirement: a network round trip to a third party for every token generated. Air-gapped deployments are impossible. |
| **Distillation from the result** | Use the fine-tuned model as a teacher to train a 0.5B student you then own (CS-09). | You *can* do this over the API — it is just an inference workload — but you are paying hosted inference prices per teacher token, and the provider's terms may forbid it (§4.11). |
| **Full hyperparameter control** | Rank, alpha, target modules, LR schedule, warmup ratio, optimiser, `max_seq_length`, packing, gradient checkpointing, `neftune_noise_alpha` (CS-13 §7). | Three knobs, one of which is a *multiplier*. That is the entire surface. |

#### 4.1.2 The four costs you take on

| Cost | Magnitude | Where you feel it |
|---|---|---|
| **Per-token inference markup** | Fine-tuned inference is ~1.5–2× the base model's per-token rate, forever. | Every call, for the life of the deployment. §11.5. |
| **No batch economics on your side** | You cannot amortise by running a bigger GPU batch. You pay per token regardless of utilisation. | High-volume, low-latency workloads. |
| **Data egress and retention policy dependence** | Your training data sits on someone else's disk until you delete it. | Procurement, DPIAs, regulated industries. §4.13. |
| **Platform risk** | Deprecation, model retirement, pricing change, or shutdown. For this platform, all four. | The moment you have a product depending on the model ID. §16.8. |

#### 4.1.3 Where hosted genuinely wins

Being even-handed matters here, because the "just self-host everything" reflex is also wrong:

| Situation | Hosted wins because |
|---|---|
| You have no GPU budget and no MLOps person | Total fixed cost is a credit card. |
| You need to ship this sprint | Upload → job → deploy is under an hour of work, and the serving is already done. |
| Your volume is spiky and low | Pay-per-token beats paying for an idle A100 at $1.50–3.00/hr. |
| You need the *specific* base model's capabilities | You cannot get `gpt-4.1`-class quality from an 8B open model at the same context length. |
| You are prototyping whether fine-tuning helps *at all* | Cheapest possible experiment. Do it, learn the answer, then decide where the production run lives. |
| Compliance forbids GPU procurement for your team | The provider's SOC 2 / DPA is easier to sign than a GPU cluster is to stand up. |

#### 4.1.4 The decision in one line

```
Do you need the WEIGHTS (on-prem, quantise, merge, distil, embed in a product you ship)?
├── YES → hosted is disqualified. Go to CS-13 (data) → CS-15/16/17 (train) → CS-10/11 (quantise).
└── NO  → Do you need it cheaper than a shorter prompt against the base model?
          ├── YES → hosted is probably still disqualified (§11.5 — the 2× markup is hard to beat).
          └── NO  → hosted is the right answer, and this module is the how.
```

#### 4.1.5 The "endpoint, not weights" consequence nobody mentions

There is a second-order effect that costs teams weeks: **a hosted fine-tune breaks your evaluation
infrastructure's assumption that a model is a file.**

| Eval-harness assumption | Open-weight reality | Hosted reality |
|---|---|---|
| `model_path` is a directory | True | False — it is a string, and it is *scoped to your org* |
| A checkpoint can be copied to a test runner | `cp -r` | Impossible |
| An eval can run offline / in CI | True | Needs a network call and a live API key in CI |
| A regression suite can pin a model version | `revision="step-1200"` | You pin a model ID, and the provider can retire it |
| A model can be deleted and restored from a bucket | True | Deleting the job loses the model |
| Cost per eval run is ~$0 (your own GPU) | True | Every eval run is a billed inference call |

> **Beyond the video:** the practical fix is to treat the fine-tuned model ID as a *versioned
> configuration value*, not as a code constant. Put it in a config table with `created_at`,
> `base_model`, `training_file_id`, `trained_tokens`, and the eval score, and have your eval harness
> read it from there. When (not if) the provider retires the base model, you have a one-row migration
> instead of a grep across the repo.

---

### 4.2 The JSONL schema and its validation rules

This is the section to memorise. Everything else in the module is arithmetic and policy; the schema
is the part you will use on day one and the part that fails silently.

#### 4.2.1 The schema, minimally and maximally

**Minimal valid line** (this is what the API needs):

```json
{"messages":[{"role":"user","content":"How long does the warranty last on a new smartphone?"},{"role":"assistant","content":"Most smartphones include a one-year limited warranty that covers manufacturing defects."}]}
```

**The shape the video actually uses** — `system` + `user` + `assistant`, one conversation per line,
all ten lines carrying the identical system prompt (`data.jsonl`, rows 1–10):

```json
{"messages":[{"role":"system","content":"You are a customer support assistant for a smartphone company. You are friendly, concise, and provide only factual answers related to smartphones."},{"role":"user","content":"How long does the warranty last on a new smartphone?"},{"role":"assistant","content":"Most smartphones include a one-year limited warranty that covers manufacturing defects. Exact details are available in the official documentation."}]}
```

**Maximal shape** (multi-turn, with a tool call — supported, not demonstrated):

```json
{"messages":[
  {"role":"system","content":"You are a support agent with access to order lookup."},
  {"role":"user","content":"Where is order 88231?"},
  {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup_order","arguments":"{\"id\":\"88231\"}"}}]},
  {"role":"tool","tool_call_id":"call_1","content":"{\"status\":\"in_transit\",\"eta\":\"2026-10-02\"}"},
  {"role":"assistant","content":"Order 88231 is in transit and should arrive by 2 October."}
]}
```

The instructor walks the shape out loud and gets it right [15:19]–[15:52]: *"the key name is the
message and the value will be the list. Now inside this list you will be having the different
different role like user, assistant, system… and then you can keep the content."* He then points out
that `content` can be a tool call rather than a string [15:42]–[15:49] — correct.

#### 4.2.2 The seven validation checks, verbatim from the notebook

The notebook's markdown table (cell 25) is the cookbook's check list. Reproduced exactly, because
the error keys are what you will see in your own validator's output:

| # | Validation check | Code condition | What it ensures | Error key raised |
|---|---|---|---|---|
| 1 | Example type | `isinstance(ex, dict)` | Each JSONL line is a valid JSON object | `data_type` |
| 2 | `messages` key exists | `ex.get("messages")` | Conversation structure is present | `missing_messages_list` |
| 3 | Required keys in message | `"role"` and `"content"` | Every message follows chat format | `message_missing_key` |
| 4 | No extra keys | allowed keys only | Prevents unsupported fields | `message_unrecognized_key` |
| 5 | Valid role names | `system` / `user` / `assistant` / `function` | No invalid or typo roles | `unrecognized_role` |
| 6 | Valid content | non-empty string or `function_call` | Message text is usable | `missing_content` |
| 7 | Assistant present | at least one `assistant` role | Supervised learning signal exists | `example_missing_assistant_message` |

Note what is **not** on the list, and why each omission bites:

| Missing check | Consequence |
|---|---|
| Duplicate detection | The video's 10 rows are 3 unique conversations and nothing flags it. |
| Class/behaviour balance | A dataset that is 95% refusals produces a model that refuses (§14). |
| Length distribution | One 30,000-token example can eat your truncation budget silently. |
| Train/validation overlap | Your "holdout" leaks and your eval is fiction. |
| Encoding | A non-UTF-8 file uploads and fails server-side. The video's file-open uses `encoding="utf-8"` [27:20]–[27:27]. |
| PII scan | Nobody in the pipeline will ever tell you that you fine-tuned on a customer's phone number. |

> **Beyond the video:** build this validator as a **CI gate that exits non-zero**, not as a notebook
> cell that prints "No errors found". A validator you have to remember to run is a validator that
> runs on the one dataset where you already checked by hand. Wire it to `pytest` with a fixture file
> of deliberately-broken rows — the notebook already ships one (`data2.jsonl`) — and assert that each
> check fires.

#### 4.2.3 The validator, annotated

The notebook's implementation (cell 27). Comments added; the logic is unchanged.

```python
# notebook cell 26-27 — verbatim structure
from collections import defaultdict

format_errors = defaultdict(int)          # histogram of violation type → count

for ex in data:                           # data = [json.loads(line) for line in file]
    # --- CHECK 1: is this JSONL line even a JSON object? ---------------------
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue                          # nothing below can run on a non-dict

    # --- CHECK 2: is the `messages` key present and non-empty? ---------------
    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue                          # no conversation, no further checks

    for message in messages:
        # --- CHECK 3: required keys -----------------------------------------
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        # --- CHECK 4: no unrecognised keys ----------------------------------
        # NOTE: this allow-list is the 2024 list. `tool_calls` and `tool_call_id`
        # are absent, so a modern tool-use dataset will be flagged here.
        if any(k not in ("role", "content", "name", "function_call", "weight")
               for k in message):
            format_errors["message_unrecognized_key"] += 1

        # --- CHECK 5: role must be one of four ------------------------------
        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1

        # --- CHECK 6: content must be a non-empty string, unless it is a call
        content = message.get("content", None)
        function_call = message.get("function_call", None)
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    # --- CHECK 7: at least one supervised turn ------------------------------
    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")
```

Two things about this code are load-bearing and easy to miss:

1. **Checks 3, 4, 5 and 6 are inside one loop and none of them `continue`.** One malformed message
   can increment up to four counters. Your error histogram is therefore *not* a count of *examples*
   affected — it is a count of *violations*. The instructor reads the output as if it were per-example
   ("in all 10 example at one place… this assistant is missing" [37:01]–[37:08]) and in that specific
   case he is right, because `example_missing_assistant_message` sits outside the loop.
2. **Check 6's condition `(not content and not function_call)` treats `""` and `None` identically.**
   So does `0`, `[]` and `{}`. If you ever generate content programmatically, a falsy-but-intended
   value silently becomes `missing_content`.

> **Correction:** at [31:57]–[32:00] the instructor describes check 3 as *"we are checking the
> required key. Okay, role and content"* and then at [32:00]–[32:02] *"then we are checking there is
> no extra key"* — both correct — but he never says the thing that matters most about check 4: **the
> allow-list is stale.** It permits `name`, `function_call` and `weight` and rejects `tool_calls`,
> `tool_call_id` and `refusal`, which are all valid in the current chat schema. A team that adopts
> this validator unmodified and then fine-tunes a tool-using model will see
> `message_unrecognized_key` on every single row and conclude their data is broken when their
> *validator* is. Fix: drive the allow-list from a constant, and update it deliberately.

#### 4.2.4 `data.jsonl` vs `data2.jsonl` — and the third dataset nobody mentions

The instructor deliberately ships a second file with a defect so you can see the validator fire. The
defect is real and the demonstration works [36:08]–[37:08]. But there are **three** datasets in play
across the two notebooks, and they are not the same file. That matters, because the video's headline
chat demo runs against the *worst* of the three.

Verified by counting the files directly (script in §12.2):

| Dataset | Where | Bytes | Rows | Unique conversations | Unique `user` turns | Missing `assistant` | Used for |
|---|---|---|---|---|---|---|---|
| `data.jsonl` (repo) | `LLM Fine-Tuning-20-GPT-Finetuning/data.jsonl` | 4,239 | 10 | **10** | 10 | 0 | Second notebook's job (`file-Hn8ooHUNEzwcPjBasULS25`, `gpt-4o-2024-08-06`) |
| `data2.jsonl` (repo) | same folder | 4,060 | 10 | 10 | 10 | **1** (line 1) | Nothing. It exists to make the validator fail. |
| *inline dataset* | first notebook, cell 7; uploaded as `file-PsuEt1x4gLPqY8f4sDD49t` | 6,429 | 10 | **4** | **4** | 0 | The first notebook's job, and the model the video chats with at [1:03:02] |

The inline dataset in the first notebook is the redundant one. Its ten rows are four questions —
warranty, apps-on-both-platforms, holiday discount, repair — repeated to fill ten lines, with an
identical system prompt every time. The repo's `data.jsonl` is the clean one: ten distinct support
questions, including the "can you recommend a good laptop" out-of-scope probe that teaches the
refusal behaviour.

So the picture is:

- The validator's "No errors found" verdict [35:10]–[35:13] is correct for the file it was run on.
- The file the video *first* trained on has an effective size of **four** examples.
- The file with a genuine, instructive defect (`data2.jsonl`) is the one that would have taught the
  most, and it is never uploaded.

The lesson is not that the video is careless. It is that **the validator measures the wrong thing
entirely**: it can tell you a file is *legal*, never that a file is *good*, and it says nothing at all
about redundancy. A 6,429-byte file of four repeated conversations passes every one of the seven
checks, and so does a 4,239-byte file of ten distinct ones.

> **Beyond the video:** add an eighth check the platform will never run for you — a **uniqueness
> ratio**. Compute
> `len({tuple((m["role"], m["content"]) for m in ex["messages"]) for ex in data}) / len(data)`
> and refuse to train below ~0.9. On the repo's `data.jsonl` that ratio is 1.0 (fine); on the first
> notebook's inline dataset it is 0.4, and it would have stopped the job. On a real 1,000-row support
> export, a ratio below 0.9 almost always means your export job ran twice, your dedup key is wrong,
> or your synthetic generator collapsed into a loop.

> **Beyond the video — the check that would have caught this at the door:** uniqueness is necessary
> but not sufficient. A dataset can be 100% unique and still teach nothing, if every example is the
> same *task* with different surface content. Compute two more numbers before you spend money:
> **(a) task entropy** — cluster the assistant turns by embedding and count clusters; ten clusters
> from 1,000 rows means your data is one behaviour repeated. **(b) label balance** — the fraction of
> rows that are refusals, the fraction that are tool calls, the fraction that are answers. A dataset
> that is 95% answers produces a model that never refuses; a dataset that is 40% refusals produces
> the over-refusing model of CS-13 §4.9.3. Neither number is visible anywhere in this module's
> tooling, and both cost about twenty lines of Python.

---

### 4.3 The message roles and the system message

#### 4.3.1 What each role means to the trainer

The hosted trainer converts each message into a token sequence with role delimiters, and computes
cross-entropy on the `assistant` spans. The roles behave as follows:

| Role | Tokenised as | In the loss? | Billed as training tokens? | What it actually does |
|---|---|---|---|---|
| `system` | A leading turn, delimited | **No** | **Yes** | Sets persona and hard constraints for the whole conversation. |
| `user` | A turn, delimited | **No** | **Yes** | The conditioning input. |
| `assistant` | A turn, delimited; terminator included | **Yes** | **Yes** | The only thing the model is taught to produce. |
| `function` / `tool` | A turn, delimited | No (it is context) | Yes | Returns tool output to the model. |

Two consequences that surprise people:

1. **You pay for tokens you do not learn from.** The system prompt in the video's data is 41 tokens
   and is identical on all ten rows, so at `n_epochs=3` you pay for 123 tokens of system prompt that
   teach nothing. On a real dataset with a 1,500-token system prompt and 50,000 rows over 3 epochs
   you would pay for **225 million tokens** of pure repetition — at `gpt-4.1-mini`'s $3.00/M that is
   **$675 for boilerplate**. §4.6.4 works this through.
2. **The system message is part of the trained context, so it becomes part of the model's identity.**
   Every row sharing one system prompt means the resulting model is *conditioned* on that prompt
   appearing. Change the system prompt at inference and you are off-distribution. This is the hosted
   analogue of CS-13's train/serve template mismatch, and it is the number one silent failure in
   this module (§9.4).

#### 4.3.2 The system-message discipline the video gets right, and the part it skips

The instructor's data uses a single, well-formed system prompt with three properties worth copying
(`data.jsonl`, all rows):

```text
You are a customer support assistant for a smartphone company.
You are friendly, concise, and provide only factual answers related to smartphones.
```

| Property | Present? | Why it matters |
|---|---|---|
| **Role** ("customer support assistant") | Yes | Names the persona the model should adopt. |
| **Style** ("friendly, concise") | Yes | Two adjectives that the model can actually act on. |
| **Scope constraint** ("only… related to smartphones") | Yes | This is what produces the refusal on row 9 ("Can you recommend a good laptop for work?" → *"I'm here to assist only with smartphone-related questions."*) |
| **Length guidance** | No | "Concise" is vague. A token-count target is not. |
| **Output format** | No | No schema, no markdown policy. The model will produce prose. |
| **Escalation rule** | No | Nothing tells it what to do when it does not know. |

Row 9 is the most valuable row in the dataset and deserves to be called out. It is a **negative
example with a soft refusal** — the only row that teaches the model what *not* to do. Everything else
is a positive example. If you keep one idea from this subsection: **the marginal value of a dataset
is in its negative space.** A dataset of 1,000 in-scope questions teaches the model to answer. A
dataset of 900 in-scope questions plus 100 well-crafted out-of-scope probes teaches it *when not to*.

> **Beyond the video:** the instructor never mentions that the system message must be *identical*
> in training and serving, and his chat-test code repeats it by hand-typing the same string
> [1:02:53]–[1:02:55]. That is a latent bug. Store the system prompt once as a constant, use it in
> the data-generation script and in the inference call, and add an assertion that every training row
> carries exactly that string:
>
> ```python
> SYSTEM_PROMPT = ("You are a customer support assistant for a smartphone company. "
>                  "You are friendly, concise, and provide only factual answers "
>                  "related to smartphones.")
>
> def assert_system_prompt(data, expected=SYSTEM_PROMPT):
>     bad = [i for i, ex in enumerate(data)
>            if not ex["messages"] or ex["messages"][0]["role"] != "system"
>            or ex["messages"][0]["content"] != expected]
>     if bad:
>         raise ValueError(f"{len(bad)} rows have a non-canonical system prompt: {bad[:10]}")
> ```
>
> On a 10-row demo this is pedantry. On a 50,000-row dataset assembled from four different export
> scripts over six months, it is the difference between a working model and a week of debugging.

#### 4.3.3 Multi-turn conversations and where the supervision lands

The video's data is strictly single-turn (system → user → assistant). Multi-turn is supported and
changes the economics:

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"turn 1 question"},
  {"role":"assistant","content":"turn 1 answer"},
  {"role":"user","content":"turn 2 follow-up"},
  {"role":"assistant","content":"turn 2 answer"}
]}
```

Here **both** assistant turns are supervised, and both user turns are billed but not learned from.
Three practical consequences:

| Consequence | Detail |
|---|---|
| A 6-turn conversation counts as **one example**, not six | This makes your `MIN_TARGET_EXAMPLES=100` threshold misleading: 100 six-turn conversations is 600 supervised turns. |
| Your "average tokens per example" explodes | The video measures ~56 tokens/example on single-turn data [45:51]–[45:53]. Real multi-turn support conversations average 400–1,200 tokens. Budget accordingly. |
| Long conversations hit the per-example cap | Anything above the model's fine-tuning context is truncated. Which end gets truncated is provider-specific and usually the *end* — which means your final assistant turn, the one you most want supervised, may be silently cut. §4.5.5. |

> **Beyond the video:** for multi-turn data, always compute a **truncation histogram** — the
> percentage of examples above 50%, 75%, 90% and 100% of the per-example cap. If more than 2% of your
> data sits above 90% of the cap, restructure (split long conversations into windows with overlapping
> context) rather than shipping a dataset where a random 2% of your supervision is chopped off. This
> is the single most common silent quality loss in hosted fine-tuning, and no vendor will warn you.

---

### 4.4 The base-model line-up, and the fact that it churns

#### 4.4.1 What the pricing page showed him

At [19:52]–[20:06] the instructor reads the fine-tuning model list off the pricing page. His spoken
list, as transcribed, is:

> *"these many model we can finetune like 04 mini 04 mini okay this is the date which when this model
> was like published then here is a GPD 4.1 GPD 4.1 mini nano 40 mini and the other model also"*
> [19:52]–[20:06]

Decoded, that is: `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o-mini` — with the
"date" column being the model snapshot date in the ID. He then chooses to demo the cheapest:
*"in this video I will show you the finetuning using this GPT 4.1 nano because it is having the
minimum pricing"* [20:58]–[21:02].

#### 4.4.2 The base-model line-up, with the churn made explicit

| Model | Fine-tuning era | Status as of Sept 2026 | The video's use |
|---|---|---|---|
| `babbage-002`, `davinci-002` | 2023-08 → 2024 | Retired | — |
| `gpt-3.5-turbo-0613` / `-1106` / `-0125` | 2023-08 → 2024/25 | **Retired** from fine-tuning | First notebook's job; the model the video chats with at [1:03:02] |
| `gpt-4o-mini-2024-07-18` | 2024-08 → 2025 | Deprecated for fine-tuning | Named by the instructor at [19:52] |
| `gpt-4o-2024-08-06` | 2024-08 → 2025 | Deprecated for fine-tuning | Second notebook's job (`cell-78`), commented-out base in `cell-4` |
| **`gpt-4.1-2025-04-14`** | 2025-04 → sunset | SFT available on the wind-down track | Named at [20:06] |
| **`gpt-4.1-mini-2025-04-14`** | 2025-04 → sunset | SFT available on the wind-down track | Named at [20:06] |
| **`gpt-4.1-nano-2025-04-14`** | 2025-04 → sunset | SFT available on the wind-down track | **The video's demo model** [54:44]–[54:47] |
| `o4-mini-2025-04-16` | 2025-04 → sunset | **Reinforcement** fine-tuning only, billed **per hour** | Not used |

Three observations from that table:

1. **The video's demo model is one of three survivors, and all three are on the sunset track.** The
   instructor could not have known this in 2025; he was reading a live pricing page correctly.
2. **The first notebook's `gpt-3.5-turbo` job is now impossible to reproduce on this platform.**
   Anyone following the repo as-is will get an error, not a model.
3. **`o4-mini` is billed differently** — reinforcement fine-tuning is priced per hour of training
   compute, not per token. That is a genuinely different cost model and it belongs in the same
   sentence as `$100/hour` whenever someone claims "fine-tuning is cheap".

> **Correction:** at [1:03]–[1:12] the instructor lists the platform's supported methods as *"SFT,
> vision fine-tuning, DPO, RLHF"* and again at [5:55]–[6:04] *"direct preference optimization… DPO
> reinforcement finetuning."* The fourth method is **reinforcement fine-tuning (RFT)**, not RLHF.
> The distinction is not pedantry: classical RLHF (CS-24) trains a separate reward model on human
> preference pairs and then runs PPO against it; RFT takes a **grader** — a Python function, a
> unit-test harness, or an LLM judge — and optimises directly against its score. There is no reward
> model, no preference dataset, and no PPO. The instructor himself describes the grader correctly at
> [18:46]–[18:56] (*"we required few additional thing like a grader… you can manually grade the
> output whether it is a good or bad, otherwise you can use the LLM model"*) without connecting it to
> the mislabel. In an interview, saying "hosted RLHF" when you mean RFT is an immediate tell.

#### 4.4.3 How to choose a base model when the line-up is churning

The transcript's advice is implicit (*"pick nano because it's cheapest"*) and correct as far as it
goes. The production version is a five-question filter:

| Question | Why | What to do |
|---|---|---|
| **Is the base model you want still on the fine-tuning list?** | Most models are inference-only. | Check the current fine-tuning model table, not the pricing page — they differ. |
| **What is your per-example token budget?** | A 1M-context base model does not mean a 1M-context *fine-tuning* context. | Measure your p95 example length, then confirm the fine-tuning context covers it. |
| **Is the smallest model that works the right choice?** | Capacity, not price, is usually the binding constraint. | Run a 200-row SFT on nano *and* mini *and* a base-model prompt. Compare on a held-out set before committing. |
| **What is the inference-price ratio?** | Nano FT inference is 2× nano base. Mini FT inference is 2× mini base. | Compute the break-even prompt length (§11.5) for each candidate. |
| **When does this model retire?** | The base model's deprecation date *is* your product's end date. | Get it in writing from the provider's deprecations page and put it in your roadmap. |

> **Beyond the video:** a rule the transcript never states — **always fine-tune the smallest model
> that passes your evaluation, and always evaluate the smaller one first.** The reasoning is
> economics, not capability: a `gpt-4.1-nano` fine-tune that hits 94% on your task costs 2.5× less to
> train and ~4× less to serve than a `gpt-4.1-mini` fine-tune that hits 96%. Those two points of
> quality only matter if you have a business justification for them, and you usually do not. The
> inverse error — starting with the flagship because "quality" — costs 10–20× per experiment and
> slows your iteration loop to the point where you stop iterating.

<!-- CONTINUE -->
