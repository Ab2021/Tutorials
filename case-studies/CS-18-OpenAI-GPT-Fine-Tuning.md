# CS-18 — OpenAI GPT Fine-Tuning Masterclass (Hosted SFT)

| Field | Value |
|---|---|
| **Module** | Hosted / API Fine-Tuning (Managed SFT) |
| **Source video(s)** | `LLM Fine-Tuning 20: OpenAI(GPTs) Fine-Tuning Masterclass \| Supervised FT \| Token & Cost Analysis` |
| **Transcript file(s)** | `LLM_Fine-Tuning_20_OpenAIGPTs_Fine-Tuning_Masterclass_Supervised_FT_Token_Cost_A.txt` |
| **Companion code** | `LLM Fine-Tuning-20-GPT-Finetuning/openai_api_and_finetuning_of_gpt_model.ipynb` (34 KB — the **abandoned** earlier run: `gpt-3.5-turbo`, a 4-unique-row inline dataset, auto hyperparameters resolving to `n_epochs=10`, and a 404 at inference), `openai_api_and_finetuning_of_gpt_model (1).ipynb` (96 KB — **the run the video narrates**: validator, tiktoken, both token counters, the epoch policy, the billing arithmetic, the `gpt-4o-2024-08-06` job), `data.jsonl` (10 rows, 10 unique, 0 missing assistant), `data2.jsonl` (10 rows, 10 unique, line 1 deliberately malformed), `hand-wriiten-notes.pdf` |
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
| **`gpt-4.1-nano-2025-04-14`** | 2025-04 → sunset | SFT available on the wind-down track | **Named by the instructor as his demo model** [54:44]–[54:47] — but see the discrepancy note below |
| `o4-mini-2025-04-16` | 2025-04 → sunset | **Reinforcement** fine-tuning only, billed **per hour** | Not used |

Three observations from that table:

1. **The video's named demo model is one of three survivors, and all three are on the sunset track.**
   The instructor could not have known this in 2025; he was reading a live pricing page correctly.
2. **The first notebook's `gpt-3.5-turbo` job is now impossible to reproduce on this platform.**
   Anyone following the repo as-is will get an error, not a model.
3. **`o4-mini` is billed differently** — reinforcement fine-tuning is priced per hour of training
   compute, not per token. That is a genuinely different cost model and it belongs in the same
   sentence as `$100/hour` whenever someone claims "fine-tuning is cheap".

> **Beyond the video — the narration and the notebook disagree about which model was trained, and you
> should know which one to trust for what.**
>
> | Source | Says | Where |
> |---|---|---|
> | The narration | *"This one GPD 4.1 nano 2025414"* — i.e. `gpt-4.1-nano-2025-04-14` | [54:44]–[54:47] |
> | The pricing discussion | nano's rate, **$1.50 per 1M training tokens** | [47:31], notebook cell 67 |
> | **The notebook's job object** | **`model="gpt-4o-2024-08-06"`** | run notebook, cell 78 |
>
> Both cannot be true of the same job, and the notebook artefact is the harder evidence: a
> `FineTuningJob` object is what the server returned, whereas the narration is a sentence spoken while
> scrolling. The likely explanation is that he *intended* nano, read nano's price, and the cell he
> executed carried a `gpt-4o` snapshot from earlier in the session — the notebook's cell 4 does carry
> `# gpt-4o-2024-08-06` as a commented-out line, which is consistent with exactly that kind of editing
> drift.
>
> **What this changes, and what it does not.** The **token arithmetic** (§4.6.2) is completely
> model-independent — `672 tokens/epoch × 3 epochs = 2,016 billed tokens` is true whichever base model
> runs the job. The **price** is not: `$1.50/M` is nano's rate, and a `gpt-4o` fine-tune was priced
> roughly an order of magnitude higher. So the video's headline figure, **$0.003024**, is the correct
> arithmetic applied — at most — to the model he *said* he was training, and the model the notebook
> recorded would have cost meaningfully more.
>
> **The transferable lesson is not "he made a mistake".** It is that **the model ID is the one field
> you cannot reconstruct from a video, a bill, or a memory** — which is precisely why `metadata` and a
> model registry (§16.2) exist. A run whose `metadata` says
> `{"base": "gpt-4.1-nano-2025-04-14", "data_version": "v7"}` never has this problem.

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

---

### 4.5 Token accounting — where the bill comes from

You cannot negotiate with the tokeniser. The provider counts, and the count appears in
`job.trained_tokens` when the job completes. This subsection teaches you to predict that number to
within a few percent before you spend anything.

#### 4.5.1 First: what `tiktoken` actually is

The instructor introduces the local counting tool like this:

> *"openai provides you one library, the library name is tiktoken… using this tiktoken I'm going to
> load this model, CL1 — sorry, CL 100K base model. So this is my encoding model guys. What it will
> do? It will create a token out of this sentence and it will assign an ID to this particular token"*
> [39:50]–[40:11]
>
> *"So this is also a transformer based model."* [40:14]–[40:16]

> **Correction:** *"this is also a transformer based model"* [40:14]–[40:16] is wrong. `tiktoken` is
> a **byte-pair-encoding (BPE) tokeniser and merge table** — a lookup algorithm over a static
> vocabulary file, with no weights, no layers, no forward pass, and no GPU. `cl100k_base` is the
> *name of a merge table* (roughly 100,000 merges) used by the `gpt-4`, `gpt-3.5-turbo` and
> text-embedding-ada-002 generation of models. Nothing in the tokenisation step is learned at
> inference time; the merges were learned once, from a corpus, and frozen. The distinction matters
> practically: because it is a pure lookup, `tiktoken` runs in microseconds on CPU, which is why the
> cost-estimation code in this module can run inside a Colab CPU runtime while the training itself
> happens in a datacentre. Confusing a tokeniser with a model also leads people to expect the token
> count to be *approximate* for the model they are training. It is not approximate for the model
> family it belongs to — see §4.5.4.

The notebook's demonstration is clean and correct. `Hello, how are you?` is 19 characters, 4
whitespace-separated words, and **6 tokens** (cell 39):

| Token | Token ID (notebook) |
|---|---|
| `Hello` | 9906 |
| `,` | 11 |
| ` how` | 1268 |
| ` are` | 527 |
| ` you` | 499 |
| `?` | 30 |

The transcript renders the instructor reading this table as *"hello is being represented by this
9906. A comma is being represented by this 11. How is being represented by this 12868 then R then U
and then this question mark"* [41:00]–[41:09] — note that the spoken ID "12868" disagrees with the
notebook's `1268`, and the spoken "R then U" disagrees with the notebook's ` are` / ` you`. **Trust
the notebook over the auto-transcript.** The lesson that survives either way is the ratio:

| Unit | Count for `Hello, how are you?` | Ratio to characters |
|---|---|---|
| Characters | 19 | — |
| Words (whitespace split) | 4 | 0.21 words/char |
| **Tokens** | **6** | **0.32 tokens/char** |

The rule of thumb that follows — **English prose is roughly 1 token per 4 characters, or ~0.75
tokens per word** — is good to ±15% on prose and *catastrophically wrong on code, JSON, non-Latin
scripts, and long numeric IDs*, where tokens/character can reach 1.5–2.0. Never estimate a
fine-tuning budget from word counts.

#### 4.5.2 The two local counters, and the overhead they model

The notebook defines two counting functions (cells 42 and 58):

```python
# notebook cell 42 — the "naive" counter used for the video's headline numbers
def count_total_tokens(messages):
    return sum(len(encoding.encode(m["content"])) for m in messages)

def count_assistant_tokens(messages):
    return sum(len(encoding.encode(m["content"]))
               for m in messages if m["role"] == "assistant")
```

```python
# notebook cell 61 — the "cookbook" counter that models per-message overhead
def num_tokens_from_messages(messages, tokens_per_message=2, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message                       # role delimiter overhead
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 2                                            # reply priming
    return num_tokens
```

The difference between them is the thing you have to internalise: **the provider does not bill you
for the characters in your JSON. It bills you for the tokens after the conversation has been
rendered into the model's own format** — role markers, turn separators, and a priming suffix are all
real tokens on the wire.

| Counter | Model | Total for the 10-row video dataset | Where it appears |
|---|---|---|---|
| `count_total_tokens` | Sum of content tokens only | **562** | Transcript [46:38]–[46:42]; notebook cell 44 `[59,62,51,62,51,54,54,53,57,59]` |
| `num_tokens_from_messages` | Content + 2/msg overhead + 2 priming | **672** | Notebook cell 65 |
| Ratio | — | **+19.6%** | — |

That 19.6% is your overhead, and it is worst on *short* examples because the per-message cost is
fixed. On the video's 56-token examples, overhead is a fifth of the bill. On 2,000-token examples it
is under 0.5%. Which means:

> **Beyond the video:** the overhead constant is **the** place where a small-data fine-tune gets
> expensive per unit of learning, and it is why very short examples are a bad deal. Compare two ways
> of spending 400,000 billed tokens at `gpt-4.1-nano`'s $1.50/M (i.e. $0.60):
>
> | Dataset shape | Rows | Supervised tokens | Overhead tokens | Overhead % | Learning per dollar |
> |---|---|---|---|---|---|
> | 56-token examples (video-shaped) | ~5,900 | ~4,700 × 3 epochs | ~1,180 × 3 | 19.6% | very low — 5,900 near-duplicate snippets teach one behaviour |
> | 2,000-token examples | 200 | ~1,970 × 1 epoch | ~30 | 0.5% | far higher — 200 rich, distinct conversations |
>
> The second dataset is not merely cheaper *per token*; it is cheaper *per unit of behaviour
> learned*, because the fixed per-message overhead is amortised and because 200 long conversations
> carry more distinct conditional structure than 5,900 short near-duplicates. The video's 56-token
> average is a property of its 3-unique-question dataset, not a virtue.

#### 4.5.3 Reading the distribution, not the mean

The notebook computes means and maxima (cells 46–47) and the video reads them aloud:

> *"the average token per example is 56 around 56. Average token is there in every example and the
> maximum token is 62… average assistant token which is around 21 token in every output message
> there is on an average 21 tokens and maximum is around 25."* [45:51]–[46:16]

The numbers are right (mean 56.2 → 56; max 62; mean assistant 21.2 → 21; max 25), and the instructor
dismisses the analysis with *"this is not like having any meaning but yeah this is just for the
analysis"* [46:16]–[46:23]. **That dismissal is the most costly sentence in the video**, because the
distribution is the only thing that predicts your bill and your truncation risk. See §4.5.5.

The cookbook's `print_distribution` function — which the notebook has, commented out in cell 41 —
prints min/max, mean/median, and p5/p95:

```python
def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
```

Note the bug in the original: `np.quantile(values, 0.1)` and `0.9` are the **p10 and p90**, not the
p5 and p95 the label claims. Fix it or your "p95" is really a p90 and you will under-provision the
token cap.

| Statistic | What it is for | Why the mean is not enough |
|---|---|---|
| mean | Rough cost estimate on homogeneous data | Hides a long tail that a per-example cap will truncate |
| median | The "typical" example | On a skewed distribution, mean ≫ median means a few examples dominate |
| **max / p99** | Truncation risk | If max > per-example cap, you are silently losing data |
| **p95** | Capacity planning | Set your own `max_seq_length` (open-weight) near p99, not near max |
| assistant-token mean | The behaviour density | The ratio supervised/(supervised+prompt) tells you how much of your spend teaches anything |

> **Beyond the video — the metric that actually predicts model quality:** compute the **supervised
> token fraction** = assistant tokens ÷ total tokens, per example, and then look at the *spread*.
> On the video's data it is 212/562 = **37.7%** (or 212/672 = 31.5% with overhead). That is a
> healthy ratio for a chat model — roughly a third of every billed token is a token the model is
> learning to produce. Compare:
>
> | Task shape | Typical supervised fraction | Implication |
> |---|---|---|
> | Short chat reply | 30–40% | Efficient. |
> | Summarisation of a 2,000-token document | 5–10% | You are mostly paying to re-read the input 3× per epoch. |
> | Classification into one label | 0.1–1% | **Fine-tuning by SFT is the wrong tool.** Use a classifier head or a smaller model. |
> | Long-context extraction with a short JSON output | 1–3% | Consider whether a *base* model + `response_format` already does this (§4.12). |
>
> If your supervised fraction is below ~5%, do the arithmetic in §11 before you upload: you may be
> about to spend most of your budget teaching the model to reproduce your own prompts.

#### 4.5.4 The tokeniser you use is not the tokeniser that bills you

Both notebooks do this:

```python
encoding = tiktoken.get_encoding("cl100k_base")   # notebook cell 33
```

`cl100k_base` is the vocabulary of the `gpt-4` / `gpt-3.5-turbo` generation. It does **not** tokenise
identically to `o200k_base`, which is what the `gpt-4o` and `gpt-4.1` families use. The two differ
most on:

- non-English text (o200k is materially better for non-Latin scripts);
- code and punctuation-heavy strings;
- long runs of digits;
- whitespace patterns.

For English prose the two counts typically land within a few percent of each other, which is why the
video's numbers look plausible. But the direction of the error is systematic, and on non-English
data it can be 20%+. The notebook's own data contains a typographic apostrophe (`I'm` in row 9,
U+2019 rather than U+0027) which the two vocabularies may split differently — exactly the kind of
character that makes a "few percent" estimate wrong.

```python
# The correct local estimate for a gpt-4.1-* fine-tune (2025+ vocabularies)
import tiktoken
encoding = tiktoken.get_encoding("o200k_base")

def billed_tokens_estimate(messages, max_tokens_per_example, tokens_per_message=3, tokens_per_name=1):
    """Upper-bound estimate of what one example will cost per epoch."""
    n = 0
    for m in messages:
        n += tokens_per_message
        for k, v in m.items():
            if isinstance(v, str):
                n += len(encoding.encode(v))
            if k == "name":
                n += tokens_per_name
    n += 3                                     # reply priming
    return min(n, max_tokens_per_example)      # the cap is applied AFTER counting
```

> **Beyond the video:** the constants in the video's counter (`tokens_per_message=2`, `+2`) differ
> from the ones in the commented-out cell (`tokens_per_message=3`, `+3`) and from the cookbook's
> documented defaults. Nobody in the video notices. The difference is 4 tokens per example — noise on
> a 56-token example (7%) and irrelevant on a 2,000-token example. **Do not lose sleep over the
> constant. Do lose sleep over the cap.** The cap in the next subsection is worth three orders of
> magnitude more.

#### 4.5.5 `MAX_TOKENS_PER_EXAMPLE = 16385` is a stale constant, and the failure is silent

The notebook carries this line (cell 54):

```python
# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 16385
```

and uses it in the billing sum (cell 64):

```python
n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length)
                                  for length in total_tokens_per_example)
```

and in the truncation warning (cell 63):

```python
n_too_long = sum(l > 16385 for l in total_tokens_per_example)
print(f"\n{n_too_long} examples may be over the 16,385 token limit, "
      f"if they are crossing the limit they will be truncated during fine-tuning")
```

Three things are true about `16385` and only one of them is in the video:

1. **It is a real constant from the 2024 cookbook**, derived from a 16,384-token context plus one.
   Correct for the `gpt-3.5-turbo` / `gpt-4o-mini` fine-tuning era.
2. **It is not the fine-tuning context of the `gpt-4.1` family.** The demo model
   `gpt-4.1-nano-2025-04-14` is not a 16k-context model. Using this constant against a `gpt-4.1`
   fine-tune makes the billing sum **over-count** (it caps too aggressively) and makes the
   truncation warning **lie** (it flags examples as safe that are not, or claims truncation where
   none occurs) — depending on which direction the real cap falls.
3. **Truncation is silent.** The `print` at cell 63 emits `0 examples may be over the 16,385 token
   limit` for this dataset and nothing else happens. For a dataset that *is* over, the API does not
   fail the job; it trains on a shortened example. The instructor states the mechanism correctly
   [63:00 area, cell 63 output]: *"if they are crossing the limit they will be truncated during
   fine-tuning."* What he does not state is the operational consequence: **you will never find out
   which rows were truncated, or that it happened at all, except by looking at `trained_tokens` and
   noticing it is lower than your own sum.**

| Symptom | What it means | Diagnostic |
|---|---|---|
| `job.trained_tokens` ≪ your `billing_tokens × epochs` | Examples were capped | Recompute with the *correct* per-example cap for your base model |
| Model ignores the tail of long inputs | Truncation removed the end of the prompt | Check whether the provider truncates head or tail |
| Model stops mid-sentence on long outputs | Truncation removed the end of the *assistant* turn — the supervised part | Cap your own examples below the limit before upload |

> **Beyond the video — the production rule:** never let the platform be the thing that decides what
> gets cut. Pre-truncate yourself, deliberately, with a documented policy:
>
> ```python
> def truncate_example(ex, cap, keep="tail"):
>     """Guarantee the assistant turn survives. Returns (example, was_truncated)."""
>     total = num_tokens_from_messages(ex["messages"])
>     if total <= cap:
>         return ex, False
>     msgs = ex["messages"]
>     if keep == "tail":
>         # drop the OLDEST user/assistant pairs first, never the final assistant turn
>         while num_tokens_from_messages(msgs) > cap and len(msgs) > 2:
>             del msgs[1]                      # keep index 0 (system) and the last pair
>     else:
>         # hard-cut the system prompt, which is usually the boilerplate
>         msgs[0]["content"] = msgs[0]["content"][:200]
>     return {"messages": msgs}, True
> ```
>
> Then assert that the truncated fraction is under a threshold you chose on purpose (2% is a sane
> default) and **fail the pipeline** if it is not. A silent 15% truncation rate is the single most
> common cause of "we fine-tuned and it got worse on long inputs".

---

### 4.6 Cost arithmetic — the worked end-to-end budget

This is the section the module exists for. The instructor attempts this arithmetic on camera and gets
a structurally wrong answer; correcting it teaches you the whole cost model.

#### 4.6.1 What the pricing page has on it

Reading the fine-tuning pricing table at [20:08]–[20:40], the instructor identifies the columns
correctly *the first time*:

> *"this is the training cost, this is per hour basis. Now this is the input cost — so the input cost
> means the message which you are providing to the LLM… this is the cache cost, means if you're going
> to provide the same input again, OpenAI is going to pick that particular input from the cache
> memory… Now this is the output cost… the pricing is per 1 million tokens, per 10 lakhs tokens."*
> [20:08]–[20:40]

He then says, correctly, that the input/output columns are the **inference** prices:

> *"Now here is the cost for the inferencing. So after the finetuning, whenever we are going to
> inference the model… this cost will come from the input, cache input and the output."*
> [20:43]–[20:57]

Then, twenty minutes later, he retracts it:

> *"So guys, as I told you, this is an inferencing input and output token price. But guys, this
> statement was the wrong. So this input and output token price, it is with respect to the training
> only."* [38:01]–[38:16]

> **Correction:** the retraction at [38:01]–[38:16] is the error. The fine-tuning pricing table has
> **four distinct columns**, and they are not all training:
>
> | Column | What it is | When you pay it |
> |---|---|---|
> | **Training** | per 1,000,000 **training tokens processed** (dataset tokens × epochs) | once, per job |
> | **Input** | per 1,000,000 **inference** input tokens against the fine-tuned model | every call, forever |
> | **Cached input** | per 1,000,000 inference input tokens that hit the prompt cache | every call that repeats a prefix |
> | **Output** | per 1,000,000 **inference** output tokens | every call, forever |
>
> The instructor's *first* reading [20:43]–[20:57] was right and he talked himself out of it. The
> practical damage is severe in one direction: a reader who believes the input/output columns are
> training costs concludes that **inference on a fine-tuned model is free**, and then discovers the
> real per-call bill only after shipping. The second, subtler damage: the Training column is a
> **per-token** price, and reading it as per-hour (which he does at [37:47]–[37:53]) is what produces
> the $0.75 figure in §4.6.3.

#### 4.6.2 The correct arithmetic for the video's own job

Everything needed is in the notebook and the transcript. Assembled:

| Input | Value | Source |
|---|---|---|
| Training rows | 10 | `data.jsonl` line count |
| Content tokens (no overhead) | 562 | notebook cell 44; transcript [46:38]–[46:42] |
| Billed tokens with per-message overhead | **672** | notebook cell 65 |
| `n_epochs` requested | 3 | transcript [55:44]–[55:47]; notebook cell 56 |
| Base model | `gpt-4.1-nano-2025-04-14` | transcript [54:44]–[54:47] |
| Training price | **$1.50 per 1,000,000 tokens** | transcript [47:31]; notebook cell 67 |
| Inference input price | $0.20 per 1,000,000 tokens | transcript [47:36] |
| Inference output price | $0.80 per 1,000,000 tokens | transcript [47:44] |
| USD→INR rate used | 91 | transcript [49:07]–[49:13] |

```
billed_training_tokens = 672 tokens/epoch × 3 epochs           = 2,016 tokens
training_cost          = 2,016 × ($1.50 / 1,000,000)           = $0.003024
                                                               ≈ ₹0.275 at 91 INR/USD
```

That is the **entire** training bill for the video's job: **three tenths of a cent, or 27 paise.**
The notebook computes exactly this at cell 67:

```text
$1.50 / 1,000,000 = $0.0000015 per token
2016 × 0.0000015 = $0.003024
$0.003024 × 91 = ₹0.275184
```

> **Beyond the video — why this is the most useful number in the module:** it makes the *iteration
> cost of the data* visible. Nine hundred and ninety-seven of the first thousand experiments you run
> cost under a dollar. The expensive thing in fine-tuning is not the training run; it is the human
> hours spent collecting and curating the data, and the only way that stops being true is when you
> scale `n_epochs` or dataset size into the millions of tokens. Compare:
>
> | Dataset | Rows | Tokens/row | Epochs | Billed tokens | `gpt-4.1-nano` @ $1.50/M | `gpt-4.1-mini` @ $3.00/M |
> |---|---|---|---|---|---|---|
> | Video demo | 10 | 67 | 3 | 2.0 K | **$0.003** | $0.006 |
> | Small real | 500 | 400 | 3 | 0.60 M | **$0.90** | $1.80 |
> | Medium real | 5,000 | 400 | 3 | 6.0 M | **$9.00** | $18.00 |
> | Large real | 50,000 | 400 | 2 | 40.0 M | **$60.00** | $120.00 |
> | Very large | 200,000 | 1,200 | 3 | 720.0 M | **$1,080.00** | $2,160.00 |
>
> Read the last row against §11.5 before you commit to it: at that spend, a rented A100 and an
> open-weight 8B model are the same order of magnitude *and* you keep the weights.

#### 4.6.3 Where the video's $0.75 and $0.702396 come from — and why they do not reconcile

The instructor reports two figures:

> *"training, let's say my training is going to be run for the half an hour… So here is my training
> cost, this is my training cost, this is the training cost **$0.75**."* [48:08]–[48:29]
>
> *"Now what is the total cost guys? So the total cost will be this one… this is the total cost
> **0.702396**."* [48:31]–[48:43]
>
> *"0.75 you can multiply it with 91… so it is around 68 rupees or 69, so roughly 69 to 70 rupees
> I will get."* [49:04]–[49:20]

> **Correction:** three separate errors are stacked here, and each one is worth naming.
>
> **1. There is no hourly training cost for supervised fine-tuning.** The instructor states the
> training price as *"the cost of the GPU on an hourly basis"* [37:45]–[37:53] and then as *"$1.5 for
> using the GPU… this is the hourly rate, right? Per hour training compute"* [47:31]–[47:48]. The
> SFT Training column is **per 1,000,000 training tokens**, not per hour. Wall-clock duration does
> not enter the bill at all: a 10-token job that takes 40 minutes in the queue and a 500M-token job
> that takes 6 hours are priced by the same formula, and neither has an hourly term. It is
> **reinforcement** fine-tuning that is billed hourly (the `o4-mini` track). Multiplying
> `0.5 h × $1.50/h = $0.75` is therefore arithmetic on a quantity that does not exist. The correct
> training cost is $0.003024 — **248× smaller.**
>
> **2. The stated total is smaller than one of its own components.** $0.75 of training plus
> *"0.23 something"* of token cost [48:04]–[48:08] cannot total $0.702396. Whichever of the three
> numbers is the misprint, the report is internally inconsistent and should not be used as a budget
> template. (The most likely reconstruction of `0.702396`: it is close to
> `10 examples × 16,385 max tokens × $1.50/M × 3 epochs = $0.737`, i.e. the cost you get if you let
> the `MAX_TOKENS_PER_EXAMPLE` cap, rather than the actual token count, drive the sum. Note that this
> is the *same* stale 16,385 constant from §4.5.5 and it over-counts this dataset by **244×**.)
>
> **3. The conclusion he draws is wrong in the direction that matters.** *"0.75 you can multiply it
> with 91… roughly 69 to 70 rupees"* frames fine-tuning as a ~₹70 decision. The true figure for this
> job is **₹0.275**. The framing cost is not the money; it is the *decision quality*. A team that
> believes each experiment costs $0.75 runs few experiments and treats each as precious. A team that
> knows the truth runs two hundred and converges on a good dataset in a week.

#### 4.6.4 The four-term cost model to carry in your head

```
TOTAL =  TRAINING          +  VALIDATION        +  INFERENCE                +  ENGINEERING
         tokens × epochs      (rate varies)       input + cached + output      your time
         × train_price        × train_price       × call volume
         ────────────         ─────────────       ────────────────────────     ──────────
         one-off              one-off             recurring, monthly           dominant
```

| Term | Formula | The video's job | A 5k-row production job |
|---|---|---|---|
| Training | `billed_tokens × n_epochs × train_price` | $0.0030 | $18.00 (`gpt-4.1-mini`, 6.0 M tokens) |
| Validation | `val_tokens × n_epochs × train_price` (provider-dependent) | — (none used) | ~$1.80 on a 10% holdout |
| Inference | `calls × (P_in·r_in + P_out·r_out + P_cached·r_cached)` | ~$0.0002/call | $25–140/month depending on volume |
| Engineering | 20–200 h of data work | ~3 h | 80–300 h |

> **Beyond the video — the validation-file billing question.** Whether the validation file's tokens
> are billed at the training rate varies by provider and by era. Two safe practices: **(a)** size
> your validation set as a percentage of training (10–20%) and treat its tokens as a small linear
> adder to the training estimate rather than trying to get an exact answer; **(b)** verify against
> the provider's meter after your first job by comparing `job.trained_tokens` against your own
> training-file sum. If `trained_tokens` comes back ~10–20% above your training-file estimate, the
> validation file is being billed. If it matches, it is not.

#### 4.6.5 The comparison nobody runs: hosted fine-tune vs a *shorter prompt on the base model*

Before you spend anything, answer this: **could I just delete some prompt tokens?**

The fine-tuned model's per-token inference price is higher than the base model's — for the `gpt-4.1`
family it is a clean **2×** on both input and output:

| Model | Base input | FT input | Base output | FT output | Markup |
|---|---|---|---|---|---|
| `gpt-4.1-nano` | $0.10/M | **$0.20/M** | $0.40/M | **$0.80/M** | 2.0× |
| `gpt-4.1-mini` | $0.40/M | **$0.80/M** | $1.60/M | **$3.20/M** | 2.0× |
| `gpt-4.1` | $2.00/M | **$4.00/M** | $8.00/M | **$16.00/M** | 2.0× |
| `gpt-4o` (historical) | $2.50/M | $3.75/M | $10.00/M | $15.00/M | 1.5× |

So the fine-tune wins **only if the prompt you can delete is worth more than the markup you pay on
everything else.** With a 4:1 output:input price ratio (which every model above has), the algebra
collapses to something you can do in your head:

$$P_f \;<\; \frac{P_b}{2} \;-\; 2O \qquad\text{(hosted FT wins at a 2× markup, 4:1 output:input ratio)}$$

where $P_b$ = base prompt tokens, $P_f$ = fine-tuned prompt tokens, $O$ = output tokens.

Three worked cases, all `gpt-4.1-mini`:

| Case | Base prompt $P_b$ | Output $O$ | Break-even $P_f$ | Actual $P_f$ | Base $/call | FT $/call | Verdict |
|---|---|---|---|---|---|---|---|
| **A** — 8-shot support prompt | 1,200 | 150 | < 300 | 400 | $0.000720 | $0.000800 | **FT loses.** You cannot delete enough prompt. |
| **B** — 30-shot extraction prompt | 3,000 | 90 | < 1,320 | 700 | $0.001344 | $0.000848 | **FT wins**, saves $0.000496/call |
| **C** — 2-shot, short | 450 | 200 | < −175 (impossible) | 300 | $0.000500 | $0.000880 | **FT loses at any prompt length** — the output-side markup alone exceeds the base total. |

Case B recomputed line by line so you can audit it:

```text
CASE B — gpt-4.1-mini, 200,000 calls/month

BASE MODEL (no fine-tune)
  input : 3,000 tok × $0.40/M = $0.001200
  output:    90 tok × $1.60/M = $0.000144
  per call                    = $0.001344
  × 200,000 calls             = $268.80 / month

FINE-TUNED (gpt-4.1-mini, prompt cut to 700 tok)
  input :   700 tok × $0.80/M = $0.000560
  output:    90 tok × $3.20/M = $0.000288
  per call                    = $0.000848
  × 200,000 calls             = $169.60 / month

MONTHLY SAVING = $99.20
TRAINING COST  = 5,000 rows × 420 billed tok × 2 epochs × $3.00/M = $12.60
PAYBACK        = $12.60 / $99.20 = 0.13 months ≈ 4 days
YEAR 1 NET     = 12 × $99.20 − $12.60 − (validation, ~$1.30) = $1,176.50 saved
```

> **Beyond the video — the strategic consequence of the break-even formula.** The formula says the
> only fine-tunes that pay for themselves are the ones that **replace a large block of few-shot
> examples**. Which means:
>
> - If your prompt is short (a system message and a one-line instruction), fine-tuning will *never*
>   pay for itself on hosted inference economics. You fine-tune for **quality or format compliance**,
>   and you should say so out loud, because "it's cheaper" will be false.
> - If your prompt is long *because of examples*, fine-tuning is a straight prompt-compression trade
>   and it usually wins by a wide margin.
> - If your output is long relative to your input, the 2× output markup makes hosted fine-tuning
>   structurally uncompetitive — **and that is the single strongest argument for open-weight
>   fine-tuning in this handbook.** A fine-tuned 8B model on your own GPU has a marginal output cost
>   of zero. It is only the fixed cost that differs, and fixed cost amortises.

---

### 4.7 The three hyperparameters, and what the API does when you do not set them

#### 4.7.1 The call as the video writes it

```python
client.fine_tuning.jobs.create(
    training_file=training_file_id,          # "file-Hn8ooHUNEzwcPjBasULS25"
    model="gpt-4.1-nano-2025-04-14",         # the snapshot ID, not the alias
    suffix="second-finetune-model",          # becomes part of the model name
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {
                "batch_size": 16,
                "learning_rate_multiplier": 1.0,
                "n_epochs": 3
            }
        }
    }
)
```

The instructor builds this call field by field, correctly, at [54:19]–[55:47], reading each
hyperparameter aloud. His framing of each:

| Knob | His words | Verdict |
|---|---|---|
| `batch_size` | *"batch size is 16, means in how many batches we are passing the data in, like in one batch we are passing 16 rows. But guys, this is irrelevant as of now because I just have 10 rows right now."* [55:09]–[55:22] | Directionally right, and the observation about 10 rows is exactly the right instinct — see §4.7.3 on clamping. |
| `learning_rate_multiplier` | *"this is 1.0. Zero learning rate, you know, to stabilize the training… this is again a mathematical concept of the gradient of this model, the neural network."* [55:24]–[55:39] | Muddled. It is a **multiplier**, not a learning rate, and 1.0 means "provider default", not "zero". See the Correction below. |
| `n_epochs` | *"number of epochs — so for how many epochs you are running the training. I'm running it for the three epochs."* [55:41]–[55:47] | Correct. |

#### 4.7.2 The full knob table

| Param | What it does | Default | Documented guidance | Safe range | Too high → | Too low → | SDK key |
|---|---|---|---|---|---|---|---|
| `n_epochs` | Full passes over the dataset. Linear multiplier on the training bill. | `"auto"` → resolved by the target-example policy (§4.8) | Start at 1–2; increase only while validation loss is still falling | 1–3 for datasets ≥ 1k rows; 3–10 only for datasets < 100 rows | Memorisable training rows → the model recites; validation loss rises while train loss falls | Underfit: the behaviour change never takes | `method.supervised.hyperparameters.n_epochs` |
| `batch_size` | Examples per forward/backward pass in the provider's trainer | `"auto"` → **~0.2% of training examples, capped at 256**, floored at 1 | Larger batches work better on larger datasets; do not set it by hand below ~50 rows | 1–16 for tiny data, `auto` otherwise | Fewer, larger steps: slower convergence per step, and an LR mismatch if you raise it without raising the LR multiplier | Noisier gradient, unstable loss, and (in the API) an LR mismatch in the other direction | same path, `batch_size` |
| `learning_rate_multiplier` | A **multiplier** applied to the provider's internal base LR. Not an absolute LR. | `"auto"` → one of **0.05 / 0.1 / 0.2** depending on the final batch size (legacy guidance); the observed resolved value on a tiny dataset is **2.0** | Legacy completion-era guidance: experiment in **0.02–0.2**. Current chat-era guidance found in the wild: **0.5–2**, with the API accepting **0.0–5.0** | 0.5–2 if you must touch it | Loss spikes, divergence, NaN, or a model that forgets everything it knew | Effectively no learning; the job "succeeds" and the model is unchanged | same path, `learning_rate_multiplier` |
| `seed` | Best-effort determinism for the run | random | Set it if you want to compare two runs that differ in exactly one variable | any int | — | — | top-level `seed` |
| `suffix` | Human-readable tag appended to the model name | none | Use a naming convention that encodes the data version | — | Names collide and you cannot tell jobs apart | You cannot map a model ID back to a dataset | top-level `suffix` |
| `validation_file` | Holdout file for computing validation loss during training | `None` | **Always set it.** 10–20% of the data. | 10–20% split | Costs tokens to bill; slows the job slightly | You get no overfitting signal at all (§4.9) | top-level `validation_file` |
| `integrations` | Push events to Weights & Biases or similar | `[]` | Useful for real runs; overkill for a 10-row demo | — | — | You are polling the API by hand | top-level `integrations` |
| `metadata` | Up to 16 key/value pairs attached to the job | `None` | Put your `data_version`, `git_sha`, `owner` here | — | — | You cannot reconstruct what produced a model | top-level `metadata` |

> **Correction:** at [55:24]–[55:39] the instructor describes `learning_rate_multiplier: 1.0` as
> *"zero learning rate, you know, to stabilize the training."* Two things are wrong. **(1)** It is
> not a learning rate — it is a **multiplier on the provider's chosen base learning rate**, which
> you cannot see. `learning_rate_multiplier=1.0` means "use the default base rate as-is", not "use
> zero". **(2)** The actual learning rate for the job is
> `lr = base_lr × learning_rate_multiplier`, and `base_lr` is set by the provider and is not exposed.
> So a `learning_rate_multiplier` of 2.0 with a base of 0.05 is 0.1; a multiplier of 0.1 with a base
> of 2.0 is 0.2. **The same numeric value can mean opposite things across providers and across
> versions of the same provider's stack.** The practical rule: treat this knob as *ordinal, not
> cardinal* — "the same as default", "more aggressive", "less aggressive" — and never reason about
> the absolute value. If you need a real learning rate, that is an argument for open-weight training
> (CS-13 §7.3), where you set it yourself.

#### 4.7.3 What the API does with hyperparameters you do not control

The instructor's demo job (from the *first* notebook, with `n_epochs` left as `"auto"`) is the perfect
worked example of the automatic policy, because the API echoes back exactly what it chose:

```python
# notebook cell 16 — the create() call, with NO hyperparameters block
client.fine_tuning.jobs.create(
    training_file="file-PsuEt1x4gLPqY8f4sDD49t",
    model="gpt-3.5-turbo",
    suffix="first finetune model"
)
```

```text
# notebook cell 17 — what the server returned on the next poll
FineTuningJob(
    hyperparameters=Hyperparameters(
        batch_size=1,
        learning_rate_multiplier=2.0,
        n_epochs=10                    # ← auto, resolved to 10
    ),
    model='gpt-3.5-turbo-0125',
    status='running',
    ...
)
```

Three auto-resolutions are visible in that single object, and every one of them is predictable:

| Knob | Requested | Resolved | Rule that produced it |
|---|---|---|---|
| `n_epochs` | `"auto"` | **10** | 10 examples × `TARGET_EPOCHS`(3) = 30 < `MIN_TARGET_EXAMPLES`(100) → `min(25, 100 // 10)` = **10** |
| `batch_size` | `"auto"` | **1** | 0.2% of 10 examples = 0.02 → floored to the minimum of **1** |
| `learning_rate_multiplier` | `"auto"` | **2.0** | Provider's LR policy for a batch size of 1 |

The `n_epochs` resolution is the important one. **A 10-row dataset is trained for 10 epochs by
default.** The instructor later sets `n_epochs=3` explicitly, which is why the job in the video costs
a third of what the auto job would have. He never mentions that a default job on 10 rows would have
been 3.3× more expensive and 3.3× more overfit.

> **Beyond the video:** read `job.hyperparameters` back **every single time** and log it next to the
> model ID. It is the only record of what actually happened. A job created six months ago with
> `"auto"` everywhere is unreproducible unless you recorded the resolved values, because the
> provider's policy can change under you without an API version bump.

---

### 4.8 The automatic epoch policy, in full

The notebook reproduces the cookbook's reference implementation as a commented block (cell 68). Here
it is, restored to executable form:

```python
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(data)

if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
```

Decoded, the policy has exactly three regimes and two sharp cliffs:

```
                      n_train_examples
                      10        100      5,000     8,333   8,400      200,000
                       │          │         │        │       │            │
  REGIME 1             │  REGIME 2 (flat)    │  REGIME 3 ────────────────────►
  "too little data,    │  "target epochs"    │  "too much data,
   crank the epochs"   │                     │   throttle the epochs"
                       │                     │
  n_epochs = min(25, 100 // N)   n_epochs = 3   n_epochs = max(1, 25000 // N)
                       │                     │
  10 rows  → 10 ep     │  100 rows → 3 ep    │  200,000 rows → 1 ep
  50 rows  → 2 ep      │  5,000  rows → 3 ep │  50,000 rows  → 1 ep
  99 rows  → 1 ep      │  8,333  rows → 3 ep │  8,400 rows   → 2 ep  ◄── cliff
```

Worked values across the range — put this table next to your dataset size before you create a job:

| Training rows | `N × TARGET_EPOCHS` | Regime | Resolved `n_epochs` | Total billed passes over the data |
|---|---|---|---|---|
| 10 | 30 | 1 | **10** | 100 examples-worth |
| 25 | 75 | 1 | **4** | 100 examples-worth |
| 33 | 99 | 1 | **3** | 99 examples-worth |
| 50 | 150 | 2 | **3** | 150 |
| 100 | 300 | 2 | **3** | 300 |
| 1,000 | 3,000 | 2 | **3** | 3,000 |
| 8,333 | 24,999 | 2 | **3** | 24,999 |
| 8,400 | 25,200 | 3 | **2** | 16,800 |
| 12,500 | 37,500 | 3 | **2** | 25,000 |
| 25,000 | 75,000 | 3 | **1** | 25,000 |
| 200,000 | 600,000 | 3 | **1** | 200,000 |

Two properties of this policy are worth stating explicitly:

1. **Below 100 examples, the policy tries to hold the number of *example-observations* constant at
   ~100.** That is why `n_epochs = 100 // N`. It is a sensible heuristic: it guarantees at least
   ~100 gradient-relevant observations regardless of dataset size. The cost is that the same 100
   observations are *repeated*, which is overfitting by construction — you learn those 100 rows
   extremely well and nothing else.
2. **Above 25,000 examples, the policy keeps the number of *example-observations* at ~25,000 by
   dropping `n_epochs` to 1.** This caps cost but means very large datasets get a single pass. That
   is usually the right call, and it also means the marginal value of the 300,000th training row is
   approximately zero against this policy.

> **Beyond the video — the practical takeaway is a contradiction to fix, not a rule to follow.**
> The policy is designed to bound *cost*, not to produce the best model, and the two regimes pull in
> opposite directions:
>
> - If you have **50 good examples**, the policy gives you 2 epochs. That is probably *too few* —
>   with 50 high-quality curated examples, 6–10 epochs is where the behaviour change actually lands
>   (LIMA's 1,000-example result was with a much larger base model, but the "small data, more
>   epochs" intuition holds). **Set it explicitly.**
> - If you have **200,000 examples**, the policy gives you 1 epoch. That is probably *right*, but
>   the 500M-token bill at `gpt-4.1-mini` is $1,500 — at which point open-weight training costs a
>   tenth as much and gives you the weights.
>
> **The concrete rule: always set `n_epochs` explicitly, and treat the auto policy as a fallback you
> have read the source of.** The instructor sets `n_epochs=3` explicitly in the video, which is the
> right instinct even though he does not say why.

---

### 4.9 The validation file, overfitting, and the metrics the API returns

#### 4.9.1 The missing file

Both of the video's notebooks create jobs with `validation_file=None` — visible in every
`FineTuningJob` object in cells 17, 78 and 79. The instructor demonstrates the *dashboard's*
validation-Upload field at [1:00:42]–[1:00:48] (*"then validation data, upload the validation data"*)
and never uses it in code.

That omission has a specific consequence: **without a validation file, the hosted API gives you no
signal about generalisation at all.** With one, you get a holdout loss curve computed on data the
model never trained on. That curve is the only automatic overfitting detector in the entire workflow.

| | No validation file | Validation file set |
|---|---|---|
| Train loss curve | Yes (via result files / events) | Yes |
| **Holdout loss curve** | **No** | Yes |
| Overfitting detectable? | Only by inference-time testing after the fact | Yes, during training |
| Cost | Lower | +10–20% of training tokens |
| Data required | All rows train | 10–20% of rows held out |
| Video's choice | **This one** | — |

> **Beyond the video — the split you should actually ship.** Use **three** sets, not two:
>
> | Set | Share | Purpose | Touched how often |
> |---|---|---|---|
> | Training | 70–80% | What the model learns on | Every job |
> | Validation | 10–15% | Early-stop / epoch selection, passed to `validation_file` | Every job |
> | **Test / regression** | 10–15% | The honest number you report | **Once per release, never during training** |
>
> The platform has no concept of a test set. If you pass your regression set as `validation_file`,
> you are tuning on it, and it stops being a test set the moment you look at it twice. Freeze it in
> a file the training script cannot read.

#### 4.9.2 Reading the returned metrics

After a supervised fine-tune, the job object and its result files expose the following. Names vary
by provider and by API version; the *shape* is stable across all hosted vendors:

| Field | Where | What it means | How it lies to you |
|---|---|---|---|
| `job.status` | job object | `validating_files` → `queued` → `running` → `succeeded` / `failed` / `cancelled` | `succeeded` means *the job ran*, not that the model is any good |
| `job.trained_tokens` | job object | Total billed training tokens | If it is lower than your estimate, examples were truncated (§4.5.5) |
| `job.fine_tuned_model` | job object | The deployment ID | `null` until success; `null` is not an error |
| `job.error` | job object | Populated on failure | The message is often generic; check `result_files` |
| `job.hyperparameters` | job object | **What actually ran** | Not what you sent (§4.7.3) |
| `usage_metrics` | job object | Token usage summary | Provider-dependent; do not assume it equals `trained_tokens` |
| `result_files` | job object | CSV id(s) with the step-level metrics | The **only** place the loss curves live |
| `train_loss` | result CSV | Cross-entropy on the training rows | Falls monotonically almost always — near-useless on its own |
| `valid_loss` | result CSV | Cross-entropy on the holdout rows | **The number that matters.** Rising = overfitting |
| `full_valid_loss` | result CSV | Valid loss at end-of-epoch boundaries | The one to plot; step-level valid loss is noisy |
| `train_mean_token_accuracy` | result CSV | % of training tokens predicted correctly | Goes to ~100% on small datasets regardless of quality |
| `valid_mean_token_accuracy` | result CSV | Same on holdout | A more readable proxy than loss for format-learning tasks |
| Job events | `client.fine_tuning.jobs.list_events(...)` | A stream of `message` strings (`"Step 100/300: training loss=..."`) | Cheap to poll; do not build an alerting system on it |

The canonical diagnostic shape, and the four ways it goes wrong:

```
train_loss  │                                   ┌── 4. "loss → 0"
            │                              ┌────┘      memorisation
            │                        ┌─────┘
            │                 ┌──────┘  3. healthy
            │        ┌────────┘
            │ ┌──────┘
            └────────────────────────────────────────────► steps
valid_loss  │
            │        ┌─────────────┐ 1. healthy: falls then flattens
            │ ┌──────┘             └───────────
            │ │  ┌──────────────────────────── 2. OVERFIT: rises while
            │ │  │                                 train loss still falls
            └─┴──┴────────────────────────────────────────► steps
```

| Pattern | Diagnosis | Action |
|---|---|---|
| Both fall, valid flattens | Healthy | Stop here. More epochs will not help. |
| Train falls, **valid rises** | **Overfitting** | Cut `n_epochs`; add data; add diversity |
| Both flat from step 1 | Underfitting / LR too low / data too uniform | Raise epochs; raise LR multiplier; check the data has signal |
| Both fall to ~0 | Memorisation (typical on < 100 rows) | You have a template, not a model. Get more data. |
| Loss spikes then recovers | LR slightly too high, or a pathological example | Inspect the longest examples; smooth over it |
| Loss NaN | LR multiplier far too high | Reset to 1.0 and halve the dataset's longest examples |
| Valid loss **below** train loss | Almost always a leak, not a miracle | Check for duplicate rows across the split (§4.9.3) |

#### 4.9.3 The overfitting story, quantified for this dataset

The video's first job ran `n_epochs=10` on a dataset with **4 unique conversations** (the inline
6,429-byte file). Do the arithmetic:

| Quantity | Value |
|---|---|
| Unique conversations | 4 |
| Rows | 10 |
| Epochs (auto) | 10 |
| **Total forward/backward passes over each unique conversation** | **25** (10 rows ÷ 4 unique × 10 epochs) |
| Distinct gradient steps (batch_size=1) | 100 |
| Distinct information content | 4 Q/A pairs |

Twenty-five exposures to each of four Q/A pairs, at an LR multiplier of 2.0, on a base model that
already knows how to answer support questions. The result is not a better support agent. It is a
model that has memorised four answers and their exact phrasing.

The video's own demonstration is consistent with this and does not notice it. When the instructor
finally chats with that model [1:03:02]–[1:03:16], he asks:

> *"What warranty does a smartphone come with?"* — the question is one of the four in the training
> set, and the training answer begins *"Smartphones typically come with a one-year limited warranty
> covering manufacturer defects."* The model replies in fluent, on-brand support prose.

That is a memorisation check, and it passes for a reason that has nothing to do with generalisation.
The correct test is the **out-of-scope probe that is already in his own dataset but not in the model
he chats with**: *"Can you recommend a good laptop for work?"* The model he trained (the 6,429-byte
file) has never seen it. The model he *could* have trained on the repo's `data.jsonl` has.

> **Beyond the video — three cheap overfitting detectors the platform will not run for you:**
>
> 1. **Phrasing-shift test.** Take 20 training questions and paraphrase each. A model that generalises
>    answers the paraphrase. A model that memorised answers it with the *original* question's
>    phrasing bleeding through, or fails.
> 2. **Out-of-scope refusal rate.** Hold out 20 in-scope and 20 out-of-scope prompts. Report both
>    numbers. Fine-tuning on all-positive data drives the out-of-scope refusal rate to zero — the
>    model starts answering laptop questions — and nobody notices until production.
> 3. **Verbatim-recall rate.** Generate 100 completions from held-out prompts and check whether any
>    n-gram of length ≥ 13 appears verbatim in the training set. If it does, you have memorised
>    customer data and you have a privacy finding, not just a quality finding.

---

### 4.10 The other three training methods on the same platform

The instructor opens the video by walking the platform's method selector at [1:03]–[1:12] and again
in a table at [13:01]–[13:32]. Four methods are on offer. He covers one of them.

| Method | API value | What it optimises | Data shape | Billing unit | In this video? |
|---|---|---|---|---|---|
| **Supervised fine-tuning (SFT)** | `method.type = "supervised"` | Next-token prediction on your demonstrations | `messages` JSONL | **Per 1M training tokens** | **Yes** — the whole video |
| **Vision fine-tuning** | `method.type = "supervised"` + image content parts | Same loss, with images in the context | `messages` JSONL with `image_url` parts | Per 1M training tokens | No — table row only |
| **Direct Preference Optimization (DPO)** | `method.type = "dpo"` | Relative preference between a chosen and a rejected completion | `messages` JSONL **+ `chosen` / `rejected` sibling arrays** | Per 1M training tokens | No — mentioned, never shown |
| **Reinforcement fine-tuning (RFT)** | `method.type = "reinforcement"` | A grader/reward function scores rollouts; policy is updated by RL | prompts + a grader (Python or a model) | **Per hour of training compute** | No — mentioned as "RLHF", never shown |

> **Correction:** at [1:03]–[1:12] and again at [13:20] the instructor lists the fourth method as
> **"RLHF"**. The platform's own name for it is **reinforcement fine-tuning (RFT)**, and the
> distinction matters more than a label. Classical **RLHF** (CS-24) means: train a reward model on
> human preference pairs, then optimise the policy against that reward model with PPO, with a KL
> penalty anchoring the policy to the SFT model. **RFT** as offered here means: you supply a
> **grader** — a Python function or a model-based rubric that scores a rollout as correct or
> incorrect — and the platform runs the RL loop against *that grader* directly. There is no learned
> reward model and no human preference data in the loop. The consequence is practical and large:
> **RFT needs a task with an automatically verifiable answer** (a math result that matches, code
> that passes tests, a JSON blob that validates against a schema). It is useless for "make the tone
> friendlier", because you cannot write a grader for that. And it is billed **per hour**, not per
> token, which changes the cost model completely (§11.6).

#### 4.10.1 DPO on the hosted API, concretely

DPO is the one non-SFT method from that table you are most likely to actually need, because it is the
correct tool for **"the model can do the task but I want it to prefer answer B over answer A"** —
tone, refusal behaviour, verbosity, safety. It is also the method most often confused with SFT.

The data contract extends the SFT one. Each row still has a `messages` list — the *prompt context* —
and then two parallel completions:

```json
{"messages": [{"role": "system", "content": "You are a support agent for a phone retailer."},
              {"role": "user", "content": "My phone arrived cracked. What do I do?"}],
 "chosen":   [{"role": "assistant", "content": "I'm sorry about that. Send a photo of the damage to support@example.com within 48 hours and we'll dispatch a replacement the same day."}],
 "rejected": [{"role": "assistant", "content": "You should have checked it before signing. Contact the courier."}]}
```

| Aspect | SFT | DPO |
|---|---|---|
| Per row you supply | 1 completion | **2** completions (chosen + rejected) |
| Signal | "produce this" | "produce this **rather than** that" |
| Data effort | 1× | ~2×, **and** you must be able to rank pairs |
| Minimum viable rows | 10 (demo) / 100 (real) | 100+ before the signal is meaningful; pairs are the scarce resource |
| Typical `n_epochs` | 1–3 | **1** — DPO overfits preference pairs extremely fast |
| Failure mode | Memorisation | **Likelihood collapse**: both chosen and rejected probabilities sink; the model gets worse at the task it was good at |
| When it is the right tool | New behaviour, new format, new domain | Existing behaviour, wrong ranking |

> **Beyond the video — the ordering rule that saves a project.** DPO is **stage two**, never stage
> one. If your model cannot produce the target format at all, DPO has nothing to rank and you will
> train on noise. The pipeline is:
>
> ```
> base model  →  SFT on demonstrations  →  DPO on preference pairs  →  ship
>                (teaches the behaviour)     (fixes the ranking)
> ```
>
> Running DPO on a base model that has never seen the task is a common and expensive mistake: the
> chosen and rejected strings are both out-of-distribution, the implicit reward is meaningless, and
> the run "succeeds" while making the model strictly worse. See CS-25 for the DPO loss in full and
> CS-24 for the RLHF pipeline it replaces.

The platform also exposes DPO-specific knobs beyond the SFT three, most notably `beta` (the strength
of the KL anchor to the reference model). Higher `beta` = stay closer to the SFT model; lower `beta` =
chase the preference signal harder and risk collapse. Defaults are sane; if you are tuning `beta` you
should be plotting `rewards/chosen` and `rewards/rejected` and watching for both to fall together.

---

### 4.11 Distilling a hosted model into a small one you own

This is the escape hatch from the endpoint-only constraint in §4.1, and it is the single most
practically valuable thing in this module that the video does not mention.

The situation: the platform is winding down new fine-tuning jobs (§16.8), you cannot download the
fine-tuned weights under any circumstances, and you have already paid to teach a hosted model
something valuable. **What you can always do is call it.**

```
┌──────────────────────┐        1. generate 50k–500k           ┌────────────────────┐
│  Teacher: hosted FT  │ ────────  completions on your  ──────► │  synthetic JSONL   │
│  model you cannot    │           prompt distribution          │  {messages: [...]} │
│  download            │                                        └─────────┬──────────┘
└──────────────────────┘                                                  │
                                                                          │ 2. filter
                                                                          │    (dedupe, length,
                                                                          │     dedup by n-gram,
                                                                          │     drop malformed)
                                                                          ▼
┌──────────────────────┐        4. serve locally                ┌────────────────────┐
│  Student: 1–8B open  │ ◄────────  (vLLM / Ollama)  ────────── │  SFT the student   │
│  weights you own     │           at 1/10 the cost             │  (CS-13, CS-23)    │
└──────────────────────┘                                        └────────────────────┘
```

Why it works: you are not extracting weights, you are **sampling the teacher's input→output
function**. What transfers is the *behaviour* the fine-tune taught — its format, its tone, its
refusal boundary — because that is exactly what the teacher's outputs encode. What does **not**
transfer is the teacher's raw capability ceiling; a 1B student trained on a 400B teacher's outputs
will not become a 400B model. It will become a 1B model that has the *habits* of one.

| Factor | Effect on distillation quality |
|---|---|
| Number of teacher samples | The dominant term. 1k is a demo; 10k is weak; 100k+ approaches the teacher on the narrow task |
| Prompt distribution coverage | The second dominant term. Samples must cover the *production* prompt distribution, not a hand-picked easy slice |
| Teacher sampling temperature | Slightly above 0 (0.7–1.0) gives variety; temperature 0 gives a student that is a lookup table |
| Diversity filtering | Dedupe near-identical generations. Without it the student over-fits a handful of phrasings |
| Student size | Bigger student = closer to teacher, but past ~30% of teacher size the returns collapse and you should have trained the student's base directly |
| **Legal basis** | **Read the provider's terms.** See below. |

> **Beyond the video — the terms-of-service question you must answer before writing the pipeline.**
> Training a competing model on a provider's outputs is restricted by most major providers' terms,
> usually with a clause prohibiting using outputs "to develop models that compete with" the provider.
> The practical distinctions:
>
> | Use | Usually permitted? | Note |
> |---|---|---|
> | Distilling to serve **your own product** on your own hardware | Commonly yes | The most defensible case |
> | Distilling to **resell model access** | Commonly restricted | This is the case the clauses target |
> | Distilling to build a **general-purpose competitor** | Almost always prohibited | Read the clause before you spend the tokens |
> | Using the samples as **eval data** | Commonly yes | Much safer framing than "training data" |
>
> The safe engineering pattern: generate the samples for a *narrow, product-specific* task, keep the
> prompt distribution documented, and get the legal reading in writing before the first training run
> rather than after. CS-09 covers the distillation pipeline end to end; the compliance checklist
> lives in AP-01.

---

### 4.12 The thing you probably want instead: structured outputs

The most common reason an engineer reaches for fine-tuning is **"the model won't reliably return
JSON."** It is worth being blunt: **for that problem, fine-tuning is the wrong tool, and the API has
had the right one for years.**

The failure mode looks like this. You ask for a JSON object; you get prose, or a JSON blob wrapped in
markdown fences, or a valid object with a key renamed, or a truncated object because the model hit
`max_tokens` mid-string. You add "respond ONLY with valid JSON" to the system prompt. It works ~90%
of the time. In production, 10% of calls throw a parse exception, and your retry loop quietly triples
your bill.

Fine-tuning *does* improve that, because you are demonstrating the exact output shape hundreds of
times. But you are paying a training bill and taking on a model-versioning burden to solve a problem
that has an exact, free, deterministic solution.

| Need | Right tool | Why |
|---|---|---|
| **Guaranteed valid JSON matching a schema** | **Structured outputs / `response_format` with a JSON schema** | Constrained decoding makes invalid output *impossible*, not unlikely. Zero training cost. |
| Guaranteed choice from a fixed label set | Structured outputs with an `enum` | Same mechanism |
| Guaranteed call to a specific tool with typed args | Tool/function calling with a schema | Same mechanism |
| The model does not know your **domain vocabulary** | Fine-tuning (or RAG) | The constraint is knowledge, not shape |
| The model's **tone** is wrong | Fine-tuning | Tone is not expressible in a schema |
| The model **refuses** things it should answer (or vice versa) | Fine-tuning (SFT, then DPO) | Behaviour, not shape |
| The output must be **shorter/cheaper** — same task, fewer tokens | Fine-tuning | The win is prompt-token reduction (§4.6.5), not correctness |
| The task needs a **private, unshipped** model | Open-weight training (CS-13) | Hosted gives you no artefact |

```python
# The correct answer to "make it return JSON", with no training run at all.
response = client.chat.completions.create(
    model="gpt-4.1-mini",                        # NOT a fine-tuned model
    messages=[{"role": "user", "content": "Extract the order."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "order",
            "strict": True,                      # <- this is the whole trick
            "schema": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku":      {"type": "string"},
                                "quantity": {"type": "integer"}
                            },
                            "required": ["sku", "quantity"],
                            "additionalProperties": False
                        }
                    },
                    "status": {"type": "string", "enum": ["new", "shipped", "cancelled"]}
                },
                "required": ["order_id", "items", "status"],
                "additionalProperties": False
            }
        }
    }
)
# Guaranteed to parse. Guaranteed to have those keys. Guaranteed `status` is one of three strings.
```

**The combined pattern is the one to remember**, and it is stronger than either alone:

```
structured outputs  →  guarantees the SHAPE
fine-tuning         →  guarantees the CONTENT and the TONE inside that shape
```

You do not choose between them. If you need a support agent that always returns
`{"reply": str, "escalate": bool}` in your brand's voice, you use `response_format` for the contract
and fine-tuning for the voice. If you only needed the contract, you needed one of the two, and it was
not the one this video is about.

> **Correction-adjacent — a claim the video never makes but implies.** The instructor frames
> fine-tuning throughout as the mechanism that makes the model "behave" [13:40]–[14:05], and the
> notebook's whole demo is a model that answers in a consistent house style. Nowhere does he claim
> fine-tuning is needed for *structure* — but the omission matters, because it is the single most
> common reason practitioners reach for it and get a worse result at a higher price. Treat "the model
> returns badly-shaped output" as a schema problem until you have proven it is not.

---

### 4.13 Privacy and data retention — what the platform actually promises

Fine-tuning means uploading your data. The instructor touches this only obliquely, at
[22:17]–[23:00], where he puts his API key in a Colab environment variable and moves on. The
questions an enterprise reviewer will ask have specific answers:

| Question | Answer | Consequence for you |
|---|---|---|
| Is my fine-tuned model private to my organisation? | Yes — a fine-tuned model is accessible only to your organisation | It is not a public model, but it is not *yours* either: no weights, no export |
| Does my fine-tuning data train the provider's base models? | **No.** API and fine-tuning data are not used to train base models by default | This is the default, not an opt-in — you do not need to configure anything |
| How long is my uploaded file retained? | **Until you delete it.** Training files persist until explicitly deleted | Data hygiene is your job. Write the deletion into the pipeline, not the runbook |
| How long are API requests retained? | Up to **30 days**, for abuse monitoring | Prompts and completions in your logs exist provider-side for a month |
| Can I get zero data retention (ZDR)? | Yes, on eligible endpoints and with an approved org configuration | ZDR organisations **cannot** opt into data sharing — the two are mutually exclusive |
| What is "data sharing", and is it on? | An **org-owner** setting that permits your traffic to be used for training in exchange for discounted inference or complimentary tokens. **Off by default.** | Off unless an owner turned it on. Verify before an audit, not during one |
| Does the free-token/shared-traffic programme cover fine-tuning? | **No. Fine-tuning training, fine-tuned models themselves, evals, and tool use are excluded** from that programme | The discount applies to base-model inference on shared traffic. Do not budget for it covering your FT bill |
| Who can read my training file? | Provider staff under the abuse-monitoring retention window | If the data is regulated, that window is the finding |

> **Beyond the video — the three-line policy that answers 90% of security questionnaires.** Put this
> in your design doc *before* the first upload:
>
> ```text
> 1. RETENTION: training files are deleted by our pipeline N days after job completion
>    (default: immediately on success). Nothing relies on the provider retaining them.
> 2. NO PII IN TRAINING DATA: examples are scrubbed of names, emails, phone numbers, account
>    IDs and payment data before upload. If a field is needed for the task, it is synthesised.
> 3. NO SHARED-TRAFFIC OPT-IN: the organisation is not enrolled in the data-sharing programme.
>    If anyone proposes enrolling for the discount, the exclusion list above is the rebuttal.
> ```
>
> **The uncomfortable one is #2, not #1.** The instructor's own demo dataset is customer-support
> data — exactly the category that carries PII in real deployments. A fine-tuning file is a
> permanent copy of your worst-case examples sitting in a third party's storage, and unlike a
> database it has no row-level access control, no audit log you can query, and no way to redact a
> single row after the fact. Scrub before upload.

---

## 5. The End-to-End Pipeline

The video's workflow is twelve stages long and the instructor walks all of them. Below is the whole
pipeline, with the exact artefacts each stage produces.

```
  (1) FRAME          (2) COLLECT         (3) SCRUB            (4) FORMAT
  decide FT is       gather/generate     remove PII,          messages JSONL,
  the right tool     demonstrations      dedupe, balance      one object/line
      │                   │                  │                     │
      ▼                   ▼                  ▼                     ▼
  decision doc        raw pairs          clean pairs           data.jsonl
                                                                   │
  (5) SPLIT          (6) VALIDATE        (7) COUNT            (8) COST
  train/val/test     run the 7 checks    tiktoken +           tokens × epochs
      │              before uploading     billable counter     × $/1M
      ▼                   │                  │                     │
  train.jsonl             ▼                  ▼                     ▼
  val.jsonl          error histogram    token + $ estimate    go / no-go
                                                                   │
  (9) UPLOAD         (10) CREATE         (11) WATCH           (12) EVALUATE
  Files.create        jobs.create         poll status,          holdout eval,
      │              upload+start         loss curves          paraphrase test,
      ▼                   │                  │                  A/B vs base
  file-XXXX               ▼                  ▼                     │
  ftjob-XXXX          job.hyperparams    result CSV                 ▼
                                                              ship / rollback
```

### Stage-by-stage contract

| # | Stage | Input | Operation | Output | Failure mode if skipped |
|---|---|---|---|---|---|
| 1 | **Frame** | The observed problem | Decide FT vs prompting vs RAG vs structured outputs | A decision, written down | You fine-tune a knowledge problem and get a confident hallucinator (CS-05) |
| 2 | **Collect** | Production logs, SMEs, or a stronger model | Produce input→output demonstrations | Raw pairs | You train on data that does not match production's distribution |
| 3 | **Scrub** | Raw pairs | PII removal, dedupe, label balance, outlier trimming | Clean pairs | A permanent PII copy in a third party's storage (§4.13); duplicates inflate eval |
| 4 | **Format** | Clean pairs | Serialise to `{"messages": [...]}` per line | `data.jsonl` | Any of the seven validation errors (§4.2.2) |
| 5 | **Split** | `data.jsonl` | Deterministic split; **dedupe across the split** | `train.jsonl`, `val.jsonl`, `test.jsonl` | Leakage → validation loss below train loss → false confidence |
| 6 | **Validate** | `train.jsonl` | Run every structural check locally | Error histogram, expected all-zeros | The API rejects the file and you lose a round-trip; worse, `missing_content` slips through and bills you |
| 7 | **Count** | `train.jsonl` | Tokenise with the **right** encoding; add per-message overhead | Token count + per-epoch bill | You budget 562 and pay for 672 (§4.5.2) |
| 8 | **Cost** | Token count, `n_epochs`, price | `tokens × epochs × $/1M`, plus the inference break-even | Go / no-go | You discover at scale that inference dominates (§11) |
| 9 | **Upload** | `train.jsonl`, `val.jsonl` | `client.files.create(file=..., purpose="fine-tune")` | `file-XXXXXXXX` | You re-upload the same file every run and cannot map models to data |
| 10 | **Create** | File IDs, model ID, hyperparameters | `client.fine_tuning.jobs.create(...)` | `ftjob-XXXXXXXX` | You leave `"auto"` everywhere and cannot reproduce (§4.7.3) |
| 11 | **Watch** | Job ID | Poll `retrieve`; stream `list_events`; download `result_files` | Loss curves, `trained_tokens` | Overfitting runs to completion and you ship it (§4.9) |
| 12 | **Evaluate** | The finished model ID | Holdout eval vs the base model on the **same** prompts | A decision with a number attached | You ship a model that is worse, and cannot tell (§12) |

### The nine API calls that make up the whole workflow

The video demonstrates exactly four of these at [50:37]–[55:07]; the rest are required for a run you
can defend.

```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---- 1. upload the training file -------------------------------------------------
train_file = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
val_file   = client.files.create(file=open("val.jsonl",  "rb"), purpose="fine-tune")

# ---- 2. inspect the upload (the check the video skips) ---------------------------
info = client.files.retrieve(train_file.id)
print(info.id, info.bytes, info.status)      # status must reach "processed"

# ---- 3. create the job -----------------------------------------------------------
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,             # <- never None in a real run
    model="gpt-4.1-mini-2025-04-14",         # snapshot ID, not the alias
    suffix="support-v3",
    method={"type": "supervised",
            "supervised": {"hyperparameters": {
                "n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": "auto"}}},
    seed=42,
    metadata={"data_version": "2026-09-01", "owner": "platform-team"}
)

# ---- 4. poll until terminal ------------------------------------------------------
import time
while client.fine_tuning.jobs.retrieve(job.id).status in ("validating_files", "queued", "running"):
    time.sleep(30)
final = client.fine_tuning.jobs.retrieve(job.id)
print(final.status, final.trained_tokens, final.fine_tuned_model)

# ---- 5. stream the human-readable events (optional, for logs) --------------------
for ev in client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=100).data:
    print(ev.created_at, ev.level, ev.message)

# ---- 6. pull the metrics CSV -----------------------------------------------------
csv_id = final.result_files[0]
metrics = client.files.content(csv_id).text       # step, train_loss, valid_loss, ...

# ---- 7. call the model (the only call that runs in production) -------------------
resp = client.chat.completions.create(
    model=final.fine_tuned_model,                  # "ft:gpt-4.1-mini-2025-04-14:org:support-v3:AbCd1234"
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": user_turn}],
    temperature=0.2,
)

# ---- 8. clean up: delete the uploaded files once the job is done ------------------
client.files.delete(train_file.id)
client.files.delete(val_file.id)

# ---- 9. when the model is retired or superseded, delete it too --------------------
client.models.delete(final.fine_tuned_model)
```

> **Beyond the video — the two stages the video skips entirely are stages 1 and 12**, and they are the
> two that decide whether the project was worth doing. He never asks whether fine-tuning was the
> right tool (§4.12 has the answer for the most common case), and he never compares the fine-tuned
> model against the base model on a held-out set. His only evaluation is a single ad-hoc chat turn
> against a question that is in the training data.
>
> **A fine-tuning run without a paired baseline is not an experiment; it is a purchase.** Before you
> create a job, write down the number the fine-tuned model must beat — accuracy, format-compliance
> rate, latency, and cost per 1k calls — measured on the *same* held-out set with the *same* prompts
> against the base model. If you cannot state that number in advance, you cannot evaluate the result,
> and you will ship on vibes.

---

## 6. Hands-On Code (annotated)

Everything here comes from the two companion notebooks. The full path is
`D:\Finetuning\_source\repo\Complete-LLM-Finetuning-main\LLM Fine-Tuning-20-GPT-Finetuning\`.

| Notebook | Size | Which run | What it demonstrates |
|---|---|---|---|
| `openai_api_and_finetuning_of_gpt_model.ipynb` | 34 KB | The **earlier**, abandoned run | `gpt-3.5-turbo`, a 4-unique-row inline dataset, auto hyperparameters resolving to 10 epochs, and a 404 at inference |
| `openai_api_and_finetuning_of_gpt_model (1).ipynb` | 96 KB | **The run the video narrates** | validator, tiktoken, the two token counters, the auto-epoch policy, the billing arithmetic, the `gpt-4o-2024-08-06` job |

The first notebook is the more instructive of the two, because it is the failure.

### 6.1 Environment setup

```python
# The video's environment, at [22:17]-[23:00]. Colab with a CPU runtime — no GPU needed, because
# the training happens on OpenAI's servers [22:04]-[22:14].
!pip install --quiet openai tiktoken
```

```python
# The API key as a Colab secret, never inline. The instructor sets this in the Colab key panel
# and reads it with userdata; the equivalent env-var form is below.
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])   # never client = OpenAI(api_key="sk-...")
```

> **Beyond the video:** `pip install openai` with no pin is a reproducibility hazard for an SDK whose
> major versions have changed the fine-tuning API shape more than once — `fine_tuning.jobs.create`
> itself did not exist in the 0.x SDK. Pin it: `pip install "openai>=1.40,<3"`, record the resolved
> version in your run log, and note that the `method={"type": "supervised", ...}` nesting replaced the
> flat `hyperparameters={...}` argument in the 2.x SDK.

### 6.2 Dataset inspection — the part that prevents wasted jobs

```python
import json

DATASETS = ["data.jsonl", "data2.jsonl"]

def audit(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    uniq_convs = len({tuple((m["role"], m["content"]) for m in r["messages"]) for r in rows})
    users      = {m["content"] for r in rows for m in r["messages"] if m["role"] == "user"}
    assts      = {m["content"] for r in rows for m in r["messages"] if m["role"] == "assistant"}
    missing    = sum(1 for r in rows if not any(m["role"] == "assistant" for m in r["messages"]))
    sys_prompts = {next((m["content"] for m in r["messages"] if m["role"] == "system"), None)
                   for r in rows}
    print(f"{path:<14} rows={len(rows):>3}  unique_convs={uniq_convs:>3}  "
          f"unique_users={len(users):>3}  unique_assts={len(assts):>3}  "
          f"missing_assistant={missing}  system_prompts={len(sys_prompts)}")
    return rows

for p in DATASETS:
    audit(p)
```

Actual output on the repo's two files:

```text
data.jsonl     rows= 10  unique_convs= 10  unique_users= 10  unique_assts= 10  missing_assistant=0  system_prompts=1
data2.jsonl    rows= 10  unique_convs= 10  unique_users= 10  unique_assts= 10  missing_assistant=1  system_prompts=1
```

**These two files are byte-for-byte equivalent for training purposes except that `data2.jsonl` is
deliberately broken on line 1.** It exists only to make the validator fire.

| File | Bytes | Rows | Unique convs | Missing assistant | Uploaded in the video? | Purpose |
|---|---|---|---|---|---|---|
| `data.jsonl` | 4,239 | 10 | **10** | 0 | **Yes** | The training file |
| `data2.jsonl` | 4,060 | 10 | **10** | **1** | **No** | The negative fixture for the validator |
| first-notebook **inline** dataset | 6,429 | 10 | **4** | 0 | Yes (earlier run) | The actual cause of the 10-epoch overfit |

The third row is the one that matters and the one that is easy to get wrong. The **first** notebook
does not read a file. It builds its rows from a Python list, and that list has only **4 unique
conversations**, repeated to fill 10 rows. That is the dataset behind the `file-PsuEt1x4gLPqY8f4sDD49t`
upload — 6,429 bytes, which is larger than either `.jsonl` file precisely because the rows are
duplicated rather than distinct.

> **Beyond the video — the audit above is the cheapest possible intervention and it is the one that
> catches the expensive mistakes.** Five numbers, computed before a single byte is uploaded:
>
> | Check | Threshold | What it catches |
> |---|---|---|
> | `unique_convs / rows` | **> 0.95** | Duplicates silently reweighting your training set |
> | `unique_users / rows` | **> 0.95** | One prompt template repeated (the model learns the template, not the task) |
> | `missing_assistant` | **0** | The `example_missing_assistant_message` error, caught locally |
> | `system_prompts` | **1**, or a deliberate small set | Inconsistent personas; a model that averages two voices |
> | `unique_assts / unique_users` | **≈ 1** | Multiple conflicting answers for the same question — the single worst data defect for SFT |

### 6.3 The validator, with the bug the video leaves in

The instructor builds this live and hits a real bug on camera, which is the best five minutes of the
video for anyone learning to write data tooling.

**Attempt 1 — cell 14, this crashes:**

```python
# notebook cell 14 — reproduced verbatim from the video, EXCEPT for the line marked below
import json

def validate(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    format_errors = {}
    for ex in data:
        if not isinstance(ex, dict):
            format_errors["data_type"] = format_errors["data_type"] + 1   # <-- KeyError, every time
            continue
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] = format_errors["missing_messages_list"] + 1
            continue
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] = format_errors["message_missing_key"] + 1
            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] = format_errors["message_unrecognized_key"] + 1
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] = format_errors["unrecognized_role"] + 1
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] = format_errors["missing_content"] + 1
        if not any(m.get("role", None) == "assistant" for m in messages):
            format_errors["example_missing_assistant_message"] = \
                format_errors["example_missing_assistant_message"] + 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

validate("data2.jsonl")
```

```text
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
Cell In[14], line 9
      7 for ex in data:
      8     if not isinstance(ex, dict):
----> 9         format_errors["data_type"] = format_errors["data_type"] + 1
KeyError: 'data_type'
```

**Attempt 2 — cells 17–23, the fix:**

```python
from collections import defaultdict

def validate(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    format_errors = defaultdict(int)                 # <-- the whole fix, one line
    for ex in data:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        if not any(m.get("role", None) == "assistant" for m in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

validate("data.jsonl")     # -> No errors found
validate("data2.jsonl")    # -> Found errors:  example_missing_assistant_message: 1
```

The two lines that matter are the import and the constructor: `defaultdict(int)` means a missing key
reads as `0` and is created on first increment, so `+= 1` never raises. The instructor's diagnosis on
camera — *"it is not initialized, we need to initialize it"* [33:41] — is right, and his fix is
exactly the cookbook's.

**A subtler defect in that code, which the video does not catch.** Look at the `continue` statements:

```python
if not isinstance(ex, dict):
    format_errors["data_type"] += 1
    continue                     # correct: no `messages` to inspect
messages = ex.get("messages", None)
if not messages:
    format_errors["missing_messages_list"] += 1
    continue                     # correct
for message in messages:         # <- if `messages` is the wrong TYPE, this raises TypeError
```

`messages` can be a string, an int, or a dict. `for message in messages` over a string iterates
characters; over a dict it iterates keys; over an int it raises `TypeError` and the whole validation
aborts with no error histogram. The official cookbook has the identical hole. One extra guard closes
it:

```python
if not isinstance(messages, list):
    format_errors["messages_not_a_list"] += 1
    continue
```

**And the structural trap that produced the module's most-cited Correction:** the role allow-list in
the code above is `("system", "user", "assistant", "function")`. That is the **completion-era** list
from the original cookbook.

> **Correction:** the validator's recognised-role tuple
> `("system", "user", "assistant", "function")` that the notebook (cells 14 and 23) copies from the
> cookbook is **out of date and will produce false errors on modern data.** The current chat API
> accepts **`system`, `user`, `assistant`, `tool`** — `tool` replaced `function` when the tool-calling
> API landed, and `function` is now legacy-only. Consequences, concretely:
>
> | Your data | Copied validator says | Reality |
> |---|---|---|
> | A row with `{"role": "tool", "tool_call_id": "call_1", "content": "..."}` | `unrecognized_role: 1` — **false positive** | Valid; the API accepts it |
> | A row with `{"role": "developer", ...}` | `unrecognized_role: 1` — false positive | Accepted by newer reasoning-era models as a system-equivalent |
> | A row with `{"role": "assistant", "function_call": {...}}` | passes | Legacy; migrate to `tool_calls` |
> | A row with `{"role": "assistant", "content": null, "tool_calls": [...]}` | **`missing_content: 1` — false positive** | This is the *correct* modern shape for an assistant turn that only calls a tool |
>
> The third and fourth rows are the dangerous ones, because the validator does not merely complain —
> it counts a **correct** row as broken, and then the `missing_content` check's per-message counting
> inflates your error histogram in a way that looks like a data problem. The general lesson is bigger
> than this key list: **a copied allow-list is a versioned artefact, and yours is now older than the
> API.** Pin it beside your SDK version and re-read it whenever you upgrade.

An allow-list that is correct as of the current API, and closes the type hole:

```python
VALID_ROLES = ("system", "developer", "user", "assistant", "tool")   # `function` = legacy opt-in
NAME_OK_ROLES = ("system", "user", "assistant", "tool")              # `name` is per-role in the docs
```

### 6.4 Token counting — tiktoken and the two counters

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")   # what the video uses [39:50]-[40:17]
```

The demo (cell 33) tokenises a short string and prints each token's ID, and the notebook's token
table is the authoritative record of what the encoding actually produced:

| Token text | Notebook token ID | The transcript says | Verdict |
|---|---|---|---|
| `Hello` | `9906` | "9906" | Agrees |
| `,` | `11` | "11" | Agrees |
| ` how` | `1268` | "12868" | **Transcript is garbled** — extra digit |
| ` are` | `527` | "R then U…" | **Transcript is garbled** — speech-to-text mangled the token text |
| ` you` | `499` | — | — |
| `?` | `30` | "30" | Agrees |

The six token IDs sum to **12,241**, which is the cell's printed total and matches the notebook. Where
the auto-transcript and the notebook disagree, the notebook wins — every time.

> **Correction (restated — the full version is in §4.5.1):** the instructor's *"this is also a
> transformer based model"* [40:14]–[40:16] is wrong. `tiktoken` is a **BPE merge table** — a
> deterministic lookup with no weights and no forward pass. The three consequences that matter *to the
> counting code below*: it is **exact** (the same string always gives the same count, so the arithmetic
> in §4.6 is not an estimate), it is **fast** (millions of tokens per second on one CPU core, so there
> is no excuse for not counting), and its **vocabulary is frozen at release** — `cl100k_base` and
> `o200k_base` are different fixed tables, so you cannot count a `gpt-4.1` model correctly with the
> `cl100k_base` encoder the video uses (§4.5.4).

**The two counters, side by side.** The difference between them is the difference between a correct
budget and a 16% underestimate.

```python
# ---- Counter A: content only. This is what the notebook prints. --------------------
def count_total_tokens(examples):
    """Total content tokens across the dataset, ignoring all per-message overhead."""
    return sum(len(encoding.encode(m["content"]))
               for ex in examples for m in ex["messages"])

# ---- Counter B: the cookbook's billing estimate, overhead included. -----------------
def num_tokens_from_messages(messages, tokens_per_message=3,
                             tokens_per_name=1, tokens_per_reply=3):
    """
    Estimated tokens the API will bill.

    tokens_per_message  : per-message priming overhead (<|im_start|>role\n ... <|im_end|>\n)
    tokens_per_name     : extra when a `name` field is present
    tokens_per_reply    : the assistant reply priming tokens at the end of the conversation
    """
    num = tokens_per_reply
    for message in messages:
        num += tokens_per_message
        for key, value in message.items():
            num += len(encoding.encode(value))
            if key == "name":
                num += tokens_per_name
    return num

total_a = count_total_tokens(data)
total_b = sum(num_tokens_from_messages(ex["messages"]) for ex in data)
print(total_a, total_b, f"{(total_b - total_a) / total_a:+.1%}")
# -> 562 672 +19.6%
```

| Quantity | Value | What it counts |
|---|---|---|
| `count_total_tokens` | **562** | Content characters only |
| `num_tokens_from_messages` | **672** | Content + 3 tokens per message + 3 per reply |
| Messages in the dataset | 21 | 10 system + 10 user + 10 assistant, minus the missing one in `data2` |
| Overhead | **110 tokens** | `21 × 3 + 10 × 3 + ...` |
| **Overhead fraction** | **+19.6%** | On a 672-token dataset |
| Billed at 3 epochs | **2,016** | `672 × 3` |
| Billed if you used count A | **1,686** | `562 × 3` — a 16% underestimate |

> **Beyond the video — the overhead is a *constant per message*, so it does not stay at 19.6%.** It is
> amortised away as your examples get longer, which means the error from using the wrong counter
> moves in the opposite direction from what you would expect:
>
> | Example length (content tokens) | Messages | Overhead | Overhead % | Cost error if you use counter A |
> |---|---|---|---|---|
> | 56 (the video's) | 2.1 | 9 | **+16%** | 16% under-budget |
> | 200 | 2.1 | 9 | +4.5% | 4.5% under-budget |
> | 600 | 2.1 | 9 | +1.5% | 1.5% under-budget |
> | 2,000 | 2.1 | 9 | +0.45% | Negligible |
>
> So counter A is *nearly* fine for a production dataset of long examples and *badly* wrong for the
> short-example demos everyone builds first. **Use counter B. It costs nothing and it is correct in
> both regimes.** The one case where the distinction bites hardest is the very small dataset, which is
> exactly the case where a surprise bill is most annoying.

### 6.5 The billing arithmetic, exactly as the notebook computes it

Cell 67 of the run notebook is the only place the video's cost figure is stated with its derivation,
and it is correct:

```python
# notebook cell 67 — the instructor's own arithmetic, reproduced
n_epochs = 3
billed_tokens_per_epoch = 672          # = sum(num_tokens_from_messages(ex) for ex in data)
total_billed_tokens = billed_tokens_per_epoch * n_epochs          # 2016

price_per_million = 1.50               # USD per 1,000,000 training tokens, gpt-4.1-nano
price_per_token   = price_per_million / 1_000_000                 # 0.0000015

training_cost_usd = total_billed_tokens * price_per_token         # 0.003024
training_cost_inr = training_cost_usd * 91                        # 0.275184
```

```text
672 tokens/epoch × 3 epochs = 2,016 billed training tokens
2,016 × $0.0000015        = $0.003024
$0.003024 × ₹91           = ₹0.275184
```

**Three paise.** That is the honest headline number for the demo, and it is worth sitting with,
because it explains why he is comfortable experimenting so casually — and why the mental model he
builds ("fine-tuning is cheap") is the exact wrong lesson to carry into a real dataset.

The scaling, from the same arithmetic:

| Dataset | Content tokens | **Billed** tokens/epoch | × 3 epochs | `gpt-4.1-nano` @ $1.50/M | `gpt-4.1-mini` @ $3.00/M | `gpt-4.1` @ $25/M |
|---|---|---|---|---|---|---|
| Video demo (10 rows) | 562 | 672 | 2,016 | **$0.0030** | $0.0060 | $0.050 |
| 100 rows | 5,620 | 6,720 | 20,160 | $0.030 | $0.060 | $0.50 |
| 1,000 rows | 56,200 | 67,200 | 201,600 | $0.302 | $0.605 | $5.04 |
| 10,000 rows | 562,000 | 672,000 | 2,016,000 | $3.02 | $6.05 | $50.40 |
| 100,000 rows | 5.62M | 6.72M | 20.16M | $30.24 | $60.48 | $504.00 |
| 1,000,000 rows | 56.2M | 67.2M | 201.6M | $302.40 | $604.80 | **$5,040.00** |

*(`gpt-4.1` at ~$25/M is a 2026 list figure; verify before budgeting. `gpt-4.1-nano` at $1.50/M and
`gpt-4.1-mini` at ~$3.00/M are the figures confirmed for this module.)*

Read the last column against the previous table in §4.6.5. **At a million training rows on the
flagship model, training alone costs $5,040 — and that is the *cheap* half of the bill.** Inference at
production volume on that model is many multiples of it, and the fine-tuned model carries a 2×
inference markup over the base. This is the arithmetic that ends most hosted fine-tuning projects, and
it is nowhere in the video.

```python
# The one function to put in your repo: what will this dataset cost, all-in, before you upload.
def budget(training_rows, val_rows, avg_content_tokens, n_epochs,
           train_price_per_m, in_price_per_m, out_price_per_m,
           calls_per_month, avg_prompt_tokens, avg_completion_tokens):
    MSG_OVERHEAD = 9      # 3 per message + 3 per reply, for a 2.1-message average row

    train_tokens = training_rows * (avg_content_tokens + MSG_OVERHEAD)
    val_tokens   = val_rows      * (avg_content_tokens + MSG_OVERHEAD)
    train_usd    = (train_tokens * n_epochs + val_tokens * n_epochs) / 1e6 * train_price_per_m

    prompt_tokens     = calls_per_month * avg_prompt_tokens
    completion_tokens = calls_per_month * avg_completion_tokens
    infer_usd = (prompt_tokens / 1e6 * in_price_per_m) + (completion_tokens / 1e6 * out_price_per_m)

    return {"train_usd": train_usd, "infer_usd_per_month": infer_usd,
            "train_share_of_first_year": train_usd / (train_usd + 12 * infer_usd)}

# 10,000 rows, 3 epochs, gpt-4.1-mini, 500k calls/month, 800-token prompts, 200-token completions
b = budget(9_000, 1_000, 562, 3, 3.00, 0.80, 3.20, 500_000, 800, 200)
# -> train_usd: 6.05   infer_usd_per_month: 720.00   train_share_of_first_year: 0.0007
```

**Training is 0.07% of the first-year bill.** That single line is the most important number in this
module and the video never computes it.

### 6.6 Creating the job — the video's four calls

The instructor demonstrates the four API calls at [50:37]–[55:07]: upload the file, create the job,
list the jobs, retrieve the model. Two of the details he gets exactly right are the ones most people
get wrong.

```python
# [50:37]-[52:10] — upload. `purpose="fine-tune"` is mandatory and easy to omit.
training_file = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
print(training_file.id)          # -> file-Hn8ooHUNEzwcPjBasULS25
```

```python
# [52:15]-[54:16] — create. Note the snapshot ID and the suffix.
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-4o-2024-08-06",            # a dated SNAPSHOT, not the moving alias
    suffix="second-finetune-model",
    method={"type": "supervised",
            "supervised": {"hyperparameters": {
                "batch_size": 16,
                "learning_rate_multiplier": 1.0,
                "n_epochs": 3}}}
)
print(job.id)                             # -> ftjob-XXXXXXXX
```

| Detail | The video's value | Why it is the right instinct |
|---|---|---|
| `model` | `"gpt-4o-2024-08-06"` | A **dated snapshot**, not `"gpt-4o"`. The alias moves under you; the snapshot does not. The notebook's cell 4 even carries `# gpt-4o-2024-08-06` as a comment — the alias is what he first wrote, and the snapshot is what he shipped. |
| `suffix` | `"second-finetune-model"` | The suffix is the one human-readable handle on the model. It is also, in the video's case, the *only* record that this was the second attempt — there is no `metadata` and no `seed`. |
| `training_file` | a `file-...` ID | Not a path, not a filename. The two-step upload-then-reference is deliberate: it lets the same validated file back multiple jobs, and it is what makes the file's retention independent of the job. |
| `validation_file` | omitted | The one omission that costs the most (§4.9.1) |

> **Beyond the video — add the two fields he omits, on every real job.** `seed=42` and
> `metadata={"data_version": ..., "git_sha": ..., "owner": ...}`. Between them they turn an
> unreproducible run into a documented one, and they cost nothing. Concretely, the question you will
> be asked six months later is *"which data trained the model that is currently serving?"* — and
> without these fields the only honest answer is "I don't know". The model ID itself does not encode
> the dataset; the suffix is a free-text label you will forget the meaning of.

### 6.7 The chat call, and the 404 the first notebook earns

The first notebook's final cell is the most instructive code in either file, because it fails:

```python
# first notebook, cell 20 — the video's first attempt at inference
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:personal::9XbBvXXXXXXXXXXXXXXXXX",
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": "What warranty does a smartphone come with?"}]
)
```

```text
NotFoundError: Error code: 404 - {'error': {'message':
  'The model `ft:gpt-3.5-turbo-0125:personal::9XbBvXXXXXXXXXXXXXXXXX` does not exist
   or you do not have access to it.', 'type': 'invalid_request_error',
   'param': None, 'code': 'model_not_found'}}
```

| The apparent reading | The actual reading |
|---|---|
| "The fine-tuned model is broken" | The job never reached `succeeded` — it is listed as `cancelled` in the notebook. **A cancelled job produces no model**, so the ID is a placeholder that was never minted. |
| "My API key lacks access" | The key is fine; the model string is simply not a real deployment ID. |
| "The `ft:` ID format is wrong" | The format is right and the ID was never issued. |

The second notebook's chat cell works, and it is the only evaluation the video performs anywhere:

```python
# second notebook — the working call, against the model whose job succeeded
response = client.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:personal::XXXXXXXX",
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user",   "content": "What warranty does a smartphone come with?"}],
    temperature=0.2,
)
print(response.choices[0].message.content)
```

**Decoding the `ft:` model ID** is worth doing once, because everything about a fine-tuned model's
identity is in that string:

```text
ft: gpt-4o-2024-08-06 : personal : second-finetune-model : AbCd1234
│   │                   │          │                       │
│   │                   │          │                       └─ random job hash (NOT guessable)
│   │                   │          └───────────────────────── the `suffix` you passed
│   │                   └──────────────────────────────────── the org slug
│   └──────────────────────────────────────────────────────── the BASE snapshot ID
└──────────────────────────────────────────────────────────── fixed literal prefix
```

Two properties follow. **(1)** The base model is in the ID, so a base-model deprecation is
mechanically detectable by string match across your config — which is exactly the alert you want to
have wired up when a vendor announces a sunset (§16.8). **(2)** The base model is *also* in the ID
because **you do not get a standalone model** — a fine-tuned model is a set of adapter weights
attached to a specific base snapshot, addressed as a whole. You cannot port it, re-base it, or
extract it.

> **Beyond the video — the 404 in the first notebook is the single best debugging lesson in the
> module, and it has nothing to do with this specific error.** The instructor's `model_not_found`
> arose because he chatted with a model before the job finished. The generic version of that mistake
> is an **unconditional `sleep` instead of a status poll**, and it is endemic in tutorial code:
>
> ```python
> # What tutorial code does — a race condition with a fixed delay.
> create_job()
> time.sleep(600)                      # "10 minutes should be enough"
> chat("ft:gpt-4o-...")                # 404 if the queue was slow; 404 if the job failed
>
> # What production code does — poll to a TERMINAL state, then branch.
> TERMINAL = {"succeeded", "failed", "cancelled"}
> while True:
>     j = client.fine_tuning.jobs.retrieve(job.id)
>     if j.status in TERMINAL:
>         break
>     time.sleep(30)
> if j.status != "succeeded":
>     raise RuntimeError(f"fine-tune {j.id} ended {j.status}: {j.error}")   # <- never skip this
> model_id = j.fine_tuned_model                                          # guaranteed non-null here
> ```
>
> The failure signature is diagnostic: **a 404 on a `ft:` model means "not ready or not yours"**, and
> the two are distinguished by checking `jobs.retrieve()` for a `succeeded` status — nothing else.
> It is never a credentials problem, and retrying it never helps.

---

## 7. Hyperparameters & Configuration — Every Knob

§4.7 covered the three training hyperparameters in depth. This section is the complete configuration
surface for a hosted fine-tuning pipeline — the training knobs, the job knobs, the file-API knobs and
the **inference** knobs, because the last group is where the money and the latency actually live and
nobody thinks of them as "configuration".

### 7.1 Training hyperparameters (recap table, full detail in §4.7)

| Param | What it does | Typical | Safe range | Too high → | Too low → | API key |
|---|---|---|---|---|---|---|
| `n_epochs` | Full passes over the dataset; linear cost multiplier | `3` for small data, `1–2` for large | 1–3 (≥1k rows); 3–10 (<100 rows) | Memorisation; valid loss rises | Behaviour change never lands | `method.supervised.hyperparameters.n_epochs` |
| `batch_size` | Examples per optimiser step in the provider's trainer | `"auto"` (~0.2% of rows, ≤256, ≥1) | `"auto"` unless you have a reason | Slower convergence; LR mismatch | Noisy gradient; LR mismatch | `...hyperparameters.batch_size` |
| `learning_rate_multiplier` | **Multiplier** on the hidden base LR | `"auto"` | 0.5–2 (chat-era); legacy 0.02–0.2 | Divergence, forgetting | No learning at all | `...hyperparameters.learning_rate_multiplier` |
| `beta` (DPO only) | KL anchor strength to the reference model | `"auto"` | 0.01–0.5 | Model barely moves from SFT | Likelihood collapse on both chosen and rejected | `method.dpo.hyperparameters.beta` |

### 7.2 Job-level configuration

| Param | What it does | Typical | Why it matters | Omit it and… |
|---|---|---|---|---|
| `model` | The base model to train | A **dated snapshot ID** | The alias moves; the snapshot does not | Your job silently trains a different base next month |
| `training_file` | `file-...` ID from `files.create` | required | Provenance | — |
| `validation_file` | `file-...` ID for holdout loss | 10–20% of the data | The only automatic overfitting detector | You get no generalisation signal at all |
| `suffix` | Up to 18 chars appended to the model name | `"support-v3"` | The only human handle on the artefact | You cannot map models to data |
| `seed` | Best-effort determinism | `42` | A/B comparisons need it | Two runs differ and you cannot attribute the difference |
| `metadata` | Up to 16 string key/value pairs | `{"data_version": ..., "git_sha": ...}` | Answers "what trained this?" in six months | The answer is "I don't know" |
| `integrations` | Push events to W&B / similar | `[]` in a demo | Loss curves without polling | You poll by hand |
| `method.type` | `"supervised"` / `"dpo"` / `"reinforcement"` | `"supervised"` | Selects the trainer | Defaults are version-dependent — set it |
| `hyperparameters.n_epochs` | see §7.1 | `3` | Cost × quality | Auto policy decides (§4.8) |

### 7.3 File-API configuration

| Param | What it does | Typical | Gotcha |
|---|---|---|---|
| `file` | The file object to upload | `open("data.jsonl", "rb")` | **Binary mode.** Text mode with an explicit encoding works on Linux and breaks on Windows |
| `purpose` | What the file is for | `"fine-tune"` | Omitting or mistyping it means the file cannot be used as training data |
| `info.bytes` | Server-side size | — | Compare to the local `os.path.getsize`. A mismatch means a truncated upload |
| `info.status` | Processing state | must reach `processed` | `uploaded` is not ready; creating a job on an unprocessed file fails |
| `client.files.delete(id)` | Removes the file | after job success | **Retention is your responsibility** — files persist until deleted (§4.13) |

### 7.4 Inference configuration — the knobs nobody calls "configuration"

Once you have a fine-tuned model, the parameters you pass at **call time** determine your bill far
more than anything on this page:

| Param | Effect on cost | Effect on quality | Recommended for a fine-tuned model |
|---|---|---|---|
| `model` | The fine-tuned ID carries a **~2× input / ~2× output markup** over the base | — | The FT model, obviously — but see §4.6.5 for when the markup is not worth it |
| `messages` length | **Linear in input cost.** Every token you can delete is money | The system prompt you trained with must be present and byte-identical | Trim the system prompt only if you trained without it |
| `max_tokens` | Caps the most expensive side (output is 4× input at these ratios) | Too low truncates JSON mid-object | Set it. Never leave it unbounded on a structured output |
| `temperature` | None | 0–0.3 for extraction/classification; 0.7+ for chat | **0.2** is the video's choice and a sane default |
| `top_p` | None | Interacts with temperature | Leave at default unless you have an eval |
| `response_format` | Adds constrained decoding; no training cost | **Guarantees** the schema | Use it in addition to the fine-tune, not instead (§4.12) |
| `frequency_penalty` / `presence_penalty` | None | Can break a fine-tune's learned style | **0** for a fine-tuned model |
| `stop` | Truncates output early = cheaper | Necessary for some templates | Set it if your format has a terminator |
| `seed` | None | Reproducibility for evals | Set it in your eval harness |
| `n` | Linear in output cost | — | 1 in production; >1 only in evals |
| `stream` | No cost change | Time-to-first-token, not total | Use it in production for perceived latency |
| Prompt caching | Cached input at ~25% of the input price (e.g. $0.05/M vs $0.20/M) | None | **Put the stable system prompt first.** This is the single largest inference-cost lever you control |

### 7.5 The three-way interaction you must understand

`n_epochs`, `batch_size` and `learning_rate_multiplier` do not act independently. They form one
coupled system:

```
total_optimiser_steps = n_epochs × (n_examples / batch_size)

effective_learning_signal ≈ learning_rate_multiplier × total_optimiser_steps
```

Read that second line as a *budget*, not a formula. Three configurations with the same effective
signal:

| Config | Steps (1,000 examples) | LR multiplier | Signal | Note |
|---|---|---|---|---|
| A | `3 × (1000/16)` = 187 | 1.0 | 187 | The sane default |
| B | `1 × (1000/16)` = 62 | 3.0 | 186 | Same signal, **1/3 the training cost**, but a coarser trajectory |
| C | `3 × (1000/1)` = 3,000 | 0.06 | 180 | Same signal, **16× the cost**, and a very noisy path |

**B is the interesting one**: cutting epochs and raising the LR multiplier gives the same learning
signal for a third of the money. It is also the configuration most likely to fail, because the
multiplier is applied to a hidden base rate and a 3.0 multiplier is well outside the documented
experimentation range. The practical rule: **when cost matters, cut `n_epochs` first and leave the LR
multiplier alone** — the epoch cut is predictable and the LR change is not.

And the batch-size interaction, which is the one that catches people out:

| Change | Effect on the optimiser | Effect on convergence |
|---|---|---|
| Raise `batch_size` 2× | Half as many steps; gradient is the mean of 2× as many examples, so it is *less* noisy | Usually **raise** the LR with it, or convergence slows |
| Lower `batch_size` 2× | Twice as many steps, each noisier | Usually **lower** the LR, or the noise prevents settling |

This is why `"auto"` is the right answer for `batch_size` on almost every run: the provider's LR
policy was fitted to the batch size it chose. Setting `batch_size` by hand without also setting the
LR multiplier desynchronises the two.

> **Correction:** at [55:09]–[55:22] the instructor describes `batch_size` as *"in how many batches we
> are passing the data in, like in one batch we are passing 16 rows"*, and adds that it is *"irrelevant
> as of now because I just have 10 rows."* The first half is a description of what the number
> *is*; the second half is the right instinct; but two things are missing that make the number
> actually matter:
>
> **(1) `batch_size` is a trainer hyperparameter coupled to the learning rate, not a data-loading
> instruction.** "How many rows per batch" is the *implementation*; the *effect* is on the number of
> optimiser steps and the variance of each gradient estimate, which is why the provider changes the
> learning rate when you change the batch size (§7.5). Setting it manually is a commitment to tune the
> other two around it.
>
> **(2) On a 10-row dataset the server has no choice.** The automatic rule is ~0.2% of examples
> (0.02 rows here) floored at 1, so it resolves to `batch_size=1` — which is confirmed directly by
> the first notebook's job object (§4.7.3). The instructor's explicit `16` on a 10-row dataset is
> therefore a request the trainer cannot honour as written: there are not 16 rows in a batch, there is
> one batch of 10. **The value is accepted, echoed back in `job.hyperparameters`, and is not
> meaningful.** That gap between "what I sent" and "what happened" is exactly why §4.7.3 says to read
> the resolved hyperparameters back.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 The primary decision: hosted or not

| Situation | Hosted FT? | Instead use | Why |
|---|---|---|---|
| You need an endpoint today and have no GPU budget | **Yes** | — | Zero infrastructure; training runs on the provider's hardware |
| Dataset < 1,000 rows on a **supported base model** | **Yes** | — | Exactly the regime where hosted is cheapest and fastest |
| You need the **weights** for on-prem / air-gapped serving | **No** | Open-weight (CS-13) | Hosted gives you no artefact, ever |
| You need a **quantised** or **merged** deployment | **No** | Open-weight (CS-13, CS-23) | There is no quantisation control and nothing to merge |
| You need **control over the learning rate, scheduler, or loss** | **No** | Open-weight (CS-13) | Three hyperparameters, one of them hidden behind a multiplier |
| You need **RLHF/PPO/GRPO** with your own reward model | **No** | Open-weight (CS-24, CS-26) | RFT takes a grader, not a reward model |
| The bottleneck is **knowledge**, not behaviour | **No** | RAG (CS-05) | Fine-tuning teaches form, not facts |
| You only need **guaranteed JSON** | **No** | Structured outputs (§4.12) | Deterministic, free, and exact |
| You are on a **supported model** and cost at scale is fine | **Yes** | — | Operational simplicity is a real feature |
| You are on a **sunset** base model | **No** | Migrate, or open-weight | See the wind-down timeline (§16.8) |
| You need **vision** fine-tuning | **Yes** | — | One of the few hosted-only capabilities with no easy open-weight equal at the same effort |
| You need to train **larger than ~70B** and serve it | **No** | Hosted inference without FT, or accept the cost | You cannot host it yourself at sane cost |

### 8.2 STOP conditions — signals you are about to waste a week

```
STOP if any of these is true:

  [ ] You cannot state, in one sentence, the behaviour change you are buying.
      ("Better answers" is not a behaviour change. "Always returns {"reply","escalate"}
       and never promises a delivery date" is.)

  [ ] You have fewer than ~50 examples AND no budget to generate more.
      With 10 rows you will produce a template, not a model (§4.9.3).

  [ ] The base model already does the task acceptably at temperature 0 with a good
      system prompt. You are about to pay for a behaviour you already have.

  [ ] The failure is a schema/format problem. Use response_format (§4.12).

  [ ] The failure is a knowledge problem. Use RAG (CS-05). Fine-tuning on facts
      produces a confident hallucinator.

  [ ] There is no held-out set and no plan to build one. You cannot evaluate the
      result, so you cannot ship it or defend it (§12).

  [ ] The base model is on the sunset list and you are planning a multi-year system
      on it (§16.8).

  [ ] Your data contains PII and no scrubbing step exists (§4.13).

  [ ] Your inference volume is high enough that the 2x markup dominates: check
      P_f < P_b/2 - 2O (§4.6.5) before you start, not after.

  [ ] Somebody has said "we'll figure out the eval later."
```

### 8.3 The one-question test

If you only remember one thing from this section:

> **"Am I buying a *behaviour*, a *shape*, or a *fact*?"**
>
> - **Behaviour** → fine-tune. (Tone, refusal boundary, verbosity, format *style*, domain vocabulary.)
> - **Shape** → structured outputs / tool calling. (Never fine-tune for this alone.)
> - **Fact** → RAG. (Fine-tuning cannot reliably install facts and reliably makes hallucination worse.)

The video is entirely about the first answer, and it never distinguishes the three. That distinction is
the difference between a project that returns its cost and one that produces a model nobody can
explain the purpose of.

---

## 9. Pros · Cons · Limitations · Silent Failure Modes

Four separate categories. The distinction between the last two matters and is routinely blurred:
a **limitation** is something you know you cannot do; a **silent failure mode** is something you
believe you did.

### 9.1 Pros — what you genuinely get

| # | Pro | Evidence / mechanism | Who cares |
|---|---|---|---|
| 1 | **Zero infrastructure** | Training runs on the provider's hardware [22:04]–[22:14]. A Colab **CPU** runtime is sufficient — the video's own environment is CPU-only. | Teams with no GPU budget |
| 2 | **No serving stack to build** | The output is an HTTPS endpoint, not a `.safetensors` file. No vLLM, no autoscaling, no CUDA. | Small teams; anyone shipping this week |
| 3 | **Cost is metered and small at demo scale** | The video's job: **$0.003024** (§6.5). A real 1,000-row job on a nano model: ~$0.30. | Anyone who needs a number before an approval meeting |
| 4 | **Pay-per-token, no idle cost** | There is no GPU sitting idle at 3am. You pay for training tokens once and inference tokens as used. | Finance |
| 5 | **The base models are strong and current** | You inherit the provider's research. A `gpt-4.1` fine-tune starts from a model you could not train from scratch at any price. | Quality ceiling |
| 6 | **Three hyperparameters means three ways to be wrong** | Compare CS-13's config surface: LoRA rank, alpha, dropout, target modules, quantisation, scheduler, warmup, gradient checkpointing. | Everyone; this is underrated |
| 7 | **Data-quality discipline transfers** | The validator, the token accounting, the holdout split (§5) are the same skills you need for open-weight work. Nothing here is wasted if you migrate. | Career |
| 8 | **Vision fine-tuning is a real hosted-only capability** | No open-weight equivalent matches the effort-to-result ratio. | Multimodal teams |
| 9 | **DPO and RFT are one flag away** | The same platform offers preference and grader-based training without a new toolchain. | Alignment work |
| 10 | **The provider handles the abuse-monitoring/compliance wrapper** | Data-sharing defaults are off; retention windows are documented (§4.13). | Compliance reviewers |

### 9.2 Cons — the real costs

| # | Con | Magnitude | Mitigation |
|---|---|---|---|
| 1 | **2× inference markup** | Every production call costs twice the base model's price | Compute the break-even **before** starting (§4.6.5); shorten the prompt |
| 2 | **Inference dominates lifetime cost** | 0.07% train / 99.93% inference in the §6.5 worked budget | Treat the training bill as a rounding error and optimise the prompt |
| 3 | **You cannot leave** | No weights, no export, no portability. Migrating means re-training from scratch somewhere else | Distil your own outputs into an open model *before* you need to (§4.11) |
| 4 | **You are on the vendor's deprecation clock** | The 2026 wind-down (new FT jobs restricted) and base-model sunsets are not under your control | Detect the base ID in your `ft:` string and alert on it (§6.7) |
| 5 | **The learning rate is hidden** | `learning_rate_multiplier` is a multiplier on an undisclosed base | Treat it as ordinal; move to open-weight if you need cardinal control |
| 6 | **No evaluation is provided beyond loss curves** | No benchmark, no regression suite, no quality score | Build §12's harness; it is not optional |
| 7 | **Automatic epoch policy optimises cost, not quality** | 10 rows → 10 epochs; 200k rows → 1 epoch (§4.8) | Set `n_epochs` explicitly |
| 8 | **Dataset minimums are a floor, not a target** | 10 examples is the API's minimum; the useful floor is ~50–100 with real diversity | Generate synthetic data if you are short (CS-09) |
| 9 | **Billing is not fully transparent in advance** | The auto-policy token count is a server-side computation; `trained_tokens` is the only ground truth | Budget with a margin; reconcile against `trained_tokens` after the first job |
| 10 | **Your data leaves your control** | Uploaded files persist until you delete them; prompts sit in abuse-monitoring logs for 30 days | Scrub pre-upload; delete on success (§4.13) |
| 11 | **Prompt templates become hard requirements** | The system prompt you trained with must be present at inference, byte-identical, or quality degrades | Version the system prompt with the model ID |
| 12 | **The platform is actively de-emphasising FT** | The stated reason for the wind-down is that newer base models follow instructions well enough that prompting is cheaper and faster (§16.8) | Bet on it *now* only where it clearly wins; keep the RAG/structured-outputs alternatives live |

### 9.3 Hard limitations — things that are simply impossible

| # | Limitation | Why it is structural | The workaround, and its cost |
|---|---|---|---|
| 1 | No weight download | The model is a provider-side artefact addressed by an opaque ID | Distillation into an open student (§4.11), at the cost of the capability ceiling |
| 2 | No merging | Nothing to merge | None |
| 3 | No quantisation control | You do not control the serving stack | None for the hosted endpoint; the distilled student is quantisable |
| 4 | No local/offline serving | Requires the provider's API | Distil and serve the student (CS-09) |
| 5 | No custom loss or objective | You get SFT, DPO, or grader-RL | Open-weight training (CS-24, CS-26) |
| 6 | No architectural changes | You cannot touch layers, attention, or the tokenizer | None |
| 7 | No training-data introspection after the fact | You cannot ask which examples influenced a behaviour | Influence functions are an open-weight technique |
| 8 | No exact reproducibility guarantee | `seed` is best-effort | None; treat runs as samples |
| 9 | No fine-tuning of models you cannot fine-tune | Only the listed base models are eligible | Check the list; it churns (§4.4) |
| 10 | No versioning of the fine-tuned model yourself | You cannot snapshot, tag, or roll back your own FT model | Keep a config registry that maps job ID → data version, and treat the model ID as immutable |

### 9.4 Silent failure modes — looks fine, is broken

**This is the most important table in the module.** Every entry here has the same shape: the job
reports `succeeded`, the chat demo looks good, and the system is wrong.

| # | Symptom you observe | What is actually happening | Why it is silent | Detection |
|---|---|---|---|---|
| 1 | Chat replies are fluent and on-brand | The model memorised the **4 unique conversations** in the training set (§4.9.3) | A memorised answer and a generalised answer are indistinguishable on a question that is in the training set | **Paraphrase test**: ask 20 reworded versions of training questions (§4.9.3) |
| 2 | `succeeded`, `fine_tuned_model` populated, everything green | The model learned **nothing** — LR multiplier too low, or all rows truncated | A fine-tune that changes nothing still returns a model, and its outputs still look like the base model's outputs (which are good) | Generate 50 completions from the FT and the base; if they are byte-identical, nothing happened |
| 3 | Out-of-scope questions get answered confidently | All-positive training data drove the refusal boundary to zero | Refusals look like failures, so nobody notices they stopped — until a customer gets a made-up warranty term | Hold out 20 in-scope + 20 out-of-scope; report two rates, not one (§4.9.3) |
| 4 | Loss curve looks healthy; eval is fine | **Train/validation leak**: duplicated rows on both sides of the split | Validation loss comes out *lower* than train loss, which reads as "great generalisation" | Dedupe by content hash **across** the whole dataset before splitting |
| 5 | Format compliance improved dramatically | The **format** was taught and the **content** got worse | Format compliance is trivially measurable and content quality is not, so the dashboard shows a win | Content-level eval on the same holdout; human spot-check on 50 outputs |
| 6 | Prompt cache hit rate was 90%, now it is 0% | The fine-tune changed the system prompt's tokenisation or you changed the prompt | Cost climbs gradually; nothing errors | Alert on cache-hit rate and on cost-per-1k-calls |
| 7 | The fine-tuned model is slower than the base | The FT model may be served on different hardware, and a 2× price often comes with a latency difference | Latency regressions appear in p99 dashboards nobody watches during a launch | A/B both models on the same 200 prompts, compare p50 **and** p99 |
| 8 | The model works in the playground | The playground fills in defaults your code does not (`temperature`, system prompt) | Playground and production diverge by configuration, not by model | Evaluate through the exact production code path, not the console |
| 9 | Job cost is 10× your estimate | `n_epochs` resolved to auto (10 on a small dataset) instead of your intended 3 (§4.7.3) | `succeeded` does not mean "ran as I intended" | Read `job.hyperparameters` back and log `trained_tokens` |
| 10 | Nothing fails; the bill is enormous | The break-even was never computed; the 2× markup at volume exceeded the training cost by orders of magnitude (§4.6.5) | Each individual call is cheap; the aggregate is not | Monthly cost-per-1k-calls against the base-model baseline |
| 11 | The 404 that "goes away if you retry" | The job is still running, or it failed and you never checked (§6.7) | A retry loop converts a hard failure into an intermittent one | Poll to a terminal state; raise on non-`succeeded` |
| 12 | Model quality drifts down over two weeks | **Silent re-deployment**: the FT model's ID is stable but its base was deprecated and remapped, or someone edited the shared system prompt | Neither change touches your code, so nothing in CI fires | Pin the system prompt and the base snapshot; assert on both in your eval job |
| 13 | Validation loss is excellent from step 1 | You passed the **training** file as `validation_file` | Nothing errors — the API does not check that they differ | Assert `training_file != validation_file` in your job-creation wrapper |
| 14 | The fine-tune improved the metric you optimised and broke a different one | SFT is a blunt instrument: it moves everything correlated with the target | You are measuring one metric | Maintain a **regression suite** of 50–200 prompts across all behaviours, not just the target one |

> **Beyond the video — the one detection that catches failures 1, 2, 3, 5, 12 and 14 in a single
> job.** None of them are visible in the loss curves. All of them are visible in a **paired comparison
> against the base model on a frozen prompt set**:
>
> ```python
> import json, hashlib
> from collections import Counter
>
> base_id, ft_id = "gpt-4.1-mini-2025-04-14", "ft:gpt-4.1-mini-2025-04-14:org:support-v3:AbCd1234"
>
> def run(model, prompts, temperature=0.2):
>     out = []
>     for p in prompts:
>         r = client.chat.completions.create(
>             model=model, temperature=temperature, seed=42, max_tokens=512,
>             messages=[{"role": "system", "content": SYSTEM_PROMPT},
>                       {"role": "user",   "content": p}])
>         out.append(r.choices[0].message.content)
>     return out
>
> prompts = [json.loads(l)["prompt"] for l in open("frozen_eval_prompts.jsonl", encoding="utf-8")]
> base, ft = run(base_id, prompts), run(ft_id, prompts)
>
> identical = sum(a == b for a, b in zip(base, ft))
> print(f"identical outputs: {identical}/{len(prompts)}")   # >90% => the fine-tune did nothing (failure #2)
>
> # Failure #1/#5: is the FT model reciting training data?
> train_text = " ".join(json.dumps(ex) for ex in training_examples)
> def max_ngram_overlap(text, n=13):
>     toks = text.split()
>     grams = {" ".join(toks[i:i+n]) for i in range(max(0, len(toks)-n+1))}
>     return sum(1 for g in grams if g in train_text)
> print("verbatim 13-gram hits:", [max_ngram_overlap(t) for t in ft])
> ```
>
> `identical outputs` near zero means the model changed. Near 100% means it did not — and that number
> appears **nowhere** in the platform's UI, in the loss curves, or in the video. It is the first thing
> to compute and the cheapest.

---

## 10. Exceptions, Edge Cases & Gotchas

1. **The 10-example minimum is not the useful minimum.** The API accepts 10 rows
   [14:18]–[14:27]; the instructor's own framing is *"minimum you can keep 10 examples… 50 to 100
   example are the good one."* Treat 10 as the API's guard rail and 50–100 as the smallest thing worth
   a job. Below ~50, generate synthetic data (CS-09) rather than shipping a template.

2. **A `data2.jsonl`-style broken file fails at *upload*, not at *job creation*.** The API's
   `validating_files` state rejects structural errors before a job runs. Two consequences: the failure
   is cheap (you lose a round trip, not a training run), and the local validator is a convenience, not
   a safety net — the server would have caught it anyway. Run it locally because round trips are slow,
   not because they are dangerous.

3. **`example_missing_assistant_message` is the error the video demonstrates, and it is the one that
   matters least.** It fires only when a row has *no* assistant turn at all. A far more common and
   more damaging defect — an assistant turn whose content is an empty string `""` — trips
   `missing_content` instead, and the video never shows it.

4. **A single row with `"role": "assistant", "content": null` is valid but trips the copied
   validator.** That is the correct shape for an assistant turn that only calls a tool. See §6.3.

5. **Windows line endings.** A `.jsonl` produced on Windows without `newline="\n"` can carry `\r\n`,
   and some parsers then see a trailing `\r` in the last string of each line. The training usually
   works; the *validator* reports a character mismatch you will spend an hour on. Write with
   `open(path, "w", encoding="utf-8", newline="\n")`.

6. **UTF-8 BOM.** A file saved from Excel or Notepad may start with `﻿`, which makes line 1
   unparseable while lines 2..n are fine. Symptom: `json.decoder.JSONDecodeError` on the first line
   only. Fix: `encoding="utf-8-sig"` on read.

7. **One malformed line kills the whole file's validation, but not the whole file's *content*.**
   `json.loads` raises on the first bad line and the loop stops — so a file with 9,999 good rows and
   one bad row at line 4,000 appears to have 3,999 rows. Catch per line:
   `for i, line in enumerate(f, 1): try: json.loads(line) except: errors.append(i)`.

8. **The `name` field is per-role, and misusing it is a silent quality cost.** `name` on a `user`
   message distinguishes speakers in a multi-user transcript; on a `system` or `assistant` message it
   is far less standard. It also costs `tokens_per_name` per occurrence (§6.4) and, in some
   trainer versions, is dropped entirely. If a behaviour depends on it, verify it survived.

9. **Multi-turn rows are truncated, and the truncation is silent.** The automatic policy's
   `MAX_TOKENS_PER_EXAMPLE` (16,385 in the reference implementation) drops the *oldest* messages of
   an over-long row. A long conversation therefore loses its system prompt first. Symptom: the model
   behaves as though it has no persona on long inputs only (§4.5.5).

10. **The system prompt is part of the model, not part of the request.** Train with a system message
    and you have effectively baked it in: it must be supplied byte-identically at inference, or
    quality drops. Train *without* one and you must never add one. The worst configuration is train
    *with* and serve *without*, which is what happens when the inference code path is written by a
    different team.

11. **`temperature` interacts with a fine-tune more sharply than with a base model.** A fine-tune
    narrows the output distribution; `temperature=1.2` on a fine-tuned model produces repetition and
    incoherence faster than on the base. Use 0–0.3 for extraction, ≤1.0 for chat.

12. **The suffix is capped and silently truncated.** Over-long suffixes lose their tail, and if the
    tail encoded your version number, two models become indistinguishable. Keep the suffix ≤18
    characters and put the full version in `metadata`.

13. **Two jobs with the same suffix and the same base produce colliding-looking IDs.** They are
    distinct models; the hash differs. Do not key anything on the suffix alone.

14. **A cancelled job still costs money.** Tokens billed before cancellation are charged. The first
    notebook's job is `cancelled`; the spend happened. Cancel early if you see a divergence, but do
    not expect a refund.

15. **`trained_tokens` can be lower than your estimate, and that is not an error.** It reflects the
    *post-truncation*, post-split count. A large gap means truncation was aggressive and your long
    examples were silently shortened.

16. **A base-model deprecation breaks a fine-tuned model's inference path even though the FT model
    has its own ID.** The FT model is an adapter on a base snapshot; when the snapshot is retired the
    FT model follows. The video's `gpt-4o-2024-08-06` job is on exactly this track (§4.4.2).

17. **The playground and the API are not the same client.** The playground injects a default system
    prompt and default parameters. A model that "works in the playground" may fail through your code
    and vice versa. Always evaluate through your own code path.

18. **Fine-tuning does not fix a retrieval problem, and the failure is worse than the original
    problem.** A base model that lacks a fact will say it does not know. A *fine-tuned* model that
    lacks a fact has been trained to produce confident in-domain answers, so it will invent one in
    your house style. Hallucination gets *more* plausible, not less.

19. **The dataset you validate is not always the dataset that trains.** Between your validator and the
    trainer sit the split, the truncation, and (for chat-format train files) the platform's own
    template application. Any of the three can change your data. Reconcile `job.trained_tokens`
    against your prediction; that is your only window into what actually ran.

20. **A fine-tuned model cannot be used as a `validation_file`-trained model's teacher without a
    ToS check.** Distillation is permitted for narrow product tasks and commonly prohibited for
    building competing models (§4.11). This is a legal exception to an engineering rule, and it is
    the one exception in this list you cannot engineer around.

---

## 11. Cost, Compute & Memory

### 11.1 There is no GPU-hour line in a hosted fine-tuning budget

The instructor reaches for GPU framing twice and both times it is wrong, in a way that reveals the
shape of the whole cost model:

> [21:02]–[21:06]: *"$1 uh $150 per hour"* — reading the pricing page.
> [37:45]–[37:53]: *"cost of the GPU on a hourly basis."*
> [47:20]–[47:48]: *"$1.5 for using the GPU… hourly rate."*

> **Correction:** there is no hourly GPU charge for supervised fine-tuning on this platform. Training
> is billed **per 1,000,000 training tokens**, computed as `billed_tokens × n_epochs × price_per_M`.
> The words "GPU" and "hourly" appear nowhere in the SFT billing model. The instructor corrects
> himself once — his later, accurate framing is *"here is the cost for the inferencing"*
> [20:43]–[20:57] and the notebook's cell 67 arithmetic is right (§6.5) — and then **un-corrects**
> himself at [38:01]–[38:16] when he doubts his own correct reading.
>
> **The one place an hourly rate does appear is reinforcement fine-tuning (RFT)**, which *is* billed
> per hour of training compute — and that is the likely source of the confusion. SFT/vision/DPO:
> per token. RFT: per hour. Two different methods, two different billing models, and the instructor
> carries the second one into the first.

This matters for a practical reason: **an hourly mental model makes you unable to budget.** "How many
hours will 5,000 examples take?" has no answer you can compute in advance. "How many tokens will
5,000 examples bill?" is exact, deterministic, and computable in a Python cell before you upload
anything (§6.5).

### 11.2 The four-line cost model

Every hosted fine-tuning project's lifetime cost is the sum of four terms, and they differ by orders
of magnitude:

```text
LIFETIME COST =  T  +  V  +  I  +  M

  T  training      = (train_tokens + val_tokens) × n_epochs × train_price_per_M
  V  validation    = one-off, ~20% of T                      (often ignored; it is real but small)
  I  inference     = calls × (prompt_tokens × in_price + completion_tokens × out_price)
  M  maintenance   = engineer time to re-run, re-eval and migrate  ← almost never budgeted
```

| Term | Scaling | Typical share of a 12-month bill | Lever you control |
|---|---|---|---|
| `T` training | One-off, linear in data × epochs | 0.01–2% | Dataset size, `n_epochs`, base-model tier |
| `V` validation | One-off, ~20% of `T` | <0.5% | Split size |
| `I` inference | Linear in **volume**; the FT model carries a **~2× markup** on both directions | **95–99.9%** | Prompt length (the big one), output length, caching, call volume |
| `M` maintenance | Engineer-days; unbounded | Unbounded | Whether you built an eval harness before you needed one |

**`I` is the bill.** Everything else is a rounding error at any real volume, and the video's framing —
"fine-tuning costs three paise" — is true and completely misleading, because at 100k rows on a
flagship model `T` is $5,040 while `I` is a monthly charge that never stops.

### 11.3 Worked end-to-end budget — "fine-tune a support-triage assistant"

The scenario, with numbers. This is the budget the brief asks for, and every figure is computable
before a single byte is uploaded.

```text
SCENARIO
  Task          Classify + draft a reply for inbound retail support tickets
  Base model    gpt-4.1-mini  (train $3.00/M · in $0.80/M · cached $0.20/M · out $3.20/M, fine-tuned)
                              (base: in $0.40/M · cached $0.10/M · out $1.60/M)
  Training set  5,000 rows, 400 content tokens each (long system prompt + ticket + reply)
  Validation    800 rows, same shape  (16% split — at the top of the recommended band)
  Epochs        3
  Production    200,000 calls/month
  Prompt        900 tokens with the base model's long instruction prompt
                500 tokens with the fine-tuned model (the behaviour is baked in; the
                    prompt shrinks by the amount the fine-tune replaced)
  Output        120 tokens
```

**Step 1 — training tokens (the deterministic part).**

| Line | Arithmetic | Value |
|---|---|---|
| Billed tokens per training row | `400 content + 9 overhead` | 409 |
| Training tokens per epoch | `5,000 × 409` | 2,045,000 |
| Validation tokens per epoch | `800 × 409` | 327,200 |
| Total per epoch | `2,045,000 + 327,200` | 2,372,200 |
| **Total billed training tokens** | `2,372,200 × 3 epochs` | **7,116,600** |
| **Training cost** | `7,116,600 / 1e6 × $3.00` | **$21.35** |

**Step 2 — inference, base model (the baseline you must beat).**

| Line | Arithmetic | Value |
|---|---|---|
| Input cost per call | `900 / 1e6 × $0.40` | $0.000360 |
| Output cost per call | `120 / 1e6 × $1.60` | $0.000192 |
| Cost per call | | **$0.000552** |
| **Monthly** | `200,000 × $0.000552` | **$110.40** |

**Step 3 — inference, fine-tuned model (the 2× markup meets the shorter prompt).**

| Line | Arithmetic | Value |
|---|---|---|
| Input cost per call | `500 / 1e6 × $0.80` (2× base rate) | $0.000400 |
| Output cost per call | `120 / 1e6 × $3.20` (2× base rate) | $0.000384 |
| Cost per call | | **$0.000784** |
| **Monthly** | `200,000 × $0.000784` | **$156.80** |
| **Monthly delta vs base** | `$156.80 − $110.40` | **+$46.40 (worse)** |

**Step 4 — the verdict, and it is not the one the project sponsor expects.**

| Line | Value |
|---|---|
| Training cost (one-off) | $21.35 |
| Inference delta | **+$46.40 / month — the fine-tune is *more* expensive to run** |
| **Payback period** | **Never.** There is no payback; the cost is strictly higher, forever. |
| Break-even check | `P_f < P_b/2 − 2O` → `500 < 900/2 − 2×120` → `500 < 210` → **FALSE** |

```python
# The whole budget in eleven lines. Run this BEFORE you create a job.
def total_cost(train_rows, val_rows, avg_tokens, epochs, train_price,
               calls, base_prompt, ft_prompt, out_tokens,
               base_in, base_out, ft_in, ft_out, months=12):
    billed = avg_tokens + 9
    train  = (train_rows + val_rows) * billed * epochs / 1e6 * train_price
    base_m = calls * (base_prompt/1e6*base_in + out_tokens/1e6*base_out)
    ft_m   = calls * (ft_prompt  /1e6*ft_in   + out_tokens/1e6*ft_out)
    return {"train_one_off": round(train, 2),
            "base_per_month": round(base_m, 2),
            "ft_per_month":   round(ft_m, 2),
            "payback_months": round(train / (base_m - ft_m), 1) if ft_m < base_m else None}

print(total_cost(5_000, 800, 400, 3, 3.00,
                 200_000, 900, 500, 120,
                 0.40, 1.60, 0.80, 3.20))
# {'train_one_off': 21.35, 'base_per_month': 110.4, 'ft_per_month': 156.8, 'payback_months': None}
```

**The honest conclusion, and the decision most teams get wrong.** The fine-tune makes the system
**more expensive to operate** and it never pays back. If the fine-tune is still worth doing, it is
worth doing for a reason that is *not* on this spreadsheet:

- the base model's replies violate the brand voice in ways customers notice;
- the base model promises delivery dates it cannot keep, which is a **liability**, not a cost line;
- the base model's refusals on legitimate requests cost more in support tickets than the delta.

Those are real reasons. **"It will be cheaper" is not, and you can now prove it in eleven lines of
Python before you spend anything.** A fine-tune whose only justification is cost reduction should be
stopped at §8.2.

**Now the version where it *does* pay back.** Suppose the fine-tune replaces a 900-token prompt with a
**180-token** one — a realistic outcome if most of the base prompt was few-shot examples and format
instructions that the fine-tune absorbed:

| Line | Value |
|---|---|
| FT input per call | `180 / 1e6 × $0.80` = $0.000144 |
| FT output per call | $0.000384 |
| FT cost per call | $0.000528 |
| FT monthly | **$105.60** |
| Delta vs base | **−$4.80 / month (better)** |
| Payback | `$21.35 / $4.80` = **4.4 months** |
| Break-even check | `180 < 210` → **TRUE** |

**A $4.80/month saving.** That is what a genuinely successful hosted fine-tune is worth on a 200k-call
monthly volume: a few dollars, realised over four months, on a $21 training bill. Multiply by ten
(2M calls) and it is $48/month against a $21 one-off. **The margins are real and thin.** Anyone who
tells you fine-tuning is a cost-optimisation play at this scale has not done this arithmetic.

### 11.4 Inference dominance, visualised

At 5,000 training rows, the crossover where inference overtakes training happens in the first
**five days** of production traffic at 200k calls/month:

```text
$ cumulative
│                                                   ╱  inference (FT, 500-tok prompt)
│                                                 ╱     $156.80/mo, forever
│                                               ╱
│                                             ╱     ╱  inference (base, 900-tok prompt)
│                                           ╱     ╱     $110.40/mo, forever
│                                         ╱     ╱
│                                       ╱     ╱
│  $21.35 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╱─ ─ ─╱─ ─ ─ ─ ─  training (one-off)
│                                    ╱     ╱
│                                  ╱     ╱
│    0 ────────────────────────╱─────╱──────────────────► time
│         day 1        day 4  day 5   day 8
                             ▲
                             └── the training cost is already the smaller half
```

| Training rows | Training cost (`gpt-4.1-mini`, 3 epochs) | Days of 200k-call/month traffic to exceed it |
|---|---|---|
| 100 | $0.43 | **0.1 days** |
| 1,000 | $4.27 | **1.2 days** |
| **5,000** | **$21.35** | **5.8 days** |
| 20,000 | $85.39 | 23 days |
| 100,000 | $426.96 | 116 days |
| 500,000 | $2,134.80 | 1.6 years |

**Even the largest realistic training set is out-earned by production traffic within two years.** There
is no dataset size at which training cost is the thing to worry about, and there is no dataset size at
which the 2× inference markup goes away.

### 11.5 The open-weight comparison — and the honest crossover

The comparison that matters is not "hosted vs open-weight training cost" (open-weight training is
nearly free for a LoRA). It is **"hosted per-call forever" vs "own a GPU"**, and the crossover is
volume-driven:

| Monthly calls | Hosted base (`gpt-4.1-mini`, 900/120) | Hosted FT (500/120) | Self-host a 3B LoRA on 1× RTX 4090, 24/7 | Cheapest |
|---|---|---|---|---|
| 20,000 | $11.04 | $15.68 | $252 | **Hosted base** |
| 100,000 | $55.20 | $78.40 | $252 | **Hosted base** |
| **500,000** | $276.00 | $392.00 | **$252** | *crossover ≈ 460k calls* |
| 1,000,000 | $552.00 | $784.00 | $252 | **Self-host** |
| 5,000,000 | $2,760.00 | $3,920.00 | $756 (3 GPUs) | **Self-host** |
| 50,000,000 | $27,600.00 | $39,200.00 | $2,520 (10 GPUs) | **Self-host, 10×** |

*Assumptions: a 3B model on one 24 GB card running a modern server, ~2,000–4,000 aggregate tok/s, at
$0.35/GPU-hour, 730 hours/month. A 7B model roughly halves throughput, moving the crossover to ~230k
calls. Serverless/scale-to-zero GPU changes the low end entirely — at 16% average utilisation, a
scale-to-zero provider charged per second can undercut a 24/7 reservation by 5×.*

| Factor | Hosted FT | Open-weight on your own GPU |
|---|---|---|
| Up-front cost | ~$0–$500 | GPU rental or capex; $0 if you use a free notebook |
| Marginal cost per call | **Non-zero, forever, with a 2× markup** | **~$0** (fixed GPU cost, amortised) |
| Break-even volume | — | **~230k–460k calls/month** for a 3B student |
| Time to first endpoint | **Minutes** | Hours to days |
| Ops burden | **None** | Serving stack, autoscaling, monitoring, on-call |
| Weights | **Never yours** | Yours |
| Quality ceiling | The provider's flagship | A model you can actually train and host |
| Data residency | The provider's region | Your region |
| Migration path | **None — you rebuild from zero** | Change the base, re-run |

> **Beyond the video — the asymmetry that decides this, and it is not in either column.** The hosted
> column is a *floor*: your cost per call can never go below the markup, and it scales linearly with
> success, forever. The self-hosted column is a *step function*: a fixed monthly bill that covers the
> first ~500k calls and can be scaled by adding GPUs at linear cost — but you own the ops.
>
> So the real question is not "which is cheaper today". It is **"is my traffic going to grow, and do I
> have the team to run a serving stack?"** If traffic will 10×, the hosted path's cost 10×'s with it
> while the self-hosted path moves in steps. If traffic is flat and small, hosted is unbeatable and
> the ops burden is worth more than the savings.
>
> And the trap underneath both: **you cannot migrate a hosted fine-tune to your own GPU.** The
> decision at 20k calls/month determines your architecture at 5M calls/month, because the only way
> out of the hosted column is to *retrain* — which is where §4.11's distillation pipeline earns its
> place. Build it before you need it, or pay the rebuild cost at the worst possible moment.

### 11.6 RFT's per-hour billing, as the exception that proves the rule

Reinforcement fine-tuning is billed **per hour of training compute**, not per token, and that inverts
the cost model completely:

| Method | Billing unit | Cost predictability | What makes the bill grow |
|---|---|---|---|
| SFT / vision / DPO | Per 1M training tokens | **Exact** — computable in advance, offline | Dataset size × epochs |
| **RFT** | **Per hour of training compute** | **Not computable in advance** — depends on rollout count, grader latency, wall-clock | How long the RL loop runs; a slow grader is a direct cost |

The practical consequence for RFT: **your grader's latency is a line item.** A grader that calls
another model per rollout, for 10,000 rollouts per step across 200 steps, is 2 million grader calls —
and the training clock is running the whole time. Optimise the grader (deterministic Python where
possible; a small model with caching where not) before you optimise anything else.

### 11.7 Memory and compute: what you are not paying for

There is no VRAM formula to compute for hosted fine-tuning, and that is the point. What follows is
what the provider is absorbing on your behalf, and why the hosted price is what it is.

| Resource | What a 7B full fine-tune needs | What the provider runs for your job |
|---|---|---|
| Model weights (bf16) | 14 GB | 14 GB+ |
| Gradients | 14 GB | 14 GB |
| AdamW optimiser states | **56 GB** (2 moments × 4 bytes × params) | 56 GB |
| Activations (batch 4, 2k seq) | ~10–20 GB with checkpointing | similar |
| **Total, full fine-tune** | **~95–105 GB** | Multi-GPU, tensor-parallel |
| **Total, LoRA (r=16)** | **~20–24 GB** on a 24 GB card | — |
| Total, **API fine-tune** | **0 GB** | All of the above, and it is why the price has a markup |

**One line closes this section.** A single RTX 4090 (24 GB) can LoRA-fine-tune a 7B model. It cannot
full-fine-tune one. The hosted API's entire value proposition at small scale is that you do not have
to know the difference — and its entire limitation at large scale is that knowing the difference
would have let you do it yourself for a fraction of the marginal cost.

---

## 12. Evaluation — How To Know It Worked

The video's entire evaluation is one chat turn against a question that appears in the training set
[1:03:02]–[1:03:16]. Everything below exists because that is not enough.

### 12.1 The four metrics the platform gives you, and what each one hides

| Metric | Source | What it actually measures | How it lies |
|---|---|---|---|
| `train_loss` | result CSV | Cross-entropy on rows the model is fitting | Falls monotonically on almost any dataset; on 10 rows it reaches ~0 and means nothing |
| `valid_loss` | result CSV | Cross-entropy on the holdout | **The only honest learning signal the platform provides** — and it is `None` if you did not pass `validation_file` |
| `train_mean_token_accuracy` | result CSV | % of teacher-forced tokens predicted correctly | On a memorised 10-row set this reaches ~100% while generalisation is zero |
| `valid_mean_token_accuracy` | result CSV | Same, on holdout | A more readable proxy than loss for *format* learning; blind to content quality |
| *(nothing)* | — | Correctness, tone, refusal behaviour, format compliance, latency, cost | **All unmeasured.** The platform has no concept of any of these. |

> **Beyond the video — the single most useful reframing in this section.** Loss is a *token-level
> language-modelling* metric. Your task is not language modelling. "Did it emit the right token 99% of
> the time" and "did it produce a correct, on-brand, safe answer" are different questions, and a model
> can maximise the first while failing the second completely — because the tokens it gets wrong are
> exactly the load-bearing ones (a warranty duration, a price, an `escalate: true`).
>
> **The rule: loss selects between checkpoints. Task metrics decide whether to ship.** Never ship on a
> loss curve.

### 12.2 The five-layer evaluation stack

Each layer costs more than the one above it and catches what the one above it cannot.

| Layer | What it is | Cost | Catches | Misses |
|---|---|---|---|---|
| **0. Loss** | Platform result CSV | Free | Nothing useful for shipping; gross divergence | Everything task-related |
| **1. Structural** | Does the output parse and match the schema? | Free, instant, deterministic | Format regressions — the highest-frequency production failure | Whether the content is any good |
| **2. Exact / reference** | String or JSON-field equality against a golden answer, where the task has one | Hours to build the goldens | Classification, extraction, routing — anywhere correctness is checkable | Free-text quality |
| **3. Model-as-judge** | A stronger model scores outputs against a rubric, pairwise vs the base model | Cents per hundred | Tone, helpfulness, faithfulness, refusal quality — at scale | Systematic judge bias; correlation with humans is imperfect |
| **4. Human** | 50–200 blind pairwise comparisons | Engineer-days | Everything the layers above missed; the ground truth | Cost; it does not scale, so it does not run in CI |

**The practical protocol, in order:**

```text
1. Freeze a test set. 100-500 prompts, held out, never trained on, never used to pick epochs.
2. Run BOTH models (base and fine-tuned) on the SAME prompts, same temperature, same seed.
3. Layer 1: structural pass rate for both.   <-- catches the failures that page you at 3am
4. Layer 2: task accuracy where checkable.   <-- the number you report
5. Layer 3: pairwise judge, FT vs base.      <-- the number that correlates with user perception
6. Layer 4: 50 blind human pairs.            <-- the number that settles arguments
7. Ship only if the FT model wins or ties on layers 1-2 and wins on 3-4.
   A tie on quality with a 2x inference markup is a LOSS.
```

### 12.3 A minimal, runnable eval harness

```python
"""
eval_ft.py — paired evaluation of a fine-tuned model against its base.
Every layer is a function returning a number. Run it in CI on every prompt or model change.
"""
import json, os, statistics, hashlib
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()

BASE_MODEL = "gpt-4.1-mini-2025-04-14"
FT_MODEL   = os.environ["FT_MODEL_ID"]          # set by the training pipeline

# ---------------------------------------------------------------- layer 1: structural
REQUIRED_KEYS = {"reply", "escalate"}
VALID_ESCALATE = {True, False}

def structural_ok(text: str) -> bool:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not REQUIRED_KEYS.issubset(obj):
        return False
    return isinstance(obj["escalate"], bool) and isinstance(obj["reply"], str)

# ---------------------------------------------------------------- layer 2: exact / reference
def exact_ok(text: str, golden: dict) -> bool:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return obj.get("escalate") == golden["escalate"]      # the checkable half of the task

# ---------------------------------------------------------------- layer 3: model-as-judge
JUDGE_RUBRIC = """You are comparing two customer-support replies to the same ticket.
Answer with exactly one token: A, B, or TIE.
Judge on: (1) factual accuracy, (2) brand voice, (3) whether it over-promises,
(4) whether it asks for the right next step. Do not reward length."""

def judge(prompt: str, a: str, b: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4.1", temperature=0.0, max_tokens=4,
        messages=[{"role": "system", "content": JUDGE_RUBRIC},
                  {"role": "user", "content": f"TICKET:\n{prompt}\n\nREPLY A:\n{a}\n\nREPLY B:\n{b}"}])
    return r.choices[0].message.content.strip().upper()

def balanced_judge(prompt: str, a: str, b: str) -> str:
    """Run both orders and only count agreement — kills position bias."""
    f, r = judge(prompt, a, b), judge(prompt, b, a)
    if f == "TIE" or r == "TIE": return "TIE"
    if f == "A" and r == "B": return "A"
    if f == "B" and r == "A": return "B"
    return "TIE"                                    # inconsistent => no preference

# ---------------------------------------------------------------- runner
def run(model: str, cases: list[dict]) -> list[str]:
    out = []
    for c in cases:
        r = client.chat.completions.create(
            model=model, temperature=0.2, seed=42, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": c["prompt"]}])
        out.append(r.choices[0].message.content)
    return out

def main():
    cases = [json.loads(l) for l in open("test_set.jsonl", encoding="utf-8") if l.strip()]
    base, ft = run(BASE_MODEL, cases), run(FT_MODEL, cases)

    report = {
        "n": len(cases),
        "structural_base": sum(structural_ok(t) for t in base) / len(cases),
        "structural_ft":   sum(structural_ok(t) for t in ft)   / len(cases),
        "exact_base": sum(exact_ok(t, c["golden"]) for t, c in zip(base, cases)) / len(cases),
        "exact_ft":   sum(exact_ok(t, c["golden"]) for t, c in zip(ft,   cases)) / len(cases),
        "identical_outputs": sum(a == b for a, b in zip(base, ft)) / len(cases),
    }

    wins = {"A": 0, "B": 0, "TIE": 0}               # A = base, B = fine-tuned
    for c, b, f in zip(cases, base, ft):
        wins[balanced_judge(c["prompt"], b, f)] += 1
    report["judge_base_win"] = wins["A"] / len(cases)
    report["judge_ft_win"]   = wins["B"] / len(cases)
    report["judge_tie"]      = wins["TIE"] / len(cases)

    print(json.dumps(report, indent=2))
    # Gate: fail CI if the fine-tune lost structural ground or produced identical output.
    assert report["structural_ft"] >= report["structural_base"], "format regression"
    assert report["identical_outputs"] < 0.90, "fine-tune did not change behaviour (silent failure #2)"

if __name__ == "__main__":
    main()
```

### 12.4 How each evaluation approach lies to you

| Approach | The lie | Why | The fix |
|---|---|---|---|
| **Train loss** | "It's learning" | It always falls | Ignore it; look at `valid_loss` |
| **Validation loss** | "It generalises" | True only if the split is leak-free and the task is language-modelling-like | Dedupe across the split; add task metrics |
| **Your own chat test** | "It works" | You asked a question from the training set (§4.9.3) | Paraphrase test; frozen test set |
| **Exact match** | "92% accurate" | Goldens written by the same person who wrote the training data share its blind spots | Source goldens from a different author, or from production |
| **Model-as-judge** | "The FT wins 68%" | Judges have **position bias**, **length bias**, and a preference for fluent prose over correct prose | Balanced two-order judging; length-controlled prompts; spot-check against humans |
| **Model-as-judge (same family)** | "The FT wins" | If the judge shares a base model with the contestant, it may prefer its own family's style | Use a different family as judge, or a rubric with explicit criteria |
| **Human eval** | "The FT wins 8/10" | 10 comparisons cannot distinguish 60% from 80%; raters know which is which | ≥50 pairs, blind, reported with a confidence interval |
| **A/B in production** | "The FT wins" | Traffic is not random; novelty effects fade; downstream metrics lag | Interleave, hold out a permanent control, wait two weeks |
| **Latency p50** | "It's faster" | p99 is what your users experience and what your SLO is written against | Compare p99 on the same prompt set |
| **Cost per call** | "It's cheaper" | Ignores the training bill, the eval bill, and the engineer-weeks | Total cost of ownership over 12 months (§11.3) |
| **Public benchmark** | "It scores higher" | Your task is not on the benchmark, and a fine-tune can raise a benchmark while lowering your task metric | Never ship on a public benchmark delta |

> **Beyond the video — the number that decides whether a fine-tune was worth running, in one line.**
> Compute `identical_outputs` from the harness above **first**, before any judge or human evaluation.
>
> | `identical_outputs` | Reading |
> |---|---|
> | ~1.00 | **The fine-tune did nothing.** LR multiplier too low, rows truncated, or the wrong file was uploaded. Stop and investigate before spending on eval. |
> | 0.5–0.9 | The behaviour changed in a limited, probably *format-related* way. Expected for a small dataset. |
> | 0.1–0.5 | Substantive behaviour change. Now you need layers 2–4 to find out whether it is the *right* change. |
> | < 0.1 | Near-total behaviour replacement. On a small dataset this is usually **memorisation**, not generalisation (§4.9.3). |
>
> The video's demo would land near 0.0 on its own training question and near 1.0 on the base model's
> general behaviour — which is precisely the signature of a model that has learned four answers. That
> number is one API call per prompt away and the video never computes it.

---

## 13. Comparison Tables

### 13.1 Hosted fine-tuning vs the four nearest alternatives

| Dimension | **Hosted FT** (OpenAI) | **Open-weight FT** (CS-13) | **Prompt engineering** | **RAG** (CS-05) | **Structured outputs** (§4.12) |
|---|---|---|---|---|---|
| **Deliverable** | An endpoint ID | Weights you own | A prompt string | An index + a prompt | A schema |
| Weights | **Never** | **Yours** | n/a | n/a | n/a |
| Cost — one-off | $0.003–$500 (token-billed) | GPU-hours; $0–$300 typical for LoRA | $0 | Index build; $10–$1,000+ | $0 |
| Cost — per call | Base price **× 2** | Your GPU, ~$0 marginal | Base price (longer prompt) | Base price + retrieval latency | Base price |
| Time to first result | **Minutes** | Hours to days | **Minutes** | Days | **Minutes** |
| Data needed | 50–1,000 rows typical; 10 minimum | Same, plus a GPU | 0 | A corpus | 0 |
| Teaches **behaviour** | Yes | Yes | Partially (fragilely) | No | No |
| Teaches **facts** | Poorly, and dangerously | Poorly | No | **Yes** | No |
| Guarantees **shape** | No | No | No | No | **Yes, exactly** |
| Ops burden | **None** | Serving stack + on-call | None | Vector DB + freshness | None |
| Control of the loss / LR | **Three knobs** | All of them | n/a | n/a | n/a |
| Quantisation / merging | **Impossible** | Yes | n/a | n/a | n/a |
| Privacy | Vendor-hosted; retention windows | Your infrastructure | Vendor API | Vendor API (+ your index) | Vendor API |
| Vendor lock-in | **Total** | Low | Low | Medium | Low |
| Break-even vs prompting | **~230k–460k calls/month** (or a behaviour prompting cannot reach) | Same | — | — | — |
| **Use when** | No GPUs, a supported base model, a behaviour change, small data | You need the artefact, the control, or the volume | The behaviour is already close | The problem is knowledge | The problem is shape |

### 13.2 The hosted fine-tuning methods, head to head

| | **SFT** | **Vision FT** | **DPO** | **RFT** |
|---|---|---|---|---|
| Teaches | A new behaviour from demonstrations | The same, over images | A **preference** between two behaviours | A **verifiable skill** |
| Data shape | `messages` | `messages` + `image_url` | `messages` + `chosen` + `rejected` | prompts + a grader |
| Human effort per row | 1 completion | 1 completion + an image | **2 completions, ranked** | A grader you must write |
| Billing | Per 1M tokens | Per 1M tokens | Per 1M tokens | **Per hour** |
| Typical `n_epochs` | 1–3 | 1–3 | **1** | n/a |
| Signature failure | Memorisation on small data | Same | Likelihood collapse | Reward hacking the grader |
| Prerequisite | — | — | **An SFT model** (§4.10.1) | Automatically verifiable answers |
| The video covers it | **Yes** | Mentioned | Mentioned, not shown | Mentioned (as "RLHF"), not shown |

### 13.3 `gpt-4.1` family tiering for fine-tuning

| | `gpt-4.1-nano` | `gpt-4.1-mini` | `gpt-4.1` |
|---|---|---|---|
| Training (per 1M) | **$1.50** | **~$3.00** | **~$25.00** |
| FT input (per 1M) | $0.20 | $0.80 | ~$6.00 |
| FT cached input | $0.05 | $0.20 | ~$1.50 |
| FT output (per 1M) | $0.80 | $3.20 | ~$24.00 |
| Markup over its own base | 2× (all three prices) | 2× | 2× |
| 10,000-row job, 3 epochs (672 tok/row) | **$3.02** | $6.05 | $50.40 |
| Best for | Classification, routing, extraction, high volume | The default choice; balanced quality and cost | Rarely worth a fine-tune — prompt the base model instead |
| Relative risk of wasting money | Low | Low | **High** — $25/M training on a model you could prompt |

**Read the last row seriously.** Fine-tuning the flagship is a decision that is almost never
justified: a $50 training run is cheap, but the **inference** markup on a flagship is brutal at
volume, and the flagship's base instruction-following is good enough that most of what you would
teach it, it already does. The tiering rule that holds up in practice:

> **Fine-tune the smallest model that can do the task when prompted. Prompt the largest model you
> can afford when you cannot.** The gap between them is where money is wasted.

### 13.4 Hosted fine-tuning vs the same task solved four other ways

The scenario is the video's own: a support agent that always replies in a fixed house style.

| Approach | Effort | Behaviour match | Cost / 1k calls | Weights | Verdict |
|---|---|---|---|---|---|
| Base + long few-shot prompt | 1 hour | 70% | $0.55 | No | Start here. Measure. |
| Base + structured outputs | 2 hours | Shape 100%, style 70% | $0.55 | No | Fixes the schema half exactly |
| **Hosted SFT (this module)** | 1 day | **90%** | **$0.78** | **No** | Justified only by the missing 20% of *behaviour* |
| Open-weight LoRA (CS-13) | 3 days | 85% | ~$0 marginal above GPU | **Yes** | Justified by volume or by needing the weights |
| RAG over the support corpus (CS-05) | 2 days | 70% + *facts* | $0.62 | No | Justified when the problem is *knowledge* |
| **Hosted SFT + structured outputs** | 1 day | **Shape 100%, style 90%** | $0.78 | No | **The right combination when you need both** |

---

## 14. Debugging Playbook

### 14.1 Data and file errors

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | `KeyError: 'data_type'` | Dict key read before assignment — the video's own bug at cell 14 | Traceback points at the first `+= 1` | `format_errors = collections.defaultdict(int)` (§6.3) |
| 2 | Job fails in `validating_files` | Any of the seven structural errors | Run the local validator; the API's error message is terse | Fix the file; re-upload. Never re-run the same file ID |
| 3 | `messages is not a list` / `TypeError` mid-validation | `messages` is a string, dict or int | `print(type(ex.get("messages")))` on the offender | Guard with `isinstance(messages, list)` (§6.3) |
| 4 | `unrecognized_role` on valid `tool` rows | The copied allow-list is the completion-era one | Compare your `VALID_ROLES` constant to the current API docs | Widen to `("system","developer","user","assistant","tool")` (§6.3) |
| 5 | `missing_content` on correct tool-call rows | An assistant turn with `content: null` and `tool_calls` populated | Inspect the row: does it have `tool_calls`? | Allow null content when `tool_calls` or `function_call` is present |
| 6 | `JSONDecodeError` on line 1 only | UTF-8 BOM from an editor | `open(p,'rb').read(3) == b'\xef\xbb\xbf'` | Read with `encoding="utf-8-sig"` |
| 7 | Validator reports fewer rows than the file has | An unhandled exception on one line aborts the comprehension | Count lines vs parsed rows | Parse per line in a `try/except`, collect line numbers (§10.7) |
| 8 | A stray `\r` in the last field of each row | Windows line endings | `repr(line[-5:])` | Write with `newline="\n"` |
| 9 | `file.bytes` ≠ local file size | Truncated upload, or you uploaded the wrong path | Compare `info.bytes` to `os.path.getsize(path)` | Re-upload; assert the sizes match in your pipeline |
| 10 | "File already exists / cannot be used" | `purpose` was not `"fine-tune"` | `client.files.retrieve(id).purpose` | Re-upload with the right purpose |

### 14.2 Training failures

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 11 | `loss: NaN` | LR multiplier far too high, or one pathological example | Look at `train_loss` in the result CSV: it climbs then NaNs | Reset to `learning_rate_multiplier=1.0`; find and trim the longest examples |
| 12 | Loss flat from step 1 | LR multiplier too low; data has no signal; wrong file | Compare loss to the base model's expected loss on this data | Raise the multiplier modestly; audit the dataset (§6.2) |
| 13 | **Loss goes to ~0** | Memorisation — small dataset, too many epochs | `valid_loss` rising while `train_loss` → 0 | Cut `n_epochs`; add data; this is the video's `gpt-3.5` job exactly |
| 14 | `valid_loss` **below** `train_loss` | **Leak** — duplicate rows across the split | Content-hash every row; look for hashes appearing on both sides | Dedupe **before** splitting; re-run |
| 15 | `valid_loss` noisy and unreadable | Validation file too small (a 10-row validation set is 10 samples) | `len(val_rows)` | Use ≥100 validation rows, or read `full_valid_loss` (end-of-epoch) instead of step-level |
| 16 | Loss spikes mid-run then recovers | One outlier example, or LR slightly high | Find the longest rows in the dataset | Trim the top 0.1% by length; keep the LR |
| 17 | `trained_tokens` far below your estimate | Truncation dropped messages (§4.5.5) | Compare longest rows against the `MAX_TOKENS_PER_EXAMPLE` constant | Restructure long rows into multiple examples, or accept and re-budget |
| 18 | Job `succeeded` but `fine_tuned_model` is `null` | A short poll window; the model ID populates slightly after the status flips | Re-`retrieve` after 30s | Poll on `fine_tuned_model is not None`, not on `status` alone |
| 19 | Cost 3–10× your estimate | Auto `n_epochs` resolved high on a small dataset (§4.8) | `job.hyperparameters` — always | Set `n_epochs` explicitly, every time |
| 20 | Job stuck in `queued` for hours | Provider-side capacity; not something you control | `list_events` for queue messages | Wait; or retry at a different time; do not cancel and restart blindly — a restart rejoins the same queue and loses the progress |

### 14.3 Inference and serving failures

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 21 | **`404 model_not_found`** on a `ft:` ID | Job not `succeeded`, or cancelled, or the ID is stale — the video's first-notebook error (§6.7) | `client.fine_tuning.jobs.retrieve(job_id).status` | None — the model will never exist. Fix the job and re-run |
| 22 | Model responds, but as the **base** model | You passed the base model ID, or the fine-tune learned nothing | `identical_outputs` from §12.3 | Check the ID string starts with `ft:`; if it does, the run was a no-op (§9.4 #2) |
| 23 | Quality drops in production but not in tests | The system prompt differs from training, or `temperature`/`penalties` differ | Diff the exact request payloads | Version the system prompt with the model ID; assert it in CI |
| 24 | Output is truncated mid-JSON | `max_tokens` too low | Count completion tokens on failures | Raise `max_tokens`; add `response_format` so the decoder cannot open an object it cannot close |
| 25 | Cost per call doubled overnight | Prompt grew, cache hit rate fell, or the FT model replaced the base | Cost dashboard split by model ID | Alert on cost-per-1k-calls; pin the prompt |
| 26 | Latency regressed after the switch | FT models are not always served on the same hardware | p50 **and** p99 for both models on the same prompts | Accept it, or revert; latency is a product metric |
| 27 | Model starts answering out-of-scope questions | All-positive training data erased the refusal boundary (§9.4 #3) | Out-of-scope holdout rate | Add refusal examples to the training set and re-run — it is a data fix, not a prompt fix |
| 28 | Two teams' requests behave differently | Two different system prompts, one trained identity | Log the full request per call | One system prompt per model ID, owned in a single place |
| 29 | Repeated phrases / degeneration | `temperature` too high for a fine-tuned model | Sample 20 outputs at the production temperature | Drop to ≤1.0; fine-tunes narrow the distribution |
| 30 | Fine-tuned model is **worse** than the base at everything | Overfit, or the training data contained corrections of the base model's *good* answers | Paired eval (§12.3); inspect 20 training rows for quality | Roll back to the base model ID — this is a one-line config change, which is the one advantage of the hosted setup |

### 14.4 The loss-curve decision tree

```
loss behaviour?
│
├── train ↓, valid ↓ then flat  ────────────► HEALTHY. Stop here. More epochs will hurt.
│
├── train ↓, valid ↑  ──────────────────────► OVERFIT.
│                                             → cut n_epochs to the step where valid was lowest
│                                             → add data (or generate it)
│                                             → increase diversity, not volume
│
├── train ~flat, valid ~flat from step 1 ───► NOT LEARNING.
│                                             → raise learning_rate_multiplier 2x, once
│                                             → confirm the file is the one you think it is
│                                             → confirm the base model can do the task when prompted
│
├── train → ~0, valid → ~0 ─────────────────► MEMORISED. (Typical < 100 rows.)
│                                             → you have a template, not a model
│                                             → the only fix is more distinct data
│
├── train ↓, then SPIKES, then recovers ────► LR slightly high, or one bad example.
│                                             → find the longest rows; trim the top 0.1%
│
├── train → NaN ────────────────────────────► LR far too high / bad row.
│                                             → multiplier back to 1.0; re-run
│
└── valid BELOW train ──────────────────────► LEAK, not a miracle.
                                              → dedupe across the split; re-run
```

---

## 15. Applied Case Studies

### 15.1 The one the video actually ran — and what it really demonstrates

| Field | Value |
|---|---|
| **Scale** | 10 rows, 4,239 bytes, 1 system prompt, **$0.003024** of training |
| **Goal as stated** | A smartphone-support agent with a fixed house voice |
| **Goal as achieved** | A model that reproduces the training answers in the training style |
| **Config** | `gpt-4o-2024-08-06`, `n_epochs=3`, `batch_size=16` (not honoured), `learning_rate_multiplier=1.0`, `validation_file=None` |
| **Result** | `succeeded`; a chat turn against a question that is in the training set returns a fluent, on-brand answer |
| **What went wrong first** | The `gpt-3.5-turbo` job with an inline 4-unique-row dataset and auto hyperparameters resolved to **10 epochs**; it was cancelled, and the chat test 404'd (§6.7) |
| **The honest evaluation** | Never performed. `validation_file` was never set, the base model was never run on the same prompt, and the test question was in the training data |
| **What it is genuinely good for** | Proving the *mechanics* — upload, validate, count, budget, create, chat — end to end for three paise |
| **What it does not prove** | Anything about the model's quality, generalisation, or cost-effectiveness |

**The transferable lesson:** the mechanics are the easy half and they are what the video teaches well.
The hard half — a held-out set, a paired baseline, a break-even calculation — is absent, and it is
absent from almost every fine-tuning tutorial you will read. Build that half yourself, and build it
before the first job, because the artefacts it needs (a frozen test set, the base model's baseline
numbers) cannot be reconstructed afterwards.

### 15.2 Retail support triage — the cost-justified case

| Field | Value |
|---|---|
| **Situation** | 2M support tickets/month. An 1,800-token prompt (12 few-shot examples + policy + format rules) on `gpt-4.1-mini`, ~40M input tokens and 4M output tokens monthly |
| **Constraint** | Inference cost is the second-largest line in the department's budget |
| **Why FT** | The 12 few-shot examples are the *entire* prompt, and they are exactly what a fine-tune encodes |
| **Data** | 8,000 labelled historical tickets, 12% held out, 400-token average, deduped by content hash |
| **Config** | `gpt-4.1-mini`, `n_epochs=2`, `batch_size="auto"`, `learning_rate_multiplier="auto"`, `validation_file` set, `seed=42`, `metadata={"data_version":"2026-08-11"}` |
| **Cost** | Training: `(7,040 + 960) × 409 × 2 / 1e6 × $3.00` = **$19.62**. Inference after: prompt drops 1,800 → 240 tokens |
| **Result** | Base: `(1800×$0.40 + 200×$1.60)/1e6 × 2M` = **$2,080/mo**. FT: `(240×$0.80 + 200×$3.20)/1e6 × 2M` = **$1,664/mo**. Saves **$416/mo**, payback in **1.4 days** |
| **What went wrong first** | The first run used the **raw** ticket text including customer names and order IDs. A privacy review caught it before upload. The second run scrubbed with a regex pass plus a manual review of 200 samples |
| **Second thing that went wrong** | Round 1 used `validation_file=None`; the run looked fine and shipped. Two weeks later the escalation rate rose — the model had learned to *always* escalate, because the historical data was biased toward tickets that got escalated. Only a paired eval against the base model caught it |
| **The lesson** | `P_f < P_b/2 − 2O` → `240 < 900 − 400 = 500` → **TRUE**, and this is the rare case where the arithmetic and the behaviour both say yes. Note that the win came from the prompt collapse (1,800 → 240), not from the fine-tuning being cheap |

### 15.3 The fine-tune that should not have been run

| Field | Value |
|---|---|
| **Situation** | A fintech team needs an extraction endpoint returning a fixed JSON schema. The base model returns valid JSON "about 88% of the time" |
| **Proposed solution** | Fine-tune on 5,000 hand-labelled extraction examples |
| **Cost of the proposal** | $3,000 of annotation, 3 weeks of engineer time, a $15 training run, and a permanent 2× inference markup |
| **The actual solution** | `response_format={"type":"json_schema","strict":true}` with the exact schema |
| **Cost** | **$0** and one afternoon |
| **Result** | Structural validity went from 88% to **100%** — not 99%, 100%, because constrained decoding makes invalid output *unrepresentable* |
| **What the team actually needed fine-tuning for** | Nothing, at first. Three months later they fine-tuned on **800** examples to fix a *field-value* problem the schema could not express: the model was putting the wrong value in a `category` enum, because the enum labels were ambiguous. The schema guaranteed the shape; only the data could teach the mapping |
| **The lesson** | Separate the two failures. **Shape failures** are a schema problem. **Content failures inside a valid shape** are a fine-tuning problem. This team almost spent $3,000 and a 2× markup to fix a shape failure |

### 15.4 Regulated-industry migration — the wind-down case

| Field | Value |
|---|---|
| **Situation** | A healthcare-adjacent team runs a `gpt-4o-2024-08-06` fine-tune in production. The base snapshot is deprecated and the platform is restricting new fine-tuning jobs (§16.8) |
| **Constraint** | No GPU infrastructure, no ML engineers, 18 months of accumulated training data and prompt tuning, and a compliance requirement that prompts and completions stay in-region |
| **Why it is hard** | There is no migration. The FT model's weights are unreachable, the base is sunsetting, and the data is the only asset that survives |
| **The path taken** | **(1)** Freeze the current model's behaviour as **120,000 teacher completions** over the production prompt distribution (§4.11). **(2)** Filter: dedupe, drop malformed, drop the shortest 5%. **(3)** LoRA-fine-tune a 7B open model (CS-13) on 4×A100 for ~6 GPU-hours. **(4)** Serve it on 2×L40S in-region. **(5)** Run the §12.3 harness: **91% agreement** with the hosted model on the frozen test set |
| **Cost** | Distillation generation: ~$340 in API calls at 120k samples. Training: ~$15 of GPU time. Serving: ~$1,100/month, flat |
| **What went wrong first** | The first distillation run sampled the teacher at **temperature 0**, producing 120,000 near-identical completions. The student learned a lookup table and scored **62%** agreement. Raising the teacher's temperature to 0.9 and adding a dedupe pass took it to 91% |
| **The lesson** | **Start the distillation before the deadline.** The 18-month data asset was worth far more than the model, and generating the teacher samples took a week of wall-clock. A team that begins this migration the week the deprecation notice lands ships a worse model under time pressure. Instrument the base-model ID inside your `ft:` string (§6.7) and alert on announcements |

### 15.5 Preference tuning for a refusal boundary — the DPO case

| Field | Value |
|---|---|
| **Situation** | A consumer app's assistant gives medical-adjacent advice it should decline. SFT runs make it *worse*: adding refusal demonstrations makes it refuse everything, including legitimate questions |
| **Why SFT fails here** | Refusal is a **boundary**, and SFT teaches it as a **category**. Every refusal example makes "refuse" more likely in general, including for in-scope questions |
| **The fix** | DPO. 1,400 preference pairs where `chosen` is the correct response (helpful for in-scope, declining for out-of-scope) and `rejected` is the SFT model's own wrong-side response |
| **Why the pairs are cheap** | The rejected half is **sampled from the model being fixed** — no human writing required, only human *ranking*, and often only human *verification* of an automatic ranker |
| **Config** | `method={"type":"dpo","dpo":{"hyperparameters":{"n_epochs":1,"beta":"auto"}}}`, on top of the SFT checkpoint |
| **Result** | Out-of-scope refusal rate 41% → **94%**; in-scope helpfulness *unchanged* at 97% (the SFT approach had dropped it to 68% while raising refusals to 88%) |
| **What went wrong first** | A `n_epochs=3` DPO run **collapsed**: `rewards/chosen` and `rewards/rejected` both fell, and the model became terse and unhelpful on everything. DPO almost always wants **1 epoch** |
| **The lesson** | SFT teaches *what to do*; DPO teaches *which of two things to prefer*. A boundary is a relative judgement, which is why DPO wins here and SFT cannot. And the ordering matters: DPO was run **on top of** an SFT checkpoint, never on the base model (§4.10.1) |

---

## 16. Production Considerations

### 16.1 Serving

```python
# The production wrapper: everything that must be true, asserted at import time.
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL_ID      = os.environ["FT_MODEL_ID"]           # "ft:gpt-4.1-mini-2025-04-14:org:support-v3:AbCd1234"
SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()
PROMPT_SHA    = "3f9a1c2e"                          # recorded at training time; asserted below

def call(user_turn: str, *, timeout: float = 20.0) -> str:
    assert MODEL_ID.startswith("ft:"), f"not a fine-tuned model: {MODEL_ID}"
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8] == PROMPT_SHA, \
        "system prompt drift: this model was trained with a different prompt"
    r = client.chat.completions.with_raw_response.create(
        model=MODEL_ID, temperature=0.2, max_tokens=512, timeout=timeout,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": user_turn}],
    )
    log_request(r.headers.get("x-request-id"), MODEL_ID, PROMPT_SHA)
    return r.parse().choices[0].message.content
```

| Concern | Hosted answer | What you must build |
|---|---|---|
| Uptime | The provider's SLA | A fallback to the base model on error/timeout |
| Autoscaling | Automatic | Nothing |
| Rate limits | Per-org token-per-minute and request-per-minute ceilings | A client-side limiter with retry-and-backoff on 429 |
| Latency | Provider-dependent | Your own timeout + fallback; never let a slow call block a user |
| Cost control | A dashboard | Per-team/per-feature accounting; alert on cost-per-1k-calls |
| Prompt identity | **Nothing** | The `PROMPT_SHA` assertion above — the single highest-value line in this section |

### 16.2 Versioning and rollback

The hosted setup's one genuine operational advantage: **rollback is a string change.**

```python
# A model registry keyed by the values you control.
REGISTRY = {
    "support-v1": {"id": "ft:gpt-4.1-mini-2025-04-14:org:support-v1:Aa11", "sha": "1a2b3c4d"},
    "support-v2": {"id": "ft:gpt-4.1-mini-2025-04-14:org:support-v2:Bb22", "sha": "3f9a1c2e"},
    "support-v3": {"id": "ft:gpt-4.1-mini-2025-04-14:org:support-v3:Cc33", "sha": "5e6f7a8b"},
}
ACTIVE = os.environ.get("ACTIVE_MODEL", "support-v2")      # env-var rollback, no deploy required
```

| What to version | Where | Why |
|---|---|---|
| The model ID | Config, not code | Rollback without a deploy |
| **The system prompt, hashed** | Alongside the model ID | Train/serve skew is the #1 silent regression (§9.4 #12) |
| The training data version | `job.metadata` + your data lake | Answers "what trained this?" |
| The job ID | Your registry | Links the model to its hyperparameters and metrics |
| The eval scores at ship time | Your registry | The baseline the new version must beat |
| The base snapshot ID (parsed from the `ft:` string) | A monitored alert | Fires when a sunset is announced (§6.7) |

**What you cannot version:** the model itself. There is no `ft:...:v2` you can pin to a byte-identical
artefact, and no way to restore a deleted fine-tuned model. If a model is in production, **do not
delete it**, even after its successor ships — deletion is irreversible and it is the only rollback
target you have.

### 16.3 Monitoring and drift

| Signal | How to measure | Alert threshold | What it catches |
|---|---|---|---|
| **Cost per 1,000 calls** | Provider dashboard, split by model ID | >10% week-over-week | Prompt growth, cache collapse, a runaway retry loop |
| **Cache hit rate** | Cached-input tokens ÷ total input tokens | <50% of the training-time baseline | Prompt drift; a stable prefix that stopped being stable |
| **Structural failure rate** | Parse failures ÷ total calls, from your own logs | >0.5% | Format regression — the highest-severity production failure |
| **p99 latency** | Your own instrumentation | >2× the launch baseline | Provider-side changes; your own prompt growth |
| **Output length distribution** | Mean and p95 completion tokens | A shift >20% | The model getting terser or more verbose after an unannounced change |
| **Refusal rate** | Count of a known refusal signature | Any move >5 points | A boundary that drifted |
| **Escalation rate** (task-specific) | Your downstream business metric | Any move >10% | The thing that actually matters; the earliest real signal |
| **`identical_outputs` vs the base** | Weekly eval job (§12.3) | Any move toward 1.0 | A silent revert to base-model behaviour |

> **Beyond the video — data drift on the *input* side is the failure nobody instruments.** Everything
> in the table above watches the model. Nothing watches whether production prompts still look like the
> training prompts. Compute a cheap embedding for every production prompt, keep a reference centroid
> from the training set, and alert when the cosine distance crosses a threshold. A fine-tune trained
> on 2024 support tickets will degrade on 2026 tickets even if the model is byte-identical and healthy
> — and the only warning you get is a slow decline in the business metric, three months late.

### 16.4 Regression tests

```yaml
# .github/workflows/ft-regression.yml — the eval harness as a merge gate.
name: ft-regression
on:
  pull_request:
    paths: ["prompts/**", "model_registry.py", "training/**"]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python eval_ft.py --test-set test_set.jsonl --fail-under-structural 0.99
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          FT_MODEL_ID:    ${{ vars.FT_MODEL_ID }}
```

| Test | Gate | Why this one |
|---|---|---|
| Structural pass rate | **100%** | A schema break is a production outage |
| Task accuracy vs recorded baseline | ≥ baseline − 1 point | Catches a regression before a user does |
| Out-of-scope refusal rate | within ±5 points of baseline | Catches the boundary collapse of §9.4 #3 |
| `identical_outputs` vs base | **< 0.90** | Catches a no-op fine-tune |
| Cost per 1k calls | ≤ 1.1× baseline | Catches prompt bloat |
| p99 latency | ≤ 1.5× baseline | Catches a hardware or provider change |
| Verbatim 13-gram recall | 0 hits | Catches memorisation **and** a privacy finding |

### 16.5 Guardrails

Fine-tuning does not remove the need for input and output guardrails, and it slightly increases it —
a fine-tuned model is *more* confident in-domain, which means a hallucination arrives in a more
persuasive voice.

| Guardrail | Position | Implementation |
|---|---|---|
| Input: PII detection and redaction | Before the call | Regex + a classifier; log the redaction, not the value |
| Input: prompt-injection screening | Before the call | The fine-tune does **not** make the model injection-resistant |
| Input: scope classifier | Before the call | A cheap model decides "in-domain?" and routes; this is often the better home for a boundary than the fine-tune |
| Output: **schema validation** | After the call | `response_format` makes this a check, not a hope (§4.12) |
| Output: policy checks | After the call | Banned claims (delivery dates, prices, medical advice) via regex *and* a classifier |
| Output: groundedness | After the call | If RAG is in the loop, verify claims against retrieved context |
| Fallback path | On any failure | The base model with the long prompt — which is also your permanent control group |

### 16.6 Cost governance

| Control | Mechanism |
|---|---|
| Per-team attribution | Separate API keys or projects per consuming team; the provider's usage view splits by key |
| Budget alerting | Daily cost query against a threshold; alert at 50/80/100% |
| Prompt budget in CI | A unit test that tokenises the system prompt and fails if it exceeds N tokens — prompt growth is the #1 cost regression |
| Cache-first design | Stable prefix first, variable content last; measure the hit rate as an SLO |
| Output caps | `max_tokens` set on every call path; an unbounded output is an unbounded bill |
| Training-job approval | A job above $X requires a signed-off budget (§11.3's function output attached) |
| Retention | A scheduled job that deletes training files N days after job completion |

### 16.7 The compliance angle

| Requirement | Hosted posture | Action |
|---|---|---|
| Data residency | Region-pinned endpoints where offered | Pin the region in config and assert it |
| Zero data retention | Available on eligible endpoints with an approved org configuration | Request it before the first regulated workload |
| No training on your data | **Default** — API data is not used to train base models | Verify no one enabled data sharing at the org level (§4.13) |
| Right to erasure | Training data deletes on request; **a fine-tuned model cannot be un-trained** | The only remedy is deleting the model and retraining. Design for this: keep the data versioned and the pipeline re-runnable |
| Audit trail | Job objects, file records, usage logs | Export `metadata` and the model registry into your own system of record — the provider's console is not an audit log |
| Model provenance | The `ft:` ID encodes the base snapshot | Parse and store it; it is your deprecation early-warning system |
| **Automated decision-making disclosures** | Your obligation, not the provider's | If the fine-tune informs decisions about people, it is in scope for the same review a rules engine would be |

> **Beyond the video — the one compliance property unique to fine-tuning, and the one most often
> missed.** In a RAG system, deleting a document removes it from the answer path: the index is
> rebuilt, the fact is gone. **In a fine-tuned model, deletion is not possible.** A fact, a name, a
> pattern absorbed into the weights cannot be located, cannot be excised, and — because the weights
> are not even yours — cannot be surgically edited. The only remedy is to delete the model and retrain
> on the corrected data, which means **your training pipeline must be cheap and re-runnable for the
> lifetime of the model.** If a full retrain takes three weeks of an engineer's attention, you do not
> have a deletion capability; you have a deletion *intention*. This is the strongest argument for
> keeping the pipeline scripted end to end and for keeping the dataset small, curated and versioned.

### 16.8 The platform wind-down — and what to do about it

This module dates faster than any other in the handbook, and the platform's own direction of travel is
the largest single fact in it. The published position, as of this writing:

| Date | Change | Affects |
|---|---|---|
| **2026-05-07** | No **new** fine-tuning job can be created by an organisation that has **never** run one | New adopters — i.e. exactly the reader of this module |
| **2026-07-02** | No new job for an organisation with no fine-tuned-model **inference** traffic in the trailing 60 days | Lapsed users; a dormant fine-tune cannot be revived |
| **2027-01-06** | Existing active customers can no longer create **new** jobs | Everyone still on the platform |
| **Until base-model deprecation** | Existing fine-tuned models **continue to serve** | Production systems already running |

**The stated reason is the most important sentence in the module, and it is not a billing change:**
newer base models follow instructions and output formats well enough that **prompting is cheaper and
faster than fine-tuning**. The vendor is telling you that the capability gap that justified hosted
fine-tuning has narrowed — which is precisely the argument §4.12 and §13.1 make from first
principles, arriving from the other direction.

> **Beyond the video — re-verify every date on this page before you rely on it.** The dates above are
> the published position at the time of writing; vendor roadmaps move, and a policy that is announced
> as a wind-down is occasionally reversed or extended. The *shape* of the situation is what to plan
> around: **hosted fine-tuning is being de-emphasised for new users, while existing fine-tuned models
> keep serving until their base is retired.** Check the current status page and the model deprecation
> list before you commit a multi-year system to this path, and treat any specific date here as a
> prompt to go and look rather than as a fact.

**The migration decision tree, for a team that has a hosted fine-tune in production today:**

```
Do you have a hosted fine-tune serving traffic?
│
├── NO, and you are considering one
│     │
│     ├── Can you reach the behaviour with prompting + structured outputs?  ──► DO THAT. Stop.
│     │
│     ├── Is the volume above ~450k calls/month?  ──► Open-weight from day one (CS-13).
│     │                                              You will need the weights eventually.
│     │
│     └── Otherwise ──► Hosted is fine for now, BUT:
│                        (a) set validation_file, seed and metadata
│                        (b) script the pipeline end to end
│                        (c) start capturing production prompts for a future distillation
│
├── YES, and your base model is NOT on the sunset list
│     │
│     ├── Keep serving. Do not rebuild for its own sake.
│     └── START the distillation capture now (§4.11). It costs ~$340 and it is your
│         only exit. Doing it under deadline pressure produces a worse model (§15.4).
│
└── YES, and your base model IS on the sunset list
      │
      ├── 1. Freeze the behaviour: 100k+ teacher completions over the production
      │      prompt distribution, temperature 0.7-1.0, deduped.
      ├── 2. Fine-tune an open student (CS-13) and measure agreement on the frozen
      │      test set. Target ≥90% before you switch.
      ├── 3. Serve in-region; keep the hosted model live as the control group.
      └── 4. Cut over on the eval numbers, not on the deadline.
```

**The three things that survive a platform migration, and the one that does not:**

| Asset | Survives? | Note |
|---|---|---|
| Your **dataset** | **Yes** | The only durable asset. Version it, curate it, keep the licence clean |
| Your **prompt and schema contracts** | Yes | They port to any model |
| Your **eval harness** | Yes | It is what lets you switch models with evidence |
| Your **fine-tuned model** | **No** | Not the weights, not the behaviour, not the ID |

**The strategic conclusion.** The lesson of this module is not "use hosted fine-tuning" or "avoid it".
It is that **the hosted path is a rental, and the rent can be changed.** Everything valuable that you
build — a validated dataset, a frozen eval set, a scripted pipeline, a documented prompt contract —
is portable to any other model. The model is not. Spend your effort accordingly: **the data and the
evaluation are the assets; the endpoint is a commodity.** A team that internalises that is fine-tuning
on a hosted platform with the exit already built, and it makes no difference to them when the wind-down
completes.

---

## 17. Common Misconceptions

**1. "Fine-tuning is cheap."**
The *training* is cheap — the video's job costs **$0.003024** (§6.5). The *product* is not. The
fine-tuned model carries a **~2× inference markup on every call, forever**, and at any real volume
that is 95–99.9% of the lifetime bill (§11.2). The video's own arithmetic is correct and its framing
is the most expensive thing in it.

**2. "Fine-tuning teaches the model new facts."**
It teaches **form**: format, tone, refusal boundaries, domain vocabulary, output conventions. A model
fine-tuned on documents about your product does not reliably contain your product's facts — and it
becomes **more** confident in-domain, so its hallucinations get more persuasive. Facts belong in RAG
(CS-05). This is the single most costly misconception in the field.

**3. "I need to fine-tune to get reliable JSON."**
You need **structured outputs**. Constrained decoding makes malformed output unrepresentable — 100%,
not 99%. It costs nothing and takes an afternoon. §4.12, §15.3.

**4. "More epochs means a better model."**
More epochs means a model that has memorised your dataset. On the video's `gpt-3.5` job the automatic
policy chose **10 epochs on 4 unique conversations** — 25 exposures each (§4.9.3). The correct reading
of a falling training loss is "the model is fitting", not "the model is learning".

**5. "10 examples is enough to fine-tune."**
10 is the API's **minimum**, not a viable dataset. The instructor's own framing is *"minimum you can
keep 10 examples… 50 to 100 example are the good one."* With 10 rows you produce a template. With 10
rows and 10 epochs you produce a template that has memorised itself.

**6. "`batch_size` is how many rows I pass in each batch."**
It is a trainer hyperparameter coupled to the learning rate. Changing it changes the number of
optimiser steps and the variance of each gradient, which is why the provider's automatic LR policy
depends on it (§7.5). On a 10-row dataset it is not honoured at all.

**7. "`learning_rate_multiplier: 1.0` means zero learning rate."**
It means "use the provider's default **base** learning rate". The actual rate is
`base_lr × multiplier`, where `base_lr` is not exposed (§4.7.2). The knob is ordinal — "same as
default", "more", "less" — never cardinal.

**8. "The platform charges me for GPU hours."**
Supervised, vision and DPO fine-tuning are billed **per 1,000,000 training tokens**. Only
**reinforcement fine-tuning** is billed per hour, and the instructor appears to carry that model over
by mistake (§11.1).

**9. "If the job says `succeeded`, the model is good."**
`succeeded` means the process exited cleanly. A fine-tune with a bad LR multiplier, a truncated
dataset, or a duplicated one still returns `succeeded` and still returns a model — one that is
byte-identical to the base model, or worse (§9.4 #2, #4).

**10. "The loss curve tells me whether it worked."**
Loss is a token-level language-modelling metric. It has no opinion about whether your answers are
correct, on-brand, or safe (§12.1). And without a `validation_file` you do not even get the holdout
half of it — which is the configuration both of the video's notebooks use.

**11. "I own the fine-tuned model."**
You own an **ID string**. Not weights, not a checkpoint, not a portable artefact. You cannot download
it, merge it, quantise it, serve it locally, or move it to another provider (§4.1, §9.3).

**12. "I can migrate my fine-tune to another provider later."**
There is no migration. There is a **rebuild**, and the only asset that crosses the boundary is your
dataset. Plan the exit before you need it: capture production prompts and distil your own outputs into
an open student while the endpoint still works (§4.11, §15.4).

**13. "Fine-tuning is the same skill on every platform."**
The **data contract** is portable — `{"messages": [...]}` is a de-facto standard. The **job API**, the
**hyperparameters**, the **pricing** and the **model line-up** are not, and all four churn monthly
(§4.4). Budget for re-reading the docs at every vendor upgrade.

**14. "A bigger base model will give me a better fine-tune."**
Rarely worth it. The flagship costs ~8× more to train and, more importantly, carries a much higher
inference markup at volume — while its base instruction-following is already good enough that most
of what you would teach it, it does when prompted (§13.3). Fine-tune the **smallest** model that can
do the task when prompted.

**15. "Fine-tuning will reduce my inference cost."**
Only if the fine-tune lets you **delete prompt tokens**, and only if the reduction beats the 2×
markup: `P_f < P_b/2 − 2O` (§4.6.5). In the §11.3 worked budget the fine-tune made the system
**$46.40/month more expensive**, permanently, with no payback. Fine-tuning is a behaviour purchase
that sometimes comes with a cost benefit, never a cost-optimisation play.

**16. "The evaluation is the last step."**
The evaluation is the **first** step, because the frozen test set and the base model's baseline
cannot be reconstructed after the fact — and because a fine-tune that cannot be evaluated cannot be
shipped or defended (§5, §12). The video performs exactly one evaluation, on a question from its own
training set.

**17. "I should delete the training file once the job succeeds."**
Yes — and this is the rare case where the correct action is the counter-intuitive one. Uploaded files
persist until explicitly deleted, and a fine-tuning file is a permanent copy of your data at a third
party (§4.13). Delete on success; keep your own copy in versioned storage, which is the asset that
matters anyway.

**18. "A fine-tuned model can be un-trained."**
It cannot. A fact or a name absorbed into weights cannot be located or excised — and since the
weights are not yours, it cannot even be surgically edited. The only remedy is to **delete the model
and retrain**, which means your pipeline must stay cheap and re-runnable for the model's whole life
(§16.7). For a regulated dataset this is the requirement that determines the architecture.

---

## 18. Key Takeaways

1. **You are buying a behaviour, not a model.** Hosted fine-tuning returns an endpoint ID — no
   weights, no merge, no quantisation control, no local serving, no artefact of any kind.

2. **Three hyperparameters, one of which is a lie.** `n_epochs`, `batch_size`,
   `learning_rate_multiplier` — and the third is a multiplier on a base rate you cannot see, so treat
   it as ordinal and never reason about its absolute value.

3. **Training is billed per million tokens, exactly.** `billed_tokens × n_epochs × price_per_M`, with
   `billed_tokens` including ~3 tokens of per-message overhead plus 3 for the reply. The video's job:
   `672 × 3 = 2,016 tokens × $1.50/M = $0.003024 ≈ ₹0.275`.

4. **The auto-epoch policy is a cost bound, not a quality recommendation.** Below 100 examples it
   holds observations constant at ~100 (`n_epochs = 100 // N`), which means **10 rows → 10 epochs**.
   Above 25,000 examples it drops to 1. Always set `n_epochs` explicitly; always read
   `job.hyperparameters` back.

5. **Inference is the bill.** 95–99.9% of lifetime cost, carrying a **2× markup over the base model**
   in both directions, forever. Budget it before you train, not after.

6. **The break-even, in one inequality:** `P_f < P_b/2 − 2O`. Fine-tuning wins on cost only if the
   fine-tuned prompt (`P_f`) is less than half the base prompt (`P_b`) minus twice the output tokens
   (`O`). Run this before you create a job.

7. **"The model won't return valid JSON" is a schema problem.** Structured outputs make malformed
   output unrepresentable, cost nothing, and take an afternoon. Fine-tune for tone and content
   *inside* the schema, never for the schema itself.

8. **Loss selects between checkpoints; task metrics decide whether to ship.** And if
   `identical_outputs` between the fine-tuned and base models is near 1.0, the fine-tune did nothing —
   compute that number first, before any judge or human evaluation.

9. **On a 10-row dataset you produce a template, not a model.** 10 is the API's minimum; 50–100
   distinct, high-quality rows is the smallest thing worth a job. The video's own best run used four
   unique conversations.

10. **The system prompt is part of the model.** Train with one and you must serve with it,
    byte-identically, or quality silently degrades. Hash it, store it beside the model ID, and assert
    it at import time.

11. **The data contract is portable; nothing else is.** `{"messages": [...]}` works everywhere. The
    job API, the hyperparameters, the pricing and the model line-up churn monthly — and the platform
    is actively de-emphasising fine-tuning because prompting newer base models is cheaper and faster.

12. **The exit is distillation, and you should build it before you need it.** Sample your fine-tuned
    model's outputs over the production prompt distribution, at temperature ~0.9, dedupe, and train an
    open student. ~$340 of API calls buys you a model you own (§15.4).

13. **A fine-tuned model cannot be un-trained.** Deletion is the only remedy for a data-removal
    request, which means the pipeline has to stay re-runnable for the model's whole life. This is the
    constraint that determines the architecture in any regulated setting.

14. **The data and the evaluation are the assets; the endpoint is a commodity.** Everything that
    survives a platform migration is something you built outside the platform. Spend the effort there.

15. **A fine-tuning run without a paired baseline is a purchase, not an experiment.** Before the first
    job, write down the number the fine-tuned model must beat, measured on a frozen held-out set with
    the same prompts against the base model. If you cannot state it in advance, you cannot evaluate
    the result.

---

## 19. Self-Check Questions

**Q1.** What does hosted fine-tuning return, and name four things an open-weight fine-tune returns
that it does not.

**Q2.** Write the JSONL schema for one training row with a system, user and assistant message. Name
the seven structural checks a validator should perform.

**Q3.** A dataset of 672 billed tokens per epoch is trained for 3 epochs on a model priced at $1.50
per million training tokens. What is the training cost, and what would it be if you had used
`count_total_tokens` (562) instead of the billing counter?

**Q4.** You have 8,000 training examples and pass `n_epochs="auto"`. How many epochs does the
documented policy resolve, and what does it resolve for 10 examples?

**Q5.** What is the difference between `batch_size` and `learning_rate_multiplier`, and why does the
second one's absolute value mean nothing without knowing the provider's base rate?

**Q6.** State the break-even inequality for hosted fine-tuning. With a base prompt of 900 tokens and
200 output tokens, what is the largest fine-tuned prompt that still wins?

**Q7.** A fine-tune reports `succeeded`, and in a chat test it answers a training question fluently.
Name three things that could still be wrong and the specific test that would detect each.

**Q8.** When is DPO the right method and when is it the wrong one? What must exist before you run it?

**Q9.** Your colleague wants to fine-tune to guarantee valid JSON. What do you recommend instead, and
what is the one case where fine-tuning is still needed?

**Q10.** Your fine-tuned model is in production and the base snapshot is announced for deprecation.
Describe the migration, why it must begin immediately, and what the one asset that survives is.

<details>
<summary><b>Answers</b></summary>

**A1.** An endpoint — a model ID string of the form `ft:<base>:<org>:<suffix>:<hash>` that you call
over HTTPS. It does not return: (1) downloadable weights, (2) anything you can merge, (3) any control
over quantisation or the serving stack, (4) any ability to serve locally or offline. It also returns
no control over the loss, the scheduler, or the architecture.

**A2.**
```json
{"messages": [{"role": "system",    "content": "You are a support agent."},
              {"role": "user",      "content": "What warranty does it come with?"},
              {"role": "assistant", "content": "One year, covering manufacturer defects."}]}
```
The seven checks: `data_type` (each line is a dict), `missing_messages_list` (the `messages` key
exists and is non-empty), `message_missing_key` (every message has `role` and `content`),
`message_unrecognized_key` (no unexpected keys), `unrecognized_role` (`role` is in the allow-list),
`missing_content` (`content` is a non-empty string, or a `function_call`/`tool_calls` is present), and
`example_missing_assistant_message` (the row contains at least one assistant turn).

**A3.** `672 × 3 = 2,016` tokens; `2,016 / 1e6 × $1.50 = $0.003024`. With the content-only counter you
would have computed `562 × 3 = 1,686` tokens and `$0.002529` — a **16.4% underestimate**, because
the 110 tokens of per-message overhead (3 per message + 3 for the reply, across 21 messages) are real
billed tokens.

**A4.** 8,000 examples: `8,000 × 3 = 24,000`, which is between `MIN_TARGET_EXAMPLES` (100) and
`MAX_TARGET_EXAMPLES` (25,000), so `n_epochs = 3`. 10 examples: `10 × 3 = 30 < 100`, so
`n_epochs = min(25, 100 // 10) = 10`. The policy holds the number of *example observations* at ~100
in the low regime and ~25,000 in the high regime.

**A5.** `batch_size` is the number of examples per optimiser step: it sets how many steps a run
contains and how noisy each gradient estimate is. `learning_rate_multiplier` scales a **base learning
rate chosen by the provider and not exposed to you**, so its absolute value is meaningless across
providers and across versions of the same provider's stack — `1.0` means "use the default", not "zero"
and not "1e-5". Because the two interact (a larger batch usually wants a larger LR), changing one
without the other desynchronises the run, which is why `"auto"` is the right answer for `batch_size`
on nearly every job.

**A6.** `P_f < P_b/2 − 2O`. With `P_b = 900` and `O = 200`: `P_f < 450 − 400 = 50`. The fine-tuned
prompt must be under **50 tokens** — which is essentially impossible for a task that needs
instructions, so fine-tuning on cost grounds **loses** in this configuration. (This is why the prompt
collapse has to come from deleting few-shot examples, not from shortening instructions.)

**A7.** (1) **Memorisation** — the model learned the training answers rather than the task; detect
with a paraphrase test on reworded versions of the training questions. (2) **Out-of-scope collapse** —
all-positive training data erased the refusal boundary; detect by holding out 20 in-scope and 20
out-of-scope prompts and reporting both rates. (3) **A no-op** — the fine-tune changed nothing and you
are talking to the base model wearing a new ID; detect by computing `identical_outputs` between the
fine-tuned and base models over 50 prompts (near 1.0 means nothing happened). A fourth: a **leaked
split**, detectable by `valid_loss` coming out *below* `train_loss`.

**A8.** DPO is right when the model **can already perform the task** and you need it to *prefer* one
of two behaviours — tone, refusal boundaries, verbosity, safety. It is wrong when the model cannot
produce the behaviour at all, because then there is nothing to rank and the implicit reward is
meaningless. A **supervised fine-tune must exist first**; DPO runs on top of an SFT checkpoint, never
on a base model. It also wants `n_epochs=1` — DPO overfits preference pairs extremely fast, and the
signature failure is likelihood collapse, where the probabilities of both the chosen and the rejected
responses fall together.

**A9.** Use **structured outputs** — `response_format={"type":"json_schema","strict":true}` —
which makes invalid output structurally unreachable (100%, not 99%) at zero training cost. You still
need fine-tuning when the *values inside* a valid shape are wrong: the shape is guaranteed by the
schema, but the mapping from input to a correct field value is a behaviour that only data can teach
(§15.3).

**A10.** Freeze the behaviour by sampling the existing model over the production prompt distribution
(100k+ completions, temperature 0.7–1.0, deduped), then fine-tune an open-weight student on those
samples and validate it against a frozen test set before cutting over (§4.11, §15.4). It must begin
immediately because the teacher is only callable while the base model still serves, and generating
the samples is a week of wall-clock that cannot be compressed at the end. The one asset that survives
is the **dataset** — with the eval harness and the prompt/schema contract alongside it. The model
itself does not survive: not the weights, not the ID, not the behaviour.

</details>

---

## 20. Cross-References

| Relationship | Module | Why |
|---|---|---|
| **Contrasts with** | **CS-13 — Instruction Fine-Tuning** | The same SFT loss, trained on your own GPU with full control. Every capability lost in §4.1 is a capability it has |
| Contrasts with | **CS-03 — Fine-Tuning Framework Landscape** | Positions hosted APIs at the extreme "less code" end of the control/complexity axis |
| Builds on | **CS-05 — RAG** | The alternative for knowledge problems; the two compose (fine-tune the form, retrieve the facts) |
| Builds on | **CS-09 — Synthetic Data & Distillation** | The pipeline that recovers an artefact from an endpoint-only model (§4.11) |
| Needed by | **CS-23 — LoRA / QLoRA** | The technique that makes the open-weight alternative cheap enough to be the migration target |
| Needed by | **CS-25 — DPO** | Hosted DPO is one flag away; the loss and the theory live there |
| Needed by | **CS-24 — RLHF / PPO** | Contrast with RFT's grader-based, per-hour-billed model (§4.10) |
| Contrasts with | **CS-26 — GRPO** | Verifiable-reward training without a learned reward model; closer to RFT than RLHF is |
| Related | **CS-17 — Axolotl** | The config-file-driven open-weight trainer; the migration destination |
| Related | **CS-19 — Gemini / Vertex fine-tuning** | The same three-layer abstraction (data contract → job → endpoint) on another vendor; the same churn problem |
| Related | **CS-20 — Small Language Models** | The distillation target; the reason a 1–8B student is viable |
| Related | **CS-21 — Multimodal fine-tuning** | Where hosted vision fine-tuning is compared against open-weight VLMs |
| Related | **CS-22 — Embedding fine-tuning** | A different objective (contrastive) on a different artefact |
| Related | **CS-28 — Capstone** | The end-to-end project that assembles this module's pipeline with the eval harness |
| **Ethics / compliance** | **AP-01** | The distillation ToS question (§4.11), PII handling (§4.13), and automated-decision disclosures (§16.7) |
| Interview practice | **IQ-18** | The question bank built from this module |

---

## Appendix A — Instructor's Verbatim Key Claims

Quoted directly from the transcript with timestamps. The **Verdict** column is this module's
assessment: whether the claim holds today, and what changed. Read this table as the fastest possible
audit of the video's reliability.

### A.1 Positioning and method selection

| Timestamp | Verbatim claim | Verdict |
|---|---|---|
| [0:16]–[0:30] | Positions this video after the Axolotl one, as the hosted-API counterpart | Correct framing. Axolotl is CS-17; the contrast is the spine of §4.1 |
| [1:03]–[1:12] | Lists four training methods and names the fourth **"RLHF"** | **Wrong name.** The platform calls it **reinforcement fine-tuning (RFT)**, and it is not classical RLHF — it takes a grader, not a learned reward model, and is billed per hour. See §4.10 |
| [13:01]–[13:32] | Reads the methods-and-models table from the dashboard | Accurate as a description of what was on screen at the time; the model column is now almost entirely retired (§4.4.2) |
| [14:18]–[14:27] | *"minimum you can keep 10 examples… 50 to 100 example are the good one"* | **Correct, and better than most tutorials.** 10 is the API minimum; 50–100 is the practical floor. Garbled in the auto-transcript but the meaning is clear |
| [15:19]–[15:52] | Walks the `messages` JSONL schema out loud, role by role | Correct, and the schema is still exactly this today — it is the one part of this module that has not moved |
| [15:52]–[16:00] | *"not going to fine-tune on tool data"* | Correct for this dataset. The tool/function-call path is real but is a different data shape (§6.3) |
| [16:57]–[17:01] | *"I just kept the 10 rows just to demonstrate you"* | Honest framing — and it is exactly why the module's cost and quality conclusions do not generalise from this run (§15.1) |

### A.2 Pricing and cost

| Timestamp | Verbatim claim | Verdict |
|---|---|---|
| [19:52]–[20:06] | Reads the model list off the pricing page | Accurate at the time; the line-up has turned over almost completely (§4.4.2) |
| [20:08]–[20:40] | Reads the four pricing columns — training, input, cached input, output | **Correct**, and it is the right four columns. This is the part to keep |
| [20:43]–[20:57] | *"here is the cost for the inferencing"* | **Correct**, and it is the single most important sentence in the video — inference is the bill (§11.2) |
| [21:02]–[21:06] | *"$1 uh $150 per hour"* | **Wrong.** No hourly charge exists for supervised fine-tuning; billing is per million training tokens (§11.1) |
| [37:45]–[37:53] | *"cost of the GPU on a hourly basis"* | **Wrong**, same as above. The hourly model belongs to RFT only |
| [38:01]–[38:16] | Retracts his own correct reading of the pricing columns and reverts to hourly framing | **The retraction is the error.** This is the most consequential mistake in the video, because it is the one that gets carried into a budget spreadsheet |
| [45:51]–[46:23] | Reads the dataset statistics — 56, 62, 21, 25 tokens across the example | Consistent with the notebook; the transcript garbles individual digits in places |
| [46:38]–[46:54] | 562 total content tokens, 212 output tokens | **Matches the notebook exactly.** This is the `count_total_tokens` figure (§6.4) |
| [47:20]–[47:48] | The nano price block, then *"$1.5 for using the GPU… hourly rate"* | Price correct, framing wrong — $1.50 is per **million training tokens**, not per hour |
| [48:04]–[48:43] | *"0.23 something"*, *"$0.75"* training cost, *"$0.702396"* total | **Three stacked errors**, none of which reconcile with the notebook's own `$0.003024`. See §4.6.3 |
| [49:04]–[49:20] | Multiplies by 91 to get ₹ | Correct method (USD→INR), and the notebook's own cell 67 computes ₹0.275184 correctly from $0.003024 |

### A.3 Tokenisation, epochs and hyperparameters

| Timestamp | Verbatim claim | Verdict |
|---|---|---|
| [39:50]–[40:17] | Introduces `tiktoken` and says *"this is also a transformer based model"* | **Wrong.** `tiktoken` is a BPE lookup with a fixed merge table — no neural network, fully deterministic (§6.4) |
| [41:00]–[41:09] | Reads the token-ID table aloud | The notebook's table is authoritative; the transcript garbles several IDs (`12868` for `1268`; `"R then U"` for `" are"` / `" you"`). **Trust the notebook** |
| [47:20] region | Treats the 562-token figure as the billing basis | **Underestimate.** The billed count is 672 — the per-message overhead is real (§6.4) |
| [50:37]–[55:07] | Demonstrates the four API calls: upload, create, list, retrieve | **The best five minutes of the video**, and the mechanics are still accurate |
| [53:52]–[54:16] | Uses the `suffix` parameter | Correct, and worth doing — but it is the *only* provenance field he sets. No `seed`, no `metadata` (§6.6) |
| [55:09]–[55:22] | *"batch size is 16, means in how many batches we are passing the data in… this is irrelevant as of now because I just have 10 rows"* | **The second half is the good instinct.** `batch_size` is a trainer hyperparameter coupled to the LR, not a data-loading instruction, and on 10 rows the server resolves it to 1 (§7.5) |
| [55:24]–[55:39] | Describes `learning_rate_multiplier: 1.0` as *"zero learning rate, you know, to stabilize the training"* | **Wrong on both counts.** It is a multiplier, not a rate, and 1.0 means "provider default", not zero (§4.7.2) |
| [55:41]–[55:47] | *"number of epochs — for how many epochs you are running the training. I'm running it for the three epochs."* | **Correct**, and setting it explicitly is the right call given the auto policy resolves 10 on this dataset (§4.8) |

### A.4 Job, dashboard and evaluation

| Timestamp | Verbatim claim | Verdict |
|---|---|---|
| [1:00:52]–[1:00:00] | Reads the job fields off the dashboard | Accurate; the dashboard's field set has since changed shape |
| [1:00:42]–[1:00:48] | Points at the validation-data upload field | **He demonstrates it and never uses it.** `validation_file=None` in every job in both notebooks — the source of the module's central evaluation gap (§4.9.1) |
| [1:01:00]–[1:01:05] | *"15 to 20 minutes… 10 to 15 minutes"* for the job to complete | Plausible for a 10-row job in a queue. Wall-clock is queue-dependent and not a tunable |
| [1:03:02]–[1:03:16] | Chats with the finished model: *"What warranty does a smartphone come with?"* | The **only** evaluation in the video — and the question is in the training set, so a fluent answer proves memorisation works (§4.9.3) |
| *(never)* | A paired comparison against the base model on held-out prompts | **Absent.** This is the single largest gap in the video and the subject of §12 |
| *(never)* | A statement of what the model is worse at | **Absent.** No failure analysis anywhere |
| *(`data2.jsonl`)* | The broken dataset is validated on camera and never uploaded | Correct behaviour — it exists solely to make the validator fire |

### A.5 What the companion notebooks add beyond the narration

| Artefact | Notebook | Why it matters |
|---|---|---|
| `validate()` with `KeyError` | Run notebook, cell 14 | The live bug and the live fix — `defaultdict(int)` at cells 17–23 (§6.3) |
| The tiktoken token table | Run notebook, cell 39 | **Authoritative** where the transcript is garbled (`Hello`=9906, `,`=11, ` how`=1268, ` are`=527, ` you`=499, `?`=30) |
| `num_tokens_from_messages` | Run notebook, cells 42, 61 | The billing counter — the source of **672**, not 562 |
| The epoch-policy constants | Run notebook, cells 50/54/55/56 | `TARGET_EPOCHS=3`, `MIN/MAX_TARGET_EXAMPLES=100/25000`, `MIN/MAX_DEFAULT_EPOCHS=1/25`, `MAX_TOKENS_PER_EXAMPLE=16385` (§4.8) |
| The billing sum and cost text | Run notebook, cells 64–67 | `672 → 2016 tokens → $0.003024 → ₹0.275184`. **The correct arithmetic, in the instructor's own hand** |
| Job creation with `gpt-4o-2024-08-06` | Run notebook, cell 78 | `batch_size:16, learning_rate_multiplier:1.0, n_epochs:3`; cell 4 carries `# gpt-4o-2024-08-06` as a comment |
| The abandoned `gpt-3.5-turbo` job | First notebook, cell 16–17 | `hyperparameters=Hyperparameters(batch_size=1, learning_rate_multiplier=2.0, n_epochs=10)` — **the auto policy resolving to exactly the value §4.8 predicts** |
| The inline 4-unique-row dataset | First notebook, cell 7 | The real cause of the 10-epoch overfit; uploaded as `file-PsuEt1x4gLPqY8f4sDD49t` (6,429 bytes) |
| The 404 `model_not_found` | First notebook, cell 20 | Chatting with a cancelled job's model — the debugging lesson of §6.7 |
| `data.jsonl` / `data2.jsonl` | repo | 10 rows / 10 unique / 0 missing, and 10 rows / 10 unique / **1 missing** respectively (§6.2) |

---

## Appendix B — Reference Links & Papers

### B.1 Primary sources for this module

| Source | Path / location | Used for |
|---|---|---|
| Video transcript | `_source\transcripts\LLM_Fine-Tuning_20_OpenAIGPTs_Fine-Tuning_Masterclass_Supervised_FT_Token_Cost_A.txt` | Every timestamped claim in Appendix A |
| Run notebook (the video's) | `_source\repo\Complete-LLM-Finetuning-main\LLM Fine-Tuning-20-GPT-Finetuning\openai_api_and_finetuning_of_gpt_model (1).ipynb` | Validator, tiktoken table, both counters, epoch constants, billing arithmetic, job creation |
| First notebook (abandoned run) | `…\openai_api_and_finetuning_of_gpt_model.ipynb` | The auto-policy confirmation (`n_epochs=10`, `batch_size=1`, LR 2.0), the 404, the 4-unique inline dataset |
| Training data | `…\data.jsonl` (4,239 bytes, 10 rows, 10 unique, 0 missing assistant) | The training corpus |
| Negative fixture | `…\data2.jsonl` (4,060 bytes, 10 rows, 10 unique, **1 missing assistant**) | The validator demo |

### B.2 Platform documentation to re-check before relying on anything here

| Topic | What to verify | Why it moves |
|---|---|---|
| **Fine-tuning availability** | The current wind-down status and dates (§16.8) | Announced policy with dated tiers; re-check before committing |
| **Supported base models** | The eligible-model list | Turned over completely between the video and now (§4.4.2) |
| **Pricing** | Training, input, cached input, output — per model | The four columns the instructor reads are the right four; the numbers change |
| **Hyperparameter defaults** | The auto rules for `n_epochs`, `batch_size`, `learning_rate_multiplier` | Documented values have changed between API generations (§4.7.2) |
| **Dataset requirements** | Minimum examples, the seven validation errors, maximum file size | Minimums have moved; the seven checks have been stable for years |
| **Data retention & sharing** | Retention windows, ZDR eligibility, the data-sharing default, what the free-token programme excludes | Fine-tuning training and fine-tuned models are **excluded** from the shared-traffic programme (§4.13) |
| **Structured outputs** | `response_format` schema support and the `strict` flag | The correct tool for the most common request (§4.12) |
| **Method availability** | Whether SFT / vision / DPO / RFT are all still offered per model | Method-by-model support is not uniform |
| **SDK version** | The fine-tuning API surface in the installed `openai` package | `method={"type":"supervised",...}` replaced the flat `hyperparameters` argument within the 2.x line |

### B.3 Concepts and techniques cited in this module

| Concept | Where it is covered properly | One-line relevance here |
|---|---|---|
| Instruction fine-tuning / SFT | **CS-13** | The same loss, full control, your own GPU |
| Synthetic data & distillation | **CS-09** | The exit from an endpoint-only model (§4.11) |
| Retrieval-augmented generation | **CS-05** | The right tool when the problem is knowledge, not behaviour |
| LoRA / QLoRA | **CS-23** | What makes the migration target cheap enough to be real |
| DPO | **CS-25** | The hosted preference-tuning path in full (§4.10) |
| RLHF / PPO | **CS-24** | What RFT is *not* |
| GRPO | **CS-26** | Verifiable-reward training without a learned reward model |
| Framework landscape | **CS-03** | Where hosted APIs sit on the control/complexity axis |
| Small language models | **CS-20** | Why a 1–8B student is a viable migration target |
| Structured outputs / constrained decoding | §4.12 of this module | Guarantees shape; the answer for "it won't return JSON" |
| Tokenisation (BPE) | §4.5 of this module, and the tokenizer sections of **CS-13** | Deterministic, frozen, and the basis of every cost estimate |
| Preference data collection | **CS-25**, **AP-01** | Where `chosen`/`rejected` pairs come from, and the ethics of collecting them |
| Model deprecation management | §16.8 of this module | Parsing the base ID out of the `ft:` string as an early-warning system |

### B.4 The four numbers to memorise from this module

```text
1.  1.50        USD per 1,000,000 training tokens, gpt-4.1-nano  (2026 list)
2.  2x          the inference markup on a fine-tuned model, both directions
3.  100 // N    the automatic epoch count for a dataset below 100 examples
4.  P_f < P_b/2 - 2O    the break-even inequality for fine-tuning on cost grounds
```

---

*End of CS-18.*
