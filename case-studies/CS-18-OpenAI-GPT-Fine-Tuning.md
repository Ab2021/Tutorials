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

<!-- CONTINUE -->
