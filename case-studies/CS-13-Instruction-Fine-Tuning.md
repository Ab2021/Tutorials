# CS-13 — Instruction Fine-Tuning (SFT)

| Field | Value |
|---|---|
| **Module** | Supervised Fine-Tuning / Alignment Stage 2 |
| **Source video(s)** | LLM Fine-Tuning 15: Instruction Fine-Tuning Explained \| Domain-Specific FineTuning with Hugging Face |
| **Transcript file(s)** | `LLM_Fine-Tuning_15_Instruction_Fine-Tuning_Explained_Domain-Specific_FineTuning.txt` |
| **Companion code** | `LLM Fine-Tuning-15-Instruction Fine-Tuning Explained -Domain-Specific Fine-Tuning with Hugging Face/Instruction_finetuning_on_domain_specific_dataset.ipynb`, `pharma_instruction_data.jsonl`, `pharma_instruction_data.csv`, `tinyllama-instruction.zip` (2.7 MB → LoRA adapter, not a full model) |
| **Prerequisites** | CS-01 (the three-stage lifecycle), CS-02 (transfer learning), CS-06 (HF `Trainer`/`datasets`), CS-12 (domain-adaptive continued pretraining — the stage that runs *before* this one) |
| **Neighbours** | CS-14 (preference alignment: RLHF/PPO/DPO/ORPO), CS-15 (LLaMA-Factory), CS-16 (Unsloth), CS-17 (Axolotl), CS-18 (OpenAI SFT), CS-23 (LoRA/QLoRA deep dive) |
| **Difficulty** | Beginner to run, Expert to run *well*. The code is 20 lines; the data is 20 weeks. |
| **Hands-on required** | Yes — the notebook runs end to end on a free Colab T4 in under 5 minutes because the dataset is 5 rows |
| **Estimated study time** | 8h theory + 8h practical (build a 1k-row dataset yourself; that is the real exercise) |

---

## 0. Executive Summary

- **SFT = next-token prediction on `(prompt, response)` pairs, with the prompt's tokens masked out of the loss.** That is the whole mechanism. Everything else in this module is data curation, chat templating, masking hygiene, and knowing what SFT cannot do. The instructor's framing is exactly right and worth quoting: at every stage — pretraining, SFT, preference alignment — *"the model will always predict the next token in whatever way we have a data"* [11:44]. SFT is not a new objective. It is a **new data distribution for the same objective**.
- **The single most important number in this module: 1,000.** LIMA (Meta, 2023) showed that **1,000 carefully curated instruction/response pairs** fine-tuned onto a 65B base model produced a chatbot competitive with models trained on 50,000+ examples plus RLHF. The corollary, which the instructor states as *"one great example beats ten mediocre ones"* in spirit [24:00]–[25:00], is that **data quality dominates every hyperparameter in this module**. If you take one thing away: spend your time on the dataset, not on the LR sweep.
- **The thesis war you must be able to referee:** is SFT *teaching* the model new knowledge, or *eliciting* behaviour the base model already has? The correct answer is **mostly elicitation, with a thin, fragile layer of injection**. LIMA's authors call it the *superficial alignment hypothesis*: "a model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used." Counter-evidence exists and matters: SFT *does* inject narrow facts, but it takes 10–100 paraphrases per fact, the recall is brittle, and it is wiped by the next preference-tuning run. Full treatment in §4.1.
- **The masking decision is the highest-leverage line of code you will write.** Train on prompt + response (what the notebook does in cell 37) and you spend 60–90% of your loss teaching the model to *generate your prompt*. Mask the prompt (`-100`) and 100% of the gradient goes where you want it. The instructor recommends *not* masking [52:36] — **see the `Correction` callout in §4.4; the industry default is the opposite**, and the reason is arithmetic, not taste.
- **Silent bug #1 in the companion notebook:** `padding="max_length"` + `labels = input_ids.copy()` means **padding tokens are trained as targets.** On this 5-row dataset that is ~84% of the loss signal. §14 gives the one-line assertion that catches it. **Silent bug #2:** the notebook tokenizes with `TinyLlama-1.1B-Chat-v1.0`'s tokenizer but loads `TinyLlama-1.1B-intermediate-step-1431k-3T` as the model, and formats with `### Instruction:` / `[INST]` templates the model was never pretrained on. §4.3 explains why train/serve template mismatch is the most common production SFT failure.
- **Five data formats cover ~99% of the ecosystem:** Alpaca (`instruction`/`input`/`output`), ShareGPT (`conversations` with `from`/`value`), OpenAI chat (`messages` with `role`/`content`), DPO-ready pairs (`prompt`/`chosen`/`rejected`), and completion-only (`prompt`/`completion`). §4.2 renders the *same* Metformin row from `pharma_instruction_data.jsonl` in all five, and shows exactly which framework key reads which field (LLaMA-Factory `dataset_info.json`, TRL `messages`, Unsloth `train_on_responses_only`).
- **Chat templates are a contract, not a formatting detail.** `tokenizer.apply_chat_template(..., tokenize=False)` is the only correct way to render a prompt. If you train with `<|im_start|>` and serve with `### Instruction:`, you get a model that looks fine in your eval notebook and produces gibberish behind the API. §4.3 has the inspection one-liners.
- **SFT overfitting does not look like overfitting.** Val loss can be flat or falling while the model becomes **verbose, rigidly formatted, over-refusing, and worse at everything else**. §4.9 teaches the four signatures and the mitigations (epochs 1–3 not 10; LoRA not full FT; 5–20% general-data replay; eval on out-of-domain prompts).
- **SFT is often 90% of the value.** It is the stage that turns a text completer into something you can put behind a product. DPO/ORPO (§CS-14) are refinements that *require* a competent SFT checkpoint underneath — preference tuning on a raw base model is a known way to burn a week.
- **Cost reality for the reader's own project:** QLoRA SFT of an 8B model on 10k examples (mean 512 tokens) for 3 epochs = ~15.4M tokens ≈ **1.5 GPU-hours ≈ $2–4 on a rented A100**; full fine-tuning the same run is ~3.6 single-GPU-hours but needs **~160 GB of optimizer+weight state** and therefore 2–4× A100-80GB. §11 has the arithmetic.
- **STOP conditions that actually matter:** (1) if the knowledge you need changes weekly, use RAG (CS-04), not SFT; (2) if you have fewer than ~100 *genuinely good* examples and no budget to make more, prompt-engineer first; (3) if your metric is "does it know fact X", SFT is the wrong tool and continued pretraining (CS-12) or retrieval is the right one.

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world without SFT

| Failure | What it looks like | Root cause |
|---|---|---|
| **The brilliant autocomplete** | You ask a base Llama "List three contraindications for metformin." It replies with *"List three contraindications for metformin is a common question asked by patients who..."* — it completes your sentence instead of answering it. | A base model is a document simulator. Its training distribution is the web, where a question is followed by more prose *about* the question, not an answer to it. |
| **The pharma hallucination** | A raw Llama is dropped into a pharma company and confidently invents drug-interaction terminology. The instructor's exact framing [16:31]–[17:35]: *"this model does not know the pharma specific term... so it will hallucinate."* | Domain vocabulary was a rounding error in pretraining. The model has the *shape* of medical language and none of the *facts*. |
| **The "we prompted it, it's fine" plateau** | 40-shot prompts in a giant system message; latency 4× a fine-tune; cost per call 6×; still 12% format violations. | Prompting is a runtime tax that never amortises. Every call re-pays for the behaviour. |
| **The refusal machine** | You fine-tune on a safety-heavy dataset and now the model refuses to summarise a legal contract because it contains the word "terminate". | Safety data with no task data → the dominant behaviour in the loss becomes refusal. §4.9. |
| **The output that breaks the parser** | Your JSON extractor works 91% of the time; the other 9% costs an on-call engineer an hour a week. | Base models were never constrained to a schema. SFT is how you buy format compliance. |
| **The tone that isn't yours** | The model is correct and unusable — it writes like a Reddit thread, not like your company's support desk. | Style is a *distribution over formats*, which is exactly what SFT controls. |

### 1.2 The state of the art before instruction tuning

- **Pretraining + prompting.** GPT-3-era (2020): write a clever prompt, get whatever the model gives you. Zero-shot was unreliable, few-shot cost thousands of tokens per call, and there was no way to make the model *reliably* emit a format.
- **Pretraining + task-specific fine-tuning (pre-2021).** One model per task: a sentiment classifier, a summariser, an NER tagger. BERT-style (CS-07). This worked but did not scale — 30 tasks meant 30 checkpoints and 30 datasets with 30 label schemas.
- **Pretraining + supervised "instruction" datasets (FLAN, T0 — 2021/2022).** Fine-tune one model on ~60–1,800 NLP tasks phrased as instructions. This proved the *format* transfers across unseen tasks, but the tasks were academic and the outputs were not conversational.
- **The 2022 inflection: Self-Instruct → Alpaca.** `text-davinci-003` generated 52,000 instruction/response pairs from 175 hand-written seed tasks for about **$500**. Fine-tuning LLaMA-7B on that produced a model that subjectively behaved like a chat assistant. The recipe was now public, cheap, and reproducible — and it is the direct ancestor of every dataset you will use.
- **The remaining gap:** none of these told you *how much* data you actually need, or how to tell good data from bad. Enter LIMA (§4.1).

### 1.3 The naive approach and precisely why it fails

**Naive approach: "I have 5,000 documents. I'll convert each one to `{"instruction": "Summarise this", "output": <the doc>}` and fine-tune."**

This is the single most common first attempt and it fails for four separable reasons:

1. **You taught the model to copy, not to summarise.** Your target output *is* the input. The loss rewards reproduction. The model learns `output ≈ input` and learns nothing about the mapping you actually want.
2. **The instruction is constant, so it carries no information.** With one instruction repeated 5,000 times, the conditional distribution `p(output | instruction)` has zero variance in the conditioning variable. There is nothing to learn about instructions. You have paid full fine-tuning cost for a 5,000-example continued-pretraining run with a decorative prefix.
3. **The prompt tokens dominate the loss if you do not mask.** With a long document as the prompt and a short answer as the target, an unmasked run spends >95% of its gradient on document tokens.
4. **There is no negative space.** Every example says "yes, this is a summary". Nothing says "no, this is not", and nothing distinguishes a good summary from a bad one. LIMA's insight is that *diversity of tasks and phrasing* is what creates the conditional structure the model needs.

The correct version of this naive plan is: **first** do domain-adaptive continued pretraining on the raw documents (CS-12) — the instructor's exact recommendation [18:24]–[19:30], which he flags as the step most tutorials skip: *"this particular stage is very much important"* — and **then** hire or synthesise 1,000 *questions with answers* about those documents, where the question was written by someone who did not have the answer in front of them.

---

## 2. First-Principles Mental Model

### 2.1 The analogy: SFT is teaching a method actor their script, not their life

Imagine an actor who has read the entire internet (pretraining), then gets a 1,000-page script (SFT). The script does not teach them chemistry, or French, or how to ride a horse — they already absorbed a working model of all of that from the internet. What the script teaches is **which of their thousand personas to be when someone says "action"**, and **what a scene looks like**: enter, deliver, pause, exit. Swap the script for another one and they are still competent; they just play a different role.

That is why LIMA's 1,000 examples work, and why 1,000 *random* examples do not: the script has to show the *range* of scenes you care about. A script with 1,000 variations of "enter, say hello, exit" produces an actor who can only do entrances.

**Where this analogy breaks.** Two places. (1) An actor can learn a genuinely new fact from a script — "your character is left-handed" — and SFT can too, but far more weakly than the analogy suggests: it takes many repetitions and it is the first thing forgotten. (2) The actor does not lose their other skills from reading a bad script; an LLM **does** — this is catastrophic forgetting, and it is the dominant failure mode of over-long SFT runs (§4.9). The analogy's biggest lie is that the script is free.

### 2.2 The mechanism, stated precisely

Pretraining minimises, over the whole web:

$$L_{PT} = -\sum_{t} \log p_\theta(x_t \mid x_{<t}), \qquad x \sim \mathcal{D}_{web}$$

SFT minimises the *same* functional form over a *different* distribution, with a *selective* mask:

$$L_{SFT} = -\sum_{t \in \mathcal{R}} \log p_\theta(x_t \mid x_{<t}), \qquad x \sim \mathcal{D}_{instruct},\ \ \mathcal{R} = \text{response token positions only}$$

Two things changed and one did not:

| | Pretraining | SFT |
|---|---|---|
| Objective | next-token CE | next-token CE (**identical**) |
| Data distribution | web text, ~10T tokens | instruction/response pairs, typically 1M–500M tokens |
| Supervision mask | every token | **response tokens only** (assistant spans) |
| Tokenisation / template | raw text | **structured**: role markers, turn separators |
| Gradient signal per example | ~1 token of supervision per token | ~1 token per token, but *all of it on the behaviour you want* |

**The entire value of SFT comes from the delta in the data distribution and the mask.** The instructor makes this point twice [11:44], [26:13] and it is the right mental anchor: you are not changing the learning rule, you are changing *what the model sees when it applies it*, and *where the loss is computed*.

### 2.3 Why the mask matters — the information argument

Consider one row of `pharma_instruction_data.jsonl`, rendered as a string:

```
### Instruction:
Explain the mechanism of action of Metformin.
### Input:

### Response:
Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake
and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose.
```

The prompt is 25 tokens; the response is 32 tokens. Now compute what each masking choice supervises:

| Masking scheme | Tokens contributing to loss | Fraction of loss on the *answer* | What else the model is trained to produce |
|---|---|---|---|
| No mask (notebook cell 37) | 57 | 56% | The entire prompt — including the literal token `### Instruction:` and the question itself |
| Prompt masked (notebook cell 54) | 32 | **100%** | Nothing but the answer |
| Prompt + template header masked (industry default) | 29 | **100%** | Nothing but the answer, and the model is never trained to emit the header it just received |

The failure mode of "no mask" is not a small inefficiency — it is a *different task*. You are training a model that, given `### Instruction:\nExplain the mechanism of action of Metformin.\n### Input:\n\n### Response:\nMetformin activates...`, learns that the highest-probability continuation of `### Instruction:` is `\nExplain the mechanism of action of Metformin.` That is prompt-completion, i.e. a very expensive way to build an autocomplete that occasionally answers questions.

**The exception, stated honestly:** there are cases where you *want* prompt tokens in the loss — §4.4.4. They are rare and none of them is the case in this notebook.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **SFT** (Supervised Fine-Tuning) | Fine-tuning a pretrained LM with next-token cross-entropy on `(prompt, response)` pairs where the response is human- or model-authored. | The stage that converts a base model into an assistant. Sometimes called IFT (instruction fine-tuning) or SFT-1. | "Supervised" here means "the target tokens were chosen by a human", not "there is a separate classification head". The instructor's explanation [12:59] is correct but incomplete — he never mentions the mask. |
| **IFT** (Instruction Fine-Tuning) | Synonym for SFT when the data is instruction-shaped. | Interchangeable in practice; the instructor uses IFT for the video title and SFT for the concept [1:52]–[2:06]. | Not a different technique. |
| **Base model** | A model trained only on pretraining (next-token on web text). Llama-3.1-8B vs Llama-3.1-8B-Instruct. | The starting point for SFT; knowing which you have prevents the #1 setup bug. | People fine-tune `-Instruct` checkpoints when they mean to build a domain model, then fight the existing refusal behaviour (see §4.9, over-refusal). |
| **Chat model / Instruct model** | A base model that has already been through (SFT → preference tuning). | Fine-tuning one of these is "second-stage SFT" and needs less data and lower LR. | The two are not interchangeable when it comes to learning rate: an instruct model is already near a loss basin. |
| **Prompt / instruction** | The conditioning text: everything up to and including the assistant header. | The masked-out region. | "Instruction" sometimes means just the task sentence, "prompt" the whole rendered string. In the Alpaca format they are different fields. |
| **Response / completion / target** | The tokens whose loss you compute. | The only thing SFT actually teaches. | In `trl`'s legacy API, `completion` is the target; in the OpenAI API, the same field is a `messages` array. |
| **Loss masking / prompt masking / response-only loss** | Setting `labels = -100` on non-assistant token positions so cross-entropy ignores them. | The single highest-leverage implementation detail. §4.4. | `-100` is not magic; it is `CrossEntropyLoss(ignore_index=-100)`'s default. Any sentinel works if you pass `ignore_index`. |
| **`ignore_index`** | The label value that `torch.nn.CrossEntropyLoss` skips. Default `-100`. | Explains why `-100` appears everywhere. | `-100` is *not* masked in the input; it is only in the label tensor. |
| **Chat template / chat format** | A Jinja2 template that renders a message list into the exact string the model was trained on: `<|im_start|>user\n...<|im_end|>\n`. | Train/serve mismatch here is the most common production SFT failure. §4.3. | A chat template is *not* a prompt template you can invent. It must match pretraining or you are fine-tuning on noise. |
| **ChatML** | The `<|im_start|>role\ncontent<|im_end|>` format, introduced by OpenAI, adopted by Qwen, Yi, Nous/Hermes, and many others. | The de-facto open standard; the most common `chat_template` you will read. | Not the same as the OpenAI *API* chat format (which is JSON, not a string). |
| **Special tokens** | Tokens outside the natural-text vocabulary: `<|im_start|>`, `<|eot_id|>`, `[INST]`, `<start_of_turn>`. | They carry the role structure. Truncating or `strip()`ing them silently destroys your training signal. | `tokenizer(text)` vs `tokenizer(text, add_special_tokens=False)` differ by the BOS/EOS — matters when you compute mask offsets. |
| **BOS / EOS** | Begin/end-of-sequence. `<s>`/`</s>` for Llama-family, `<bos>`/`<eos>` for Gemma. | The EOS token is what teaches the model to **stop**. If your response spans lack EOS, the model never learns to end a turn. | Many pipelines strip EOS to "avoid double tokens" and then wonder why the model rambles forever. |
| **PAD token** | Placeholder used to make a batch rectangular. | If `-100` is not applied to pad positions, **the model is trained to generate padding**. Silent, common, catastrophic. | `tokenizer.pad_token = tokenizer.eos_token` is the usual fix, and it is exactly what makes the pad-as-target bug invisible. |
| **Alpaca format** | `{"instruction", "input", "output"}`. | The most widely supported SFT schema; the default in LLaMA-Factory, Axolotl, and most HF datasets. | `input` is often the empty string `""`, not missing. Code that checks `if example["input"]` behaves differently from code that checks `if "input" in example`. |
| **ShareGPT format** | `{"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}`. | The multi-turn standard; required for anything conversational. | `from`/`value` are the *legacy* keys; LLaMA-Factory lets you rename them via `tags`. |
| **OpenAI chat format** | `{"messages": [{"role": "system"\|"user"\|"assistant", "content": ...}]}`. | The format the OpenAI fine-tuning API consumes and the format TRL/Unsloth use natively. | Roles are `system`/`user`/`assistant`/`tool` — *not* `human`/`gpt`. |
| **DPO-ready pair** | `{"prompt", "chosen", "rejected"}`. | The output of SFT is the input to DPO (CS-14). Getting the schema right at SFT time saves a re-export. | `prompt` may be a string or a message list; `chosen`/`rejected` are the *completions only*, not full conversations, in TRL's default schema. |
| **Completion-only format** | `{"prompt": "...", "completion": "..."}`. | The legacy TRL/OpenAI-completions format; still everywhere in older code. | Loss masking is implicit (the `prompt` field is never trained), which is why old code has no `-100` in it. |
| **LIMA** | Meta's 2023 paper "Less Is More for Alignment": 1,000 curated examples, 65B LLaMA, no RLHF. | The empirical anchor for "data quality ≫ quantity". §4.1. | LIMA did *not* prove 1k is optimal — it proved 1k *curated* beats 52k *uncurated* on a model that already had the capability. |
| **Superficial alignment hypothesis** | LIMA's claim that pretraining learns capabilities and alignment learns *which subdistribution of formats* to use. | The strongest statement of the "elicitation, not injection" thesis. | Often over-stated. It holds well for style/format, weakly for facts. §4.1.3. |
| **Self-Instruct** | Wang et al. 2022: bootstrap instruction data from a model using 175 seed tasks. | The origin of the synthetic-data pipeline; Alpaca is its most famous output. | The seeds matter more than the model. 175 diverse seeds ≫ 10,000 similar ones. |
| **Alpaca** | The 52k Self-Instruct dataset generated with `text-davinci-003`. | The canonical starter dataset. | Its licence is a landmine — OpenAI's ToS forbade using its outputs to train competing models. §4.8. |
| **Decontamination** | Removing training examples that overlap your evaluation set. | Otherwise your eval measures memorisation. | 13-gram overlap (GPT-3 paper) is the standard; sentence-level embedding dedup misses verbatim leakage. |
| **Packing** | Concatenating multiple short examples into one `max_seq_length` window so no compute is spent on pad tokens. | 2–5× throughput on short-response data. | Requires position-id resets per sequence. §4.5.3 and the Beyond-the-video callout. |
| **NEFTune** | Adding uniform noise to the embedding layer during SFT only. | Free 3–30% win on conversational benchmarks; one config flag. §7.9. | Inference is unchanged — that is the point. |
| **`neftune_noise_alpha`** | The noise scale parameter (typical 5). | The only knob. | Must be 0/None at inference; the flag is train-time only in HF. |
| **Format compliance rate** | % of generations that parse against your schema. | The KPI that actually matters for product SFT. §12. |
| **Refusal rate** | % of in-domain requests the model declines. | Rises with over-training; the counter-metric to format compliance. §4.9. |
| **Win rate** | Pairwise-judge preference for your model vs a baseline. | The standard subjective metric; also the easiest to fool. §12. |
| **Catastrophic forgetting** | Loss of general capability after narrow fine-tuning. | The hidden cost of a long SFT run; measured by MMLU/GSM8K delta. §4.9.4. |
| **Alignment tax** | The general-capability cost of alignment. | Why replay data exists. | Often measured wrong — compare against the *SFT-start* checkpoint, not the base, when judging DPO. |
| **Replay / rehearsal data** | Mixing 1–20% general instruction data into a domain SFT run. | The cheapest, most reliable forgetting mitigation. §4.9.5. | Replay of *pretraining* text (5–10%) also helps but conflicts with the mask (you must not mask it). |
| **Adapter** | A small trained delta (LoRA matrices) that can be merged or served separately. | 2.7 MB per 1.1B model instead of 2.2 GB. | The notebook's zip is 2.7 MB — it is an adapter, which is why loading it with `AutoModelForCausalLM` is wrong (§10.2). |
| **Merging** | `model.merge_and_unload()` — folding adapter weights into the base. | Needed for vLLM/single-artefact deploys. | Irreversible. Keep the adapter and the base hash if you might need to un-merge. |
| **`max_seq_length`** | The truncation window. | Truncating mid-response trains the model to stop mid-sentence. §7.7. | Not the same as the model's context window. Set it to the p99.5 of *your* token lengths, not to 8192 "because it fits". |
| **Gradient checkpointing** | Recompute activations during the backward pass instead of storing them. | ~60–70% activation memory saved for ~25–30% more time. | The single best memory/time trade in SFT. §4.5.4. |
| **Gradient accumulation** | Summing gradients over N micro-batches before stepping. | Decouples batch size from VRAM. | Interacts with the LR schedule: `num_training_steps` is computed *after* accumulation, so changing the accumulation changes your warmup length. §7.6. |
| **Effective batch size (tokens)** | `micro_batch × seq_len × grad_accum × n_gpus`. | The number that matters for stability: target 32k–128k tokens/step. | Examples/step is a useless unit when sequence lengths vary 10×. |
| **Epoch** | One pass over the dataset. | For SFT the useful range is 1–3. More is almost always worse. §7.3. | With 1k–10k examples, "epoch" is a coarse knob; think in optimizer steps instead. |
| **Warmup** | Linear LR ramp at the start. | Prevents the first steps from destroying a pretrained model. 3–10% of steps. §7.4. | With 3 total steps (this notebook) warmup is meaningless — see §7.4's degenerate case. |
| **LLM-as-judge** | Using a strong model to score generations. | The only practical way to measure open-ended SFT quality. | Pairwise with position swap, or it is measuring the judge's position bias. §12.4. |
| **Position bias** | Judges prefer whichever response appears first (or second). | Fixable only by evaluating both orders and averaging. | Measured at 10–15 points of win rate in published audits. §12.4. |
| **Verbosity bias** | Judges prefer longer answers. | The reason SFT over-training shows up as length inflation. §4.9.1. | Correlated with, but distinct from, position bias. |
| **Over-refusal** | The model declines benign in-domain requests. | The most common *product* failure of a safety-heavy or over-trained SFT. §4.9.3. | Diagnosed with a fixed 200-prompt benign suite, not by vibes. |
| **Format rigidity** | The model only works when the prompt matches the training template exactly. | A direct consequence of over-fitting the template. §4.9.2. | Test by mutating the prompt: drop the system message, add whitespace, reorder fields. |
| **Decontamination set** | Your held-out eval prompts, excluded from training by n-gram overlap. | Without it, eval numbers are fiction. | Also apply to your *judge* prompts. |
| **`assistant_only_loss`** | TRL's `SFTConfig` flag to compute loss only on assistant spans, using the `{% generation %}` markers in the chat template. | The modern, template-aware way to mask. §4.4.3. | Requires a template with generation tags; not all Hub templates have them. |
| **`train_on_responses_only`** | Unsloth's helper that patches a `Trainer` to mask all but the response spans. | Same idea, works with templates that lack generation tags. §4.4.3. | String-matching based — if your template splits the marker across tokens, it silently no-ops. |
| **Dataset card / licence** | The Hub metadata describing provenance and permitted use. | Alpaca's OpenAI-ToS problem is the canonical cautionary tale. §4.8. | "Available on the Hub" ≠ "licensed for commercial training". |
| **Data version hash** | A content hash (`datasets` fingerprint, git-LFS SHA, DVC hash) pinned to a model artefact. | Without it you cannot reproduce or roll back. §16.1. | `load_dataset("csv", data_files=...)` gives you no fingerprint at all. |
| **Regression suite** | A frozen set of prompts with expected properties, run on every candidate checkpoint. | The only defence against shipping a regression. §16.5. | Assert *properties* (parses as JSON, contains no refusal phrase), not exact strings. |

---

## 4. Deep Dive — How It Actually Works

### 4.1 What SFT actually does to a base model: the two theses

This is the most important conceptual question in the module, and the one most likely to be asked in an interview at L3+.

#### 4.1.1 Thesis A — "SFT teaches the model to *respond*" (the format/behaviour thesis)

The claim: a base model already contains, in superposition, the capability to answer questions, summarise, translate, and write code. Pretraining on the internet necessarily included millions of Q&A pages (Stack Overflow, Reddit, Quora), tutorial pages, and documentation. The capability is *present but not elicited*: given `"Explain the mechanism of action of Metformin."` a base model's highest-probability continuation is more prose *about* the sentence, because that is what the web mostly contains next.

Evidence for Thesis A:

1. **LIMA (Meta AI, Zhou et al., 2023).** 1,000 curated prompt/response pairs, no RLHF, no preference model, fine-tuned onto LLaMA-65B. Result: 43% of the time, LIMA's output was preferred over or tied with GPT-4 in human pairwise comparison; 58% vs. Bard; 65% vs. Alpaca-65B. The paper's own conclusion: *"a model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."* This is the **superficial alignment hypothesis**.
2. **The 1,000 examples were curated for diversity, not volume.** LIMA's authors deliberately sampled from three sources — 750 from community Q&A (Stack Exchange, wikiHow, Reddit), 200 from *authored* examples (the model's own failures, hand-fixed), 50 from Super-NaturalInstructions — and applied strict filters. Volume was not the variable that moved.
3. **The instructor's own observation [28:59]:** ChatGPT's "successful mantra" is instruction fine-tuning. Note carefully what he describes: pretraining first on the whole internet, *then* Q&A data from Reddit/Quora/GitHub/Stack Overflow [42:00], [54:21]–[54:37]. The second stage did not add a capability that was absent; it changed which behaviour was emitted.
4. **InstructGPT's own finding (Ouyang et al., 2022):** 1.3B-parameter InstructGPT was preferred to 175B GPT-3 on human prompts. The 128× smaller model had *less* knowledge. What it had was better *elicitation*.
5. **Format compliance is near-100% learnable from a few hundred examples.** If SFT were teaching a capability, you would expect a scaling curve. You get a step function.

#### 4.1.2 Thesis B — "SFT injects knowledge"

The claim: fine-tuning on `(question, fact)` pairs writes new facts into the weights.

Evidence for Thesis B:

1. **It demonstrably works for narrow, repeated facts.** Train on 500 paraphrases of "Acme-7 is our internal ticketing system" and the model will state it. The information is in the weights — you can measure it with a probe on the MLP layers.
2. **Domain SFT improves domain QA benchmarks.** Med-PaLM/Meditron-style runs improved medical QA by double digits — though careful ablations attribute most of the gain to *continued pretraining* on domain text (CS-12), not to the SFT stage.
3. **The instructor's own demo is a (weak) instance:** after SFT the TinyLlama-1.1B model emits pharma vocabulary that the base model did not. But note *where* it learned the vocabulary: the **non-instructional (continued-pretraining) stage on the PDF** [31:46]–[32:38], not the 5-row SFT. This is the instructor's actual workflow, and it is the correct one: *"first teach your model on your domain specific data set on your PDF text and all and later on perform some instruction tuning"* [54:10]–[54:19].

#### 4.1.3 The correct answer

**SFT primarily teaches *how to respond*. It injects new facts only weakly, expensively, and unreliably.**

| Property | Behaviour/format (elicitation) | Factual content (injection) |
|---|---|---|
| Examples needed | 100–1,000 | 10–100 paraphrases *per fact* |
| Learned after 1 epoch? | Yes | Partially |
| Survives 200 more steps of SFT? | Yes | Often overwritten |
| Survives a DPO run afterwards? | Yes | Frequently erased |
| Generalises to unseen phrasings? | Yes (that is the point) | No — the model matches surface forms |
| Reliable at inference? | ≥95% format compliance achievable | Recall is patchy and confidence is uncalibrated |
| Cost-efficient alternative | — | RAG (CS-04) for facts; continued pretraining (CS-12) for terminology |
| Where it lives in the weights | Output-distribution shift, strongly in the last layers and the unembedding | Scattered MLP writes, hard to find and easy to damage |

Practical decision rule:

```
Do you need the model to behave differently?  → SFT. (1k–50k curated examples.)
Do you need the model to know different things? → continued pretraining (CS-12) for
                                                  terminology/vocabulary, RAG (CS-04)
                                                  for retrievable facts, and only
                                                  then SFT to make it answer in
                                                  your house style.
Do you need both, from scratch?                → CS-12 then CS-13, in that order.
```

> **Beyond the video:** the sharpest modern statement of this is the "SFT memorizes, RL generalizes" line of work (Chu et al., 2025): models trained with SFT on a reasoning task fit the training distribution and fail to extrapolate, while RL-trained models generalise — even though SFT reaches higher *training* accuracy. The practical reading for a production SFT engineer: **do not expect SFT to teach a new reasoning procedure**; expect it to teach the *format* that lets your RL or DPO stage (CS-14) do the teaching.

> **Beyond the video — the "eliciting, not teaching" framing, stated for an interview:** *SFT is a prior-shifting operation. You are reweighting a distribution that already exists in the base model, not writing a new one into it. The evidence is that 1,000 examples change behaviour, that a 1.3B model can beat a 175B model on human preference despite knowing less, and that capability benchmarks barely move while format compliance goes to ~100%. The counter-evidence — that facts can be injected — is real but bounded: it takes roughly an order of magnitude more data per fact than per behaviour, and the injection is the first thing a subsequent preference-tuning stage destroys.*

#### 4.1.4 Why this matters operationally

If you believe Thesis B, you build a 200k-row synthetic dataset and train for 5 epochs, and you ship a model that is confidently wrong in a new style.

If you believe Thesis A correctly, your plan looks like this:

| Question you are answering | Dataset you build | Size | Stage |
|---|---|---|---|
| Does it speak pharma? | Raw PDFs, plain text | 10M–1B tokens | CS-12 (continued pretraining) |
| Does it answer like a pharma professional? | Curated Q&A about the pharma material | 1k–20k rows | **CS-13 (this module)** |
| Does it prefer *our* answer over the alternative? | Chosen/rejected pairs | 1k–20k pairs | CS-14 (DPO/ORPO) |
| Does it never say X? | Safety pairs + refusal examples | 200–2k rows | CS-13, mixed with task data |

---

### 4.2 The data-format landscape — the same example in five formats

One row from `pharma_instruction_data.jsonl` (line 1 of 5) is used throughout this subsection so the differences are impossible to miss:

```json
{"instruction": "Explain the mechanism of action of Metformin.", "input": "", "output": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."}
```

#### 4.2.1 Format 1 — Alpaca (`instruction` / `input` / `output`)

The instructor introduces exactly this schema [20:14]–[21:10] and names Alpaca as its source. He then shows the "with input" variant (his summarisation example, where `instruction` = "summarize the following paragraph" and `input` = the paragraph) and the "without input" variant (`instruction` = "explain the mechanism of the mRNA format", `input` = empty) — and he gives the rule for which you use [22:17]–[23:16]:

> *"if instruction uses extra text... we are giving an instruction to our model with respect to this input and then we are generating this output. Now, the second line: if instruction is a self-contained context or dialogue style or conversational data then you can keep input empty."*

That is the whole Alpaca convention, correctly stated.

```json
{
  "instruction": "Explain the mechanism of action of Metformin.",
  "input": "",
  "output": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."
}
```

The notebook then flattens it into a single training string (cell 31) — this is the **Alpaca-style flat text** that the trainer actually consumes:

```text
### Instruction:
Explain the mechanism of action of Metformin.
### Input:

### Response:
Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose.
```

Three notes that matter:

- The literal `### Input:` line is emitted **even when `input` is empty** — the notebook's f-string does `{example['input']}` which produces the string `None` when the CSV parser yields `None`, or an empty line when it yields `""`. The instructor's own printed output shows `### Input:\nNone` [cell 35]. **A trained model learns to emit the literal token `None`** as part of its format. Fix with `example.get("input") or ""`. This is a real, shipped bug class.
- This flat format is what LLaMA-Factory calls `alpaca` but it is *not* the format LLaMA-Factory trains on — LLaMA-Factory applies the model's **chat template** to the three fields. The flat string above is a hand-rolled template, which is precisely the train/serve mismatch risk of §4.3.
- `input` being present-but-empty vs. absent changes behaviour of naive format detection. Always normalise.

#### 4.2.2 Format 2 — ShareGPT (`conversations` with `from` / `value`)

The instructor describes this as "conversational data... user and assistant" [21:44]–[21:56] and separately as "system, user and assistant" [21:56]. ShareGPT is the schema for that:

```json
{
  "conversations": [
    {"from": "system", "value": "You are a pharmacology assistant for Acme Pharma. Answer concisely and cite the mechanism class."},
    {"from": "human", "value": "Explain the mechanism of action of Metformin."},
    {"from": "gpt", "value": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."}
  ]
}
```

Why it exists: it is the dump format of the ShareGPT browser extension, which captured real human↔ChatGPT conversations. Its virtue is that it is *natively multi-turn* — the Alpaca schema has no place to put turn 2. Every serious conversational SFT dataset (OpenAssistant, UltraChat, WildChat, Tulu) is in this shape or a near relative.

The role names are the trap: `human`/`gpt` (ShareGPT), `user`/`assistant` (OpenAI), and any custom pair a dataset author invented. LLaMA-Factory lets you map them:

```json
{
  "pharma_sharegpt": {
    "file_name": "pharma_sharegpt.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "system": "system" },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
}
```

**Multi-turn loss masking.** With ShareGPT data, "the response" means *every* assistant turn, not just the last one. A correct mask is per-turn. If you naively mask up to the *last* `gpt` turn you throw away all earlier turns' supervision; if you mask nothing you teach the model to produce the user's messages. §4.4.3 shows the template-driven way to do it.

#### 4.2.3 Format 3 — OpenAI chat (`messages` with `role` / `content`)

The format the OpenAI fine-tuning API consumes, the format `tokenizer.apply_chat_template` consumes, and the format TRL's `SFTTrainer` and Unsloth consume natively:

```json
{
  "messages": [
    {"role": "system", "content": "You are a pharmacology assistant for Acme Pharma. Answer concisely and cite the mechanism class."},
    {"role": "user", "content": "Explain the mechanism of action of Metformin."},
    {"role": "assistant", "content": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."}
  ]
}
```

This is the format to build in. It is a strict superset of ShareGPT (role names normalised, `content` a string), it renders through any chat template, it converts to Alpaca in three lines, and it is what the preference-tuning stages want. If you get to choose one internal schema:

> **Store in `messages`. Convert on export.** One canonical internal format, one converter per training framework. Every dataset bug I have seen in production SFT came from three teams each inventing their own intermediate JSON.

Two API-level details that differ from the file format and bite people:

- The OpenAI **fine-tuning API** expects a `messages` array in a JSONL file where **each line is a complete conversation**, the last message is the assistant's, and there is a `weight` field (0/1) available for loss masking. It does *not* accept `system`-first requirements — a system message is optional.
- The OpenAI **Chat Completions** runtime format is the same shape, which is why prompt-format drift between training and serving is invisible until you run a real evaluation.

#### 4.2.4 Format 4 — DPO-ready pairs (`prompt` / `chosen` / `rejected`)

Not an SFT format strictly — it is what you *produce* on the way out of SFT and consume in CS-14 — but you should design your dataset so the export exists:

```json
{
  "prompt": "Explain the mechanism of action of Metformin.",
  "chosen": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose.",
  "rejected": "Metformin is a biguanide that has been used for decades. It is generally well tolerated and is often the first-line therapy for type 2 diabetes. Mechanism of action is complex and involves multiple pathways."
}
```

TRL's `DPOTrainer` accepts two schemas: (a) `{"prompt", "chosen", "rejected"}` where `chosen`/`rejected` are **completions only** (strings); (b) `{"chosen": [messages...], "rejected": [messages...]}` where the prompt is inferred as the shared prefix. Which one you have determines whether the tokenizer renders the prompt once or twice. Getting it wrong adds a silent duplicate-prompt to half your pairs. See CS-14 §4 for the full schema discussion.

**Why this is in the SFT module:** because the best moment to capture `rejected` is *while you are curating SFT data*. If you are already paying a domain expert to write the good answer, pay them 20% more to write/select a plausible-but-wrong answer in the same session. That single decision saves a complete relabelling project two weeks later.

#### 4.2.5 Format 5 — Completion-only (`prompt` / `completion`)

The oldest schema in the stack, still in the TRL docs and in every pre-2024 tutorial:

```json
{
  "prompt": "### Instruction:\nExplain the mechanism of action of Metformin.\n### Input:\n\n### Response:\n",
  "completion": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."
}
```

Two properties define it:

1. **The prompt is already rendered.** The training code does not touch a chat template; the dataset author baked the exact string in.
2. **Masking is implicit and total.** The loss is computed on `completion` tokens only, always. There is no `-100` in the dataset because the framework constructs the labels itself.

Why it survives: it is the only format where you can express arbitrary context (retrieved documents, tool outputs, few-shot examples) without fighting a template. Why it is dangerous: the prompt string is now *data*, so a mistyped `### Response:` in one row silently changes the task for that row, and nothing will complain.

```json
// TRL legacy: one JSONL line per example
{"prompt": "...", "completion": "..."}
```

#### 4.2.6 One example, five formats — side by side

| Format | Key structure | Rendered by | Prompt masking | Multi-turn | Framework |
|---|---|---|---|---|---|
| **Alpaca** | `instruction`/`input`/`output` | you (f-string) *or* chat template | manual `-100` or LLaMA-Factory's `train_on_prompt: false` | No | LLaMA-Factory (`alpaca`), Axolotl (`alpaca`), `datasets` |
| **ShareGPT** | `conversations[]` of `{from, value}` | chat template | per-turn, framework or manual | **Yes** | LLaMA-Factory (`sharegpt`), Axolotl (`sharegpt`), FastChat |
| **OpenAI chat** | `messages[]` of `{role, content}` | `apply_chat_template` | `assistant_only_loss=True` | **Yes** | TRL `SFTTrainer`, Unsloth, OpenAI API |
| **DPO pair** | `prompt`/`chosen`/`rejected` | chat template, 2× | implicit (chosen+rejected only) | Both forms | TRL `DPOTrainer`, `ORPOTrainer` (CS-14) |
| **Completion-only** | `prompt`/`completion` | nobody — pre-rendered | implicit and total | No (stuff it in the prompt) | TRL legacy `SFTTrainer`, OpenAI legacy completions |

#### 4.2.7 Conversion utilities you will actually use

```python
# --- alpaca -> messages (the one conversion everyone writes, badly) ---
def alpaca_to_messages(row, system=None):
    """Alpaca's `input` is not a message; it is context appended to the instruction.
    Concatenating with two newlines is the convention used by Axolotl/LLaMA-Factory."""
    user = row["instruction"]
    extra = (row.get("input") or "").strip()
    if extra:
        user = f"{user}\n\n{extra}"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": row["output"]})
    return {"messages": msgs}

# --- sharegpt -> messages ---
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}

def sharegpt_to_messages(row):
    return {"messages": [
        {"role": ROLE_MAP[c["from"].lower()], "content": c["value"]}
        for c in row["conversations"]
    ]}

# --- messages -> alpaca (for tools that only eat alpaca) ---
def messages_to_alpaca(row):
    msgs = row["messages"]
    sys = next((m["content"] for m in msgs if m["role"] == "system"), "")
    usr = [m["content"] for m in msgs if m["role"] == "user"]
    ast = [m["content"] for m in msgs if m["role"] == "assistant"]
    return {
        "instruction": (sys + "\n\n" if sys else "") + (usr[0] if usr else ""),
        "input": "\n\n".join(usr[1:]),
        "output": ast[-1] if ast else "",
    }

# --- messages -> dpo pair (only valid when you have a rejected completion) ---
def messages_to_dpo(row, rejected: str):
    msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    return {"prompt": msgs, "chosen": row["messages"][-1]["content"], "rejected": rejected}
```

> **Beyond the video:** LLaMA-Factory's `dataset_info.json` is the cleanest formalisation of this entire section, and reading it once will save you a week. The four keys are `file_name`, `formatting` (`alpaca` | `sharegpt` | and a few less common ones), `columns` (which dataset column maps to which logical field), and `tags` (which *values* inside the data mean user/assistant/system). Axolotl's YAML does the same job with `type: alpaca|sharegpt|completion` plus `field_instruction`, `field_input`, `field_output`, `field_messages`, `roles_to_train`. If your dataset does not fit either schema, write a converter to `messages` rather than inventing a sixth format.

---

### 4.3 Chat templates in depth

A chat template is a Jinja2 program that turns a message list into the exact byte string the model was trained on. It lives in `tokenizer_config.json` under `chat_template`, and it is exposed as `tokenizer.chat_template` (a string).

#### 4.3.1 What the templates actually look like

| Family | Model examples | Template shape | Turn-end token |
|---|---|---|---|
| **ChatML** | Qwen 2/2.5, Yi, Hermes, OpenHermes, InternLM | `<\|im_start\|>role\ncontent<\|im_end\|>\n` | `<\|im_end\|>` |
| **Llama-2** | Llama-2-Chat, CodeLlama-Instruct | `<s>[INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant </s>` | `</s>` |
| **Llama-3** (and 3.1/3.2/3.3) | Llama-3.1-8B-Instruct, TinyLlama-Chat-v1.0 (Zephyr-style) | `<\|begin_of_text\|><\|start_header_id\|>role<\|end_header_id\|>\n\ncontent<\|eot_id\|>` | `<\|eot_id\|>` |
| **Mistral / Mixtral** | Mistral-7B-Instruct-v0.2/v0.3 | `<s>[INST] user [/INST]assistant</s>` (no system role in v0.1/v0.2) | `</s>` |
| **Gemma / Gemma-2** | Gemma-2-9b-it | `<start_of_turn>user\ncontent<end_of_turn>\n<start_of_turn>model\ncontent<end_of_turn>` | `<end_of_turn>` |
| **Phi-3 / Phi-4** | Phi-3-mini-4k-instruct | `<\|user\|>\ncontent<\|end\|>\n<\|assistant\|>\ncontent<\|end\|>` | `<\|end\|>` |
| **Zephyr** | TinyLlama-1.1B-Chat-v1.0, Zephyr-7B | `<\|user\|>\ncontent</s>\n<\|assistant\|>\ncontent</s>` | `</s>` |
| **Alpaca-flat** | (not a real model template) | `### Instruction:\n...\n### Input:\n...\n### Response:\n...` | (EOS) |

The instructor's own data prep uses two of these — `[INST] {question} [/INST] {answer}` in notebook cell 17 (Mistral/Llama-2 style), and `### Instruction:` / `### Response:` in cell 31 (Alpaca-flat). Neither is the template of the model being trained. This is the single most instructive thing in the whole notebook, and §10.1 dissects it.

#### 4.3.2 The API — how to inspect and use a template

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 1. SEE the template (it is Jinja2; read it before you trust it)
print(tok.chat_template)

# 2. Render a conversation WITH tokenisation (the normal path)
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Explain the mechanism of action of Metformin."}],
    tokenize=True,
    add_generation_prompt=True,      # append the assistant header so the model continues
    return_tensors="pt",
)
print(ids.shape)                      # e.g. torch.Size([1, 24])

# 3. Render WITHOUT tokenising (so you can read the string, count, or edit it)
print(tok.apply_chat_template(
    [{"role": "system", "content": "You are a pharmacology assistant."},
     {"role": "user",   "content": "Explain the mechanism of action of Metformin."},
     {"role": "assistant", "content": "Metformin activates AMPK ..."}],
    tokenize=False,
))
```

`add_generation_prompt=True` is the flag that decides whether the rendered string *ends with the assistant's opening header* (inference: the model must continue) or ends after the assistant's content (training: the assistant turn is present and will be masked/unmasked). Getting this backwards is a silent bug — the model sees `Explain X.` with no assistant header and has to guess that it should answer.

**Token counts, measured** (illustrative, Qwen2.5 tokenizer, the same conversation rendered three ways):

| Rendering | Tokens | Notes |
|---|---|---|
| Raw user text only | 12 | No roles, no template — what a naive `tokenizer(prompt)` gives you |
| ChatML with `add_generation_prompt=True` | 24 | `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` |
| ChatML with `system` + assistant turn | 61 | Training string; ~25 tokens are the response |

The template adds **~12 tokens of overhead per turn**. For a conversation with 5 turns that is 60 tokens of positional budget consumed by scaffolding — one of the reasons long multi-turn SFT is expensive, and why packing (§4.5.3) matters.

#### 4.3.3 The failure mode: train with one template, serve with another

The mechanism: the model learns `p(next token | preceding tokens)`. The role markers are tokens it must condition on. If at training time the model always saw `<|im_start|>assistant\n` before an answer, it learns that *this specific token sequence* is the strongest cue to "start answering in the assistant style". At serving time, a different harness sends `### Response:\n` or nothing at all. The cue is absent. The model falls back to the behaviour that is unconditional — typically continuation of the prompt, or a mush of formats it saw in pretraining.

This is why a model can look excellent in the SFT notebook (same template) and broken behind an API (different template). **It is not a "quality" problem; it is an interface mismatch, and no amount of extra training data fixes it.**

The mismatch taxonomy, worst to least obvious:

| Mismatch | Symptom | Which side is usually wrong |
|---|---|---|
| Completely different markers (`[INST]` vs ChatML) | Model ignores the instruction; continues the prompt; or emits the raw template text | Serving harness applies no template, or applies the wrong one |
| Same markers, missing `add_generation_prompt` | Model answers then keeps generating fake user turns | Serving harness forgot the assistant header |
| Wrong BOS handling (`<s>` present at train, absent at serve) | Degraded but still functional; quality loss of a few % | Tokenizer loaded with `add_special_tokens=False` on one side |
| Missing `</s>`/`<\|eot_id\|>` in the training response spans | Model never stops; rambles past the answer | Dataset rows built by string concatenation without the EOS |
| System prompt present at train, absent at serve | Model behaves as if it has no persona; ignores house style | Serving harness drops the system message (some servers only accept user/assistant) |
| Whitespace differences (`\n\n` vs `\n`) | Subtle quality drop, more format drift | Hand-rolled templates |

**The rule:** there is exactly one place in your codebase that renders prompts, and both the training script and the inference server import it. In practice:

```python
# prompt_contract.py  — the single source of truth, imported by training AND serving
from transformers import AutoTokenizer

_TEMPLATE_PATH = "templates/acme_pharma_v3.jinja"

def get_tokenizer(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    tok = AutoTokenizer.from_pretrained(model_id)
    with open(_TEMPLATE_PATH) as f:
        tok.chat_template = f.read()        # pin the template as a versioned artefact
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"              # REQUIRED for the manual mask in §4.4.2
    return tok

def render_inference(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        ([{"role": "system", "content": system}] if system else []) +
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
```

Ship `templates/acme_pharma_v3.jinja` alongside the adapter, version it with the dataset hash, and assert in CI that the server's rendered string equals the trainer's rendered string for a golden conversation.

#### 4.3.4 Inspecting a template and adding your own

```python
# Which template is actually in this checkpoint?
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(tok.chat_template[:400])

# Does the template support tools / system / generation markers?
print("tools" in tok.chat_template, "system" in tok.chat_template,
      "generation" in tok.chat_template)   # 'generation' -> assistant_only_loss works

# Render with a system prompt and see the exact bytes
print(repr(tok.apply_chat_template(
    [{"role": "system", "content": "S"},
     {"role": "user", "content": "U"},
     {"role": "assistant", "content": "A"}],
    tokenize=False)))
# '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>...'
```

Adding a custom template (needed when the base model has none, e.g. a freshly continued-pretrained checkpoint from CS-12):

```jinja
{# templates/acme_pharma_v3.jinja — ChatML plus an explicit generation block #}
{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
```

```python
tok.chat_template = open("templates/acme_pharma_v3.jinja").read()
tok.save_pretrained("./merged-model")     # persists into tokenizer_config.json
```

The `{% generation %}` markers are what `assistant_only_loss=True` looks for (TRL ≥ 0.12). Without them, TRL falls back to a warning and no mask — a silent failure with a very quiet log line.

> **Beyond the video:** the special tokens must exist in the vocabulary *and* be marked special, or they get split into pieces and lose their meaning. Check with:
> ```python
> for t in ["<|im_start|>", "<|im_end|>"]:
>     ids = tok.encode(t, add_special_tokens=False)
>     print(t, ids, tok.convert_ids_to_tokens(ids))
> # <|im_start|> [151644] ['<|im_start|>']   <- good: one token
> # <|im_start|> [27, 91, ...]              <- bad: it was split; the marker carries no signal
> ```
> If a marker splits, `tokenizer.add_special_tokens({"additional_special_tokens": [...]})` + `model.resize_token_embeddings(len(tokenizer))` fixes it — and now your newly added embeddings are randomly initialised, so they need a warmup phase with the rest of the model frozen. That is a CS-12 topic, not a CS-13 one, but it is why "I added a custom token and my model got worse" happens.

---

### 4.4 Loss masking / prompt masking — the single most important implementation detail

#### 4.4.1 The mechanism at the tensor level

A causal LM forward pass returns `logits` of shape `[batch, seq_len, vocab]`. The loss shifts them by one:

```python
# what Trainer does internally, verbatim in spirit
shift_logits = logits[..., :-1, :].contiguous()   # predict position t+1 from position t
shift_labels = labels[..., 1:].contiguous()
loss = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,          # <- positions with label -100 contribute ZERO to both
)                               #    the numerator and the denominator of the mean
```

Three consequences that are worth being able to state in an interview:

1. **`-100` positions are removed from the denominator too.** The loss is a mean over *non-ignored* positions. So masking changes both what is learned and the *scale* of the reported loss. Two runs with the same data and different masks are not comparable by loss value. Ever.
2. **The shift is why the first token of the response cannot be predicted from the response itself.** Position `t` of the logits predicts `shift_labels[t+1]`. If you mask everything up to and including the assistant header, the first *trained* label is the first response token, predicted from the last header token. That is exactly right.
3. **Attention is not affected by labels.** The masked prompt tokens still act as context — they are in `input_ids` and attended to. You are not deleting the prompt; you are deleting *the gradient that teaches the model to produce it*.

#### 4.4.2 The manual mask — and the notebook's two implementations

**Implementation A — the notebook's default (cell 37), no mask at all:**

```python
def tokenize_fn(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()   # <-- trains on EVERYTHING, pads included
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
```

**Implementation B — the notebook's masking variant (cell 54):**

```python
def tokenize_and_mask(example):
    text = example["text"]
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    input_ids = enc["input_ids"]

    response_marker = "### Response:"
    response_start = text.find(response_marker)
    if response_start != -1:
        response_token_start = len(tokenizer(text[:response_start])["input_ids"])
    else:
        response_token_start = 0        # marker missing -> masks NOTHING (silent no-op)

    labels = input_ids.copy()
    labels[:response_token_start] = [-100] * response_token_start
    enc["labels"] = labels
    return enc
```

**The instructor's verdict [52:36]–[52:58], verbatim:**

> *"Both training you can do — you can try out with this one as well as the previous one. You can check which one is giving you the effective result. According to me, the previous one is a good one because here actually my model is learning onto the entire string... Now here, till response we are masking and after that only we are allowing the model to learn through some tokens."*

> **Correction:** this recommendation is backwards, and it is the most consequential error in the video. **The industry default is to mask the prompt.** Three reasons, in order of force:
> 1. **Gradient dilution.** On the `pharma_instruction_data.jsonl` rows, ~44% of tokens are prompt. On a realistic dataset with retrieved context (a 2,000-token passage and a 100-token answer), it is >95%. You are spending 95% of your compute teaching the model to reproduce passages it will be *given* at inference.
> 2. **It trains a competing objective.** An unmasked model learns `p(prompt)`, i.e. it learns to *generate prompts*. The instructor is right that the model "learns the entire string" — that is the problem. Prompt generation is the highest-frequency pattern in the data (every row has one) and it is the behaviour you least want.
> 3. **It breaks the length distribution.** Prompt tokens are cheap to predict (highly predictable, low loss) and response tokens are expensive. A mixed loss means early stopping triggers on the prompt-fit, which happens first, while the response is still being learned.
>
> When the instructor says "I have seen in many places people are following that" [53:00] about *masking*, that is because masking is the standard. He has the observation right and the conclusion inverted.
>
> That said, he is not *crazy*: for tiny datasets (<100 rows) with short prompts and short answers, and models that already chat, unmasked training does sometimes produce more coherent prompt-following, because the model is rehearsing the whole in-context pattern at every step. It is a degenerate case of replay. It is not a strategy you should build a pipeline on.

**The third implementation — the one to actually write:**

```python
def tokenize_with_mask(example, tokenizer, max_len=1024):
    """Template-aware masking: mask everything up to and including the assistant
    header, keep the response (and its EOS), and mask padding. Works for ChatML,
    Llama-3, Zephyr, and anything with a textual assistant header."""
    messages = example["messages"]

    # 1. Render the FULL conversation, but let the template mark the assistant spans.
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 2. Render the conversation truncated at the LAST assistant header, so we can
    #    count exactly how many tokens precede the response we want to train on.
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    enc_full = tokenizer(full, truncation=True, max_length=max_len,
                         padding="max_length", return_tensors=None,
                         add_special_tokens=False)
    # add_special_tokens MUST match the call above. `full` is tokenized with the
    # default (True) in the naive version, while `prompt_only` is tokenized with
    # False — so on any tokenizer that prepends a BOS, n_prompt is short by one and
    # the whole mask shifts by a token, silently supervising the last prompt token.
    n_prompt = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])

    labels = list(enc_full["input_ids"])
    # mask prompt
    labels[:n_prompt] = [-100] * n_prompt
    # mask PADDING — the bug the notebook ships (see §14, row 1)
    labels = [-100 if a == 0 else l for l, a in zip(labels, enc_full["attention_mask"])]

    enc_full["labels"] = labels
    return enc_full
```

> **No extra pad-token mask is needed.** An earlier draft of this function ended with
> `if tokenizer.pad_token_id is not None: pass` — a no-op stub. The genuine worry behind it
> is that `pad_token == eos_token` (true for Llama-3, Qwen, and most modern families), so a
> naive `labels[labels == pad_token_id] = -100` would also mask every real EOS and teach the
> model never to stop. The `attention_mask` line above is the correct fix: it masks pads by
> *position*, not by *token id*, so real EOS tokens inside the response survive. That is why
> the id-based approach is the classic "model never stops generating" bug and this one isn't.

**The same example, as a mask tensor.** Rendered with TinyLlama's SentencePiece tokenizer (token *boundaries* below are illustrative — your exact ids and counts will differ, the structure will not):

| idx | token | `attention_mask` | labels (no mask, notebook A) | labels (prompt-masked, notebook B) | labels (correct: prompt + header + pad masked) |
|---|---|---|---|---|---|
| 0 | `<s>` | 1 | `<s>` | -100 | -100 |
| 1 | `▁###` | 1 | 1 | -100 | -100 |
| 2 | `▁Instruction` | 1 | 2 | -100 | -100 |
| 3 | `:` | 1 | 3 | -100 | -100 |
| 4 | `\n` | 1 | 4 | -100 | -100 |
| 5 | `Explain` | 1 | 5 | -100 | -100 |
| … | … | … | … | -100 | -100 |
| 21 | `###` | 1 | 21 | -100 | -100 |
| 22 | `▁Response` | 1 | 22 | -100 | -100 |
| 23 | `:` | 1 | 23 | -100 | -100 |
| 24 | `\n` | 1 | 24 | -100 | -100 |
| 25 | `Met` | 1 | 25 | 25 | 25 |
| 26 | `formin` | 1 | 26 | 26 | 26 |
| … | … | … | … | … | … |
| 55 | `.` | 1 | 55 | 55 | 55 |
| 56 | `</s>` | 1 | 56 | 56 | 56 |
| 57 | `<pad>` | **0** | **`<pad>` ← BUG** | **`<pad>` ← BUG** | **-100** |
| … | `<pad>` × 454 | 0 | `<pad>` × 454 | `<pad>` × 454 | -100 |
| 511 | `<pad>` | 0 | `<pad>` | `<pad>` | -100 |

Read the last three columns' row 57–511 carefully:

- **Column "no mask" (the notebook's default):** 455 of 512 label positions are `<pad>`, and 25 of the remaining 57 are prompt. Only **32 of 512 positions (6.3%)** supervise the actual answer. The model spends 93.7% of its gradient learning to emit the prompt and then to emit `<pad>` forever.
- **Column "prompt-masked" (the notebook's variant):** the prompt is gone, but the **padding is still trained** — 455 positions of `<pad>`. The mask fixed 44% of the problem and left 89% of the actual waste in place. Since `pad_token = eos_token` for TinyLlama, this is literally training the model to emit end-of-sequence tokens in long runs. That is a plausible contributor to the incoherent generations shown in the video's final comparison [55:00]–[55:05].
- **Column "correct":** the loss is computed over 32 positions, all of them the answer.

The generalisable lesson: **`attention_mask == 0` must imply `labels == -100`.** One assertion:

```python
# the assertion that catches the notebook's bug in one line
def assert_no_pad_in_loss(tok, n=200):
    for i in range(n):
        ex = tokenize_with_mask(dataset[i], tok)
        for l, a in zip(ex["labels"], ex["attention_mask"]):
            assert a == 1 or l == -100, f"row {i}: pad position trained as a target"
    print("ok: no padding in the loss")
```

#### 4.4.3 The framework-native ways to do it

**TRL `SFTConfig(assistant_only_loss=True)`** (TRL ≥ 0.12) uses the `{% generation %}` markers in the chat template to build the mask. It is the only method that is correct across multi-turn conversations, tool calls, and templates that place a generation header *inside* a turn.

```python
from trl import SFTTrainer, SFTConfig

cfg = SFTConfig(
    output_dir="out",
    assistant_only_loss=True,        # mask everything that is not an assistant span
    packing=False,                   # keep off while debugging masks
    max_length=1024,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)
trainer.train()
```

If the tokenizer's template lacks `{% generation %}`, TRL raises `ValueError: The chat template ... does not contain the generation tag` in recent versions — older versions warned and silently did **not** mask. Pin your TRL version and test with the assertion above.

**Unsloth's `train_on_responses_only`** works with any template by string-matching the instruction and response parts:

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,
                                         target_modules=["q_proj","k_proj","v_proj","o_proj",
                                                         "gate_proj","up_proj","down_proj"])
trainer = ...  # a standard TRL SFTTrainer

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
# verify: the first non--100 label must be the first response token
batch = next(iter(trainer.get_train_dataloader()))
row = batch["labels"][0]
print(row[row != -100][:8])        # should be the response's first tokens, not the prompt's
```

**LLaMA-Factory**: no code — set `train_on_prompt: false` in the dataset entry's preprocessing or use the `mask_history` option for sharegpt multi-turn.

**Axolotl**: `train_on_inputs: false` (and `roles_to_train: ["assistant"]` for chat templates).

**The manual `-100` route is still legitimate** when your data is completion-only, when your template is exotic, or when you need per-token weighting. Just never ship it without the padding assertion.

#### 4.4.4 When you *should* train on the prompt (the real exceptions)

| Situation | Why it helps | How much of the prompt |
|---|---|---|
| **Continued pretraining mixed into SFT** (replay, CS-12) | Replay text is not a prompt/response pair; masking it would leave nothing to learn | 100% (it is the whole example) |
| **Very small dataset (<200 rows) on a base model that has never seen the format** | Whole-string rehearsal accelerates format adoption before the response-only signal has enough support | 100%, for the first ~10% of steps, then switch to masked |
| **Teaching a model to *emit* a template** (rare; e.g. you are deliberately training a prompt generator) | The prompt is the target | 100%, by design |
| **Prefix-LM / fill-in-the-middle objectives** | The objective is bidirectional on the prefix | Prefix unmasked by design |
| **Chain-of-thought distillation where the reasoning trace is the target and the question is the prompt** | You want the trace trained; the question is *not* what you want trained. → mask the question, train the trace. This is still masking, but it is worth stating because people confuse "train the CoT" with "train the prompt" | 0% of question, 100% of trace |
| **Unmasked SFT as a data-quality probe** | If unmasked training *helps* materially, it usually means your prompt distribution is carrying signal your responses lack (i.e. your answers are degenerate). Treat it as a diagnostic, not a strategy | — |

Everything else: mask. If you want the "the model learns the whole row" benefit, get it from **replay of high-quality general instruction data** (where the response is also masked correctly) rather than from unmasking your domain rows.

---

### 4.5 Memory, compute, and the packing question

#### 4.5.1 The VRAM arithmetic, worked

Training memory decomposes into five terms. For a model with `N` parameters, batch size `B`, sequence length `S`, `L` layers, hidden size `d`, and `a` attention heads:

| Term | fp16 full FT | LoRA (fp16 base) | QLoRA (4-bit base) |
|---|---|---|---|
| Weights | `2N` bytes | `2N` | `0.5N` (NF4) + ~0.13N (quant constants, double-quant) |
| Gradients | `2N` | `2 × N_adapter` (~0.005N) | `2 × N_adapter` |
| Optimizer (AdamW: fp32 m + v) | `8N` | `8 × N_adapter` | `8 × N_adapter` |
| fp32 master weights | `4N` | — | — |
| Activations (no ckpt) | `≈ B · S · L · d · (≈ 34) bytes` | same | same (~20% less: bf16 only) |
| Activations (grad ckpt, recompute) | `≈ 2N + B·S·L·d·2` | same | same |

**Worked example — 7B model, `S=2048`, `B=1`, `L=32`, `d=4096`:**

- Full FT fp16 with AdamW: `2N + 2N + 8N + 4N = 16N` = 16 × 7e9 = **112 GB static**. Activations without checkpointing: `1 × 2048 × 32 × 4096 × 34 ≈ 9.1 GB`; with gradient checkpointing `≈ 2N + 2048×32×4096×2 ≈ 14 GB + 0.5 GB`. **≈ 127 GB with checkpointing → 2× A100-80GB minimum, realistically 4× for headroom.** Without checkpointing you are at ~121 GB + 9 GB and you will OOM on 2×80 at `B=2`.
- LoRA (r=16, q/k/v/o + MLP, ~0.6% of params = 42M trainable): `14 GB weights + 0.084 GB grads + 0.34 GB optimizer + 14 GB (2N for checkpoint recompute) ≈ 29 GB`, plus activations. **Fits on 1× A100-40GB at `S=2048, B=4`, and on 1× 24 GB card at `S=1024, B=2`.**
- QLoRA: `3.5 GB (4-bit) + 0.42 GB adapter state + activations ≈ 9–12 GB`. **Fits on a 12 GB card; comfortably on a free Colab T4 (16 GB) at `S=2048, B=1, ga=16`.**

The notebook's run is far below any of these: TinyLlama-1.1B, LoRA r=8, `max_length=512`, `B=1`. Roughly 2.2 GB of weights + adapter state — it fits on CPU.

#### 4.5.2 Sequence length is a quadratic tax

Flash-Attention makes attention compute linear in `S`, but the *activation* term and the MLP term stay linear in `B·S`, and if you disable flash-attention, attention memory is `O(S²)`. Doubling `S` from 1024 to 2048 doubles compute and activation memory per example, and *halves* the number of examples per batch at fixed memory. Practical guidance:

| Your data's p99 token length | `max_seq_length` to set | Why |
|---|---|---|
| 180 | 256 | 30% of compute saved vs 512; nothing is truncated |
| 600 | 768 | headroom for the longest 0.5% |
| 2,400 | 2,560–3,072 | round up to a power-of-two-friendly bucket |
| 12,000 (long-context QA) | 16,384 + packing | here you need flash-attn and checkpointing; consider a long-context base |

Measure it, do not guess:

```python
import numpy as np
lengths = [len(tokenizer(render(row)).input_ids) for row in ds]
print({p: int(np.percentile(lengths, p)) for p in (50, 90, 95, 99, 99.5, 100)})
```

#### 4.5.3 Padding vs packing

Padding wastes compute; packing recovers it. With the notebook's data (mean length ~57 tokens, window 512), **89% of every forward pass is padding**. On a real dataset with 20-turn conversations where the assistant turns are short, packing typically gives **2–5× throughput**.

| | Padding | Packing |
|---|---|---|
| Compute efficiency | `mean_len / max_len` (often 20–50%) | ~95–99% |
| Implementations | default | TRL `packing=True`, HF `DataCollatorWithFlattening`, `neat_packing` in Axolotl |
| Position ids | natural | **must be reset per packed sequence** |
| Attention | full causal within example | full causal within example; must not cross examples |
| Loss masking | per token | per token; must survive the concatenation |
| Eval comparability | straightforward | harder to map a loss back to an example |
| Risk | none | cross-contamination if position ids are wrong |

> **Beyond the video — packing + flash-attention, precisely.** Naively concatenating sequences and running a plain causal attention mask lets token 3 of sequence B attend to sequence A. The fix has two halves: (1) pass per-sample `position_ids` that restart at 0 for each packed sequence — flash-attention v2's `flash_attn_varlen_func` uses `cu_seqlens` derived from position-id discontinuities to build a block-diagonal mask; (2) in HF, `trl`'s `packing=True` handles this via `padding_free` collation and, from `transformers>=4.44`, `DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)` returns the `position_ids`+`cu_seq_lens` pair that `flash_attention_2` consumes directly. If you implement packing yourself and forget the reset, your loss will look *better* than it should (the model gets free context from unrelated examples) and your production quality will be worse than your offline eval. The symptom is a model that is fine on single-turn prompts and erratic on anything long. Also note that with packing, a row whose response is fully truncated still occupies a slot, so re-check the "no example is 100% masked" assertion after enabling it. The full recipe — `attn_implementation="flash_attention_2"`, `padding_free=True`, `packing=True`, and the `position_ids` check that proves the reset happened — is in CH-13 §5.4.

---

### 4.6 Data curation — the part that determines success more than any hyperparameter

The instructor identifies the problem correctly and bluntly [23:20]–[24:14]:

> *"Now the biggest question: how you are going to generate this instruction data? Because in every company — any XYZ company, any ABC company — everywhere you will NOT find this kind of data."*

and then gives the three sourcing options: manual authoring, expert/human annotation, and LLM-synthesised data [24:14]–[25:43]. What he does not give you is the part that decides whether the model works: **how many, how diverse, and how filtered.**

#### 4.6.1 Quantity — the LIMA hypothesis vs the 100k+ reality

The empirical anchors:

| Dataset / study | Examples | Base model | Outcome |
|---|---|---|---|
| **LIMA** (Meta, 2023) | **1,000** curated | LLaMA-65B | 43% win/tie vs GPT-4, 58% vs Bard — no RLHF |
| **AlpaGasus** (Chen et al., 2023) | **9,000** (filtered from Alpaca's 52k by GPT-4 quality score) | LLaMA-7B/13B | Beat the full 52k Alpaca on 5 of 6 benchmarks — *with 17% of the data* |
| **Deita** (Liu et al., 2023) | **6,000** (quality+complexity+diversity scorer) | LLaMA-7B | Beat 100k-example baselines (UltraChat, WizardLM, ShareGPT) on MT-Bench |
| **Alpaca** (Stanford, 2023) | 52,000 synthetic | LLaMA-7B | First cheap public instruct recipe |
| **FLAN v2 / collection** (Google) | ~15M across 1,800 tasks | T5 / PaLM | Best-in-class task generalisation at pretraining scale |
| **Tulu 3 SFT mixture** (AI2, 2024) | ~500k+ (with PersonaHub synthetic) | Llama-3.1-8B/70B | Current open SOTA-class SFT mixture |
| **UltraChat 200k** | 200k multi-turn | — | Standard conversational mixture component |

**The reconciliation.** LIMA and AlpaGasus do not say "use 1,000 examples". They say: *for the purpose of teaching format and behaviour to a model that already has the capability, curated small data beats uncurated large data.* Scale it up and the same principle holds with a twist: as you go past ~10k examples, you are no longer buying behaviour — you are buying **coverage** (more task types, more domains, more refusal boundaries, more edge cases) and **robustness** (the same behaviour under more surface variation).

Practical tiers, with what each buys you:

| Tier | Size | What it is for | Realistic outcome | Failure if you stop here |
|---|---|---|---|---|
| **Smoke test** | 50–200 | Does my pipeline run? Does the loss go down? Does the template round-trip? | Model adopts the format ~80% of the time | No breadth; unusable in production |
| **Behavioural core (LIMA tier)** | **1,000–5,000** | Format compliance, tone, task-type adoption, refusal shape | 90–95% format compliance on in-distribution prompts | Breaks on prompt styles you did not include |
| **Product tier** | 5,000–50,000 | The above + edge cases + long-tail tasks + multi-turn | 95%+ format compliance, handles 90% of production traffic | Costs more to curate; needs a real filter pipeline |
| **Capability tier** | 50,000–500,000 | Broad assistant behaviour, agentic/tool use, multi-domain | Competitive with open instruct models on MT-Bench | Diminishing returns per dollar; needs dedup at scale |
| **Frontier mixture** | 1M+ | General assistants from base models | Where Tulu 3 / OLMo 2 / Nemotron live | You are now running a data organisation |

**Rule of thumb for a domain project:** **2,000–10,000 high-quality, domain-specific rows** is the sweet spot for the enterprise case. Below 1,000 you are exposed to template over-fit and idiosyncrasy; above ~30,000 on a single domain, you are mostly re-teaching the same behaviour with more noise unless you have deliberately engineered diversity (task type, length, difficulty, format).

**Compute reality for a small dataset.** This notebook's dataset is 5 rows. With `B=1`, `ga=8`, and 3 epochs, `len(dataloader) = 5`, so `steps_per_epoch = ceil(5/8) = 1` and **total optimizer steps = 3**. That is why the saved checkpoint is literally `checkpoint-3` [49:05]. Three gradient updates. The LR schedule is meaningless, warmup is meaningless, and the model has learned essentially nothing except a nudge in the direction of the response format — which is exactly what the video's final comparison shows (the instruction-tuned model still emits the same pharma-flavoured continuation, and in the side-by-side it produces "expect output test case test case" [55:00]). The instructor is honest about this: *"my data set is very small, my model is also very small, so it might not give the correct answer"* [54:00].

#### 4.6.2 Diversity — the axis people forget

LIMA's curation budget went to **diversity of task type**, not per-example polish. If you only have 1,000 rows, each row has to earn its place.

Three diversity axes, all measurable:

**(a) Task-type coverage.** Enumerate the tasks your product needs and count rows per task. If one task is >25% of the dataset, it will dominate the behaviour.

```python
from collections import Counter
import re

TASK_PATTERNS = {
    "explain":   r"\b(explain|describe|how does|mechanism)\b",
    "summarise": r"\b(summar|tl;dr|condense)\b",
    "extract":   r"\b(extract|list|identify|find all)\b",
    "classify":  r"\b(classify|categor|label|is this)\b",
    "generate":  r"\b(write|draft|compose|generate)\b",
    "transform": r"\b(translate|convert|rewrite|reformat)\b",
    "reason":    r"\b(why|calculate|compare|which is better)\b",
    "refuse":    r"\b(ignore previous|jailbreak|system prompt)\b",
}
def task_type(instr: str) -> str:
    for t, p in TASK_PATTERNS.items():
        if re.search(p, instr, re.I):
            return t
    return "other"

print(Counter(task_type(r["instruction"]) for r in ds).most_common())
# [('extract', 312), ('explain', 208), ('summarise', 121), ... , ('other', 39)]
```

**(b) Instruction-verb diversity.** Count distinct leading verbs. Self-Instruct formalised this: a generator that produces "Summarize the following..." 4,000 times has produced one example with 4,000 spellings.

```python
verbs = Counter(r["instruction"].split()[0].lower() for r in ds)
print(f"distinct leading verbs: {len(verbs)} over {len(ds)} rows")
print(verbs.most_common(10))
# distinct leading verbs: 43 over 1000 rows   <- healthy
# distinct leading verbs: 6 over 1000 rows    <- your synthetic generator is in a rut
```

**(c) Length diversity.** Self-Instruct's own guidance is explicit: *"1 instruction should be diverse in length."* A dataset where every response is 40–60 tokens produces a model with a fixed output length — which then truncates or pads every real answer.

```python
resp_lens = [len(tokenizer(r["output"]).input_ids) for r in ds]
print({p: int(np.percentile(resp_lens, p)) for p in (5, 25, 50, 75, 95, 100)})
# {5: 21, 25: 46, 50: 78, 75: 149, 95: 402, 100: 1180}
```

Target: a **10–20× spread** between p5 and p95, with a deliberate tail of very short answers ("Yes." / "No, because §4 applies.") and a few very long ones. The short ones teach the model to stop; the long ones teach it to structure. Models trained on a narrow length band develop **length rigidity** — the most common "our fine-tune is weirdly verbose for simple questions" complaint.

Add two more diversity axes that are cheap and high-value:

| Axis | Why | How to check |
|---|---|---|
| **Language / register** | If 5% of your traffic is Spanish or terse, include 5% Spanish or terse rows | language-ID histogram (`fastText lid.176`) |
| **Refusal / boundary** | The model must know what to decline. Without boundary rows, refusal is decided by the base model's RLHF, which may be wrong for your domain | count rows matching `TASK_PATTERNS["refuse"]`; target 2–5% |
| **Multi-turn** | Any conversational product needs ≥20% multi-turn rows | count rows with >2 messages |
| **Negative / "I don't know"** | The single most valuable row type and the least common. A model with no "I don't know" examples will hallucinate rather than abstain | target 3–8% of rows |

#### 4.6.3 Quality filters

Run these in order. Each is cheap; the order matters because dedup before judging saves money.

| # | Filter | Rule of thumb | Tool | Catches |
|---|---|---|---|---|
| 1 | **Schema validation** | Required keys present, types correct, non-empty response | `pydantic` | Broken rows that crash training at step 900 |
| 2 | **Length filter** | Drop responses <2 tokens or >`max_seq_length` | `len(tokenizer(x))` | Truncated targets, one-word non-answers |
| 3 | **Exact dedup** | Hash the normalised `(instruction, output)` pair | `hashlib`, `datasets.unique` | Copy-paste storms |
| 4 | **Near dedup** | MinHash-LSH, Jaccard ≥ 0.8 on 5-grams within the instruction field | `datasketch`, `text-dedup` | Paraphrase farming |
| 5 | **Decontamination vs eval** | Drop any row sharing a 13-gram with any eval prompt or reference answer | `nltk` n-grams, `text-dedup` | Eval-score fiction |
| 6 | **Repetition detection** | Drop responses whose self-BLEU or compression ratio indicates a loop | `gzip` ratio, `self-BLEU` | `"Yes, yes, yes, ..."` |
| 7 | **Refusal / meta removal** | Drop "As an AI language model...", "I cannot", "I'm sorry" unless the row is *intended* as a refusal | regex + classifier | The refusal machine (§4.9.3) |
| 8 | **LLM-as-judge quality score** | 1–5 rubric on helpfulness / correctness / relevance; keep ≥4 | GPT-4o / Claude / a local 70B judge | Plausible-looking wrong answers |
| 9 | **Complexity score** | 1–5 on reasoning depth; **deliberately balance** the distribution | Deita-style scorer | An all-easy dataset produces a shallow model |
| 10 | **PII / secret scrub** | Regex + `presidio` for emails, phones, keys, patient IDs | `presidio`, custom | Legal liability in the weights (irreversible) |
| 11 | **Language ID** | `fastText lid.176` — drop rows in languages you did not intend | `fasttext` | Mixed-language training collapse |

**Decontamination is not optional.** The GPT-3 paper's standard: compute all 13-grams of each training example, drop the example if any 13-gram appears in any evaluation set. Thirteen tokens is long enough that accidental collision is negligible and short enough to catch a reworded sentence.

```python
def ngrams(tokens, n=13):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

EVAL_NGRAMS = set()
for row in eval_set:
    EVAL_NGRAMS |= ngrams(tokenizer(row["prompt"]).input_ids, 13)
    EVAL_NGRAMS |= ngrams(tokenizer(row["answer"]).input_ids, 13)

def is_contaminated(row):
    return bool(ngrams(tokenizer(row["instruction"] + row["output"]).input_ids, 13) & EVAL_NGRAMS)

before = len(ds)
ds = ds.filter(lambda r: not is_contaminated(r))
print(f"dropped {before - len(ds)} contaminated rows ({100*(before-len(ds))/before:.2f}%)")
# 0.0% -> suspicious. 15%+ -> your eval set is inside your training set, and every
# metric you have reported so far is fiction.
```

**The "one great example beats ten mediocre ones" principle, with evidence.** AlpaGasus filtered 52k Alpaca rows to 9k using GPT-4 quality scores and *improved* every benchmark — the removed 43k rows were not neutral, they were actively harmful (they taught the model to produce the style of the low-quality rows). Deita's 6k beat 100k+ baselines. The mechanism is straightforward: SFT is a **mean-seeking** estimator. Every bad row pulls the output distribution toward bad outputs, and there is no loss term that says "this row is wrong". There is no negative example in SFT. The only defence is the filter.

Quantitatively: if 20% of your rows are mediocre, roughly 20% of your gradient is pointing at mediocre behaviour, and the model has no way to tell which is which. Removing them is a *higher* expected-value action than doubling your learning rate sweep.

> **Beyond the video:** the Deita scorer is worth implementing because it is 40 lines and it operationalises "quality" as three measurable scores:
> ```python
> # Deita-style: score each row on complexity, quality, and diversity; select a subset
> COMPLEXITY_PROMPT = "Score the reasoning depth of this instruction from 1 (trivial) to 5 (requires multi-step reasoning):\n{instruction}\nScore:"
> QUALITY_PROMPT    = "Score the correctness and helpfulness of this response from 1 (poor) to 5 (excellent):\nQ: {instruction}\nA: {response}\nScore:"
> # Then: select a target N by iterating rows in descending (complexity + quality),
> # greedily adding a row only if its embedding distance to all selected rows exceeds
> # a threshold (that is the diversity term).
> ```
> The key insight is the *interaction*: high-quality-but-similar rows are redundant, and diverse-but-low-quality rows are poison. You want the high-complexity, high-quality, low-redundancy frontier. Selecting 6k rows this way outperformed 100k+ raw rows in the paper's own evaluation.

#### 4.6.4 Synthetic data generation for bootstrapping

The instructor describes the mechanism [25:00]–[25:43]: *"here to LLM only you can provide some prompt, some guides, to the GPT model, to the Claude model... those models only can prepare your instruction data set. Let's say you have a plain text, this text you are feeding to your GPT model and you are saying 'can you prepare some question and answer from this particular text?'"* That is the standard pipeline, and it is how Alpaca, UltraChat, WizardLM, and the Tulu 3 mixture were made.

The four families, with their failure modes:

| Method | How it works | Best for | Silent failure |
|---|---|---|---|
| **Seed + self-instruct** | 175 hand-written seeds → model generates new instructions → filter by similarity to existing → repeat | Task-type breadth | Mode collapse: by round 4 everything looks like round 1. Fix: cap rounds at 3–4, resample seeds |
| **Document-to-QA** | Chunk a corpus, prompt "write 3 Q&A pairs answerable only from this chunk" | Domain knowledge, RAG-adjacent data | The model writes questions whose answers are *in the chunk*, so the answer is a copy → teaches copying. Fix: generate the question from the chunk, then answer *with the chunk hidden*, and discard pairs you cannot answer |
| **Evol-Instruct** | Take a seed instruction and evolve it: add constraints, deepen, concretise, increase reasoning steps | Difficulty and complexity | Evolved instructions drift into nonsense ("write a 12-paragraph haiku in Latin") |
| **Persona / Magpie-style** | Prompt a *pre-trained* model with only the chat-template prefix and let it generate the user turn, then answer | Cheap, naturally diverse, no seed needed | Generates user turns that leak the template; needs a strong classifier to filter |

**The three-part filter for any synthetic data** (all three are required):

1. **Answerability.** With the source text withheld, can the reference answer still be judged correct? Use an LLM judge with the source removed. If the judge says "cannot determine", the row teaches the model to bluff.
2. **Uniqueness.** Embedding-dedup against the existing pool (cosine ≥ 0.9 → drop). Synthetic generators produce near-duplicates by the thousand.
3. **Instruction–response non-overlap.** Token-level Jaccard between instruction and response < ~0.3. Above that, the row is a copy task, and you are training a summariser you did not ask for.

> **Beyond the video — the licensing reality of synthetic data.** If you generate with a commercial API (OpenAI, Anthropic, Google), read the terms before you train a model you intend to ship. The canonical case: **Alpaca** was generated with `text-davinci-003`, and OpenAI's terms of use at the time prohibited using outputs "to develop models that compete with OpenAI". Stanford replaced the dataset (see the `text-davinci-003` → `gpt-4` "Alpaca-GPT4" lineage, and the "cleaned alpaca" descendants), but thousands of downstream projects had already shipped on the original. Current status, as of the 2025-era tooling this book describes: OpenAI's business terms permit fine-tuning *their* models on *your* data and treat API outputs as yours to use, but the "compete with us" clause has been read by many legal teams as restricting training third-party base models on their outputs. Anthropic's and Google's terms differ again. **The practical rule: for anything commercial, generate your synthetic data with a model you have the right to train on — typically an open-weights model you run yourself — or get legal sign-off in writing.** Secondary datasets inherit this risk: "Alpaca-cleaned" is cleaned for *correctness*, not relicensed. §4.6.6 has the dataset-licence table.

#### 4.6.5 The curation pipeline, end to end

```text
raw sources                 →  normalise        →  filter              →  score           →  select        →  freeze
──────────────────────────     ──────────────      ──────────────────     ─────────────     ────────────      ────────
PDFs / tickets / CRM        →  messages[]      →  schema + length     →  judge 1–5       →  Deita-style  →  v1 hash
human experts               →  messages[]      →  exact + MinHash 0.8 →  complexity 1–5  →  diversity    →  eval split
LLM synthesis (document)    →  messages[]      →  13-gram decontam    →  lang-ID         →  balance      →  DVC/HF rev
LLM synthesis (seeds)       →  messages[]      →  refusal strip      →  PII scrub       →  dedup        →  lock seed
                                                                                          →  cap per task
```

Every arrow is a script with a version. If you cannot re-run the pipeline and get byte-identical output, you cannot reproduce your model — see §16.1.

#### 4.6.6 Dataset licensing — the table you need before you ship

| Dataset | Rows | Licence / status | Commercial use | Notes |
|---|---|---|---|---|
| `tatsu-lab/alpaca` | 52k | CC-BY-NC-4.0 (dataset) **but** generated by `text-davinci-003` | ⚠️ Legal review required | The canonical "check this before you ship" case |
| `vicgalle/alpaca-gpt4` | 52k | Apache-2.0 (claimed) | ⚠️ Same provenance problem, different generator | The replacement most people use |
| `databricks/databricks-dolly-15k` | 15k | CC-BY-SA-3.0, **human-written** | ✅ Yes (share-alike) | The safe classic; 15k, 7 categories |
| `OpenAssistant/oasst1` / `oasst2` | 88k / 130k | Apache-2.0 | ✅ Yes | Multi-turn, human, multilingual |
| `HuggingFaceH4/ultrachat_200k` | 200k | MIT | ✅ Yes | GPT-3.5/4-generated; MIT-licensed by the publisher |
| `allenai/tulu-3-sft-mixture` | ~1M | ODC-BY-1.0 (per-component mix) | ✅ Mostly (check components) | Best current open mixture; read the per-source table |
| `teknium/OpenHermes-2.5` | ~1M | Mixed per-source | ⚠️ Component-dependent | The most-used community mixture; audit components |
| `microsoft/orca-math-word-problems-200k` | 200k | MIT | ✅ | Math-specific |
| `nvidia/OpenMathInstruct-2` | 14M | CC-BY-4.0 | ✅ | Math, synthetic from Llama-3.1 |
| Your own expert-authored data | — | Yours | ✅ | The only dataset with no legal ambiguity |
| Your own API-generated data | — | Provider ToS applies | ⚠️ Read the ToS | See the Beyond-the-video note above |

Rule: **the dataset licence covers the dataset; it does not cover the model's outputs.** For a dataset produced by a commercial API, the generating model's terms are what matter, and they are often stricter than the dataset card's licence field.

---

### 4.7 Overfitting in SFT — it does not look like classic overfitting

If you come from classical ML, you expect: train loss ↓, val loss ↑, stop. In SFT, on a 2k-row dataset, you will often see **train loss ↓, val loss ↓ or flat, and the model gets worse in four specific, recognisable ways.** The held-out loss is a *weak* signal (§12.1) and it will not save you. Learn the four signatures.

#### 4.7.1 Signature 1 — Verbosity

The model answers a yes/no question with four paragraphs. Response length drifts up 30–80% over a long SFT run while task accuracy stays flat. It happens because longer responses contain more tokens, and SFT maximises total likelihood over tokens, so the model is rewarded for producing more high-probability filler ("It's important to note that...", restating the question, adding caveats).

Diagnostic:

```python
# mean response length on a FIXED prompt set, across checkpoints
for ckpt in ["checkpoint-200", "checkpoint-400", "checkpoint-800"]:
    outs = generate_batch(ckpt, EVAL_PROMPTS)          # same 100 prompts
    print(ckpt, np.mean([len(tokenizer(o).input_ids) for o in outs]))
# 200 -> 84.1   400 -> 96.7   800 -> 138.2     <- verbosity inflation, stop at 400
```

#### 4.7.2 Signature 2 — Format rigidity

The model works perfectly on prompts that exactly match your training template and degrades sharply on trivial mutations: an extra system message, a reordered field, "Summarize" vs "Summarise", leading whitespace, a missing `### Input:` header, or a prompt in a different language.

Diagnostic — prompt-mutation sweep. The gap between "canonical" and "mutated" accuracy is your rigidity score:

```python
MUTATIONS = {
    "canonical":   lambda p: p,
    "no_system":   lambda p: p.replace("You are a helpful assistant.\n\n", ""),
    "whitespace":  lambda p: "  " + p.strip() + "\n",
    "case_shift":  lambda p: p.replace("Summarize", "summarize"),
    "header_drop": lambda p: p.replace("### Input:\n\n", ""),
    "rephrased":   lambda p: p + "\n(Be concise.)",
}
```
A healthy model loses <5 points between canonical and mutated. A model at 30+ points is over-fitted to the template and will fail the moment a different client library renders the prompt. This is also the *most expensive* form of overfitting, because it will not show up until integration.

#### 4.7.3 Signature 3 — Over-refusal

The model declines legitimate in-domain requests. Root causes, in order of frequency: (a) your dataset contains many refusal rows; (b) you fine-tuned an `-Instruct` model and the safety prior got amplified; (c) your domain vocabulary trips the base model's safety classifier; (d) you trained too long, and the most "assistant-like" behaviour in the loss is the refusal template.

Diagnostic: a frozen 200-prompt benign suite, scored for refusal phrases.

```python
REFUSAL_MARKERS = [
    "I cannot", "I can't", "I'm unable", "I am unable", "I'm sorry",
    "As an AI", "I must decline", "against my", "not appropriate", "I won't",
]
def refusal_rate(model, tokenizer, prompts):
    outs = generate_batch(model, tokenizer, prompts)
    return sum(any(m.lower() in o.lower() for m in REFUSAL_MARKERS) for o in outs) / len(outs)

print(refusal_rate(m, tok, BENIGN_200))    # must be <2% for a domain assistant
```

#### 4.7.4 Signature 4 — Catastrophic forgetting of general ability

The domain behaviour improves and everything else degrades: MMLU −6, GSM8K −12, and the model can no longer follow an instruction phrased in a way that appears nowhere in your dataset. This is the dominant risk of **full fine-tuning** and of any run longer than ~3 epochs.

Diagnostic:

```python
# run these BEFORE you train, on the base model, and after every run
BEFORE = {"mmlu": 0.624, "gsm8k": 0.481, "ifeval_format": 0.41, "held_out_domain": 0.58}
AFTER  = {"mmlu": 0.581, "gsm8k": 0.392, "ifeval_format": 0.93, "held_out_domain": 0.79}
#            -6.9%        -18.5%            +52 pts              +21 pts
# verdict: the domain win is real but you paid 18 points of math reasoning.
#          Re-run with replay + LoRA + 2 epochs.
```

Run at least one general benchmark (MMLU 500-question subset, or a 200-prompt IFEval slice) on every candidate. It costs ~10 minutes and it is the only way you will notice.

#### 4.7.5 The mitigation matrix

| Mitigation | Effect on forgetting | Effect on domain fit | Cost | When |
|---|---|---|---|---|
| **Fewer epochs** (3 → 1) | Large win | Small loss (if data is good) | Free | Always try first |
| **LoRA instead of full FT** | Large win (base frozen) | Small loss | Free (CS-23) | Almost always for <50k rows |
| **Replay 5–20% general instruction data** | Large win | Neutral to small win | +5–20% compute | Whenever you have a general SFT set |
| **Replay 5–10% raw pretraining text** | Large win | Neutral | +5–10% compute | Full FT on a narrow domain; **must not be masked** (§4.4.4) |
| **Lower LR** (2e-4 → 5e-5 for LoRA) | Moderate win | Small loss | Free | When loss is still falling but the model is drifting |
| **Early stopping on a domain + general composite metric** | Moderate | Neutral | Free | Always |
| **`neftune_noise_alpha=5`** | Neutral to small win | Small win | Free | When you cannot afford more data |
| **KL-to-base regulariser** (rare in SFT) | Large win | Small loss | +1 forward/step | Research-grade; standard in RLHF, unusual in SFT |
| **Data dedup/quality filtering** | Moderate win | Large win | One-off effort | Always |

The order matters: **replay > epochs > LoRA > LR**. If you have to pick one and you are already using LoRA at 2–3 epochs, add replay.

#### 4.7.6 The epoch-vs-loss interaction

SFT loss and SFT quality decouple after roughly the end of epoch 1. The reason: the *first* epoch is spent learning the format and the task distribution (large loss drop, large quality gain). The *later* epochs are spent memorising surface forms and response lengths (small loss drop, quality flat or negative).

| Epoch | Typical train loss | Format compliance | Verbosity | General ability | Verdict |
|---|---|---|---|---|---|
| 1 (partial) | 1.42 | 78% | baseline | −1 pt | Underfit but safe |
| 1 | 0.91 | 92% | +5% | −2 pt | **Usually the sweet spot for >20k rows** |
| 2 | 0.63 | 96% | +18% | −4 pt | **Sweet spot for 1k–10k rows** |
| 3 | 0.44 | 97% | +34% | −7 pt | Watch closely; stop at 2 if possible |
| 5 | 0.19 | 98% | +61% | −14 pt | Overfit |
| 10 | 0.06 | 98% | +90% | −26 pt | Degenerate; the model is a template engine |

**The interaction to remember:** dataset size and epoch count trade off. 1k rows × 3 epochs ≈ 3k example-views, which is where the quality curve turns. 100k rows × 1 epoch = 100k views, and 3 epochs = 300k views, which for most models is well past the turn. The rule of thumb: **stop at ~2–10k example-views per distinct row for small datasets, ~1–2 views for large ones.** And measure, do not assume — the table above is a prior, not a law.

---

### 4.8 The "SFT then what" path

```text
                 ┌───────────────────────────────────────────────────────────────┐
   pretraining → │  SFT (this module)   →   [optional] DPO / ORPO  →  [optional] RL (PPO/GRPO)
   (web text)    │  ~90% of the value       ~8% of the value        ~2% of the value
                 │  1k–500k rows            1k–50k pairs           10k+ verifiable tasks
                 └───────────────────────────────────────────────────────────────┘
                        ↑                          ↑                        ↑
                   behaviour + format        preference over pairs    verifiable reward
```

Why SFT alone is often 90% of the value:

1. **SFT establishes the interface.** Without it, DPO has nothing to align; you cannot compute a preference over two completions from a model that does not produce completions in your format.
2. **DPO's gains require a competent SFT base.** Published practice (InstructGPT, Zephyr, Tulu 2/3, and every open replication) uses an SFT checkpoint as the DPO/ORPO starting point. Preference-tuning a raw base model is possible but consistently underperforms SFT-first, because the preference signal is sparse (one bit per pair, and only implicit) relative to the dense token-level signal of SFT.
3. **DPO is a refinement, and refinements have small effect sizes.** Typical reported DPO gains over SFT on MT-Bench are ~0.2–0.5 points out of 10, and on AlpacaEval ~5–15 win-rate points. Real, but an order of magnitude less than the base→SFT jump.
4. **DPO can *undo* SFT knowledge.** Preference pairs are usually stylistic; the DPO gradient can pull the model away from the SFT-distribution facts. This is the standard "our model got chattier and slightly less accurate after DPO" report.
5. **RL needs verifiable rewards.** GRPO (CS-26) on math/code works because the reward is a program. For open-ended domain text there is usually no verifiable reward, so RLHF needs a reward model, which is another project.

The decision table:

| Your situation | Do SFT? | Then preference training? | Then RL? |
|---|---|---|---|
| Format compliance for a narrow task | Yes (1k–5k) | No | No |
| Domain assistant, house style, "we don't say X" | Yes (2k–20k) | **Yes (DPO/ORPO, 1k–5k pairs)** — this is where style constraints live | No |
| Reasoning/coding improvement on verifiable tasks | Yes (cold start, 1k–10k) | Optional | **Yes (GRPO, verifiable reward)** |
| Knowledge injection | Continued pretraining first (CS-12), then SFT | No | No |
| Latency/cost reduction (small model) | Distillation (CS-08/09) then SFT | No | No |
| Safety/refusal behaviour | Yes (mixed with task data, 2–5% boundary rows) | Yes (safety pairs) | Sometimes (RLHF) |

> **Beyond the video:** ORPO (CS-27) folds the SFT and preference objectives into one loss, removing the need for a separate SFT checkpoint. It is attractive when you only have pairs and no clean SFT set, and it is *not* a replacement for SFT when you have a real SFT dataset — published comparisons show ORPO matching SFT+DPO while being simpler, not beating it by much. If you already have 5,000 good SFT rows, running SFT first and ORPO second is the low-risk path.

---

## 5. The End-to-End Pipeline

```text
 STAGE 0            STAGE 1             STAGE 2            STAGE 3           STAGE 4
 Corpus prep   →    Continued      →    Data curation  →   SFT run      →    Eval & ship
                    pretraining                            (this module)
 ────────────       ────────────        ────────────       ────────────       ───────────
 PDFs, tickets,     raw text,          instruction /      template +         golden set,
 CRM, wiki, docs    ~10M–1B tokens     response pairs     masking +          win-rate,
     │                   │                  │             LoRA/QLoRA         regression
     ↓                   ↓                  ↓                 ↓                  ↓
 dedup, PII,        CS-12 run          filters §4.6.3     §6 code below       §12 harness
 chunk, clean                          → 2k–20k rows
```

| # | Stage | Input | Operation | Output | Failure mode if skipped or botched |
|---|---|---|---|---|---|
| 0 | **Corpus prep** | raw docs (PDF/DOCX/tickets) | dedup, PII scrub, chunk, language filter | clean text corpus + provenance log | Model memorises PII (irreversible); duplicated text over-weights some topics |
| 1 | **Continued pretraining (CS-12)** | clean corpus | next-token CE on raw text, 1–3 epochs, full FT or LoRA | domain-adapted base model | Model lacks the vocabulary; the SFT stage then has to teach vocabulary *and* format with 5k rows — it cannot (§4.1) |
| 2 | **Data curation** | source docs + experts + a generator model | author/synthesise → filter → score → balance → freeze | `sft_v1.jsonl` (messages) + `sft_v1.eval.jsonl` | The single highest-risk stage. Bad data cannot be fixed by hyperparameters |
| 3 | **Template + masking contract** | `sft_v1.jsonl` | choose template, render, mask, assert | tokenised dataset + a versioned `.jinja` | Train/serve mismatch; pad-in-loss (§14) |
| 4 | **The SFT run** | tokenised dataset, base checkpoint | LoRA/QLoRA/full FT, 1–3 epochs | adapter + metrics | Overfit signatures (§4.7); forgetting |
| 5 | **Evaluation** | adapter, golden set, base model | format compliance, win rate, refusal rate, forgetting check | scorecard + go/no-go | Shipping a regression you cannot see |
| 6 | **Merge & serve** | adapter + base | `merge_and_unload`, quantise, serve | serving artefact + version tag | Serving a different template than you trained on |
| 7 | **Regression suite** | production traffic sample + golden set | property assertions on every candidate | CI gate | Silent quality drift over model/version changes |

**Stage-level detail for the one that matters most (stage 2).** The notebook's flow, and the one to copy, is:

```text
load_dataset(csv|json) → map(format_example) → map(tokenize_fn) → Trainer
                     ↑                        ↑                  ↑
                 §4.2 formats            §4.4 masking       §7 hyperparams
```

The instructor walks exactly this path [42:20]–[44:05]: load the CSV, map a `format_example` to a single `text` column, map a tokenizer, train. His own summary of why the single-string step is necessary is the correct one [42:49]–[43:16], [26:13]: *"whatever data goes to the LLM, it goes as a single string."*

---

## 6. Hands-On Code — the notebook, annotated

The notebook is `Instruction_finetuning_on_domain_specific_dataset.ipynb`. It contains 53 cells: a first half that loads the CS-12 (non-instruction) adapter and checks it is not instruction-following, a second half that does the SFT, and a consolidated final cell that re-does everything in one block. Below is the whole thing, cleaned, annotated, and with the bugs marked. **Line-for-line, this is the module.**

### 6.1 Setup and the two models

```python
# cell 1 — the entire import surface of SFT in 2025. Note how small it is.
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
```

```python
# cells 2–3 — the base model, and the pad-token fix you must never skip.
model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

tokenizer = AutoTokenizer.from_pretrained(model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # <- REQUIRED; also the source of the §14 pad bug
```

**Why this model:** TinyLlama-1.1B is 1.1B params, 2.2 GB in fp16, and trains a LoRA adapter on a free Colab T4. The instructor says so explicitly [31:49]: *"this model is quite small, that's why I have used it; otherwise you can use any sort of a model if you have a good infrastructure."* For a production domain model the equivalent 2025 choice is `meta-llama/Llama-3.1-8B` or `Qwen/Qwen2.5-7B` with QLoRA, or a 1.5B–3B model if you are latency-bound.

`TinyLlama-1.1B-intermediate-step-1431k-3T` is a **base** checkpoint: 1.1B params, trained on 3T tokens (the "1431k-3T" = checkpoint at step 1431k of the 3T-token run). It has **no chat template** and has never been instruction-tuned. That is the correct starting point if you are doing the CS-12 → CS-13 pipeline, and it is why the model's pre-SFT behaviour is "continue the sentence" rather than "answer the question".

### 6.2 Loading the stage-1 (non-instruction) model

```python
# cells 4–6 — unpack the adapter from the previous video, then load it.
import zipfile
with zipfile.ZipFile("/content/tinyllama-lora.zip", "r") as zip_ref:
    zip_ref.extractall()          # -> /content/checkpoint-5 (a LoRA ADAPTER, 2.7 MB)

model_path = "/content/checkpoint-5"
non_instruction_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
```

> **Correction:** `AutoModelForCausalLM.from_pretrained` on a directory containing `adapter_config.json` + `adapter_model.safetensors` does **not** load the fine-tuned model. A 2.7 MB zip cannot contain a 1.1B-parameter checkpoint (that would be ~2.2 GB in fp16). The correct call is:
> ```python
> from peft import PeftModel
> base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
>                                             torch_dtype=torch.float16, device_map="auto")
> non_instruction_model = PeftModel.from_pretrained(base, "/content/checkpoint-5").merge_and_unload()
> ```
> Depending on the `transformers` version, the notebook's call either raises `OSError: no file named pytorch_model.bin` or silently constructs a **randomly initialised** model with the right architecture. The second case is worse: the "non-instruction model" output shown in the video is coherent pharma text, which means either the zip contains a merged model or the notebook was run against a different path at recording time. Do not copy this cell. (Cell 43's comment block shows the author knew about `PeftModel.from_pretrained` + `merge_and_unload` — he just did not use it.)

### 6.3 Confirming the model is *not* instruction-following

```python
# cells 7–10 — the "before" measurement. Copy this habit.
prompt = "Clinical trials demonstrated that combining Atorvastatin with Ezetimibe"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = non_instruction_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.8,        # sampling: needed for a fair qualitative read
    top_p=0.9,              # nucleus sampling: 90% of the probability mass
    do_sample=True,
    repetition_penalty=1.1, # damp the degenerate loops small models fall into
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# "Clinical trials demonstrated that combining Atorvastatin with Ezetimibe is a safe and
#  effective treatment for the ... hypercholesterolemia."
```

This is the key baseline: **the model completes the sentence**. It does not answer a question, because a question was not asked — this is a *prefix*. §6.6 repeats the experiment with an actual instruction to show the difference.

### 6.4 Inspecting both data sources

```python
# cells 15–27 — a pre-built Hub dataset, to show how someone else's schema looks.
from datasets import load_dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations", split="train")
# Dataset({features: ['Context', 'Response'], num_rows: 3512})
```

The instructor's read [36:43]–[37:07]: 3,512 rows, two columns, `Context` and `Response`. Note the **capitalised** column names — every dataset has its own convention, which is why §4.2's converters exist.

```python
# cell 17 — the instructor's first formatting attempt (Mistral/Llama-2 markers)
def format_row(example):
    question = example["Context"]
    answer = example["Response"]
    example["Text"] = f"[INST] {question} [/INST] {answer}"
    return example

# cell 18 — the second attempt (custom markers), which OVERWRITES the first in the notebook
def format_row(example):
    question = example["Context"]
    answer = example["Response"]
    example["Text"] = f"[Context] {question} [/Response] {answer}"
    return example

formatted_dataset = dataset.map(format_row)
```

Two things worth stating plainly:

1. **`[Context]` / `[/Response]` are invented markers**, not any model's template. Fine as a demonstration of "you can put any string here"; fatal if you train on it and then serve with `apply_chat_template` (§4.3.3).
2. **Cell 18 redefines `format_row`**, so only the second version is ever called by cell 19. If you are reading the notebook in order and wondering which one ran: the second one.

```python
# cells 22–27 — round-trip through CSV and JSONL.
import pandas as pd
df = pd.DataFrame(dataset)
df.to_csv("mental_health_counseling_conversations.csv", index=False)
df.to_json("mental_health_counseling_conversations.jsonl", orient="records", lines=True)

dataset = load_dataset("csv",  data_files="...csv",    split="train")   # cell 26
dataset = load_dataset("json", data_files="...jsonl",  split="train")   # cell 27
```

The instructor's point [38:00]–[40:23]: *"in which format actually you will get a data for the instruction fine-tuning... you can keep the data in any sort of a format. I have converted into the JSON, I converted into the CSV, and both ways it is possible."* Correct — `datasets.load_dataset` will read either, and the file extension is the only thing that decides the parser.

### 6.5 The real dataset: `pharma_instruction_data.jsonl`

All five rows, verbatim from the repo (this is the entire training set):

```json
{"instruction": "Explain the mechanism of action of Metformin.", "input": "", "output": "Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose."}
{"instruction": "List two non-glycemic benefits of Metformin mentioned in the text.", "input": "", "output": "Improved cardiovascular outcomes; anti-inflammatory effects. (Also suggested: anticancer potential via mTOR inhibition.)"}
{"instruction": "Why is combining Atorvastatin with Ezetimibe clinically beneficial?", "input": "", "output": "Because Ezetimibe blocks intestinal cholesterol absorption (NPC1L1) and Atorvastatin inhibits hepatic HMG-CoA reductase, providing additive LDL-C reduction—useful when statin response is inadequate."}
{"instruction": "Summarize the key advantages and ongoing research directions for mRNA vaccines.", "input": "The success of mRNA vaccines against SARS-CoV-2 has opened new pathways for rapid vaccine development. mRNA platforms enable flexible design and quick adaptation to emerging viral variants such as BQ.1 and XBB.1.5. Phase-II clinical trials have shown strong immunogenicity with elevated neutralizing antibody titers and robust CD8⁺ T-cell responses. Ongoing research is exploring thermostable formulations and self-amplifying mRNA constructs to enhance global distribution and cost-efficiency.", "output": "mRNA platforms enable rapid, flexible vaccine design and quick variant updates (e.g., BQ.1, XBB.1.5). Phase-II data show strong immunogenicity with robust neutralizing antibodies and CD8⁺ T-cell responses. Current work targets thermostable and self-amplifying formulations to improve distribution and cost."}
{"instruction": "From the passage, extract two benefits and two challenges of AI in pharmaceutical R&D.", "input": "Artificial intelligence (AI) is transforming pharmaceutical research by accelerating target identification, molecular docking, and compound screening. Deep learning models trained on large-scale biological datasets can predict protein–ligand binding affinities and optimize lead compounds. Integrating AI-driven insights with laboratory automation is reducing discovery timelines from years to months. However, challenges remain regarding interpretability, bias mitigation, and regulatory validation for AI-generated molecules.", "output": "Benefits: (1) Faster target ID, docking, and screening; (2) Better protein–ligand affinity prediction and lead optimization. Challenges: (1) Limited interpretability and bias risks; (2) Regulatory validation of AI-generated molecules."}
```

The CSV view of the same file (note how the quoting changes and how an empty `input` becomes an empty field between two commas):

```csv
instruction,input,output
Explain the mechanism of action of Metformin.,,"Metformin activates AMP-activated protein kinase (AMPK), ..."
List two non-glycemic benefits of Metformin mentioned in the text.,,Improved cardiovascular outcomes; anti-inflammatory effects. (Also suggested: anticancer potential via mTOR inhibition.)
"From the passage, extract two benefits and two challenges of AI in pharmaceutical R&D.","Artificial intelligence (AI) is transforming ...","Benefits: (1) Faster target ID, docking, and screening; ..."
```

**Format comparison, and why JSONL wins here:**

| Property | CSV (`pharma_instruction_data.csv`, 2.5 KB) | JSONL (`pharma_instruction_data.jsonl`, 2.7 KB) |
|---|---|---|
| Multi-line answers | Requires RFC-4180 quoting; a stray `"` or newline breaks the row | Native — one JSON object per line, newlines inside strings are escaped |
| Nested/multi-turn (ShareGPT, messages) | Impossible without embedding JSON in a CSV cell | Natural |
| Type fidelity | Everything is a string; `None` vs `""` becomes ambiguous (row 1 and 2 show this) | `null` / `""` / absent are distinct |
| Streaming/append | Awkward (header handling) | Perfect — append one line |
| Human editing in a spreadsheet | Easy | Painful |
| Diffable in git | Line-oriented, but quoting noise | One example per line, clean diffs |
| `load_dataset` support | `load_dataset("csv", data_files=...)` | `load_dataset("json", data_files=...)` |
| Best for | Small, flat, hand-edited sets | **Everything else — this is the production default** |

The instructor keeps both [41:15]–[42:05] and says so, which is exactly right for a 5-row teaching set. At 50k rows, CSV will cost you a day of quoting bugs.

### 6.6 The formatting step — and the template decision

```python
# cell 30 — load it.
from datasets import load_dataset
dataset = load_dataset("csv", data_files="/content/pharma_instruction_data.csv", split="train")

# cell 31 — flatten to a single training string (the Alpaca-flat template).
def format_example(example):
    prompt = (f"### Instruction:\n{example['instruction']}\n"
              f"### Input:\n{example['input']}\n"
              f"### Response:\n{example['output']}")
    return {"text": prompt}

dataset = dataset.map(format_example)
```

The rendered first row (the notebook prints it in cell 35, and the `None` is real):

```text
### Instruction:
Explain the mechanism of action of Metformin.
### Input:
None
### Response:
Metformin activates AMP-activated protein kinase (AMPK), which increases glucose uptake and fatty-acid oxidation while inhibiting hepatic gluconeogenesis, thereby lowering blood glucose.
```

> **Correction:** the `None` is a bug you should never ship. The CSV parser yields `None` for the empty `input` field, and the f-string interpolates it as the literal characters `N`, `o`, `n`, `e`. The model then learns that the token sequence `### Input:\nNone\n### Response:` is followed by a good answer — harmless-ish here, but it *teaches the model to emit the word "None"* as part of its format, and it will do so at inference (which the instructor's own printed output confirms). Fix: `example.get('input') or ''`, or omit the `### Input:` block entirely when empty. This is the single most common formatting bug in hand-rolled SFT pipelines.

**What the template should have been.** TinyLlama-1.1B-Chat-v1.0's template is Zephyr-style:

```jinja
{# actual TinyLlama-1.1B-Chat-v1.0 template, abridged #}
{% for message in messages %}
{% if message['role'] == 'user' %}<|user|>
{{ message['content'] }}</s>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}</s>
{% endif %}
{% endfor %}
```

If you are starting from `TinyLlama-1.1B-intermediate-step-1431k-3T` (a *base* model with no template), you have a choice, and the choice matters:

| Option | Template | Pro | Con |
|---|---|---|---|
| A | Add TinyLlama-Chat's Zephyr template | Matches the model family; if you later want to compare with the Chat model, comparable | The base model has never seen `<|user|>`; you are teaching a new marker from scratch |
| B | ChatML (`<\|im_start\|>`) | Industry standard; portable across Qwen/Yi/Hermes | Also unseen by this base model; adds 4 special tokens to learn |
| C | Alpaca-flat (`### Instruction:`) | Pure text, no special tokens; the base model has plausibly seen `### Instruction:` in pretraining data | No role mechanism; multi-turn is awkward |

For a **base** model, option C is actually defensible for a smoke test (it is what the notebook does) — the markers are ordinary text the model already has embeddings for. For anything with multi-turn, tool calls, or a system prompt, use B. What is *not* defensible is training with one and serving with another.

### 6.7 Tokenisation — the masked and unmasked variants

Unmasked (cell 37, the run that produced the video's results):

```python
def tokenize_fn(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
```

Masked (cell 54, the variant the instructor demonstrates last):

```python
def tokenize_and_mask(example):
    text = example["text"]
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    input_ids = enc["input_ids"]

    response_marker = "### Response:"
    response_start = text.find(response_marker)
    if response_start != -1:
        response_token_start = len(tokenizer(text[:response_start])["input_ids"])
    else:
        response_token_start = 0

    labels = input_ids.copy()
    labels[:response_token_start] = [-100] * response_token_start
    enc["labels"] = labels
    return enc

tokenized = dataset.map(tokenize_and_mask, batched=False)
```

What is right about this code:

- The offset is computed by re-tokenising the prefix, not by counting characters. Character-count-to-token-index is the classic wrong way and it is off by an unbounded amount.
- `labels[:n] = [-100] * n` correctly masks **everything before** the response marker, so the model is never trained to generate the instruction.

What is wrong with it, in order of severity:

| # | Problem | Consequence | Fix |
|---|---|---|---|
| 1 | Padding positions keep their pad-token labels (§4.4.2) | 89% of the loss is "emit `<pad>`" on this dataset; 20–70% on realistic ones | `labels[a == 0] = -100` |
| 2 | `response_token_start = 0` when the marker is missing | Masks **nothing** → silently reverts to full-text training on that row | `raise` or drop the row |
| 3 | `padding_side` is not asserted | With left padding, `labels[:n]` masks the *pads and the start of the prompt*, leaving the instruction unmasked | `tokenizer.padding_side = "right"` before mapping |
| 4 | The marker itself (indices for `### Response:`) is **not** masked | The model is trained to emit `### Response:` — mild, but at inference the template already emitted it, so the model may double it | Extend the mask past the header |
| 5 | Works only for single-turn Alpaca-flat | Silently trains on user turns in ShareGPT data | Use `assistant_only_loss=True` (§4.4.3) |
| 6 | Assumes `text.find` finds the marker at the *task* boundary | If an answer contains the literal string `### Response:`, the offset is wrong | Use the template-driven approach |

An assertion you should run once per dataset and never think about again:

```python
# fail loudly if any row's response is entirely masked out
def audit_masks(tokenized):
    bad = 0
    for ex in tokenized:
        supervised = sum(1 for l in ex["labels"] if l != -100)
        if supervised == 0:
            bad += 1
        elif supervised == len(ex["labels"]):
            bad += 1    # nothing masked at all
    print(f"rows with 0 supervised tokens: {bad}")
    return bad
audit_masks(tokenized)
```

A row with zero supervised tokens is not "harmless" — it contributes `0/0` to the mean loss, which is `NaN` on many `transformers` versions, and it silently wastes a slot in every batch. Filter them out.

### 6.8 The LoRA configuration

```python
# cell 41 — the instructor walks every field [46:53]–[47:56].
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,       # decoder-only LM
    r=8,                                 # rank of the update matrices
    lora_alpha=16,                       # scaling; effective scale = alpha / r = 2.0
    lora_dropout=0.05,                   # regularisation on the adapter
    target_modules=["q_proj", "v_proj"], # which linear layers get adapters
    bias="none",                         # do not train bias terms
)

instruction_model = get_peft_model(non_instruction_model, lora_config)
instruction_model.print_trainable_parameters()
# trainable params: 1,126,400 || all params: 1,101,375,488 || trainable%: 0.1023
```

The numbers, checked: for TinyLlama (`hidden_size=2048`, `num_layers=22`, `num_heads=32`, `num_key_value_heads=4`, `head_dim=64`), each LoRA pair costs `r × (in + out) = 8 × (in + out)` params. With `r=8` on `q_proj` and `v_proj`:

| Module | Shape | Params at `r=8` |
|---|---|---|
| `q_proj` | 2048 → 32 × 64 = **2048** | 8 × (2048 + 2048) = 32,768 |
| `v_proj` | 2048 → 4 × 64 = **256** | 8 × (2048 + 256) = 18,432 |
| **Per layer** | | **51,200** |
| **× 22 layers** | | **1,126,400** |

That reproduces the notebook's 1,126,400 **exactly** — no hand-waving required.

> **Correction (self-):** an earlier draft of this section said `q_proj`/`v_proj` are both `2048×2048`, giving `44 × 32,768 = 1,441,792`, and dismissed the gap as "varies by whether biases are included and by the exact GQA configuration." That was wrong on both counts. `v_proj` in a GQA model projects to `num_key_value_heads × head_dim` = **4 × 64 = 256**, not 2048 — the K and V projections are exactly the tensors GQA shrinks, which is the whole point of GQA. The notebook's number is not approximate; it is the exact figure, and the discrepancy is fully explained by the KV-head count. The lesson generalises: **when a parameter count disagrees with your arithmetic, look for the tensor that isn't square before you reach for an explanation involving biases.**

The point stands: **~1.1M trainable parameters, ~0.1% of the model.** The adapter artifact is 1,126,400 × 4 B ≈ **4.3 MiB** in fp32 (≈2.1 MiB in fp16) — note that the zip in the repo is ~2.7 MB, which is the *compressed archive of the adapter directory*, not the raw fp32 tensors; do not read the zip size as the parameter footprint (§10.2).

The instructor's own description of the fields [47:14]–[47:43] is accurate: *"r is representing the rank... lora_alpha is a scaling factor... lora_dropout: dropout probability... target_modules: which layer to tune... I want to tune query and value... trade-off between cost and quality."* The one thing he does not mention is that **`alpha/r` is the number that matters** — `alpha=16, r=8` means the adapter's output is scaled by 2.0, and if you halve `r` to 4 you should halve `alpha` to 8 to keep the effective scale constant. Changing `r` without changing `alpha` changes your effective learning rate. Full treatment in CS-23.

For a domain SFT on a modern model, the standard target-module set is the full attention + MLP projection list:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]   # ~8× more adapter params, better quality
```

### 6.9 Training arguments and the run

```python
# cell 44
args = TrainingArguments(
    output_dir="./tinyllama-instruction",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 1 × 8 = 8 examples
    learning_rate=2e-4,              # LoRA-appropriate; ~10× a full-FT LR
    fp16=True,                       # T4/A100; prefer bf16 on Ampere+ (no loss scaling needed)
    logging_steps=20,
    save_total_limit=1,
    report_to="none",                # <- no curve anywhere; use "wandb" in real work
)

trainer = Trainer(
    model=instruction_model,
    args=args,
    train_dataset=tokenized,
)

trainer.train()
```

**The arithmetic of this run, precisely:**

| Quantity | Value | Derivation |
|---|---|---|
| Examples | 5 | `pharma_instruction_data.csv` has 5 data rows |
| Micro-batch | 1 | `per_device_train_batch_size=1` |
| Gradient accumulation | 8 | effective batch = 8 examples |
| Steps per epoch | 5 | `len(dataloader) = 5` (5 examples / batch 1) |
| Optimizer steps per epoch | 1 | `ceil(5 / 8) = 1` |
| Epochs | 3 | |
| **Total optimizer steps** | **3** | → the saved artefact is literally `checkpoint-3` [49:05] |
| Tokens per step | 8 × 512 = 4,096 | dominated by padding (89%) |
| **Actual response tokens trained on** | 5 rows × ~32 tokens × 3 epochs ≈ **480** | vs. ~61,000 pad-token targets if unmasked |
| Learning-rate schedule | None in practice | 3 steps with a cosine schedule and default warmup |

Read that last block again. **Three gradient updates.** The checkpoint name in the video is not a coincidence — it is the whole run. This is a *pipeline demonstration*, and the instructor says so repeatedly [50:40], [54:00]: *"my data set is very small, I just trained it for a very simple epoch... if you are doing it on a full scale with a good model with a good data set, with a huge data set, that definitely this technique will work."*

Two changes that would make this a real run, with no change to the code structure:

```python
args = TrainingArguments(
    output_dir="./tinyllama-instruction-real",
    num_train_epochs=2,               # not 3 — with 2k rows, 2 is the right prior
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # effective batch 16 examples
    learning_rate=1e-4,               # LoRA; drop to 5e-5 if the loss is still falling at the end
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,                        # no GradScaler, more stable than fp16
    gradient_checkpointing=True,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    report_to="wandb",                # you cannot debug what you cannot see
    seed=42,
    data_seed=42,
)
```

### 6.10 Saving, loading, and the comparison

```python
# cells 46–48
trainer.save_model("/content/tinyllama-instruction")   # adapter_model.safetensors + config
tokenizer.save_pretrained("/content/tinyllama-instruction")
instruction_model = AutoModelForCausalLM.from_pretrained(
    "/content/tinyllama-instruction/checkpoint-3", device_map="auto"
)
```

> **Correction (again):** `save_model` writes a **PEFT adapter**, and `AutoModelForCausalLM.from_pretrained` on that directory does not restore the fine-tuned weights. The correct load is:
> ```python
> from peft import PeftModel
> base = AutoModelForCausalLM.from_pretrained(
>     "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
>     torch_dtype=torch.float16, device_map="auto")
> instruction_model = PeftModel.from_pretrained(base, "/content/tinyllama-instruction/checkpoint-3")
> instruction_model = instruction_model.merge_and_unload()   # optional: fold into base weights
> ```

This matters for the video's own result: at [50:09] the "instruction model" output is *"Clinical trials demonstrated that combining Atorvastatin with Ezetimibe is a safe and effective treatment..."* — textually identical to the non-instruction model's output at [35:54]. If the adapter was not actually loaded (or was loaded into a randomly-initialised base), that is exactly what you would see. §14 has this as a diagnostic row.

The final comparison harness (cell 56) is the right instinct and worth copying with one fix:

```python
questions = [
    "Explain the mechanism of action of Metformin.",
    "List two advantages of combining Atorvastatin with Ezetimibe.",
    "Summarize how mRNA vaccines work and mention one current research focus.",
]

for q in questions:
    print("Question:", q)

    print("\n--- Non-instruction model ---")
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    print(tokenizer.decode(
        non_instruction_model.generate(**inputs, max_new_tokens=80)[0],
        skip_special_tokens=True))

    print("\n--- Instruction-tuned model ---")
    prompt = f"### Instruction:\n{q}\n### Input:\n\n### Response:\n"   # MUST match training
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(tokenizer.decode(
        instruction_model.generate(**inputs, max_new_tokens=100)[0],
        skip_special_tokens=True))
    print("=" * 80)
```

**The fix that matters:** the non-instruction model is prompted with the *raw question* while the instruction model is prompted with the *templated* prompt. That is the correct comparison for this experiment (the base model has no template), but it means the comparison is measuring two things at once — the format and the fine-tune. For a clean read, also run the base model on the templated prompt, and the instruction model on the raw question. The gap between those four numbers is the honest story:

| Prompt shape | Base model | SFT model |
|---|---|---|
| Raw question | continues the question (fails) | usually continues / degrades |
| Templated | continues the template (fails) | answers (works) |

A model that only works in the bottom-right cell is **format-rigid** (§4.7.2), and that is the *expected* outcome of a 3-step SFT on 5 rows. It is also, at scale, the outcome you are paying to avoid.

### 6.11 What to change for your own data — checklist

| Line | Change | Why |
|---|---|---|
| `model = "TinyLlama/..."` | Your base or your CS-12 checkpoint | Do not SFT a base model whose vocabulary lacks your domain terms |
| `load_dataset("csv", data_files=...)` | `load_dataset("json", data_files=...)` or `load_dataset("your-org/your-ds")` | JSONL avoids all CSV quoting bugs; a Hub dataset gets a version hash for free |
| `format_example` | `apply_chat_template(..., tokenize=False)` | One template source for train and serve |
| `example['input']` | `example.get('input') or ''` | Kills the literal `None` |
| `tokenize_fn` / `tokenize_and_mask` | Template-aware mask + pad assertion | §4.4.2 |
| `max_length=512` | `max_length=<p99.5 of your tokenised lengths, rounded up>` | Do not truncate your answers |
| `padding="max_length"` | `padding=False` + a padding collator, or `packing=True` | 89% of this run is padding |
| `r=8, target_modules=["q_proj","v_proj"]` | `r=16` (or 32) + all 7 projection modules | `q/v` only was a 2021 compromise |
| `fp16=True` | `bf16=True` on Ampere/Hopper | No loss scaling, more stable |
| `num_train_epochs=3` | 1–3, decided by an eval run, not by default | §4.7.6 |
| `report_to="none"` | `"wandb"` | You cannot debug an invisible run |

---

## 7. Hyperparameters & Configuration — Every Knob

### 7.1 The master table

| Param | What it does | This notebook | Typical | Safe range | Too high → | Too low → | Framework flag |
|---|---|---|---|---|---|---|---|
| `learning_rate` (LoRA/QLoRA) | Step size on adapter params | **2e-4** | 1e-4 – 2e-4 | 5e-5 – 3e-4 | Loss spikes, format collapse, refusal | Nothing learned in 3 epochs; needs 10+ epochs | `learning_rate` |
| `learning_rate` (full FT) | Step size on all params | n/a | 1e-5 – 2e-5 | 5e-6 – 5e-5 | Catastrophic forgetting in <1 epoch; loss spike to NaN | Underfit at 3 epochs; looks "not learning" | `learning_rate` |
| `num_train_epochs` | Passes over the data | **3** | 1–3 | 1–3 (never >5) | Verbosity, rigidity, forgetting (§4.7) | Format not adopted; task underfit | `num_train_epochs` |
| `per_device_train_batch_size` | Micro-batch | **1** | 1–8 | 1–16 | OOM | Slow, noisy gradients | `per_device_train_batch_size` |
| `gradient_accumulation_steps` | Micro-batches per step | **8** | 4–16 | 1–64 | Fewer optimizer steps than a schedule needs | Effective batch too small → unstable | `gradient_accumulation_steps` |
| **Tokens per optimizer step** | The real batch size | 4,096 | 32k–128k | 16k–256k | LR needed is higher; memory pressure | Noisy loss, unstable LR | product of the three above × `seq_len` × GPUs |
| `lr_scheduler_type` | Schedule shape | default (linear) | `cosine` | `cosine`, `linear`, `constant_with_warmup` | — | `constant` with a high LR over-trains late | `lr_scheduler_type` |
| `warmup_ratio` / `warmup_steps` | LR ramp | default (0) | 0.03–0.10 | 0.01–0.15 | Too few real steps at full LR | First steps damage pretrained weights; loss spike | `warmup_ratio` |
| `max_length` / `max_seq_length` | Truncation window | **512** | p99.5 of your data | 256–4096 | Wasted compute, OOM | **Answers truncated → model stops mid-sentence** | `max_length` (TRL), `max_seq_length` (Unsloth) |
| `padding` | How batches are made rectangular | `"max_length"` | `False` + collator, or `packing=True` | — | `max_length` wastes 20–90% | — | `padding` |
| `packing` | Concatenate examples into one window | not set | `True` for short data | — | Cross-contamination if position ids are not reset | 2–5× wasted compute | `packing` (TRL `SFTConfig`) |
| `padding_free` | Flatten without pack | not set | `True` w/ FA2 | — | — | — | `padding_free` |
| `gradient_checkpointing` | Recompute activations | not set | `True` | — | +25–30% time | OOM at long sequences | `gradient_checkpointing` |
| `optim` | Optimizer | default AdamW | `adamw_torch` / `paged_adamw_8bit` | — | — | — | `optim` |
| `bf16` / `fp16` | Numeric precision | **`fp16=True`** | `bf16=True` on Ampere+ | one of them, not both | Overflow/NaNs (fp16 with high LR) | Slow, more memory | `bf16`, `fp16` |
| `lora_r` | Adapter rank | **8** | 16 | 4–64 | Overfits small datasets; more memory | Underfits; cannot learn the task | `r` (peft) |
| `lora_alpha` | Adapter scaling | **16** | `r` or `2r` | keep `alpha/r ∈ [0.5, 4]` | Effectively raises the LR → instability | Adapter has no effect → "fine-tune did nothing" | `lora_alpha` |
| `lora_dropout` | Adapter dropout | **0.05** | 0.05–0.1 | 0–0.1 | Underfit (with `r=8` and 5 rows) | Overfit on small data | `lora_dropout` |
| `target_modules` | Which linears get adapters | **`q_proj, v_proj`** | all 7 projections | see §7.8 | More memory, slower | Quality ceiling: `q/v` only underperforms `all-linear` by 2–5 points on domain tasks | `target_modules` |
| `neftune_noise_alpha` | Embedding noise (train only) | not set | 5 | 0–15 (0 = off) | Degrades long-form and already-clean data | — | `neftune_noise_alpha` |
| `seed` / `data_seed` | Reproducibility | not set | 42 | any fixed int | — | — | `seed`, `data_seed` |
| `save_total_limit` | Checkpoints kept | **1** | 2–3 | ≥2 | Disk full | You cannot roll back a bad run | `save_total_limit` |
| `report_to` | Experiment tracking | **`"none"`** | `"wandb"` | — | — | **You cannot see the loss curve** | `report_to` |
| `assistant_only_loss` | Template-aware masking | n/a (manual) | `True` | — | Silently no-ops without `{% generation %}` | Prompt-in-loss (§4.4) | `assistant_only_loss` (TRL `SFTConfig`) |
| `train_on_prompt` | LLaMA-Factory's mask switch | n/a | `false` | — | Prompt learned | — | LLaMA-Factory dataset YAML |
| `train_on_inputs` | Axolotl's mask switch | n/a | `false` | — | Prompt learned | — | Axolotl YAML |
| `max_grad_norm` | Gradient clipping | default 1.0 | 1.0 | 0.3–1.0 | Slow learning | Loss spikes propagate | `max_grad_norm` |
| `weight_decay` | Regularisation | default 0 | 0–0.1 | 0–0.1 | Underfit | — | `weight_decay` |
| `group_by_length` | Batch similar lengths together | not set | `True` (padding mode) | — | — | Wasted padding | `group_by_length` |

### 7.2 Learning rate — the full-FT vs LoRA split

The instructor uses `2e-4` with LoRA [48:35], and that is the standard LoRA LR. The reason full FT uses 10–20× less is worth being able to explain: **LoRA's gradient only touches the low-rank adapter, and the adapter's output is `BAx · (alpha/r)`.** With `alpha/r = 2` and `A` initialised from a Gaussian, the effective per-step perturbation of the frozen weights is small relative to a full-FT update — so you compensate with a bigger LR. Additionally, full FT moves *every* weight including the embeddings and the output head, which are the parameters most responsible for keeping the model's general ability; large steps there are what cause forgetting.

| Method | LR | Why |
|---|---|---|
| Full FT, base model | 1e-5 – 2e-5 | Minimum perturbation of a pretrained optimum |
| Full FT, instruct model | 5e-6 – 1e-5 | Already at a good optimum; you are nudging a style |
| LoRA / QLoRA | 1e-4 – 2e-4 | Only the adapter moves; `alpha/r` scales it down |
| LoRA on a 1k-row dataset, 1 epoch | 5e-5 – 1e-4 | Fewer steps → the same total displacement needs a smaller per-step rate |
| Continued pretraining (CS-12) | 1e-5 – 3e-5 (full FT), 1e-4 (LoRA) | Same logic, but you want *more* displacement |

Diagnostic for "is my LR right": look at the loss curve in the first 50 steps. A healthy LoRA run drops from ~2.5 (random-ish on TinyLlama) to ~1.2 within 50–100 steps. A curve that oscillates by ±0.4 between consecutive logging steps is too high. A curve that is flat to three decimals after 200 steps is too low.

### 7.3 Epochs — the single most-abused knob

The rule and its reasoning are in §4.7.6. The compressed version:

- **1 epoch** when you have >20k rows, or when you are fine-tuning an already-instruct model, or when you cannot measure forgetting.
- **2 epochs** is the default for 1k–10k rows.
- **3 epochs** is the maximum for domain SFT, and only with replay + a general-capability eval.
- **>3 epochs** is a red flag in a code review. If you need more than 3 epochs to learn the task, your task is under-specified, your LR is wrong, or your data is too small for the behaviour you want.

The instructor uses 3 on 5 rows, where it is harmless (3 optimizer steps) but also meaningless.

### 7.4 Warmup — and the degenerate case

Warmup protects the pretrained weights from the first, largest steps, when the optimizer's second-moment estimates are still garbage (AdamW's bias correction is poor for the first ~10 steps). 3–10% of total steps, linear.

**The degenerate case in this notebook:** with 3 total optimizer steps and default `warmup_steps=0`, there is no warmup, and a cosine schedule over 3 steps is nearly a constant. **Schedule-shaped hyperparameters are meaningless below ~200 optimizer steps.** If your run is under 200 steps, the only things that matter are the LR, the data, and the epoch count. This is a useful thing to say out loud in an interview — it shows you know which knobs are inert at small scale.

Practical formula for choosing warmup:

```python
num_steps = (len(ds) // (micro_bs * grad_accum * n_gpus)) * epochs
warmup_steps = max(10, int(0.05 * num_steps))
# 5,000 rows, bs 4, ga 4, 1 GPU, 2 epochs -> 625 steps -> warmup 31
# 5 rows, bs 1, ga 8, 1 GPU, 3 epochs      -> 3 steps   -> warmup is irrelevant
```

### 7.5 Scheduler — cosine vs linear

| Schedule | Shape | When |
|---|---|---|
| **Cosine** | Slow decay, fast finish | Default for SFT. Reaches a lower final loss at the same step count |
| **Linear** | Constant decay | Fine; the HF default. Marginal difference from cosine |
| **Constant with warmup** | Flat then drop to 0 | Short runs (<300 steps) where scheduling does nothing anyway |
| **Cosine with restarts / WSD** | Multi-phase | Research; warmup-stable-decay is popular for long pretraining runs (CS-12), not SFT |

The practical difference between cosine and linear on a 2-epoch domain SFT is within noise. Pick cosine, move on, and do not spend a sweep on it.

### 7.6 Batch size — measure it in tokens, not examples

"Batch size 16" is meaningless when one example is 40 tokens and another is 3,900. The stability-relevant quantity is **tokens per optimizer step**:

```python
tokens_per_step = micro_bs * max_length * grad_accum * n_gpus
# notebook: 1 × 512 × 8 × 1 = 4,096 tokens/step    <- tiny; gradient is very noisy
# typical:  4 × 1024 × 8 × 1 = 32,768 tokens/step   <- good default
# large:    8 × 2048 × 8 × 8 = 1,048,576            <- 1M tokens/step, needs LR tuning
```

Targets:

| Tokens/step | Effect | Typical setup |
|---|---|---|
| < 8k | Very noisy; LR must be lower; loss curve is sawtooth | Single small GPU, long sequences |
| 16k–32k | Workable minimum for stable SFT | `bs 2 × 2048 × ga 8` on a 24 GB card |
| **32k–128k** | **The sweet spot** | `bs 4 × 2048 × ga 8` on an A100 |
| 256k+ | Needs LR scaling and often more warmup; diminishing returns | Multi-GPU full FT |

If you increase tokens/step by 4×, increase LR by roughly 2× (square-root scaling), not 4×. The instructor's `2e-4 / bs 8 / 512` is a valid small-scale point; scaling the same LR to a 64× larger batch would be unstable.

### 7.7 `max_seq_length` and the truncation danger

This is the quietest catastrophic bug in SFT. If `truncation=True` cuts a row before (or in the middle of) the response, then:

- With prompt masking: the row has **few or zero supervised tokens**. Zero → NaN loss on some versions, or a wasted row. Few → the model is trained to produce an answer that stops mid-sentence.
- Without prompt masking: the model is trained on a document that just ends. Harmless-ish, and therefore *worse* — you will not notice.

The diagnostic is two lines and should be in every pipeline:

```python
lens = [len(tokenizer(render(r)).input_ids) for r in ds]
too_long = sum(1 for l in lens if l > MAX_LEN)
print(f"{too_long}/{len(ds)} rows exceed max_length={MAX_LEN} "
      f"({100*too_long/len(ds):.2f}%)")

# and the one that actually matters: how many rows lose their RESPONSE to truncation?
def response_lost(row, tokenizer, max_len):
    full = tokenizer(render(row)).input_ids[:max_len]
    head = tokenizer(render_prompt_only(row)).input_ids
    return len(head) >= max_len      # the response never even starts
print(sum(response_lost(r, tokenizer, MAX_LEN) for r in ds))
```

Target: **<0.5%** of rows lose their response. If more, either raise `max_seq_length` or drop/filter those rows explicitly. Do not let the tokenizer decide silently.

### 7.8 `target_modules` — where the quality is

| Configuration | Trainable params (7B) | Typical quality vs `all-linear` | When |
|---|---|---|---|
| `q_proj, v_proj` (the notebook) | ~4M (0.06%) | −2 to −5 points on domain tasks | 2021-era default; fine for a smoke test |
| `q_proj, k_proj, v_proj, o_proj` | ~8M | −1 to −2 points | Attention-only; cheapest useful config |
| **all 7 projections** (`q,k,v,o,gate,up,down`) | ~20M (0.3%) | **baseline** | The 2024+ default. `target_modules="all-linear"` in peft ≥ 0.8 |
| + `lm_head`, `embed_tokens` | +300M | Usually worse unless you are adding vocabulary | Rare; only for new-token injection |

```python
# peft >= 0.8: one string instead of a list
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",     # every nn.Linear except the output head
    bias="none",
)
```

### 7.9 `neftune_noise_alpha`

NEFTune adds uniform noise `U(-α, α)` to the *embedding* output on every forward pass during training only:

$$\tilde{e}_i = e_i + u_i, \qquad u_i \sim \mathcal{U}\left(-\frac{\alpha}{\sqrt{L \cdot d}}, \frac{\alpha}{\sqrt{L \cdot d}}\right)$$

where `L` is the sequence length and `d` the embedding dimension. Inference is unchanged — there is no runtime cost, no architecture change, and no extra data.

```python
args = TrainingArguments(
    ...,
    neftune_noise_alpha=5,     # paper's default; 10-15 for long sequences
)
```

Reported effect (Jain et al., 2023): +29.8% on AlpacaEval for LLaMA-2-7B, +8.7% for 13B, with no loss on MMLU/GSM8K. The mechanism is regularisation of the embedding space: the model cannot memorise exact token sequences, so it learns more robust surface-form mappings — which is precisely the failure mode (§4.7.2 format rigidity) that SFT produces.

**When it does not help:** if your data is already large and clean (Tulu-3-scale ablations found it neutral-to-negative there), or if you are training on long documents where the noise perturbs semantic content. Treat it as a cheap 1-line experiment, not a mandatory flag.

> **Beyond the video — the four "free" wins that are not in the course, in priority order:**
> 1. **`assistant_only_loss=True`** (or equivalent masking) — the largest single effect, and it is not a hyperparameter, it is correctness.
> 2. **`neftune_noise_alpha=5`** — one line, 3–30% subjective win on conversational data.
> 3. **`target_modules="all-linear"`, `r=16`** — 2–5 quality points over `q/v`, r=8.
> 4. **`packing=True` + `bf16`** — 2–5× throughput, which buys you more epochs of *measurement*, which is where the real gains are.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 Should you SFT at all?

| Situation | SFT? | Instead use | Why |
|---|---|---|---|
| Model must reliably emit a schema/format | **Yes** | — | Format compliance is exactly what SFT buys (§4.1) |
| Model must adopt a house tone/register | **Yes** | — | Style is a distribution over formats |
| Model must know facts that change weekly | No | **RAG** (CS-04) | Retraining per fact change is a pipeline anti-pattern |
| Model must know *terminology* for a fixed domain | Sort of | **Continued pretraining** (CS-12) then SFT | SFT alone cannot teach vocabulary efficiently (§4.1.2) |
| Model must improve at verifiable reasoning | Partially | **RL/GRPO** (CS-26) with SFT cold-start | SFT memorises; RL generalises |
| Model must be smaller/cheaper | No | **Distillation** (CS-08/09) + quantisation (CS-10) | SFT does not reduce size |
| Model must respect a content policy | **Yes** (2–5% boundary rows) | + DPO for the fine judgements (CS-14) | Refusal boundaries are behaviour, and behaviour is SFT-shaped |
| You have <100 good examples and no way to make more | Not yet | Prompt engineering + few-shot | SFT on 80 rows is a coin flip on format rigidity |
| Your task is "answer from this document" and the document changes | No | RAG | You will be retraining forever |
| You need latency <200 ms with a 70B model | Yes for the small model | Distil, then SFT the small model | SFT is the last step of a compression pipeline |

### 8.2 STOP conditions — signals SFT is the wrong tool

1. **You cannot state the behaviour change in one sentence.** "It should be better" is not a target. "It should answer in ≤2 sentences with a citation ID" is.
2. **Your evaluation is 'look at some outputs'.** You have no way to know if it worked, so you will ship a regression.
3. **The requirement is a fact that changes.** RAG.
4. **The requirement is 'know everything about our 40,000-page corpus'.** Continued pretraining, or RAG. Not SFT.
5. **You have fewer than ~200 rows and no budget for more.** Prompt engineer; revisit with data.
6. **The base model cannot do the task at all, even with a perfect prompt.** 50-shot prompting with a strong model is the ceiling test. If a well-prompted 70B cannot do it, SFT on 2k rows will not teach it — you need capability, which comes from scale or distillation.
7. **You are fine-tuning to fix a bug in the base model.** Fine-tuning does not fix tokenizer bugs, context-window limits, or a template mismatch in your server.
8. **You are fine-tuning because it is 2026 and everyone is.** The honest first move is almost always retrieval, prompting, or a better base model.

### 8.3 Which SFT configuration?

```text
How many curated rows do you have?
├─ < 200            → do not SFT yet. Prompt-engineer, or generate more data.
├─ 200 – 1,000      → LoRA r=16 all-linear, 2-3 epochs, LR 1e-4, replay 20%, expect rigidity.
├─ 1,000 – 10,000   → LoRA r=16-32 all-linear, 2 epochs, LR 1e-4, replay 10%, eval every 50 steps.
├─ 10,000 – 100,000 → LoRA r=32 or full FT, 1-2 epochs, LR 1e-4 (LoRA) / 1e-5 (full), replay 5%.
└─ > 100,000        → full FT or LoRA r=64, 1 epoch, LR 1e-5, replay 1-5%, pack, multi-GPU.

Is the base model already instruction-tuned?
├─ Yes → LR ÷ 2, epochs 1-2, expect faster convergence and more over-refusal.
└─ No  → the CS-12 pipeline first if the domain vocabulary is unfamiliar.

Is your data multi-turn?
├─ Yes → assistant_only_loss=True (or train_on_responses_only) + verify per-turn masking.
└─ No  → single-turn mask; still verify with the pad assertion.

Are responses long (>1,024 tokens)?
├─ Yes → packing=False, gradient_checkpointing=True, bf16, watch truncation.
└─ No  → packing=True, 2-5× throughput, position-id reset check.
```

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros

| # | Advantage | Evidence / magnitude |
|---|---|---|
| 1 | Directly controls behaviour, tone, and format | Format compliance 40% → 95%+ is a routine SFT result |
| 2 | Cheap at small scale | 1k rows, QLoRA, one GPU-hour, a few dollars |
| 3 | No reward model, no RL machinery, no preference pairs | The simplest of the three alignment stages |
| 4 | Dense supervision | 1 loss term per token, versus 1 bit per pair in DPO |
| 5 | Runs anywhere | 7B QLoRA fits in 8–12 GB |
| 6 | Composable with everything else | Distil→SFT, SFT→DPO, SFT→quantise, SFT→serve |
| 7 | Reproducible when the data is versioned | Same seed + same data hash + same code = same adapter |
| 8 | Works on base models with no existing chat behaviour | No RLHF prior to fight |

### 9.2 Cons

| # | Disadvantage | Reality |
|---|---|---|
| 1 | Data curation is 80% of the effort and cannot be automated away | 2–6 weeks of expert time for a real domain set |
| 2 | Poor at injecting facts | 10–100 paraphrases per fact; brittle recall |
| 3 | Overfits in ways that val loss does not show | §4.7 |
| 4 | Catastrophic forgetting on long runs | MMLU −6 to −14 points is typical at 5 epochs |
| 5 | Amplifies whatever is in the data — including refusals and biases | No negative examples to correct it |
| 6 | The template becomes a hard contract | §4.3.3 |
| 7 | Requires a real evaluation harness to be worth doing | Without one you are optimising blind |
| 8 | Hyperparameters are mostly inert at small data size | 3 optimizer steps: LR is the only live knob |

### 9.3 Hard limitations

| Limitation | Why it is hard (not a tuning issue) |
|---|---|
| Cannot add capability the base model lacks | Capability comes from pretraining scale or distillation (CS-08/09) |
| Cannot reliably store retrievable facts | Facts need repetition SFT does not provide; and later stages erase them |
| Cannot fix tokenizer/context-window limits | Architectural, not behavioural |
| Cannot be evaluated by loss alone | SFT loss measures fit to your data, not quality |
| Cannot generalise to prompts far outside the training distribution | This is definitional: SFT is a distributional reweighting |
| Cannot be undone cleanly if the data was memorised | PII in the weights is not removable by re-fine-tuning |

### 9.4 Silent failure modes — looks fine, is broken

| # | Silent failure | What you see | What is actually happening | Detection |
|---|---|---|---|---|
| 1 | **Padding trained as targets** | Loss goes down nicely; generations are choppy or terminate oddly | 20–90% of loss is "emit `<pad>`/<eos>" | `assert attention_mask==0 → labels==-100` |
| 2 | **Prompt unmasked** | Loss is lower than expected; model occasionally recites the instruction | Gradient spent learning to generate prompts | Inspect `labels[:20]` — should be all −100 |
| 3 | **Train/serve template mismatch** | Great in the eval notebook, bad in the API | The role-marker cue is absent at inference | Render the same conversation through both paths; `assert train_str == serve_str` |
| 4 | **Adapter not actually loaded** | "The fine-tune didn't do anything" | `AutoModelForCausalLM.from_pretrained(adapter_dir)` built a random model | Compare a weight hash before/after loading; print `print_trainable_parameters` at load |
| 5 | **Response truncated away** | Loss plateaus; model stops mid-sentence on long inputs | `max_length` cuts before the response starts | `response_lost` count (§7.7) |
| 6 | **No EOS in the response spans** | Model never stops generating | The data was built by string concat without `<|eot_id|>`/`</s>` | Check the last label of each row equals the EOS id |
| 7 | **All-`-100` rows** | Occasional NaN loss or a suspiciously flat curve | 0/0 in the mean | `audit_masks` (§6.7) |
| 8 | **Wrong tokenizer for the model** | Quality much worse than benchmarks suggest | Different vocab → different ids → garbage embeddings | `assert model.config.vocab_size == len(tokenizer)` |
| 9 | **`padding_side` mismatch with a manual mask** | Instruction appears trained, response partially masked | `labels[:n]` indexes the wrong end | `assert tokenizer.padding_side == "right"` when using prefix masks |
| 10 | **Loss averaged over a different denominator across runs** | "Run B has lower loss so it's better" | Different masks → different means | Compare *task metrics*, never cross-run loss |
| 11 | **Evaluation set contaminated** | Eval accuracy implausibly high | Train rows overlap eval prompts | 13-gram decontamination counter |
| 12 | **Data ordering effect** | Final checkpoint is worse than the mid-run one | The last batch's distribution dominates the final weights | Always evaluate ≥3 checkpoints, not just the last |
| 13 | **`report_to="none"`** | Nothing to debug with | No curve exists | Turn on wandb/tensorboard |
| 14 | **Chat template silently not applied** | Model ignores the system message | `tokenizer(text)` used instead of `apply_chat_template` | Print the rendered string in the training script |

---

## 10. Exceptions, Edge Cases & Gotchas

1. **Multi-turn masking.** In ShareGPT/`messages` data, "the response" is *every* assistant turn. Masking only the final turn discards most of your supervision; masking nothing teaches the model to write user turns. Use `assistant_only_loss=True` or `train_on_responses_only`, and verify by printing the first 40 labels of a multi-turn row.
2. **System-prompt handling varies by template.** Llama-2 wraps it in `<<SYS>>` inside the first `[INST]`; Mistral v0.1/v0.2 has no system role at all and some templates silently *drop* it; ChatML has a first-class system turn. If your template drops the system message, your model never learns to condition on it — and at serving time your harness will send one anyway.
3. **Empty `input` fields.** `None` vs `""` vs missing (§6.6). Normalise at ingest; never interpolate a Python `None` into an f-string that becomes training data.
4. **Instruction and response overlapping.** If the answer is a substring of the prompt, you are training a copy task. Check token-Jaccard; treat >0.5 as a red flag.
5. **Very short responses teach stopping.** Include them deliberately (10–20% of rows under 30 tokens) or your model will ramble. This is the cheapest, most reliable anti-verbosity intervention.
6. **A single long response can dominate an epoch.** One 4,000-token example contributes 40× the loss of a 100-token one. Either cap response length, or weight rows. `len(resp) > 3 × median` is a reasonable flag.
7. **BOS duplication.** Some pipelines add `<s>` at the tokenizer *and* the template adds `<|begin_of_text|>`. Two BOS tokens degrade quality subtly. Check `tokenizer(rendered)["input_ids"][:3]`.
8. **`tokenizer.pad_token = tokenizer.eos_token` makes the pad bug invisible.** The pad id *is* the eos id, so "training on pads" looks like "training on eos" and nothing errors. This is exactly why it survives code review.
9. **Left-padding for generation, right-padding for training.** If you use the same tokenizer object for both and flip `padding_side` at some point in the script, your manual mask offsets break (§4.4.2 #3).
10. **LoRA on a model you also fully fine-tuned is not additive.** Two adapters trained sequentially do **not** compose by simple addition unless they were trained on the same base with the same target modules. The clean pattern is: merge adapter 1, then attach a fresh adapter 2 to the merged model (the notebook's cell 43 comment describes exactly this, correctly).
11. **`merge_and_unload` is irreversible and template-coupled.** Keep the adapter, the base model id/hash, and the `.jinja` template together as one versioned artefact.
12. **Evaluating with `do_sample=False` changes the verdict.** Greedy decoding flatters format compliance and understates diversity problems. Report both greedy and sampled numbers.
13. **Quantisation after SFT can undo format compliance.** NF4/AWQ quantisation of a small SFT model sometimes introduces format drift (missing closing tags). Always re-run the regression suite on the *quantised* artefact, not the fp16 one.
14. **The same dataset can be used for SFT and DPO.** If you export `prompt/chosen/rejected` from your SFT curation (using a weaker model's answer as `rejected`), make sure the `chosen` string is *byte-identical* to the SFT target. Any divergence means the two stages pull in different directions.
15. **Distillation data is SFT data.** If you generated the dataset by sampling a teacher model (CS-08/09), your SFT model inherits the teacher's format quirks, refusals, and stop-token habits. Always include a small human-written slice as a tiebreaker.
16. **`max_length` applies per row, not per conversation.** A 3-turn conversation that fits in 1,024 tokens total will still be truncated if any *single* rendered row exceeds it — but conversely, a conversation exceeding the window gets cut at the end, silently dropping the final (and most important) assistant turn. Truncate from the *beginning* of the conversation, not the end, if the last turn is the one you train on.
17. **Seed everything.** `transformers.set_seed(42)` covers `random`, `numpy`, `torch`; it does *not* cover `datasets` shuffling (`data_seed`), CUDA nondeterminism (`torch.use_deterministic_algorithms(True)`, with a performance cost), or dataloader worker seeding (`dataloader_num_workers=0` for reproducibility).
18. **SFT on a model with a broken chat template is a trap.** Some community uploads have a `chat_template` that does not match the weights (a re-upload with a merged adapter and the wrong template). Always render and *read* the string before you train.

---

## 11. Cost, Compute & Memory

### 11.1 VRAM formula

$$\text{VRAM} \approx \underbrace{P_w \cdot N}_{\text{weights}} + \underbrace{P_g \cdot N_{\text{train}}}_{\text{grads}} + \underbrace{8 \cdot N_{\text{train}}}_{\text{AdamW } m,v} + \underbrace{4 \cdot N}_{\text{fp32 master (full FT)}} + \underbrace{c \cdot B \cdot S \cdot L \cdot d}_{\text{activations}}$$

where `c ≈ 34` bytes without gradient checkpointing and `c ≈ 2` with it (plus a one-off `2N` for the recompute buffer), and `N_train` is the number of trainable parameters.

### 11.2 VRAM by method — measured/derived, batch size 1, seq 2048

| Model | Full FT fp16 | LoRA fp16 (r=16, all-linear) | QLoRA 4-bit NF4 (r=16) |
|---|---|---|---|
| 1B (TinyLlama) | 16 GB + act | 4 GB + act | **1.5 GB** + act |
| 3B (Llama-3.2-3B) | 48 GB | 8 GB | **3 GB** |
| 7–8B (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B) | **112 GB** static → 2–4× A100-80 | 16 GB + act (~24–30 GB total) | **6 GB** + act (~9–12 GB total) |
| 13B | 208 GB → 4× A100-80 minimum | 28 GB + act | 9 GB + act |
| 70B | 1,120 GB → 16× A100-80 (needs ZeRO-3) | 145 GB + act → 2× A100-80 | 40 GB + act → **1× A100-80** |

Practical GPU mapping (QLoRA/LoRA):

| GPU | VRAM | Comfortable ceiling |
|---|---|---|
| Colab T4 | 16 GB | QLoRA 7–8B @ seq 1024, bs 1, ga 16 |
| RTX 4090 / L4 | 24 GB | QLoRA 8B @ seq 2048, bs 2; LoRA 3B fp16 |
| A100-40 | 40 GB | LoRA 8B fp16 @ seq 2048 bs 4; QLoRA 13B |
| A100-80 / H100-80 | 80 GB | Full FT 3B; LoRA 13–30B; QLoRA 70B |

### 11.3 Worked example — the reader's real project

**Setup:** domain SFT of `Llama-3.1-8B` (base), 8,000 curated rows, mean rendered length 512 tokens, QLoRA r=16 all-linear, 2 epochs, `bs 2 × ga 8 × seq 1024` on one A100-40.

| Quantity | Value |
|---|---|
| Training tokens | 8,000 × 512 × 2 epochs = **8.19M tokens** |
| Tokens per optimizer step | 2 × 1024 × 8 = 16,384 |
| Optimizer steps | 8.19M / 16,384 = **500 steps** |
| Measured throughput (QLoRA 8B, A100-40, gc on) | ≈ 4,000 tokens/s |
| Wall clock | 8.19M / 4,000 = **2,048 s ≈ 34 min** |
| Rent (A100-40 @ $1.50/hr on a 2025 spot market) | **≈ $0.85** |
| Storage | adapter 160 MB + base 16 GB (fp16) or 5 GB (4-bit) |
| **Eval cost** | 200 prompts × 2 models × ~600 tokens ≈ 0.24M tokens through a judge → **<$1** |
| **Human review** | 2 reviewers × 3 h × $50/h = **$300** ← 99% of the project cost |

**The point of that table:** compute is nearly free. **The dataset and the evaluation are the cost.** A 2,000-row expert-authored domain dataset at 20 minutes per row is ~670 person-hours ≈ **$33,000** — 40,000× the GPU bill. Anyone who tells you the hard part of SFT is the GPU has not done it.

The notebook's own run, for contrast:

| Quantity | Value |
|---|---|
| Model / data | TinyLlama-1.1B / 5 rows / 512 tokens |
| Tokens seen | 5 × 512 × 3 = 7,680 (≈ 480 non-pad) |
| Optimizer steps | 3 |
| Wall clock on a T4 | ~15 seconds |
| Cost | **<$0.001** |

### 11.4 Full fine-tuning cost, for completeness

| Model | Method | Hardware | Wall clock (8k × 512 × 2 ep) | Cost @ $1.50/GPU-hr |
|---|---|---|---|---|
| 8B | Full FT + ZeRO-3 | 4 × A100-80 | ~1.2 h | **$7.20** |
| 8B | LoRA fp16 | 1 × A100-40 | ~0.6 h | $0.90 |
| 8B | QLoRA | 1 × A100-40 | ~0.6 h | $0.90 |
| 70B | QLoRA | 1 × A100-80 | ~6 h | $9.00 |
| 70B | Full FT + ZeRO-3 | 16 × A100-80 | ~3 h | **$72.00** |

(Estimates, extrapolated from throughputs of 1,000–1,500 tokens/s/GPU full FT, 4,000 LoRA, 2,000 QLoRA-70B. Real numbers move ±40% with `packing`, sequence length, and whether flash-attention is on.)

---

## 12. Evaluation — How To Know It Worked

### 12.1 The five metrics, and how each one lies to you

| Metric | What it measures | Cost | How it lies |
|---|---|---|---|
| **Held-out loss** | Fit to your data distribution | Free | **The weakest signal available.** A model at epoch 5 has lower val loss and is worse. Only useful to detect divergence/NaN |
| **Exact match / task metric** | Correctness on closed-form tasks | Free | Only exists for closed tasks (extraction, classification, JSON). Useless for open generation |
| **Format compliance rate** | % that parses against your schema | Free | Can be gamed: a model that always emits an empty JSON `{}` scores 100% |
| **Refusal rate** | % of benign prompts declined | Free | Aggregate hides that it refuses a specific sensitive-but-legitimate category |
| **Win rate vs base (LLM judge)** | Subjective quality | ~$1 / 100 pairs | Position bias, verbosity bias, self-preference, and it is not ground truth |
| **Forgetting delta** | General ability retained | ~$0.50 / benchmark run | Noisy on 500-question subsets; do not read ±2% as real |

### 12.2 Held-out loss — when it is actually informative

Rules for making loss non-useless:

1. **Use the same mask on train and eval.** Otherwise the two numbers are on different denominators (§4.4.1) and the comparison is meaningless.
2. **Compare checkpoints within a run, never across runs** with different data.
3. **Watch for the plateau, not the minimum.** Quality turns before loss flattens (§4.7.6). Plot loss *and* mean response length *and* format compliance on the same axes.
4. **A rising eval loss with a falling train loss is the classic signal — but a *flat* eval loss with a falling train loss is far more common in SFT and equally bad.**

```python
# minimal held-out evaluation that is actually comparable
def evaluate_loss(trainer, tokenized_eval):
    out = trainer.evaluate(eval_dataset=tokenized_eval)
    print({k: round(v, 4) for k, v in out.items()})
    # sanity: the eval set must have the same masking policy as the train set
    ex = tokenized_eval[0]
    assert sum(1 for l in ex["labels"] if l != -100) > 0
    return out
```

### 12.3 Format compliance and refusal rate — the two free metrics you must have

```python
import json, re

REFUSAL_MARKERS = ("I cannot", "I can't", "I'm unable", "I am unable", "I'm sorry",
                   "As an AI", "I must decline", "I won't", "not able to help")

def format_compliance(outputs, validator):
    return sum(1 for o in outputs if validator(o)) / len(outputs)

def json_validator(text):
    try:
        obj = json.loads(text.strip().strip("`"))
        return isinstance(obj, dict) and len(obj) > 0     # non-empty, not just parseable
    except Exception:
        return False

def citation_validator(text):
    return bool(re.search(r"\[(?:CT|DOC|REF)-\d{3,}\]", text))

def refusal_rate(outputs):
    return sum(any(m.lower() in o.lower() for m in REFUSAL_MARKERS) for o in outputs) / len(outputs)

# Report all three on a FROZEN 200-prompt set, at every checkpoint.
print(f"format   {format_compliance(OUT, json_validator):.3f}")   # want >= 0.95
print(f"refusal  {refusal_rate(OUT):.3f}")                        # want <= 0.02
print(f"mean len {np.mean([len(tok(o).input_ids) for o in OUT]):.1f}")  # watch for drift
```

### 12.4 The LLM-judge protocol, including its biases

Pairwise judging is the standard for open-ended output, and it is wrong by default.

**The protocol (five steps, none optional):**

1. **Pairwise, not pointwise.** "Which is better, A or B?" is far more reliable than "rate this 1–5" — scales drift between prompts and judges.
2. **Swap positions and average.** Run every pair twice: `(A,B)` and `(B,A)`. If the same model wins both orders, it is a real win. If A wins the first and B the second, it is a **tie** (the judge is position-biased and the pair is indistinguishable). Never report a win rate without the swap.
3. **Blind and shuffle order within a run**, and strip model-identifying strings.
4. **Calibrate against humans.** Label 100 pairs yourself. If your judge agrees <70%, your judge is measuring something else. Report the agreement number alongside the result.
5. **Fix the judge's prompt and version it.** A judge prompt change is a metric change; old numbers are not comparable.

**The biases, measured:**

| Bias | Magnitude | Fix |
|---|---|---|
| **Position bias** | 10–15 points of win rate attributable to order alone | Evaluate both orders; treat discordant pairs as ties |
| **Verbosity bias** | Longer answers win ~60–70% of "equal quality" pairs | Length-control the comparison, or penalise length in the judge rubric |
| **Self-preference** | A GPT-4 judge prefers GPT-4 outputs by ~10 points | Use a different judge family than the model you trained |
| **Sycophancy / rubric drift** | Judges agree with whatever framing the prompt implies | Do not put the hypothesis in the judge prompt |
| **Formatting bias** | Markdown-bulleted answers beat prose at equal content | Report format compliance separately and hold it constant |

```python
JUDGE_PROMPT = """You are comparing two responses to the same user request.
Judge ONLY on: factual correctness, completeness, and adherence to the requested format.
Ignore length, markdown, and writing style. Answer with a single letter.

USER REQUEST:
{request}

RESPONSE 1:
{r1}

RESPONSE 2:
{r2}

Which response is better? Answer "1", "2", or "tie"."""

def judge_pairwise(judge_client, request, r_a, r_b, model="gpt-4o"):
    """Returns 'A', 'B', or 'tie' with position-bias correction."""
    fwd = judge_client(JUDGE_PROMPT.format(request=request, r1=r_a, r2=r_b))
    rev = judge_client(JUDGE_PROMPT.format(request=request, r1=r_b, r2=r_a))
    a_score = 0
    if "1" in fwd[:3]: a_score += 1          # A in slot 1
    if "2" in rev[:3]: a_score += 1          # A in slot 2
    if a_score == 2: return "A"
    if a_score == 0: return "B"
    return "tie"                              # the two orders disagree

def win_rate(judge_client, pairs, model_a, model_b):
    from collections import Counter
    res = Counter(judge_pairwise(judge_client, p["request"],
                                 model_a(p), model_b(p)) for p in pairs)
    n = sum(res.values())
    return {"win": res["A"]/n, "loss": res["B"]/n, "tie": res["tie"]/n, "n": n}
```

### 12.5 The minimal eval harness — put this in your repo

```python
# eval_sft.py — run after every training run; exits non-zero on regression
import json, re, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

GOLDEN = json.load(open("eval/golden_200.json"))          # [{"prompt","must_contain","must_not_contain"}]
REFUSAL = ("I cannot", "I can't", "I'm unable", "As an AI", "I must decline")
BASE_ID = "meta-llama/Llama-3.1-8B"
TEMPLATE = open("templates/acme_pharma_v3.jinja").read()

def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"                              # generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16,
                                                device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok

@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens=256, greedy=True):
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        gen = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=not greedy, temperature=0.7 if not greedy else None)
        out.append(tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return out

def score(tok, outs):
    n = len(outs)
    contains = sum(all(s.lower() in o.lower() for s in g["must_contain"])
                   for g, o in zip(GOLDEN, outs)) / n
    forbidden = sum(any(s.lower() in o.lower() for s in g["must_not_contain"])
                    for g, o in zip(GOLDEN, outs)) / n
    refusal = sum(any(m.lower() in o.lower() for m in REFUSAL) for o in outs) / n
    mean_len = float(np.mean([len(tok(o).input_ids) for o in outs]))
    return {"must_contain_rate": contains, "must_not_contain_violation_rate": forbidden,
            "refusal_rate": refusal, "mean_len": mean_len}

def main(adapter):
    model, tok = load(adapter)
    outs = generate(model, tok, [g["prompt"] for g in GOLDEN])
    m = score(tok, outs)
    print(json.dumps(m, indent=2))
    # ---- the gates. Tune these once, then never loosen them. ----
    assert m["must_contain_rate"]      >= 0.90, "factual regression"
    assert m["must_not_contain_violation_rate"] <= 0.02, "format/content regression"
    assert m["refusal_rate"]           <= 0.03, "over-refusal"
    assert m["mean_len"]               <= 320,  "verbosity inflation"
    print("PASS")

if __name__ == "__main__":
    main(adapter="out/final")
```

Run this **three** times: on the SFT start checkpoint, on the candidate, and on the currently-served model. A candidate that beats the start but loses to the server is not shippable.

### 12.6 What "good" looks like — the scorecard

| Metric | Bad | Acceptable | Good | Excellent |
|---|---|---|---|---|
| Format compliance (held-out prompts) | <80% | 90% | 96% | 99%+ |
| Refusal rate (benign 200) | >8% | 5% | 2% | <1% |
| Win rate vs base (swap-corrected) | <50% | 55% | 65% | 75%+ |
| MMLU delta vs start checkpoint | <−5 pts | −3 | −1 | 0 |
| Response-length drift vs epoch 1 | >+50% | +25% | +10% | <+5% |
| Prompt-mutation gap (§4.7.2) | >25 pts | 15 | 8 | <5 |
| Contaminated eval rows | >2% | 0.5% | 0% | 0% |

---

## 13. Comparison Tables

### 13.1 SFT vs the other stages

| Dimension | Continued pretraining (CS-12) | **SFT (this module)** | DPO (CS-25) | ORPO (CS-27) | PPO/RLHF (CS-24) | GRPO (CS-26) |
|---|---|---|---|---|---|---|
| Data shape | raw text | (prompt, response) | (prompt, chosen, rejected) | same as DPO | prompts + reward model | prompts + verifier |
| Data volume | 10M–1B tokens | 1k–500k rows | 1k–50k pairs | 1k–50k pairs | 10k–1M prompts | 10k+ verifiable tasks |
| Signal density | 1 per token | 1 per token (masked) | 1 bit per pair | 1 bit per pair + SFT loss | scalar reward | scalar reward |
| Teaches | vocabulary, domain fluency | **format, tone, task behaviour** | preference between outputs | both at once | reward-shaped behaviour | verifiable reasoning |
| Cost (8B, QLoRA) | 20–200 GPU-h | 1–10 GPU-h | 1–5 GPU-h | 2–8 GPU-h | 20–200 GPU-h | 10–100 GPU-h |
| Needs a reward model | No | No | No | No | **Yes** | No (verifier) |
| Needs a reference model | No | No | Yes | No | Yes | Yes |
| Risk | forgetting general ability | forgetting, rigidity, over-refusal | undoing SFT knowledge | both | reward hacking | reward hacking |
| Typical win over the previous stage | +10–20 domain benchmark pts | **+30–50 pts format compliance** | +5–15 win-rate pts | ≈ SFT+DPO | +5–20 win-rate pts | +10–30 task pts |
| Complexity | Low | Low | Medium | Medium | **High** | Medium-High |

### 13.2 SFT implementation methods

| Method | Trainable | VRAM (8B) | Quality | Speed | When |
|---|---|---|---|---|---|
| **Full FT** | 100% | 112 GB static | Best at large data | 1× | >50k rows, multi-GPU, you own the hardware |
| **LoRA** | ~0.3–1% | 16–30 GB | 95–99% of full FT | 1–2× | The default for <50k rows |
| **QLoRA** | ~0.3–1% | 6–12 GB | 92–98% of full FT | 0.6–0.9× | One consumer GPU, prototyping |
| **DoRA** | ~0.3–1% | like LoRA | +0–2 pts over LoRA | 0.9× | When you want the last point |
| **LoRA + `all-linear`** | ~0.6% | 20–30 GB | ≈ full FT on domain tasks | 1× | **Recommended default** |
| **Prompt/prefix tuning** | <0.1% | lowest | 80–90% of LoRA | 1× | Research; rarely the right production call |

### 13.3 Frameworks for SFT

| Framework | Interface | Masking | Multi-turn | Packing | Best for |
|---|---|---|---|---|---|
| **HF `Trainer`** (the notebook) | Python | **Manual** | Manual | No | Learning; full control; tiny data |
| **TRL `SFTTrainer`** | Python | `assistant_only_loss=True` | Yes | `packing=True` | The standard Python default |
| **Unsloth** | Python | `train_on_responses_only` | Yes | Yes | 2–4× faster, single consumer GPU (CS-16) |
| **LLaMA-Factory** | WebUI/YAML/CLI | `train_on_prompt: false` | Yes | Yes | No-code; 100+ models (CS-15) |
| **Axolotl** | YAML | `train_on_inputs: false` | Yes | Yes (`neat_packing`) | Reproducible config-driven pipelines (CS-17) |
| **OpenAI fine-tuning API** | REST | implicit (messages) | Yes | n/a | When the base is a GPT model (CS-18) |
| **Vertex AI** | REST/CLI | implicit | Yes | n/a | Gemini models (CS-19) |

### 13.4 Should you SFT a base model or an instruct model?

| | Base + SFT | Instruct + SFT |
|---|---|---|
| Starting behaviour | Continuation | Assistant-like |
| Data needed for format | More (you teach the template from scratch) | Less (template already learned) |
| Over-refusal risk | Low | **High** (the safety prior is already sharp) |
| Catastrophic forgetting risk | Higher (full FT on a base drifts more) | Lower (already near an assistant optimum) |
| LR | 2e-4 LoRA / 2e-5 full | **half** of the base numbers |
| Epochs | 2–3 | 1–2 |
| Best when | You already did CS-12 domain adaptation; you want maximal control | You want a style/format change on a capable model, fast |
| Typical production choice | Domain-specific assistants with unusual formats | Most product fine-tunes |

---

## 14. Debugging Playbook

| # | Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|---|
| 1 | Loss decreases smoothly but generations are choppy/truncated | Padding trained as targets | `assert all(l == -100 for l, a in zip(labels, attn) if a == 0)` | Mask `attention_mask == 0` positions |
| 2 | Loss is suspiciously low from step 0 | Prompt unmasked and/or eval contaminated | Print `labels[:30]`; run 13-gram decontamination | Mask the prompt; clean the eval set |
| 3 | Loss goes to ~0 quickly and the model recites the training data | Overfit; dataset too small / too many epochs | Sample 20 generations; check verbatim overlap with train rows | Fewer epochs, more data, LoRA instead of full FT, add dropout |
| 4 | Loss is flat at ~2.5 for 500 steps | LR too low, or the adapter is attached to nothing | `model.print_trainable_parameters()`; check `target_modules` matched (a wrong name silently attaches zero adapters in some peft versions) | Raise LR 3–10×; fix `target_modules` |
| 5 | Loss oscillates ±0.5 between logging steps | LR too high / batch too small | Plot the raw curve; check tokens/step | Lower LR 3×; increase `gradient_accumulation_steps` |
| 6 | Loss is NaN | All-`-100` row; fp16 overflow; LR too high | `audit_masks`; grep the log for the first NaN step; try `bf16` | Filter zero-supervision rows; switch to bf16; lower LR; `max_grad_norm=1.0` |
| 7 | Loss spikes at a specific step then recovers | One pathological example (very long, or a degenerate target) | Find the example at that step (accumulation × step index) | Filter outliers; clip gradients; shuffle with a fixed seed to reproduce |
| 8 | Train loss ↓, eval loss ↑ | Classic overfit | Compare epochs | Stop earlier; add data; add replay |
| 9 | Train loss ↓, eval loss flat | Overfit in the SFT-specific sense | Check verbosity, rigidity, refusal, MMLU | §4.7 mitigations |
| 10 | Eval loss is much lower than train loss | Different masking, or eval data is easier/duplicated | Compare the mask applied to both sets | Apply the same masking policy |
| 11 | Model ignores the system prompt at serving time | Template mismatch or dropped system role | Render the serving string; compare bytes with training | Pin one template; serve it; check the server does not strip `system` |
| 12 | Model double-emits the response header (`### Response:` twice) | Header trained as a target (not masked) | Check whether `labels` for the header indices are `-100` | Extend the mask past the header |
| 13 | Model answers but never stops | No EOS in the training response spans | Check the last non-`-100` label equals `eos_token_id` | Append EOS when building the target string; do not strip it |
| 14 | Model produces the literal word "None" or "nan" | `None`/NaN interpolated into the template | grep the rendered dataset | `example.get("input") or ""` |
| 15 | Identical output to the base model → "nothing happened" | Adapter not loaded (see §6.2/§6.10); LR too low; zero trainable params | Hash a weight before/after loading; `print_trainable_parameters()` | Load with `PeftModel.from_pretrained` |
| 16 | Quality drops for prompts phrased differently | Format rigidity | Prompt-mutation sweep (§4.7.2) | More instruction-style diversity; fewer epochs; NEFTune |
| 17 | Answers get longer every checkpoint | Verbosity inflation | Mean-length tracking on fixed prompts | Stop earlier; add short-answer rows; length-normalised early stopping |
| 18 | Refuses legitimate requests | Over-refusal | Benign 200-prompt suite | Remove refusal rows; lower epochs; mix task data; lower LR |
| 19 | MMLU/GSM8K collapse | Catastrophic forgetting | Run a general benchmark before/after | LoRA, fewer epochs, 5–20% replay, lower LR |
| 20 | Multi-turn conversations degrade after turn 1 | Only the final assistant turn was masked/kept | Count trained tokens per turn | `assistant_only_loss=True`; verify per-turn |
| 21 | Training OOMs at step X | A long sequence in the batch | Log `input_ids.shape` per batch; `group_by_length=True` | Lower `max_length`; raise `gradient_accumulation_steps`, lower micro-batch; gradient checkpointing |
| 22 | Eval loss is fine, but the judge prefers the base model | The SFT taught your data's quirks, not the task | Run the win-rate harness with position swap | Improve data quality; reduce epochs; check the judge for verbosity bias |
| 23 | Different runs give different results with the same config | Seeds not fixed, dataloader workers, CUDA nondeterminism | `set_seed(42)`, `data_seed=42`, `dataloader_num_workers=0` | Fix seeds; otherwise treat ±3 points as noise |
| 24 | Loss decreases but the model's format is intermittent (80–90%) | Not enough examples / epochs; or the format is hard | Format-compliance curve per checkpoint | More epochs on more data; add 200 pure-format rows |
| 25 | Adapter works in Python, fails in vLLM | LoRA not enabled in the server, or wrong template at the server | `--enable-lora --lora-modules` and the served chat template | See §16.3 |

---

## 15. Applied Case Studies

### 15.1 The video's own case study — 5 rows, TinyLlama, and what it actually proves

| | |
|---|---|
| **Situation** | A pharma company wants an assistant that understands its terminology and answers in a consistent clinical format [16:31]–[17:35]. Before CS-12 continued pretraining, the base model does not know the term "pharmacovigilance" and hallucinates around it. |
| **Why SFT** | The requirement is not knowledge, it is *behaviour*: answer the question, in the right structure, in the domain register, without rambling. That is exactly the axis SFT moves (§4.1). |
| **Data** | `pharma_instruction_data.jsonl`, 5 rows, `instruction`/`input`/`output`. Two rows use `input` (mRNA vaccines, AI in pharma R&D); three leave it empty. Converted from a 5-row CSV in the video [40:45] "because CSVs are ambiguous about empty fields". |
| **Base model** | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` — a base model, not a chat model [31:46]. |
| **Technique** | LoRA r=8, α=16, dropout 0.05, `q_proj`/`v_proj` [46:53]; 3 epochs; bs 1 × ga 8; LR 2e-4; fp16 [48:11]. |
| **Result** | 3 optimizer steps. ~480 response tokens supervised against ~61,000 pad targets (§6.9). The video demonstrates the pipeline end-to-end and gets fluent-looking output [50:09]. |
| **What went wrong first** | Almost everything that makes this a good teaching example: the tokenizer came from a different checkpoint than the model; `format_example` wrote the literal string `None` into training rows; the "instruction-tuned" model and the "non-instruction" model produced byte-identical output; only 5 rows were used; and the prompt was not masked. |
| **The honest verdict** | This notebook is a **mechanism demonstration, not a recipe**. It shows you every moving part — dataset → format → tokenize → mask → LoRA → Trainer → generate — at a scale where you can read every tensor. It is not evidence that the technique works at 5 rows, and the video never claims it is. Take the pipeline, throw away the hyperparameters, and supply your own 2,000 rows. |

The video's own concluding instruction, verbatim [54:10]:

> *"first teach your model on your domain specific data, i.e. non-instructional fine-tuning, and then instruction tuning"*

and the comparison verdict at [53:50]: *"According to me, the previous one is a good one"* — i.e. the **unmasked** variant produced the better-looking answer on the three test questions. This is a 3-question evaluation on a 5-row dataset and carries no statistical weight (§4.4.3, §12.1). Do not take it as a masking recommendation.

### 15.2 Regulated-domain assistant — 4,000 rows, 8B base, QLoRA

| | |
|---|---|
| **Situation** | A mid-size insurer needs a claims-triage assistant. Input: a free-text claim description. Output: `{"category": ..., "urgency": "routine\|priority\|urgent", "missing_documents": [...], "rationale": "..."}`. The output must be valid JSON 100% of the time — a downstream service parses it. |
| **Why SFT** | The task is a fixed transformation with a strict schema. RAG cannot enforce a schema, and prompt engineering got to ~88% valid JSON with a 4,000-token prompt on every call. SFT moves it to ~99.5% *and* cuts the prompt to 300 tokens. |
| **Pipeline** | 3,000 real historical claims (de-identified, human-relabelled where the original adjuster's decision was wrong — 11% of rows) + 1,000 synthetic from a stronger model, all human-reviewed. |
| **Data shape** | OpenAI `messages` internally; exported to ShareGPT for LLaMA-Factory. Multi-turn not needed → single turn. 40% of rows carry a system prompt (the schema spec), 60% do not — deliberately, so the model works in both serving modes. |
| **Config** | `Llama-3.1-8B-Instruct` + QLoRA r=16 `all-linear`, LR 1e-4, 2 epochs, bs 2 × ga 8 × seq 2048, cosine, warmup 5%, `neftune_noise_alpha=5`, packing on, `assistant_only_loss=True`. |
| **Cost** | 3,000 rows × ~900 tokens × 2 epochs = 5.4M tokens → ~25 min on one A100-40 → **$0.63**. Dataset creation: **$18,000** mostly in human relabelling. Eval: 6 h of a clinician's time. |
| **Result** | JSON validity 88% → 99.6%. Urgency accuracy 71% → 89% against the relabelled gold set. Over-refusal 0% → 0.4% (acceptable). MMLU −1.2 pts. |
| **What went wrong first** | (1) Epoch 3 with LR 2e-4 pushed JSON validity to 99.8% but mean response length from 210 → 480 tokens and urgency accuracy *down* to 84% — the model learned to write plausible-sounding rationales that argued for the wrong urgency. Fixed by stopping at 2 epochs and lowering LR to 1e-4. (2) The first eval set leaked: 14 of 200 eval rows had near-duplicates in train (found by 13-gram overlap), inflating the reported validity to 99.9%. Decontaminated, it was 98.9%. (3) The serving stack stripped the system prompt for 2 weeks before anyone noticed — the model had a 60/40 split so it degraded gracefully to 94% rather than breaking, which is why the 60/40 split was the right call. |

### 15.3 Style/format transfer onto a capable instruct model — 800 rows

| | |
|---|---|
| **Situation** | A developer-tools company wants its docs assistant to answer in "house voice" (second person, no hedging, always ends with a runnable snippet). The base is already a strong assistant; only the *style* is wrong. |
| **Why SFT** | Pure behaviour change. The best evidence in the field says this needs hundreds of examples, not thousands (§4.1, LIMA). |
| **Data** | 800 hand-written pairs by 3 technical writers over 4 days: 400 real support tickets with a house-voice answer, 400 synthetic variations of 60 seed prompts (paraphrase + format variations). Deliberately includes 120 rows where the *correct* answer is "this cannot be done in the current version" — to prevent the style tuning from also teaching over-confidence. |
| **Config** | `Qwen2.5-7B-Instruct` + LoRA r=16 `all-linear`, LR **5e-5** (half the usual, because the model is already aligned and we want minimal drift), 1 epoch, bs 4 × ga 4, seq 1536, cosine, warmup 3%. |
| **Cost** | 800 × 600 × 1 = 0.48M tokens → ~8 min on an L4 (24 GB). **<$0.05** in GPU time. |
| **Result** | House-voice adherence (human-rated, 100 samples): 34% → 91%. Win rate vs base with position swap: 68% / 19% / 13% tie. MMLU unchanged (within noise). Response length +9%. |
| **What went wrong first** | (1) At 3 epochs, the model began *inserting* snippets into answers where a snippet was not appropriate — format rigidity appearing exactly as §4.7.2 predicts; the style had become a compulsion. (2) The first version of the dataset had all-nice answers, and the tuned model stopped saying "this is not supported"; the 120 negative rows fixed it. (3) The serving stack's default `max_tokens` was 256, truncating the trailing snippet — the format compliance metric was measuring the server, not the model. |

### 15.4 Multi-turn tool-calling agent — 12,000 rows, the hard case

| | |
|---|---|
| **Situation** | A support agent that must call 6 internal tools across up to 9 turns, holding slot state. |
| **Why SFT** | The tool-call *syntax* and the decision policy are a format. Without SFT, a 7B model emits a valid-looking call ~70% of the time and forgets the slot in ~30% of 5-turn conversations. |
| **Data** | 12,000 conversations synthesised by replaying 4,000 real ticket threads against a stronger model with a tool sandbox, keeping only trajectories where the tool results were ground-truth-verifiable. Distribution: 60% ≤3 turns, 30% 4–6 turns, 10% 7–9 turns — the length mix is deliberate, because training mostly on short conversations produces a model that loses the thread (§10.1). |
| **Config** | `Mistral-7B-Instruct-v0.3` + LoRA r=32 `all-linear` (higher rank — the task is harder than style), LR 8e-5, 2 epochs, bs 1 × ga 16, seq 4096, packing on, `assistant_only_loss=True`, gradient checkpointing on. |
| **Result** | Valid tool-call syntax 71% → 98.2%. Multi-turn slot retention (7-turn) 68% → 91%. Wrong-tool rate 14% → 4%. |
| **What went wrong first** | (1) **Training on tool-result messages.** The first run masked only the *user* turns, so the model was trained to predict its own tool's JSON output — it started hallucinating plausible-but-fake API responses instead of emitting a call. `assistant_only_loss=True` fixed it. This is the single most common agent-SFT bug. (2) Packing across conversation boundaries without position-id resets caused cross-conversation attention bleed — turn 9 of conversation A attended to conversation B. Fixed with `DataCollatorWithFlattening(return_position_ids=True, return_flash_attn_kwargs=True)` (see §4.5). (3) The eval used single-turn prompts, so it reported 99% while production failed on turn 3 — the eval set now mirrors the turn-length distribution of production logs. |

### 15.5 The "we tried SFT and it did not work" post-mortem

The most instructive case, because it is the most common.

| Symptom | Actual root cause | What they believed |
|---|---|---|
| Output identical to base | The LoRA adapter was loaded with `AutoModelForCausalLM.from_pretrained(adapter_dir)`, which silently ignored the adapter weights (the notebook's own bug, §6.2) | "SFT does nothing on this model" |
| Answers fluent but ignoring instructions | `labels = input_ids.copy()` — the model was trained to generate the *prompt*, so it produced plausible continuations of the user's text rather than answers | "the model is too small" |
| Format right in the notebook, wrong in the API | Notebook used the tokenizer's default template; the API server used a hand-rolled one with `\n\nHuman:` | "the fine-tune is unstable" |
| Quality good for the 20 demo prompts, bad in the field | 3 of the 20 demo prompts were in the training set; the eval set was the last 200 rows of the training file | "it overfits, we need RLHF" |
| Refuses 30% of legitimate requests | 40% of the training rows were refusals copied from a safety dataset | "the base model is over-cautious" |

**Total cost of that project before the post-mortem:** $1,900 (mostly the labelling contract) and 6 weeks. The fix took 2 days. Every one of those five failures is detectable in under an hour with the harness in §12.5.

---

## 16. Production Considerations

### 16.1 Dataset versioning — the artifact that matters most

The adapter is regenerable in 30 minutes. The dataset is not. Treat it like source code.

```
data/
  pharma_sft/
    v1.0.0/
      train.jsonl               # 8,000 rows, sha256 in the manifest
      eval.jsonl                # 200 frozen rows, NEVER used for training
      MANIFEST.json             # see below
      CHANGELOG.md              # what changed and why, one line per row-batch
    v1.1.0/
      ...
```

```json
{
  "dataset_id": "pharma_sft",
  "version": "1.1.0",
  "created_utc": "2025-09-14T10:22:31Z",
  "n_train": 8214,
  "n_eval": 200,
  "sha256": {
    "train.jsonl": "9f2c...",
    "eval.jsonl": "41ab..."
  },
  "sources": {
    "human_authored": 3100,
    "human_relabelled": 214,
    "synthetic_gpt4o": 4900
  },
  "filters_applied": ["dedup_minhash_0.9", "decontam_13gram", "len_p99_clip", "json_valid"],
  "template_sha256": "c7d1...",
  "seed": 42,
  "parent_version": "1.0.0",
  "notes": "added 214 clinician-corrected rows; removed 88 rows that overlapped the eval set"
}
```

Three rules:

1. **The eval set is frozen and versioned with the dataset.** Changing it invalidates every historical number.
2. **The training run records the dataset hash and the template hash**, not just the version string. If a hash is missing, the run is unreproducible.
3. **Never mutate a version in place.** A row fix is a patch version; a distribution shift is a minor version (semver for data).

### 16.2 Reproducibility

```python
import os, random, numpy as np, torch, transformers

def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # required for deterministic CUDA
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

# In TrainingArguments:
TrainingArguments(
    seed=42,
    data_seed=42,
    dataloader_num_workers=0,       # worker RNG is a silent nondeterminism source
    dataloader_drop_last=False,     # keep the tail batch; changing it changes the step count
    full_determinism=True,          # slower; set False if you accept ±small drift
)
```

**What reproducibility you can actually get:**

| Level | Achievable | Requires |
|---|---|---|
| Same machine, same GPU, same libs, `full_determinism=True` | Bitwise identical | All of the above + no flash-attention (flash-attn is not deterministic in backward) |
| Same machine, `full_determinism=False` | ±0.1% loss curves | Seeds only |
| Different GPU (A100 → H100) | ±2–5% on downstream eval | Seeds + tolerance bands |
| Different library minor version (TRL 0.11 → 0.12) | Different results | Pin exact versions in `requirements.txt` with `==` |

The practical rule: **reproduce the pipeline deterministically, and the model statistically.** Write down the seed, pin the versions, and set your eval gates with enough margin that ±5% model variance does not flip the decision.

### 16.3 Merging and serving

```python
# 1. Merge the adapter into the base weights (one-time, ~2 min on CPU, ~30 s on GPU)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Llama-3.1-8B"
ADAPTER = "out/final"
MERGED = "out/merged"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="cpu")
tok = AutoTokenizer.from_pretrained(BASE)
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained(MERGED, safe_serialization=True)
tok.save_pretrained(MERGED)
# CRITICAL: the merged repo must carry the SAME chat template the training used
tok.chat_template = open("templates/acme_pharma_v3.jinja").read()
tok.save_pretrained(MERGED)
```

| Serving route | Command / config | Notes |
|---|---|---|
| **vLLM, merged** | `vllm serve out/merged --max-model-len 8192` | Simplest; best throughput |
| **vLLM, adapter hot-swap** | `vllm serve $BASE --enable-lora --lora-modules pharma=out/final --max-lora-rank 16` | Multiple adapters, one base; **must** pass `--chat-template templates/acme_pharma_v3.jinja` |
| **TGI** | `text-generation-launcher --model-id out/merged` | Reads `chat_template` from the tokenizer config |
| **Ollama** | create a `Modelfile` with `TEMPLATE """..."""` | The `TEMPLATE` block must reproduce the training template exactly, or the model degrades |
| **Transformers** | `tokenizer.apply_chat_template(...)` | Reference implementation; use it to generate the golden strings for the other routes |

**The three serving checks everyone forgets:**

1. **Template byte-equality.** Render the same conversation through the training tokenizer and through the server, and diff the strings. Do it in CI.
2. **Special-token handling.** Merged models need `--skip-special-tokens` semantics preserved; a server that decodes `<|eot_id|>` into visible text will confuse the client.
3. **`max_tokens` is long enough** for the longest expected answer. Truncating the tail of a structured output is the most common "the model is broken" production report (§15.3).

### 16.4 Regression suites and rollout

```yaml
# .github/workflows/sft-regression.yml  (fragment)
- name: Render-template check
  run: python scripts/check_template_parity.py --template templates/acme_pharma_v3.jinja
- name: Golden-set eval
  run: python eval_sft.py --adapter out/final        # the harness from §12.5
- name: Forgetting check
  run: python eval_general.py --adapter out/final --max-mmlu-drop 3.0
- name: Latency check
  run: python bench_latency.py --adapter out/final --p95-max-ms 1200
```

**Rollout protocol:**

| Step | Action | Gate |
|---|---|---|
| 1 | Offline eval on the frozen 200 | All §12.5 gates pass |
| 2 | Shadow traffic (log-only, 1% of real requests) | Format compliance within 1 pt of the incumbent; no new failure class in logs |
| 3 | Canary 5% | p95 latency within 20%; refusal rate within 1 pt; no user escalations attributable |
| 4 | 25% → 50% → 100% | Each step holds for 24 h with no regression |
| 5 | Keep the previous adapter warm for 30 days | One-flag rollback (`--lora-modules pharma=v_prev`) |

**Rollback is a config change, not a retrain.** If your rollback requires training, you do not have a rollback.

### 16.5 Compliance and governance

| Requirement | SFT-specific action |
|---|---|
| Data provenance | Every row carries a `source` field. You must be able to answer "where did this knowledge come from" per row |
| Licensing | Dataset licence tracked in `MANIFEST.json`; **Alpaca and other `text-davinci-003`-derived sets carry an OpenAI-ToS restriction on commercial use** (§4.6.8) |
| PII | Scan both the prompt and the response spans; de-identified before the row enters the dataset, not after |
| Right to erasure | Row-level deletion from a versioned dataset means retraining. Design for it: keep the dataset small enough to retrain in a day, and record which adapter version each request was served by |
| Model cards | Record: base model, dataset version + hash, template hash, hyperparameters, eval numbers, known failure modes, intended use |
| Audit | Log `(request_id, model_version, dataset_version, template_version)` for every served request |
| Safety evaluation | Run a red-team set *after* SFT. Fine-tuning on domain data measurably weakens safety behaviour, and the size of that degradation is model- and data-dependent — never assume the base model's safety carried over |

---

## 17. Common Misconceptions

| # | People believe | Actually | Because |
|---|---|---|---|
| 1 | "Fine-tuning teaches the model new facts." | SFT teaches *how to answer*, and injects facts only weakly and unreliably. | The superficial-alignment hypothesis (§4.1). Knowledge is 10⁴–10⁶× more token-efficient to install in prompt/RAG context than in weights, and later alignment stages erode what SFT did inject. |
| 2 | "More epochs = better SFT." | Almost always worse after 2–3. | The training data is a *sample of a behaviour*, not the behaviour itself. Past the point where the format is learned, extra epochs only sharpen the sample's idiosyncrasies. |
| 3 | "Loss is the metric." | Held-out loss is the *weakest* available signal for SFT. | A model at 5 epochs has lower loss and produces worse output. Track format compliance, refusal rate, length drift, and win rate. |
| 4 | "The prompt should be trained too — it is more signal." | Masking the prompt is the industry default and the correct default. | Training on the prompt dilutes the gradient with a task you do not want (copying) and biases the loss towards long prompts. |
| 5 | "SFT needs 50,000 examples." | 1,000 curated examples produced a model that wins 43% of the time against GPT-4 (§4.1). 9k filtered beat 52k unfiltered. | Diversity and correctness per example matter far more than count. |
| 6 | "Any instruct model works with any template." | Templates are a per-model contract. | The weights were trained to expect specific control tokens. A mismatch produces a model that looks fine and performs badly (§4.3.4). |
| 7 | "The chat template only matters at serving." | It matters identically at training, and the *training* string is the one people get wrong. | The default template in a tokenizer is not necessarily the one the model was trained with. Always render and read it. |
| 8 | "LoRA is nearly as good, so just use QLoRA." | QLoRA is 92–98% of full FT on domain tasks and *lower* on tasks needing new knowledge or precise recall. | The 4-bit base cannot represent the fine distinctions that knowledge injection requires. Use QLoRA to prototype, then re-run with LoRA/full FT if the eval says so. |
| 9 | "SFT then DPO — DPO is where the quality comes from." | SFT is ~90% of the achievable value; DPO's gains *require* a good SFT base. | DPO is a preference between two outputs. If both are bad, the preference signal teaches almost nothing (§4.8). |
| 10 | "The model must be re-served after every checkpoint." | Merge the adapter, and rolling back is a config change. | Adapters decouple the training artifact from the serving artifact. |
| 11 | "If the eval number went up, it improved." | With 200 eval rows, ±3 points is noise. | Binomial standard error at p=0.9, n=200 is ≈2.1 points. Use 500+ rows and a confidence interval, or accept that you cannot resolve small differences. |
| 12 | "We can evaluate with the same model we trained." | Self-preference bias is ~10 points. | Use a different judge family; calibrate against ≥100 human labels. |
| 13 | "Packing is just a speed trick." | Packing changes the attention structure and needs position-id resets. Done wrong, it silently corrupts training. | Sequences in a pack attend to each other unless boundaries are enforced (§4.5.4). |
| 14 | "Catastrophic forgetting is only a full-FT problem." | LoRA reduces it; it does not eliminate it. Multi-epoch LoRA on narrow data still drops general benchmarks by 2–5 points. | The adapter still shifts the shared representation, and in multi-epoch runs the drift compounds. |
| 15 | "A bigger base model needs more SFT data." | The opposite is closer to the truth: a more capable base needs *less* data to reach a given behaviour. | Capability is already there; you are eliciting a format from a larger space of already-learned behaviours. |

---

## 18. Key Takeaways

1. **SFT is next-token prediction on `(prompt, response)` pairs with the prompt's tokens excluded from the loss.** Everything else in this module is detail on that sentence.
2. **The number to remember is 1,000.** LIMA reached 43% win/tie against GPT-4 with 1,000 curated examples on a 65B base. Your 8,000 rows are not the reason your model is good; your *curation* is.
3. **SFT elicits format far more effectively than it injects knowledge.** Design for behaviour: schema, tone, register, refusal policy, stopping. Get facts from retrieval and from continued pretraining (CS-12).
4. **Masking the prompt is the single most important implementation detail.** `labels[prompt_positions] = -100`, and mask the padding too. Everything you care about — gradient dilution, length balance, format compliance — follows from it.
5. **Padding is not free; it is target tokens.** With `padding="max_length"` and unmasked labels, the notebook's run spends 61,000 label positions on `<pad>` and 480 on the answer.
6. **Read the rendered training string.** The chat template is a contract with the weights; a mismatch produces fluent, wrong behaviour and no error message.
7. **The five data formats are the same information.** Alpaca, Alpaca-flat, ShareGPT, OpenAI chat, DPO pairs, completion-only. Pick the one your framework eats; convert with a versioned script you can unit-test.
8. **One great example beats ten mediocre ones, and it is measurable.** 9k filtered from 52k beat the full 52k; 6k curated beat 100k+.
9. **Diversity beats volume, on three axes:** task type, instruction phrasing, and response length. Measure all three with code, not intuition.
10. **Epochs 1–3, and quality turns before loss does.** If your 5-epoch run has the best loss and the worst output, nothing is wrong with your training loop.
11. **Batch size in tokens, not examples.** 32k–128k tokens per optimizer step is the useful band; `global_batch × seq_len` is the number that matters.
12. **Learn the four overfitting signatures** — verbosity, format rigidity, over-refusal, forgetting — and apply the mitigations in order: replay > fewer epochs > LoRA > lower LR.
13. **SFT is roughly 90% of the achievable value.** Do SFT well before you consider DPO; a bad SFT base makes preference tuning a rounding error.
14. **Build the eval harness before the first training run.** A frozen 200-prompt set with format compliance, refusal rate, length, and a swap-corrected win rate. Without it, every later decision is a guess.
15. **The GPU is not the cost.** A 2,000-row expert dataset at 20 minutes a row is ~$33,000; the training run it feeds is under $1.

---

## 19. Self-Check Questions

<details>
<summary>1. What exactly does SFT change about a model, and what does it not change?</summary>

It changes the *conditional distribution of the response given a prompt* — which format, register, length, structure, and refusal policy the model adopts. It does not reliably add factual knowledge: the model's knowledge and capabilities are largely established in pretraining (the superficial-alignment hypothesis), and SFT selects which subdistribution of learned behaviours to surface. It also does not change the tokenizer, and it does not raise the ceiling on reasoning or factual recall.
</details>

<details>
<summary>2. Why is prompt masking the default, and when is training on the prompt legitimate?</summary>

Masking removes the prompt tokens from the loss — both numerator and denominator. Training on them dilutes the response gradient with a copying task you do not want, biases the loss towards examples with long prompts, and lets a handful of long-prompt rows dominate the update. It is legitimate when: (a) the prompt is *generated* content you want the model to be able to produce (e.g. synthesising the user query in a conversation), (b) you are training a base model that you also want to continue as a plain LM, (c) very short prompts with a strong continuation requirement, or (d) you are matching a specific published recipe and can justify it. In all four cases the effect must be measured, not assumed.
</details>

<details>
<summary>3. What does `-100` mean in a labels tensor, and what happens to those positions in the loss?</summary>

`-100` is `ignore_index` in PyTorch's `CrossEntropyLoss`. After the shift, any target position equal to `-100` contributes nothing to the numerator *and* nothing to the denominator — it is removed from the mean entirely. The forward pass is unaffected: `-100` is in `labels`, not `input_ids`, so attention and hidden states are computed over the full sequence, including the masked prefix. The model still *reads* the prompt; it is only not *graded* on predicting it.
</details>

<details>
<summary>4. Your loss is dropping nicely but the model answers in a rigid, over-long style. What happened, and what do you change?</summary>

Two overfitting signatures — verbosity and format rigidity — appearing together, which is the normal pattern past ~2–3 epochs. The model has learned your training sample's idiosyncratic length and structure rather than the task. Change, in order of effectiveness: (1) add 5–20% replay/general instruction data to break the mode; (2) reduce epochs; (3) switch to LoRA (or lower the LoRA rank) if you were full-FT; (4) lower the LR; (5) add NEFTune; (6) stop earlier using a length-drift early-stopping rule rather than loss.
</details>

<details>
<summary>5. How do you evaluate an SFT model, and which metric is most misleading?</summary>

Four layers: (1) mechanical — held-out loss, format compliance, refusal rate, mean length drift, contamination check; (2) task — exact-match/regex/JSON-schema validity, or a domain rubric; (3) comparative — swap-corrected LLM-judge win rate against the start checkpoint; (4) regression — a general benchmark (MMLU/GSM8K subset) to measure forgetting. The most misleading is held-out loss: it is monotonically improved by exactly the behaviour you do not want (memorising the sample), so a falling loss is compatible with a worse model.
</details>

<details>
<summary>6. Walk through the mask tensor for `[BOS] + prompt(24) + response(32) + PAD`, with `max_length=512`, under `padding="max_length"` + `labels=input_ids.copy()`.</summary>

Indices 0–56 are real tokens (BOS + 24 prompt + 32 response) and 57–511 are `<pad>`. The notebook's scheme produces labels equal to input_ids everywhere — so the loss is computed over all 512 positions, including 25 prompt positions and 455 pad positions. Only 32 of 512 positions (6.3%) supervise the answer. The correct scheme sets indices 0–24 and 57–511 to `-100`, leaving 31 shifted targets (32 response positions, the last of which has no successor within the sequence) supervising the answer. With packing instead of padding, all 455 pad positions disappear.
</details>

<details>
<summary>7. When does SFT inject knowledge, and why is that not a reason to use it for facts?</summary>

SFT injects knowledge when: the fact is repeated across many paraphrases (500+), the base model has the relevant latent capability, the knowledge is not contradicted later, and no subsequent preference stage is applied. It is still the wrong tool because the token efficiency is terrible (10⁴–10⁶× worse than in-context), the injection is brittle under paraphrase, and it degrades over subsequent training. Use continued pretraining (CS-12) for the domain corpus, and retrieval (CS-04) for facts that change.
</details>

<details>
<summary>8. What is the difference between a chat template and a data format, and why does conflating them break things?</summary>

The **data format** is how your dataset stores a conversation (Alpaca columns, `conversations[]`, `messages[]`). The **chat template** is how a conversation is serialised into the single token string the model consumes, including control tokens and role markers. A format is a container; a template is a serialisation contract with the weights. Conflating them produces the classic failure of converting data correctly and then rendering it with the wrong template — the pipeline looks clean, the loss falls, and the model is subtly wrong (§4.3.4).
</details>

<details>
<summary>9. Why is `padding="max_length"` with `max_length=512` on a 5-row dataset with ~96-token examples a problem, in numbers?</summary>

Every row is padded to 512 tokens. With `labels = input_ids.copy()`, all 512 positions are targets. Total supervised positions across the run = 5 rows × 512 × 3 epochs = 7,680, of which ~7,200 are `<pad>` and ~480 are real answer tokens — 6%. The gradient spends 94% of its budget learning to predict padding, and the reported loss is dominated by an easy-to-predict constant, so it looks deceptively good. The fix is a padding-aware mask, and the better fix is packing.
</details>

<details>
<summary>10. You have 3,000 domain rows and a deadline. Which method, which hyperparameters, and which evals?</summary>

Method: LoRA (not QLoRA unless you are memory-bound) r=16, `target_modules="all-linear"`, α=16, dropout 0.05 on an *instruct* base if one exists. Hyperparameters: LR 1e-4, 2 epochs, cosine, warmup 5%, `tokens_per_step` ~64k (`bs 4 × ga 4 × seq 4096`, or `bs 2 × ga 8 × seq 2048` on a smaller GPU), `max_seq_length` at the 95th percentile of your rendered lengths with `response_lost < 0.5%`, packing on, `neftune_noise_alpha=5`, prompt masked. Evals: a frozen 200-prompt set with format compliance ≥95%, refusal ≤2%, mean-length drift <15%, swap-corrected win rate vs the base ≥60%, and an MMLU subset drop ≤3 points. Budget: 3,000 × ~700 tokens × 2 epochs ≈ 4.2M tokens ≈ 20 minutes on one A100-40 ≈ **$0.50**.
</details>

---

## 20. Cross-References

| Module | Relationship to CS-13 |
|---|---|
| **CS-01** Foundations: Pretraining & the LLM Lifecycle | Where SFT sits as stage 2 of 3; the pretraining objective that SFT modifies |
| **CS-02** Transfer Learning & Model Fine-Tuning | The transfer-learning framing; why freezing layers predates PEFT [7:09] |
| **CS-04** Fine-Tuning vs RAG vs Agents | **Read before you start SFT.** The decision table for whether you need it at all |
| **CS-06** Hugging Face Masterclass | `AutoModel`, `AutoTokenizer`, `Trainer`, datasets, `push_to_hub` — the API layer this module assumes |
| **CS-12** Domain-Adaptive Continued Pretraining on Your Own PDFs | **The prerequisite in practice.** Non-instructional FT first, then instruction FT [18:24]–[19:30] |
| **CS-14** The Alignment Map: RLHF, PPO, DPO, ORPO | Where SFT fits and what comes after; the preference-stage theory |
| **CS-15** LLaMA-Factory | No-code SFT; `dataset_info.json`, `formatting: alpaca\|sharegpt`, `train_on_prompt: false` |
| **CS-16** Unsloth | The 2–4× faster SFT path; `train_on_responses_only` |
| **CS-17** Axolotl | YAML-driven SFT; `train_on_inputs: false`, `roles_to_train`, `neat_packing` |
| **CS-18** OpenAI GPT Fine-Tuning | Managed SFT; the `messages` format; token pricing; when not to self-host |
| **CS-19** Vertex AI / Gemini Fine-Tuning | The same, on Gemini |
| **CS-20** Small Language Models | SFT a 0.5–3B model; the data-efficiency argument is strongest here |
| **CS-23** LoRA & QLoRA — the PEFT Deep Dive | The adapter mechanics this module uses; rank/α/target-module theory |
| **CS-24** RL Fundamentals & RLHF with PPO | What SFT feeds into; why a reward model needs a good SFT policy |
| **CS-25** DPO | The stage after SFT; DPO pairs (`prompt`/`chosen`/`rejected`) are built from SFT data |
| **CS-26** GRPO | Verifiable-reward training; needs an SFT'd model with the right output format first |
| **CS-27** ORPO | SFT + preference in one stage; the alternative to "SFT then DPO" |
| **CS-28** Capstone: The Complete End-to-End Pipeline | SFT as the middle stage of the full pipeline |
| **CS-09** Knowledge Distillation II | Where synthetic instruction data comes from; teacher→student SFT |
| **CH-13** Instruction Fine-Tuning Cheat Sheet | The compressed version of this module |
| **IQ-13** Instruction Fine-Tuning Interview Questions | 113 questions from this material |

---

## Appendix A — The Instructor's Key Claims, Verbatim

| # | Timestamp | Claim (verbatim) | Verdict |
|---|---|---|---|
| 1 | [2:47] | Fine-tuning is "taking a pre-trained model and training it further on your own data" | Correct; the definition assumes stage 2 or 3 |
| 2 | [5:20] | The three stages: pretraining, then fine-tuning, then preference alignment | Correct; matches the standard lifecycle |
| 3 | [7:09]–[8:57] | Full fine-tuning updates all weights; partial fine-tuning freezes early layers; PEFT/LoRA is the modern form | Correct, and the historical framing (freeze-then-LoRA) is right |
| 4 | [10:06] | "SFT" comes from having an input column and an output column, i.e. supervision | Correct in effect; note that "supervised" here means *labelled targets*, which pretraining also has |
| 5 | [11:44] | Next-token prediction is the objective at every stage; only the data changes | **Correct, and the most important sentence in the video** |
| 6 | [15:24] | Non-instruction fine-tuning teaches the domain's language, tone, and terminology | Correct — this is continued pretraining (CS-12) |
| 7 | [16:31]–[17:35] | The pharma model does not know the pharma-specific term, "so it will hallucinate" | Correct, and the example is well chosen |
| 8 | [17:45] | "You can fine-tune directly on instruction data but this is not a good practice" | **Directionally right, overstated.** Many teams SFT a strong instruct base directly and get excellent results. The recommendation holds when the domain vocabulary is absent from the base |
| 9 | [18:24]–[19:30] | Recommended order: non-instructional FT on the company corpus first, then instruction FT | Correct for domain adaptation; unnecessary if the base already knows the domain |
| 10 | [19:59] | The purpose of instruction FT is to understand the instruction and generate a well-structured, relevant answer | Correct — the behaviour thesis in one sentence |
| 11 | [20:14] | Alpaca format: `instruction`, `input`, `output` | Correct |
| 12 | [21:35]–[22:05] | Four schema variants: `instruction/input/response`, `context/answer`, `user/assistant`, `system/user/assistant` | Correct; these map onto the five formats in §4.2 |
| 13 | [22:17] | Leave `input` empty when there is no additional context; do not duplicate the instruction into it | Correct, and this is exactly the trap CS-12's notebook falls into |
| 14 | [23:20] | "The biggest question: how you are going to generate this instruction data… in every company… you will NOT find this kind of data" | **Correct and underrated.** The dataset is the project |
| 15 | [24:14]–[25:43] | Three sourcing options: manual writing, expert annotation, LLM synthetic generation | Complete and correct; add "existing logged interactions", which is usually the best source |
| 16 | [26:13] | The conversation must be formatted as a single string before tokenization | Correct — this is the chat template |
| 17 | [27:43]–[28:59] | Instruction FT is required because a base model only predicts the next token; ChatGPT's behaviour comes from instruction tuning | Correct; the framing "the mantra is instruction-response, instruction-response" is a good mnemonic |
| 18 | [31:46] | Base model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Correct model ID, but note the notebook tokenizes with `TinyLlama-1.1B-Chat-v1.0` — a mismatch (§6.6) |
| 19 | [36:30]–[37:07] | Dataset: `Amod/mental_health_counseling_conversations`, 3,512 rows, `Context`/`Response` columns | Correct; the dataset is used only as a formatting demo, not as real training data |
| 20 | [43:00] | Format as `### Instruction:` / `### Input:` / `### Response:` | Correct — this is the Alpaca-flat text form, and it matches the TinyLlama Zephyr template |
| 21 | [44:47] | Tokenize with `truncation=True, padding="max_length", max_length=512` and `labels = input_ids.copy()` | **Mechanically correct, pedagogically hazardous.** `max_length` padding plus unmasked labels trains on 455 pad tokens per row (§6.7) |
| 22 | [45:40]–[46:31] | The masking technique: set the prompt's labels to −100; "this is very effective — we can entirely copy this input ID to the labels" | **The technique is exactly right — and then he recommends not using it.** See #24 |
| 23 | [46:53]–[47:56] | LoRA: r=8, α=16, dropout 0.05, `q_proj`/`v_proj`; TrainingArguments with 3 epochs, bs 1, ga 8, LR 2e-4, fp16 | Correct API; the hyperparameters are tutorial-scale, not production (§7) |
| 24 | [51:19]–[52:58] | After walking through the masking code: "According to me, the previous one is a good one" — i.e. prefer the **unmasked** variant | **Correction (§4.4.3):** masking is the industry default. The recommendation rests on a 3-question eyeball comparison; the unmasked model is being trained to generate the prompt and on 455 pad tokens |
| 25 | [49:05] | The saved checkpoint directory is `checkpoint-3` | Correct, and the name is explainable: 5 rows / bs 1 = 5 steps/epoch; with ga=8 that is 1 optimizer step per epoch; × 3 epochs = **3** |
| 26 | [50:09] | The instruction-tuned model's answer to "what is your name" | **Identical to the non-instruction model's output at [35:54]** — a diagnostic that the adapter was not actually loaded (§6.2) |
| 27 | [54:10]–[54:19] | "First teach your model on your domain specific data, i.e. non-instructional fine-tuning, and then instruction tuning" | **Correct and the most operationally useful sentence in the video.** Cross-ref CS-12 |
| 28 | [55:05] | The comparison of the three questions across the two models | Presented as a verdict; it is a 3-sample eyeball test with no controls. Treat as illustration, not evidence |

---

## Appendix B — Papers, Datasets, Models & Tools

### Papers

| Paper | arXiv | Why it matters here |
|---|---|---|
| **LIMA: Less Is More for Alignment** (Zhou et al., 2023) | 2305.11206 | 1,000 examples; the superficial-alignment hypothesis; 43% win/tie vs GPT-4 |
| **InstructGPT / Training language models to follow instructions with human feedback** (Ouyang et al., 2022) | 2203.02155 | SFT → RM → PPO; 1.3B InstructGPT preferred over 175B GPT-3 |
| **Alpaca / Self-Instruct** (Taori et al., 2023; Wang et al., 2022) | 2212.10560 | 52k synthetic instructions from `text-davinci-003`; the format and the licence problem |
| **AlpaGasus: Training a Better Alpaca with Fewer Data** (Chen et al., 2023) | 2307.08701 | 9k filtered rows beat 52k — the quality-over-quantity evidence |
| **What Makes Good Data for Alignment? (Deita)** (Liu et al., 2023) | 2312.15685 | 6k curated (evol-complexity + quality + diversity) beats 100k+ |
| **LIMA-adjacent: Exploring the Impact of Instruction Data Scale** (Zhou et al., 2023) | 2312.02465 | Scaling instruction data: the marginal value of row 10,000 is near zero |
| **How Far Can Camels Go? (Tulu)** (Wang et al., 2023) | 2306.04751 | Open instruction mixtures; the value of mixture diversity |
| **Tulu 3: Pushing Frontiers in Open Language Model Post-Training** (Lambert et al., 2024) | 2411.15124 | A modern, fully-documented open SFT+DPO recipe |
| **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) | 2106.09685 | The adapter math this module's configs use (see CS-23) |
| **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) | 2305.14314 | 4-bit NF4 + paged optimizers; the 7B-on-one-GPU result |
| **NEFTune: Noisy Embeddings Improve Instruction Finetuning** (Jain et al., 2023) | 2310.05914 | +5–15 pts on judged quality from embedding noise; 5–10 lines of code |
| **Judging LLM-as-a-Judge (MT-Bench)** (Zheng et al., 2023) | 2306.05685 | Position/verbosity/self-enhancement bias; the swap protocol |
| **The False Promise of Imitating Proprietary LLMs** (Gudibande et al., 2023) | 2305.15717 | Why imitation SFT produces a shallow style match |
| **Physics of Language Models / SFT memorizes, RL generalizes** (Chu et al., 2025) | 2501.17161 | The current best evidence on what SFT does and does not install |
| **Textbooks Are All You Need (phi)** (Gunasekar et al., 2023) | 2306.11644 | Synthetic "textbook quality" data — the case for curated synthesis |

### Datasets

| Dataset | Size | Format | Licence note |
|---|---|---|---|
| `tatsu-lab/alpaca` | 52k | Alpaca | **CC-BY-NC 4.0 + OpenAI-ToS restriction on the outputs** |
| `databricks/databricks-dolly-15k` | 15k | Alpaca | CC-BY-SA 3.0; **human-written, commercially usable** |
| `Open-Orca/OpenOrca` | 4.2M | ShareGPT | MIT-ish; GPT-4 derived — check ToS |
| `HuggingFaceH4/ultrachat_200k` | 200k | ShareGPT | MIT (model-generated) |
| `teknium/OpenHermes-2.5` | ~1M | ShareGPT | Mixed upstream licences — audit before commercial use |
| `Amod/mental_health_counseling_conversations` | 3,512 | `Context`/`Response` | CC-BY-NC-4.0; used in the video as a formatting demo only |
| `pharma_instruction_data.jsonl` (this module's companion) | 5 | Alpaca | Course material; **too small to train on — it is a schema example** |
| `garage-bAInd/Open-Platypus` | 25k | Alpaca | CC-BY-4.0, filtered for contamination |
| `allenai/tulu-3-sft-mixture` | 939k | OpenAI messages | ODC-BY; the current best-documented open mixture |
| `microsoft/orca-math-word-problems-200k` | 200k | completion | MIT; the verifiable-reward (GRPO) starter set |

### Models used or referenced

| Model | Params | Note |
|---|---|---|
| `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | 1.1B | **The video's base model.** Non-instruct |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Where the notebook's tokenizer came from (a mismatch, §6.6) |
| `meta-llama/Llama-3.1-8B` / `-Instruct` | 8B | The practical default for domain SFT in 2025 |
| `meta-llama/Llama-3.2-1B/3B` | 1B/3B | Small-model SFT (CS-20) |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Strong instruct base; excellent template hygiene |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | `[INST]` template; tool-calling workhorse |
| `google/gemma-2-9b-it` | 9B | `<start_of_turn>` template |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Synthetic-data-native small model |

### Tools

| Tool | Version cited | Use here |
|---|---|---|
| `transformers` | ≥4.45 | `apply_chat_template`, `Trainer` |
| `peft` | ≥0.13 | `LoraConfig`, `PeftModel`, `merge_and_unload` |
| `trl` | ≥0.12 | `SFTTrainer`, `SFTConfig(assistant_only_loss=True, packing=True)` |
| `datasets` | ≥3.0 | loading, mapping, `Dataset.from_json` |
| `bitsandbytes` | ≥0.44 | 4-bit NF4 (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`) |
| `unsloth` | ≥2024.10 | 2–4× faster SFT (CS-16) |
| `llama-factory` | ≥0.9 | No-code SFT (CS-15) |
| `axolotl` | ≥0.5 | YAML SFT (CS-17) |
| `vllm` | ≥0.6 | `--enable-lora` serving |
| `flash-attn` | ≥2.6 | Needed for real packing throughput |
| `minhash` / `datasketch` | — | Near-duplicate detection |

### Reference implementations to read

1. **TRL's `sft_trainer.py`** — the `assistant_only_loss` path is the cleanest reference for correct masking.
2. **LLaMA-Factory's `dataset_info.json`** — the canonical mapping from on-disk formats to internal ones.
3. **Unsloth's `train_on_responses_only`** — the most robust practical masking implementation, because it derives the boundary from the template rather than from string matching on your data.
4. **`alignment-handbook`** (HuggingFace) — Zephyr/Tulu-style SFT+DPO recipes with real hyperparameters.
5. **The `tatsu-lab/stanford_alpaca` training script** — historically important; note it is the origin of the `-100` convention spreading through the ecosystem.

---

*End of CS-13. Companion artifacts: `CH-13-Instruction-Fine-Tuning.md` (cheat sheet), `IQ-13-Instruction-Fine-Tuning.md` (113 interview questions). Source: video 15 of 32, transcript `LLM_Fine-Tuning_15_Instruction_Fine-Tuning_Explained_Domain-Specific_FineTuning.txt`, notebook `Instruction_finetuning_on_domain_specific_dataset.ipynb`, datasets `pharma_instruction_data.jsonl` / `.csv`.*
