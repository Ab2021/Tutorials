# IQ-15 — Interview Questions: LLaMA-Factory (No-Code / YAML Fine-Tuning)

| Field | Value |
|---|---|
| **Module** | Frameworks / Tooling — the config-driven branch (the "no-code" trainer) |
| **Pairs with** | **CS-15** (case study), **CH-15** (cheat sheet) |
| **Total questions** | **113** (30 L1 + 32 L2 + 24 L3 + 10 L4 + 17 L5) + 40 rapid-fire + 5 whiteboard |
| **Levels covered** | L1 screening → L5 incident response |
| **Source material** | CS-15 §0–§20, CH-15 §1–§13, `train_gemma_qlora.yaml`, `data/dataset_info.json`, `how-to-save-in-dataset_info.txt`, `code/common/memory.py --table` |
| **Also contains** | Rapid-fire true/false (40), 5 coding/whiteboard tasks with grading notes, numbers to memorize, full answers to CS-15's 10 self-check questions |
| **Time to work through** | ~6 h at interview pace; ~75 min for a revision pass (L1 + L2 + rapid fire + numbers) |

---

## How To Use This File

- **L1 — Fundamentals & Vocabulary (30).** Can you place the tool and name its two registries (datasets, templates) at all? A candidate who cannot say what `template:` does has read the README and nothing else.
- **L2 — Applied & Implementation (32).** Registry entries, YAML keys, the `key=value` override, the export config. This is the level that separates "I ran the WebUI once" from "I shipped a pipeline."
- **L3 — Analysis & Trade-offs (24).** Mechanism-level: *why* a wrong template is silent, *why* ZeRO-3 is not automatically better, *why* export dequantises, *why* the converter emits empty samples instead of raising.
- **L4 — System Design & Scenario (10).** Full prompts: requirements → constraints → design → trade-offs → failure modes. 8–12 minutes each, out loud.
- **L5 — Debugging & Incident Response (14).** A symptom and a clock. Say what you check first, second, third — and which of the seven other things that look identical you are ruling out.

**The meta-rule that decides most of this module's interviews:** *LLaMA-Factory is a compiler with the type-checker disabled.* A type mismatch in your data (a role tag the template table does not know) prints one warning and emits an example with an empty prompt and an empty response; a wrong ABI (`template: llama3` on a Gemma model) trains happily on tokens the model has never seen. Nothing fails the build. So the candidate who says "the loss went down, so it worked" fails, and the candidate who says "the loss curve cannot see a template mismatch — show me the tokenised sample" passes. That single instinct is worth more than every hyperparameter in the YAML.

**Answer format used throughout:** a direct **Answer**, then **Why the interviewer asks this** (what is actually being probed), then **Trap** (the plausible-but-wrong answer that gets candidates rejected).

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. In one sentence, what is LLaMA-Factory?**

- **Answer:** A config-driven orchestration layer **over** the Hugging Face stack — it wraps `transformers`, `peft`, `bitsandbytes` and `trl` and exposes them as one YAML file (or a Gradio WebUI), plus a dataset registry and a chat-template registry. The instructor's own framing is exact: *"they haven't written the complete code from scratch, no… they are using the Hugging Face libraries only"* [7:25].
- **Why the interviewer asks this:** It is the module's load-bearing fact, and it predicts both halves of the tool's behaviour — its superpower (any HF model works) and its weakness (every HF bug is your bug, and you cannot out-run the backend's limits).
- **Trap:** Calling it "a new training stack" or "a model." It is neither; it is a thin layer, and the repo's own `train_gemma_qlora.yaml` is reproducible in ~60 lines of raw HF.

**Q2. What is the "no-code" surface, concretely?**

- **Answer:** **LLaMA Board** — the Gradio WebUI launched by `llamafactory-cli webui` (or `create_ui()` + `ui.launch()`). It covers model → method → quantisation → template → stage → dataset → hyperparameters → train → chat → export, and it writes out a YAML config. Its most pedagogically important button is **Preview command**, which converts the GUI state into the equivalent CLI invocation. It is a *local* Gradio app, not a hosted service — *"it is not providing you any sort of a GPU and all"* [18:22].
- **Why the interviewer asks this:** "No-code" is a marketing word until you can name the artifact it produces. The WebUI's output is a YAML file — which is the same artifact the CLI consumes, which is why the no-code path is not a dead end.
- **Trap:** Thinking the WebUI is a platform you log into, or that `share=True` is a deployment. It is a tunnel that expires, on an app with no auth — do not expose it.

**Q3. What is `stage:`, and what does changing it change?**

- **Answer:** `stage` selects the training **objective and workflow**: the short forms are `pt` (pretrain), `sft`, `rm` (reward model), `ppo`, `dpo`, `kto`. It decides which loss is computed, which workflow module runs (`train/<stage>/workflow.py`), and — critically — **which dataset schema is required**. `stage: rm`/`ppo` need a reward model; `stage: dpo`/`kto` need preference-shaped data, not instruction data.
- **Why the interviewer asks this:** It is the first line of any config and it determines the entire shape of the run, including whether you now need a reference model or a reward model you do not have.
- **Trap:** Changing `stage: dpo` without changing the dataset to a `ranking: true` preference file. You get a run that trains on the wrong schema — or an error that surfaces deep in the data pipeline rather than at the key that caused it. Also note ORPO/SimPO are commonly expressed as **loss variants under the preference stage** (`pref_loss: orpo|simpo|hinge|ipo`) rather than as separate `stage` values; check your installed version, because CH-15's decision tree and CS-15's key table describe this differently.

**Q4. What is `finetuning_type:`, and what are its three values?**

- **Answer:** *How* weights are updated: `lora`, `freeze`, or `full`. `freeze` trains only the last N layers (default 2) and/or named modules; `full` trains every weight; `lora` attaches adapters. It is orthogonal to `stage` — you can run `stage: dpo` with `finetuning_type: lora`.
- **Why the interviewer asks this:** It is the memory/speed dial and the second-most-common confusion in the framework, immediately after `template`.
- **Trap:** Believing `freeze` is cheaper than LoRA. It usually uses **more** VRAM, because it still stores full-precision gradients and optimizer state for the unfrozen layers.

**Q5. How do you enable QLoRA in LLaMA-Factory?**

- **Answer:** Two keys: `finetuning_type: lora` **plus** `quantization_bit: 4` (optionally `quantization_method: bnb`, the documented stable default). There is **no `finetuning_type: qlora`** — `finetuning_type` is a `Literal["lora", "freeze", "full"]`. The repo's own `train_gemma_qlora.yaml` proves the point: line 7 is `finetuning_type: lora`, line 35 is `quantization_bit: 4`, and the word "qlora" appears nowhere in the file.
- **Why the interviewer asks this:** It is the single most common newcomer config error and the fastest way to tell whether a candidate has actually written a config or only read a blog post.
- **Trap:** Writing `finetuning_type: qlora` and assuming it must work because the file is named `train_gemma_qlora.yaml`. Argument parsing fails — `qlora` is not a member of the Literal.

**Q6. What does `template:` control, and why is it the highest-risk key?**

- **Answer:** It controls **two** things, and this is why it is not cosmetic: (1) the exact token string rendered from your `messages` — including where the prompt ends and the response begins, which is what derives the loss mask; and (2) the tokenizer's special tokens, via `fix_special_tokens`, which force BOS/EOS/PAD to the model family's expected values. A wrong template raises **no error**, trains on a token string the model has never seen, and produces a model that is *worse than the base model* at inference.
- **Why the interviewer asks this:** It is the #1 silent failure in the whole module (CS-15 §9.4 row 1) and the cause of the complaint "I fine-tuned it and it got worse."
- **Trap:** "The tokenizer's own template is close enough." That is a different code path — see Q7.

**Q7. What is the difference between `template: default`, `template: empty`, and omitting `template`?**

- **Answer:** Three paths, two of which behave nothing like their names. `template: default` is a **registered plaintext template** emitting `Human: …\nAssistant: …\n` with a `System:` prefix, `replace_jinja_template=True`, `efficient_eos=False`. `template: empty` is bare `{{content}}` concatenation with no role markers (the pretraining fallback). **Omitting** `template` is the third path: the framework checks `tokenizer.chat_template`, and if it is a string it logs a warning and derives a template via `parse_template(tokenizer)`; otherwise it falls back to `empty`.
- **Why the interviewer asks this:** It inverts the single most widely repeated belief about the tool — that `default` means "use the model's own chat template." Upstream's own docs have been flagged as stale on exactly this point.
- **Trap:** Writing `template: default` on a Llama-3 model and believing you are on ChatML. You are on Vicuna-format plaintext, and the run will look healthy.

**Q8. What is `data/dataset_info.json`, and what is it the API of?**

- **Answer:** A single hand-edited JSON object whose **keys are dataset names**. It is the entire dataset API: every dataset — local file, HF Hub repo, or ModelScope — needs an entry, because the YAML references the *key*, never a path. The instructor is explicit that this holds even for Hub data: *"even if you are going to read data from the Hugging Face, in that case also you will have to make an entry inside this particular file"* [37:33].
- **Why the interviewer asks this:** It is the thing that actually blocks beginners, and it is the structural difference between LLaMA-Factory and Axolotl (which puts the data contract in the config) or Unsloth/TRL (which put it in code).
- **Trap:** Putting a file path or a Hub URL in the YAML's `dataset:` field. That gives `ValueError: Undefined dataset … in dataset_info.json.` — a loud failure, which is the good kind.

**Q9. What are the `formatting` values, and what happened to `format`?**

- **Answer:** Current releases read **`formatting`**, with values `alpaca` | `sharegpt` | `openai`. `openai` is not a distinct converter — it is sharegpt plus a `tags` remap for `{messages:[{role,content}]}` data. The key `format` is the older spelling; when it is present and `formatting` is absent, the value is **silently ignored** and the schema defaults to `alpaca`. Preference (`ranking: true` + `chosen`/`rejected`), KTO (a `kto_tag` column) and pairwise (`pref_loss: sigmoid`) are **not** `formatting` values in current releases, despite still circulating as `"format": "preference"` in tutorials — including this course's own notes.
- **Why the interviewer asks this:** It tests whether you have read the schema or copy-pasted one, and it maps to the framework's nastiest asymmetry: the alpaca case fails **silently**, the sharegpt case fails **loudly**, so alpaca-only tutorials propagated the bug for a year.
- **Trap:** Writing `"format": "sharegpt"` and losing a day. With `formatting` unset, sharegpt rows are parsed as alpaca and the converter looks for `instruction`/`output` fields that do not exist.

**Q10. What are the default ShareGPT role tags, and why does it matter?**

- **Answer:** `user_tag: "human"` and `assistant_tag: "gpt"` — with `role_tag: "from"` and `content_tag: "value"`. They are **not** `user`/`assistant`. If your data says `{"from": "user", …}`, the sharegpt converter does not raise: it logs `Invalid role tag in […].` at WARNING level, logs `Skipping this abnormal example.`, and then **still emits the example with `prompt, response = [], []`**. You get a dataset of unchanged length containing empty samples.
- **Why the interviewer asks this:** It is the second silent failure of the framework (CS-15 §9.4 row 2) and it produces the most confusing symptom in the module: a dataset that is clearly full of content, and a loss of ~0 or `nan`.
- **Trap:** Assuming "it's JSONL chat data, it must be the standard shape." `user`/`assistant` is what most JSONL chat data on the internet looks like, and it is not what the default tag table accepts. Fix by renaming the fields or by adding a `tags` block.

**Q11. What is `cutoff_len`, and what is the failure it causes?**

- **Answer:** The maximum tokenised length **after** the template is rendered; longer examples are truncated. It is a hard truncation applied post-template, so the template's own tokens (≈30–60 for a Gemma-style format with a system turn) count against your budget. If the prompt is long, the **response** can be cut off entirely, leaving samples whose labels are a handful of tokens.
- **Why the interviewer asks this:** It is the most common cause of "the loss looks fine and the model learned nothing," and it is arithmetic the candidate can be asked to do live (Q79, Q103).
- **Trap:** Setting `cutoff_len` from a tutorial instead of from your data. The correct derivation is `cutoff_len ≈ p99(response tokens) × 1.2 + template overhead`, confirmed by dumping one tokenised sample.

**Q12. What is `val_size`, and what is its default?**

- **Answer:** The fraction (or count) of each dataset held out for validation. Its default is **0** — meaning no held-out split and no eval loss unless you set it. Setting it also requires an eval strategy (`eval_strategy: steps|epoch`) to do anything at all.
- **Why the interviewer asks this:** It is the difference between a run that can detect overfitting and one that cannot, and the default is the wrong answer for anything except a smoke test.
- **Trap:** Two of them. (a) Leaving it at 0 and reading a falling train loss as success. (b) Believing `val_size` is an evaluation protocol — it slices your *training file*, so it shares provenance, annotator and template with the training data and measures memorisation of a distribution, not generalisation. A real holdout is a separate registry key used via `eval_dataset`.

**Q13. What does `llamafactory-cli export` do?**

- **Answer:** It loads the base model plus the adapter, computes `W_merged = W_base + (α/r)·(B@A)` for every targeted module, and writes a standalone safetensors model (sharded by `export_size` in GB). Training **saves adapters**; `export` **merges** them. The merged artifact has no `peft` dependency at load time — it is what you hand to a serving stack.
- **Why the interviewer asks this:** It is the last step before serving and the last place a good run gets quietly ruined, and it is where the training-time and serving-time artifacts diverge.
- **Trap:** Putting `quantization_bit` in the merge YAML. The framework's own example config states the rule in capitals: *"DO NOT use a quantized model or `quantization_bit` when merging LoRA adapters."*

**Q14. Name the `llamafactory-cli` verbs and what each is for.**

- **Answer:** `train` (run a stage from YAML), `chat` (terminal REPL over base + adapter), `export` (merge), `api` (OpenAI-compatible HTTP server, `infer_backend: vllm|sglang|huggingface`), `webui`/`webchat` (LLaMA Board / chat-only UI), `eval` (perplexity/BLEU/ROUGE/MMLU-family), `env` (environment report), `version`, `help`. The older module path `python -m llamafactory.cli <verb>` still works.
- **Why the interviewer asks this:** It is concrete evidence of hands-on use, and `env` is what you paste into a bug report while `export` is what you actually ship.
- **Trap:** Depending on `eval`. It is marked for deprecation in current `main` and raises `NotImplementedError`; if your pipeline needs it, pin a release tag rather than discovering this in a deadline week.

**Q15. What is `adapter_name_or_path` for?**

- **Answer:** Chaining stages without restarting from base. `stage: dpo` with `adapter_name_or_path: ./runs/my_sft` starts DPO from *your* SFT adapter rather than from the raw base model — which is the correct order (SFT, then preference tuning). It is also what you set in an inference YAML to chat with a trained adapter.
- **Why the interviewer asks this:** The SFT → DPO handoff is the single most common multi-stage pipeline, and getting it wrong means preference-tuning a model that has never been instruction-tuned.
- **Trap:** Putting the SFT output in `model_name_or_path` instead. It must be `adapter_name_or_path`; the base stays in `model_name_or_path`.

**Q16. What does `mask_history: true` do, and when do you need it?**

- **Answer:** In multi-turn data, it masks every assistant turn **except the last**, so loss is computed only on the final response. Default is `false`, which means every assistant turn receives loss.
- **Why the interviewer asks this:** Multi-turn data with synthetic or low-quality early turns is extremely common, and the default silently trains on all of it — changing your effective dataset size and quality without changing the loss curve's shape.
- **Trap:** Believing `mask_history` and `ignore_pad_token_for_loss` are the same knob. They are not: the latter governs **padding** positions, and setting it `false` teaches the model to emit pad tokens. Prompt masking is a property of the template's prompt/response split, with `train_on_prompt` and `mask_history` as the two explicit overrides.

**Q17. What does `report_to: none` prevent?**

- **Answer:** HF `Trainer`'s `report_to="wandb"` behaviour when `wandb` is installed — which on the CLI presents an **interactive prompt** ("create a W&B account / use an existing account / don't visualize my result"). In a non-TTY job that prompt is not a question, it is a hang.
- **Why the interviewer asks this:** It is a small key with an operational consequence, and it is the test of whether a candidate has run training in CI rather than in a notebook.
- **Trap:** Relying on `WANDB_DISABLED=true` as your CI policy. It works, but the YAML key is the artifact you commit; put `report_to: none` in the config and let the environment variable be the belt-and-braces.

**Q18. What is `packing`, and what is `neat_packing`?**

- **Answer:** `packing` concatenates short samples into full-length sequences to eliminate padding waste — 2–4× throughput on short-data SFT. `neat_packing` is packing **without cross-sample attention**, i.e. each packed sample keeps its own attention window so it cannot attend to its neighbours. `packing` is auto-on for `pt` and off for `sft`; `neat_packing` defaults to `false`.
- **Why the interviewer asks this:** It is a throughput knob that becomes a data-corruption bug when applied by reflex: without `neat_packing`, two examples can be concatenated into one training row and the model learns to continue someone else's answer.
- **Trap:** Turning `packing: true` on for chat data and leaving `neat_packing` off. Rule of thumb: `packing` for raw-text CPT, `packing + neat_packing` for short-instruction SFT, and neither if your samples already fill `cutoff_len`.

**Q19. Where does the WebUI sit in a production workflow?**

- **Answer:** It is a **first-run and teaching tool**, not a job runner. Use it to learn the parameter space (the instructor's own point: *"just by reading this UI you will get so much knowledge regarding which parameter you need to choose"* [30:31]), to run one-off experiments, and to reach `Preview command`, which teaches you the CLI. Then move to the CLI, because the WebUI has no job queue, no retry, no sweep engine, no multi-node, no git integration, and a browser tab is not a scheduler.
- **Why the interviewer asks this:** It tests whether "no-code" means "toy" (wrong) or "on-ramp" (right), and whether you know the exact boundary.
- **Trap:** Either dismissiveness ("the WebUI is a toy" — its run produced a working adapter in 4–5 minutes) or over-reliance ("we'll run the sweep from the UI").

**Q20. One line each: LLaMA-Factory vs Unsloth vs Axolotl vs TRL vs torchtune.**

- **Answer:** LLaMA-Factory = YAML + WebUI + a dataset/template registry, broadest model coverage. Unsloth = hand-patched Triton kernels for single-GPU speed (CS-16). Axolotl = YAML at scale, best-in-class multi-GPU/FSDP2/ZeRO (CS-17). TRL = a Python library where you own the loop and can write a new loss. torchtune = readable single-file PyTorch recipes with FSDP2.
- **Why the interviewer asks this:** The whole of IQ-03 in one answer, applied. A candidate who places LLaMA-Factory correctly will answer every follow-up correctly.
- **Trap:** Treating them as competitors on one axis. They are on three different axes (abstraction, memory regime, parallelism) and they compose — LLaMA-Factory can route through Unsloth with `use_unsloth: true`.

**Q21. What is `lora_target: all`, and why is it the safe default?**

- **Answer:** It attaches adapters to every `nn.Linear` module, resolving names per architecture and sidestepping the `target_modules` matching problem entirely. On Gemma it resolves to the seven attention and MLP projections (`q,k,v,o,gate,up,down`). Note it does **not** include `nn.Embedding`, which is why the vocabulary is untouched and why `lora_target: all` on a 2B Gemma gives ~9.81 M trainable parameters.
- **Why the interviewer asks this:** The "trainable params printed as 0.000%" failure is common enough that the framework added a blunt fix for it, and knowing *why* shows you understand module-name matching.
- **Trap:** Hand-listing `q_proj, v_proj` for an unfamiliar architecture and getting near-zero trainable parameters with a loss that barely moves. Also the inverse: on a large model, `all` inflates optimizer state and checkpoint size — and on MoE models it includes every expert projection.

**Q22. What is `quantization_bit: 8` for, versus `4`?**

- **Answer:** Both quantise the frozen base for training; `4` is NF4 QLoRA, `8` is int8. `8` costs roughly +1.3 GB at 2B-scale over `4` and gives slightly better quality in the base path; `4` is the default practical recipe. `quantization_bit` is a **training-time** setting — it is not a serving quantization format.
- **Why the interviewer asks this:** It separates "4-bit = QLoRA = the only 4-bit thing" from an understanding that there is a spectrum, and it sets up the export question (Q13, Q63).
- **Trap:** Confusing training-time 4-bit (NF4 via `bitsandbytes`) with post-training inference quantization (GPTQ/AWQ/GGUF). The exported artifact is bf16/fp16; you quantise it again, separately, for serving.

**Q23. What does the framework write into `output_dir` that you should keep?**

- **Answer:** The adapter (`adapter_model.safetensors` + `adapter_config.json`), `trainer_state.json` (which contains the eval-loss history), `trainer_log.jsonl` and `loss.png` (if `plot_loss: true`), and — the underrated one — **its own resolved config**: the post-defaults, post-CLI-override YAML as the run actually executed. The instructor points at exactly this: *"this entire command configuration will be saved over here inside this YAML"* [49:18].
- **Why the interviewer asks this:** It is the framework's audit trail, and it is written at the *start* of training, so a run that crashes at step 1 still leaves a misleading artifact — you pair it with your own commit/hash files.
- **Trap:** Treating the adapter directory as a model. It contains no base weights; copying it elsewhere without the base and a matching `peft`/`transformers` gives you nothing loadable.

**Q24. What is `IGNORE_INDEX` in this stack?**

- **Answer:** `-100`, the label value that means "no loss here." Every masked position — the prompt, the padding, and (with `mask_history`) earlier assistant turns — carries `-100`. Counting non-`-100` positions in one sample is the single most useful diagnostic in the framework.
- **Why the interviewer asks this:** It is the concrete form of "the loss curve cannot see a masking bug," and it makes the token-dump diagnostic (Q44) legible.
- **Trap:** Confusing "masked" with "ignored by the model." The tokens are still in `input_ids` and still attend; they simply contribute nothing to the gradient. `train_on_prompt: false` does not remove the prompt, it removes its loss.

**Q25. Does LLaMA-Factory support a custom loss function?**

- **Answer:** Not without editing the source. There is no config key for a custom objective; you would be editing `train/<stage>/workflow.py`, which is exactly the code that conflicts on your next upgrade. The honest boundary is: register a new *dataset converter* or *template* (both designed extension points), but for a new *loss*, write the loop in TRL or fork a torchtune recipe.
- **Why the interviewer asks this:** It is the module's central "when is this the WRONG tool" question, and the answer must come with the mechanism, not just the verdict.
- **Trap:** "I can subclass the Trainer and pass it in." You can, but you are then maintaining a fork of a fast-moving project for a capability that TRL gives you for free — and every hour spent there is an hour that will conflict with the next release.

**Q26. What is `optim:` and what should a LoRA run set it to?**

- **Answer:** The optimizer. For LoRA/QLoRA, `optim: paged_adamw_8bit` (bitsandbytes) is the standard choice — it halves optimizer-state bytes and pages to CPU under pressure. For QLoRA it is nearly free, because the optimizer state covers only the adapter slice (~137 MB at 2B rank 8); for **full** fine-tuning it is the difference between 16 and 10 bytes/param and therefore between 4×A100 and 2×A100 at 8B.
- **Why the interviewer asks this:** It shows you know which memory term dominates in which regime — and that the answer is different for LoRA and full FT.
- **Trap:** Setting `optim: paged_adamw_8bit` for a **full** fine-tune and expecting it to be free: page faults to host memory can dominate step time, and 8-bit Adam has a small quality cost.

**Q27. What is `neftune_noise_alpha`, and what is the recommended value?**

- **Answer:** NEFTune — uniform noise added to the input embeddings during training only (the paper's default is **5**). It costs nothing at inference and reliably improves instruction-following on short SFT datasets. It is off by default and is the single highest-leverage same-money change to a standard SFT config.
- **Why the interviewer asks this:** It distinguishes someone who tunes a config from someone who copies one. It is exposed in the WebUI the instructor demoed and is not used in the demoed run.
- **Trap:** Treating it as a regulariser to add along with dropout without measuring. It changes the training distribution; A/B it on your frozen probe set like anything else.

**Q28. What does `FORCE_TORCHRUN=1` do?**

- **Answer:** Routes `train` through `torchrun`, which is what you need for more than one GPU. It is not needed on a single GPU, and setting `deepspeed:` with a multi-GPU JSON on one GPU without it gives you ZeRO-3's overhead with none of its benefit — and some versions complain about the device count, because ZeRO-3 assumes ≥2 processes.
- **Why the interviewer asks this:** It is the entire multi-GPU surface of the tool: one environment variable, plus a DeepSpeed JSON path. Knowing that is knowing the boundary between the WebUI and a real cluster job.
- **Trap:** Assuming the WebUI's "device count" makes a multi-node job. It covers a single node's GPUs; multi-node needs `NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`.

**Q29. Which two registries must agree for a run to be correct?**

- **Answer:** The **dataset registry** (`<dataset_dir>/dataset_info.json` — key → file, formatting, columns, tags) and the **template registry** (`src/llamafactory/data/template.py` — name → the exact format strings and special-token forcing). The YAML's `dataset:` must be a key in the first; the YAML's `template:` must be a name in the second; and the template must match the model family.
- **Why the interviewer asks this:** Every silent failure in CS-15 §9.4 traces to one of these two registries, and a candidate who names them has the mental model rather than a list of symptoms.
- **Trap:** Saying "the tokenizer." The tokenizer's own `chat_template` is only consulted when `template` is omitted, and it is model-repo data that can change between revisions — which is precisely why the framework pins the token string in Python instead.

**Q30. Why does the framework have a `template:` key at all, rather than just using the tokenizer?**

- **Answer:** Because `apply_chat_template` reads `tokenizer_config.json`, which is **model-repo data and drifts**: a repo can change its template between revisions, a fine-tune can ship a template the base never had, and many templates assume a system message your data does not contain. Pinning the token string in Python makes the recipe version-dependent on *your code*, not on a Hub revision — and gives the framework a place to force `fix_special_tokens`. It also gives it a place to derive the prompt/response split, which is where the loss mask comes from.
- **Why the interviewer asks this:** It is the design rationale for the whole module, and it converts "template is a gotcha" into "template is the mechanism."
- **Trap:** "It's a convenience wrapper for `apply_chat_template`." It is a *replacement* for it, and the two produce different bytes.

---

## Level 2 — Applied & Implementation

**Q31. Write the `dataset_info.json` entry for `data/support/chat.jsonl` whose rows are `{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}`.**

- **Answer:**

```json
{
  "support_chat": {
    "file_name": "support/chat.jsonl",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": { "role_tag": "role", "content_tag": "content",
              "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system" }
  }
}
```

Key points: `file_name` is **relative to `dataset_dir`** (default `data`), so `"support/chat.jsonl"`, not `"data/support/chat.jsonl"`. The `tags` block is mandatory here because the defaults expect `from`/`value` with roles `human`/`gpt`; without the remap every row is skipped with a warning and emitted empty. And the YAML uses `dataset: support_chat` — the key.

- **Why the interviewer asks this:** It is the framework's equivalent of "reverse a linked list": trivial if you have used the tool, impossible to bluff.
- **Trap:** Omitting the `tags` block. This is the highest-frequency real-world failure in the module, and it fails **silently**.

**Q32. Write the `dataset_info.json` entry for an alpaca file, and for a preference file.**

- **Answer:**

```json
{
  "acme_sft": {
    "file_name": "acme/sft_v3.jsonl",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  },
  "acme_prefs": {
    "file_name": "acme/prefs_v2.jsonl",
    "ranking": true,
    "columns": { "prompt": "instruction", "query": "input",
                 "chosen": "chosen", "rejected": "rejected" }
  }
}
```

The `columns` in the alpaca entry are the *defaults* and are therefore redundant — harmless here, but a bad habit if your field names differ and you assume the mapping is automatic. For preference data, the load-bearing key is **`ranking: true`**, not a `formatting` value and not a `columns` entry.

- **Why the interviewer asks this:** It tests the schema's real shape rather than the shape tutorials teach, and it distinguishes "preference" from "a formatting called preference."
- **Trap:** Writing `"formatting": "preference"` (a stale key that no longer exists) or omitting `ranking: true`. Without `ranking: true` the rows load as ordinary alpaca data and you train a DPO stage on instruction-shaped samples.

**Q33. Write the registry entry for a dataset that lives on the Hugging Face Hub.**

- **Answer:**

```json
{
  "hf_unix_commands": {
    "hf_hub_url": "harpomaxx/unix-commands",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  }
}
```

`hf_hub_url` **overrides** `file_name` (resolution order: `hf_hub_url`/`ms_hub_url` → `script_url` → `cloud_file_name` → `file_name`). Add `"subset"` and `"split"` for multi-config datasets. The YAML still refers to the key: `dataset: hf_unix_commands`.

- **Why the interviewer asks this:** It is the exact case the instructor walked through [43:06]–[43:54], and it tests the one rule people most want to break — that Hub data also needs a registry entry.
- **Trap:** Putting the Hub repo id straight into `dataset:` and expecting a download. You get `ValueError: Undefined dataset … in dataset_info.json.` — loud, at least.

**Q34. Write the registry entry that makes a raw-text file work with `stage: pt`.**

- **Answer:**

```json
{
  "my_plain_text": {
    "file_name": "my_custom_data3.json",
    "formatting": "alpaca",
    "columns": { "prompt": "text" }
  }
}
```

Three lines of JSONL, each `{"text": "…"}`; the non-obvious move is the `prompt → text` column remap, which is how pretraining data is expressed through the alpaca converter. Used with `stage: pt`.

- **Why the interviewer asks this:** It is the only place in the module where the alpaca converter is used for something that is not instruction data, and it proves you understand that `columns` is a *type declaration* rather than a cosmetic alias.
- **Trap:** Using `stage: sft` with this entry "because it's SFT-shaped data." The converter then yields an **empty response** and the whole sequence receives loss — which is sometimes what you want (unsupervised SFT) and sometimes destroys your instruction structure. Decide deliberately; do not arrive there by accident.

**Q35. What are the LoRA defaults in LLaMA-Factory, and what do they resolve to in the demo run?**

- **Answer:** `lora_rank: 8`, `lora_alpha: 2 × rank = 16`, `lora_dropout: 0.0`, `lora_target: all`. On Gemma-1.1-2B with `lora_target: all`, that resolves to seven projections per layer × 18 layers = **9 805 824 ≈ 9.81 M trainable parameters = 0.39 %** of the 2.51 B model. Scaling: rank 16 → 19.6 M; rank 32 → 39.2 M; rank 64 → 78.4 M (3.1 %).
- **Why the interviewer asks this:** It is a number you can derive live — `Σ r × (in + out)` over targeted modules — and deriving it beats remembering it. It also exposes the 2 %/0.4 % intuition for "how much am I actually training."
- **Trap:** Assuming `all` includes the embedding. It does not (`nn.Embedding` is not `nn.Linear`), which is why vocabulary adaptation needs `additional_target: embed_tokens,lm_head` and why the 2B parameter count barely moves with rank.

**Q36. What learning rate do you use for LoRA, full FT and DPO in LLaMA-Factory?**

- **Answer:** LoRA/QLoRA: **1e-4 – 2e-4** (`2e-4` is the common default in the cheat sheet's starter config; the course's PDF says 5e-5 to 2e-4 for QLoRA). Full FT: **1e-5 – 2e-5**. DPO: **~5e-6**, roughly 10–40× lower than SFT. GRPO: ~1e-6.
- **Why the interviewer asks this:** It is the most common catastrophic misconfiguration in the module — a full-FT learning rate applied to a LoRA run, or an SFT learning rate applied to DPO. Both produce runs that *look* healthy.
- **Trap:** "Same LR, they're all fine-tuning." LoRA tolerates 10–100× the full-FT LR because the adapter starts at zero and only a tiny subspace moves. Applying `2e-5` to a LoRA run does not break anything — it gives you a model that has barely moved, with a perfectly healthy loss curve. That is how people conclude "LoRA doesn't work."

**Q37. How do you compute how many optimiser steps a LLaMA-Factory run will take, and why does that number change another key?**

- **Answer:** `steps_per_epoch = ceil(n_train / (per_device_train_batch_size × gradient_accumulation_steps × n_gpus))`. The demo: `max_samples: 1000`, `val_size: 0.1` → 900 train; batch 1 × accum 4 → **225 optimiser steps** per epoch, 1 epoch. That number sets the useful value of `eval_steps` (target ≈ steps_per_epoch / 5 → ~45) and of `save_steps`, and it determines whether `warmup_ratio: 0.03` is 7 steps or 1.
- **Why the interviewer asks this:** It is the arithmetic that turns a config you copied into a config you understand, and it catches the `eval_steps` no-op (Q38) before you waste the run.
- **Trap:** Computing it and then leaving `eval_steps: 200` in place. You get one evaluation, near the end, and you read it as "the run was stable."

**Q38. Your config has `eval_steps: 200` and `eval_strategy: steps`. What actually happens?**

- **Answer:** With 225 steps per epoch and 1 epoch, evaluation fires at step 200 and then at end-of-training — **one meaningful evaluation point**, not a curve. Change it to ~45 (`steps_per_epoch / 5`) and you get five points. If you also want mid-run checkpoints, you must change `save_strategy` from `epoch` to `steps` separately — in a single-epoch run, `save_strategy: epoch` writes exactly one checkpoint, at the end.
- **Why the interviewer asks this:** It is a silent no-op that requires only arithmetic to predict, and it is a common answer to "why does my run tell me nothing?"
- **Trap:** Assuming `eval_steps` is a maximum. It is an interval; if it exceeds the run length, you effectively evaluate once.

**Q39. What is the order of operations between `max_samples` and `val_size`, and why does it matter?**

- **Answer:** `max_samples` truncates **first**, then `val_size` slices the truncated set. So `max_samples: 1000, val_size: 0.1` gives 900/100 — not "10 % of your real dataset." The dangerous combination is `max_samples: 100` for a quick test with `val_size: 0.1` retained: you validate on **ten examples**, a number with a confidence interval so wide it is decoration.
- **Why the interviewer asks this:** It is a composition bug that no error message can catch, and the correct fix (set `val_size: 0` for smoke tests, or hold out a real dataset via `eval_dataset`) requires knowing the order.
- **Trap:** Leaving `max_samples` in the production config. Its own help text says *"For debugging purposes"* — it has exactly one legitimate production use, a throwaway smoke test.

**Q40. How do you validate a LLaMA-Factory config without spending GPU time?**

- **Answer:** Two 30-second checks. (1) A **1-example, 1-step run**: `max_samples: 1`, `max_steps: 1`, `output_dir: /tmp/lf-smoke`, `report_to: none`, `save_strategy: no`, `plot_loss: false` — this exercises the registry, the converter, the template and the tokeniser, and fails in under a minute if any of them is wrong. (2) The **registry API check**:

```python
from llamafactory.data.parser import get_dataset_list
for attr in get_dataset_list(["my_custom_data", "my_chat_data"], dataset_dir="data"):
    print(attr)   # my_custom_data(file, file_name='…', formatting='alpaca', …)
```

- **Why the interviewer asks this:** Everyone wastes GPU hours on a broken registry. The candidate who names the smoke test has a working process, not just knowledge.
- **Trap:** Running the real job as the test. The first run on a new dataset should always be `max_samples: 1`, because the failure modes are all categorical — a wrong key, a wrong tag, a wrong template — and none of them need 225 steps to appear.

**Q41. How do you dump one tokenised sample and read the loss mask?**

- **Answer:**

```python
from transformers import AutoTokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args

model_args, data_args, training_args, finetuning_args, _ = get_train_args(
    {"stage": "sft", "model_name_or_path": "google/gemma-1.1-2b-it",
     "template": "gemma", "dataset": "my_custom_data", "dataset_dir": "data",
     "cutoff_len": 1024, "output_dir": "/tmp/dump"})
tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
template = get_template_and_fix_tokenizer(tok, data_args)
ds = get_dataset(template, model_args, data_args, training_args, stage="sft")

row = ds["train"][0]
print("TOKENS:", tok.convert_ids_to_tokens(row["input_ids"][:80]))
print("LABELS:", row["labels"][:80])                    # -100 == masked == no loss
n_sup = sum(1 for x in row["labels"] if x != -100)
print(f"SUPERVISED TOKENS: {n_sup} of {len(row['labels'])}")
```

Reading it: 5–15 % supervised is normal for short-answer SFT; **0 or near 0 means everything is masked**, which is the bug 90 % of the time; ~100 % means `train_on_prompt` is on or you are on the `pt` path; special tokens that look nothing like the model's own means a wrong template; response text cut mid-sentence means `cutoff_len` is too small.

- **Why the interviewer asks this:** It is the one diagnostic that finds most template and dataset bugs, and the supervised-token count is the single number the loss curve cannot show you.
- **Trap:** Reading the loss instead. Loss measures fit to *your tokenised data* — and when the tokenised data is the thing that is broken, it falls anyway.

**Q42. How do you list the templates your installed version actually has?**

- **Answer:**

```bash
python -c "from llamafactory.data.template import TEMPLATES; print(len(TEMPLATES), sorted(TEMPLATES)[:20])"
# or, on some versions: from llamafactory.extras.constants import TEMPLATES
```

Do this before configuring a new model, because template names track releases.

- **Why the interviewer asks this:** It is the difference between configuring from the source of truth and configuring from a blog post. The instructor's own registry spans 100+ families, and names like `gemma` / `gemma2` / `gemma3` and `phi` / `phi_small` / `phi4` are **not** aliases and have no fallback chain.
- **Trap:** Assuming a name you saw in a tutorial exists in your version. A typo gives `ValueError: Template … does not exist.` — the *only* loud failure in the entire pipeline, and therefore the one to be grateful for.

**Q43. How do you chain SFT into DPO? Write the relevant keys.**

- **Answer:**

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct      # the ORIGINAL base
adapter_name_or_path: ./runs/qwen7b_summ_sft      # your SFT output_dir
stage: dpo
finetuning_type: lora
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid
dataset: acme_summaries_prefs                     # registry entry with "ranking": true
cutoff_len: 3072
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5e-6                               # ~10-40x lower than SFT
```

- **Why the interviewer asks this:** It is the canonical two-stage pipeline and it exercises three separate facts at once: the base/adapter split, the preference dataset schema, and the learning-rate discontinuity.
- **Trap:** Keeping the SFT learning rate (`2e-4`). The observed failure is that chosen and rejected log-probabilities both collapse and the model emits 12-word summaries for everything. The second trap is `pref_beta` — carrying `0.5` over from intuition over-constrains the KL and the model barely changes.

**Q44. Write the export/merge config, and name the one key that must not be in it.**

- **Answer:**

```yaml
model_name_or_path: google/gemma-1.1-2b-it     # the ORIGINAL base, not a 4-bit copy
adapter_name_or_path: ./gemma_lora_sft_output  # the training output_dir
template: gemma                                # the merged tokenizer config inherits it
finetuning_type: lora
export_dir: ./models/gemma_lora_sft_merged
export_size: 2                                 # safetensors shard size, GB
export_device: cpu                             # cpu is safest; no GPU memory needed
export_legacy_format: false                    # false -> safetensors, true -> .bin
```

The forbidden key is **`quantization_bit`**. Everything else is a preference; that one is arithmetic.

- **Why the interviewer asks this:** Export is where frameworks stop being interchangeable and where the last silent failure lives. It also tests whether you know the output dtype story (Q63).
- **Trap:** Reusing the training YAML for export "because it has the model name in it." The notebook does exactly this and it works by accident — and that training YAML contains `quantization_bit: 4`, which is precisely what a merge must not have.

**Q45. How do you launch a multi-GPU run, and how do you serve the result?**

- **Answer:** Multi-GPU: `FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train train.yaml` plus a `deepspeed: examples/deepspeed/ds_z2_config.json` (or `ds_z3_config.json`) key. Multi-node adds `NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT`. Serving: `llamafactory-cli chat inference.yaml` for a REPL, or `API_PORT=8000 llamafactory-cli api inference.yaml infer_backend=vllm` for an OpenAI-compatible endpoint (`POST /v1/chat/completions`), with `API_KEY` set before exposing it.
- **Why the interviewer asks this:** It compresses the whole operational surface into one answer, and the `infer_backend: vllm` detail separates "I know it has an API verb" from "I know it can be production-ish."
- **Trap:** Using `infer_backend: huggingface` (the default) for production. It is a demo server; at >50 QPS merge and move to vLLM/TGI/SGLang yourself.

**Q46. Two things you must record to make a run reproducible beyond the YAML.**

- **Answer:** The YAML pins only the recipe's **shape**. You also need: (1) the **base model revision** (`revision: <commit-sha>`, not `main`); (2) the **dataset file hash** and row count; (3) the **framework commit** (`git rev-parse HEAD` of your clone, since you installed with `pip install -e .` from a moving `main`); and (4) the **backend versions** (`llamafactory-cli env > output_dir/env.txt`). Four versioned objects total, plus the registry entry, plus the adapter's own `base_model_name_or_path`.
- **Why the interviewer asks this:** It is the production answer to "how would you reproduce or roll back?", and it is the concrete remedy for the unvalidated-base-path class of bug.
- **Trap:** "I have the YAML, so it's reproducible." The framework's API churns (renames, the `USE_V1` launcher, `eval` deprecation), so `pip install -e .` from `main` next month is a different program.

**Q47. When would you set `use_rslora: true`, and when `pissa_init: true`?**

- **Answer:** `use_rslora` at **rank ≥ 32**: it scales by `α/√r` instead of `α/r`, which is what stops large ranks from *losing* their advantage under the default scaling. `pissa_init: true` initialises `A`/`B` from the base SVD instead of randomly, for faster convergence — and it needs `pissa_convert: true` to export in the conventional form. Related: `loraplus_lr_ratio` (4–16) sets a higher LR on the `B` matrix and is the answer when a rank-8 adapter converges too slowly.
- **Why the interviewer asks this:** It tests whether you treat PEFT variants as a menu with preconditions rather than a quality dial, and the `use_rslora` precondition is genuinely counter-intuitive.
- **Trap:** "Higher rank is better, so rank 64 beats rank 16." Often worse: more capacity to memorise small data, a bigger adapter, and — under `α/r` with α held at `2r` — a *smaller* effective update. Above rank 32, set `use_rslora: true` and raise α.

**Q48. How do you add a fixed system prompt, and how do you do tool-calling data?**

- **Answer:** System prompt: either `default_system: "You are…"` in the YAML (overrides the template's own), or a row-level `"system"` field plus `"columns": {"system": "system"}`, or a `{"from": "system", "value": "…"}` turn **first** in the conversation. Tool calling: sharegpt rows with `function_call` and `observation` turns in the correct alternating positions, a `tools` field containing a JSON **string**, `tool_format: qwen` (or your model's native syntax) in the YAML, and a template that has a tool formatter.

- **Why the interviewer asks this:** Tool-calling SFT is the most fragile dataset shape in the framework — it couples role positional rules, the `tool_format` string, and the template's tool formatter, and each of the three fails quietly.
- **Trap:** Training without setting `tool_format`. You get the template's default serialisation, which may not be what your inference stack emits — and if the training string and serving string differ by one character, the model learns a dialect nobody speaks.

**Q49. How do you override a single hyperparameter without editing the YAML?**

- **Answer:** Append `key=value` pairs to the CLI invocation — CLI overrides win over the YAML:

```bash
llamafactory-cli train train_gemma_qlora.yaml learning_rate=2e-4 num_train_epochs=3 output_dir=./runs/lr2e4
```

This is the mechanism that makes sweeps cheap: one committed base YAML plus a shell loop that varies exactly one key, which is also how you avoid the "I copied the config and now the two variants have drifted" problem.

- **Why the interviewer asks this:** It is the difference between a config as documentation and a config as an executable artifact, and it is the idiomatic way to run a sweep in this tool.
- **Trap:** Varying `output_dir` and forgetting `overwrite_output_dir: true`'s counterpart risk. Two runs sharing an `output_dir` with `overwrite_output_dir: true` means the second silently destroys the first; sweeps must vary the path.

**Q50. What is `tokenized_path` and when do you set it?**

- **Answer:** A directory to cache the **tokenised** (arrow) dataset. Preprocessing is often the expensive part — tokenising a 500 k-example dataset can take longer than training it. Set `tokenized_path: /fast-disk/tok/run-v3` and subsequent runs start in seconds instead of minutes.
- **Why the interviewer asks this:** It is the single biggest wall-clock saver on re-runs, it is exposed as a key and is absent from the video's config, and naming it separates someone who has iterated on a real dataset from someone who has run a demo.
- **Trap:** Naming the cache path after the run rather than the **data version**. Point it at a path named after your data version, or you will silently reuse tokens for changed data — the same class of bug as stale caches everywhere.

**Q51. What does `overwrite_cache: true` actually invalidate?**

- **Answer:** It forces re-tokenisation rather than reusing the arrow cache. The subtlety is that the cache is keyed by dataset name/path, **not by file mtime** — so editing a file in place can silently reuse stale tokens. The practical rule: `true` while iterating; `false` for frozen data (faster) but then change something else when the data changes.
- **Why the interviewer asks this:** It is the cache-invalidation semantics of a framework that hides its caching, and the failure is "I changed the data and nothing changed."
- **Trap:** Confusing `overwrite_cache` with `overwrite_output_dir`. One is about tokens, the other is about where your adapter lands; they are unrelated keys with confusingly similar names.

**Q52. How do you enable FlashAttention, and what happens if you ask for it on a T4?**

- **Answer:** `flash_attn: auto | fa2 | sdpa | disabled` — `auto` picks the best available. FA2 is an **exact** IO-aware kernel giving 2–4× attention speedup and KV memory linear in sequence length. It does not build on Turing (T4, sm_75); there you want `sdpa` or `xformers`. Note that the WebUI's "booster" dropdown (`flash_attn` / `use_unsloth` / `use_liger_kernel`) is the same choice in a different costume, and `auto` is the safe setting.
- **Why the interviewer asks this:** It tests whether you know FlashAttention is exact (not an approximation) and whether you will notice a hardware incompatibility before it becomes an install failure.
- **Trap:** "FlashAttention is an approximation, so my eval numbers differ." It is exact; any difference is dtype, kernel selection, or nondeterminism.

**Q53. What does `use_unsloth: true` do in a LLaMA-Factory YAML?**

- **Answer:** Routes the model through Unsloth's patched kernels while keeping the LLaMA-Factory config surface — the bridge between the two tools. You get Unsloth's speed on a **narrow supported-architecture list**; on an unsupported model you get stock HF behaviour without a loud warning.
- **Why the interviewer asks this:** It is the composition question, and it shows the frameworks are on different axes rather than competing: config-driven breadth (LLaMA-Factory) plus kernel-level speed on one GPU (Unsloth).
- **Trap:** Expecting the 2× Unsloth headline from the flag alone. Against a *tuned* baseline (FA2 + packing + bf16 compute dtype + all-seven-projections LoRA), the honest Unsloth delta is ~1.2–1.4× time and 15–30 % VRAM; the 2–4× figure is measured against library **defaults**.

**Q54. How do you run a hyperparameter sweep with this tool?**

- **Answer:** A shell loop over a committed base YAML, varying one key and one `output_dir` per cell:

```bash
for lr in 1e-4 2e-4 3e-4; do
  llamafactory-cli train base.yaml learning_rate=$lr \
    output_dir=./runs/sweep-lr-$lr report_to=none save_strategy=no plot_loss=false
done
```

There is no built-in sweep engine and no Bayesian search — you script it. The cost-saving move is the same as everywhere: sweep on a 10 % data slice with `max_steps` capped, then run the winner on full data.

- **Why the interviewer asks this:** It is a "how would you actually do the work" question, and it also tests whether you would run a sweep inside the WebUI (you would not — no queue, no retry).
- **Trap:** Sweeping the LR before checking the data. In every CS-15 case study the first failure was a data/registry/template problem, not a hyperparameter problem — and three of the four had a perfectly healthy loss curve.

**Q55. How do you evaluate a LLaMA-Factory adapter properly?**

- **Answer:** A **separate evaluation run** with its own registry key that never appears in any `dataset:` list:

```yaml
model_name_or_path: google/gemma-1.1-2b-it
adapter_name_or_path: ./gemma_lora_sft_output
template: gemma                    # MUST equal the training template
finetuning_type: lora
stage: sft
dataset: my_holdout_200           # never trained on
predict_with_generate: true
max_new_tokens: 256
do_sample: false                  # greedy: comparable run to run
```

And — the part most teams skip — run the **same** eval against the base model every time, so you report `{base_score, tuned_score, delta}` rather than an absolute.

- **Why the interviewer asks this:** Without the base baseline you are measuring nothing; the delta is the only number that matters. And `template`, `cutoff_len` and `default_system` must be byte-identical to training or you are measuring the template, not the model.
- **Trap:** Using `val_size` as the evaluation protocol. A `val_size` slice shares provenance, annotator and often template with the training file; it measures memorisation of a distribution, not generalisation.

**Q56. How do you serve 40 customer-specific adapters over one base model?**

- **Answer:** Export the **adapters**, not 40 merged models, and serve them with a multi-adapter server (vLLM `--enable-lora`). Forty rank-8 adapters at a 2B scale is ~800 MB of artifacts and one base resident in VRAM; forty merged models is 40 × 5 GB = 200 GB of storage, 40 VRAM-resident bases, and no way to patch a base-model CVE without re-merging all forty. Keep **one** merged model for any pinned single-tenant deployment.
- **Why the interviewer asks this:** It is the strongest practical argument for the adapter-first workflow, and it tests whether you understand that the adapter is the version-control artifact and the merged model is the deployment artifact.
- **Trap:** Deleting the adapter after merging. You have thrown away the portable 20 MB artifact and kept the bulky derivative.

**Q57. What three files does a merged deployment need alongside the weights, and for what?**

- **Answer:** The tokenizer files including `tokenizer_config.json` (**carrying the chat template**), the merged `*.safetensors` plus `config.json`, and — in your registry, not on disk — the recorded `(base revision, adapter hash, template)` triple. The merged artifact inherits the template from the export YAML's `template:` key, which is why you keep that key in a merge config rather than deleting it as unused.
- **Why the interviewer asks this:** It is the serving contract: the identical `template` and `default_system` used in training must be used at inference, and the model's `stop_words` must be honoured. A merged model served with a different template is a different model.
- **Trap:** Hand-writing a `generate()` loop for serving. You then re-implement the template and the stop words, which is exactly the bug class that produces a model that never stops generating.

**Q58. Your run has `save_strategy: epoch` and `num_train_epochs: 1`. What is wrong?**

- **Answer:** You get exactly **one** checkpoint, at the end. If the run diverges at step 150 you have nothing to fall back to, and on a spot instance a preemption at hour 22 of a 24-hour job loses everything. For single-epoch runs use `save_strategy: steps, save_steps: <steps_per_epoch/4>, save_total_limit: 3`.
- **Why the interviewer asks this:** Checkpoint cadence is the key operational link between the trainer config and the infrastructure config (spot instances, preemption, disconnects) — and it is set in the *trainer's* YAML, not in the scheduler's job spec.
- **Trap:** Setting `save_strategy: steps` with a tiny step count and no `save_total_limit`. That is disk exhaustion, and the older checkpoints are deleted before you needed them if you set the limit to 1.

**Q59. How do you resume a LLaMA-Factory run?**

- **Answer:** `resume_from_checkpoint` pointed at a checkpoint directory. The exception that matters: resume works within a run directory, but the framework does not version the *config* alongside the checkpoint in a way that guarantees the same tokenisation. **Resume only with an identical YAML** — if you changed `cutoff_len` or `template`, the tokenised cache and the checkpoint disagree, and you are fine-tuning a model on a distribution it was not mid-way through.
- **Why the interviewer asks this:** "Resume just works" is the assumption that turns a preemption from an inconvenience into a corrupted run.
- **Trap:** Resuming with a raised `cutoff_len` "because it's a superset." The tokenisation differs, so the step count and data order differ, and the checkpoint's optimizer state no longer corresponds to the data it saw.

**Q60. What does the WebUI's `output_dir` default tell you about `overwrite_output_dir`?**

- **Answer:** The WebUI's default is a **timestamped** subfolder (`saves/<Model>/lora/train_2025-12-11-17-40-51`), because `overwrite_output_dir` is not set — HF `Trainer` appends a run name. The notebook's CLI run sets `overwrite_output_dir: true`, so artifacts land directly in `./gemma_lora_sft_output`. That is the entire explanation for why the two paths in the companion notebook differ.
- **Why the interviewer asks this:** It is a small, concrete detail that proves you have compared the UI run and the CLI run rather than reading one of them, and it has a real consequence: one gives you a predictable artifact path, the other gives you an accumulating directory.
- **Trap:** Enabling `overwrite_output_dir: true` in a sweep. It is right for a one-off demo and dangerous the moment two runs share a path.

**Q61. How do you check that a template renders the right thing *before* training?**

- **Answer:** Run `llamafactory-cli chat` against the **base** model with the candidate template. If the untrained model already produces sensible, well-formatted answers to a plain question, the template is right. If it rambles, refuses oddly, or continues your prompt instead of answering it, the template is wrong — and no amount of training will fix it. This costs 60 seconds and replaces the token dump when you do not want to import internal APIs.
- **Why the interviewer asks this:** It is the highest-value pre-flight check in the cheat sheet, and it is the one that saves the wasted run. It also works without touching `get_train_args`/`get_dataset`, whose import paths move between releases.
- **Trap:** Testing the *fine-tuned* model to decide whether the template was right. By then the run is spent, and the answer is confounded by the training.

**Q62. What does the framework's own `how-to-save-in-dataset_info.txt` get wrong, and what breaks?**

- **Answer:** Both keys in it are stale. It uses `"format"` where current releases read `"formatting"`, and `"path"` where they read `"file_name"` — resolved **relative to `dataset_dir`**, so the literal `"path": "data/my_dataset/my_data.json"` with the default `dataset_dir: data` would point at `data/data/my_dataset/my_data.json`. Consequences: with `file_name` absent the loader raises `KeyError: 'file_name'` (loud, lucky); with `format` present and `formatting` absent the value is **silently ignored** and the schema defaults to `alpaca` — which happens to be correct for the alpaca example in that file, masking the bug entirely, and wrong for the sharegpt example, which then fails looking for `instruction`/`output` fields.
- **Why the interviewer asks this:** It is the origin story of the most-copied wrong config in the ecosystem, and it explains the asymmetry — because the alpaca case fails silently, alpaca-only tutorials propagated the typo for a year.
- **Trap:** Copying the file's sharegpt block. Unlike the alpaca block it does not degrade gracefully, and the failure surfaces deep in the converter rather than at the key you mistyped.

---

## Level 3 — Advanced, Internals & Theory

**Q63. Why does export dequantise, and what does that mean for a 4-bit training run?**

- **Answer:** The merge is `W_merged = W_base + (α/r)·(B@A)`, cast into the base model's dtype and written as safetensors. A 4-bit `bitsandbytes` base is an *approximation* with per-block scales, and `quantize(W) + Δ` is not `quantize(W + Δ)` — the LoRA delta was learned against the quantised forward pass, so adding it to a dequantised approximation compounds two errors and can overflow the quantisation range. So export must load the base in full precision; therefore **a run trained with `quantization_bit: 4` produces a bf16/fp16 merged artifact**, and it must be **re-quantised separately** for serving (GPTQ/AWQ/GGUF — CH-10/CH-11). That is why some people's exported models are "unexpectedly large."
- **Why the interviewer asks this:** It is the clearest example in the module of training-time and serving-time artifacts being different objects, and it is a two-line question that separates memorised knowledge from understood knowledge.
- **Trap:** (a) Putting `quantization_bit` in the merge YAML because "the training config had it." (b) Expecting a 4-bit export. Neither error is loud — (a) can produce garbage from an export that "succeeded."

**Q64. Explain, mechanically, why a wrong `template` is silent.**

- **Answer:** Three mechanisms compose. (1) The template supplies the **format strings**; a wrong one still produces a valid token sequence — it is just a sequence the model never saw during post-training. (2) It supplies the **prompt/response split**, which derives the loss mask; a wrong split still yields a mask, just a differently-placed one. (3) `fix_special_tokens` forces the tokenizer's BOS/EOS/PAD to the family's expected values; a wrong template forces the *wrong* family's values. None of these is a type error, so nothing raises. Cross-entropy then measures fit to *your* tokenised strings, and those strings are what is broken — so loss falls smoothly and the model is worse than the base at inference.
- **Why the interviewer asks this:** It is the module's central insight and it cannot be answered by reciting "template is important." The interviewer is looking for the chain from format string → mask → special tokens.
- **Trap:** "It must have raised a warning." It does not warn about a *mismatch* — only about a *missing* template (which triggers `parse_template` and a `logger.warning_rank0`).

**Q65. Why does the sharegpt converter emit empty samples instead of raising on an unknown role tag?**

- **Answer:** Because role matching is **positional**: odd turns must be `user_tag` or `observation_tag`, even turns must be `assistant_tag` or `function_tag`. The converter's design tolerates malformed rows so that one bad row in a 500 k dataset does not kill a run — it logs `Invalid role tag in […].` at WARNING, logs `Skipping this abnormal example.`, and then still appends an example with `prompt, response = [], []`. The dataset keeps its length; the content is gone.
- **Why the interviewer asks this:** It is the mechanism behind the second-most-common silent failure, and the fix (add a `tags` block, or rename the fields) follows from understanding the positional rule rather than from memorising `human`/`gpt`.
- **Trap:** Grepping for "error" in the log. You must grep for `Invalid role tag` — or, better, count supervised tokens in one sample, which catches this class and three others at once.

**Q66. Why is `freeze` usually *more* VRAM than `lora` at equal trainable-parameter count?**

- **Answer:** Because the memory regime is different, not just the parameter count. LoRA stores adapters and their optimizer state in fp16/bf16 over a tiny slice (~14 bytes per *trainable* parameter, and there are few). `freeze` trains the last N layers' *full-precision* weights, so it holds fp16/bf16 gradients plus fp32 AdamW moments plus an fp32 master copy — 14 bytes per trainable parameter again, but on a much larger and much less controllable slice, and with no quantization of the base. It is a middle ground in *quality*, not in memory.
- **Why the interviewer asks this:** It is a genuine misconception that survives because "fewer layers" sounds cheaper, and it requires you to separate parameter count from memory regime.
- **Trap:** Recommending `freeze` as the "low-VRAM option between LoRA and full." If you are memory-bound the answer is LoRA + `quantization_bit: 4`, which at 7B is 4.9 GB against 14.7 GB for bf16 LoRA and 91.6 GB for full fine-tuning.

**Q67. Walk the memory arithmetic of the demo run and say where each byte goes.**

- **Answer:** Model is Gemma-1.1-2B (2.51 B params), `quantization_bit: 4`, `batch 1`, `cutoff_len 1024`, `gradient_checkpointing: true`, `lora_rank 8`, `lora_target: all`.

| Component | GB | Derivation |
|---|---|---|
| 4-bit NF4 base weights | ~1.35 | 2.51 B × ~0.54 B/param with double quant |
| LoRA params + grads + AdamW + fp32 master | ~0.14 | 9.81 M × 14 B/param |
| Activations (gradient checkpointing on) | ~0.5 | 18 layers × 1 × 1024 × 2048 × 2 B ≈ 76 MB, plus one block's recomputation buffer |
| CUDA context, cuBLAS/cuDNN workspaces, bnb dequant buffers, fragmentation | 1.0–2.5 | unavoidable overhead |
| **Realistic peak** | **≈ 5.5–6.5** | on a 15 GB Colab T4 |

Without gradient checkpointing the same run holds per-layer attention scores (`1 × 8 × 1024² × 2 = 16.8 MB`) plus MLP intermediates (`1 × 1024 × 16384 × 2 = 33.6 MB`) — roughly 0.9 GB of persistent activations plus transients — pushing the peak to ~9–10 GB. That margin is the entire reason the flag exists.

- **Why the interviewer asks this:** It is the module's sizing question, and the *decomposition* — base, adapter state, activations, overhead — is what transfers to every other model. The single most common wrong answer is to compute weights and stop.
- **Trap:** Forgetting the 1–2.5 GB of overhead. A "3 GB" model routinely peaks over 5 GB, and that gap is where OOMs live.

**Q68. Scale that recipe: what fits where for 7B, 13B and 70B?**

- **Answer:** From `code/common/memory.py --table` (single GPU, gradient checkpointing on):

| Model | `finetuning_type: full` | `lora` (bf16 base) | `lora` + `quantization_bit: 4` |
|---|---|---|---|
| 1B | 14.5 GB | 2.4 GB | 0.9 GB |
| 3B | 39.4 GB | 6.4 GB | 2.2 GB |
| **7B** | **91.6 GB** | **14.7 GB** | **4.9 GB** |
| 8B | 104.7 GB | 16.7 GB | 5.6 GB |
| 13B | 170.0 GB | 27.1 GB | 9.0 GB |
| 70B | 913.8 GB | 144.5 GB | 46.8 GB |

Add 10–20 % for allocator and framework overhead. So: 7B full FT does not fit on one 80 GB card (91.6 GB before activations); 7B bf16 LoRA fits a 24 GB card comfortably; 7B QLoRA fits a 16 GB free T4. 70B QLoRA at 46.8 GB needs one A100-80 or FSDP+QLoRA across 2×24 GB.

- **Why the interviewer asks this:** These are the numbers to have on instant recall, and the ~6× full-vs-LoRA and ~3× LoRA-vs-QLoRA ratios are the two rules of thumb that let you answer any sizing question in ten seconds.
- **Trap:** Quoting the table as a *ceiling*. It is a model at a fixed batch and sequence length; raising `cutoff_len` or `per_device_train_batch_size` moves it, and long-sequence OOMs are the ones that appear mid-run rather than at step 1.

**Q69. ZeRO-2 or ZeRO-3 for a LoRA run on four GPUs? Why is ZeRO-3 not automatically better?**

- **Answer:** **ZeRO-2.** ZeRO-1 shards optimizer state, ZeRO-2 additionally shards gradients, ZeRO-3 additionally shards **parameters** — which means every forward and backward pass must all-gather them. For a LoRA run, the weights already fit (LoRA exists precisely because they do), so ZeRO-3 buys nothing and costs per-layer communication on every step. CH-15 states it flatly: `ds_z2_config.json` for LoRA on multi-GPU, `ds_z3_config.json` for full FT when the weights do not fit.
- **Why the interviewer asks this:** It is the module's canonical "the bigger number is not the better number" question, and it is exactly the answer the interviewer wants to hear phrased in terms of *what is being communicated for what benefit*.
- **Trap:** "ZeRO-3 maximises memory efficiency, so use ZeRO-3." It is the most memory-efficient and the most communication-hungry; on PCIe without NVLink it can make a job several times slower than ZeRO-2, or slower than a single GPU. The related trap is setting `deepspeed:` on a single GPU — ZeRO-3 assumes ≥2 processes.

**Q70. Why does the framework's `template: default` not mean "use the model's own template," and what does that tell you about the design?**

- **Answer:** `default` is a registered plaintext template emitting `Human: …\nAssistant: …\n` with a `System:` prefix and `replace_jinja_template=True`; the auto-detect behaviour lives on the *omitted* path, via `parse_template(tokenizer)`. The design lesson is that "which template did I get" is answered by the code path taken, not by the key's name — and the framework's own docs have been flagged as stale on exactly this point. The practical rule: prefer an explicit, family-correct template name over both `default` and omission, because an explicit name is version-pinned to your code rather than to a Hub revision that can change under you.
- **Why the interviewer asks this:** It is a precise-internals question that has a real consequence (Vicuna-format plaintext on a Llama-3 model), and it demonstrates the candidate reads source rather than blog posts.
- **Trap:** Using `template: default` "to be safe." It is one of the least safe values in the registry.

**Q71. `stage: pt` on a text-only dataset — what happens to instruction data, and what is the correct way to do unsupervised SFT?**

- **Answer:** `stage: pt` uses the pretraining path and consumes your instruction data as **raw text**, destroying its structure. The correct way to do unsupervised SFT on text is `stage: sft` with a `prompt → text` column mapping (the `my_plain_text` registry entry), which the alpaca converter accepts and which yields an empty response so the whole sequence receives loss.
- **Why the interviewer asks this:** It is a trap where two similar-sounding options produce very different corpora, and the fix is a `columns` remap rather than a stage change — which is only obvious if you understand that `columns` is a type declaration.
- **Trap:** "I want continued pretraining, so `stage: pt`." That is right for genuinely raw text and wrong for data that happens to be stored in an instruction file.

**Q72. Why does packing corrupt chat data, and what exactly does `neat_packing` change?**

- **Answer:** Packing concatenates samples to fill `cutoff_len`. In chat data the boundary between examples is an EOS/special-token structure, and if that is handled wrongly two examples become one training row — the model learns to continue someone else's answer. `neat_packing` prevents **cross-sample attention**, so each packed sample keeps its own attention window and cannot attend to its neighbours; the packing is then a batching optimisation rather than a data transformation. `packing` is auto-on for `pt`, off for `sft`; `neat_packing` is off by default.
- **Why the interviewer asks this:** It is the module's clearest example of a throughput knob that is also a correctness knob, and the honest answer names the mechanism rather than a rule.
- **Trap:** Turning on `packing: true` alone for SFT "for the 2–4× throughput" and then debugging a model that answers the wrong question.

**Q73. How is the loss mask derived, and why can't you fix a masking problem with `ignore_pad_token_for_loss`?**

- **Answer:** The mask comes from the **template's** prompt/response split: the format strings say where the prompt ends, and everything before the split point gets `IGNORE_INDEX = -100`. Two explicit overrides exist: `train_on_prompt` (do not mask the prompt — raises `ValueError` if the template has `efficient_eos`, an explicit guard) and `mask_history` (mask earlier assistant turns). `ignore_pad_token_for_loss` governs **padding** positions only; setting it `false` teaches the model to emit pad tokens, which is a bug rather than a feature. It is `true` by default and should stay there.
- **Why the interviewer asks this:** It is the most commonly conflated trio of keys in the framework, and the answer requires knowing that masking is a template property rather than a flag.
- **Trap:** Writing `ignore_pad_token_for_loss: false` "so the model learns the prompts too." It does not do that, and it does do harm.

**Q74. LoRA rank 64 versus rank 16 — and what does `α/r` have to do with it?**

- **Answer:** Often **worse** at rank 64 on small data: more capacity to memorise, a larger adapter, and — under the default `α/r` scaling with α held at `2r` — a **smaller effective update** as r grows, so the adapter trains *harder* to move the same distance. The fix for large ranks is `use_rslora: true` (scaling by `α/√r`) plus a proportionally larger α. In the demo run, rank 8 on `lora_target: all` gives 9.81 M trainable params (0.39 %); rank 16 → 19.6 M; rank 32 → 39.2 M; rank 64 → 78.4 M (3.1 %).
- **Why the interviewer asks this:** It is a "do you understand the parameterisation" question disguised as a "which value" question, and the `α/r` interaction is the part most candidates miss.
- **Trap:** "Higher rank, more capacity, better model." Above rank ~64 on a dataset under ~10 k examples, higher rank mostly buys overfitting plus optimizer state. The exception direction also matters: a vocabulary-heavy task (the CS-15 legal case study) saw +4 points from rank 32 over rank 16 — rank is a capacity dial, and capacity should match the task.

**Q75. The instructor says "BF16 is for the higher-end GPU, fp16 for the lower-end GPU." What is the mechanism, and when is he right?**

- **Answer:** fp16 has a 5-bit exponent, so attention logits above ~65 504 overflow to `inf` and `softmax(inf)` produces `nan` that propagates through the graph. bf16 has fp32's exponent range and cannot overflow that way, but it needs hardware support (Ampere and later, plus some 30-series). So the rule is **hardware**, not preference: `fp16: true` on a T4/P100/V100, `bf16: true` on A100/H100/L4/4090/30-series. The observed symptom of getting it wrong is a `nan` loss at a random step — often around step 200 — when an unusually long or unusual example arrives.
- **Why the interviewer asks this:** It is a rule of thumb the instructor states and the candidate must be able to *derive*, because deriving it produces the fallback (4× lower LR + `max_grad_norm: 1.0`) when bf16 genuinely is unavailable.
- **Trap:** Treating bf16 as strictly better and setting it on a pre-Ampere card. The failure is `RuntimeError: "addmm_impl_cpu_" not implemented for 'BFloat16'` or a silent CPU fallback, depending on the stack.

**Q76. Why is QLoRA slightly worse than bf16 LoRA at the same adapter size?**

- **Answer:** Because the **frozen base is quantised**, and the adapter cannot fully compensate for quantisation error in the weights it is adapting to: the LoRA delta was learned against the 4-bit forward pass, so the error is in the function being approximated, not only in the adapter. At `quantization_bit: 8` the gap is small; at 4 in the *base path* it is measurable on hard tasks. If you have the VRAM, `finetuning_type: lora` **without** `quantization_bit` is strictly better quality for the same adapter size.
- **Why the interviewer asks this:** It is the honest version of "QLoRA is free," and in an interview where a candidate claims QLoRA costs nothing it is the follow-up that finds the boundary.
- **Trap:** Two opposite ones. "QLoRA loses quality because the model is 4-bit" — the adapter trains in bf16 and the measured gap is typically under a point. And "QLoRA is free" — it is a *memory* trade that is usually, but not always, the right one.

**Q77. Why is LLaMA-Factory's start-up overhead 20–40 seconds per run, and when does that matter?**

- **Answer:** Python imports (torch, transformers, triton, bitsandbytes), registry and template resolution, and model loading — all before the first step. It matters exactly when the run is short: on a ≤3B model with a total compute budget under an hour, the overhead is a meaningful fraction of a 10-minute run, and Unsloth or a plain script wins. It does not matter at 7B+ with a real dataset, where the same 30 seconds is noise.
- **Why the interviewer asks this:** It is a "when is this the wrong tool" question at the small end, and it is the argument for having two frameworks rather than one (Q107).
- **Trap:** Over-optimising it. The fix is not to shave imports; the fix is to recognise that a 5-minute experiment on a 1B model is a different tool's job.

**Q78. Why can a merged model be worse than the adapter it was merged from, even when the merge is correct?**

- **Answer:** Two distinct reasons. (1) **Precision**: the merge is `W + (α/r)·BA`, a floating-point accumulation; merging a bf16-trained adapter into an fp16-loaded base rounds the delta and shifts outputs, most visibly at long context. Merge in the base's exact dtype, on CPU (`export_device: cpu`, `export_legacy_format: false`). (2) **Template drift**: the merged artifact inherits the tokenizer config from the export YAML's `template:` key, and if you serve it with a different template or without the training `stop_words`, the model is fluent and wrong. Neither shows up as an error.
- **Why the interviewer asks this:** It is the "it worked in the notebook and not behind the API" bug, decomposed into the two independent causes — arithmetic and semantics.
- **Trap:** A/B-ing the merged model against the base and concluding the fine-tune failed. Compare adapter-vs-merged on a fixed prompt set *first*; if they disagree, the problem is the export, not the training.

**Q79. Why is a multi-GPU QLoRA run often *slower* than a single-GPU one?**

- **Answer:** `bitsandbytes` 4-bit layers are not sharded under plain DDP: each rank holds a **full copy of the quantised base**, and quantise/dequantise kernels run per-rank with no gain from the extra devices. You pay all-gather of gradients every step and get no memory benefit — the opposite of the point. The fix is either one GPU with enough VRAM, or FSDP+QLoRA / `deepspeed` with `zero3_init_flag`, or simply not quantising when the model fits unquantised across the pool.
- **Why the interviewer asks this:** It is the counter-intuitive scaling result, and it tests whether "more GPUs" is being reasoned about as a bandwidth-and-sharding question instead of a compute question.
- **Trap:** Diagnosing it as a broken config. The config is correct; the strategy is wrong for the setup.

**Q80. Why can the loss be low, the curve smooth, and the model still worse than the base?**

- **Answer:** Because the loss measures fit to *your* tokenised strings, and there are four ways those strings can be wrong while remaining perfectly learnable: the template mismatch (wrong format, wrong mask, wrong special tokens), the empty-sample bug (`Invalid role tag` → `prompt, response = [], []`), a `cutoff_len` that truncates the answer away so the model learns prompt→truncated-nothing, and a mask that puts loss on the prompt so the model learns to generate user turns. In every case the model becomes a *better* predictor of the wrong objective.
- **Why the interviewer asks this:** It is the module's thesis, and it is the question that separates "I ran the notebook" from "I have shipped a fine-tune."
- **Trap:** Declaring the run healthy because the curve is smooth. The correct diagnostic is a **held-out generation comparison against the base model** with the same template — not the training loss.

**Q81. Why does `report_to: none` belong in every local config?**

- **Answer:** Three reasons. (1) Without it the default can be `wandb` or `all`, and a machine with no credentials either blocks on an interactive login prompt (which hangs a batch job with no log output — Q110-adjacent) or silently fails to log. (2) It removes a network dependency from a job you may need to run air-gapped. (3) It keeps `trainer_log.jsonl` the single source of truth, which is what your own tooling reads. Turn it back on (`report_to: wandb`) only when the run is long enough that remote curves earn their keep.
- **Why the interviewer asks this:** It is a small line that indicates whether the candidate has actually run this in CI, and it links to the hang-in-startup failure mode.
- **Trap:** Setting it once and forgetting the *sweep* case, where a wrapper re-enables logging for one run and every subsequent job inherits the credential failure.

**Q82. What does the WebUI's place in the workflow actually buy you, and what does it structurally prevent?**

- **Answer:** It buys a `train_web.py` server on port 7860 with five tabs (Train, Evaluate/Predict, Chat, Export, and the dataset/config editors), a live loss chart, and — the genuine benefit — an *interactive* template check: chat with the base model in the Chat tab under the candidate template before committing to a multi-hour run. It structurally prevents reproducibility: the WebUI's `output_dir` carries a **timestamp**, so the notebook's two paths (the reported checkpoint dir vs the export path) differ by design, and config state lives in the server rather than in git.
- **Why the interviewer asks this:** It tests whether the candidate treats the GUI as a *probe* (good) or as the *system of record* (bad), and it is the setup for the pre-flight question — the same check belongs in CI, headless.
- **Trap:** "The WebUI is for beginners." It is not; it is the fastest way to answer "is my template right?" interactively, and that check is worth more than any dashboard.

**Q83. Why is there no schema validation between `model_name_or_path` and `template`, and what would you do about it?**

- **Answer:** Because the two are resolved in different registries with no declared relation — the template registry is a flat `TEMPLATES` dict and the model path is an open string; the framework cannot know that `Qwen/Qwen3-4B` implies the `qwen3` template. So the mismatch surfaces only at generation time, if at all. What you do about it is a **pre-flight gate** (Q94): assert the resolved template against a small allowlist keyed by model family, dump one tokenised sample and check the supervised-token fraction, and chat with the base model before training starts.
- **Why the interviewer asks this:** It is the "what would you build" bridge from a defect to engineering practice, and it is a Level 4 question wearing Level 3 clothes.
- **Trap:** Assuming a newer version fixed it. It has not; the `ValueError(f"Template {data_args.template} does not exist.")` guard fires only for a template name that is genuinely absent from the registry, never for one that exists and does not match the model.

**Q84. How do you reason about the quality budget across SFT → DPO?**

- **Answer:** As a sequence of small, individually-verifiable steps, each with its own eval, where **DPO cannot repair a broken SFT** — it only sharpens a preference *between two candidate responses that the SFT model already produces*. So: SFT first (LoRA, rank matched to task complexity), score with a held-out generation set, then DPO with `adapter_name_or_path` pointing at the SFT adapter, `pref_beta: 0.1`, `learning_rate: 5e-6` (an order of magnitude below SFT's 1e-4) and `pref_loss: sigmoid`. If SFT already produces both responses reliably and the preference is really about formatting or safety, DPO is the wrong lever — a better SFT dataset is cheaper.
- **Why the interviewer asks this:** It tests whether the candidate has a *pipeline* mental model or a list of stages, and whether they know the order of magnitude gap in learning rate that chaining implies.
- **Trap:** Starting DPO from the base model. It runs, loss falls, output is nonsense — the same silent-failure shape as a template mismatch, one level up.

**Q85. What are the real costs of `lora_target: all`?**

- **Answer:** It attaches adapters to every linear layer including the MLP (`gate`, `up`, `down`) rather than only `q_proj`/`v_proj`. Quality improves — more of the network is adapted, and the effect is largest for format and style changes — but trainable parameters scale ~2–3× (7B rank 16: ~19.6 M for q/v against ~39–160 M for all), optimizer state scales with them, and the adapter's deploy-time delta is bigger. `all` is a safe *default* precisely because LoRA is low-rank; it is not free, and on a tight VRAM budget `lora_target: q_proj,v_proj` is the first thing to try.
- **Why the interviewer asks this:** It is the question that catches the candidate who has memorised the setting but not the arithmetic behind it — and the arithmetic (`Σ r × (in + out)` per adapted matrix) is checkable against the observed 9.81 M.
- **Trap:** "It adapts the whole model." It adapts a low-rank slice of every linear; the base weights remain frozen and the merged artifact is exactly base + delta.

**Q86. What makes `stage: rm` and `stage: ppo` a different kind of job from `sft`/`dpo`?**

- **Answer:** They are the only two stages that need **more than one model in memory at once**. `rm` trains a reward model on `chosen`/`rejected` pairs and is architecturally an SFT with a scalar head — cheap, and reusable. `ppo` then requires the policy, a **frozen reference copy of the policy**, the reward model, and a value head — four sets of weights, plus a rollout loop, which is why PPO is the stage people abandon for DPO. `dpo`/`orpo`/`simpo` get you the preference objective with two models (policy + reference) and no sampling loop, which is exactly why they displaced PPO for most teams.
- **Why the interviewer asks this:** It is the memory-and-complexity dimension of the stage question, and it explains *why* the field moved from RLHF-Proximal to Direct — not because DPO is theoretically better but because it fits.
- **Trap:** Choosing `ppo` for a small team with no prior RL experience. The failure mode is not a crash; it is a reward-hacked policy or a job that cannot be sized on your hardware.

---

## Level 4 — System Design & Scenario

*Each of these is a 10–15 minute whiteboard. Expected shape: requirements → constraints → design → trade-offs → failure modes. The interviewer is grading the questions you ask and the failure modes you name, not the YAML.*

**Q87. Design a fine-tuning harness for 12 model families behind one compliance boundary.**

- **Requirements:** 12 families (Llama/Gemma/Qwen/Mistral/Phi/DeepSeek + others), one internal package, all runs audit-logged, no training code changes between families.
- **Constraints:** 3 engineers, 8×A100-80 mixed with 4×24 GB, no internet from the training nodes, every run attributable to a ticket.
- **Design:** LLaMA-Factory as the execution engine behind a **thin internal wrapper**: a `family → (template, lora_target, dtype)` allowlist table (the validation the framework lacks), a config template per family, and a `dataset_info.json` registry vendored into the repo so dataset keys are versioned alongside configs. Pin the framework version in the image; freeze model revisions (`revision: <sha>`) so `parse_template` cannot drift under you. Run `llamafactory-cli train|export` only through the wrapper, with the export step mandatory and the merged artifact registered with the `(base revision, adapter hash, template)` triple.
- **Trade-offs:** Losing the WebUI (fine — it is a probe, not a system of record); some flexibility vs TRL, regained by the wrapper writing YAML; one framework to patch instead of twelve scripts.
- **Failure modes:** family added without a `template` entry (blocked by the gate); a model revision moving on the Hub (blocked by pinned revisions); template/tokenizer drift after a framework upgrade (blocked by a golden-token regression test).
- **Follow-up the interviewer may ask:** "Why not one framework per family?" Because the *coverage* is the framework's value proposition — the same YAML trains a different family by changing two keys — and the compliance cost of N toolchains is N times the audit surface.

**Q88. Design SFT → DPO → export for a legal summariser with a fixed budget.**

- **Requirements:** 7B base, 8k-token documents, summarise into a fixed clause template, one 24 GB GPU, a budget of ~40 GPU-hours.
- **Constraints:** Peak VRAM 24 GB, no multi-node, quality must beat the base model measurably on a held-out set of 200 documents.
- **Design:** QLoRA at `cutoff_len: 4096` (not 8192 — you cannot fit it, and truncation must be verified not assumed) with `packing: false` for chat data and `gradient_checkpointing: true`; `lora_rank: 32` because the CS-15 legal case found rank mattered more than usual on clause-heavy text; 3 epochs, `learning_rate: 1e-4`, `lr_scheduler_type: cosine`, `warmup_ratio: 0.1`; `val_size: 0.05` explicitly, because the default is 0. Then DPO: `stage: dpo` with `adapter_name_or_path` at the SFT adapter, `pref_beta: 0.1`, `learning_rate: 5e-6`, `pref_loss: sigmoid`. Export with `export_device: cpu` and **no `quantization_bit`**, then compare the merged artifact against the base on the 200 held-out documents under the training template.
- **Trade-offs:** rank 32 costs ~39 M trainable params instead of ~9.8 M — accepted for capacity; 4096 tokens truncates long documents — accepted, but *measured* (histogram the token lengths first), because a summariser that never sees the last 30 % of a contract learns to summarise the first 70 %.
- **Failure modes:** truncated targets (the loss falls and the summaries are plausible-but-incomplete — the worst outcome in a legal setting); template mismatch making the model fluent in the wrong format; DPO before SFT is verified.
- **Follow-up:** "Where does the 40-hour budget go?" Roughly 1/3 SFT, 1/4 DPO (preference data is smaller), the rest to eval runs and export — which is why the eval set is defined before training, not after.

**Q89. Design multi-tenant serving for 40 fine-tuned adapters.**

- **Requirements:** 40 adapters over ~6 bases, p95 under 2 s for 200-token completions, a new adapter ships weekly.
- **Constraints:** Two L40S, no retraining during business hours, each tenant isolated.
- **Design:** Serve the **adapters**, not 40 merged models: vLLM with `--enable-lora --lora-modules`, one base per engine, adapters loaded from the registry at start-up with a max-LoRA-rank budget. Every adapter is stored with its `(base revision, template, stop_words, adapter hash)` record; the serving layer refuses to load an adapter whose base revision does not match the base being served. Export/merge is reserved for the tenant who needs a standalone artifact.
- **Trade-offs:** LoRA serving costs some throughput versus merged weights on the hot path but makes a new tenant a config change rather than a redeploy; isolation is at the process level per base, not per tenant — accepted if tenants share a trust boundary, otherwise one engine per tenant.
- **Failure modes:** a template mismatch at serving (the training YAML's template is not recorded → model answers fluently in the wrong format); rank heterogeneity forcing the engine to a high max rank and losing the batching benefit; an adapter silently falling back to the base when its name is misspelled — which is why the serving layer validates adapter names against the registry at start-up.
- **Follow-up:** "How do you roll back one tenant?" Repoint that adapter name to the previous hash; this is why adapters ship as immutable artifacts, never as mutable paths.

**Q90. The 14B model is 4× the cost of the 3B and gives +1 point. Argue both sides.**

- **Answer-shape:** The interviewer wants to see the candidate *not* default to "bigger is better." The cost case: 14B full FT needs 170 GB (table) so it is LoRA-only by construction, at 27.1 GB bf16 or 9.0 GB at 4-bit versus 3B at 6.4 GB / 2.2 GB; latency is ~4× at inference; iteration cost is ~4×, so a 5-experiment day becomes a 1.25-experiment day. The quality case: +1 point on the eval set is real but is it *outside the eval's confidence interval*? With 200 eval examples a 1-point delta is usually inside the noise — so the honest answer is "the delta may be noise; measure it, then decide." If it is real, the next question is *where* it comes from: if it is a long-context or world-knowledge effect, a 3B with better data will not close it; if it is a formatting effect, it will.
- **Why the interviewer asks this:** It tests whether the candidate reasons with intervals and budgets rather than with a leaderboard, and whether they can be argued out of a default.
- **Failure mode to name:** Shipping the 14B, then discovering the eval set was 200 examples and the +1 was noise — after the serving bill has been committed.

**Q91. Design the air-gapped path.**

- **Requirements:** A cluster with no internet, a gated base model, a private dataset, and a regulatory requirement that no bytes leave the network.
- **Constraints:** No Hub access at runtime, no telemetry, reproducible in six months.
- **Design:** Pre-fetch everything into an artifact store: model weights at a pinned `revision:`, the tokenizer, and the framework's dataset files. Disable telemetry and network logging (`report_to: none`, no `WANDB_*` in the environment); serve the private dataset through a local `dataset_dir` and vendor `dataset_info.json` into the repo. Vendor the *template registry* itself if you have custom templates, because it is Python source, not config. Pin the framework version in the image and record it in the run manifest.
- **Trade-offs:** You lose `trust_remote_code` conveniences and Hub dataset shortcuts, so every dataset becomes a local file and every model a local path — more ops work, fully auditable.
- **Failure modes:** a `load_dataset("…")` Hub call reintroduced by a well-meaning engineer (blocked by a no-egress test in CI); a gated model with a 401 (Q108) or a token baked into an image.

**Q92. You are handed a 900-line hand-written HF + PEFT training script. Migrate it or keep it?**

- **Answer-shape:** Migrate if the script is *standard* — LoRA/full SFT, alpaca/sharegpt data, `Trainer`, `peft` — because LLaMA-Factory reproduces it in ~40 lines of YAML and buys you the registry, checkpointing, DeepSpeed, export and the multi-family coverage for free. Keep it if the script contains any of: a **custom loss**, a **custom collator**, a **custom training loop** (curriculum, custom sampling, multi-objective), or a training-time dependency the framework does not own (Q1's "wrong choice" list). The migration check is arithmetic: reproduce one run and diff `trainer_log.jsonl` — the same loss trace to within floating-point noise means the migrate was faithful; a different trace means the two are not doing the same thing, and you now know which one you actually wanted.
- **Trade-offs:** YAML is a *less expressive* language than Python, deliberately — you trade flexibility for reproducibility and coverage. The failure mode of migrating a script you did not understand is that its quirks (a custom pad token, a mask tweak) disappear silently.
- **Failure mode to name:** migrating, seeing a similar-looking loss curve, and never diffing the supervised-token count. "It looks the same" is not evidence.

**Q93. Design a spot-instance 70B multi-node job that must not lose a week of work.**

- **Requirements:** 3×8×A100-80 spot, ~4 days of training, preemption with a ~2-minute warning.
- **Constraints:** 70B full FT is 913.8 GB — do not attempt it on this pool; 70B QLoRA is 46.8 GB and fits one node. So the first design decision is to *shrink the job*: QLoRA on one 8×A100 node with `ds_z2_config.json`, or FSDP+QLoRA across two.
- **Design:** `save_strategy: epoch` with `save_total_limit` sized so checkpoints survive; the spot warning triggers a checkpoint write via the instance's shutdown hook; `resume_from_checkpoint` restarts with **byte-identical YAML** (the Q59 rule); `overwrite_output_dir: false`; checkpoints on durable storage, not instance-local disk. Note `sharding` on multi-GPU checkpoints and the DeepSpeed sharded-to-unsharded conversion before export.
- **Trade-offs:** more frequent saves cost throughput and disk; `save_strategy: steps` with a small interval is the safer extreme and the reason to size checkpoint storage first.
- **Failure modes:** a resumption with a *different* YAML (silently restarting from step 0 or from a mismatched optimizer state); spot termination during the export step; the `Bus error (core dumped)` from too many DataLoader workers (Q109).

**Q94. Design the pre-flight gate every run must pass.**

- **Requirements:** fail in under two minutes, before any GPU-hour is spent, for every config in the repo.
- **Answer — the gate has six checks:**
  1. **Config validity**: every key exists for this framework version; `formatting` not `format`; `file_name` not `path`; no `quantization_bit` in an export config; `val_size > 0`.
  2. **Registry resolution**: every `dataset:` key exists in `dataset_info.json` and its `file_name` resolves inside `dataset_dir`.
  3. **Template allowlist**: the resolved template is in the family's allowed set (the check the framework does not do).
  4. **Token-and-label dump**: render one sample, print token strings and labels, and assert the supervised-token fraction is in the expected band (5–15 % for sharegpt chat; 100 % only for `stage: pt`; **0 % is a hard failure**).
  5. **Base-model chat probe**: generate from the *base* model with the training template and `stop_words`, and assert the output stops at the expected stop token.
  6. **Sizing**: predicted peak VRAM from the table plus 15 % overhead is under the device budget.
- **Why the interviewer asks this:** It is the answer to "the framework has no validation," and it is a *design* answer — each check maps to a specific silent failure the candidate should be able to name.
- **Failure mode to name:** the gate that only checks YAML syntax. It passes 100 % of the runs that fail.

**Q95. Argue the case for LLaMA-Factory versus Unsloth and Axolotl for a 5-person team that fine-tunes weekly.**

- **Answer-shape:** Not a leaderboard — a fit argument. **LLaMA-Factory** if the team's variability is *across models and data formats*: 100+ families, `dataset_info.json` as a schema layer, `stage:` covering pt→sft→rm→ppo→dpo in one tool, a WebUI for interactive template checks, and the export path — but a Python-side ceiling (no custom loss inside the YAML) and no multi-GPU advantage of its own. **Unsloth** if the workload is *one or two models, single-GPU, and iteration time dominates*: 2–4× versus naive HF defaults, honestly ~1.2–1.4× and 15–30 % VRAM against a *tuned* FA2 baseline, and no GUI or config layer. **Axolotl** if the team is *multi-node at scale* — FSDP2, ZeRO 1–3, 20+ dataset `type:` strategies, `preprocess --debug` as a first-class data check. The fit argument for a weekly-cadence 5-person team is usually LLaMA-Factory **plus** a second tool for the one model that is hot, and the interview answer should say so instead of picking a winner.
- **Why the interviewer asks this:** It is the comparison the module exists inside, and the honest answer contains the *decomposition* of the headline speed claims rather than a number.
- **Trap:** Quoting "2× faster and 60 % less memory" as a framework comparison. That figure is against untuned defaults; LLaMA-Factory with FA2 and the right batch size closes most of it.

**Q96. Design versioning and rollback for 40 production adapters.**

- **Answer:** An adapter is not a directory, it is a **record**: `(base model, base revision, adapter hash, template name, stop_words, dataset registry hash, framework version, eval score)`. Store adapters immutably (content-addressed by hash), never overwrite a path; promote by moving a *pointer* per tenant. Rollback is repointing the pointer, and it must be possible without a retrain — which requires the merged artifact and the adapter to both be retained, plus the exact template. The eval score is part of the record so that a rollback can be justified, and the `(base revision, template)` pair is what lets you detect that a *base* has moved underneath every adapter at once — the failure that turns a one-tenant bug into a forty-tenant one.
- **Why the interviewer asks this:** It is the operations question that a training-framework interview usually skips, and it is where "I ran the notebook" candidates run out of road.
- **Failure mode to name:** mutable `output_dir` paths. Two runs writing the same directory is how you discover that your production adapter is from last Tuesday's experiment.

---

## Level 5 — Debugging & Incident Response

*Each item is a symptom as it presents in a real incident channel. The expected answer is: most likely cause → cheapest discriminating test → fix → how you would have prevented it.*

**Q97. "The loss curve is textbook — 2.4 down to 0.6 with no spikes. The model is worse than the base at everything."**

- **Most likely cause:** The template/model mismatch — the format strings do not match what the base was post-trained on, so the model was trained on strings it has never seen. Second: empty samples from an unrecognised role tag.
- **Cheapest discriminating test (2 minutes, no GPU):** Render one sample with `template:` and print the token strings, the labels, and the supervised-token fraction. 5–15 % supervised with the response text present is healthy; a plausible-looking render with the *wrong* special tokens is the bug; `prompt, response = [], []` shows as a supervised fraction of 0 %.
- **Fix:** Set the family-correct template by name; if the data uses its own field names, add a `tags` block to the registry entry rather than reformatting the file.
- **Prevention:** Chat with the **base** model under the candidate template before training (Q61) — a template mismatch is visible in the base model's output immediately. Then gate it (Q94).
- **Trap:** Lowering the learning rate. The run is converging perfectly; it is converging on the wrong objective. Any hyperparameter change "fixes" it by making the same wrong thing happen more slowly.

**Q98. "Loss is 0.000 from step 1 and never moves."**

- **Most likely cause:** Label leakage into the prompt — the answer is in the input, so the task is trivial. Usually produced by a converter that put the response text in both `prompt` and `response`, or by a `prompt`/`response` column mapping that aliases the same column to both.
- **Cheapest test:** Print one rendered sample. If the response text appears twice, or if the prompt field literally contains the answer, that is it. Also check the supervised fraction — near 100 % on chat data is the tell.
- **Fix:** Correct the `columns` map so `prompt` and `response` are distinct fields; if the source has a single `text` column holding a whole conversation, use the sharegpt `conversations` shape instead of aliasing.
- **Prevention:** The token/label dump in the gate, plus a "loss should start near ln(vocab) ≈ 9–10 for a fresh model" sanity expectation. A first-step loss under ~1.0 on a chat dataset is a data bug, not a good model.
- **Trap:** Reading 0.000 as "perfect convergence" and shipping. This model will be an expensive copy of the input.

**Q99. "Training was healthy for ~200 steps and then the loss went to `nan`."**

- **Most likely cause:** fp16 overflow. fp16's 5-bit exponent overflows around 65 504; an attention logit exceeding it produces `inf` and `softmax(inf)` → `nan`, which then poisons every parameter. The step-200 timing is the signature — it is the 200th example, not the 200th step, and it is usually an outlier in length or content.
- **Cheapest test:** Reload the last checkpoint and re-run with `bf16: true` (if the GPU supports it) — if the `nan` disappears, confirmed. On pre-Ampere hardware, instead drop the learning rate 4× and set `max_grad_norm: 1.0`, then re-run.
- **Fix:** `bf16: true` on Ampere+; otherwise lower LR + clipping, and consider filtering the extreme-length tail out of the dataset.
- **Prevention:** Choose the dtype by *hardware* at config time, not by preference; add the length histogram to the pre-flight so you know the tail before you meet it.
- **Trap:** Restarting from scratch with the same config and hoping. The outlier is still in the dataset and the run will fail again at the same place.

**Q100. "Loss is stuck at 2.30 and has not moved for 800 steps."**

- **Most likely cause:** Nothing is being trained. Either the adapter never attached (wrong `lora_target` for this architecture's module names — e.g. `all` resolving to no linear layers on an unusual architecture), or the labels are all `IGNORE_INDEX = -100` so every token is masked and the loss is a constant. 2.30 ≈ ln(10) is the value for a uniform distribution over ~10 classes; a truly frozen model gives a constant at ln(vocab).
- **Cheapest test:** Print the count of trainable parameters at start-up — a `lora` run reporting 0 trainable params is the whole answer; and dump one sample's labels.
- **Fix:** Correct `lora_target` to the architecture's actual projection names; correct the `columns`/`template` so a real response region is supervised.
- **Prevention:** Assert `trainable_params > 0` and `supervised_fraction > 0` in the gate.
- **Trap:** Raising `learning_rate` by 100×. If no parameter is receiving gradient, LR is irrelevant.

**Q101. "CUDA OOM at step 1, in a config that ran fine yesterday."**

- **Most likely cause:** The *shape* changed, not the model: `cutoff_len` raised, `per_device_train_batch_size` raised, `gradient_accumulation_steps` lowered (which changes nothing about activation memory but tempts people to raise the batch), `gradient_checkpointing` off, `packing: true` (which makes every batch exactly `cutoff_len` — you lose the short-sample savings), or the DeepSpeed config dropped so the weights are no longer sharded.
- **Cheapest test:** `nvidia-smi` at start-up plus the config diff against yesterday's YAML — the config is in git, so `git diff` answers it in seconds. Look for a `deepspeed:` line that vanished.
- **Fix / prevention:** Put the resolved config in the run manifest and diff manifests, not memories. Predict peak from the table plus 15 % and assert it in the gate.
- **Trap:** Reducing `cutoff_len` first when the actual change was `packing: true`. Fixing the wrong knob costs a day of confusion.

**Q102. "OOM at step 1 is gone, but the job OOMs at step ~1400, near the end."**

- **Most likely cause:** Fragmentation plus a long-tail sample. The allocator has reserved and released different shapes for 1400 steps; a batch whose sequences are much longer than typical needs a contiguous block that no longer exists. Q101's "changed shape" and this are different incidents — same error string, different causes.
- **Cheapest test:** Re-run from the last checkpoint with a smaller `per_device_train_batch_size` and `expandable_segments:True` (`PYTORCH_CUDA_ALLOC_CONF`). If it now completes, it is fragmentation, not a leak.
- **Fix:** Reduce batch size, enable `expandable_segments`, add `save_strategy: steps` so a late OOM never costs the run.
- **Prevention:** Length histogram in the pre-flight so the tail is known; `save_strategy: steps` as policy for any run over an hour.
- **Trap:** Diagnosing a "memory leak" in the trainer. It is nearly always allocator behaviour at the tail of the length distribution, and a leak would show as a monotonically rising peak, not one spike.

**Q103. "OOM at the *evaluation* step, after training finished."**

- **Most likely cause:** `per_device_eval_batch_size` defaults independently of the train batch size, and generation-based eval (`predict_with_generate: true`) allocates the KV cache for `max_new_tokens` on top of the model. Training finished and then eval tried to hold a full generation batch.
- **Cheapest test:** Check whether the OOM message names an eval step; if so, re-run only the eval with `per_device_eval_batch_size: 1`.
- **Fix:** `per_device_eval_batch_size: 1–2`, `max_new_tokens` set to what you actually need, and for large models turn generation eval off entirely (`do_train` only) and evaluate in a separate run.
- **Prevention:** Separate train and eval configs — evaluation is a cheap, repeatable, independently-schedulable job and should not be coupled to a 12-hour training process.
- **Trap:** "The eval set is small, it should be cheap." It is small in *rows* and expensive in *KV cache*.

**Q104. "Eval printed exactly one line and then the run ended."**

- **Most likely cause:** `val_size: 0` — the default. No validation split was created, so evaluation had nothing to iterate over and the trainer emitted a single summary line. This is the single most common "I thought I was evaluating" incident.
- **Cheapest test:** Look for the dataset sizes in the start-up log — a validation set of 0 examples is printed there.
- **Fix:** `val_size: 0.05` (or a fixed count) explicitly in every config; note the order of operations — `max_samples` applies *before* the split, so a `max_samples: 1000` debug run with `val_size: 0.05` validates on 50 examples.
- **Prevention:** The gate asserts `val_size > 0`; CI fails the config otherwise.
- **Trap:** Reading the single line as "evaluation ran and the score was X." It is a summary of an empty loop.

**Q105. "The model never stops generating."**

- **Most likely cause:** `stop_words` are missing or wrong for the training template. The model learned to terminate with a token that the inference config does not treat as a stop condition — a template-shaped output with no matching stop string.
- **Cheapest test:** Generate a fixed 200 tokens and read the raw output. If the answer is correct and then continues with a plausible next user turn, it is a stopping problem, not a training problem — and the continuation text tells you which format the model learned, which identifies the template.
- **Fix:** Set `stop_words` to the template's EOS/end-of-turn strings in both the Chat tab and the served config (Q57: the deployment record must include the template and stop strings, not just the weights).
- **Prevention:** The base-model probe in the gate; `stop_words` recorded in the adapter manifest alongside `(base revision, template)`.
- **Trap:** Retraining with more data. The model is fine; the sampler is wrong.

**Q106. "The export succeeded and the merged model produces word salad."**

- **Most likely cause:** The merge was performed on a **quantised base** — a `quantization_bit` left in the export YAML — so `quantize(W) + Δ` was computed where `W + Δ` was meant. Second: the merge ran on the wrong adapter path or the wrong base revision.
- **Cheapest test:** Compare adapter vs merged on ten fixed prompts. If the adapter is coherent and the merge is not, the export is the fault; if both are bad, the training is (return to Q97).
- **Fix:** Remove `quantization_bit` from the export config, set `export_device: cpu`, `export_legacy_format: false`, and export in the base's dtype. Re-merge.
- **Prevention:** Keep export configs in a separate directory from training configs so a quantised training config cannot be edited into an export config; add the "no `quantization_bit` in an export" check to the gate.
- **Trap:** Assuming the export is a copy operation. It is an arithmetic merge with dtype rules.

**Q107. "Four-GPU run: throughput is *lower* than one GPU."**

- **Most likely cause:** Wrong sharding strategy for the job — `ds_z3_config.json` on a LoRA run, which all-gathers parameters every forward and backward for no memory benefit (Q69) — or plain DDP with `quantization_bit: 4`, where every rank holds a full quantised copy (Q79). Both are "correct configs" that are wrong strategies.
- **Cheapest test:** Compare step time across the two configs at fixed batch; then check whether the base is sharded at all in the logs. Also confirm the GPUs are actually being used (`nvidia-smi` during the run) rather than one rank doing the work.
- **Fix:** `ds_z2_config.json` for LoRA; a single GPU for QLoRA with a model that fits; ZeRO-3 only for full FT where the weights genuinely exceed one device.
- **Prevention:** A one-line rule in the team's runbook — *ZeRO-2 for LoRA, ZeRO-3 for full fine-tuning, never DeepSpeed on one GPU* — plus a nightly throughput check on a tiny model so regressions are caught in CI rather than in production.
- **Trap:** Blaming PCIe. Interconnect matters, but a strategy that communicates for no benefit is slow even on NVLink.

**Q108. "The job hangs at start-up. No logs, no error, nothing on `nvidia-smi`."**

- **Most likely cause:** An interactive prompt or a network wait before the first log line. The classic is a telemetry/experiment-tracker login (`report_to` resolving to `wandb` with no credentials), which can block on a prompt a batch job cannot answer. Second: a gated model requiring `huggingface-cli login` (a 401 that presents as a hang on some stacks), or `HF_ENDPOINT` unreachable.
- **Cheapest test:** `strace`/`py-spy dump` the live process to see what it is blocked on; run with `HF_HUB_OFFLINE=1` and `report_to: none` and see whether it proceeds.
- **Fix:** `report_to: none` in every config; `HF_TOKEN` in the environment (never in the image); pre-fetch weights for air-gapped nodes (Q91).
- **Prevention:** The gate runs the config with no network and fails if it blocks; the manifest records the resolved `report_to`.
- **Trap:** Waiting. A hang with zero output before any import is a prompt or a socket, not a slow disk.

**Q109. "`Bus error (core dumped)` with `dataloader_num_workers: 8`."**

- **Most likely cause:** Shared-memory exhaustion. Each worker uses `/dev/shm` for the batch transfer, and in containers `/dev/shm` is often the 64 MB default — the dataloader needs far more than that to pass a padded batch.
- **Cheapest test:** `df -h /dev/shm` while the job runs, or set `dataloader_num_workers: 0` and watch the error vanish.
- **Fix:** Raise the container's `--shm-size`, or lower `dataloader_num_workers` to 2–4, or both.
- **Prevention:** Size `/dev/shm` in the image/compose file as a documented requirement, and prefer 2–4 workers — more workers rarely help when the GPU is the bottleneck and the data is tokenised.
- **Trap:** Treating `Bus error` as a hardware fault and replacing the node.

**Q110. "A 401 on a model you have downloaded before."**

- **Most likely cause:** The model is gated and the credential is absent or expired in this environment — a fresh container with no `HF_TOKEN`, an expired token, or an unaccepted licence on the Hub for this account. The "I downloaded it before" detail is the trap: the local cache is usually irrelevant because the config points at a Hub path.
- **Cheapest test:** `huggingface-cli whoami` and a direct `hf_hub_download` of one file outside the framework, so the error comes from the client rather than from the trainer's traceback.
- **Fix:** Set `HF_TOKEN` (env or secret manager), accept the licence, and for reproducibility move to a pinned local path with `revision: <sha>` so production never depends on a Hub credential at all.
- **Prevention:** Pre-fetch into the artifact store at build time; treat runtime Hub access as an anti-pattern in a training cluster.
- **Trap:** Bypassing with `trust_remote_code: true`, which is unrelated and increases the attack surface.

**Q111. "I changed the dataset and nothing changed in the metrics."**

- **Most likely cause:** The tokenised cache. LLaMA-Factory caches preprocessed/tokenised datasets; the cache key may not capture a change in the underlying file (or the same `tokenized_path` was reused), so the old tokenised data is served. Second: the registry key still points at the old `file_name` — you edited a file nothing references.
- **Cheapest test:** Set `overwrite_cache: true` on one run and compare. If the metrics move, it was the cache. Separately, print the resolved `file_name` and row count from the start-up log against what you believe you edited.
- **Fix:** `overwrite_cache: true` for the run that must reflect the change; delete or namespace `tokenized_path` per data version.
- **Prevention:** Version the dataset registry alongside configs and key `tokenized_path` by the data hash, so a changed file cannot reuse a stale cache.
- **Trap:** Concluding that your data change did not help. You have not tested the change yet.

**Q112. "The adapter loads but the trainer warns about unexpected keys."**

- **Most likely cause:** The adapter was saved against a different module structure than the model now being loaded — a different `lora_target`, a different architecture revision, or a base model whose linear layers are named differently (fused vs unfused QKV). The mismatched keys are dropped, so the model loads *without* part of the adapter and generates plausible but wrong output.
- **Cheapest test:** Compare `adapter_config.json`'s `target_modules` and `r` against the training YAML and the base's module names; count how many keys were skipped in the warning.
- **Fix:** Load the adapter against the exact base revision it was trained on; if the base has genuinely changed, retrain rather than force-loading.
- **Prevention:** The adapter record (Q96) stores target modules and base revision; the serving layer refuses a mismatch rather than warning about it.
- **Trap:** Suppressing the warning with `ignore_mismatched_sizes`-style flags. It converts a loud partial failure into a silent one, which is the exact failure class this module is about.

**Q113. "Rerunning the identical config produced a different final loss."**

- **Answer:** Most of the difference is expected. Sources, in rough size order: **nondeterministic GPU kernels** (attention and reduction order; `torch.use_deterministic_algorithms(True)` costs throughput and does not cover every op), **data order** under multi-worker loading and sharding across ranks, **`packing`** interactions with batch composition, and **a different resolved template** if the model revision moved on the Hub (which is why `revision:` must be pinned). The right frame: final loss is a noisy scalar — differences under ~2–3 % on a 7B LoRA run are not evidence of anything. What must be reproducible is the *artifact record*, not the sixth decimal place.
- **Cheapest test:** Run the same config twice with `seed` fixed and data order single-worker; if the spread stays, it is kernels and reduction order rather than your data.
- **Fix / prevention:** Pin `revision`, pin the framework version, record the manifest, and treat eval scores (with their intervals) as the comparison unit — not the final training loss.
- **Trap:** Chasing bitwise reproducibility on GPU. It is a research project, not a config flag, and it is not what "reproducible" means for a fine-tune.

---

## Rapid Fire — True / False / One-Liner

*Read the statement, answer in under ten seconds, then say why. "True" with no reason scores zero.*

| # | Statement | Verdict + one-line why |
|---|---|---|
| 1 | LLaMA-Factory implements its own training loop from scratch. | **False.** It is a config-driven orchestrator over `transformers` + `peft` + `bitsandbytes` + `trl`. |
| 2 | A wrong `template:` raises an error. | **False.** It trains and converges; the model is just worse. Only a template *missing from the registry* raises. |
| 3 | `template: default` uses the tokenizer's own chat template. | **False.** It is a registered plaintext template (`Human:`/`Assistant:`); the tokenizer's template is used only when `template` is omitted. |
| 4 | The YAML `dataset:` field takes a file path. | **False.** It takes a **key** resolved through `dataset_info.json`. |
| 5 | The registry field is `format`. | **False.** It is `formatting`, and only `alpaca`, `sharegpt` or `openai` are accepted. |
| 6 | `finetuning_type: qlora` is a valid value. | **False.** QLoRA is `finetuning_type: lora` **plus** `quantization_bit: 4`. |
| 7 | `stage: dpo` requires `ranking: true` in the registry entry. | **True.** Without it the preference rows are parsed as ordinary SFT data. |
| 8 | ShareGPT's default role tags are `user` and `assistant`. | **False.** They are `human` and `gpt`; a mismatch logs `Invalid role tag` and can emit empty samples. |
| 9 | The framework validates that your `template` matches your model. | **False.** The two are resolved in separate registries with no declared relation. |
| 10 | `quantization_bit: 4` makes the adapter 4-bit. | **False.** The **base** is quantised; the LoRA adapter trains in bf16/fp16. |
| 11 | `export` can merge into a quantised base. | **False.** Merge must load the base unquantised; a `quantization_bit` in the export config produces garbage. |
| 12 | A 4-bit training run exports a 4-bit model. | **False.** Export dequantises — you get a bf16/fp16 merged artifact and must re-quantise separately for serving. |
| 13 | The merge formula is `W_base + (α/r)·(B@A)`. | **True.** Scaling is `α/r` — which is why `use_rslora` (α/√r) is the fix at large ranks. |
| 14 | `val_size` defaults to a sensible 0.05. | **False.** It defaults to **0**, so evaluation runs on nothing and prints one line. |
| 15 | `stage: pt` on an instruction dataset is fine. | **False.** It consumes the rows as raw text and destroys the instruction structure. |
| 16 | `packing: true` speeds up SFT and is safe. | **False.** Without `neat_packing` it creates cross-sample attention in chat data. |
| 17 | `neat_packing: true` prevents cross-sample attention. | **True.** Each packed sample keeps its own attention window. |
| 18 | `packing` is auto-enabled for `stage: sft`. | **False.** It is auto-enabled for `stage: pt`; off for SFT. |
| 19 | `cutoff_len` truncates and is silent about it. | **True.** Long samples are cut; the loss stays healthy while the target is lost. |
| 20 | `ignore_pad_token_for_loss: false` trains on the prompt. | **False.** It governs *padding* positions only; `train_on_prompt` is the prompt switch. |
| 21 | ZeRO-3 is the best choice for multi-GPU LoRA. | **False.** It shards weights that already fit and all-gathers them every step; ZeRO-2 is right for LoRA. |
| 22 | LoRA runs always beat `freeze` on VRAM. | **True** in practice at equal trainable count — `freeze` holds full-precision grads and fp32 moments over a larger slice. |
| 23 | `α` scales the LoRA update. | **True.** With the default `α = 2r`, the effective step shrinks as `r` grows. |
| 24 | `lora_target: all` trains every weight in the model. | **False.** It attaches low-rank adapters to every *linear*; the base stays frozen. |
| 25 | Higher `lora_rank` is always better. | **False.** Past the point where the task needs capacity it buys overfitting; rank 32 beat 16 on the CS-15 legal case, but rank 64 on 5 k rows usually loses. |
| 26 | `adapter_name_or_path` lets DPO continue from an SFT adapter. | **True.** It is how SFT → DPO chains without restarting from the base. |
| 27 | DPO needs a reward model in memory. | **False.** DPO needs policy + reference; the reward model is what `stage: rm` produces and what `stage: ppo` consumes. |
| 28 | DPO learning rate should match SFT's. | **False.** It is ~20× lower — `5e-6` against `1e-4`. |
| 29 | `stage: rm` and `stage: ppo` need more models in memory than `sft`. | **True.** PPO holds policy + reference + reward + value. |
| 30 | `report_to: none` is a harmless line to omit. | **False.** The default can block a batch job on an experiment-tracker login. |
| 31 | The WebUI writes to a deterministic `output_dir`. | **False.** It uses a timestamped directory, which is why notebook paths differ between examples. |
| 32 | The WebUI is only useful for beginners. | **False.** Its Chat tab is the fastest interactive template check there is. |
| 33 | `bf16` is universally better than `fp16`. | **False.** bf16 needs Ampere or newer; on a T4 you use fp16, and a `nan` at step ~200 is the signature of getting it wrong. |
| 34 | `max_samples` applies after the validation split. | **False.** It applies **before**, so a debug run's `val_size` is a fraction of the slice. |
| 35 | `tokenized_path` saves preprocessing time on the next run. | **True** — and it will also serve stale data if the source changed and the path is reused. |
| 36 | `overwrite_cache` and `overwrite_output_dir` control the same thing. | **False.** One invalidates the tokenised cache, the other the checkpoint directory. |
| 37 | `use_unsloth: true` makes LLaMA-Factory 2× faster than itself. | **False.** Unsloth's honest margin against a *tuned* FA2 baseline is ~1.2–1.4× time and 15–30 % VRAM. |
| 38 | LLaMA-Factory and Axolotl are near-identical tools with different names. | **False.** Overlapping YAML-and-registry territory, but Axolotl's strength is multi-node FSDP2/ZeRO and its 20+ dataset `type:` strategies. |
| 39 | TRL is the right answer when you need a custom loss. | **True.** Custom loss, custom collator or a custom loop is precisely when the YAML layer is the wrong tool. |
| 40 | A smooth loss curve is evidence the fine-tune worked. | **False.** It is evidence the run converged — on whatever tokenised strings you actually gave it. |

---

## Coding / Whiteboard Tasks

### Task 1 — Register a dataset without touching Python (5 minutes)

**Prompt.** You have `data/medical_qa.jsonl`, one JSON object per line: `{"question": "...", "answer": "...", "specialty": "cardiology"}`. Make it trainable as `medical_qa` and explain every field you write.

**Solution sketch.**

```jsonc
// dataset_info.json
{
  "medical_qa": {
    "file_name": "medical_qa.jsonl",     // relative to <dataset_dir>; NOT "path"
    "formatting": "alpaca",              // NOT "format"
    "columns": {
      "prompt":   "question",            // source field -> framework role
      "response": "answer"
    }
  }
}
```

```yaml
# train.yaml
dataset_dir: data
dataset: medical_qa                      # the KEY, not the filename
template: qwen3                          # family-correct name, never "default"
stage: sft
finetuning_type: lora
quantization_bit: 4
cutoff_len: 1024
val_size: 0.05                           # the default is 0
report_to: none
```

**Grading notes.** Full marks require: (a) `formatting`, not `format`; (b) `file_name`, not `path`, and *relative* to `dataset_dir`; (c) the `columns` map naming **source** fields on the right and framework roles on the left; (d) `dataset:` naming the **key**; (e) an explicit non-`default` template; (f) an explicit `val_size`. `max_samples` and `num_workers` are bonuses. A candidate who writes `"path": "data/medical_qa.jsonl"` has memorised the shape and not the semantics — probe with "what happens when that key is wrong?", where the correct answer is `ValueError: Undefined dataset medical_qa in dataset_info.json.` at start-up, i.e. loud.

### Task 2 — Preference data for DPO (5 minutes)

**Prompt.** You have `data/prefs.jsonl` with `{"instruction", "input", "chosen", "rejected"}`. Write the registry entry and the DPO config that continues from an existing SFT adapter at `outputs/sft_v1`.

**Solution sketch.**

```jsonc
{
  "medical_prefs": {
    "file_name": "prefs.jsonl",
    "formatting": "alpaca",
    "ranking": true,                     // <-- the line that makes it preference data
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "chosen",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

```yaml
stage: dpo
model_name_or_path: Qwen/Qwen3-4B
adapter_name_or_path: outputs/sft_v1     # chain, do not restart from base
dataset: medical_prefs
template: qwen3
finetuning_type: lora
pref_beta: 0.1
pref_loss: sigmoid                       # orpo | simpo | hinge | ipo are the alternatives
learning_rate: 5.0e-6                    # ~20x below a typical SFT LR
num_train_epochs: 1
report_to: none
```

**Grading notes.** The discriminating line is `"ranking": true` — omit it and the rows parse as SFT data, the run looks fine and the model is not preference-tuned at all. Second discriminator: `adapter_name_or_path` present (chaining) versus absent (restarting from base, which produces fluent nonsense). Third: the learning rate order of magnitude. CH-15's own `my_pairs` snippet omits `ranking` — a candidate who reproduces that snippet verbatim should be asked what `ranking` does.

### Task 3 — Explain a token/label dump (10 minutes)

**Prompt.** Given this output from dumping one rendered sample, say what is healthy and what is wrong:

```
template: qwen3
cutoff_len: 1024
tokens:  <|im_start|> user\nHow do I reset my password?<|im_end|>\n<|im_start|> assistant\nGo to Settings > Security.<|im_end|>\n
labels:  -100 -100 -100 ... -100 Go to Settings > Security . <|im_end|> -100 ...
supervised_tokens: 7 / 24   (29.2%)
trainable_params: 9805824 / 2510000000   (0.39%)
```

**Solution sketch / expected reading.**

1. **Format is correct** — the `<|im_start|>` markers match a Qwen-family template, so this is not the template-mismatch failure.
2. **The mask boundary is correct** — `-100` until the assistant header ends, then real token ids for the answer and the closing `<|im_end|>`. The EOS being supervised is intended.
3. **29.2 % is on the high side** for a short-answer chat set; the 5–15 % band assumes long prompts. Not a bug, but worth a second sample — a *median* near 100 % across many samples is the leakage signature.
4. **0.39 % trainable** matches rank 8 with `lora_target: all` on a ~2.5 B model (9.81 M params) — the arithmetic checks out.
5. What the dump **cannot** tell you: whether the template matches this base model. That is answered by chatting with the base under the same template.

**Grading notes.** Full marks require all five. The failure mode to watch for: a candidate who reads "29.2 %" and says "too high, use `train_on_prompt: false`" — it is already `false`; the number is high because the prompt is short, not because masking is broken. Second failure: reading `-100` as "a small negative loss weight." It is `IGNORE_INDEX`, the mask sentinel.

### Task 4 — Compute the LoRA parameter count and check it against the log (5 minutes)

**Prompt.** A 7B model has 32 layers, `hidden_size 4096`, `intermediate_size 11008`, and the config targets `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` at rank 16. Estimate the trainable parameters and say whether `lora_target: all` at rank 8 on a 2.5 B model giving 9.81 M is consistent.

**Solution sketch.**

```
q,k,v,o : 4 x (4096 x 4096) = 67.1 M  weights per layer
gate,up : 2 x (4096 x 11008) = 90.2 M
down    :      (11008 x 4096) = 45.1 M
total   : 202.4 M weights per layer

LoRA params per layer = r x (in + out) summed over targeted matrices
  q,k,v,o : 4 x 16 x (4096 + 4096) =  524,288
  gate,up : 2 x 16 x (4096 + 11008) = 483,328
  down    :     16 x (11008 + 4096) = 241,664
  per layer                          = 1,249,280
32 layers                            = 39,976,960  ≈ 40.0 M   (rank 16, all targets)

rank 8 => halve  => ~20.0 M for a 7B
```

For the 2.5 B case: `all` at rank 8 giving 9.81 M implies ~19.6 M at rank 16, i.e. a model with roughly a quarter of the 7B's per-layer weight count — consistent with a 2.5 B model at rank 8. The observed log (9,805,824) and the arithmetic agree to within rounding across layer counts, which is the point: **the number in the log is checkable, and checking it catches a wrong `lora_target`, a wrong rank, and a silently unfrozen base all at once.**

**Grading notes.** The formula `Σ r × (in + out)` is the required content. Candidates who say "roughly a few million" fail. Bonus: noting that rank 16 → 40 M is 0.5 % of a 7B and that the ratio (trainable/total) is what you compare across configs, not the absolute count.

### Task 5 — Design the start-up assertion block (10 minutes)

**Prompt.** Write the checks you would run before spending GPU-hours, and name the silent failure each one catches.

**Solution sketch.**

```python
# preflight.py — run before llamafactory-cli train; each check names its failure mode
CHECKS = {
  "keys_valid":        "typo'd or renamed YAML keys (stage/finetuning_type/formatting)",
  "dataset_resolves":  "dataset key absent from dataset_info.json -> Undefined dataset",
  "file_exists":       "file_name typo or wrong dataset_dir",
  "template_in_family":"template/model mismatch -> the silent quality bug (Q97)",
  "supervised_tokens": "empty samples from Invalid role tag; all-(-100) labels; prompt leakage",
  "stop_token":        "stop_words missing -> model never stops (Q105)",
  "val_size_gt_0":     "val_size: 0 default -> evaluation ran on nothing (Q104)",
  "no_quant_on_export":"quantization_bit left in an export config -> word salad merge (Q106)",
  "vram_budget":       "peak prediction from the memory table + 15% overhead vs device (Q101)",
  "trainable_gt_0":    "adapter attached to nothing -> flat loss (Q100)",
}
```

**Grading notes.** Score on the mapping, not the syntax: ten checks each tied to a *named* failure mode is a senior answer; five generic assertions ("config parses") is a junior one. The two checks candidates most often miss are `template_in_family` (because the framework does not do it, so nothing reminds them) and `no_quant_on_export` (because training and export configs are often edited copies of each other). Reward anyone who proposes running the block in CI against a 100-row sample and asserting the *loss starts near ln(vocab)* rather than asserting a target loss value.

---

## Cheat Sheet of Numbers To Memorize

| Number | Value | Why it appears |
|---|---|---|
| 7B full fine-tune VRAM | **91.6 GB** | From `code/common/memory.py --table`. Does not fit one 80 GB card. |
| 7B LoRA (bf16 base) | **14.7 GB** | Fits a 24 GB card with room for batch. |
| 7B QLoRA (`quantization_bit: 4`) | **4.9 GB** | Fits a free 16 GB T4. The whole reason QLoRA exists. |
| 13B / 70B, same three columns | **170.0 / 27.1 / 9.0** and **913.8 / 144.5 / 46.8** GB | The scaling rules of thumb: ~6× full vs LoRA, ~3× LoRA vs QLoRA. |
| Overhead multiplier | **+10–20 %** on the table | Allocator, cuBLAS workspaces, fragmentation, bnb dequant buffers. |
| Demo run trainable params | **9,805,824 ≈ 9.81 M (0.39 %)** | Rank 8, `lora_target: all`, ~2.5 B model. The number to check arithmetic against. |
| Rank 16, all-targets, 7B | **≈ 40 M** | `Σ r × (in + out)` over q/k/v/o/gate/up/down × 32 layers. |
| Supervised-token fraction, healthy chat SFT | **5 – 15 %** | Higher is fine for short-answer data; **0 % is always a bug**; ~100 % on chat data suggests leakage. |
| Healthy first-step loss | **≈ ln(vocab) ≈ 9–10** | A fresh model is uniform over the vocabulary. A first-step loss under ~1.0 is a data bug. |
| fp16 overflow threshold | **~65,504** | Attention logits above it → `inf` → `softmax` → `nan`. The arithmetic reason bf16 exists. |
| Default `val_size` | **0** | Evaluation runs on nothing and prints one line. Always override. |
| SFT learning rate | **1e-4** | Typical LoRA SFT. |
| DPO learning rate | **5e-6** | ~20× lower; `pref_beta: 0.1`. Running DPO at SFT's LR is a common self-inflicted failure. |
| QLoRA's two lines | `finetuning_type: lora` + `quantization_bit: 4` | There is no `qlora` value, ever. |
| `use_rslora` threshold | **rank ≥ 32** | Scaling switches from `α/r` to `α/√r`; below that the default scaling is fine. |
| `neftune_noise_alpha` | **5** | The NEFTune recommendation for instruction tuning. |
| Template registry | `src/llamafactory/data/template.py` | A flat `TEMPLATES` dict; the source of truth for valid template names. |
| Dataset registry | `<dataset_dir>/dataset_info.json` | Key → file. The YAML refers to the **key**. |
| Missing-template error | `ValueError: Template X does not exist.` | The *only* loud template failure. A mismatch is silent. |
| Missing-dataset error | `ValueError: Undefined dataset X in dataset_info.json.` | Loud, at start-up. Dataset problems are the cheap ones. |
| Silent role-tag warning | `Invalid role tag in […]. Skipping this abnormal example.` | Grep for this string; the sample is emitted with `prompt, response = [], []`. |
| IGNORE_INDEX | **-100** | The loss-mask sentinel everywhere in the framework. |
| Start-up overhead | **20–40 s** | Imports + registry + model load. Matters only for very short runs. |
| ZeRO rule | **ZeRO-2 for LoRA, ZeRO-3 for full FT** | ZeRO-3 shards weights that LoRA runs already fit, and pays all-gather for it. |
| Unsloth's honest margin | **~1.2–1.4× time, 15–30 % VRAM** | Against a *tuned* FA2 baseline — not the 2×/60 % headline, which is against untuned defaults. |
| Multi-adapter storage | **~20 MB per adapter** (r=8, 2B) vs ~5 GB per merged model | 40 adapters = 800 MB; 40 merges = 200 GB. The argument for LoRA serving. |

---

## Answers To The Self-Check Questions From CS-15

**1. A colleague sets `finetuning_type: qlora`. What will LLaMA-Factory do, and what is the correct config?**
It will fail argument parsing: `finetuning_type` is a `Literal["lora", "freeze", "full"]` and `qlora` is not a member. The correct config is `finetuning_type: lora` **plus** `quantization_bit: 4` — nothing else changes. QLoRA is not a training method in the config's vocabulary; it is the *composition* of a 4-bit frozen base with a LoRA adapter, which is exactly what the repo's `train_gemma_qlora.yaml` encodes (a file named "qlora" containing `finetuning_type: lora`).

**2. Your run completes with a healthy-looking loss curve, but the model is worse than the base at inference. List the four most likely causes, in the order you would check them.**
(a) **Template mismatch** — render one sample and compare its tokens against the model's own `apply_chat_template` output; a wrong format string produces a perfectly learnable sequence the base never saw. (b) **Role-tag or `formatting` mismatch** — `grep "Invalid role tag"` in the log, then count supervised vs `-100` tokens in one sample. (c) **Truncated responses** from too small a `cutoff_len` — histogram response token lengths against the limit. (d) **Adapter not loaded, or wrong base/precision at inference** — print `base_model_name_or_path` from `adapter_config.json` and compare it to what you actually loaded. All four are invisible in the loss curve, which is precisely why the loss curve is not the check.

**3. Registry entry for `data/support/chat.jsonl` with `{"messages":[{"role":"user",...},{"role":"assistant",...}]}` rows.**

```jsonc
{
  "support_chat": {
    "file_name": "support/chat.jsonl",     // relative to dataset_dir (default "data")
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role", "content_tag": "content",
      "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
    }
  }
}
```

The `tags` block is mandatory here. The sharegpt defaults expect `from`/`value` with roles `human`/`gpt`; without the remap every row fails the positional role check, logs `Invalid role tag`, and is emitted with `prompt, response = [], []` — a dataset of the correct length containing nothing.

**4. `template: default`, `template: empty`, and omitting `template`.**
`default` is a **registered plaintext template**: `Human: …\nAssistant: …\n` with a `System:` prefix, `replace_jinja_template=True`, `efficient_eos=False`. `empty` is bare `{{content}}` concatenation with no role markers, intended for pretraining. **Omitting** `template` is a third path entirely: the framework looks for `tokenizer.chat_template`, logs a warning if it is a string, and derives a template via `parse_template(tokenizer)`; failing that it falls back to `empty`. Two of the three behave nothing like their names suggest, which is why an explicit family-correct name is the only safe choice.

**5. Why is `train_gemma_qlora.yaml` named "qlora" when it contains `finetuning_type: lora`?**
Because the filename describes the *recipe*, not a config key. QLoRA is quantising the frozen base to 4-bit (`quantization_bit: 4`) while training LoRA adapters (`finetuning_type: lora`). Since there is no `qlora` value to put in the file, a config that wants to be honest about being QLoRA has to say so in its name.

**6. `max_samples: 1000`, `val_size: 0.1`, `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, `eval_steps: 200`, `num_train_epochs: 1`, `logging_steps: 10` — optimiser steps, eval events, log lines?**
Train examples `1000 × (1 − 0.1) = 900`. Steps per epoch `900 / (1 × 4) = 225`. With one epoch → **225 optimiser steps**. `eval_steps: 200` fires at step 200 plus the end-of-training evaluation → **1–2 eval events** (one meaningful). `logging_steps: 10` → **≈22 log lines**. The actionable conclusion is that `eval_steps: 200` is too coarse for a 225-step run — roughly `steps / 5 ≈ 45` is the value that gives you a curve rather than two points. Note also the order of operations: `max_samples` applied *before* `val_size`, so both numbers are downstream of the 1000.

**7. You must serve 40 customer-specific adapters over one base model. What do you export, and why not the other option?**
Export the **adapters** and serve them with a multi-adapter server (vLLM `--enable-lora`). Forty adapters at ~20 MB (r=8 on a 2B base) is 800 MB of artifacts sharing one VRAM-resident base. The other option — forty merged models — is 40 × ~5 GB = **200 GB** of storage, forty separate VRAM-resident copies of the same base, and no way to patch a base-model vulnerability without re-merging all forty. Keep one merged model for a pinned single-tenant deployment; use adapters for multi-tenancy.

**8. Why can merging a LoRA adapter into a 4-bit base produce garbage, and how does the framework phrase the rule?**
A 4-bit base stores an approximation: per-block quantised weights plus scales. `quantize(W) + Δ` is not `quantize(W + Δ)` — the LoRA delta was learned against the quantised forward pass, so adding it to a dequantised approximation compounds two errors and can overflow the quantisation range. The framework's own merge example states the rule in capitals: *"DO NOT use a quantized model or `quantization_bit` when merging LoRA adapters."* Practically: delete `quantization_bit` from the merge YAML, point `model_name_or_path` at the original full-precision base, and set `export_device: cpu`. The consequence worth saying out loud is that a 4-bit training run produces a **bf16 merged artifact**, which must be re-quantised separately for serving.

**9. "I set `ignore_pad_token_for_loss: false` so the model learns the prompts too." What is wrong with that sentence?**
Two things. First, `ignore_pad_token_for_loss` governs **padding** positions, not prompt tokens — setting it `false` teaches the model to generate pad tokens, which is a bug rather than a feature, and it is `true` by default for that reason. Second, prompt masking is not controlled by that flag at all: it is a property of the **template**, whose prompt/response split decides which label positions become `-100`, with `train_on_prompt` and `mask_history` as the two explicit overrides. To actually train on the prompts, set `train_on_prompt: true` — and be ready for the framework to raise `ValueError` if the template has `efficient_eos`, because that combination is unsupported.

**10. A working config moved from a T4 to an A100 goes `nan`. One-line fix and the mechanism.**
Change `fp16: true` to `bf16: true`. fp16 has a 5-bit exponent, so attention logits above ~65,504 overflow to `inf`, and `softmax(inf)` produces `nan` that propagates through the entire graph — usually appearing at a random-looking step when an unusually long or unusual example arrives. bf16 has fp32's exponent range and cannot overflow this way, which is why the instructor's rule is *"BF16 is for the higher-end GPU… fp16 is for the lower-end GPU"* [48:09]. The direction of the fix is what makes this a good question: the bug appeared when moving *up* in hardware, because the config kept the low-end dtype on a high-end card. If bf16 is genuinely unavailable, the fallback is a 4× lower learning rate plus `max_grad_norm: 1.0`.

---

## Cross-References

| Relationship | Module | Why |
|---|---|---|
| Pairs with | **CS-15** (LLaMA-Factory case study) | The source this bank is built on; every answer here has a section there. |
| Pairs with | **CH-15** (LLaMA-Factory cheat sheet) | The copy-paste companion; use it for the exact YAML shapes quoted here. |
| Builds on | **IQ-03** / **CS-03** (framework landscape) | Where LLaMA-Factory sits between Unsloth, Axolotl, TRL and torchtune. |
| Builds on | **CS-06** (Hugging Face) | `Trainer`, tokenizers, and the Hub this framework orchestrates. |
| Builds on | **CS-13** (instruction fine-tuning) | The alpaca/sharegpt shapes and prompt masking assumed throughout L1–L2. |
| Builds on | **CS-13 §6.8** / **CS-11 §4.11** (LoRA & QLoRA) | `lora_rank`, `lora_alpha`, `lora_target`, and why QLoRA = 4-bit + LoRA. (The `CS-23` LoRA deep dive cited elsewhere in this repo was never written) |
| Contrasts with | **CS-16** (Unsloth) | Single-GPU speed — the `use_unsloth: true` bridge and the honest 1.2–1.4× decomposition. |
| Contrasts with | **CS-17** (Axolotl) | The other YAML framework; multi-node FSDP2/ZeRO and `preprocess --debug`. |
| Uses | **CS-10** / **CS-11** / **CH-10** / **CH-11** (quantization) | What `quantization_bit` does, and where to re-quantise a merged artifact for serving. |
| Uses | **CS-14 §4.6.3–4.6.9** (DPO / ORPO) | `stage: dpo`, `pref_beta`, `pref_loss` variants. The DPO and ORPO deep dives are sections *of* CS-14; the planned `CS-25`/`CS-27` modules were never written |
| Uses | **CS-18** / **CS-19** (hosted fine-tuning) | The buy-vs-build comparison when a YAML toolchain is more than you need. |
| Used by | **IQ-14** (alignment questions) | Level 3–5 here assumes the preference-optimisation vocabulary there. |
| Quick reference | **CH-15** §4 (snippets), §7 (VRAM calculator), §11 (error messages) | The three sections to have open while running this bank as an interview. |



