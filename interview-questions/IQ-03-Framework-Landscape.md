# IQ-03 — Interview Questions: The Fine-Tuning Framework Landscape

| Field | Value |
|---|---|
| **Module** | Tooling / Ecosystem Selection |
| **Pairs with** | **CS-03** (case study), **CH-03** (cheat sheet) |
| **Total questions** | **102** (30 L1 + 30 L2 + 22 L3 + 10 L4 + 10 L5) |
| **Levels covered** | L1 screening → L5 incident response |
| **Source material** | CS-03 §0–§20, Appendix A (instructor's verbatim claims), Appendix B |
| **Also contains** | Rapid-fire true/false (40), 5 coding/whiteboard tasks with grading notes, numbers to memorize, answers to CS-03's 10 self-check questions |
| **Time to work through** | ~6 h at interview pace; 90 min for a revision pass (L1+L2+numbers+r rapid-fire only) |

---

## How To Use This File

- **L1 — Fundamentals & Vocabulary (30).** Can you place a tool on the right layer and define the terms? A candidate who fails L1 has installed frameworks without reading them.
- **L2 — Applied & Implementation (30).** Config keys, CLI invocations, and the exact flag that does the thing. This is the level that separates "I read the README" from "I ran it."
- **L3 — Analysis & Trade-offs (22).** Mechanism-level reasoning: why multi-GPU QLoRA is slower, what changes at 13B, why PEFT won the format war.
- **L4 — System Design & Scenario (10).** Full prompts: requirements → constraints → design → trade-offs → failure modes. Answer these out loud, in 8–12 minutes each.
- **L5 — Debugging & Incident Response (10).** A symptom and a clock. Say what you check first, second, third, and what you would change before the next run.

**The meta-rule that decides most of this module's interviews:** *it depends on the parallelism gate.* If a candidate answers "which framework?" without first asking "how many GPUs and how big is the model?", they have missed the entire module. Below ~13B on one GPU the framework barely affects the outcome; above it, the framework decides whether the job exists at all (CS-03 §13.4).

**Answer format used throughout:** a direct **Answer**, then **Why the interviewer asks this** (what is actually being probed), then **Trap** (the plausible-but-wrong answer that gets candidates rejected).

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. Sort these into trainer / parallelism library / serving engine / orchestrator: DeepSpeed, vLLM, FastChat, SkyPilot, ColossalAI, OpenLLM, Axolotl, FSDP2.**

- **Answer:** Trainers: Axolotl (and ColossalAI, which is also a parallelism library). Parallelism libraries: DeepSpeed, FSDP2, ColossalAI. Serving engines: vLLM, OpenLLM, FastChat. Orchestrator: SkyPilot. FastChat additionally carries an evaluation harness (MT-Bench / LLM-as-judge).
- **Why the interviewer asks this:** The video's own slide puts four non-trainers in a "Top 10 fine-tuning frameworks" list. If you cannot sort the layers, you will adopt a serving engine expecting a trainer — the first and most expensive of the four wrong decisions (CS-03 §1.1).
- **Trap:** Saying "DeepSpeed is a fine-tuning framework" (it is a memory/parallelism library that lives *inside* the others) or "FastChat is a trainer" (its `train_lora.py` exists but is dated; its load-bearing surface is serving + MT-Bench evaluation).

**Q2. Is DeepSpeed a trainer? What does it actually give you?**

- **Answer:** No. It is a distributed-training library: ZeRO-1/2/3 memory partitioning, CPU/NVMe offload (ZeRO-Offload, ZeRO-Infinity), ZeRO++, TP/PP/expert parallelism, activation checkpointing, MoE support, and AutoTP. `DeepSpeed-Chat` adds SFT+RM+PPO, but that is an application built on top.
- **Why the interviewer asks this:** The video explicitly gets this right (`[3:51]`), then immediately gets the comparison wrong — so the follow-up is whether *you* can hold the distinction.
- **Trap:** "It's similar to TRL." That is a **category error**: TRL implements post-training *losses* (SFT/DPO/GRPO/PPO), DeepSpeed implements *parallelism and memory management*. They are orthogonal and compose — TRL's `DPOTrainer` runs on DeepSpeed.

**Q3. Name the three axes that actually separate these tools.**

- **Answer:** (1) **Abstraction level** — how much code you write, from hosted API → GUI/YAML → config+Python → raw Python. (2) **Memory regime** — how parameters are held: full FT fp32/bf16/8-bit Adam, bf16 LoRA, QLoRA NF4. (3) **Parallelism strategy** — how the job spans GPUs: DDP, ZeRO-1/2/3, FSDP1/2, TP, PP, CP/SP.
- **Why the interviewer asks this:** Every one of the 23 frameworks in the master matrix is positioned by these three coordinates. Learn them and the matrix collapses into a decision tree.
- **Trap:** Listing frameworks instead of axes, or treating "number of stars / popularity" as an axis. Popularity is not a gate; parallelism is.

**Q4. What do ZeRO stages 1, 2 and 3 shard, and what does each cost in communication?**

- **Answer:** ZeRO-1 shards the **optimizer state** (no extra comm beyond the existing all-reduce). ZeRO-2 additionally shards **gradients** (reduce-scatter instead of all-reduce). ZeRO-3 additionally shards the **parameters** (all-gather per layer per forward/backward).
- **Why the interviewer asks this:** The stage number *is* the memory plan. A candidate who cannot state which state is partitioned cannot size a job.
- **Trap:** "ZeRO-3 is always best." It is the most memory-efficient and the most communication-hungry; on PCIe without NVLink it can make a job 3× slower than ZeRO-2, or slower than a single GPU.

**Q5. What is FSDP, and what changed in FSDP2?**

- **Answer:** FSDP (Fully Sharded Data Parallel) is PyTorch's native sharding strategy — parameters, gradients and optimizer state are sharded across ranks and all-gathered per layer. FSDP1 shards a *flattened* parameter per layer-group; **FSDP2** (`torch.distributed.fsdp.fully_shard`) shards *each parameter individually* via DTensor, giving clean `state_dict` semantics, straightforward QLoRA composition, CPU offload, and better `torch.compile` behaviour.
- **Why the interviewer asks this:** FSDP2 is the 2026 default sharding backend for new code, and "which version" is a real operational decision, not trivia (CS-03 §4.3.1).
- **Trap:** "FSDP and DeepSpeed ZeRO-3 are the same thing." Functionally close, operationally different: FSDP2 is in-tree (no extra install or version pin), DTensor-based, and per-parameter. DeepSpeed still owns ZeRO-Infinity NVMe offload and mixed TP+PP at very large scale.

**Q6. What is QLoRA, precisely?**

- **Answer:** LoRA adapters trained on top of a **4-bit NF4-quantized, frozen base model**, with the matmuls executed in bf16 (`bnb_4bit_compute_dtype=torch.bfloat16`). Only storage is 4-bit; the adapter itself trains in bf16. Base cost ≈ 0.55 bytes/param, which is why a 7B fine-tune fits on a 24 GB card.
- **Why the interviewer asks this:** It is the single most used memory regime in practice, and the most commonly half-understood.
- **Trap:** "QLoRA loses quality because the model is 4-bit." The adapter is bf16; the measured task-metric gap is typically **under 1 point**, and it is a *memory* trade rather than a quality trade when you are memory-bound.

**Q7. What two files constitute a PEFT adapter, and which field is silently unvalidated?**

- **Answer:** `adapter_config.json` (r, lora_alpha, lora_dropout, target_modules, task_type, `base_model_name_or_path`) plus `adapter_model.safetensors`. The **`base_model_name_or_path` is never validated** — a mismatch loads without error and degrades subtly.
- **Why the interviewer asks this:** This pair of files is the interop currency of the entire ecosystem; it is why a LoRA trained in Unsloth can be served by vLLM or resumed in Axolotl.
- **Trap:** Assuming the base path is checked. It is not. Always print it and compare against the base you intend to serve on, and record the **revision hash**, not just the model name.

**Q8. What is the merge formula for a LoRA, and in what precision should you merge?**

- **Answer:** `W' = W + (α/r)·BA`. Merge in **bf16 on CPU** (`merge_and_unload()` with `device_map="cpu"`), never fp16 on GPU. Merging a bf16-trained adapter into an fp16-loaded base silently rounds the delta and shifts outputs, most visibly at long context.
- **Why the interviewer asks this:** The merge is the one line that is framework-independent, and it is the last place a good run gets quietly ruined.
- **Trap:** "The merge is lossless so precision doesn't matter." It is not lossless in the arithmetic sense; it is a floating-point accumulation you can easily do in the wrong dtype.

**Q9. Distinguish SFT, CPT, and instruction tuning.**

- **Answer:** **SFT** is supervised fine-tuning on (prompt, response) pairs — cross-entropy on the response tokens. **CPT** (continued pretraining) is next-token loss on raw unlabelled domain text: it changes the base **distribution** and is the correct move for domain shift, but it does not teach instruction-following. "Instruction tuning" is SFT — the same loss, a different name for the data.
- **Why the interviewer asks this:** Confusing CPT with SFT produces the classic failed project: a model that knows the domain vocabulary but ignores your instructions.
- **Trap:** "I'll LoRA it on the domain corpus to teach it the domain." Low-rank adapters absorb *associations and formats* well and large *factual corpora* poorly — knowledge injection is CPT (CS-12) or full FT territory.

**Q10. What does DPO need that plain SFT does not?**

- **Answer:** A **reference model** — a frozen copy of the SFT model used in the KL term — plus preference pairs (chosen/rejected) rather than single responses. That reference doubles memory in naive DPO, which is why sharding or offloading is mandatory above small sizes.
- **Why the interviewer asks this:** It is the most common DPO-start OOM, and the reason "DPO on 4×A100 for a 13B" is a sharding question, not an algorithm question.
- **Trap:** "DPO is RLHF without PPO, so it needs no extra memory." The KL-to-reference term is precisely what makes it stable *and* what makes it expensive.

**Q11. What is GRPO, and what does it need that SFT/DPO do not?**

- **Answer:** Group Relative Policy Optimisation — PPO without a critic, using group-normalised advantages across multiple completions sampled per prompt. It needs a **high-throughput rollout engine** (vLLM/SGLang), a **reward function that returns a scalar per completion**, and group sampling.
- **Why the interviewer asks this:** It is the dominant 2025–26 post-training workload (RLVR for reasoning), and the video's matrix has no GRPO column at all (CS-03 §4.16).
- **Trap:** "GRPO needs a reward model." Verifiable rewards (exact match, unit tests) are enough — that is the whole point of RLVR. It also does **not** run well by calling `model.generate()` in the training loop: rollout is 60–85% of step time, and a training-only framework leaves 5–10× on the floor.

**Q12. What is NF4, and how does it differ from GPTQ and AWQ?**

- **Answer:** NF4 is 4-bit **NormalFloat** quantization, information-theoretically optimal for normally distributed weights — it is the **training-time** base format for QLoRA. GPTQ and AWQ are **post-training, weight-only** quantization formats for **inference**. GGUF is llama.cpp's container format for CPU/consumer inference.
- **Why the interviewer asks this:** Mixing the training-time 4-bit with inference quantization formats is a classic conceptual error (CS-10/CS-11 territory, not CS-03).
- **Trap:** "I'll train into a GPTQ model." You cannot; GPTQ is a post-training format. Also do not treat GGUF as a training checkpoint — LoRA→GGUF requires the merge step first, plus a tokenizer/chat-template mapping.

**Q13. What is `bitsandbytes` and what is the one practical constraint on it?**

- **Answer:** The library that provides 8-bit optimizers and `load_in_4bit` / `load_in_8bit` — the universal quantization backend for HF, Axolotl, LLaMA-Factory and Unsloth. On **Windows it effectively does not work**; it is Linux (or WSL2) only, alongside `flash-attn` and `deepspeed`.
- **Why the interviewer asks this:** It is the #1 "my install is broken" report, and the answer is environmental rather than algorithmic.
- **Trap:** Trying to fix it on native Windows. Rent a Linux box or use WSL2; do not spend the day compiling.

**Q14. What is a chat template, and why is `template:` the highest-risk key in any config?**

- **Answer:** The chat template is the exact string format the model was instruction-tuned on — where the system/user/assistant turn markers go. It is the highest-risk key because getting it wrong produces a **smooth, plausible loss curve and a broken model**: the model learns the wrong turn structure and can answer *as the user*.
- **Why the interviewer asks this:** It is the #1 silent failure mode in the entire module (CS-03 §9.3, §17.7), and the case study's first applied example turned entirely on fixing it.
- **Trap:** "The tokenizer's default template is close enough." Every framework falls back to something when the template is unset, and the fallback is almost never what the base was trained on. Set `template: <exact family>` (LLaMA-Factory) / `chat_template:` (Axolotl), or hand-write it.

**Q15. Where does FlashAttention sit, and does it change your results?**

- **Answer:** It is an IO-aware **exact** attention kernel (FA2/FA3) giving 2–4× attention speedup and memory linear in sequence length. It does not change results beyond floating-point numerics. It also does not build on Turing (T4, sm_75) — use `attn_implementation="sdpa"` or `xformers` there.
- **Why the interviewer asks this:** It is the most common "Axolotl install failed" cause and a check on whether you understand it as an exact kernel rather than an approximation.
- **Trap:** "FlashAttention is an approximation, so my eval numbers differ." It is exact; your differences come from dtype, kernel selection, and nondeterminism — not from the attention math being approximate.

**Q16. What is gradient checkpointing, and what does it cost?**

- **Answer:** Recomputing activations during the backward pass instead of storing them: ~60–70% activation-memory reduction for a **~25–35% step-time penalty**. It is the single biggest step-time tax.
- **Why the interviewer asks this:** It is the first lever when you are memory-bound and the first thing to switch off when you are not, and it changes the wall clock you quote in a cost estimate.
- **Trap:** "It's free." It is not, and at scale it is worth money — but it is still usually correct, because the alternative is an OOM.

**Q17. What does `packing` do, and when is it the wrong choice?**

- **Answer:** Packing concatenates short samples to fill the sequence length, giving ~2–3× token throughput for raw-text CPT. It is the **wrong** choice for chat/instruction data with distinct turns: with a wrong chat template or EOS handling, two examples get concatenated into one training row and the model learns to continue someone else's answer.
- **Why the interviewer asks this:** It is a throughput knob that turns into a data-corruption bug when applied by reflex.
- **Trap:** Turning packing on everywhere for speed. Rule of thumb: `packing: false` for chat data, `packing: true` for CPT on raw text.

**Q18. What is MFU, and why is it the only honest throughput number?**

- **Answer:** Model FLOPs Utilisation = achieved FLOPs ÷ peak FLOPs. It is honest because tokens/second is not comparable across different GPUs, sequence lengths and batch sizes.
- **Why the interviewer asks this:** Every vendor speed claim ("2× faster") is meaningless without a baseline and a utilisation number, and interviewers use this to test whether you quote benchmarks critically.
- **Trap:** Quoting "2× faster" without saying faster than *what*, at what batch size, on which GPU, at which sequence length.

**Q19. What is the Unsloth speed claim, and what is the honest version of it?**

- **Answer:** The claim is "2× faster with 80% less VRAM" for Qwen/Llama/Gemma/Phi/Mistral. The honest version: **≈2× step time and 60–80% VRAM reduction on single-GPU QLoRA versus a naive HF + FA2 baseline, at batch size 1–2, on the architectures it has Triton kernels for.**
- **Why the interviewer asks this:** This is the module's canonical "read the benchmark critically" question, and the video repeats the vendor number unqualified (`[10:50]`).
- **Trap:** "Unsloth is 2× faster, period." The speedup shrinks toward 1.0–1.3× when the baseline already uses FA2 + `torch.compile` + fused optimizers, when sequences are short, when your architecture is unsupported, and especially when you move to multi-GPU.

**Q20. What is a rollout engine, and where does it appear in the RL pipeline?**

- **Answer:** A high-throughput sampler (vLLM, SGLang) used inside RL training to generate completions. It sits **outside** the training engine — veRL's architecture is a training engine (FSDP/Megatron) and a rollout engine as separate processes with a data-transfer protocol between them.
- **Why the interviewer asks this:** It is the architectural reason RL frameworks look nothing like SFT trainers, and the reason RL step time is dominated by sampling.
- **Trap:** Using `model.generate()` inside the PPO/GRPO loop and then concluding "RL is impossibly slow." It is 5–10× slower than it needs to be.

**Q21. What is the abstraction-level axis, ordered from least to most code?**

- **Answer:** Hosted API (OpenAI/Vertex/Bedrock/Together/Predibase) → GUI/YAML (LLaMA-Factory WebUI, AutoTrain, ms-swift web-ui) → config + Python (Axolotl, LLaMA-Factory CLI, torchtune YAML) → raw Python (`transformers` + `trl` + `peft` + `accelerate`/DeepSpeed, LitGPT/Lightning).
- **Why the interviewer asks this:** It predicts how fast you can get to a first run versus how much control you have when the config does not express what you need.
- **Trap:** Treating "low abstraction" as "less capable." LLaMA-Factory's YAML exposes BAdam, GaLore, LongLoRA, PiSSA and QAT — more training surface than most "expert" frameworks.

**Q22. What is SkyPilot, and why is it the highest-leverage tool in the video's list?**

- **Answer:** A cloud broker / cluster orchestrator: one YAML job spec, and it finds the cheapest available capacity across 16+ clouds and Kubernetes, provisions it, rsyncs your code, runs, and tears it down. It is the only entry in the list that addresses **cost**, which is the largest line item in a real budget.
- **Why the interviewer asks this:** Cost control is a senior signal, and the pair `use_spot: true` + `--retry-until-up` + frequent checkpointing is a 60–70% discount with automatic preemption recovery.
- **Trap:** "SkyPilot is a fine-tuning framework." It runs *whatever trainer you put in the spec*. Also: spot only works if your trainer checkpoints often enough to survive preemption — `save_strategy: epoch` on spot means a preemption loses the entire run.

**Q23. What are the four hosted-API providers' positions on exporting weights?**

- **Answer:** OpenAI — **no** (you get a model ID, never weights). AWS Bedrock — **no** for hosted models (custom-import models are yours). Google Vertex — **no for Gemini, yes for Gemma/OSS** models. Together and Predibase — **yes**, PEFT adapters or merged weights, which is their differentiator.
- **Why the interviewer asks this:** Weight ownership is the question that retroactively kills most hosted fine-tuning proposals, and it is invisible until the migration is due.
- **Trap:** Assuming "hosted fine-tuning" is one category. The exportable-adapter providers are a different business model from the closed ones, and the difference decides whether you can move on-prem in six months.

**Q24. What is the difference between a merged checkpoint and a PEFT adapter, operationally?**

- **Answer:** The adapter is ~20–200 MB (`adapter_config.json` + `adapter_model.safetensors`) and is **the source of truth**; the merged checkpoint is 14+ GB and exists for serving stacks that cannot load adapters. Keep the adapter alongside every merged checkpoint — the adapter is what moves between frameworks.
- **Why the interviewer asks this:** It decides your versioning and rollback story: 50 adapters cost ~5 GB, so there is no excuse for being unable to roll back.
- **Trap:** Deleting the adapter after merging. You have thrown away the portable artifact and kept the bulky derivative.

**Q25. What is DoRA, and when is it *not* the right choice?**

- **Answer:** Weight-decomposed low-rank adaptation: it splits the update into a magnitude component and a direction component, typically +1–4 points over LoRA at roughly 2× step cost. It is not the right choice for large `r`, where the advantage disappears.
- **Why the interviewer asks this:** It tests whether you treat PEFT variants as a menu with trade-offs rather than a quality dial you turn up.
- **Trap:** "DoRA is always better than LoRA." It costs ~2× step time and its edge is `r`-dependent.

**Q26. What is HSDP?**

- **Answer:** Hybrid Sharded Data Parallel — FSDP2's hybrid mode: shard within a node and replicate across nodes (`SHARD_GRAD_OP` / `HYBRID_SHARD` strategies). It reduces cross-node all-gather traffic, which is the expensive hop.
- **Why the interviewer asks this:** It is the standard multi-node answer, and knowing it signals you have thought about interconnect topology rather than just "more GPUs."
- **Trap:** Using `FULL_SHARD` across a slow inter-node link and then blaming the framework for throughput collapse.

**Q27. What is the license situation across this landscape?**

- **Answer:** Nearly everything is **Apache-2.0** (HF stack, LLaMA-Factory, Unsloth, Axolotl, ms-swift, xTuner, veRL, OpenRLHF, DeepSpeed, ColossalAI, LitGPT, OpenLLM, FastChat, SkyPilot, LightLLM); `bitsandbytes` is MIT; **PyTorch-native pieces are BSD-3-Clause** (FSDP/FSDP2, torchtune). Only the hosted APIs are proprietary.
- **Why the interviewer asks this:** It is a procurement/compliance question that surfaces in enterprise interviews, and it is usually a non-issue — the differentiator is not license but lock-in.
- **Trap:** Worrying about open-source licenses while ignoring the *real* lock-in risk: a hosted model ID has no artifact at all, and ColossalAI/veRL checkpoints are not PEFT without a conversion step.

**Q28. What is adapter interop, and what two things actually break a move between frameworks?**

- **Answer:** Because `peft` won the format war, moving a LoRA is usually just `PeftModel.from_pretrained`. The two things that break the move are (1) the **base-model revision** (`base_model_name_or_path` is unvalidated and a different revision degrades subtly) and (2) the **chat template / prompt format**, which must match what the adapter was trained on even when the weights load perfectly.
- **Why the interviewer asks this:** It distinguishes "I know adapters are portable" from "I know precisely where portability ends."
- **Trap:** "The adapter is portable, so everything is." The weights are; the *semantics* are not. LLaMA-Factory → ms-swift loads fine and then needs `template`/`model_type` re-alignment.

**Q29. What is `stage` (LLaMA-Factory) or `training_type` (Axolotl), and what values matter?**

- **Answer:** The training method selector. LLaMA-Factory: `stage: sft | dpo | kto | orpo | ppo | rm | pretrain`. Axolotl: `training_type: sft | dpo | orpo | kto | grpo`. It is the key that decides which loss, which dataset schema, and which extra config keys are required.
- **Why the interviewer asks this:** It is the first line of any config and it determines the entire shape of the run — including whether you now need a reference model.
- **Trap:** Changing `training_type: dpo` without switching the dataset `type:` to `preference` (Axolotl) or the dataset to a preference format. You get a run that trains on the wrong schema.

**Q30. What is the difference between the "learning framework" and the "production framework" in the module's rule of thumb?**

- **Answer:** Pick exactly two: one you can debug in (`transformers` + `trl` + `peft`, where you see every tensor and the loss is 30 lines you can read) and one that does your actual job (LLaMA-Factory for breadth/UI, Axolotl for YAML-at-scale and teams, Unsloth for single-GPU speed). Everything else is a component of those two, a serving layer, or a cloud scheduler.
- **Why the interviewer asks this:** "More frameworks explored = more expertise" is the most common junior framing. A candidate who knows one framework's failure modes deeply reads as senior; one who has installed ten reads as a tourist.
- **Trap:** Listing ten frameworks as a strength. The senior answer names two and can describe their silent failure modes in detail.

---

## Level 2 — Applied & Implementation

**Q31. Give the config key for sequence length in `trl`, LLaMA-Factory, Axolotl and torchtune.**

- **Answer:** `max_length` (trl; renamed from `max_seq_length` with a deprecation shim), `cutoff_len: 2048` (LLaMA-Factory), `sequence_len: 2048` (Axolotl), `tokenizer.max_seq_len: 2048` (torchtune).
- **Why the interviewer asks this:** It is the single clearest demonstration that frameworks are "vocabulary plus execution strategy, not different mathematics" — and it is how you catch a copy-pasted-from-2024 config.
- **Trap:** Using `max_seq_length` with a modern `trl` and concluding the library is broken. Verify against your pinned version; this is the most common paste-failure in the ecosystem.

**Q32. How do you enable QLoRA in each of the four main trainers?**

- **Answer:** trl: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` + `prepare_model_for_kbit_training`. LLaMA-Factory: `quantization_bit: 4` + `quantization_method: bnb`. Axolotl: `load_in_4bit: true` + `adapter: qlora` + `optimizer: paged_adamw_8bit`. torchtune: `quantizer._component_: BitsAndBytesQuantizer`.
- **Why the interviewer asks this:** QLoRA is the default memory regime; if you cannot enable it in the framework you claim, you have not run it.
- **Trap:** Setting `quantization_bit: 4` and leaving the compute dtype at fp16 on a bf16-native model — the second-most-common NaN in the module.

**Q33. How do you register a custom dataset in LLaMA-Factory?**

- **Answer:** Add an entry to `data/dataset_info.json` mapping your dataset name to a `format` (e.g. `alpaca`, `sharegpt`), a `path`, and a `columns` mapping — then reference the name in your YAML as `dataset: my_dataset`.
- **Why the interviewer asks this:** It is the thing that actually blocks beginners, and it reveals whether you have used the tool rather than read about it.
- **Trap:** Pointing `dataset:` at a file path instead of the registered name, or forgetting the `columns` mapping when your field names differ from the format's defaults.

**Q34. What is the CLI surface of LLaMA-Factory?**

- **Answer:** `llamafactory-cli train <yaml>` (run), `webui` (LLaMA Board, Gradio), `chat <yaml>` (interactive with the adapter), `export <yaml>` (merge LoRA → safetensors), `api <yaml>` (OpenAI-compatible server on a vLLM backend).
- **Why the interviewer asks this:** It is the only truly no-code path in the video's list, and `export` is the step that turns an adapter into something a serving stack accepts.
- **Trap:** Forgetting that the WebUI is a **local** Gradio app — on a remote box you need `ssh -L 7860:localhost:7860`, which is where most first attempts stall.

**Q35. What does `export.yaml` do, and which key must stay `false`?**

- **Answer:** It merges the adapter into a deployable checkpoint: `model_name_or_path`, `adapter_name_or_path`, `template`, `finetuning_type: lora`, `export_dir`, `export_size` (shard size in GB), and `export_legacy_format: false`. Keep `export_legacy_format: false` — that is what gives you safetensors rather than the legacy pickle format.
- **Why the interviewer asks this:** The export path is where frameworks stop being interchangeable, and the legacy-format flag is a small thing with a security and interop consequence.
- **Trap:** Leaving `export_legacy_format: true` and shipping a `pytorch_model.bin` because a tutorial from 2023 said so.

**Q36. How do you launch an Axolotl run, and how do you merge with it?**

- **Answer:** `accelerate launch -m axolotl.cli.train custom-config.yaml` (or the wrapper `axolotl train custom-config.yaml`); then `axolotl inference custom-config.yaml --lora-model-dir ./outputs/sft-lora` and `axolotl merge-lora custom-config.yaml`.
- **Why the interviewer asks this:** It is the concrete evidence of hands-on use, and the `accelerate launch -m ...` form is what you need when you start adding FSDP/DeepSpeed flags.
- **Trap:** Assuming the wrapper and the module form are interchangeable on a multi-node job. The wrapper is a convenience; the `accelerate` form is where the distributed config is expressed.

**Q37. Write the Axolotl incremental-YAML ladder: what changes to go from single-GPU LoRA to QLoRA to DPO to multi-GPU?**

- **Answer:** Layer "changes only" files on a base config. QLoRA adds `load_in_4bit: true`, `bnb_4bit_compute_dtype: float16`, `bnb_4bit_quant_type: nf4`, `adapter: qlora`, `optimizer: paged_adamw_8bit`. DPO changes `training_type: dpo`, swaps the dataset to `type: preference`, and adds `dpo_beta: 0.1`. Multi-GPU adds `distributed_type: fsdp` with `fsdp: {sharding_strategy: FULL_SHARD, auto_wrap_policy: transformer, state_dict_type: full, sync_module_states: true}`.
- **Why the interviewer asks this:** This ladder is the module's clearest demonstration that a well-designed config is a diffable artifact — the reason Axolotl is a tier-1 choice for teams despite the video ranking it tier 2.
- **Trap:** Copying the whole config for each variant instead of layering. Then a change to the base model or LR silently applies to one experiment and not the others.

**Q38. What are the Unsloth install pins, and why do they matter?**

- **Answer:** `pip install unsloth` then `pip install transformers==4.56.2` and `pip install --no-deps trl==0.22.2` (plus `psutil`, and torch/xformers from the cu128 index). They matter because Unsloth's kernels and patched model classes are pinned against specific `transformers`/`trl` internals — an unpinned upgrade breaks them.
- **Why the interviewer asks this:** It is the practical cost of a kernel-level library, and it is a real reproducibility requirement rather than fussiness.
- **Trap:** Letting `pip` resolve `trl`'s dependencies normally. `--no-deps` is deliberate: the stock `trl` dependency range would pull a `transformers` version Unsloth does not support.

**Q39. Why does Unsloth set `lora_dropout=0.0`?**

- **Answer:** Unsloth patches the LoRA path for speed; a non-zero dropout requires a slower fused path, so the library recommends 0.0 "for speed and stability." If you need dropout as a regularizer on a very small dataset, you are trading the kernel advantage for it.
- **Why the interviewer asks this:** It is a specific detail that proves you read the library rather than the blog posts, and it is a genuine trade-off rather than a default to copy blindly.
- **Trap:** Setting `lora_dropout=0.1` because "that's what you do on small data" and then wondering why the kernel speedup vanished.

**Q40. What are torchtune's launch commands for single-GPU LoRA and multi-GPU FSDP2?**

- **Answer:** `tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device` for one GPU; `tune run --nnodes 1 --nproc_per_node 8 fsdp2_lora_finetune_distributed --config <cfg>` for eight. Recipes are single-file Python you fork when you need to change logic.
- **Why the interviewer asks this:** torchtune is the most conspicuous absence from the video's list and the most documented trainer on memory numbers — knowing its CLI signals you read beyond the video.
- **Trap:** Treating torchtune as config-only. The config *is* YAML, but the abstraction is "recipe," so anything beyond the exposed keys means editing a ~200-line Python file.

**Q41. What is the LitGPT CLI for a LoRA fine-tune?**

- **Answer:** `litgpt finetune lora --checkpoint_dir <dir> --data ...` (with `litgpt pretrain`, `litgpt chat`, `litgpt convert from_hf`/`to_hf` for the other stages).
- **Why the interviewer asks this:** LitGPT is very likely what the video's "LightLLM or VLM" slide intended, and the conflation is a correction worth carrying into an interview.
- **Trap:** Confusing LitGPT (Lightning AI, from-scratch readable training implementations) with **LightLLM** (ModelTC, an inference/serving engine). They are unrelated projects with different jobs.

**Q42. Which DeepSpeed config key must be set or your checkpoint is unusable?**

- **Answer:** `"stage3_gather_16bit_weights_on_model_save": true` under `zero_optimization`. Without it, a ZeRO-3 run leaves you with a sharded checkpoint that will not load as a normal model — the classic "0-byte checkpoint" symptom.
- **Why the interviewer asks this:** It is the single fact that saves a multi-day run, and it is missing from most tutorials.
- **Trap:** Discovering it after the run. The alternative recovery is `zero_to_fp32.py` on the sharded directory — possible, but you would rather not be doing it under deadline.

**Q43. What is the FSDP operational rule that saves multi-day runs?**

- **Answer:** Save with `SHARDED_STATE_DICT` during training and convert to `FULL_STATE_DICT` only at the end (or merge adapters directly from the sharded checkpoint). A `FULL_STATE_DICT` save on a 70B job gathers every shard onto rank 0 and routinely OOMs or takes 40+ minutes per checkpoint.
- **Why the interviewer asks this:** It is the difference between a 30-hour job that completes and one that dies at hour 24 while writing a checkpoint.
- **Trap:** Setting `state_dict_type: full` "because it's more portable." It is more portable and it will kill your run.

**Q44. What is the ms-swift CLI, and what is it best at?**

- **Answer:** `swift sft` / `swift rlhf` (plus `swift web-ui`, `swift infer`, `swift export`). Best at **VLM/multimodal training and the Qwen ecosystem**, plus Megatron TP+PP+CP+EP parallelism for very large MoE models.
- **Why the interviewer asks this:** ms-swift is the biggest capability omission from the video's list. If your organisation touches Qwen-VL, InternVL or DeepSeek-VL, it belongs in your top three.
- **Trap:** Reaching for it for a plain 7B text SFT — LLaMA-Factory or Unsloth is a faster path. Its advantage is multimodal and MoE-at-scale, and its documentation is Chinese-first.

**Q45. What is unusual about xTuner's config format, and what is the practical benefit?**

- **Answer:** It is a **Python** config file, not YAML (`xtuner train config.py`), so fields can be computed programmatically — e.g. `max_length` as a function of your dataset rather than a literal. It is strongest for 1B–8B single-GPU runs and InternLM/InternVL-family work.
- **Why the interviewer asks this:** It shows that the YAML-vs-Python config choice is a real ergonomic trade-off, not a stylistic preference.
- **Trap:** Recommending it as a general default. It has a smaller model list and a much smaller English-language community than LLaMA-Factory or ms-swift — it is a specialist tool.

**Q46. What does a veRL launch look like, and what is the key architectural fact about it?**

- **Answer:** A Hydra config plus a Python entry point, e.g. `python3 -m verl.trainer.main_ppo --config-name=... algorithm.adv_estimator=grpo ...`. Architecturally, the **training engine (FSDP/Megatron) and the rollout engine (vLLM/SGLang) are separate processes** with a data-transfer protocol between them — which is why it can do 70B RL at all.
- **Why the interviewer asks this:** It is the framework behind most open reasoning-model post-training recipes of 2025–26, and the separation is the insight worth having.
- **Trap:** Assuming the sampling engine is an implementation detail. RL step time is 60–85% rollout, so the rollout engine's throughput — not the trainer's — sets your wall clock.

**Q47. How does OpenRLHF differ philosophically from veRL?**

- **Answer:** OpenRLHF is **Ray + vLLM + DeepSpeed/FSDP** in a single readable repo with an actor/critic/reward/reference decomposition — optimised for readability and customisation. veRL is optimised for **maximum throughput** with a Megatron trainer and a disaggregated rollout service.
- **Why the interviewer asks this:** It is the clearest example in the module of the same problem solved with opposite philosophies, and picking between them is a real decision.
- **Trap:** "They're interchangeable." Reach for OpenRLHF when learning or customising the algorithm; reach for veRL when the bottleneck is tokens/second on a 70B actor.

**Q48. What does a hosted-SFT payload look like, and what does the response give you?**

- **Answer:** A JSONL of `messages`-format examples — `{"messages":[{"role":"system","content":...},{"role":"user",...},{"role":"assistant",...}]}` — uploaded to a job-creation call (e.g. `openai api fine_tuning.jobs.create -m gpt-4o-mini-2024-07-18 -f train.jsonl --suffix "support-tone-v3"`). The response is a **model ID** (`ft:gpt-4o-mini-...:acme:support-tone-v3:9xK2...`), never weights.
- **Why the interviewer asks this:** It is the entire mechanical difference of the hosted branch: you never load a model, and the artifact you receive is a string.
- **Trap:** Planning to "move it on-prem later." There is nothing to move. The work must be redone on an open model.

**Q49. How do you merge a LoRA from any framework, and what should you verify first?**

- **Answer:** Load the base in bf16 on CPU, `PeftModel.from_pretrained(base, adapter)`, print `model.peft_config["default"].base_model_name_or_path` and compare it against the base you intend to serve on, then `merge_and_unload()` and `save_pretrained(..., safe_serialization=True)`.
- **Why the interviewer asks this:** It is the framework-agnostic step that works on output from trl, LLaMA-Factory, Axolotl, Unsloth, ms-swift and xTuner alike — the payoff of PEFT winning the format war.
- **Trap:** Skipping the `base_model_name_or_path` check. A mismatch merges cleanly and degrades subtly, and you will attribute it to your data.

**Q50. What is the eval protocol that survives review?**

- **Answer:** (1) A frozen held-out set never used in training; (2) the **base model as baseline** through the same harness; (3) a regression set of 50–100 prompts the previous model handled correctly; (4) randomised judge order with reported agreement against 30 human labels; (5) the exact adapter hash + base revision recorded in the report.
- **Why the interviewer asks this:** Without (2) you are measuring nothing — the delta is the only number that matters, and most teams omit it.
- **Trap:** Reporting "93% accuracy" with no baseline. Also: evaluating with the judge you are optimising against, without randomising A/B order (position bias, verbosity bias, self-preference).

**Q51. Which framework has a first-class run-ID resume key, and why does it matter?**

- **Answer:** Axolotl — `wandb_project`, `wandb_name`, `wandb_mode`, `wandb_run_id`. It is the richest first-class logging config in the landscape, and the run-ID key lets you *resume and continue* a logged run rather than starting a new one.
- **Why the interviewer asks this:** Observability is one flag almost everywhere; this is the one place a framework offers something the others do not, and it matters for long spot-instance jobs.
- **Trap:** Setting `report_to=["wandb"]` without `WANDB_PROJECT` — the run lands in a default project and teams lose weeks of experiments that way.

**Q52. How do you go multi-GPU in HF `Trainer` in 2026?**

- **Answer:** Set `fsdp` / `fsdp_config` or `deepspeed="ds_config_zero3.json"` in `TrainingArguments` and launch with `accelerate launch`. Native tensor parallelism is available via `tp_plan="auto"`.
- **Why the interviewer asks this:** The video claims "G4 multi-GPU is not very good" for HF `[25:51]` — **wrong for 2026**. FSDP2 and ZeRO-3 are first-class paths; what is true is that HF's *defaults* are not tuned for multi-GPU.
- **Trap:** Repeating the video's claim in an interview. The correct nuance: the capability exists, but you must configure it explicitly, and it is a ~30-minute learning cost rather than a capability gap.

**Q53. What is the difference between `per_device_train_batch_size` and Axolotl's `micro_batch_size`?**

- **Answer:** They are the same quantity under different names. Effective batch = micro batch × `gradient_accumulation_steps` × data-parallel ranks. Axolotl: `micro_batch_size`; HF/LLaMA-Factory: `per_device_train_batch_size`; torchtune: `batch_size`.
- **Why the interviewer asks this:** Batch arithmetic is where OOM and instability are decided, and the renaming is a Rosetta-stone row.
- **Trap:** Setting `gradient_accumulation_steps` in *both* the YAML and the DeepSpeed JSON — some versions multiply the two. Set it once.

**Q54. What does `dpo_beta` / `pref_beta` control, and what are its failure directions?**

- **Answer:** The KL strength toward the reference model, typically 0.1 (safe range 0.01–0.5). Too high → the preference signal is ignored (the model stays near the reference). Too low → drift and degenerate outputs.
- **Why the interviewer asks this:** DPO's one number is also its most mis-set one, and the failure modes point in opposite directions.
- **Trap:** Copying `0.1` while running full-FT DPO at a LoRA learning rate. That combination — not `beta` — is what produces degenerate repetition by step 200.

**Q55. What learning rate do you use for LoRA vs full FT vs DPO?**

- **Answer:** LoRA 1e-4–3e-4 (default 2e-4); full FT 1e-5–5e-5 (default 2e-5); DPO 5e-6 (and 5e-7–1e-6 for many LoRA DPO setups); GRPO ~1e-6.
- **Why the interviewer asks this:** It is the most common catastrophic failure in the module: a full fine-tune run at LoRA's learning rate.
- **Trap:** Using the same LR across methods because "they're all fine-tuning." The correct mental model is that the LR is 10–100× smaller than pretraining and differs by an order of magnitude between full FT and adapter tuning.

**Q56. How do you use gradient checkpointing in each of the four trainers?**

- **Answer:** `gradient_checkpointing=True` (HF `TrainingArguments`), `gradient_checkpointing: true` (LLaMA-Factory and Axolotl), `model.set_gradient_checkpointing`-equivalent in the torchtune recipe config — and in Unsloth specifically `use_gradient_checkpointing="unsloth"` for the cheaper patched path.
- **Why the interviewer asks this:** It is the first memory lever and the one with a quantified cost (+25–35% step time), so it belongs in any sizing estimate you quote.
- **Trap:** Assuming Unsloth's `"unsloth"` mode always applies. On an unsupported architecture it falls back to standard checkpointing **without warning loudly**, so you get the VRAM profile of the standard path.

**Q57. What is the `lora_target: all` setting in LLaMA-Factory, and why is it the safe default?**

- **Answer:** It attaches adapters to every linear layer, avoiding the `target_modules` name-matching problem entirely. On an unfamiliar architecture, guessing the module names is the fastest route to a silent no-op.
- **Why the interviewer asks this:** The "trainable params printed as 0.000%" failure is common enough that frameworks added a blunt fix for it.
- **Trap:** Hand-listing `q_proj, v_proj` for a new architecture and getting zero trainable parameters with a loss that barely moves. Print `model` and copy exact `*_proj` names, or use `all-linear`.

**Q58. How do you spot that a fine-tune produced a PEFT adapter you can actually move?**

- **Answer:** The output directory contains `adapter_config.json` + `adapter_model.safetensors`. That is produced by trl, LLaMA-Factory, Axolotl, Unsloth, ms-swift, xTuner and torchtune (on HF models). It is **not** produced by veRL/OpenRLHF full checkpoints, ColossalAI chunked checkpoints, or hosted APIs.
- **Why the interviewer asks this:** It is a five-second check that predicts whether you have lock-in, and it is the operational form of the module's interop story.
- **Trap:** Assuming everything emits a PEFT adapter. ColossalAI's Gemini checkpoints need a `zero_to_fp32`-style conversion, and veRL needs a `save_hf_weights`-style flag.

**Q59. What three things do you version per model release?**

- **Answer:** The triple `(base model revision hash, adapter hash, chat template file)`. A model is not a file; it is those three things, and they must be tagged together in the registry.
- **Why the interviewer asks this:** It is the production answer to "how do you roll back?", and it is the concrete remedy for the unvalidated-`base_model_name_or_path` class of bug.
- **Trap:** Versioning only the adapter. The same adapter on a different base revision — or with a different template at serve time — is a different model.

**Q60. What is the regression-test practice that is worth the most for the least effort?**

- **Answer:** A frozen set of 50–200 prompts with expected properties, run in CI on every candidate adapter — plus a moderation/refusal slice, because SFT on unguarded data will happily unlearn refusals. It is the single highest-value 4 hours a fine-tuning team can spend.
- **Why the interviewer asks this:** Fine-tuning removes safety behaviour, and the failure is silent until production. An interviewer is checking whether you gate on more than loss.
- **Trap:** Gating only on the target-metric improvement. Without the regression set you cannot see what the run broke, and the A/B should measure the business metric, not the judge score, in the final gate.

---

## Level 3 — Analysis & Trade-offs

**Q61. Do the bytes-per-parameter arithmetic for a 70B full FT and show which sharding strategies fit on 8×80 GB.**

- **Answer:** bf16 weights (2) + bf16 grads (2) + fp32 AdamW (8) = **12 bytes/param = 840 GB unsharded**. ZeRO-2 on 8 GPUs: `2 + 2/8 + 8/8` = 3.25 B/param → 227 GB/GPU — **does not fit**. ZeRO-3 on 8 GPUs: 12/8 = 1.5 B/param → 105 GB/GPU — **still does not fit** (needs 15–25 GB headroom for activations). ZeRO-3 on **16** GPUs: 0.75 B/param → 52.5 GB/GPU + activations ≈ 70 GB — **fits**.
- **Why the interviewer asks this:** This arithmetic is the module's feasibility test. Rows that "look slow" in a framework comparison are actually **impossible**, and the interviewer wants to see you distinguish the two.
- **Trap:** Answering from intuition ("8×H100 is a lot of memory"). At 70B full FT, 8×80 GB is not enough, and the difference between ZeRO-2 and ZeRO-3 is a factor of two in per-GPU state.

**Q62. So how *do* you do 70B on 8×80 GB?**

- **Answer:** QLoRA + ZeRO-3: base at 0.5 B/param (35 GB ÷ 8 ≈ 4.4 GB/GPU) plus sharded LoRA optimizer state ≈ **under 12 GB/GPU**, leaving room for activations. This is the standard 70B recipe.
- **Why the interviewer asks this:** It is the bridge from the arithmetic to the actual recommendation, and the number (sub-12 GB/GPU) is the surprising part that shows you have run it.
- **Trap:** Concluding "70B is impossible on 8×80 GB." It is impossible for *full FT*; it is routine for QLoRA.

**Q63. Why is multi-GPU QLoRA often *slower* than single-GPU QLoRA?**

- **Answer:** Two mechanisms. (1) ZeRO-3/FSDP must **all-gather every layer's parameters per forward/backward**, and 4-bit weights must be **de-quantized before the gather** — so you pay communication *and* de-quantization per layer per step. (2) On PCIe without NVLink, that all-gather costs more than the compute it enables, and the framework may also lose the single-GPU kernel-fusion benefit.
- **Why the interviewer asks this:** It is the most counter-intuitive operational fact in the module, and it distinguishes people who have run multi-GPU QLoRA from people who have only run it on one card.
- **Trap:** "More GPUs is always faster." The fix set is ZeRO-2 instead of 3, `offload_param: cpu`, FSDP2, or staying on one GPU. The case study's legal-domain example found QLoRA on 2×A100 was **1.4× slower per step** than bf16 LoRA — 4-bit de-quantization buys nothing when memory is not the constraint (CS-03 §15, Case 2).

**Q64. When is DeepSpeed still the right choice in 2026, and when is FSDP2 better?**

- **Answer:** DeepSpeed wins for (a) ZeRO-Infinity NVMe offload, (b) mixed TP+PP+ZeRO at very large scale, (c) MoE/expert parallelism, (d) existing `DeepSpeed-Chat` RLHF pipelines, (e) `DeepSpeed-FastGen`/`AutoTP` for inference. FSDP2 wins as the default for new single-codebase projects: PyTorch-native (no extra install or version pinning), per-parameter DTensor sharding, clean `state_dict` semantics, and predictable composition with `torch.compile`, `torchao` float8 and QLoRA.
- **Why the interviewer asks this:** It is the 2026 re-evaluation of a 2023-era default, and the answer "let the YAML pick the backend" is the mature position.
- **Trap:** Declaring DeepSpeed obsolete. Both Axolotl and LLaMA-Factory support either, which is exactly why the choice should live in config rather than in your architecture.

**Q65. Why does framework choice barely matter below 13B and matter enormously above it?**

- **Answer:** Below: the algorithms are identical (every framework calls the same cross-entropy / DPO loss), data quality dominates by roughly 10:1, no framework can rescue a broken memory plan on one GPU, and every trainer converges on the same PEFT artifact. Two frameworks with identical data/LR/batch/seed/template converge within run-to-run noise (±0.5–1 point). Above: optimizer-state sharding decides whether a 70B job's 560–840 GB of state fits **at all**, kernel fusion is a 1.5–2.5× step-time difference worth thousands of dollars, and rollout integration is 5–10× end-to-end in RL.
- **Why the interviewer asks this:** It is the module's thesis sentence, and a candidate who claims "Unsloth gives better quality" is signalling inexperience.
- **Trap:** Claiming a framework determines quality at small scale, or claiming frameworks do not matter at all. Both halves are wrong.

**Q66. What does switching frameworks actually cost?**

- **Answer:** 0.5–2 days to relearn the config vocabulary (no tooling converts YAML across frameworks), 1–3 days to re-validate the data pipeline, 0.5–1 day for the export/serve path, 1–3 days × people for retraining — **1–2 engineer-weeks total**, plus a **permanent** loss of run comparability (different RNG, LR-schedule step counting and gradient-accumulation placement mean run A ≠ run B even with the same seed).
- **Why the interviewer asks this:** It is the argument for choosing once, on the hard constraint, and it shows you understand that reproducibility is a capital asset.
- **Trap:** Treating a framework switch as a config translation. The permanent part — that your historical runs are no longer comparable to your new ones — is the expensive part and it is not recoverable.

**Q67. A model must learn a large body of new facts. Does LoRA suffice?**

- **Answer:** No, not reliably. LoRA learns new **associations and formats** efficiently; it is poor at absorbing large **factual corpora** because a low-rank update cannot move enough of the weight space. Knowledge injection is CPT (continued pretraining on the domain text, at 5e-6–1e-5 with packing and no chat template) or full FT. LoRA is also usually the wrong tool for CPT for the same reason.
- **Why the interviewer asks this:** It is the most common mismatch between a stated goal ("teach it our product catalogue") and a chosen method, and it is checkable by asking what the eval set tests.
- **Trap:** "LoRA can do anything, just raise `r`." Raising `r` is a capacity dial with sharp diminishing returns, and the boundary between "associations it can absorb" and "facts it cannot" is fuzzy but real.

**Q68. When is ColossalAI the right answer, and when is it not?**

- **Answer:** Right when you must mix **TP+PP+DP on a heterogeneous cluster with offload in one config**, and as a **reading list** — `colossalai/zero/gemini/` is the clearest reference implementation of chunk-based ZeRO-3 with offload in the open ecosystem. Not right as the 2026 default trainer for a 7B–70B SFT/DPO job: its LLM fine-tuning path (`ColossalChat`) has a lower commit cadence than Axolotl/LLaMA-Factory, and FSDP2 + ZeRO-3 + `torch.compile` cover ~95% of production demand.
- **Why the interviewer asks this:** It tests whether you can hold a nuanced "narrower than it was in 2023, still wins in two places" position rather than either recommending or dismissing it wholesale.
- **Trap:** Repeating the video's framing that ColossalAI's value is "parallel training = multi-GPU." DDP does multi-GPU in four lines; ColossalAI's differentiator is that `Gemini` treats GPU + CPU + NVMe as a **three-tier cache of parameter chunks**.

**Q69. What was the video's "LightLLM / VLM" slide actually about?**

- **Answer:** It conflates two unrelated projects. **LightLLM** (ModelTC) is a Python **inference and serving** framework — not a trainer. **LitGPT** (Lightning AI) is the from-scratch, readable **training and inference** implementation of ~20 architectures (pretrain → SFT → LoRA → DPO) that belongs in a fine-tuning Top-10. Different maintainers, different jobs, coincidentally both Apache-2.0.
- **Why the interviewer asks this:** It is a correction the case study flags, and a reader who searches "LightLLM fine-tuning" lands on the serving repo and wrongly concludes the framework is too limited.
- **Trap:** Defending the slide (the instructor does self-correct at `[19:39]` after reading the README aloud — credit where due) or assuming "LightLLM" and "LitGPT" are two names for one project.

**Q70. Where does FastChat still earn its place?**

- **Answer:** In **evaluation and multi-model serving** — MT-Bench and the LLM-as-judge harness are the reference implementations most teams copy when building an eval suite — and in the arena-style multi-model chat UI. Not as a trainer: its `train_lora.py` has not tracked modern `peft`/`trl` APIs and its dependencies are pinned to older `transformers`.
- **Why the interviewer asks this:** "Where does FastChat fit?" is a precise question with a precise answer, and it tests whether you classify tools by what they currently do rather than by what a slide calls them.
- **Trap:** Calling it a fine-tuning framework because it ships a training script. Shipped is not maintained.

**Q71. What is OpenLLM's genuine differentiator in 2026?**

- **Answer:** **Multi-LoRA serving** — one base model with N adapters loaded simultaneously and routed per request — plus one-command packaging into a containerised, OpenAI-compatible service. It competes with `vllm serve`, SGLang and TGI, which have larger communities.
- **Why the interviewer asks this:** It is the tool you reach for when you have ten customers each with their own LoRA, which connects framework selection to the multi-tenant serving pattern.
- **Trap:** Picking it as a default serving stack on popularity grounds. Its niche is adapter multiplexing; for a single merged model, vLLM is the safer default.

**Q72. Where is the hosted-vs-self-hosted break-even, and why does it move?**

- **Answer:** Roughly **5–20M tuned-model tokens per month**. A tuned `gpt-4o-mini` costs ~1.5–2× the base per token (a tuned larger model 3–4×); one rented A100-80GB running vLLM on a 7B at ~2,000 tok/s costs on the order of $0.30 per million tokens at full utilisation. Below the range, hosted is cheaper *and* simpler; above it, self-hosting wins by a growing margin.
- **Why the interviewer asks this:** It is the economic gate on the entire hosted branch, and it explains why "we fine-tuned GPT-4o-mini" projects get redone on an open model six months later.
- **Trap:** Comparing only the training cost. The per-token inference premium is what flips the sign, and it scales with your traffic, not your training budget.

**Q73. What three questions decide hosted vs self-hosted?**

- **Answer:** (1) **Must you own the weights?** If yes (regulated data, on-prem serving, no vendor dependency, distillation into a smaller model), hosted is disqualified except Vertex+Gemma and the Together/Predibase export paths — this kills ~80% of hosted proposals retroactively. (2) **What is your monthly inference volume?** (the 5–20M break-even). (3) **Which methods do you need?** Hosted SFT is universal; hosted DPO/GRPO is provider-specific and often feature-flagged; hosted reward-model training is rare.
- **Why the interviewer asks this:** It converts a vague preference into three decidable questions, which is exactly what a design interview is looking for.
- **Trap:** Answering on cost alone. Question 1 is a compliance constraint that can close the branch regardless of price, and it must be asked first.

**Q74. What is the most common silent failure in RL fine-tuning, and why is it not a framework problem?**

- **Answer:** An **exploitable reward function** — the policy satisfies the *string* rather than the *intent*. The reward curve rises monotonically while true quality falls (verbosity hacking, length proxies, format-matching without reasoning). It is a data/reward-design failure: no framework can fix it, because the framework is correctly maximising the objective you wrote.
- **Why the interviewer asks this:** It is the failure that survives every infrastructure improvement, and the case study's 32B reasoning example hit it directly (reward up, judge score down).
- **Trap:** Treating a rising reward curve as success. Always log completion length alongside reward, keep a held-out judge as a **gate** rather than an optimisation target, and add a length/KL penalty.

**Q75. What share of RL step time is rollout, and what follows from it?**

- **Answer:** **60–85%**. It follows that the rollout engine's throughput — not the trainer's — sets your wall clock, which is why the training engine and rollout engine are separate processes in veRL, and why a framework that calls `model.generate()` in the loop is 5–10× slower end-to-end.
- **Why the interviewer asks this:** It is the single number that justifies an entire architectural family, and it is the strongest argument for veRL/OpenRLHF over an SFT trainer for RL work.
- **Trap:** Optimising the trainer's MFU. You are optimising the 15–40% of the step.

**Q76. Why does kernel fusion matter more at scale than at one GPU?**

- **Answer:** Fused RoPE/RMSNorm/SwiGLU/fused-cross-entropy plus memory-efficient attention give ~1.5–2.5× step time and 60–80% activation VRAM reduction. At 1 GPU that is a convenience; at 64 GPUs it is a five-figure line item — and a framework that falls back to eager attention will OOM at 8k context where another does not.
- **Why the interviewer asks this:** It is how you convert "speed" from a vendor claim into a budget argument, and it explains the legitimate part of Unsloth's claims.
- **Trap:** Dismissing kernel work as micro-optimisation. The mechanism that makes it matter is that the cost is *per GPU-hour multiplied by GPU count*.

**Q77. Why did PEFT win the format war, and what is the resulting contract?**

- **Answer:** Because every trainer needs a portable artifact, and `peft` defined one: a directory with `adapter_config.json` (r, alpha, dropout, target_modules, task_type, `base_model_name_or_path`) and `adapter_model.safetensors`. It is a two-file interface that any framework can write and any serving stack can read, so it became the interop currency of the ecosystem.
- **Why the interviewer asks this:** It is the reason framework choice is *reversible* at small scale — you can move the adapter — and the reason the module can recommend picking two frameworks without fear.
- **Trap:** Assuming the contract is enforced. `base_model_name_or_path` is never validated, so the interface is a convention, not a check.

**Q78. What is the portability rule set that prevents most migration pain?**

- **Answer:** (1) Always record the exact `base_model_name_or_path` **including the revision hash**. (2) Always keep the raw PEFT adapter (~100 MB) alongside any merged checkpoint (~14 GB) — the adapter is what moves. (3) Always keep the tokenizer and the exact chat template as a file in the run directory. (4) Never delete the merged model, but treat the adapter as the source of truth.
- **Why the interviewer asks this:** These four rules are the operational residue of every framework in the landscape and cost nothing to follow.
- **Trap:** Keeping only the merged model because "it's the real thing." You have kept the derivative and discarded the portable original.

**Q79. What does observability look like across the landscape?**

- **Answer:** One flag nearly everywhere: `report_to=["wandb"]` (HF, LLaMA-Factory, Unsloth, ms-swift), `use_tensorboard` / `mlflow_*` (Axolotl), per-recipe loggers (torchtune), unified loggers (veRL/OpenRLHF, which also log reward/KL/entropy per component). Axolotl is the only one with a first-class **run-ID resume** key.
- **Why the interviewer asks this:** It is the cheapest dimension to compare and the one that is essentially solved — knowing that lets you spend your evaluation effort on the dimensions that are not.
- **Trap:** Treating logging differences as a selection criterion. They are one flag apart everywhere; parallelism and methods are not.

**Q80. What is the checkpoint/resume correctness issue, and why is it worth 30 hours?**

- **Answer:** A long spot job must resume **exactly**. Frameworks differ in whether optimizer state, LR-schedule position and RNG state all round-trip — and gradient-accumulation placement and step counting disagree across frameworks, so a checkpoint from one is not safely resumed in another.
- **Why the interviewer asks this:** It is the mechanism by which a 60–70% spot discount becomes either a real saving or a disaster, and it is invisible until the first preemption.
- **Trap:** Resuming the same run in a different framework. Resume in the framework that produced the checkpoint, or restart with recalculated epochs.

**Q81. Why is the video's comparison matrix structurally out of date?**

- **Answer:** It has **no column for GRPO, no column for a rollout engine, and no column for DAPO/RLVR** — yet RLVR is the dominant post-training workload of 2025–26 and is *architecturally different* from SFT/DPO (asynchronous high-throughput sampling, a reward function executed per completion, group-normalised advantages). It also uses LLaMA-1-era 13B/65B sizing and has no FSDP2 or Megatron column.
- **Why the interviewer asks this:** It is the "can you date a source?" question. The instructor himself says the matrix "could be 100% correct, could not" `[47:28]` — quote him on that rather than pretending the slide is authoritative.
- **Trap:** Memorising the matrix as fact. Know it as a *map*, with the four corrections the case study flags.

**Q82. What is the one filter you apply before any other framework comparison?**

- **Answer:** The **parallelism gate** — how many GPUs and how big is the model. It is the only column that is a hard gate: a framework that cannot shard will not run your 70B job regardless of its UI, its methods or its speed. The procedure is: filter on parallelism first (kills Unsloth for multi-node, xTuner above 13B full FT, ColossalAI for most people as a trainer), then on methods (kills the serving tools as trainers), then pick on ergonomics among survivors.
- **Why the interviewer asks this:** It is the decision *procedure*, and an interviewer is checking for a procedure rather than a preference.
- **Trap:** Starting from popularity, stars, or a headline speed claim — all soft, temporary dimensions. The gate is hard and permanent.

---

## Level 4 — System Design & Scenario

Each prompt expects: **requirements → constraints → design → trade-offs → failure modes.** Answer out loud in 8–12 minutes. The grader is listening for the arithmetic and the failure modes, not the product names.

**Q83. "You have 8×H100-80GB. You need DPO on a 70B model. You have three weeks. What do you use, and why?"**

- **Answer:**
  — **Requirements.** 70B preference tuning, ship in three weeks, so the run must be *first-try feasible* rather than maximally efficient.
  — **Constraints.** 70B full FT is 12 B/param = 840 GB unsharded; even ZeRO-3 on 8 GPUs is 105 GB/GPU and does not fit with activations. DPO additionally needs the actor **and** a reference model. So the memory regime is forced: **QLoRA on a 4-bit base with a sharded adapter** — under ~12 GB/GPU on 8 ranks, leaving room for activations. Timebox: one probe run, then the real run.
  — **Design.** **Axolotl + FSDP2 (or ZeRO-3) + QLoRA + FlashAttention-3**, `stage`/`training_type: dpo`, LoRA `r=16–32` on all projections, `sequence_len` sized to the 95th percentile of your pairs (not the max), `save_steps: 100–250` (not per epoch), `save_strategy` sharded with a single `FULL_STATE_DICT` conversion at the end. DPO LR 5e-6 (full-FT scale) or 5e-7–1e-6 for a LoRA DPO setup, `beta` 0.1 as the starting point. Expected wall clock **26–38 h** (CS-03 §1.4), which fits three weeks with room for one failed run and one hyperparameter sweep. Orchestrate with SkyPilot on spot (`use_spot: true`, `--retry-until-up`) for a 60–70% discount.
  — **Trade-offs.** QLoRA costs 4-bit de-quantization overhead on the all-gather — measurable, and worth it, because the alternative does not fit. FSDP2 over ZeRO-3 for cleaner `state_dict` semantics and QLoRA composition; ZeRO-3 is equally defensible if you want the broader offload surface. Do **not** reach for veRL here: DPO is not online RL and needs no rollout engine — reaching for it costs configuration time you do not have.
  — **Failure modes to name.** (1) ZeRO-3 `stage3_gather_16bit_weights_on_model_save` missing → unusable checkpoints, and `zero_to_fp32.py` is a poor place to be at hour 30. (2) `FULL_STATE_DICT` during training → OOM on rank 0 every save. (3) Multi-GPU QLoRA slower than single-GPU because of per-layer de-quant + all-gather over PCIe — check tokens/s vs `n=1` early, and fall back to ZeRO-2 + `offload_param: cpu` if it is. (4) Full-FT DPO LR on a sharded setup → degenerate repetition by step 200. (5) Reference model forgotten in the memory plan, so you OOM at step 0.
- **Why the interviewer asks this:** It is the canonical "does the framework decide feasibility?" question, and it has one arithmetic answer that most candidates will not do.
- **Trap:** Answering "veRL, because RL" — DPO is not online RL and needs no rollout engine — or answering "full FT with ZeRO-3 on 8 GPUs", which is 105 GB/GPU and simply does not run.

**Q84. "One 24 GB GPU, an unfamiliar model family, and you need a working SFT in two days. What do you use?"**

- **Answer:**
  — **Requirements.** A working adapter, on unknown architecture and unknown chat template, in 48 hours.
  — **Constraints.** 24 GB caps you at QLoRA for anything ≥7B. "Unfamiliar" is the binding constraint, not the hardware: time-to-first-run dominates.
  — **Design.** **LLaMA-Factory**: 100+ supported models, one YAML, `quantization_bit: 4`, `finetuning_type: lora`, `lora_target: all` (which sidesteps the `target_modules` name-matching problem entirely), `template: <exact family>` — and use `llamafactory-cli webui` (LLaMA Board) for the first run to avoid a config typo, noting it is a local Gradio app needing `ssh -L 7860:localhost:7860` on a remote box. Then `llamafactory-cli export export.yaml` to merge. If the model *is* one Unsloth supports, Unsloth is faster (~2× step time, 60–80% less VRAM) — but on day one, breadth and a working template beat speed.
  — **Trade-offs.** You give up the speed of Unsloth kernels and the debugging transparency of raw `trl`. In exchange you get a first run in 20–40 minutes instead of 2–4 hours, which is the whole budget. Note that LLaMA-Factory's online-RL paths need an external vLLM rollout — irrelevant for SFT.
  — **Failure modes.** (1) `template:` wrong → perfect loss curve, model answers as the wrong speaker — the #1 silent failure and the reason to check a *rendered* training string before launching. (2) `target_modules` hand-listed and no match → 0 trainable params, loss flat. (3) WebUI used remotely without port-forwarding. (4) `preprocessing_num_workers > 0` with a tiny dataset and a 0.05 eval split → empty eval set and a divide-by-zero on some versions.
- **Why the interviewer asks this:** It is the single most common real briefing, and the correct answer is "the framework that minimises time-to-first-run", not the fastest kernel.
- **Trap:** Choosing Unsloth for an unfamiliar architecture. Its model list is deliberately narrow; the 2× claim evaporates on an unsupported model and you lose a day.

**Q85. "4×A100-80GB, a 13B model, 12k preference pairs, DPO. Design it."**

- **Answer:**
  — **Requirements.** Preference alignment without drift into verbosity, on a fixed node.
  — **Constraints.** DPO needs actor + reference; 4×80 GB is comfortable for 13B in bf16 LoRA (no 4-bit needed) and tight for full FT with optimizer state.
  — **Design.** **Axolotl or LLaMA-Factory with DPO + FSDP2/ZeRO-2, bf16 LoRA r=16–32** (no quantization — at 80 GB it buys nothing and costs de-quantization time), `pref_beta: 0.1`, LR 5e-7–1e-6 for the LoRA DPO path, 1–2 epochs, `save_steps: 250`. Reference model sharded or offloaded. Eval gate: LLM-judge win rate **plus output length** (a +6% length change is acceptable; a monotonic length climb is verbosity hacking), plus a regression set of 50–100 prompts.
  — **Trade-offs.** LoRA rather than full FT costs a little ceiling and buys a shardable, resumable, 100 MB artifact you can serve as a multi-tenant adapter. ZeRO-2 avoids ZeRO-3's per-layer all-gather, which matters more than sharding aggressiveness at this size. Full FT is defensible at 4×80 GB but triples the checkpoint size and makes the reference model expensive.
  — **Failure modes.** (1) Full-FT DPO LR on a LoRA setup → degenerate repetition by step ~200. (2) `beta` too high → the preference signal is ignored; too low → drift. (3) Reward/judge evaluated with the same judge you optimise, unrandomised A/B order → position and verbosity bias. (4) Dataset `type:` still set to a completion format after flipping `training_type: dpo` → trains on the wrong schema. (5) Reference model unaccounted for in the memory plan → OOM at step 0.
- **Why the interviewer asks this:** It is the mid-scale case where the framework genuinely matters (sharding is mandatory) but is not yet exotic — the realistic median production job.
- **Trap:** Reaching for QLoRA by reflex on 80 GB cards. The case study's legal-domain example found QLoRA **1.4× slower per step** on 2×A100 than bf16 LoRA for exactly this reason.

**Q86. "8×H100, GRPO on a 32B model for math reasoning. Design the stack."**

- **Answer:**
  — **Requirements.** Reasoning RL against **verifiable rewards** (exact match against reference answers), which means group sampling, a scalar reward per completion, and high sampling throughput.
  — **Constraints.** Rollout is 60–85% of step time, so a training-only framework wastes 5–10×. 32B requires sharding plus, at this size, likely TP within the node.
  — **Design.** **veRL** — Megatron actor (TP=4, PP=2) plus a **disaggregated vLLM rollout at TP=4**; GRPO with group size 8, LR ~1e-6, KL coefficient ~0.001, 2 epochs. If readability and customisation matter more than peak throughput, **OpenRLHF** (Ray + DeepSpeed ZeRO-3 + vLLM TP, colocated or disaggregated). For ≤8B, `trl`'s `GRPOTrainer` or Unsloth's GRPO notebooks suffice and are far cheaper to operate. Cost: the case study's 32B example ran ~60 h and ~$1,100 on spot for pass@1 41% → 58%. Budget the verifier and the eval harness as first-class work, not an afterthought.
  — **Trade-offs.** veRL: maximum throughput, steep config surface, no QLoRA-first path (RL on a 4-bit base is fragile). OpenRLHF: single readable repo, easier reward-function iteration, lower ceiling. Colocated rollout is simpler; disaggregated is faster and needs more nodes. Do **not** try this in Axolotl/LLaMA-Factory alone — you need a rollout engine.
  — **Failure modes.** (1) **Exploitable reward** — the model emits the answer *format* without the reasoning, reward rises and the judge score falls. Add a held-out judge as a gate and a length/KL penalty; this is a reward-design failure no framework can fix. (2) `model.generate()` in the training loop → 5–10× slower. (3) No group-size/KL tuning → entropy collapse and degraded diversity. (4) Reference/KL accounting causing memory to blow up mid-run, after the sampling fleet is already provisioned.
- **Why the interviewer asks this:** It is the 2026 workload that the video's matrix cannot represent at all, so it tests whether your framework knowledge extends past the source.
- **Trap:** Answering with an SFT trainer. Or answering "PPO needs a reward model" — GRPO with verifiable rewards does not.

**Q87. "Forty customers, one base model, each needs its own adapter, refreshed monthly. Serving and training design?"**

- **Answer:**
  — **Requirements.** One base, N tenant-specific adapters, monthly refresh, isolation between tenants, predictable serving latency.
  — **Constraints.** N merged checkpoints would be 40 × 14 GB — wasteful and unrollbackable. Per-tenant adapters are 20–200 MB, so 40 of them cost single-digit GB.
  — **Design.** Train per tenant with the **same base revision** and a fixed template: Unsloth for single-GPU per-tenant runs, or LLaMA-Factory/Axolotl when a tenant's job needs more than one GPU. Serve with **multi-LoRA** — `vllm serve --enable-lora` or **OpenLLM** (whose genuine differentiator is loading and routing many adapters at once), or Predibase/LoRAX if you want it hosted. Version the triple `(base revision, adapter hash, template file)` per tenant and keep the previous adapter on disk for one-command rollback.
  — **Trade-offs.** Adapter serving adds a small per-request routing cost and requires the base revision to be pinned explicitly; merging removes that bug class at the cost of 14 GB per tenant and a redeploy per update. Adapters win decisively here. Training cost is per-tenant GPU time, so a shared data-prep and eval pipeline is where the leverage is.
  — **Failure modes.** (1) Serving an adapter against a **different base revision** than `base_model_name_or_path` — loads fine, degrades subtly. (2) An adapter merged in fp16 from a bf16-trained run. (3) Cross-tenant template drift — one tenant's adapter trained with a different prompt format and silently answering as the wrong speaker. (4) No regression set, so a monthly refresh regresses one tenant undetected. (5) Unbounded adapter count in VRAM — multiplexing is not free.
- **Why the interviewer asks this:** It connects framework selection to a business pattern, and the correct answer (adapters, not merged checkpoints) depends on understanding what a PEFT adapter *is*.
- **Trap:** Proposing 40 merged models, or forgetting that multi-tenant serving requires an explicit base revision.

**Q88. "A regulated bank. Data cannot leave the VPC. They want a support model fine-tuned. Design it."**

- **Answer:**
  — **Requirements.** A working support model, trained entirely inside the VPC, with data lineage for audit.
  — **Constraints.** **Hosted APIs are closed regardless of cost** — this is the first branch of the decision tree and is decided *before* any framework comparison (CS-03 §16). Compliance and governance sign-off happen before the framework choice; PII redaction must happen before tokenization, since no framework has a PII hook. They also have no GPU team, which caps operational complexity.
  — **Design.** Rent inside their own cloud tenancy and run a **declarative** trainer so the config is a reviewable artifact: **Axolotl** (diffable YAML, FSDP/ZeRO paths, `wandb_run_id` resume, first-class governance-friendly logging) or **LLaMA-Factory** if the model is unfamiliar and the team wants a UI. Single 80 GB node, bf16 LoRA (or QLoRA if the node is 24 GB), `template:` set exactly, `save_steps: 250`, W&B in offline mode with a sync path that actually works inside the VPC. Data lineage: store the dataset hash in run metadata; version `(base revision, adapter hash, template file)`; keep a frozen regression set of 50–200 prompts — including **refusal cases**, because SFT on unguarded data unlearns safety behaviour — run in CI on every candidate adapter. SkyPilot can bid for capacity *within* their VPC or approved providers.
  — **Trade-offs.** Self-hosting costs more per token below the 5–20M/month break-even and needs ops competence. In exchange they own the weights, pass audit, and can serve on-prem indefinitely. The honest note: they could not have done this on a hosted API, so the "hosted is cheaper" argument never applied.
  — **Failure modes.** (1) A pilot quietly run on a hosted API "just to test" — a compliance breach with no artifact at the end. (2) Redaction after tokenization. (3) `WANDB_MODE=offline` in a container that never syncs → no logs at all. (4) No refusal cases in the regression set, so guardrails silently erode. (5) Fine-tuning as a substitute for prompt engineering where RAG was the right answer (CS-04).
- **Why the interviewer asks this:** It is the enterprise-readiness question the video's second half is actually about `[42:07]`–`[45:59]`, and it tests whether you lead with constraints rather than tools.
- **Trap:** Recommending OpenAI/Bedrock/Vertex on cost grounds. Question 1 — "must you own the weights?" — disqualified them before the pricing discussion began.

**Q89. "A startup prototyping 20 different fine-tunes, and cost is the binding constraint. Design it."**

- **Answer:**
  — **Requirements.** Many cheap iterations, most of which will be thrown away; the goal is learning rate of experimentation, not peak quality.
  — **Constraints.** Budget-bound, not capability-bound. Each run must be resumable because cheap capacity means spot.
  — **Design.** **SkyPilot on top of the trainer** — `use_spot: true`, `cloud: aws,gcp,azure,lambda` to bid across providers, `any_of` fallbacks (H100 → L40S when A100 capacity is gone), `--retry-until-up`, and **managed jobs** (`sky jobs launch`) which checkpoint-resume automatically after preemption. Trainer: **Unsloth** for single-GPU QLoRA (best 24 GB experience by a wide margin) or LLaMA-Factory when the model is unfamiliar. Configs in git as diffable YAML so a 20-run sweep is 20 reviewable files.
  — **Trade-offs.** Spot is 60–70% cheaper ($1.20 vs $3.50 per GPU-hour) but preemptible, which forces `save_steps: 100` instead of `save_strategy: epoch`. Frequent checkpointing costs a little throughput and is the entire reason the discount is real. Prototyping on spot also means variable run-to-run timing, so plan on MFU/step time rather than wall clock.
  — **Failure modes.** (1) `save_strategy: epoch` on spot → a preemption loses the whole run — this is the #1 way the discount becomes a disaster. (2) No `WANDB_PROJECT` → 20 runs scattered across a default project. (3) Prototyping on a single-GPU-optimised stack and promising a 70B DPO run later, forcing a rewrite at the worst time (`§1.1` decision 2). (4) Optimising quality while the constraint is cost — at prototypes, data quality dominates 10:1.
- **Why the interviewer asks this:** SkyPilot is the only entry in the video's list that addresses cost, and cost is the largest line item; this scenario checks whether you treat cost as a design constraint.
- **Trap:** Answering with a framework comparison and no cost mechanism, or using on-demand instances for throwaway prototypes.

**Q90. "You need to fine-tune a vision-language model for document extraction. Which stack?"**

- **Answer:**
  — **Requirements.** Multimodal SFT (and possibly DPO) over images + text, with correct handling of image tokens and a custom collator.
  — **Constraints.** Standard text trainers do not have mature VLM collators, packing or DPO paths; a wrong `model_type`/template makes the model train on the wrong image tokens — and the loss still looks healthy.
  — **Design.** **ms-swift** is the reference for VLM/omni training (Qwen-VL, InternVL, LLaVA, CogVLM; `swift sft`/`swift rlhf`, a `swift web-ui`, plus Megatron parallelism for large MoE VLMs). **LLaMA-Factory** is the strong second (multimodal SFT/DPO, `llamafactory-cli webui`, 100+ models). Unsloth has VLM support (Qwen2-VL/2.5-VL, Llama-3.2-Vision) but a narrower list. Whichever you pick: verify the rendered training example *including its image tokens*, check pixel/image-token budget against your sequence length, and evaluate on held-out documents with an extraction-specific metric (field-level F1), not perplexity.
  — **Trade-offs.** ms-swift: deepest VLM support and strongest Qwen-ecosystem alignment, but Chinese-first documentation and deep config plumbing. LLaMA-Factory: easier first run, slightly less VLM depth. Unsloth: fastest single GPU, narrowest coverage.
  — **Failure modes.** (1) `model_type`/template mismatch → training on wrong image tokens, invisible in loss. (2) Image resolution/token budget silently truncating the document. (3) Packing across multimodal examples corrupting the image-text alignment. (4) Evaluating with text perplexity on a vision task. (5) Exporting through a path that drops the vision tower's config.
- **Why the interviewer asks this:** VLM work is where LLaMA-Factory's and ms-swift's collators actually earn their keep, and it is a common production need that the video never mentions.
- **Trap:** Reaching for Unsloth or Axolotl first. Narrow VLM coverage means you will write the collator yourself — which is the work the framework was supposed to do.

**Q91. "A 70B full fine-tune across multiple nodes. What does the stack look like?"**

- **Answer:**
  — **Requirements.** Full FT (not PEFT — e.g. the model must absorb a large factual corpus, or you are producing a new base), 32B–70B, multiple nodes.
  — **Constraints.** 70B full FT is 840 GB of weights+grads+optimizer state unsharded, and the optimizer state alone is 560–840 GB. Sharding is not sufficient at this size: you need **TP + PP** as well, because per-layer all-gather over an inter-node fabric does not scale.
  — **Design.** **Megatron-LM**, reached via **veRL** or **ms-swift** (both expose Megatron TP+PP+CP+EP), with 2D/3D composition and HSDP where applicable; or ColossalAI if you need `Gemini`'s three-tier GPU/CPU/NVMe chunk caching on a heterogeneous cluster. Activation checkpointing on, `FULL_STATE_DICT` only at the end, and resume correctness verified *before* the long run. Consider whether PEFT+CPT would meet the actual goal — it usually does, and it is one to two orders of magnitude cheaper.
  — **Trade-offs.** Full FT buys the highest ceiling for new knowledge and the most expensive, least reversible run in the landscape: 14 GB+ checkpoints, weeks of engineering, and a permanent break from any prior LoRA experiments. CPT + LoRA is the pragmatic alternative and often closes most of the gap.
  — **Failure modes.** (1) Reaching for FSDP2 alone — it shards but does not give you TP/PP, and it will not carry a 70B full FT across nodes. (2) Save-time gather OOM on rank 0. (3) Interconnect: TP on a slow link is worse than PP; get the topology right. (4) Resume that does not round-trip optimizer state, LR position and RNG → a 30-hour restart. (5) No pre-flight memory arithmetic, so the failure appears at step 0 after a multi-node allocation.
- **Why the interviewer asks this:** It is the top of the difficulty curve the case study describes (`§8.3` branch G), where the framework is not a convenience but the only way the job exists.
- **Trap:** Answering "ZeRO-3 on 16 GPUs" for a *full* FT — that is the right answer for 8×80 GB *QLoRA*, and the wrong one here.

**Q92. "You prototyped on Unsloth with one GPU. Now it must run on 8 GPUs in production. What is the migration?"**

- **Answer:**
  — **Requirements.** Same job, 8× the hardware, plus reproducibility and team reviewability that a notebook does not provide.
  — **Constraints.** Unsloth's multi-GPU path is narrower than the others and its kernel advantage largely does not survive the move (~2× shrinks toward 1.0–1.3× vs a tuned baseline). Its selling point was single-GPU speed; that is exactly what you are leaving behind.
  — **Design.** Treat it as a **framework switch**, not a scale-out. Move to **Axolotl** (diffable YAML, FSDP1/ZeRO-1/2/3, multi-node via `accelerate`, `wandb_run_id` resume) or **LLaMA-Factory** (ZeRO-2/3, FSDP1/2, Ray multi-node). Both accept the same PEFT adapter you already trained, so nothing is lost. Port the config key by key using the Rosetta table, re-validate the data pipeline and template, re-run a short comparison, and only then launch. Re-derive the memory plan for the new regime (bf16 LoRA if you now have 80 GB cards — no 4-bit needed).
  — **Trade-offs.** You lose Unsloth's single-GPU step time and gain sharding, checkpoint/resume correctness, and a reviewable artifact. Budget **1–2 engineer-weeks** and accept that your prior runs are permanently incomparable to the new ones — different RNG, LR-schedule step counting and grad-accum placement.
  — **Failure modes.** (1) Assuming the Unsloth notebook "just needs `nproc_per_node`". (2) Keeping the notebook as the source of truth and losing reproducibility. (3) Carrying QLoRA onto 80 GB cards, where it is *slower* per step. (4) `fsdp` + `device_map` both set — they conflict. (5) Not re-verifying the chat template after the migration, which is the one thing a framework move can silently break.
- **Why the interviewer asks this:** It is the single most likely real-world consequence of the video's tier ranking, and it is the case the case study makes for putting Axolotl in tier 1 for teams (`§4.6`).
- **Trap:** "Unsloth supports multi-GPU now, so just run it on 8." Verify per model; and even where it works, you have given up the reason you chose it without gaining the reason you would choose Axolotl.

---

## Level 5 — Debugging & Incident Response

Answer as: **what you check first, what second, what you check before the next run.** The clock is real — say what you would do in the next 15 minutes versus what you would change before relaunching.

**Q93. A run has been going 40 minutes. Loss is flat and hasn't moved. `print_trainable_parameters()` says 0.000%.**

- **Answer:** The adapter matched no modules — `target_modules` names are wrong for this architecture, so nothing trains. **First:** print `model` and copy the exact `*_proj` names, or set `target_modules="all-linear"` / LLaMA-Factory's `lora_target: all`. **Second:** confirm the optimizer was built *after* the LoRA injection (an optimizer built before freezing/injection captures a stale parameter list). **Third, before relaunching:** assert `0 < trainable < total` and print the *first trainable tensor name* in the run preamble, so this class of failure can never be silent again.
- **Why the interviewer asks this:** It is the most common "loss never moves" cause in the entire module, and the fix is 30 seconds once you know it.
- **Trap:** Turning up the learning rate. The LR is irrelevant when zero parameters are trainable — and a high LR is a second, independent bug you are about to introduce.

**Q94. Loss descends beautifully to ~0.8. The model answers *as the customer* instead of as the assistant.**

- **Answer:** Wrong **chat template** — the exact silent failure the module is built around. **First:** print the *rendered* training string for one example and diff it against the base model's expected format; you will see the turn markers in the wrong places. **Second:** set `template: <exact family>` (LLaMA-Factory) / `chat_template:` (Axolotl), or use `template: default` and hand-write the format. **Third, before relaunching:** add a rendered-example assertion to the run preamble and re-run. The case study's first applied example gained its entire result from this single fix; the loss curve gave no signal whatsoever.
- **Why the interviewer asks this:** It is the #1 silent failure in the module, and a healthy loss curve is exactly what makes it dangerous.
- **Trap:** Blaming the data or the model. The data can be perfect and the model excellent; the turn structure is a *framework* concern, which is why `template:` is the highest-risk key in any config.

**Q95. Loss goes NaN at step 4.**

- **Answer:** Usually a dtype/LR mismatch. **First:** check `bnb_4bit_compute_dtype` — a fp16 compute dtype on a bf16-native model is the second-most-common NaN; set it to bf16. **Second:** check the LR against the method — a **full FT at LoRA's 2e-4** is the single most common catastrophic failure in the module; full FT is 1e-5–5e-5. **Third:** enable gradient clipping at 1.0, verify warmup is non-zero (6–10% for FT), and inspect the batch for an outlier-length row. **Before relaunching:** log grad-norm per step and filter rows above your length percentile.
- **Why the interviewer asks this:** It is diagnosed entirely by two flags, and the interviewer wants to see you reach for dtype first rather than immediately halving the LR.
- **Trap:** "It's fp16 overflow, switch to fp32." You lose 2× memory for no benefit; the fix is the compute dtype and the LR, not the storage precision.

**Q96. Your multi-GPU QLoRA run is *slower* per step than the same run on one GPU.**

- **Answer:** Expected, and mechanical. ZeRO-3/FSDP all-gathers every layer's parameters per forward/backward, and **4-bit weights must be de-quantized before the gather** — so you pay communication *and* de-quantization per layer per step. On PCIe without NVLink the gather costs more than the compute it enables. **First:** confirm with tokens/s vs `n=1` and check for NVLink. **Second:** drop to ZeRO-2, set `offload_param: cpu`, or move to FSDP2 (cheaper schedule, CPU-offload support). **Third, before relaunching:** if the model fits on one card, run it on one card — and if memory is not the constraint at all, drop QLoRA for bf16 LoRA, which the case study measured as **1.4× faster per step** on 2×A100.
- **Why the interviewer asks this:** It is the most counter-intuitive operational fact in the landscape, and it is the module's best evidence that "more GPUs" is not a strategy.
- **Trap:** Adding more GPUs, or blaming the framework. The mechanism is arithmetic: 4-bit storage and sharded communication are in tension by construction.

**Q97. A ZeRO-3 run finished. The checkpoint directory contains 0-byte or unloadable files.**

- **Answer:** `stage3_gather_16bit_weights_on_model_save` was not set. **First:** set it `true` in the DeepSpeed config. **Second,** for the run already finished: run `zero_to_fp32.py` on the sharded directory — the weights are there, just partitioned. **Third, before relaunching:** add a post-run checkpoint-load assertion to the pipeline (load the saved checkpoint in a fresh process and generate one token); this failure should never be discovered at deployment.
- **Why the interviewer asks this:** It is the single fact that saves a multi-day run, and recovery is possible but miserable.
- **Trap:** Declaring the run lost and re-running 30 GPU-hours. The shards are complete; you are one script away from a usable model.

**Q98. The merged model is measurably worse than the adapter was during training.**

- **Answer:** Precision or base-revision mismatch at merge time. **First:** check the merge dtype — merging a bf16-trained adapter into an fp16-loaded base silently rounds the delta. Merge in **bf16 on CPU**. **Second:** print `model.peft_config["default"].base_model_name_or_path` and compare it against the base you merged into; a different *revision* is a different model and the field is never validated. **Third, before relaunching:** put both checks in the merge script and gate on them.
- **Why the interviewer asks this:** It is a regression that appears only at the last step, after a successful training run — the worst place to find it.
- **Trap:** Assuming `merge_and_unload()` is lossless. It is a floating-point accumulation you can do in the wrong dtype, on the wrong base.

**Q99. The reward curve rises monotonically. The held-out judge score is flat, and completion length has tripled.**

- **Answer:** The reward is **exploitable** — the policy is satisfying the *string* (or a length proxy), not the intent: verbosity hacking. **First:** stop the run; you are optimising an objective you do not want. **Second:** log completion length and a held-out judge score **alongside** reward, and redesign the reward (require the reasoning trace, add a length/KL penalty, or move to a stricter verifier). **Third, before relaunching:** make the held-out judge a **gate**, never an optimisation target, and keep a human-labelled slice to validate the judge itself.
- **Why the interviewer asks this:** It is the most common silent failure in RL fine-tuning, and critically, **no framework can fix it** — which is the point of the question.
- **Trap:** Raising the KL coefficient and calling it fixed. KL constrains drift from the reference; it does not make a bad objective good.

**Q100. A spot job was preempted at hour 22. Nothing resumable exists on disk.**

- **Answer:** The checkpoint cadence was wrong for spot. **First:** check whether the trainer wrote anything at all — `save_strategy: epoch` on a 30-hour job with one epoch means nothing was ever written. **Second:** if a checkpoint exists, resume in the **same framework** that produced it (optimizer state, LR-schedule position and RNG state do not round-trip across frameworks) and use Axolotl's `wandb_run_id` if available. **Third, before relaunching:** set `save_steps: 100–250`, use SkyPilot **managed jobs** (`sky jobs launch`) which auto-restart from the last checkpoint after preemption, verify a checkpoint resume in a 5-minute test run *before* the real one, and confirm the preemption handling rather than assuming `--retry-until-up` resumes rather than restarts.
- **Why the interviewer asks this:** Spot is a 60–70% discount and the most common way teams turn a saving into a loss; the whole discount is contingent on checkpoint cadence.
- **Trap:** Blaming SkyPilot or the cloud. `--retry-until-up` gets capacity; **your** checkpoint frequency is what makes preemption survivable.

**Q101. The served endpoint's outputs do not match what you measured during evaluation.**

- **Answer:** You are not serving the model you evaluated. **First:** check whether the endpoint loaded the base model and ignored the adapter, or loaded a base of a different revision. **Second:** log the model hash/revision at the endpoint and compare it with the eval run's recorded `(base revision, adapter hash, template file)` triple. **Third, before relaunching:** either merge the adapter into a single artifact (removing this entire bug class at the cost of disk) or serve the adapter with an **explicitly pinned base revision** (`vllm serve --enable-lora` with the revision set). Add the triple to the health endpoint's metadata.
- **Why the interviewer asks this:** It is the deployment-half counterpart of the adapter-mismatch bug, and it is why serving requires the same versioning discipline as training.
- **Trap:** Re-running evaluation until the numbers agree. Nothing is wrong with your eval; something is wrong with what is deployed.

**Q102. Six weeks after shipping, you discover the adapter was trained against a different base revision than the one now in production.**

- **Answer:** `base_model_name_or_path` was recorded as a name (or `null` revision) and never validated — the module's canonical latent defect. **First:** stop incremental work and quantify the damage: run the frozen regression set against the current serving triple and compare with the training-time eval; the degradation is usually subtle, not catastrophic. **Second:** retrain on the pinned revision, or pin production to the revision the adapter actually saw. **Third, before relaunching:** version the triple `(base revision, adapter hash, template file)` in the registry, pin `revision=<sha>` in every `from_pretrained`, and add a CI check that asserts the adapter's recorded base matches the serving base.
- **Why the interviewer asks this:** It is the failure that hides longest, and it demonstrates why "a model is not a file; it is those three things" is a production rule rather than a slogan.
- **Trap:** Assuming the adapter is fine because the run completed and the eval looked good. It did look good — on a different base.

---

## Rapid Fire — True Or False

Answer in under 5 seconds each. The explanation is the point, not the verdict.

| # | Statement | T/F | Why |
|---|---|---|---|
| 1 | DeepSpeed is a fine-tuning framework. | **F** | It is a parallelism/memory library that lives inside trainers. |
| 2 | FSDP2 is PyTorch-native and needs no separate install. | **T** | Ships with PyTorch, BSD-3-Clause, DTensor-based. |
| 3 | SkyPilot is a trainer. | **F** | Cloud broker/orchestrator; it runs whatever trainer you put in the spec. |
| 4 | OpenLLM and FastChat are serving/eval tools, not trainers. | **T** | FastChat's trainer is dated; its MT-Bench harness is the useful part. |
| 5 | Unsloth is 2× faster than any other framework. | **F** | ~2× vs a *naive HF QLoRA baseline*, single GPU, supported archs, small batch. |
| 6 | QLoRA always costs measurable quality. | **F** | Adapter trains in bf16; gap is typically <1 point on task metrics. |
| 7 | A PEFT adapter is `adapter_config.json` + `adapter_model.safetensors`. | **T** | The two-file contract that makes LoRAs portable. |
| 8 | `base_model_name_or_path` is validated on load. | **F** | Never validated; a mismatch loads silently and degrades subtly. |
| 9 | ZeRO-3 on 8×80 GB fits a 70B full fine-tune. | **F** | 105 GB/GPU before activations; needs 16 GPUs, or QLoRA. |
| 10 | ZeRO-3 + QLoRA fits a 70B on 8×80 GB. | **T** | Under ~12 GB/GPU — the standard 70B recipe. |
| 11 | Multi-GPU QLoRA can be slower than single-GPU QLoRA. | **T** | Per-layer all-gather plus de-quantization, worst on PCIe. |
| 12 | FlashAttention changes your model's outputs (it is approximate). | **F** | It is exact (FA2/FA3); differences come from dtype and nondeterminism. |
| 13 | FlashAttention-2 builds on a T4. | **F** | Turing (sm_75) is unsupported; use `sdpa` or `xformers`. |
| 14 | `bitsandbytes` works well on native Windows. | **F** | Effectively Linux/WSL2 only, along with `flash-attn` and `deepspeed`. |
| 15 | Gradient checkpointing is memory-free. | **F** | ~60–70% activation memory saved for ~25–35% step-time cost. |
| 16 | You can train into a GPTQ model. | **F** | GPTQ/AWQ are post-training *inference* quantization formats. |
| 17 | GGUF is a training checkpoint format. | **F** | llama.cpp's inference container; LoRA→GGUF needs a merge first. |
| 18 | NF4 is the QLoRA base format. | **T** | 4-bit NormalFloat, optimal for normally distributed weights. |
| 19 | `packing: true` is safe for chat data. | **F** | Wrong template/EOS handling concatenates examples; use it for CPT. |
| 20 | A wrong chat template shows up as a bad loss curve. | **F** | Loss descends perfectly; the model answers as the wrong speaker. |
| 21 | The correct full-FT learning rate is 2e-4. | **F** | That is LoRA's LR; full FT is 1e-5–5e-5. |
| 22 | DPO needs a reference model. | **T** | The frozen SFT copy in the KL term; it doubles naive memory. |
| 23 | GRPO requires a learned reward model. | **F** | Verifiable rewards (exact match, unit tests) suffice — that is RLVR. |
| 24 | RL step time is dominated by the trainer. | **F** | Rollout/sampling is 60–85% of step time. |
| 25 | A framework that calls `model.generate()` in the RL loop is competitive. | **F** | 5–10× slower end-to-end than an integrated vLLM/SGLang rollout. |
| 26 | DeepSpeed config needs `stage3_gather_16bit_weights_on_model_save: true`. | **T** | Otherwise ZeRO-3 checkpoints are unusable without `zero_to_fp32.py`. |
| 27 | `FULL_STATE_DICT` is the right save type during a 70B FSDP run. | **F** | It gathers to rank 0 — OOMs or takes 40+ min per checkpoint. |
| 28 | Framework choice materially changes quality below 13B on one GPU. | **F** | Same loss function; data dominates ~10:1; ±0.5–1 point of noise. |
| 29 | Framework choice decides whether a 70B job runs at all. | **T** | Optimizer-state sharding is a hard gate, not a performance dial. |
| 30 | The parallelism strategy is a soft preference. | **F** | It is the only hard gate in the comparison matrix. |
| 31 | Switching frameworks costs about a day. | **F** | 1–2 engineer-weeks, plus permanent loss of run comparability. |
| 32 | LLaMA-Factory is a beginner tool with limited training surface. | **F** | Exposes BAdam, GaLore, LongLoRA, PiSSA, QAT, DoRA, 100+ models. |
| 33 | The LLaMA-Factory WebUI is remotely accessible by default. | **F** | Local Gradio; needs `ssh -L 7860:localhost:7860`. |
| 34 | Unsloth recommends `lora_dropout = 0.0`. | **T** | Non-zero dropout forces a slower path; 0.0 is for speed/stability. |
| 35 | Unsloth's install pins (`transformers==4.56.2`, `trl==0.22.2`) are optional. | **F** | Its kernels patch specific internals; unpinned upgrades break them. |
| 36 | `template: qwen` on a Qwen model is a nitpick. | **F** | It is the highest-risk key in the config; wrong template = broken model. |
| 37 | Axolotl is the only framework with a first-class run-ID resume key. | **T** | `wandb_run_id` (plus `wandb_project`/`name`/`mode`). |
| 38 | Merging a LoRA in fp16 on GPU is fine. | **F** | Merge in bf16 on CPU; fp16 merge rounds the delta silently. |
| 39 | Hosted fine-tuning always lets you export weights. | **F** | Only Vertex+Gemma, Together and Predibase; OpenAI/Bedrock do not. |
| 40 | Hosted vs self-hosted break-even is around 5–20M tuned tokens/month. | **T** | Below it hosted is cheaper; above it self-hosting wins increasingly. |

---

## Coding / Whiteboard Tasks

**Task 1 — Write the memory-plan function.** Given `params` (billions), `gpus`, and a regime (`full_fp32`, `full_bf16`, `full_bf16_8bit`, `lora_bf16`, `qlora`), return bytes/param, total GB, per-GPU GB, and a verdict (`fits` / `fits_tight` / `does_not_fit`), assuming 15–25 GB of activation headroom on an 80 GB card.

```python
def memory_plan(params_b, gpus, regime, gpu_gb=80, headroom_gb=20):
    """Bytes per trainable parameter by regime (CS-03 §11.1)."""
    B = 1e9
    table = {
        "full_fp32":      {"w": 4, "o": 8, "g": 4},
        "full_bf16":      {"w": 2, "o": 8, "g": 2},
        "full_bf16_8bit": {"w": 2, "o": 2, "g": 2},
        "lora_bf16":      {"w": 2, "o": 0.08, "g": 0.08, "shard_w": False},
        "qlora":          {"w": 0.55, "o": 0.08, "g": 0.08},   # NF4 base + bf16 adapter
    }[regime]
    # Only the trainable slice carries optimizer+grad state; for full FT that is everything.
    trainable = 0.01 if regime in ("lora_bf16", "qlora") else 1.0
    per_param = table["w"] + (table["o"] + table["g"]) * trainable
    total_gb  = per_param * params_b * B / 1e9
    per_gpu   = total_gb / gpus
    usable    = gpu_gb - headroom_gb
    verdict   = "fits" if per_gpu <= usable * 0.8 else ("fits_tight" if per_gpu <= usable else "does_not_fit")
    return {"bytes_per_param": round(per_param, 2), "total_gb": round(total_gb, 1),
            "per_gpu_gb": round(per_gpu, 1), "verdict": verdict}

# Sanity checks against the case study's numbers:
assert memory_plan(70, 16, "full_bf16")["per_gpu_gb"] == 52.5      # 12 B/param, ZeRO-3 on 16
assert memory_plan(70,  8, "full_bf16")["verdict"] == "does_not_fit"  # 105 GB/GPU
assert memory_plan(70,  8, "qlora")["verdict"] == "fits"           # the standard 70B recipe
```

*Grading notes.* Full marks require (a) 12 bytes/param derived as 2+2+8 rather than recalled, (b) correctly noting ZeRO-3 on **8** GPUs is 105 GB and therefore a **does_not_fit**, and (c) naming the activation headroom as the reason "fits" is not simply `total/gpus ≤ 80`. Deduct heavily for a verdict of "fits" on `full_bf16` with 8 GPUs — that is the answer the whole question exists to test. Bonus for noting the LoRA/QLoRA rows carry optimizer state only on the adapter slice.

**Task 2 — Write the framework-selection function, parallelism gate first.** Input: `gpus`, `model_b`, `method` (`sft`/`dpo`/`grpo`), `unfamiliar_model` (bool), `reproducible` (bool). Output: a ranked list of 1–3 frameworks with one-line reasons.

```python
def pick_stack(gpus, model_b, method, unfamiliar_model=False, reproducible=False,
               vlm=False, must_own_weights=True, cost_bound=False):
    out = []
    if gpus == 0:
        return (["Vertex+Gemma / Together / Predibase"] if must_own_weights
                else ["OpenAI (SFT/DPO/RFT)", "Bedrock (governance)", "Vertex"])
    # 1. PARALLELISM GATE — the only hard filter
    if method == "grpo":
        if model_b <= 8:   out += [("trl GRPOTrainer", "≤8B, single node, no rollout fleet needed"),
                                   ("Unsloth GRPO notebooks", "same, faster single GPU")]
        elif gpus >= 8:    out += [("veRL", "Megatron actor + disaggregated vLLM rollout; rollout is 60-85% of step time"),
                                   ("OpenRLHF", "Ray + ZeRO-3 + vLLM, more readable, lower ceiling")]
        else:              out += [("OpenRLHF", "readable RLHF on 4-32 GPUs"),
                                   ("NOT an SFT trainer", "no rollout engine ⇒ 5-10x slower")]
    elif vlm:
        if model_b >= 7:   out += [("ms-swift", "reference for VLM/omni training + Qwen ecosystem"),
                                   ("LLaMA-Factory", "multimodal SFT/DPO, easier first run")]
        else:              out += [("LLaMA-Factory", "breadth + WebUI"), ("Unsloth", "narrow VLM list")]
    elif model_b >= 70 and gpus >= 8:
        out += [("Axolotl + ZeRO-3 + QLoRA", "840 GB unsharded; sharding + 4-bit base is the only arithmetic that fits"),
                ("torchtune FSDP2 + QLoRA", "if the model is in its curated list")]
    elif model_b > 32:     out += [("Megatron (via veRL/ms-swift)", "full FT at 32B-70B needs TP+PP, not just sharding"),
                                   ("ColossalAI Gemini", "three-tier GPU/CPU/NVMe chunk cache, heterogeneous clusters")]
    elif gpus == 1:
        if model_b <= 13 and not unfamiliar_model:
            out += [("Unsloth", "~2x step time, 60-80% less VRAM, TRL-compatible PEFT output")]
        else:
            out += [("LLaMA-Factory", "100+ models, template handled, WebUI for the first run")]
    else:  # 2-16 GPUs, 7-34B
        if reproducible:   out += [("Axolotl", "diffable YAML is the artifact; FSDP/ZeRO first-class")]
        else:              out += [("LLaMA-Factory", "fastest first run, ZeRO-2/3 + FSDP1/2")]
        out += [("torchtune", "PyTorch-native, FSDP2 + torch.compile + torchao float8")]
    if cost_bound:
        out.append(("+ SkyPilot", "use_spot + --retry-until-up + sky jobs for auto-resume (60-70% cheaper)"))
    if model_b <= 13 and gpus <= 1:
        out.append(("NOTE", "below 13B on 1 GPU the framework barely affects quality; pick the one you can debug fastest"))
    return out
```

*Grading notes.* The gate that must appear **first** is the parallelism/method filter — a candidate who starts from popularity, stars or a speed claim fails the question. Full marks require: GRPO → a rollout engine (veRL/OpenRLHF, or trl/Unsloth ≤8B); 70B → sharding + 4-bit; >32B full FT → Megatron (TP+PP), not FSDP2 alone; 1 GPU + unfamiliar model → LLaMA-Factory rather than Unsloth; cost-bound → SkyPilot with the checkpoint-cadence caveat. Bonus for stating the ≤13B "framework barely matters" note unprompted — it is the module's thesis.

**Task 3 — Write the adapter verification and merge script.** Framework-agnostic, works on output from trl, LLaMA-Factory, Axolotl, Unsloth, ms-swift or xTuner. It must refuse to merge if the base does not match.

```python
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, ADAPTER, OUT = "Qwen/Qwen2.5-7B-Instruct", "./out/qwen7b-sft-lora", "./merged/qwen7b-sft"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)

# 1. THE CHECK THAT PREVENTS THE MODULE'S MOST EXPENSIVE LATENT BUG
recorded = model.peft_config["default"].base_model_name_or_path
if recorded != BASE:
    sys.exit(f"REFUSING TO MERGE: adapter trained on {recorded!r}, merging into {BASE!r}")
print(f"base revision matches: {recorded}")

# 2. Merge in bf16 on CPU — never fp16 on GPU (silently rounds the delta)
merged = model.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)   # safe_serialization=True => safetensors
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)

# 3. Keep the template with the artifact: a model is (base revision, adapter hash, template file)
with open(f"{OUT}/chat_template.jinja", "w") as f:
    f.write(AutoTokenizer.from_pretrained(BASE).chat_template or "")

# 4. Smoke-test the round trip in a fresh process before shipping
assert AutoModelForCausalLM.from_pretrained(OUT).generate is not None
# Always keep the PEFT adapter (~100 MB) alongside the merged checkpoint (~14 GB):
# the adapter is what moves between frameworks; the merge exists only for serving stacks.
```

*Grading notes.* Must-have: the `base_model_name_or_path` assertion (this is the question's whole point — the field is **never validated**, so the check must be explicit), bf16-on-CPU merge, `safe_serialization=True`, and keeping the adapter. Deduct for merging without the check, for fp16 on GPU, and for saving `pytorch_model.bin`. Bonus for writing the chat template next to the artifact and for a load-back smoke test.

**Task 4 — Write a SkyPilot job spec for a 30-hour spot run that survives preemption.**

```yaml
# skypilot_ft.yaml — cheapest capacity across clouds, survivable preemption
# sky jobs launch -c qwen-ft skypilot_ft.yaml     # managed job: auto-resume from last ckpt
name: qwen-ft
resources:
  accelerators: {A100-80GB:8}
  cloud: aws,gcp,azure,lambda     # bid across providers
  use_spot: true                  # ~60-70% cheaper, preemptible
  disk_size: 500
  any_of:                         # fallbacks when A100 capacity is gone
    - accelerators: {H100:8}
    - accelerators: {L40S:8}
file_mounts:
  /data: ./data
  /configs: ./configs
setup: |
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
  cd LLaMA-Factory && pip install -r requirements.txt && pip install -e .
run: |
  cd LLaMA-Factory
  llamafactory-cli train /configs/qwen7b_qlora_sft.yaml
# THE CONTRACT THIS FILE DEPENDS ON — in the trainer YAML, not here:
#   save_steps: 250            # NOT save_strategy: epoch; a preemption at hour 22 must lose <250 steps
#   output_dir on the mounted/persistent volume, not container-local disk
#   resume: llamafactory-cli train --resume_from_checkpoint (or Axolotl wandb_run_id)
```

*Grading notes.* Must-have: `use_spot: true`, `--retry-until-up`/managed jobs for auto-resume, `any_of` fallbacks, and — the discriminating point — the explicit note that **the trainer must checkpoint frequently enough to survive preemption**, in the trainer's own config. A candidate who writes only the YAML has answered half the question: the discount is 60–70% *and* the job is survivable only if `save_steps` is set for it. Deduct for `save_strategy: epoch`. Bonus for naming `sky jobs launch` versus `sky launch` and the difference in resume behaviour.

**Task 5 — Whiteboard the export path for each framework, and mark where it stops being interchangeable.**

```
trl            → adapter_config.json + adapter_model.safetensors ─┐
LLaMA-Factory  → same (llamafactory-cli export merges)             │
Axolotl        → same (axolotl merge-lora)                         ├─ PEFT: interchangeable
Unsloth        → same (+ save_pretrained_merged, save_pretrained_gguf)
ms-swift       → same (swift export)                              │
xTuner         → same                                             ┘
torchtune      → PEFT for HF models; native format otherwise      ── partial
LitGPT         → LoRA is PEFT-shaped; own lit_model.pth + json    ── convert via `litgpt convert to_hf`
veRL/OpenRLHF  → full checkpoints, NOT adapters                    ── needs the HF save flag
ColossalAI     → chunked/fp32 sharded, not PEFT                    ── needs zero_to_fp32-style conversion
OpenAI/Bedrock → a model ID                                        ── TOTAL lock-in; nothing to export
Vertex+Gemma
Together
Predibase      → PEFT adapter or merged weights                   ── exportable, low lock-in
```

*Grading notes.* The line that matters is the top block: six different frameworks emitting **byte-identical** PEFT output is the reason framework choice is reversible, and a candidate must be able to say *why* (peft won the format war; the contract is two files). Must-have: hosted-closed providers marked as total lock-in, ColossalAI/veRL marked as non-PEFT without conversion, and the two interop breakers named — **base revision** and **chat template**. Deduct for treating TorchTune/LitGPT as plain PEFT without the conversion caveat.

---

## Numbers To Memorize

| Number | Value | Why it appears |
|---|---|---|
| Full FT bytes/param | **12** (bf16 w2 + g2 + fp32 AdamW 8) | 70B = 840 GB unsharded |
| 70B full FT, ZeRO-2 on 8 GPUs | 3.25 B/param → **227 GB/GPU** | Does not fit |
| 70B full FT, ZeRO-3 on 8 GPUs | 1.5 B/param → **105 GB/GPU** | Still does not fit |
| 70B full FT, ZeRO-3 on 16 GPUs | 0.75 B/param → **52.5 GB/GPU** | Fits with activations |
| 70B QLoRA + ZeRO-3 on 8×80 GB | **under 12 GB/GPU** | The standard 70B recipe |
| QLoRA base cost | **0.55 B/param** | 7B ≈ 4.5 GB + activations on 24 GB |
| 70B DPO on 8×H100 with Axolotl+FSDP2+QLoRA+FA3 | **26–38 h** | The motivating example |
| Same job, ZeRO-3 + CPU offload + QLoRA | ~70–110 h | Possible, impractical |
| Same job, HF Trainer + DDP full FT | **impossible** (1.12 TB/rank) | Rows 1–2 are not slow, they are impossible |
| Unsloth speedup, honest version | **~1.8–2.2×** step time, 60–80% VRAM | Single-GPU QLoRA vs naive HF baseline |
| Unsloth vs tuned FA2+compile baseline | **~1.0–1.3×** | Where the claim evaporates |
| LoRA LR / full-FT LR / DPO LR / GRPO LR | 2e-4 / 2e-5 / 5e-6 / 1e-6 | The most common catastrophic misconfiguration |
| LoRA rank | r=16 default, 8–64 safe | 128 for heavy style transfer; r is a capacity dial |
| `lora_alpha` | 32 (α = 2r) | Effective scale α/r; instability above α/r > 4 |
| Trainable params, 7B QLoRA r=16 | **~0.6–0.8%** | What `print_trainable_parameters()` should show |
| Effective batch in the canonical job | micro 2 × accum 4 = **8** | 5k examples → ~400 steps |
| Canonical 7B QLoRA cost | **~$0.50–0.80** (35–60 min on an L4) | Naive HF loop ~$1.30 |
| Gradient checkpointing | −60–70% activation, **+25–35% step time** | The biggest step-time tax |
| Kernel fusion gain | **1.5–2.5×** step time at scale | Why kernels are a budget line at 64 GPUs |
| Rollout share of RL step time | **60–85%** | Why RL needs a separate sampling engine |
| RL speedup from an integrated vLLM rollout | **5–10×** end-to-end | vs `model.generate()` in the loop |
| Framework switch cost | **1–2 engineer-weeks** | Plus permanent loss of run comparability |
| Quality impact of framework choice below 13B | **±0.5–1 point** (noise) | Data dominates ~10:1 |
| Framework contribution to quality | **<5%** below 13B; ~100% of *feasibility* above | The module's asymmetry |
| Hosted break-even | **5–20M tuned tokens/month** | Below: hosted wins; above: self-host wins |
| Tuned-model inference premium | **1.5–2×** (up to 3–4× for larger) | What flips the sign |
| Spot discount | **60–70%** ($1.20 vs $3.50/GPU-hr) | Contingent on checkpoint cadence |
| Spot-safe checkpoint cadence | **`save_steps: 100–250`** | Never `save_strategy: epoch` |
| SkyPilot clouds | **16+** | Unified execution across clouds and Kubernetes |
| First-run time, LLaMA-Factory vs HF trl | **20–40 min** vs 2–4 h | Why it wins on an unfamiliar model |
| 7B QLoRA peak VRAM | Unsloth **~8–10 GB**; others ~13–16 GB | On a 24 GB card, seq 2048, batch 2 |
| QLoRA vs bf16 LoRA step time at 80 GB | QLoRA **1.4× slower** | 4-bit de-quant buys nothing when memory is not the constraint |
| PEFT adapter size vs merged checkpoint | **~100 MB** vs ~14 GB | Why 50 versions cost 5 GB |
| DeepSpeed config key that saves the run | `stage3_gather_16bit_weights_on_model_save: true` | Otherwise `zero_to_fp32.py` |
| FSDP save rule | `SHARDED_STATE_DICT` during, `FULL` at the end | `FULL` gathers to rank 0 → OOM / 40+ min |
| Case-study anchor: legal LoRA r=32 vs r=16 | **+4 points** domain-vocab precision | Vocabulary-heavy tasks want more rank |
| Case-study anchor: support triage | 71% → 93% accuracy, ~50 min, ~$0.40 | After fixing the chat template |
| Case-study anchor: 32B GRPO | pass@1 41% → 58%, ~60 h, ~$1,100 spot | With a verifiable reward |
| Case-study anchor: DPO 13B | win rate 54% → 68%, length +6% | Acceptable; a monotonic length climb is not |

---

## Answers To The Self-Check Questions From CS-03

1. **Sort into trainer / parallelism / serving / orchestrator.** Trainers: Axolotl, ColossalAI (also parallelism). Parallelism libraries: DeepSpeed, FSDP2, ColossalAI. Serving: vLLM, OpenLLM, FastChat. Orchestrator: SkyPilot. FastChat is additionally an evaluation harness.
2. **1×24 GB, 5k examples — which framework and which three keys set the memory regime?** Unsloth (or LLaMA-Factory for an unfamiliar architecture). Keys: `load_in_4bit=True` (4-bit NF4 base), `bnb_4bit_compute_dtype=torch.bfloat16` (matmuls in bf16), `use_gradient_checkpointing="unsloth"`; plus LoRA `r=16, lora_alpha=32, lora_dropout=0.0`.
3. **Why is `template:` the highest-risk key in any config?** It determines the exact turn structure during training. A wrong template produces a smooth, plausible loss curve and a model that answers as the wrong speaker or in the wrong format — the failure is invisible in the loss and only appears at inference.
4. **70B DPO on 8×H100-80GB: show the bytes-per-param arithmetic.** 70B × 12 B/param (bf16 weights 2 + bf16 grads 2 + fp32 AdamW 8) = 840 GB unsharded → impossible on one 80 GB card. ZeRO-3 across 8 ranks = 105 GB/GPU → still OOM with activations. QLoRA (0.55 B/param base + ~0.08 B/param sharded adapter state) ≈ 40–45 GB total ÷ 8 ≈ 6–8 GB/GPU → fits with room for activations. So: **ZeRO-3 + QLoRA**.
5. **Multi-GPU QLoRA is slower than single-GPU on the same node. Two mechanisms.** (a) ZeRO-3 must all-gather every layer's parameters per forward/backward, and 4-bit weights must be de-quantized before the gather, so you pay both the communication and the de-quantization per layer per step; (b) on PCIe without NVLink that all-gather is slower than the compute it enables, and FSDP1/ZeRO-3 may also lose the single-GPU kernel-fusion benefit if the framework falls back to non-fused paths.
6. **Which frameworks produce a PEFT-format adapter, and what two files constitute the contract?** HF `trl`+`peft`, LLaMA-Factory, Axolotl, Unsloth, ms-swift, xTuner, torchtune (HF models), and Together/Predibase/Vertex-Gemma. The contract is `adapter_config.json` (r, alpha, dropout, target_modules, base_model_name_or_path, task_type) + `adapter_model.safetensors`.
7. **Two reasons framework choice barely affects quality below 13B, and two mechanisms by which it decides feasibility above.** Below: every framework calls the same cross-entropy/DPO loss (identical mathematics), and data quality dominates by ~10:1. Above: optimizer-state sharding decides whether a 70B optimizer state of 560–840 GB fits at all, and kernel fusion plus communication schedule decide a 3× wall-clock difference — and therefore thousands of dollars.
8. **What does GRPO need that SFT does not, and which three frameworks provide it?** A high-throughput **rollout engine** (vLLM/SGLang) to sample groups of completions per prompt, a reward function returning a scalar per completion, and group-normalised advantages. Provided by **veRL**, **OpenRLHF**, and **`trl`** (`GRPOTrainer`; Unsloth ships notebooks on top of it).
9. **Where does DeepSpeed's ZeRO-3 beat FSDP2 in 2026, and vice versa?** DeepSpeed wins on ZeRO-Infinity NVMe offload, mixed TP+PP+ZeRO at very large scale, MoE/expert parallelism, and existing `DeepSpeed-Chat` pipelines. FSDP2 wins on being PyTorch-native (no extra dependency or version pinning), per-parameter DTensor sharding, cleaner `state_dict` semantics, and composing cleanly with `torch.compile`, `torchao` float8, and QLoRA.
10. **A team fine-tuned `gpt-4o-mini` for a support bot and wants to move it on-prem in six months. What is the problem and what should they have done?** The problem: OpenAI returns a model ID, not weights, so there is nothing to move on-prem — the work must be redone. They should have either (a) fine-tuned an open model from the start (Together/Predibase export a PEFT adapter; Vertex+Gemma gives you the weights), or (b) treated the hosted run as a *pilot* with a planned 6–20M-token/month break-even review, and budgeted the re-run.

---

## Cross-References

| Relationship | Module |
|---|---|
| Builds on | **CS-01** (LLM lifecycle), **CS-02** (transfer learning), **CS-23** (LoRA/QLoRA mechanics) |
| This file's source | **CS-03** (framework landscape — all sections) |
| Companion card | **CH-03** (per-framework quick reference, decision tree, CLI surface, interop notes) |
| Needed by | **CS-05** (data prep per framework), **CS-17/18/19** (LLaMA-Factory, Unsloth, Axolotl hands-on), **CS-30/31** (serving the fine-tune), **CS-34** (evaluation) |
| Contrasts with | **CS-04** (fine-tune vs RAG vs prompting — decide *whether* before *how*), **CS-10/11** (inference quantization: GPTQ/AWQ/GGUF vs training-time 4-bit) |
| Pairs with | **CS-25** (DPO), **CS-26** (GRPO/RLVR), **CS-27** (ORPO) for which framework exposes which method; **CS-12** (continued pretraining) for the CPT rows |
| Sibling interview banks | **IQ-01** (foundations), **IQ-02** (transfer learning), **IQ-04+** |

