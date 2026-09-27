# CH-03 — Fine-Tuning Framework Landscape Cheat Sheet

**One-line purpose:** Pick a stack in 60 seconds, translate a config between any two frameworks, and know which CLI does the merge.
**Use when:** Choosing a framework, moving a config or adapter between frameworks, sizing a multi-GPU run, or answering "which one would you use for X?" under interview pressure.
**Do NOT use when:** You need the reasoning behind a recommendation — that is CS-03. This card gives the answer, not the argument.

> Pairs with **CS-03** (case study) and **IQ-03** (interview bank). Deep dives: **CS-23** (LoRA/QLoRA), **CS-25/26/27** (DPO/GRPO/ORPO), **CS-30/31** (serving), **CS-34** (evaluation).

---

## 1. The 10-Second Summary

| # | Fact |
|---|---|
| 1 | **Filter on parallelism first.** It is the only hard gate; methods/UI/speed are soft. |
| 2 | **Below 13B on one GPU, framework choice barely matters** (±0.5–1 point, data dominates 10:1). |
| 3 | **Above 13B the framework decides whether the job exists at all** — 70B full FT is 840 GB unsharded. |
| 4 | **Pick exactly two:** one learning framework (`transformers`+`trl`+`peft`) and one production framework. |
| 5 | **PEFT adapters are the interop currency** — `adapter_config.json` + `adapter_model.safetensors`. |
| 6 | **`template:` is the #1 silent failure** — perfect loss curve, model answers as the wrong speaker. |
| 7 | **Unsloth = ~2× / −60–80% VRAM**, but only vs a *naive HF single-GPU QLoRA baseline*. |
| 8 | **Axolotl is tier 1 for teams**, tier 2 for solo learners — the video has this backwards. |
| 9 | **SkyPilot is the only entry that addresses cost**, and cost is the biggest line item. |
| 10 | **Switching frameworks costs 1–2 engineer-weeks** and permanently kills run comparability. |

---

## 2. Sort The Layers Before Anything Else

The video's "Top 10 fine-tuning frameworks" mixes four different layers. Sort first or you will adopt a serving engine expecting a trainer.

| Layer | Tools | What it does | What it does NOT do |
|---|---|---|---|
| **Trainer** | HF `transformers`+`trl`+`peft`, LLaMA-Factory, Unsloth, Axolotl, torchtune, LitGPT, ms-swift, xTuner, ColossalAI (`ColossalChat`) | Runs the training loop, writes the adapter | Shard for you, serve, or rent GPUs |
| **Parallelism library** | DeepSpeed, FSDP2 (`torch.distributed`), ColossalAI (`Gemini`) | Shards params/grads/optimizer, offloads, TP/PP/CP/EP | Implements any loss; is not a trainer |
| **Serving engine** | vLLM, SGLang, TGI, OpenLLM, LightLLM, FastChat (serving) | Serves the merged model or adapters | Train anything |
| **Evaluation harness** | FastChat (MT-Bench, LLM-as-judge), LightEval | Judges model-vs-model | Train or serve |
| **Orchestrator** | SkyPilot, Kubernetes, Slurm | Finds and provisions cheap capacity, survives preemption | Train anything |
| **Hosted service** | OpenAI, Vertex AI, Bedrock, Together, Predibase | Fine-tuning *as a service* — you upload JSONL | Give you weights (mostly) |

**The five-line sanity check:** DeepSpeed is not a trainer (it lives *inside* them). SkyPilot is not a trainer (it launches them). OpenLLM/FastChat/LightLLM are not trainers (they run *after* them). DeepSpeed ↔ FSDP ↔ ColossalAI ↔ Megatron is the real comparison group, **not** DeepSpeed ↔ TRL.

---

## 3. The Three Axes

**Axis 1 — Abstraction level:**

```
less code  ◄───────────────────────────────────────────────►  more control
 hosted API        GUI/YAML            config+python        raw python
 OpenAI/Vertex/    LLaMA-Factory       Axolotl/LLaMA-       transformers +
 Bedrock/Together  WebUI, AutoTrain    Factory CLI,         trl + peft +
 /Predibase        ms-swift web-ui     torchtune YAML       accelerate/DeepSpeed
                                                           LitGPT/Lightning
```

**Axis 2 — Memory regime:**

| Regime | Weights B/param | Optimizer B/param | Fits |
|---|---|---|---|
| Full FT fp32 + AdamW | 4 | 8 | ≤1.5B/GPU |
| Full FT bf16 + AdamW | 2 | 8 | ≤3B/GPU |
| Full FT bf16 + 8-bit AdamW | 2 | 2 | ~7B/GPU |
| LoRA bf16 (no quant) | 2 | ~0.08 | ~7–13B/GPU |
| **QLoRA (NF4 + LoRA)** | **~0.55** | ~0.08 | **~7B on 24 GB** |
| QLoRA + ZeRO-3 + offload | 0.5, sharded | sharded | 70B on 4×80 GB |

**Axis 3 — Parallelism strategy:**

| Strategy | Shards what | Comm | When |
|---|---|---|---|
| **DDP** | nothing (replicates grads) | all-reduce grads, 1×/step | model + optimizer state fit on 1 GPU |
| **ZeRO-1/2/3** | optimizer / +grads / +params | all-gather params per layer (Z3) | model fits only sharded |
| **FSDP1 / FSDP2** | params+grads+optim, per-layer / per-param | all-gather per layer | same as ZeRO-3, native, simpler deps |
| **TP** | weight matrices | all-reduce per layer, high | single node, ≤8 GPUs |
| **PP** | layer ranges | point-to-point | multi-node, large depth |
| **CP / SP** | activations along seq | all-gather/reduce-scatter | long context (>8k) |
| **HSDP** | FSDP hybrid: shard in node, replicate across | lower cross-node traffic | multi-node FSDP2 |

**Where the axes collide:** QLoRA + ZeRO-3 are not free together — 4-bit weights must be **de-quantized to bf16 to be all-gathered**, so you pay de-quant cost per layer per step. That is the answer to "why is my multi-GPU QLoRA slower than single-GPU?"

---

## 4. Per-Framework Quick Reference

`△` = possible with effort / not a primary goal. `—` = not applicable (wrong layer).

| Framework | Abstraction | Methods (SFT / pref / RL) | Quantization | Multi-GPU | GUI | License | Best for |
|---|---|---|---|---|---|---|---|
| **HF `transformers`+`trl`+`peft`** | Python API + CLI (`accelerate launch`, `trl sft`) | SFT ✓ / DPO, ORPO, KTO, SimPO, CPO ✓ / PPO, GRPO, RLOO ✓ / RM ✓ / CPT ✓ / distillation ✓ | bnb 4/8-bit NF4+FP4, QLoRA, GPTQ, AWQ, HQQ, EETQ, quanto, torchao | DDP, FSDP1/**FSDP2**, DeepSpeed Z1–3 (+offload), TP via `tp_plan`, SP | — | Apache-2.0 | The reference layer; interop hub; custom losses; every architecture |
| **LLaMA-Factory** | CLI + YAML + WebUI + Python API | CPT, SFT, RM, PPO, DPO, KTO, ORPO, SimPO, GRPO, VLM SFT/DPO | bnb 4/8, GPTQ, AWQ, AQLM, HQQ, EETQ, BAdam, GaLore, LoRA+, DoRA, PiSSA, LongLoRA | DDP, ZeRO-2/3, FSDP1/2, Ray multi-node | **✓✓ LLaMA Board** (`llamafactory-cli webui`) | Apache-2.0 | Breadth: 100+ models, one-stop, no-code, fastest first run |
| **Unsloth** | Python API + notebooks (`FastLanguageModel`/`FastModel`) | SFT, CPT, DPO, ORPO, KTO, SimPO, GRPO, vision SFT | bnb 4/8-bit, dynamic 4-bit, GGUF export | DDP/FSDP for *some* archs (2025+); historically single-GPU — **verify per model** | — | Apache-2.0 | Single-GPU QLoRA where wall clock/VRAM binds (Colab/T4/4090) |
| **Axolotl** | CLI + YAML (`axolotl train x.yaml`) | CPT, SFT, DPO, KTO, ORPO, GRPO, RM, RLHF-via-TRL | bnb 4/8-bit, GPTQ training, FSDP+QLoRA | DDP, ZeRO-1/2/3 (+offload), FSDP1 (FULL_SHARD / SHARD_GRAD_OP / HYBRID_SHARD), multi-node | — | Apache-2.0 | Diffable, reproducible, **multi-GPU post-training**; config as artifact |
| **torchtune** | CLI (`tune run <recipe>`) + hackable Python recipes | Full FT, LoRA, QLoRA, DoRA, DPO, PPO, GRPO, KD, VLM | bnb NF4, torchao int8/int4/**float8**, QAT | **FSDP2** (DTensor), TP, torch.compile | — | BSD-3-Clause | PyTorch-native teams; float8; best-documented memory numbers |
| **LitGPT** | CLI (`litgpt finetune lora`) + Python recipes | Pretrain, CPT, SFT, LoRA, QLoRA, Adapter v1/v2, Adapter-LoRA, DPO, distillation | bnb 4/8-bit, GGUF/GPTQ export | DDP, FSDP1, ZeRO-1/2/3, Fabric | — | Apache-2.0 | Learning internals; pretraining a small model from scratch |
| **ms-swift / SWIFT** | CLI + YAML/JSON + Python + WebUI | CPT, SFT, RM, PPO, DPO, GRPO, ORPO, SimPO, KTO, CPO, **MLLM SFT/DPO/GRPO** | bnb, GPTQ, AWQ, AQLM, HQQ, EETQ, FP8, **QAT**, GaLore, LoRA+, DoRA, ReFT, LISA | DDP, ZeRO-2/3, FSDP1/2, **Megatron TP+PP+CP+EP**, Ray | ✓ `swift web-ui` | Apache-2.0 | **VLM/multimodal; Qwen ecosystem; MoE at scale** |
| **xTuner** | Python config + CLI (`xtuner train cfg.py`) | SFT, DPO, ORPO, RM, VLM SFT, tool/agent SFT | bnb 4-bit QLoRA, LoRA, LoRA+, DoRA, DeepSpeed offload | DDP, ZeRO-1/2/3 (+offload) | △ | Apache-2.0 | 1–8B single GPU; InternLM/InternVL family |
| **ColossalAI** | Python API + CLI | SFT, RM, PPO (`ColossalChat`); large-scale pretraining | int8/int4 inference quant (GPTQ-style) | **DP+TP+PP+SP+EP, ZeRO 1–3, Gemini offload, auto-parallel** | △ demo UI | Apache-2.0 | Cluster-scale pretraining; studying parallelism internals |
| **veRL** | Hydra YAML + Python entry points | SFT, RM, PPO, **GRPO, DAPO, RLOO, REINFORCE++**, multi-turn/tool RL, VLM RL | fp8 rollout; **QLoRA not first-class** | FSDP1/2 or **Megatron TP+PP+CP+EP** + **disaggregated vLLM/SGLang rollout** | — | Apache-2.0 | **Reasoning RL/RLVR at scale** |
| **OpenRLHF** | CLI + Python scripts (`ray job submit`) | SFT, RM (outcome+process), PPO, **GRPO, REINFORCE++**, DPO/KTO/ORPO, iterative DPO, agent RL | ZeRO offload, 8-bit Adam, vLLM-side quant | **Ray + ZeRO-3 + FSDP + vLLM TP**, colocated or disaggregated | — | Apache-2.0 | Readable/customisable RLHF on 4–32 GPUs |
| **DeepSpeed** | JSON config consumed by `accelerate`/Trainer/others | — (library); `DeepSpeed-Chat` adds SFT+RM+PPO | FP6, MoQ; not a quant front-end | **ZeRO 1/2/3, Offload, Infinity (NVMe), ZeRO++, TP, PP, EP, AutoTP** | — | Apache-2.0 | Sharding what nothing else fits; NVMe offload |
| **FSDP2** | Python API (library) | — (library) | composes with QLoRA and torchao | Per-parameter sharding (DTensor), HSDP, TP via DTensor, CPU offload | — | BSD-3-Clause | The 2026 default sharding backend for new code |
| **SkyPilot** | YAML job spec + CLI (`sky launch`, `sky jobs launch`) | — (orchestration) | — | multi-node, multi-cloud, **spot with auto-resume** | ✓ `sky dashboard` | Apache-2.0 | **Cheapest capacity + preemption recovery** |
| **OpenLLM** | CLI + Python SDK | — (serving) | AWQ/GPTQ/int8 | replica/TP serving | △ chat UI | Apache-2.0 | **Multi-LoRA serving**, packaged deploys |
| **FastChat** | CLI + Python scripts | SFT (LoRA, dated) + serving + **eval** | int8 serving | DDP (train), TP/DP (serve) | ✓ arena UI | Apache-2.0 | **MT-Bench / LLM-as-judge**; multi-model serving |
| **LightLLM** | Python API + HTTP server | — (**serving only**) | AWQ, GPTQ, FP8 | TP/DP serving | — | Apache-2.0 | Low-VRAM high-throughput serving of small/medium dense models |
| **Hosted APIs** | Console + REST | SFT ✓ / DPO provider-specific / RL rare / CPT (Nova) | opaque | opaque | ✓ console | proprietary | No GPU team, low volume, latency-critical, style/format tasks |

**How to read this table in 30 seconds.** `Methods`, `Quantization`, `GUI` and learning curve vary enormously below 7B and converge to "basically all the same" above. **`Parallelism` is the only hard gate**: a framework that cannot shard will not run your 70B job no matter how good its UI is. Procedure: filter on parallelism → filter on methods → pick on ergonomics among survivors.

---

## 5. Stack-Selection Decision Tree

```
START — how many GPUs do you have, and how big is the model?

A. NO GPU (or no ops capacity)
   must own weights? ──► Vertex+Gemma / Together / Predibase  (exportable PEFT)
   otherwise         ──► OpenAI (SFT/DPO/RFT) | Bedrock (governance) | Vertex

B. 1 GPU (16–24 GB), ≤13B, SFT/DPO
   known architecture ──► UNSLOTH              (~2× step, 60–80% less VRAM)
   unfamiliar or VLM  ──► LLaMA-FACTORY        (100+ models, template handled, WebUI)
   want to learn the mechanics ──► HF transformers + trl + peft

C. 1–2 GPUs (24–48 GB), ≤32B, must be reproducible
   config as the artifact ──► AXOLOTL
   fastest first run / UI ──► LLaMA-FACTORY

D. 4–16 GPUs, 7B–34B, SFT → DPO
   reviewable YAML        ──► AXOLOTL + FSDP2 or ZeRO-2
   breadth / no-code      ──► LLaMA-FACTORY + ZeRO-3
   PyTorch-native, float8 ──► TORCHTUNE + FSDP2

E. 8×H100, 70B, DPO  ──► AXOLOTL + ZeRO-3 + QLoRA   (~26–38 h)
   (NOT veRL — DPO is not online RL and needs no rollout engine)
   torchtune FSDP2 + QLoRA if the model is in its list

F. 8×H100, 70B, GRPO on verifiable rewards ──► veRL (Megatron actor + vLLM rollout)
   readable/customisable alternative       ──► OpenRLHF (Ray + ZeRO-3 + vLLM)
   ≤8B                                     ──► trl GRPOTrainer | Unsloth GRPO notebooks
   NEVER an SFT-only trainer — no rollout engine = 5–10× slower

G. 32B–70B FULL fine-tune, multi-node ──► MEGATRON (via veRL / ms-swift)
   FSDP2 alone is NOT enough — you need TP + PP and 1.1 TB of optimizer state

H. Multi-modal / VLM ──► ms-swift (reference) | LLaMA-Factory (easier first run)
I. Qwen / InternLM on Chinese cloud or NPU ──► ms-swift | xTuner | LLaMA-Factory
J. Cost is the binding constraint ──► any trainer + SKYPILOT spot (use_spot + sky jobs)

STOP CONDITIONS — you have the wrong framework:
  • you are about to write a custom distributed training loop   → FSDP2/DeepSpeed solved it
  • the README says "inference" or "serving" first              → OpenLLM/LightLLM/FastChat
  • you need multi-node and the multi-GPU section is 1 paragraph → Unsloth >2 GPUs, xTuner >4
  • you cannot state params × bytes/param + activations         → no framework fixes a bad GPU
  • you picked a GUI because you don't want to learn the config  → you will, under deadline
  • your reward function is a string match                       → fix the reward first
```

**The five canonical answers (memorise these, they are the interview):**

| Constraint | Answer | Why |
|---|---|---|
| 1×24 GB + 5k examples | **Unsloth** (LLaMA-Factory if unfamiliar) | ~2× step, 60–80% less VRAM, TRL-compatible PEFT out |
| 1×80 GB + 50k examples, 13B | **Axolotl, bf16 LoRA** | Reproducible YAML; no 4-bit needed at 80 GB |
| 4×A100 + 5k pairs, 13B, DPO | **Axolotl or LLaMA-Factory + FSDP2/ZeRO-2** | DPO needs actor + reference; sharding mandatory |
| 8×H100 + GRPO on 32B | **veRL** (Megatron actor + vLLM rollout) | Rollout is 60–85% of step time |
| 8×H100 + DPO on 70B | **Axolotl + ZeRO-3 + QLoRA** | 840 GB unsharded; sharding + 4-bit base is the only fit |

---

## 6. CLI Quick Reference

**Hugging Face (`transformers` + `trl` + `peft`)**
```bash
pip install "transformers>=4.48" "trl>=0.14" "peft>=0.14" "bitsandbytes>=0.44" datasets accelerate
accelerate launch train.py                       # multi-GPU / FSDP / DeepSpeed entry point
accelerate config                                # writes the distributed config interactively
trl sft --model_name_or_path Qwen/Qwen2.5-7B-Instruct --dataset_name <ds>   # CLI path
python train.py                                  # single GPU
# Output: adapter_config.json + adapter_model.safetensors
```

**LLaMA-Factory**
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
pip install -r requirements.txt && pip install bitsandbytes>=0.39.0 && pip install -e .

llamafactory-cli train  qwen7b_qlora_sft.yaml        # the run
llamafactory-cli webui                               # LLaMA Board (Gradio, LOCAL — ssh -L 7860:localhost:7860)
llamafactory-cli chat   qwen7b_qlora_sft.yaml        # interactive chat with the adapter
llamafactory-cli export export_qwen7b.yaml           # merge LoRA -> safetensors
llamafactory-cli api    qwen7b_qlora_sft.yaml        # OpenAI-compatible server (vLLM backend)
```

**Axolotl**
```bash
pip install --no-build-isolation git+https://github.com/OpenAccess-AI-Collective/axolotl.git
pip install --no-build-isolation "axolotl[flash-attn]>=0.9.1"

accelerate launch -m axolotl.cli.train custom-config.yaml   # canonical (multi-GPU-ready)
axolotl train custom-config.yaml                            # convenience wrapper
axolotl inference custom-config.yaml --lora-model-dir ./outputs/sft-lora
axolotl merge-lora custom-config.yaml                       # merge adapter into base
# NOTE: FA2 does not build on T4 (sm_75). Use attn_implementation: sdpa or xformers.
```

**Unsloth**
```bash
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
pip install unsloth
pip install transformers==4.56.2        # PINNED — Unsloth patches specific internals
pip install --no-deps trl==0.22.2      # --no-deps is deliberate
pip install psutil
```
```python
model.save_pretrained("lora_out")                                             # PEFT adapter
model.save_pretrained_merged("merged_16bit", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("gguf_out", tokenizer, quantization_method="q4_k_m")
```

**torchtune**
```bash
tune run lora_finetune_single_device --config llama3_2/3B_lora_single_device
tune run --nnodes 1 --nproc_per_node 8 fsdp2_lora_finetune_distributed --config <cfg>
tune ls                                              # list available recipes and configs
tune cp llama3_2/3B_lora_single_device ./my_config.yaml   # fork a config
```

**LitGPT**
```bash
litgpt download meta-llama/Llama-3.2-3B-Instruct
litgpt finetune lora --checkpoint_dir checkpoints/meta-llama/Llama-3.2-3B-Instruct
litgpt convert from_hf <model> --checkpoint_dir <dir>    # HF checkpoint -> LitGPT format
litgpt convert to_hf --checkpoint_dir <dir>              # back out for serving
litgpt chat --checkpoint_dir <dir>
```

**ms-swift**
```bash
pip install ms-swift -U
swift sft    --model Qwen/Qwen2.5-7B-Instruct --dataset <ds> --train_type lora
swift rlhf   --rlhf_type grpo --model <m> --reward_funcs <fn>
swift infer  --adapters output/...
swift export --adapters output/... --merge_lora true
swift web-ui                                          # Gradio training/export/chat UI
```

**xTuner**
```bash
xtuner train config.py                       # a PYTHON config, not YAML
xtuner chat <model> --adapter <path> --prompt-template internlm2_chat
xtuner convert pth_to_hf ./config.py <pth> <hf_dir>    # checkpoint conversion
```

**DeepSpeed (as consumed by others)**
```bash
# point the host trainer at a JSON config — not a standalone CLI
accelerate launch --config_file fsdp.yaml train.py
deepspeed --num_gpus=8 train.py --deepspeed ds_config_zero3.json
deepspeed zero_to_fp32.py <ckpt_dir> <output>          # recovery for missing gather flag
```

**veRL**
```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  trainer.n_gpus_per_node=8 trainer.nnodes=1
```

**OpenRLHF**
```bash
ray job submit --address="http://127.0.0.1:8265" -- \
  python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 --reward_num_nodes 1 --critic_num_nodes 1 --actor_num_nodes 1 \
    --vllm_num_engines 4 --vllm_tensor_parallel_size 2 \
    --pretrain <model> --reward_pretrain <rm>
```

**SkyPilot**
```bash
sky launch      -c qwen-ft skypilot_ft.yaml --retry-until-up -i 60   # provision + run
sky jobs launch -c qwen-ft skypilot_ft.yaml                          # MANAGED: auto-resume on preemption
sky status                     # cluster/job state
sky logs qwen-ft               # tail logs
sky down qwen-ft               # tear down
sky dashboard                  # local web UI
```

---

## 7. Config Key Rosetta Stone

One identical job — Qwen2.5-7B-Instruct, 5k Alpaca examples, 1 epoch, QLoRA r=16, seq 2048 — in four frameworks. **Every row is the same computation with a different spelling.**

| Concept | HF `trl` | LLaMA-Factory | Axolotl | torchtune |
|---|---|---|---|---|
| Base model | `model_name_or_path=` | `model_name_or_path:` | `base_model:` | `model._component_:` + `checkpoint_dir` |
| Method | `SFTConfig` / `SFTTrainer` | `stage: sft` | `training_type: sft` | recipe name (`lora_finetune_single_device`) |
| Adapter | `LoraConfig(r=16,...)` | `finetuning_type: lora` + `lora_rank: 16` | `adapter: qlora` + `lora_r: 16` | `peft._component_: LoRAConfig` |
| 4-bit | `BitsAndBytesConfig(load_in_4bit=True)` | `quantization_bit: 4` | `load_in_4bit: true` | `quantizer._component_: BitsAndBytesQuantizer` |
| Seq length | `max_length=2048` | `cutoff_len: 2048` | `sequence_len: 2048` | `tokenizer.max_seq_len: 2048` |
| Dataset | `load_dataset(...)` | `dataset: <name>` + `dataset_info.json` | `datasets: [{path:..., type: alpaca}]` | `dataset._component_: AlpacaDataset` |
| **Template** | tokenizer's `chat_template` | **`template: qwen`** | **`chat_template: qwen`** | tokenizer component's template |
| LR | `learning_rate=2e-4` | `learning_rate: 2e-4` | `learning_rate: 2e-4` | `optimizer.lr: 2e-4` |
| Batch | `per_device_train_batch_size` | `per_device_train_batch_size` | `micro_batch_size` | `batch_size` |
| Grad accum | `gradient_accumulation_steps` | `gradient_accumulation_steps` | `gradient_accumulation_steps` | `gradient_accumulation_steps` |
| Epochs | `num_train_epochs` | `num_train_epochs` | `num_epochs` | `epochs` |
| Optimizer | `optim=` | `optim:` | `optimizer:` | `optimizer._component_:` |
| Save cadence | `save_steps` / `save_strategy` | `save_steps` / `save_strategy` | `save_steps` | `checkpointer.` keys |
| Multi-GPU | `accelerate launch` + `--fsdp`/`--deepspeed` | `deepspeed: ds_z3.json` | `deepspeed:` or `fsdp:` | `tune run --nnodes 1 --nproc_per_node 8 fsdp2_...` |
| Launch | `python train.py` | `llamafactory-cli train x.yaml` | `axolotl train x.yaml` | `tune run <recipe> --config x.yaml` |
| Output | `adapter_config.json` + `adapter_model.safetensors` | same (PEFT) | same (PEFT) | same (PEFT) |
| Merge | `merge_and_unload()` | `llamafactory-cli export` | `axolotl merge-lora` | convert recipe |

**Key values that transfer unchanged across every framework:** LR by method (LoRA 2e-4 / full FT 2e-5 / DPO 5e-6 / GRPO 1e-6), `r` (16), `alpha` (32), `dropout` (0.05; 0.0 for Unsloth), warmup (3% LoRA / 6–10% full FT), epochs (1–3 SFT, 1–2 DPO), effective batch (8–64), bf16 on Ampere+, gradient checkpointing on when memory-bound, grad clipping 1.0.

**Custom-dataset registration (LLaMA-Factory) — the thing that actually blocks beginners:**
```json
{
  "my_alpaca_5k": {
    "format": "alpaca",
    "path": "data/my_dataset/my_data.json",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  },
  "my_sharegpt": {
    "format": "sharegpt",
    "path": "data/my_dataset/my_data.json",
    "columns": { "messages": "conversations" }
  }
}
```

---

## 8. Memory & Sharding Cheat

**The formula:**
```
VRAM ≈ w_bytes×Ψ + (o_bytes + g_bytes)×trainable_Ψ
     + activations(seq_len, batch, hidden, layers) × checkpointing_factor
     + framework overhead (CUDA context, cuBLAS workspaces, logits buffer)
```

| Regime | Weights B/param | Optimizer | Grad | 7B total | 70B total |
|---|---|---|---|---|---|
| Full FT fp32 + AdamW | 4 | 8 | 4 | 112 GB | 1.12 TB |
| Full FT bf16 + AdamW | 2 | 8 | 2 | 84 GB | **840 GB** |
| Full FT bf16 + 8-bit AdamW | 2 | 2 | 2 | 42 GB | 420 GB |
| LoRA bf16 r=16 | 2 | ~0.08 | ~0.08 | ~15 GB + act | ~155 GB + act |
| **QLoRA (NF4 + bf16 LoRA)** | **0.55** | ~0.08 | ~0.08 | **~4.5 GB + act** | **~42 GB + act** |

**The 70B worked ladder (the one to reproduce on a whiteboard):**
```
70B × (2 w + 2 g + 8 optim) = 12 B/param = 840 GB unsharded
  ZeRO-2 on  8×80 GB → 2 + 2/8 + 8/8 = 3.25 B/param → 227 GB/GPU  ✗ does not fit
  ZeRO-3 on  8×80 GB → 12/8 = 1.5 B/param           → 105 GB/GPU  ✗ still no (activations need 15–25 GB)
  ZeRO-3 on 16×80 GB → 12/16 = 0.75 B/param         →  52.5 GB/GPU ✓ ≈70 GB with activations
  ZeRO-3 + QLoRA on 8×80 GB → base 0.55×70=38.5 ÷8 ≈ 4.8 GB + sharded adapter → <12 GB/GPU ✓
```
**Conclusion: 70B full FT on 8×80 GB is impossible; 70B QLoRA on 8×80 GB is routine.**

**Sharding strategy picker:**

| Symptom / size | Use |
|---|---|
| Model + optimizer state fit on 1 GPU | DDP |
| Fits only sharded, single node | FSDP2 or ZeRO-2 |
| Does not fit even sharded at 8 GPUs | ZeRO-3 with 16 GPUs, or QLoRA |
| Multi-node | FSDP2 + HSDP, or ZeRO-3 |
| Very large / MoE, latency-bound | TP (+PP) via Megatron; 3D parallelism |
| Long context (>8k) | CP / sequence parallel |
| PCIe, no NVLink | ZeRO-2 (not ZeRO-3), `offload_param: cpu`, or FSDP2 |
| NVMe needed as a third memory tier | DeepSpeed ZeRO-Infinity, or ColossalAI Gemini |

**The config keys that decide whether your run is salvageable:**

| Key | Where | Value | Why |
|---|---|---|---|
| `stage3_gather_16bit_weights_on_model_save` | DeepSpeed JSON | `true` | Without it, ZeRO-3 checkpoints are unloadable (0-byte symptom) |
| `stage3_param_persistence_threshold` | DeepSpeed JSON | `1e5` | Too high → OOM despite ZeRO-3 |
| `overlap_comm`, `contiguous_gradients` | DeepSpeed JSON | `true` | Throughput |
| `fsdp_state_dict_type` | accelerate/Axolotl | `SHARDED_STATE_DICT` during, `FULL` only at the end | `FULL` gathers to rank 0 → OOM or 40+ min per save |
| `fsdp_activation_checkpointing` | accelerate | `true` | Activation memory |
| `fsdp_cpu_ram_efficient_loading` | accelerate | `true` | Load shard-by-shard instead of all on rank 0 |
| `gradient_accumulation_steps` | YAML **or** DeepSpeed JSON | set in **one** place | Some versions multiply the two |
| `save_steps` | trainer YAML | `100–250` on spot | `save_strategy: epoch` loses the run on preemption |
| `attn_implementation` | model load | `flash_attention_2` (Ampere+), `sdpa` (T4/V100), `eager` (debugging) | FA2 does not build on sm_75 |

---

## 9. Adapter Interop & Migration

**The contract — the entire interface between frameworks:**
```json
// ./out/qwen7b-sft-lora/adapter_config.json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "bias": "none",
  "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
  "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "revision": null,
  "use_rslora": false,
  "modules_to_save": null
}
```
`base_model_name_or_path` is **never validated on load**. A mismatch loads cleanly and degrades subtly — the module's most expensive latent bug.

| Framework | Produces | PEFT? | Lock-in risk |
|---|---|---|---|
| HF `trl`+`peft` | PEFT adapter | **Reference implementation** | None |
| LLaMA-Factory | PEFT (+ merged, + GGUF via export) | ✓ byte-identical | Low |
| Axolotl | PEFT | ✓ | Low |
| Unsloth | PEFT (+ merged, + GGUF) | ✓ | Low |
| ms-swift | PEFT | ✓ | Low |
| xTuner | PEFT | ✓ | Low |
| torchtune | PEFT on HF models; native otherwise | ✓ for HF models | Medium |
| LitGPT | LoRA is PEFT-shaped; own `lit_model.pth` + `lit_config.json` | ✓ via `litgpt convert to_hf` | Medium |
| veRL / OpenRLHF | Full checkpoints, **not adapters** | ✓ only with the HF save flag | **High** |
| ColossalAI (`Gemini`) | Chunked/fp32 sharded | ✗ without `zero_to_fp32`-style conversion | **High** |
| Hosted: OpenAI / Bedrock / Vertex-Gemini | **A model ID** | ✗ | **Total** |
| Hosted: Together / Predibase / Vertex-Gemma | PEFT adapter or merged weights | ✓ | Low |

**What breaks a move:**

| Move | Works? | What breaks |
|---|---|---|
| Unsloth → HF `trl` inference | Always | Nothing |
| Axolotl → vLLM serving | Always (merge, or `--enable-lora` + pinned base revision) | Serving against a different base **revision** |
| LLaMA-Factory → ms-swift | Both PEFT | `template` / `model_type` must match what you trained on |
| torchtune → Axolotl | HF-model recipes only | Native-format checkpoints are not PEFT |
| veRL full checkpoint → anything | Only with `save_hf_weights`-equivalent | Adapter tooling expects ~100 MB, not 14 GB |
| Hosted (OpenAI) → anything | **Never** | There is no artifact |
| Any → GGUF | Via `save_pretrained_gguf` or llama.cpp convert | LoRA→GGUF needs the merge first + a template mapping |

**The four portability rules that prevent 90% of migration pain:**
1. Record the exact `base_model_name_or_path` **including the revision hash**.
2. Keep the raw PEFT adapter (~100 MB) alongside any merged checkpoint (~14 GB) — the adapter is what moves.
3. Keep the tokenizer and the exact chat template as a file in the run directory.
4. Never delete the merged model, but treat the adapter as the **source of truth**.

**Versioning rule:** a model is **not a file** — it is the triple `(base revision, adapter hash, template file)`. Tag them together in the registry, and log that triple at every serving endpoint.

---

## 10. Symptom → Fix

| Symptom | Cause | Fix |
|---|---|---|
| Loss flat, `print_trainable_parameters()` shows 0.000% | `target_modules` matched nothing | Print `model`; copy exact `*_proj`; or `all-linear` / `lora_target: all` |
| Loss descends perfectly, model **answers as the user** | Wrong chat template | Print the rendered training string; set `template:`/`chat_template:` to the exact family |
| Loss NaN at step 4 | fp16 compute dtype on a bf16 model; or full FT at LoRA's 2e-4 | `bnb_4bit_compute_dtype=bf16`, `bf16=True`, full-FT LR 2e-5 |
| Loss → 0.0 very fast | Prompt tokens not masked | Inspect a collated batch for `-100` in the prompt region; `assistant_only_loss` |
| Loss ~0.3 and dropping, model emits `<pad>` | Padding not masked | `ignore_pad_token_for_loss: true`; set pad token deliberately |
| Both losses great, outputs bad | Wrong template (see #2) or eval on train data | Same fix; hold out a frozen set |
| OOM at step 0 | Optimizer state, not weights | Compute `params × bytes/param`; shard, QLoRA, 8-bit Adam, shorter seq |
| Multi-GPU **slower** than single GPU | ZeRO-3 all-gather per layer + 4-bit de-quant, over PCIe | ZeRO-2, `offload_param: cpu`, FSDP2 — or stay at 1 GPU |
| Checkpoint dir 0 bytes / unloadable | `stage3_gather_16bit_weights_on_model_save` missing | Set `true`, or run `zero_to_fp32.py` |
| Throughput drops after step ~100 | Data-loading stall or blocking checkpoint write | `preprocessing_num_workers`, `dataloader_num_workers`, async save |
| Merged model worse than the adapter | fp16 merge of a bf16 adapter; wrong base revision | Merge bf16 on CPU; verify `base_model_name_or_path` |
| Served model ≠ evaluated model | Served the base, or a different revision | Merge, or `--enable-lora` + explicit base revision; log the hash |
| FlashAttention import/build error | T4/V100 (sm_75), or torch/CUDA mismatch | `attn_implementation: sdpa` or `xformers` |
| `bitsandbytes` import error | Windows, or CPU-only torch | Linux/WSL2; install CUDA torch first |
| Reward rises, quality falls | Exploitable reward (string match, length proxy) | Redesign reward; add length/KL penalty; held-out judge as a **gate** |
| Every sample in a batch identical | Packing with wrong logic, or deduplicated data | `packing: false` for chat data; inspect decoded rows |
| Run not reproducible | Seed unset; nondeterministic kernels; different GPU count | Fix `seed`, `data_seed`, `full_determinism`; record GPU count |
| `RuntimeError` on resume | Cross-framework optimizer/scheduler mismatch | Resume in the same framework, or restart |
| Loss fine, but no new **facts** learned | LoRA cannot inject knowledge | CPT or full FT (see CS-12) |
| Run is in the wrong W&B project | `report_to=["wandb"]` without `WANDB_PROJECT` | Set the project; check offline mode actually syncs |

---

## 11. Numbers To Memorize

| Number | Value |
|---|---|
| Full FT bytes/param | **12** (bf16 w2 + g2 + fp32 AdamW 8) |
| Full FT practical rule | **~16–20 B/param** including activations |
| 70B full FT unsharded | **840 GB** (1.12 TB in fp32) |
| 70B ZeRO-3 on 8×80 GB | 105 GB/GPU — **does not fit** |
| 70B ZeRO-3 on 16×80 GB | 52.5 GB/GPU — fits |
| 70B QLoRA + ZeRO-3 on 8×80 GB | **<12 GB/GPU** |
| QLoRA base cost | 0.55 B/param; 7B ≈ 4.5 GB + activations |
| 70B DPO, 8×H100, Axolotl+FSDP2+QLoRA+FA3 | **26–38 h** |
| Unsloth honest speedup / VRAM | ~1.8–2.2× step, 60–80% less, single GPU, vs naive HF |
| Unsloth vs tuned FA2+compile baseline | ~1.0–1.3× |
| 7B QLoRA peak VRAM | Unsloth ~8–10 GB; others ~13–16 GB |
| LR: LoRA / full FT / DPO / GRPO | 2e-4 / 2e-5 / 5e-6 / 1e-6 |
| LoRA rank / alpha | 16 / 32 (α = 2r) |
| Trainable %, 7B QLoRA r=16 | ~0.6–0.8% |
| Gradient checkpointing cost | −60–70% activation, +25–35% step time |
| Kernel fusion gain | 1.5–2.5× step time at scale |
| Rollout share of RL step time | **60–85%** |
| Integrated vLLM rollout speedup | 5–10× end-to-end vs `model.generate()` |
| ZeRO stage for ≤13B / ≥34B | 2 / 3 |
| Framework switch cost | 1–2 engineer-weeks + permanent comparability loss |
| Framework quality impact below 13B | ±0.5–1 point (noise); data dominates 10:1 |
| Hosted break-even | 5–20M tuned tokens/month |
| Tuned-model inference premium | 1.5–2× (3–4× for larger models) |
| Spot discount / spot-safe cadence | 60–70%; `save_steps: 100–250` |
| PEFT adapter vs merged checkpoint | ~100 MB vs ~14 GB |
| Time to first run: LLaMA-Factory vs HF | 20–40 min vs 2–4 h |
| SkyPilot reach | 16+ clouds + Kubernetes |

---

## 12. Copy-Paste Starter Config

**A. LLaMA-Factory — the whole job in one file (1×24 GB, 5k examples, ~1.5–2.5 h):**
```yaml
# qwen7b_qlora_sft.yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all              # every linear layer — safest for an unfamiliar arch

dataset: my_alpaca_5k         # register in data/dataset_info.json FIRST
template: qwen                # MUST match the base family. Silent failure otherwise.
cutoff_len: 2048
max_samples: 5000
overwrite_cache: true
preprocessing_num_workers: 8

output_dir: ./out/qwen7b-sft-lora
logging_steps: 10
save_steps: 250               # NOT save_strategy: epoch if you are on spot
plot_loss: true
report_to: wandb
run_name: qwen7b-sft-lora

per_device_train_batch_size: 2
gradient_accumulation_steps: 4      # effective batch 8
learning_rate: 2.0e-4               # LoRA range = 1e-4-3e-4
num_train_epochs: 1
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
quantization_bit: 4                 # QLoRA
quantization_method: bnb
flash_attn: fa2
ignore_pad_token_for_loss: true
val_size: 0.05
eval_strategy: steps
eval_steps: 250
```
```bash
llamafactory-cli train qwen7b_qlora_sft.yaml
llamafactory-cli export export_qwen7b.yaml
```

**B. Axolotl — the incremental ladder (each file is "changes only"):**
```yaml
# base_sft_lora.yaml
base_model: meta-llama/Llama-2-7b-hf
training_type: sft
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
fp16: true
bf16: false
gradient_checkpointing: true
datasets:
  - path: timdettmers/openassistant-guanaco
    type: chat_template
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
optimizer: adamw_torch
learning_rate: 2e-4
output_dir: ./outputs/sft-lora
```
```yaml
# + qlora.yaml
load_in_4bit: true
bnb_4bit_compute_dtype: float16
bnb_4bit_quant_type: nf4
adapter: qlora
optimizer: paged_adamw_8bit

# + dpo.yaml  (layer on top of the SFT config)
training_type: dpo
datasets:
  - path: argilla/ultrafeedback-binarized
    type: preference
dpo_beta: 0.1

# + fsdp.yaml  (single GPU -> multi-GPU, same config)
distributed_type: fsdp
fsdp:
  sharding_strategy: FULL_SHARD
  auto_wrap_policy: transformer
  state_dict_type: sharded_optim   # FULL only at the very end
  sync_module_states: true
gradient_checkpointing: true
```

**C. The framework-agnostic merge — works on ANY PEFT adapter:**
```python
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, ADAPTER, OUT = "Qwen/Qwen2.5-7B-Instruct", "./out/qwen7b-sft-lora", "./merged/qwen7b-sft"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)

# The check that prevents the module's most expensive latent bug:
recorded = model.peft_config["default"].base_model_name_or_path
if recorded != BASE:
    sys.exit(f"REFUSING TO MERGE: adapter trained on {recorded!r}, merging into {BASE!r}")

merged = model.merge_and_unload()                            # W' = W + (alpha/r)*BA
merged.save_pretrained(OUT, safe_serialization=True)         # bf16 on CPU — never fp16 on GPU
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
# Keep the adapter (~100 MB) next to the merged model (~14 GB): the adapter is what moves.
```

**D. SkyPilot — cheapest capacity + preemption survival:**
```yaml
# skypilot_ft.yaml — sky jobs launch -c qwen-ft skypilot_ft.yaml
name: qwen-ft
resources:
  accelerators: {A100-80GB:8}
  cloud: aws,gcp,azure,lambda     # bid across providers
  use_spot: true                  # ~60-70% cheaper, preemptible
  disk_size: 500
  any_of:
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
# CONTRACT: the trainer YAML must set save_steps: 250 (not save_strategy: epoch),
# or a preemption at hour 22 loses the run. Managed jobs resume from the last checkpoint.
```

**E. The decision defaults all of the above encode:**
```text
gpus == 0                 -> hosted API (exportable: Vertex+Gemma / Together / Predibase)
1 GPU, <=13B, known arch  -> Unsloth
1 GPU, unfamiliar or VLM  -> LLaMA-Factory
2-16 GPUs, 7-34B          -> Axolotl (reproducible) | LLaMA-Factory (breadth) | torchtune (FSDP2/float8)
70B, DPO, 8 GPU           -> Axolotl + ZeRO-3 + QLoRA   (NOT veRL)
GRPO at any scale         -> veRL | OpenRLHF | trl (<=8B)  -- needs a rollout engine
>32B FULL fine-tune       -> Megatron (TP+PP), not FSDP2 alone
VLM                       -> ms-swift | LLaMA-Factory
cost is the constraint    -> any trainer + SkyPilot spot + save_steps: 250
<=13B on 1 GPU            -> pick the one you can debug fastest; the model will be the same
```

---

## 13. What To Read Next

| If you want | Go to |
|---|---|
| The reasoning, evidence and instructor quotes behind every row here | **CS-03** |
| 102 interview questions with traps, rapid-fire, and 5 whiteboard tasks | **IQ-03** |
| LoRA/QLoRA mechanics, `target_modules`, rank selection, merging | **CS-23** |
| Whether to fine-tune at all, versus RAG or prompting | **CS-04** |
| Hands-on per framework: LLaMA-Factory, Unsloth, Axolotl | **CS-17, CS-18, CS-19** |
| Which framework exposes DPO / GRPO / ORPO, and when to use each | **CS-25, CS-26, CS-27** |
| Continued pretraining (the CPT rows in the matrix) | **CS-12** |
| Inference quantization (GPTQ/AWQ/GGUF) vs training-time 4-bit | **CS-10, CS-11** |
| Serving the merged model or the adapter (vLLM, multi-LoRA) | **CS-30, CS-31** |
| Evaluation harnesses and LLM-as-judge (FastChat's real value) | **CS-34** |
| Multi-tenant adapter serving | CS-30/31; OpenLLM/Predibase LoRAX in **CS-03 §4.9, §4.18** |
