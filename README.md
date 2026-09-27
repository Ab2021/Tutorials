# The Fine-Tuning Handbook

A complete, self-contained curriculum for fine-tuning LLMs and SLMs — built from a corpus
of ~34 hours of video transcripts, cross-checked against the accompanying notebooks, and
extended well past what the videos cover.

Every module ships in three forms:

| Directory | What it is | Who it is for |
|---|---|---|
| `case-studies/` | The full treatment: mechanism, maths, pros/cons, exceptions, when-to-use, failure modes | Learning it properly, or looking up why something behaves the way it does |
| `interview-questions/` | Graded question banks (L1 recall → L5 staff-level design), each with the answer, *why the interviewer asks it*, and the trap | Interview prep, or self-testing |
| `cheat-sheets/` | Dense reference cards: decision trees, symptom→fix tables, numbers to memorise, runnable starter config | Day-to-day use while actually training something |
| `code/` | 15 numbered, runnable scripts + shared utilities | Doing the work |

---

## Start here

**If you are new to fine-tuning:** read `case-studies/CS-01` → `CS-02` → `CS-13` → `CS-14`
→ `CS-23`. That is the spine: how models are trained, what transfer learning is, how
instruction tuning works, how preference alignment works, and what LoRA actually does.

**If you have a job to do:** go to `cheat-sheets/CH-01` and follow the decision tree, then
read the matching case study's "when to use / when NOT to use" section before you commit.

**If you are preparing for interviews:** `interview-questions/IQ-01` onward, in order. Do
not skip the "Trap" lines — those are what separate a candidate who has read about
fine-tuning from one who has done it.

**If you want to run something right now:**

```bash
cd code
pip install -r requirements.txt
python 01_sft_lora.py --dry-run --data data/sample_sft.jsonl   # no GPU needed
```

---

## The curriculum

### Part I — Foundations (CS-01 … CS-06)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 01 | Foundations: Pretraining and Training | ✅ 2062 | ✅ | ✅ |
| 02 | Transfer Learning and Fine-Tuning | ✅ 1671 | ✅ | ✅ |
| 03 | The Fine-Tuning Framework Landscape | ✅ 1874 | ✅ | ✅ |
| 04 | Fine-Tuning vs RAG vs Agents | ✅ 2058 | ✅ | ✅ |
| 05 | From RNN/LSTM to Attention | ✅ 1878 | ✅ | ✅ |
| 06 | The Hugging Face Masterclass | ✅ 2394 | ⬜ | ✅ |

### Part II — Compression and Domain Adaptation (CS-07 … CS-12)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 07 | BERT Fine-Tuning (classification, NER) | ✅ 2590 | ✅ | ✅ |
| 08 | Knowledge Distillation: Foundations | ✅ 2141 | ⬜ | ⬜ |
| 09 | Distillation: LLM → SLM | ✅ 2033 | ⬜ | ⬜ |
| 10 | Quantization I: PTQ, QAT, the theory | ✅ 2052 | ✅ | ✅ |
| 11 | Quantization II: GPTQ, AWQ, GGUF | ✅ 1587 | ⬜ | ⬜ |
| 12 | Domain-Adaptive Continued Pretraining | 🚧 | ⬜ | ⬜ |

### Part III — The Alignment Stack (CS-13 … CS-15)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 13 | Instruction Fine-Tuning (SFT) | ✅ 2888 | ⬜ | ✅ |
| 14 | The Alignment Map: RLHF, PPO, DPO, ORPO | ✅ 2106 | ✅ | ✅ |
| 15 | LLaMA-Factory: no-code fine-tuning | ✅ 2323 | ⬜ | ✅ |

### Part IV — Tooling and Platforms (CS-16 … CS-19)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 16 | Unsloth: 2–4× faster, low-VRAM | 🚧 | ⬜ | ⬜ |
| 17 | Axolotl: YAML-driven training | ⬜ | ⬜ | ⬜ |
| 18 | OpenAI GPT fine-tuning | ⬜ | ⬜ | ⬜ |
| 19 | Gemini / Vertex AI fine-tuning | ✅ 2439 | ⬜ | ✅ |

### Part V — Modalities and Scale (CS-20 … CS-23)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 20 | Fine-Tuning Small Language Models | ⬜ | ⬜ | ⬜ |
| 21 | Multimodal / Vision-Language | ⬜ | ⬜ | ⬜ |
| 22 | Embedding Models and Embedding FT | ⬜ | ⬜ | ⬜ |
| 23 | LoRA & QLoRA: the PEFT deep dive | ⬜ | ⬜ | ⬜ |

### Part VI — Preference Optimisation, in Depth (CS-24 … CS-28)
| # | Module | Case study | IQ | Cheat sheet |
|---|---|---|---|---|
| 24 | RL Fundamentals & RLHF with PPO | ⬜ | ⬜ | ⬜ |
| 25 | DPO — Direct Preference Optimization | ⬜ | ⬜ | ⬜ |
| 26 | GRPO — Group Relative Policy Optimization | ⬜ | ⬜ | ⬜ |
| 27 | ORPO — Odds Ratio Preference Optimization | ⬜ | ⬜ | ⬜ |
| 28 | Capstone: the end-to-end pipeline | ⬜ | ⬜ | ⬜ |

### Appendices
| # | Module | Status |
|---|---|---|
| AP-01 | The Ethics and Philosophy of Alignment | ⬜ |

**Legend:** ✅ complete · 🚧 in progress · ⬜ not yet written

> **Status note.** This handbook is being written module by module. The table above is the
> honest state of the repo — check it before assuming a file exists. Case studies carry
> line counts so you can see how deep each one actually goes.

---

## Reading the case studies

Every case study follows the same fixed structure — sections 0–20 plus two appendices — so
you can jump straight to the part you need:

```
§0   Executive Summary                          §11  Cost, Compute & Memory
§1   The Problem This Solves                    §12  Evaluation — How To Know It Worked
§2   First-Principles Mental Model              §13  Comparison Tables
§3   Core Concepts — Exhaustive Glossary        §14  Debugging Playbook
§4   Deep Dive — How It Actually Works          §15  Applied Case Studies
§5   The End-to-End Pipeline                    §16  Production Considerations
§6   Hands-On Code (annotated)                  §17  Common Misconceptions
§7   Hyperparameters — Every Knob               §18  Key Takeaways
§8   Decision Framework — When To / NOT To      §19  Self-Check Questions
§9   Pros · Cons · Limitations · Failure Modes  §20  Cross-References
§10  Exceptions, Edge Cases & Gotchas
                                                Appendix A — Instructor's Verbatim Claims
                                                Appendix B — Reference Links & Papers
```

### Two callout types you will see throughout

> **Beyond the video:** — something the instructor did not cover that you nonetheless need
> to know to use the technique. These are additions, clearly marked as such.

> **Correction:** — a place where the instructor is wrong, imprecise, or out of date. The
> callout quotes what was said, states what is actually true, and cites the timestamp.
> Nothing is silently "fixed", because knowing *which* widely-taught claims are wrong is
> itself part of the education.

Numbers are given as numbers, with units and the assumptions behind them. Claims that are
rules of thumb are labelled as such rather than presented as law.

---

## The codebase

The scripts form a pipeline. Each one is standalone-runnable and prints the reasoning
behind what it does rather than just doing it.

```
code/
├── common/
│   ├── memory.py         VRAM / FLOPs / KV-cache / cost calculator  ← the centrepiece
│   ├── data_utils.py     dataset loaders, chat templates, LOSS MASKING
│   ├── eval_utils.py     decontamination, win-rate with position-bias correction,
│   │                     reward-hacking detection, bootstrap CIs
│   └── __init__.py       UTF-8 console fix (Windows)
├── 01_sft_lora.py             LoRA/QLoRA SFT with HF Trainer + PEFT
├── 02_sft_unsloth.py          Unsloth SFT, with the chat-template trap checked
├── 03_continued_pretraining.py  PDF → clean text → chunks → CPT
├── 04_dpo.py                  DPO with the reference-model memory trick
├── 05_orpo.py                 ORPO (no reference model needed)
├── 06_grpo.py                 GRPO with verifiable rewards
├── 07_distillation.py         sequence-level and token-level KD
├── 08_quantize.py             GPTQ / AWQ / bitsandbytes / GGUF
├── 09_merge_and_export.py     merge, verify, export GGUF, push to Hub
├── 10_embedding_finetune.py   contrastive embedding fine-tuning + recall@k
├── 11_bert_classification.py  encoder classification and NER
├── 12_multimodal_vlm.py       VLM fine-tuning (Qwen2-VL), vision-token budgeting
├── 13_openai_finetune.py      hosted fine-tuning + cost model
├── 14_vertex_gemini_finetune.py  Vertex AI, including the idle-endpoint bill
├── 15_serve_vllm.py           serving + honest benchmarking
└── data/
    ├── make_instruction_data.py   generate + FILTER SFT data
    ├── make_preference_data.py    build preference pairs, with length-bias control
    └── toy_*.jsonl                small fixtures for smoke-testing
```

### Quickstart

```bash
cd code
pip install -r requirements.txt          # see the file: install torch FIRST

# 1. How much VRAM will this take? (no GPU needed)
python common/memory.py --table

# 2. Make some data (no API key needed)
python data/make_instruction_data.py --template --n 60 --out data/sample_sft.jsonl
python data/make_preference_data.py --template --out data/sample_preference.jsonl

# 3. Validate everything without a GPU
python 01_sft_lora.py   --dry-run --data data/sample_sft.jsonl
python 04_dpo.py        --dry-run --data data/sample_preference.jsonl
python 13_openai_finetune.py --data data/sample_sft.jsonl --validate --estimate

# 4. Train (needs a GPU)
python 01_sft_lora.py --data data/sample_sft.jsonl --out out/sft-lora
```

Most scripts support `--dry-run`, which validates the data, prints the memory and cost
plan, and checks the failure modes that would otherwise waste hours of GPU time. **Run
`--dry-run` first, every time.**

### What the code is opinionated about

- **Loss masking is done properly.** `build_masked_example()` tokenizes incrementally so
  the assistant span is found exactly, sidestepping the BPE non-concatenativity bug that
  makes `tokenize(a+b) != tokenize(a)+tokenize(b)`.
- **Data is validated before the GPU is touched.** Every script checks for the failure mode
  that would otherwise be discovered after a run: label-index gaps, unsupervisable rows,
  truncated responses, length-biased preference pairs, missing image placeholders.
- **Costs are computed, not estimated.** `13_` and `14_` print the full training and
  inference bill, because for hosted fine-tuning the recurring inference cost usually
  dwarfs the one-off training cost.
- **Nothing is presented as current that dates.** Price tables and model ids carry an
  explicit "verify before you budget" warning.

---

## Conventions

- **Timestamps.** Claims attributed to the source material carry a `[hh:mm:ss]`-style
  timestamp so you can check the original.
- **Numbers over adjectives.** "2× faster" is meaningless without the baseline; the
  baseline is always stated.
- **No invented transcript content.** Everything attributed to the instructor is in the
  transcript. Where the instructor did not say something, it appears as a
  `> **Beyond the video:**` callout instead.
- **`IGNORE_INDEX = -100`** throughout, matching `torch.nn.CrossEntropyLoss`'s default
  `ignore_index`. If you use a custom loss, check its default — this is a silent bug when
  it differs.

---

## Hardware expectations

From `code/common/memory.py`, for QLoRA (4-bit) LoRA training with gradient checkpointing:

| Model | QLoRA r16 training | Inference (bf16) | Inference (4-bit) |
|---|---|---|---|
| 1.5B | ~1.1 GB | 3.9 GB | 1.7 GB |
| 3B | ~2.2 GB | 7.0 GB | 2.5 GB |
| 7–8B | ~4.9–5.6 GB | 16 GB | 4.9 GB |
| 13B | ~9.0 GB | 28 GB | 7.8 GB |
| 32B | ~21.5 GB | 62 GB | 16 GB |
| 70B | ~46.8 GB | 133 GB | 34 GB |

Full fine-tuning costs roughly 20× the QLoRA figure — see the module for the arithmetic.
Add 10–20% for allocator and framework overhead, and leave headroom.

---

## Provenance

Built from a collated corpus of YouTube transcripts on LLM/SLM fine-tuning plus the
accompanying notebook repository. The transcripts are the spine; the notebooks are used to
verify what the code actually did; everything else is extension, clearly marked.

Corrections to the source material are marked, not silently applied. If you find a claim
in these pages that is wrong, the `> **Correction:**` convention is the place to look
first — it may already be documented.
