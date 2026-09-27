# Fine-Tuning Codebase — Reference Implementations

Runnable, dependency-light reference implementations for every technique covered in the
handbook. Each script is self-contained, heavily commented, and designed to be *read* as
much as run.

## Design principles

1. **One file = one technique.** No framework soup. You can read `04_dpo.py` without
   reading `01_sft_lora.py` first.
2. **Every script has a `--dry-run`** that validates config, counts tokens, and prints a
   VRAM estimate *without* loading a model. Always run this first.
3. **No hardcoded paths.** Everything comes from `config.py` / env vars.
4. **Memory math is a first-class citizen** — `common/memory.py` implements the VRAM
   formulas so you can budget before you rent a GPU.
5. **Defaults are the safe defaults** from the handbook, not the "looks impressive in a
   demo" defaults.

## Layout

```
code/
├── README.md                     ← you are here
├── requirements.txt
├── common/
│   ├── memory.py                 VRAM / FLOPs / cost calculators
│   ├── data_utils.py             loading, formatting, masking, chat templates
│   ├── eval_utils.py             metrics, LLM-judge, win-rate
│   └── hub_utils.py              push/pull adapters, merge, version tagging
├── 01_sft_lora.py                Supervised fine-tuning with LoRA (HF Trainer)
├── 02_sft_unsloth.py             4-bit QLoRA SFT, 2× faster (Unsloth)
├── 03_continued_pretraining.py   Domain-adaptive pretraining on raw text / PDFs
├── 04_dpo.py                     Direct Preference Optimization
├── 05_orpo.py                    Odds-Ratio Preference Optimization (single stage)
├── 06_grpo.py                    GRPO with a verifiable reward (math)
├── 07_distillation.py            Teacher → synthetic data → student SFT
├── 08_quantize.py                GPTQ / AWQ / bitsandbytes post-training quantization
├── 09_merge_and_export.py        Merge LoRA, convert to GGUF, push to Hub
├── 10_embedding_finetune.py      Contrastive fine-tuning of an embedding model
├── 11_bert_classification.py     Encoder fine-tuning for classification / NER
├── 12_multimodal_vlm.py          Vision-language fine-tuning (Qwen2-VL style)
├── 13_openai_finetune.py         Hosted fine-tuning (OpenAI)
├── 14_vertex_gemini_finetune.py  Hosted fine-tuning (Vertex AI / Gemini)
├── 15_serve_vllm.py              Serve the result and benchmark it
├── configs/                      Framework-native YAML configs (LLaMA-Factory, Axolotl)
└── data/
    ├── make_instruction_data.py  Build an SFT set from your own documents
    └── make_preference_data.py   Build a chosen/rejected preference set
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Always start here — no GPU needed:
python common/memory.py --model 7B --method qlora --seq-len 2048 --batch 4

# Then dry-run the training script you want:
python 01_sft_lora.py --dry-run --data data/sample_sft.jsonl

# Then run it for real:
python 01_sft_lora.py --data data/sample_sft.jsonl --output ./out/sft-7b
```

## Hardware expectations

| Script | Minimum VRAM | Comfortable |
|---|---|---|
| `01_sft_lora.py` (7B, LoRA, seq 2048) | 12 GB (4-bit) | 24 GB |
| `02_sft_unsloth.py` (7B, QLoRA) | 8 GB | 16 GB |
| `03_continued_pretraining.py` (1.1B) | 10 GB | 24 GB |
| `04_dpo.py` (7B, LoRA, 4-bit) | 16 GB | 24 GB |
| `05_orpo.py` (7B, LoRA) | 14 GB | 24 GB |
| `06_grpo.py` (1.5B) | 16 GB | 24 GB |
| `10_embedding_finetune.py` (0.1B) | 6 GB | 12 GB |
| `11_bert_classification.py` (0.1B) | 6 GB | 12 GB |
| `12_multimodal_vlm.py` (2B VLM) | 20 GB | 40 GB |
| `13_*` / `14_*` (hosted) | 0 GB | 0 GB |

Run `python common/memory.py --help` for the full calculator.

## Reading order

If you are learning: `common/memory.py` → `01` → `04` → `05` → `06`.
If you are shipping: `common/memory.py` → `01` → `09` → `15`.
If you are optimizing cost: `03` vs `08` vs `10` — the three "cheaper than SFT" paths.

## Warning

These scripts are reference implementations, not production services. They intentionally
omit orchestration, retries, secret management, and multi-node launch. For those, see
`configs/` (Axolotl/LLaMA-Factory handle multi-GPU properly) and the production section
of each case study.
