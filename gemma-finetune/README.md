# 🚀 Gemma Fine-Tuning for Product Recommendation & Strategy

A **complete, beginner-friendly** pipeline to fine-tune Google's **Gemma-2B** model on real product reviews to generate product recommendations and brand strategy suggestions.

> [!TIP]
> **New to fine-tuning?** This guide explains everything from scratch — no prior experience required!

---

## 📋 Table of Contents

1. [What Does This Project Do?](#-what-does-this-project-do)
2. [Key Concepts Explained](#-key-concepts-explained)
3. [Hardware Requirements](#-hardware-requirements)
4. [Setup Guide](#-setup-guide)
5. [Project Structure](#-project-structure)
6. [Quick Start](#-quick-start)
7. [Configuration Guide](#-configuration-guide)
8. [Understanding torch.compile](#-understanding-torchcompile)
9. [Common Errors & Fixes](#-common-errors--fixes)
10. [Parameter Tuning Guide](#-parameter-tuning-guide)

---

## 🎯 What Does This Project Do?

This project takes a pre-trained **Gemma-2B** model (Google's open-source LLM) and teaches it to:

1. **Analyze product reviews** — Understanding sentiment, strengths, and weaknesses
2. **Generate recommendations** — BUY / SKIP / CONSIDER with reasoning
3. **Suggest brand strategy** — What the brand should do based on customer feedback

### Before Fine-Tuning
```
Input: "This phone has great battery life but the camera is terrible."
Gemma: (generic text completion, no useful recommendation)
```

### After Fine-Tuning
```
Input: "This phone has great battery life but the camera is terrible."
Gemma: 
  Recommendation: CONSIDER
  Strengths: Battery life praised by the reviewer
  Weaknesses: Camera quality is a significant concern
  Brand Strategy: Prioritize camera improvements in the next iteration...
```

---

## 📚 Key Concepts Explained

### What is Fine-Tuning?

**Fine-tuning** = taking a pre-trained model and training it further on your specific data.

Think of it like this:
- **Pre-trained Gemma** = A college graduate who knows language well
- **Fine-tuned Gemma** = That graduate after specialized training in product analysis

### What is LoRA?

**LoRA** (Low-Rank Adaptation) is a technique that makes fine-tuning much cheaper:

| Approach | Params Updated | VRAM Needed | Time |
|----------|---------------|-------------|------|
| Full fine-tuning | 2 billion (100%) | ~50 GB | Days |
| LoRA | 2.6 million (0.13%) | ~6 GB | Hours |

**How LoRA works:**
- Freezes all original model weights (2B params, unchanged)
- Adds tiny "adapter" matrices to attention layers (~2.6M params)
- Only trains these adapters — 800x fewer parameters!
- After training, adapters merge back into the model — zero inference overhead

### What is QLoRA?

**QLoRA** = LoRA + 4-bit Quantization

Normal weights: 16-bit floats (2 bytes each) → 2B params = 4 GB
4-bit weights: (0.5 bytes each) → 2B params = 1 GB

**Result**: Fine-tune a 2B model on a consumer GPU with just 6 GB VRAM!

### What is `torch.compile`?

`torch.compile()` is PyTorch's graph-mode compiler that optimizes your model:

```python
# Without torch.compile (eager mode):
#   Python executes each operation one at a time
#   → Python overhead + no cross-operation optimization

# With torch.compile:
#   PyTorch traces the computation graph
#   → Fuses operations → generates optimized GPU kernels
#   → 1.3-2x speedup after initial compilation
```

**Trade-offs:**
- ✅ 1.3-2x faster training after warmup
- ❌ First few steps are slow (compilation overhead)
- ❌ Requires Triton on Linux for the `inductor` backend
- ❌ May cause issues with some LoRA operations (handled gracefully)

---

## 💻 Hardware Requirements

### Minimum
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with ≥ 8 GB VRAM (RTX 3060+) |
| RAM | ≥ 16 GB |
| Disk | ≥ 20 GB free |
| OS | Linux (recommended) or Windows (WSL2) |
| Python | ≥ 3.10 |

### Recommended
| Component | Recommendation |
|-----------|---------------|
| GPU | NVIDIA RTX 4070+ or A100 (24+ GB VRAM) |
| RAM | 32 GB |
| OS | Ubuntu 22.04 |

### Cloud Options (if no GPU available)
- **Google Colab** — free T4 GPU (16 GB VRAM) — enough for this project!
- **Kaggle Notebooks** — free P100 GPU
- **Lambda Labs** — A100 80GB for ~$1.10/hr
- **RunPod** — A100 40GB for ~$0.79/hr

---

## 🛠 Setup Guide

### Step 1: Clone/Navigate to the project

```bash
cd /root/Downloads/AG/gemma-finetune
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> If `bitsandbytes` fails to install on Windows, use WSL2 instead.

### Step 4: Set up Hugging Face access

Gemma requires accepting Google's license:

1. Go to [huggingface.co/google/gemma-2b](https://huggingface.co/google/gemma-2b)
2. Click **"Agree and access repository"**
3. Create a token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the token:

```bash
export HF_TOKEN=hf_your_token_here
```

### Step 5: Verify your environment

```bash
python setup_environment.py
```

This runs 6 checks: Python version, CUDA, packages, bitsandbytes, torch.compile, and HF token.

---

## 📁 Project Structure

```
gemma-finetune/
│
├── config.py               # All hyperparameters (extensively documented)
├── setup_environment.py    # Pre-flight environment checks
├── data_loader.py          # Dataset download, cleaning, formatting
├── model_setup.py          # Model loading + QLoRA + LoRA setup
├── train.py                # Main training script (entry point)
├── inference.py            # Generate predictions with fine-tuned model
├── evaluate.py             # Compute ROUGE/BLEU metrics
├── run.sh                  # One-click pipeline launcher
├── requirements.txt        # All dependencies (explained)
└── README.md               # This file
```

---

## ⚡ Quick Start

### Option 1: One-click (recommended)
```bash
chmod +x run.sh
bash run.sh
```

### Option 2: Step by step
```bash
# 1. Check environment
python setup_environment.py

# 2. Train
python train.py

# 3. Evaluate (replace with your actual output path)
python evaluate.py --model_dir ./outputs/run_XXXXXXXX/final_model

# 4. Test inference
python inference.py \
    --model_dir ./outputs/run_XXXXXXXX/final_model \
    --prompt "This product exceeded my expectations!" \
    --rating 5

# 5. Interactive mode
python inference.py \
    --model_dir ./outputs/run_XXXXXXXX/final_model \
    --interactive
```

---

## ⚙️ Configuration Guide

All hyperparameters are in `config.py`. Key ones to experiment with:

### Memory vs. Quality Trade-offs

| If you need... | Change this | From → To |
|----------------|-------------|-----------|
| Less VRAM | `per_device_train_batch_size` | 4 → 1 |
| Less VRAM | `max_seq_length` | 512 → 256 |
| Less VRAM | `gradient_checkpointing` | True (keep!) |
| Better quality | `lora_r` | 16 → 32 |
| Better quality | `num_epochs` | 3 → 5 |
| Faster training | `use_torch_compile` | True |
| More stable | `learning_rate` | 2e-4 → 5e-5 |

### GPU-Specific Recommendations

| Your GPU | batch_size | seq_length | compile | Notes |
|----------|-----------|------------|---------|-------|
| RTX 3060 (8 GB) | 1 | 256 | True | Tight fit, monitor VRAM |
| RTX 3070 (12 GB) | 2 | 512 | True | Comfortable |
| RTX 4070 (16 GB) | 4 | 512 | True | Default config works |
| RTX 4090 (24 GB) | 8 | 1024 | True | Can increase LoRA rank |
| A100 (40 GB) | 16 | 1024 | True | Try lora_r=64 |
| Colab T4 (16 GB) | 2 | 512 | False | Disable compile on Colab |

---

## ⚡ Understanding `torch.compile`

### How It Works

```
Normal PyTorch (Eager Mode):
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│matmul│ → │ bias │ → │ relu │ → │matmul│  ← 4 separate GPU kernel launches
└──────┘   └──────┘   └──────┘   └──────┘

With torch.compile (Graph Mode):
┌────────────────────────────────────────┐
│    matmul + bias + relu + matmul       │  ← 1 optimized fused kernel
└────────────────────────────────────────┘
```

### Backends

| Backend | Speed | Compatibility | Requires |
|---------|-------|--------------|----------|
| `inductor` | ⭐⭐⭐ | ⭐⭐ | Triton (Linux) |
| `cudagraphs` | ⭐⭐ | ⭐⭐⭐ | CUDA |
| `eager` | ⭐ | ⭐⭐⭐ | Nothing |

### Common `torch.compile` Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `Triton not found` | Triton not installed | `pip install triton` |
| Graph break warnings | LoRA ops can't be compiled | Normal — `fullgraph=False` handles this |
| Very slow first steps | Compilation happening | Normal — subsequent steps are fast |
| CUDA OOM during compile | Compilation uses extra memory | Reduce batch_size or disable compile |
| `torch._dynamo` errors | Bug in PyTorch version | `torch._dynamo.reset()` or update PyTorch |

---

## 🔧 Common Errors & Fixes

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fixes (try in order):**
1. `per_device_train_batch_size = 1`
2. `max_seq_length = 256`
3. `gradient_accumulation_steps = 8`
4. `use_torch_compile = False`
5. `gradient_checkpointing = True` (should already be True)

### Token/Authentication Error
```
HTTPError: 401 Unauthorized
```
**Fix:**
```bash
export HF_TOKEN=hf_your_actual_token
# Make sure you've accepted the Gemma license on HuggingFace
```

### bitsandbytes CUDA Error
```
RuntimeError: CUDA Setup failed despite CUDA being available
```
**Fix:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install bitsandbytes --force-reinstall
```

### NaN Loss
```
Training loss: nan
```
**Fix:**
1. Reduce `learning_rate` to `5e-5`
2. Check data for empty/corrupted examples
3. Ensure `bf16=True` only if GPU supports it (Ampere+)

### Import Error: sentencepiece
```
ImportError: sentencepiece is not installed
```
**Fix:**
```bash
pip install sentencepiece protobuf
```

---

## 🎛 Parameter Tuning Guide

### Step 1: Sanity Check (5 min)
```python
# In config.py, set:
max_train_samples = 100
max_eval_samples = 20
num_epochs = 1
```
Run to verify the pipeline works end-to-end.

### Step 2: Baseline Run (30 min)
```python
# Default config — use as-is
max_train_samples = 5000
num_epochs = 3
```

### Step 3: Experiment
Try these one at a time and compare validation loss:

| Experiment | What to change | Expected effect |
|-----------|---------------|-----------------|
| Higher rank | `lora_r = 32` | Better learning, more VRAM |
| More data | `max_train_samples = 20000` | Better generalization |
| Lower LR | `learning_rate = 5e-5` | More stable, slower convergence |
| More epochs | `num_epochs = 5` | Watch for overfitting |
| MLP adapters | `target_modules += ["gate_proj", "up_proj", "down_proj"]` | More expressive model |

### Signs of Overfitting
- Train loss keeps decreasing but validation loss increases
- **Fix**: Reduce epochs, increase dropout, or use more data

### Signs of Underfitting
- Both train and validation loss are high and plateau
- **Fix**: Increase LoRA rank, learning rate, or number of epochs

---

## 📄 License

This project uses:
- **Gemma**: Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Amazon Reviews**: Academic research dataset
- **Code**: MIT License

---

## 🙏 Acknowledgments

- [Google DeepMind](https://deepmind.google/) — Gemma model
- [Hugging Face](https://huggingface.co/) — Transformers, PEFT, TRL, Datasets
- [McAuley Lab](https://huggingface.co/McAuley-Lab) — Amazon Reviews dataset
- [Meta AI Research](https://arxiv.org/abs/2106.09685) — LoRA paper
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
