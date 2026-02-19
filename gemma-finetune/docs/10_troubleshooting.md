# 10. Troubleshooting Guide — Every Error and How to Fix It

## Table of Contents
- [Installation Errors](#installation-errors)
- [CUDA / GPU Errors](#cuda--gpu-errors)
- [Memory Errors (OOM)](#memory-errors-oom)
- [Hugging Face / Authentication Errors](#hugging-face--authentication-errors)
- [Training Errors](#training-errors)
- [torch.compile Errors](#torchcompile-errors)
- [Data Loading Errors](#data-loading-errors)
- [Inference Errors](#inference-errors)
- [Performance Issues](#performance-issues)
- [Platform-Specific Issues](#platform-specific-issues)

---

## Installation Errors

### ❌ `pip install bitsandbytes` fails on Windows

```
ERROR: Could not build wheels for bitsandbytes
```

**Cause:** bitsandbytes has limited native Windows support.

**Fix:**
```bash
# Option 1: Use WSL2 (recommended)
wsl --install
# Then install everything inside WSL2

# Option 2: Try the Windows-specific wheel
pip install bitsandbytes-windows

# Option 3: Disable 4-bit quantization
# In config.py: use_4bit = False
# Warning: This uses much more VRAM
```

---

### ❌ `pip install triton` fails

```
ERROR: No matching distribution found for triton
```

**Cause:** Triton is Linux-only.

**Fix:**
```bash
# On Windows/macOS: Triton is not available
# torch.compile will still work with other backends

# In config.py:
use_torch_compile = False  # OR
torch_compile_backend = "eager"  # No Triton needed

# On Linux: Install CUDA toolkit first
sudo apt install nvidia-cuda-toolkit
pip install triton
```

---

### ❌ `ImportError: sentencepiece is not installed`

```
ImportError: This tokenizer requires sentencepiece
```

**Fix:**
```bash
pip install sentencepiece protobuf
```

---

### ❌ `ModuleNotFoundError: No module named 'packaging'`

```
ModuleNotFoundError: No module named 'packaging'
```

**Fix:**
```bash
pip install packaging
```

---

## CUDA / GPU Errors

### ❌ `RuntimeError: CUDA not available`

```python
>>> torch.cuda.is_available()
False
```

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| No NVIDIA GPU | Need NVIDIA GPU for this project |
| Drivers not installed | `sudo apt install nvidia-driver-535` or download from nvidia.com |
| CPU-only PyTorch installed | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| CUDA version mismatch | Check: `nvidia-smi` CUDA vs `torch.version.cuda` |

**Diagnostic commands:**
```bash
# Check if GPU is detected by OS
nvidia-smi

# Check PyTorch's CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Check GPU details
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

### ❌ `CUDA error: device-side assert triggered`

```
RuntimeError: CUDA error: device-side assert triggered
```

**Cause:** Usually a data issue — token IDs out of vocabulary range.

**Fix:**
```bash
# Run with CUDA_LAUNCH_BLOCKING to get a useful error message
CUDA_LAUNCH_BLOCKING=1 python3 train.py

# Then look for the actual error which is usually:
# "index out of range" → tokenizer vocabulary mismatch
# Fix: Make sure you're using the correct tokenizer for the model
```

---

### ❌ `RuntimeError: CUDA Setup failed despite CUDA being available`

```
RuntimeError: CUDA Setup failed despite CUDA being available.
```

**Cause:** bitsandbytes can't find CUDA libraries.

**Fix:**
```bash
# Find your CUDA installation
find / -name "libcudart.so*" 2>/dev/null

# Set the library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or if CUDA is elsewhere:
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Reinstall bitsandbytes
pip install bitsandbytes --force-reinstall

# Verify
python3 -c "import bitsandbytes; print('OK')"
```

---

## Memory Errors (OOM)

### ❌ `torch.cuda.OutOfMemoryError: CUDA out of memory`

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB
```

**This is the #1 most common error.** Here's a systematic fix order:

**Step 1: Reduce batch size** (most impact)
```python
per_device_train_batch_size = 1  # Minimum
gradient_accumulation_steps = 16  # Compensate
```

**Step 2: Reduce sequence length** (significant impact)
```python
max_seq_length = 256  # From 512
```

**Step 3: Disable torch.compile** (moderate impact)
```python
use_torch_compile = False  # Compilation uses extra memory
```

**Step 4: Enable gradient checkpointing** (should already be True)
```python
gradient_checkpointing = True
```

**Step 5: Reduce LoRA modules** (small impact)
```python
target_modules = ["q_proj", "v_proj"]  # Instead of all 4
```

**Step 6: Use smaller model** (last resort)
```python
# Gemma-2B is already the smallest. Consider:
# - Running on cloud GPU (Google Colab free T4 has 16 GB)
# - Using a smaller model like TinyLlama-1.1B
```

**Diagnostic: Check your VRAM usage**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Total:     {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
```

---

### ❌ OOM happens randomly during training (not on first step)

**Cause:** Variable-length sequences causing memory spikes.

**Fix:**
```python
# Pad all sequences to exactly max_seq_length (fixed memory per batch)
# Our data_loader.py already does this with SFTTrainer's max_seq_length

# Also clear CUDA cache periodically:
import torch
torch.cuda.empty_cache()
```

---

## Hugging Face / Authentication Errors

### ❌ `HTTPError: 401 Client Error: Unauthorized`

```
requests.exceptions.HTTPError: 401 Client Error: Unauthorized
```

**Cause:** Missing or invalid Hugging Face token.

**Fix:**
```bash
# Step 1: Get your token
# Visit: https://huggingface.co/settings/tokens
# Create a token with "read" access

# Step 2: Set it
export HF_TOKEN=hf_your_actual_token_here

# Step 3: Accept Gemma license
# Visit: https://huggingface.co/google/gemma-2b
# Click "Agree and access repository"

# Step 4: Verify
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.model_info('google/gemma-2b', token='$HF_TOKEN')
print('Access confirmed!')
"
```

---

### ❌ `GatedRepoError: Access to model is restricted`

```
GatedRepoError: 403 Client Error
You need to agree to share your contact information to access this model
```

**Fix:**
1. Go to https://huggingface.co/google/gemma-2b
2. You'll see "Gated model" banner
3. Click "Agree and access repository"
4. Wait a few minutes for access to propagate

---

## Training Errors

### ❌ Training loss is `nan` (Not a Number)

**Cause 1: Learning rate too high**
```python
learning_rate = 5e-5  # Reduce from 2e-4
```

**Cause 2: Wrong mixed precision setting**
```python
# If your GPU doesn't support bfloat16:
bf16 = False
fp16 = True
bnb_4bit_compute_dtype = "float16"  # Match!
```

**Cause 3: Bad data (empty or corrupt examples)**
```python
# Check your data:
for i, example in enumerate(train_dataset):
    if not example["text"] or len(example["text"]) < 10:
        print(f"Bad example at index {i}")
```

**Cause 4: Numerical overflow**
```python
# Try fp32 compute (slower but more stable):
bnb_4bit_compute_dtype = "float32"
fp16 = False
bf16 = False
```

---

### ❌ Training loss not decreasing (stuck)

**Diagnose:**
```
Step 100: loss=3.45
Step 200: loss=3.42
Step 300: loss=3.44   ← Not improving!
```

**Fixes (in order):**

| Fix | Why |
|-----|-----|
| Increase `learning_rate` to 5e-4 | May be too conservative |
| Increase `lora_r` from 16 to 32 | More adapter capacity |
| Add more target modules | More layers can adapt |
| Check data quality | Bad data = nothing to learn |
| Increase `num_epochs` | May need more passes |

---

### ❌ `ValueError: Trying to set a tensor of size X on a model of size Y`

**Cause:** Resuming from a checkpoint with different LoRA config.

**Fix:** Start training from scratch or use the same config as the checkpoint.

---

## torch.compile Errors

### ❌ `torch._dynamo errors`

```
torch._dynamo.exc.InternalTorchDynamoError
```

**Fix:**
```python
# Option 1: Reset dynamo
import torch._dynamo
torch._dynamo.reset()

# Option 2: Disable torch.compile
use_torch_compile = False

# Option 3: Use a different backend
torch_compile_backend = "eager"  # No compilation, for debugging
```

---

### ❌ `Unsupported: call_function ... in torch.compile`

```
torch._dynamo.exc.Unsupported: call_function: ...
```

**Cause:** An operation in the model can't be compiled (graph break).

**Fix:**
```python
# This is normal with LoRA! Our default handles it:
torch_compile_fullgraph = False  # Allow graph breaks (our default)

# If you had fullgraph=True, change to False
```

---

### ❌ Compilation is extremely slow (> 10 minutes)

**Cause:** `max-autotune` mode or the model is very large.

**Fix:**
```python
# Use default mode (faster compilation)
torch_compile_mode = "default"  # Instead of "max-autotune"

# Or disable compile for debugging
use_torch_compile = False
```

---

### ❌ `triton.compiler.errors.CompilationError`

```
triton.compiler.errors.CompilationError: ...
```

**Fix:**
```bash
# Update Triton
pip install triton --upgrade

# If still failing, use a different backend
# In config.py:
torch_compile_backend = "cudagraphs"  # Doesn't use Triton
```

---

## Data Loading Errors

### ❌ `DatasetNotFoundError`

```
DatasetNotFoundError: Dataset 'McAuley-Lab/Amazon-Reviews-2023' not found
```

**Fix:**
```bash
# Check internet connection
ping huggingface.co

# Check if the dataset name is correct
python3 -c "from datasets import load_dataset; load_dataset('McAuley-Lab/Amazon-Reviews-2023', 'raw_review_All_Beauty', split='full', trust_remote_code=True)"
```

---

### ❌ `ConnectionError` or `ReadTimeout`

```
requests.exceptions.ConnectionError: Connection aborted.
```

**Fix:**
```bash
# Retry — HuggingFace Hub may have temporary issues

# If behind a proxy:
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# If download keeps failing, try downloading the dataset manually:
pip install huggingface_hub
huggingface-cli download McAuley-Lab/Amazon-Reviews-2023 --cache-dir ~/.cache/huggingface
```

---

### ❌ `OSError: Not enough disk space`

**Fix:**
```bash
# Check disk space
df -h

# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/datasets/

# Use a smaller subset
python3 -c "
from datasets import load_dataset
ds = load_dataset('McAuley-Lab/Amazon-Reviews-2023', 
                   'raw_review_All_Beauty',
                   streaming=True)  # Streams data, no full download
"
```

---

## Inference Errors

### ❌ `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Input tensors are on CPU, model is on GPU.

**Fix:**
```python
# Move inputs to model device:
inputs = {k: v.to(model.device) for k, v in inputs.items()}
```

---

### ❌ Model generates repetitive text

```
Output: "great great great great great great..."
```

**Fix:**
```python
# Increase repetition_penalty
repetition_penalty = 1.3  # From 1.15

# Or lower temperature
temperature = 0.5  # From 0.7

# Or increase top_k filtering
top_k = 30  # From 50 (more focused)
```

---

### ❌ Model generates gibberish

```
Output: "the the for and is of to with in product..."
```

**Cause:** Model didn't learn properly from training.

**Fix:**
- Training loss didn't decrease? → See "Training loss not decreasing" above
- Too few epochs? → Increase to 5
- Too little data? → Increase `max_train_samples`
- Wrong prompt format at inference? → Check that inference uses the same `INFERENCE_TEMPLATE` as training

---

## Performance Issues

### ❌ Training is extremely slow

| Symptom | Cause | Fix |
|---------|-------|-----|
| First 3-5 steps slow | torch.compile warmup | Normal! Wait for it. |
| All steps slow | CPU bottleneck | Check if `nvidia-smi` shows low GPU util |
| All steps slow | Data loading bottleneck | Check if disk I/O is the bottleneck |
| All steps slow | Small batch size | Increase if VRAM allows |

**Check GPU utilization:**
```bash
# In a separate terminal, watch GPU usage:
watch -n 1 nvidia-smi

# GPU Util should be > 80%
# If < 30%, there's a CPU/IO bottleneck
```

---

### ❌ GPU utilization is low (< 30%)

**Cause:** Data loading is the bottleneck.

**Fix:**
```python
# In TrainingArguments:
dataloader_num_workers = 4  # Parallel data loading
dataloader_pin_memory = True  # Faster CPU→GPU transfer

# Also: Make sure dataset is cached locally (not re-downloading each epoch)
```

---

## Platform-Specific Issues

### Google Colab

```python
# Common Colab issues and fixes:

# 1. Runtime disconnects
#    → Save checkpoints frequently: save_steps = 50
#    → Use Google Drive for persistent storage

# 2. Free T4 GPU has 16 GB VRAM
#    → Use default config (should fit)
#    → Disable torch.compile if unstable: use_torch_compile = False

# 3. Install dependencies in Colab
!pip install torch transformers peft bitsandbytes datasets accelerate trl sentencepiece protobuf rouge-score nltk

# 4. Set HF token in Colab
import os
os.environ["HF_TOKEN"] = "hf_your_token"

# 5. Check GPU
!nvidia-smi
```

### WSL2 (Windows)

```bash
# Common WSL2 issues:

# 1. CUDA not detected
#    → Install CUDA toolkit inside WSL2 (not Windows):
#    sudo apt install nvidia-cuda-toolkit

# 2. Slow disk access
#    → Work inside the Linux filesystem (/home/user/), 
#    → NOT in /mnt/c/ (Windows filesystem, very slow)

# 3. Memory limits
#    → Edit .wslconfig in Windows:
#      [wsl2]
#      memory=16GB
#      swap=8GB
```

### macOS

```
macOS Notes:
  • No NVIDIA GPU → Training will be extremely slow on CPU
  • bitsandbytes doesn't work → Use use_4bit = False
  • Triton not available → Use use_torch_compile = False
  • Apple Silicon (M1/M2) → Can use MPS backend but it's limited
  
  Recommendation: Use Google Colab instead of macOS for training
```

---

## General Debugging Tips

### 1. Run setup_environment.py first
```bash
python3 setup_environment.py
# Fix ALL reported issues before trying to train
```

### 2. Start with smoke test
```python
# In config.py:
max_train_samples = 50
num_epochs = 1
use_torch_compile = False  # Eliminate as variable

# This should complete in < 5 minutes
# If this fails, fix before scaling up
```

### 3. Read the FULL error message
```
Python tracebacks go BOTTOM TO TOP.
The actual error is at the BOTTOM.
Upper lines show where it was called from.
```

### 4. Search online
```
Copy the error message (without file paths) and search:
  - GitHub Issues for the library
  - HuggingFace Forums
  - Stack Overflow
  - Reddit r/LocalLLaMA
```

### 5. Check versions
```bash
python3 -c "
import torch, transformers, peft, bitsandbytes, datasets, accelerate, trl
print(f'PyTorch:       {torch.__version__}')
print(f'Transformers:  {transformers.__version__}')
print(f'PEFT:          {peft.__version__}')
print(f'bitsandbytes:  {bitsandbytes.__version__}')
print(f'Datasets:      {datasets.__version__}')
print(f'Accelerate:    {accelerate.__version__}')
print(f'TRL:           {trl.__version__}')
print(f'CUDA:          {torch.version.cuda}')
"
```
