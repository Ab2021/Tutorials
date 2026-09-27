# CH-10 — Quantization Cheat Sheet

**One-line purpose:** Turn a fp16 checkpoint into a smaller, faster, deployable one — and know exactly what you traded away.
**Use when:** You need a model to fit in less memory, run on CPU/edge, cut serving cost, or fine-tune something that would not otherwise fit (QLoRA).
**Do NOT use when:** You need to *change* the model's behaviour (fine-tune the fp16 base first), you need better quality, or your bottleneck is the KV cache instead of the weights (quantize the KV cache instead).

---

## 1. The 10-Second Summary

1. **Quantization = a grid + a scale (+ a zero-point).** `q = round(x/s) + z`, `x̂ = s(q − z)`. → §2
2. **`Δ = (max − min)/(2^b − 1)`, `MSE = Δ²/12`.** Error is linear in the range, quadratic in the step. → §2
3. **One 60× outlier costs 3600× the MSE and 5.9 bits of an 8-bit budget.** This is why every method in this sheet exists. → §2
4. **Signed INT8 is the hardware default.** `[−127, 127]`; a zero-point that does not fit breaks the model *silently*. → §2
5. **Granularity is the cheapest accuracy you can buy.** `group_size=128` = 0.25 bits/param of metadata. → §4
6. **PTQ: calibrate and solve (minutes). QAT: simulate `round()` and fake its gradient (days).** → §3
7. **GPTQ = second-order error compensation (`H = 2XXᵀ`); AWQ = activation-aware rescaling of the salient ~1%.** Both W4A16, layer-wise, one-shot. → §9
8. **GGUF is a container, not a method.** `Q4_K_M` = 4-bit, k-quant, Medium size-mix. → §3
9. **7B = 14/7/3.5/1.75 GB at fp16/int8/int4/2-bit — plus the KV cache (128–512 KB/token) which never shrinks.** → §7
10. **Quantized weights cannot be trained.** The one exception is QLoRA over an NF4 base. Order: **fp16 → fine-tune → merge → quantize.** → §3

---

## 2. Core Formulas

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Forward quantize | `q = round(x/s) + z` | `x` real value, `s` scale, `z` zero-point, `q` stored int | `x=0.7891`, `s=0.0097205` → `q = round(81.18) = 81` |
| Dequantize | `x̂ = s(q − z)` | — | `81 × 0.0097205 = 0.787358` |
| Step size (affine) | `s = (max − min)/(q_max − q_min)` | `q_max − q_min = 2^b − 1` | span 2.0236, 8-bit → `s = 0.0079357` |
| Step size (symmetric) | `s = max|x| / q_max` | `q_max = 127` | `max = 1.2345` → `s = 0.0097205` |
| Zero-point | `z = round(−min/s)` | integer; **clip only as a last resort** | `min = −1.2345`, `s = 0.0079357` → `z = 156` |
| Error bound | `max|x − x̂| = Δ/2` | — | `Δ/2 = 0.004860` (measured max was 0.004004) |
| MSE | `MSE = Δ²/12` | uniform error in the cell | `0.0097205²/12 = 7.874e−6` |
| Effective bits | `bits + (scale_bytes + zp_bytes)/group_size` | fp16 scales = 2 bytes | 4-bit, `g=128`, sym → `4 + 2/128 = 4.25` (with zp) |
| Group metadata | `2·bytes·out·in / group_size` per layer | — | 4096×4096, `g=128`, fp16 → 131,072 scales |
| Outlier cost | `bits_spent = log2(k)` where `k` = outlier ratio | — | `log2(60) = 5.91` bits |
| MSE with outlier | `MSE × k²` | — | `k=60` → 3600× |
| KV cache | `2 · L · h_kv · d_head · seq · batch · bytes` | `L` layers, `h_kv` KV heads, `d_head` head dim | 32·8·128 → 128 KiB/token |
| Weights (bytes) | `params × bits/8 × (1 + overhead)` | overhead ≈ 6% at `g=128` | 7e9 × 0.5 × 1.06 = 3.71 GB |
| VRAM total | `W + KV + activations + CUDA ctx + slack` | ctx ≈ 0.6–1.2 GB, slack ≈ 10% | 3.71 + 2.15 + 0.10 + 0.60 + 0.65 = 7.2 GB |
| STE (QAT) | forward `round(clamp(x/s))`, backward `∂q/∂x := 1` | — | makes `round()` trainable |
| SmoothQuant / AWQ scale | `s_j = max|X_j|^α / max|W_j|^(1−α)`, `α ≈ 0.5` | `j` = input channel | grid-search `α ∈ [0,1]` |
| GPTQ objective | `min ‖WX − ŴX‖² = min tr((W−Ŵ)H(W−Ŵ)ᵀ)` | `H = 2XXᵀ` | input covariance, forward passes only |
| GPTQ compensation | `w_{k>j} -= δ_j · (H⁻¹)_{j,k}/(H⁻¹)_{jj}` | `δ_j = w_j − q_j` | pushes error into un-quantized columns |
| AWQ identity | `W·X = (W·diag(s))·(diag(s)⁻¹·X)` | exact in fp16 | free because activations are not quantized |

---

## 3. Decision Tree

```
What are you optimizing?
│
├─ FIT TRAINING in memory (you will fine-tune)
│   └─ QLoRA: NF4 (bnb_4bit_quant_type="nf4") + double_quant ON + compute bf16 + LoRA
│      • NEVER QAT unless you own the model
│      • Order: fp16 → LoRA → merge → quantize for serving
│
├─ SERVE on GPU
│   ├─ quality-first, any model            → AWQ 4-bit  (w_bit=4, q_group_size=128, version="GEMM")
│   ├─ universal ecosystem / 3-bit         → GPTQ 4-bit (bits=4, group_size=128, desc_act=False)
│   ├─ max throughput, NVIDIA, own a build → TensorRT-LLM (FP8 / INT8 / NVFP4)
│   ├─ max throughput, open stack          → vLLM + AWQ/GPTQ/FP8
│   └─ just experimenting, no conversion   → bitsandbytes load_in_4bit / load_in_8bit
│
├─ SERVE on CPU / Mac / edge / air-gapped
│   └─ GGUF + llama.cpp / Ollama / LM Studio / llamafile
│      start Q4_K_M → Q5_K_M if RAM allows → Q6_K/Q8_0 if RAM is free
│      below 4-bit: build an imatrix first
│
└─ SERVE long context (>8k)
    └─ weights at 4-bit is NOT the lever: add FP8 KV cache FIRST
       (Llama-2-7B: 512 KB/token → 17.2 GB at 32k; 8B GQA: 128 KB/token → 4.3 GB)

Model size → bit-width
├─ ≤1.5B → 4-bit with group_size 32–64, expect a real quality drop; prefer 5–6-bit if it fits
├─ 3–14B → 4-bit, group_size 128, GPTQ or AWQ            ← the sweet spot
├─ 30–70B → 4-bit standard; 3-bit acceptable; 2-bit needs rotation or QAT
└─ >70B  → check the KV cache and the host RAM for the fp16 conversion (~2 bytes/param)

Validation is NOT optional:  fp16 vs quantized, on YOUR data
  mean KL ≤ 0.1 · p95 KL ≤ 1.0 · top-1 agreement ≥ 95% · task metric within 2%
```

---

## 4. Hyperparameter Quick Reference

| Param | Default | Typical sweep | Effect | Too high → | Too low → |
|---|---|---|---|---|---|
| `bits` (GPTQ/AWQ) | 4 | 3, 4, 8 | Weight precision | No benefit above 8 | 3 needs `g=32`; 2 is a cliff |
| `group_size` / `q_group_size` | 128 | 32, 64, 128, −1 | Weights per scale | Metadata bloat (32 → 5.0 eff. bits), slower | Worse accuracy, larger quantization error |
| `desc_act` / `act_order` | `False` | True/False | Quantize columns by activation importance | ~10% slower inference; some kernels refuse | Better accuracy left on the table |
| `damp_percent` (GPTQ) | 0.01 | 0.01, 0.05, 0.1 | Diagonal loading on `H` | Biases toward RTN, loses compensation | Singular Hessian → NaN / one bad layer |
| `sym` (GPTQ) | `True` | True/False | Symmetric (z=0) vs asymmetric | — | Asymmetric is not hardware-native; rounds `z` |
| `zero_point` (AWQ) | `True` | True/False | One zero-point per group | Requires the GEMM kernel only | Slightly worse on offset distributions |
| `version` (AWQ) | `"GEMM"` | `"GEMM"`, `"GEMV"` | Kernel selection | Wrong for batch-1 (GEMV's job) | Wrong for batched serving — correct output, half throughput |
| `bnb_4bit_quant_type` | `"nf4"` | `"nf4"`, `"fp4"` | NF4 codebook vs 4-bit float | `"fp4"` loses accuracy on Gaussian weights | — |
| `bnb_4bit_use_double_quant` | `True` | True/False | Quantize the block scales | — | Wastes 0.373 bits/param |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | bf16, fp16 | Matmul dtype | fp16 can overflow on long sequences | — |
| `bnb_4bit_blocksize` (low-level) | 64 | 32, 64, 128 | NF4 block size | Metadata bloat | Slightly better accuracy, slower |
| `load_in_8bit` (bnb) | `False` | True/False | LLM.int8() mixed precision | — | — |
| `bits` / `outtype` (GGUF) | `Q4_K_M` | see §6 | k-quant type | Larger file, no benefit above `Q8_0` | `Q2_K` is a visible cliff |
| `--imatrix` (llama.cpp) | none | a `.imatrix` file | Weight the quantization error by activation importance | — | 5–15% worse ppl at 2–3 bits |
| calibration samples | 128 | 50–512 | Range/`H` estimation quality | Slower, returns flatten | Under-estimated max → clipping at inference |
| `max_calib_seq_len` (AWQ) | 128 | 128–2048 | Tokens per calibration sample | Slower | Too few tokens for `H` |
| LR (QAT) | 1e-5 → 0 | 1e-5–1e-4, cosine | Adapter/weight adaptation | Diverges, NaN | No recovery |
| epochs (QAT) | 10–20% of training | — | How long to adapt | Overfits the calibration distribution | Under-trained quantizer |

---

## 5. Copy-Paste Code Snippets

### 5.1 Minimal working example — PTQ, QAT, and a hand-rolled quantizer

```python
"""The three quantization paths, minimal and runnable. CPU-only, no downloads."""
import torch, torch.nn as nn, torch.quantization

# ── 0. a toy model ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

X, y = torch.randn(512, 2), torch.randint(0, 2, (512, 1)).float()
model = MLP()
opt, loss_fn = torch.optim.Adam(model.parameters(), lr=0.01), nn.BCELoss()
for _ in range(500):
    opt.zero_grad(); loss_fn(model(X), y).backward(); opt.step()
print("fp32  ", ((model(X) > 0.5).float() == y).float().mean().item())

# ── 1. DYNAMIC PTQ — no calibration, weights only, activations scaled per batch ─────
q_dyn = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8)          # inplace=False is the default
print("dyn   ", ((q_dyn(X) > 0.5).float() == y).float().mean().item())

# ── 2. STATIC PTQ — needs calibration data (activations quantized with FIXED ranges) ─
class QuantizableMLP(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.quant = torch.quantization.QuantStub()      # marks the input insertion point
        self.net   = base.net
        self.dequant = torch.quantization.DeQuantStub()  # marks the output
    def forward(self, x):
        return self.dequant(self.net(self.quant(x)))

model.eval()
static = QuantizableMLP(model)
static.qconfig = torch.quantization.get_default_qconfig("fbgemm")   # x86; 'qnnpack' = ARM
torch.quantization.fuse_modules(static, [["net.0", "net.1"], ["net.2", "net.3"]],
                                inplace=True)                       # fuse Linear+ReLU BEFORE prepare
prepared = torch.quantization.prepare(static, inplace=False)
for i in range(0, 128):                       # calibration: 128 unlabelled samples, forward only
    prepared(X[i:i+1])
q_static = torch.quantization.convert(prepared, inplace=False)
print("static", ((q_static(X) > 0.5).float() == y).float().mean().item())

# ── 3. QAT — the same, but with fake quantization and training in between ───────────
qat = QuantizableMLP(MLP().eval())
qat.load_state_dict({k.replace("net.", "net."): v for k, v in model.state_dict().items()},
                    strict=False)
qat.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.fuse_modules(qat, [["net.0", "net.1"], ["net.2", "net.3"]], inplace=True)
qat = torch.quantization.prepare_qat(qat, inplace=False)
qat.train()
opt = torch.optim.Adam(qat.parameters(), lr=1e-4)             # a small LR; the STE does the rest
for epoch in range(20):
    opt.zero_grad(); loss_fn(qat(X), y).backward(); opt.step()
qat.eval()                                                     # REQUIRED before convert
q_final = torch.quantization.convert(qat, inplace=False)
print("qat   ", ((q_final(X) > 0.5).float() == y).float().mean().item())
```

```python
"""The quantizer by hand — this is what every library is doing underneath."""
def quantize_tensor(t, num_bits=8):
    """Symmetric affine quantization to signed int8. Returns (q, scale, zero_point)."""
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1        # [-128, 127]
    min_val, max_val = t.min(), t.max()
    # the guard everyone forgets: a dead channel makes min == max == 0 -> 0/0
    scale = (max_val - min_val) / float(qmax - qmin + 1e-8)
    zero_point = torch.round(-min_val / scale).to(torch.int32)
    q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point

def dequantize_tensor(q, scale, zero_point):
    return (q.float() - zero_point) * scale

w = torch.tensor([0.0234, -0.1456, 0.7891, -1.2345, 0.5123, -0.0678, 0.3345, -0.9123])
q, s, z = quantize_tensor(w)
w_hat = dequantize_tensor(q, s, z)
print(f"scale {s:.7f}  codes {q.tolist()}")
print(f"max err {(w - w_hat).abs().max():.6f}   MSE {((w - w_hat) ** 2).mean():.3e}")
# scale 0.0097205  codes [2, -15, 81, -127, 53, -7, 34, -94]
# max err 0.004004   MSE 5.649e-06
```

```python
"""The straight-through estimator — the only thing QAT adds to a normal training loop."""
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x.clamp(-127, 127))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output            # the lie: the derivative of the quantizer is the identity
```

### 5.2 Common variations

```python
# ── bitsandbytes: 4-bit at load time (no artifact, no calibration) ──────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,                       # or load_in_8bit=True for LLM.int8()
    bnb_4bit_quant_type="nf4",               # NF4 codebook > "fp4" for Gaussian weights
    bnb_4bit_compute_dtype=torch.bfloat16,   # matmul dtype; bf16 avoids long-seq overflow
    bnb_4bit_use_double_quant=True,          # quantize the scales too: -0.373 bits/param
)
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             quantization_config=bnb, device_map="auto")
```

```python
# ── GPTQ: quantize your own model ───────────────────────────────────────────────────
import torch, time
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "tiiuae/falcon-rw-1b"
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map="auto")

calib = ["Quantization reduces the memory footprint of large language models.",
         "Post-training quantization needs only a small calibration set.",
         "GPTQ is a second-order, layer-wise quantization algorithm.",
         "The KV cache grows linearly with context length.",
         "AWQ protects the activation-salient channels by rescaling."]     # ⚠ demo only
dataset = [tok(t, return_tensors="pt") for t in calib]

cfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False, damp_percent=0.01)
model.quantize(dataset)                                    # needs GPU; minutes for a 1B
model.save_quantized("falcon-rw-1b-gptq", use_safetensors=True)
tok.save_pretrained("falcon-rw-1b-gptq")

# load it back and verify (this is the step everyone skips)
loaded = AutoGPTQForCausalLM.from_quantized("falcon-rw-1b-gptq", device_map="auto",
                                            use_safetensors=True, trust_remote_code=True,
                                            use_triton=False, disable_exllamav2=True)
ids = tok("What is quantization?", return_tensors="pt").to(loaded.device)
t0 = time.time()
print(tok.decode(loaded.generate(**ids, max_new_tokens=64, do_sample=False)[0],
                 skip_special_tokens=True))
print(f"{time.time() - t0:.2f}s")
```

```python
# ── AWQ: quantize your own model ────────────────────────────────────────────────────
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path, quant_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, use_cache=False)
model.quantize(tok, quant_config=quant_config,
               calib_data=["What is quantization in machine learning?"] * 10,   # ⚠ demo only
               max_calib_seq_len=128, max_calib_samples=50, n_parallel_calib_samples=1)
model.save_quantized(quant_path, safetensors=True)
tok.save_pretrained(quant_path)

# load back: fuse_layers=True is faster; version must match your batch shape
loaded = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

```python
# ── quantization damage measurement: the gate that actually catches regressions ─────
import torch, torch.nn.functional as F

@torch.no_grad()
def kl_report(ref_logits, q_logits, chunk=256):
    kls, agree, n = [], 0, 0
    for a, b in zip(ref_logits.split(chunk), q_logits.split(chunk)):
        lp, lq = F.log_softmax(a, dim=-1), F.log_softmax(b, dim=-1)   # both fp32
        kls.append(F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1))
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item(); n += a.shape[0]
    kl = torch.cat(kls)
    return dict(mean_kl=kl.mean().item(), p95_kl=kl.quantile(0.95).item(),
                top1_agreement=agree / n)
# gates:  mean_kl <= 0.10   p95_kl <= 1.00   top1_agreement >= 0.95
# ALWAYS look at p95: a fine mean with a bad p95 = broken on code/JSON/rare tokens.
```

### 5.3 Framework-specific

```python
# ── HF transformers: load any quantized checkpoint with no extra code ───────────────
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ",
                                             device_map="auto")   # reads quantization_config
```

```python
# ── optimum: GPTQ through the standard HF pipeline ─────────────────────────────────
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", torch_dtype=torch.float16)
q = GPTQQuantizer(bits=4, dataset=calib_texts, group_size=128, desc_act=False,
                  damp_percent=0.01)
model = q.quantize_model(model, tokenizer=tok)
q.save(model, tok, "falcon-gptq")
```

```python
# ── llama-cpp-python: GGUF in Python, the backend for LangChain's LlamaCpp ──────────
from llama_cpp import Llama
llm = Llama(model_path="./m-Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=99,   # -1 = all on GPU
            chat_format="llama-3")            # ← if this is wrong, output is fluent nonsense
print(llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is quantization?"}])["choices"][0]["message"])
```

```python
# ── torchao: the maintained QAT API (torch.ao.quantization's eager path is legacy) ───
from torchao.quantization import quantize_, QATConfig, Int8WeightOnlyConfig
from torchao.quantization.qat import QATConfig as _QAT
# prepare -> train -> convert, same shape as the eager API but on the torchao path:
#   model = quantize_(model, _QAT(Int8WeightOnlyConfig()))   # insert fakes
#   <train>
#   model = quantize_(model, _QAT(Int8WeightOnlyConfig(), step="convert"))
```

---

## 6. CLI Commands

```bash
# ── llama.cpp: HF → GGUF f16 → k-quant ──────────────────────────────────────────────
pip install -r llama.cpp/requirements.txt          # huggingface_hub, sentencepiece, gguf, ...
python -m llama_cpp.convert_hf_to_gguf ./model-hf --outfile m-f16.gguf --outtype f16
#   (source tree: python3 convert_hf_to_gguf.py ...   |  ⚠ convert.py is DEPRECATED)

llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M              # the quant types from §9
llama-quantize m-f16.gguf m-Q8_0.gguf   Q8_0                # ~lossless, 8 bits

# ── imatrix: the highest-value 10 minutes in GGUF quantization ──────────────────────
llama-imatrix -m m-f16.gguf -f calibration.txt -o m.imatrix -ngl 99 -c 512
llama-quantize --imatrix m.imatrix m-f16.gguf m-IQ4_XS.gguf IQ4_XS   # 5-15% better at 2-3 bit

# ── run / serve / bench ────────────────────────────────────────────────────────────
llama-cli    -m m-Q4_K_M.gguf -p "What is quantization?" -n 128 --temp 0.7 -ngl 99
llama-server -m m-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192 -np 4      # OpenAI-compatible
llama-bench  -m m-Q4_K_M.gguf -p 512 -n 128 -ngl 99                  # real t/s numbers
llama-gguf   m-Q4_K_M.gguf | head -40                                # dump metadata

# ── legacy GGML .bin (pre-Aug-2023 models, still on the Hub) ───────────────────────
wget https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/ggml-model-q4_0.bin
./main -m ggml-model-q4_0.bin -p "What is quantization in ML?" -n 100   # ⚠ old binary name

# ── Ollama ─────────────────────────────────────────────────────────────────────────
ollama create mymodel -f Modelfile      # Modelfile: FROM ./m-Q4_K_M.gguf
ollama show mymodel                     # check the chat template is present
ollama run  mymodel

# ── this handbook's runnable script ────────────────────────────────────────────────
python code/08_quantize.py --model meta-llama/Llama-3.1-8B-Instruct --eval-only
python code/08_quantize.py --model meta-llama/Llama-3.1-8B-Instruct \
       --method awq --bits 4 --group-size 128 \
       --calib-dataset ./data/domain.jsonl --calib-samples 256 --out ./out
python code/08_quantize.py --model ./llama-hf --method gguf --quant-type Q4_K_M
```

---

## 7. VRAM / Cost Calculator

**Weights only, decimal GB.** GiB = GB / 1.0737. `Q4_K_M` includes k-quant block metadata.

| Model | fp16 | int8 | **int4** | 2-bit | `Q4_K_M` | `Q5_K_M` | `Q6_K` | `Q8_0` |
|---|---|---|---|---|---|---|---|---|
| 1B | 2.0 | 1.0 | **0.5** | 0.25 | 0.6 | 0.7 | 0.9 | 1.1 |
| 1.1B (TinyLlama) | 2.2 | 1.1 | **0.55** | 0.28 | 0.62 | 0.75 | 0.92 | 1.18 |
| 3B | 6.0 | 3.0 | **1.5** | 0.75 | 1.9 | 2.3 | 2.6 | 3.3 |
| 3.8B (Phi-3-mini) | 7.6 | 3.8 | **1.90** | 0.95 | 2.13 | 2.62 | 3.07 | 4.05 |
| 7B | 14.0 | 7.0 | **3.50** | 1.75 | 3.92 | 4.81 | 5.65 | 7.44 |
| 8B (Llama-3.1) | 16.0 | 8.0 | **4.00** | 2.00 | 4.48 | 5.50 | 6.46 | 8.50 |
| 13B | 26.0 | 13.0 | **6.50** | 3.25 | 7.28 | 8.93 | 10.49 | 13.81 |
| 32B | 64.0 | 32.0 | **16.0** | 8.0 | 17.9 | 22.0 | 25.8 | 34.0 |
| 70B | 140.0 | 70.0 | **35.0** | 17.5 | 39.2 | 47.2 | 56.5 | 74.4 |

**KV cache per token (fp16)** — the number everyone forgets.

| Architecture | `L/h_kv/d_head` | KB/token | 4k | 32k | 128k |
|---|---|---|---|---|---|
| TinyLlama-1.1B | 22 / 4 / 64 | 22 | 0.09 GB | 0.74 GB | 2.95 GB |
| Llama-3.2-1B | 16 / 8 / 64 | 32 | 0.13 GB | 1.07 GB | 4.29 GB |
| Qwen2.5-7B | 28 / 4 / 128 | 56 | 0.23 GB | 1.88 GB | 7.52 GB |
| **Mistral-7B / Llama-3.1-8B** | 32 / 8 / 128 | **128** | 0.54 GB | 4.29 GB | 17.18 GB |
| Llama-3.1-70B | 80 / 8 / 128 | 320 | 1.34 GB | 10.74 GB | 42.95 GB |
| Phi-3-mini-3.8B (no GQA) | 32 / 32 / 96 | 384 | 1.61 GB | 12.88 GB | 51.54 GB |
| **Llama-2-7B (no GQA)** | 32 / 32 / 128 | **512** | 2.15 GB | 17.18 GB | 68.72 GB |
| Llama-2-13B (no GQA) | 40 / 40 / 128 | 800 | 3.36 GB | 26.84 GB | 107.37 GB |

**Full budget at 4k context, batch 1** (weights + KV + ~0.1 GB activations + 0.6 GB CUDA ctx + 10% slack):

| Model | fp16 total | int4 total | `Q4_K_M` total | Fits on |
|---|---|---|---|---|
| 1.1B | 2.9 GB | 1.5 GB | 1.6 GB | 4 GB / any Mac |
| 3.8B | 9.9 GB | 4.2 GB | 4.4 GB | 8 GB @int4 |
| 7B (no GQA) | 16.9 GB | **7.2 GB** | 7.5 GB | 12 GB @int4 |
| 8B (GQA) | 17.2 GB | **5.4 GB** | 5.7 GB | 8 GB @int4 (tight) |
| 13B | 30.1 GB | 12.0 GB | 12.4 GB | 16 GB @int4 |
| 70B | 142.1 GB | 38.5 GB | 40.2 GB | 2×24 GB, or 1×24 GB + CPU offload |

**Quantization run cost** (the fp16 model must be resident — that is the real constraint):

| Model | Method | Hardware | Time | Peak RAM/VRAM | $ (A100 @ $2.5/h) |
|---|---|---|---|---|---|
| 1.1B | AWQ / GPTQ | 1×T4 16 GB | 3–15 min | ~5 GB | ~$0.03 (free on Colab) |
| 7B | AWQ | 1×A100 80 GB | 10–20 min | ~20 GB | ~$0.80 |
| 7B | GPTQ | 1×A100 80 GB | 25–60 min | ~24 GB | ~$2.50 |
| 13B | GPTQ | 1×A100 80 GB | 1–2 h | ~40 GB | ~$5 |
| 70B | GPTQ | 1×A100 + 128 GB RAM | 4–12 h | ~150 GB | ~$30 |
| 70B | GGUF (CPU) | 16-core CPU + 150 GB RAM | 1–3 h | ~150 GB | ~$1 spot |
| any | bitsandbytes | — | **0** (load-time) | — | free |

**The saving that actually matters:**

```
decode cost ∝ bytes read per token = params × bytes_per_param
fp16 → int8 → int4  =  1.0×  →  0.5×  →  0.25× bytes read   (~2-3× speedup at batch 1)
the real win = CONCURRENCY: 4× smaller weights → ~4× more KV-cache room → ~4× the batch
at 4k context a 7B's KV cache (2.15 GB) is 60% of its int4 weights (3.5 GB); at 32k it is 5x
```

---

## 8. Symptom → Fix Lookup Table

| Symptom | Likely cause | First fix |
|---|---|---|
| Output is fluent but ignores instructions | GGUF chat template missing/wrong | Dump the metadata; compare with `apply_chat_template`; re-convert |
| Never emits EOS, always hits `max_new_tokens` | Missing `generation_config.json`; wrong EOS id | Copy the generation config + tokenizer into the quantized dir |
| Perplexity fine, JSON/code broken | Perplexity is blind to long-tail structure | Measure **p95 KL** on JSON prompts; constrain decoding; exclude `lm_head` |
| Model is *slower* than fp16 | Wrong kernel; `desc_act` unsupported; CPU offload | Check the kernel name; AWQ `GEMM`↔`GEMV`; `device_map` fully on GPU |
| Quality worse than the model card claims | Calibration corpus ≠ production; too-small model for the recipe | Re-calibrate on 128–512 of *your* samples; `group_size` 128 → 32 |
| 4-bit worse than expected on a ≤3B model | Less redundancy; 7B recipes do not transfer | `group_size=32–64`; consider 5–6 bit; validate before shipping |
| `CUDA out of memory` during quantization | fp16 model + Hessian buffers | Bigger GPU; shard; CPU/GGUF path; `low_cpu_mem_usage=True` |
| One layer quantizes badly / NaN | Ill-conditioned or singular `H` | `damp_percent` 0.01 → 0.05–0.1; more calibration tokens; same device |
| Load fails with a kernel error | `desc_act=True` or an unusual head config | `disable_exllamav2=True` / `use_exllamav2=False`; or re-quantize |
| `<unk>` tokens in GGUF output | Tokenizer mismatch (wrong source revision) | Re-convert from the exact pinned HF revision |
| Quantized model *improved* perplexity | Tokenizer/template changed, or eval leaked calibration text | Treat as a bug; diff the configs; hold out calibration data |
| Repeated-token loops after an image rebuild | Kernel/library version change in the serving image | Roll back; diff kernel versions; re-validate the artifact |
| Two runs give different weights | CUDA atomics + unpinned versions | Seed, pin, hash the output; record the GPU arch |
| `Expected all tensors to be on the same device` | Calibration data on CPU, model on GPU | `.to(model.device)` the calibration batch |
| `ExllamaV2 cannot be used with this model` | Kernel cannot fuse this checkpoint | `disable_exllamav2=True` |
| `cannot import name 'AutoAWQForCausalLM'` | `auto-awq` installed instead of `autoawq` | `pip uninstall auto-awq && pip install autoawq` |
| GGUF loads but is slow on CPU | No SIMD path for that quant type | Use `Q4_K_M`/`Q5_K_M`; check the CPU's AVX support; try `-t $(nproc)` |
| Fine-tuning a quantized checkpoint does nothing | No gradient path into integer weights | Use QLoRA (NF4 + LoRA); never fine-tune a GPTQ/AWQ/GGUF model |
| VRAM still OOMs after quantizing | The KV cache did not shrink; batch/context too large | FP8 KV cache; reduce context or batch; recompute with §7 |

---

## 9. Comparison Matrix

| | **bitsandbytes** | **GPTQ** | **AWQ** | **GGUF k-quant** | **TensorRT-LLM** |
|---|---|---|---|---|---|
| What it is | load-time RTN | layer-wise 2nd-order PTQ | activation-aware PTQ | container + block quant | compiled engine |
| Scheme | W4A16 / W8A16 | W4A16 | W4A16 | W4A16 (CPU) | W8A8 / W4A8 / W4A4 |
| Calibration | none | yes (`H`) | yes (activation magnitudes) | optional (`imatrix`) | yes |
| Time (7B) | 0 | 25–60 min | 10–20 min | 20–60 min CPU | 1–3 h build |
| 4-bit quality | good | excellent | excellent | very good | excellent |
| 3-bit | — | good | fair | `Q3_K_M` ok | — |
| 2-bit | — | poor | poor | `Q2_K` cliff | NVFP4 research |
| GPU serving | yes, slow | yes (Marlin/ExLlamaV2) | yes (Marlin/GEMM) | via `-ngl` | **fastest** |
| CPU/Mac | no | no | no | **yes** | no |
| Fine-tunable | **QLoRA only** | no | no | no | no |
| Use when | experiments, QLoRA | universal 4-bit GPU | best 4-bit quality | CPU/edge/laptop | max NVIDIA throughput |

**GGUF quant types, ranked by quality vs size**

| Type | Eff. bits | 7B size | Δppl | Verdict |
|---|---|---|---|---|
| `F16` | 16.0 | 13.0 GB | 0 | reference |
| `Q8_0` | ~8.5 | 6.7 GB | <0.01 | lossless; use if RAM is free |
| `Q6_K` | ~6.6 | 5.3 GB | ~0.01 | effectively lossless |
| `Q5_K_M` | ~5.7 | 4.8 GB | ~0.02 | safe choice |
| **`Q4_K_M`** | **~4.8** | **4.1 GB** | **~0.05** | **the default — start here** |
| `Q4_K_S` | ~4.6 | 3.9 GB | ~0.08 | only if you must |
| `Q3_K_M` | ~3.9 | 3.3 GB | ~0.2 | tight memory only |
| `Q2_K` | ~2.6 | 2.7 GB | 1–5 | the cliff — use `IQ*`+imatrix instead |

**Granularity cost table**

| Granularity | Eff. bits at 4-bit | Rel. MSE | Kernel support |
|---|---|---|---|
| per-tensor | 4.00 | 1.00× | universal |
| per-channel | 4.00 | ~0.80× | standard |
| per-group 128 | 4.25 | ~0.55× | GPTQ/AWQ/bnb/GGUF |
| per-group 64 | 4.50 | ~0.48× | GPTQ/AWQ |
| per-group 32 | 5.00 | ~0.42× | GPTQ/AWQ (slower) |

---

## 10. Numbers To Memorize

| Number | Meaning |
|---|---|
| `q = round(x/s) + z` / `x̂ = s(q − z)` | the affine quantizer |
| `Δ = (max−min)/(2^b−1)`; `MSE = Δ²/12` | step and error |
| `log2(60) ≈ 5.9` bits, MSE ×3600 | the cost of a 60× outlier |
| `[−127,127]` | signed int8 convention |
| `0.25` bits/param | group metadata at `g=128` |
| `0.373` bits/param | what double quantization recovers |
| `6.0` / `0.1%` | LLM.int8()'s threshold / extracted fraction |
| `~1%` | AWQ's salient channels |
| `H = 2XXᵀ` | GPTQ's Hessian |
| `α ≈ 0.5` | SmoothQuant / AWQ scaling exponent |
| `group_size=128`, `damp_percent=0.01`, `blocksize=64` | the three standard defaults |
| `400/200/100/50 MB` per 100M params | fp32/fp16/int8/int4 |
| `14/7/3.5/1.75 GB` | 7B at fp16/int8/int4/2-bit |
| `140/70/35/17.5 GB` | 70B at fp16/int8/int4/2-bit |
| `128 KB` / `512 KB` per token | KV for GQA 8B / no-GQA 7B |
| `0.6–1.2 GB` | CUDA context + framework overhead |
| `0.05 / 0.2–0.6 / 1–5+` Δppl | 4-bit / 3-bit / 2-bit |
| `128–512` samples | calibration for GPTQ/AWQ |
| gates: `mean KL ≤ 0.1`, `p95 KL ≤ 1.0`, `top-1 ≥ 95%` | the honest evaluation threshold |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `ValueError: ExllamaV2 cannot be used with this model` | the ExLlamaV2 kernel cannot fuse this checkpoint (usually `desc_act=True`) | `disable_exllamav2=True` / `use_exllamav2=False` |
| `ImportError: cannot import name 'AutoAWQForCausalLM' from 'awq'` | the research `auto-awq` package is installed, not `autoawq` | `pip uninstall auto-awq && pip install autoawq` |
| `NameError: name 'AutoGPTQForCausalLM' is not defined` | `auto-gptq` missing in this kernel (often a Colab kernel mismatch) | `pip install auto-gptq optimum` in the same kernel |
| `RuntimeError: Expected all tensors to be on the same device` | calibration batch on CPU while the model is on GPU | `.to(model.device)` the batch, or move the model |
| `torch.cuda.OutOfMemoryError` during `model.quantize(...)` | fp16 model + Hessian/`H⁻¹` buffers do not fit | bigger GPU, shard, or the CPU/GGUF path |
| `KeyError: 'qweight'` / `'qzeros'` on load | loaded a GPTQ checkpoint with an AWQ loader (or vice versa) | use the matching library; check `config.json` → `quant_method` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` after quantizing | `q_group_size` does not divide the layer's input dimension | pick a group size that divides all your layer widths (128 usually does) |
| `UserWarning: TypedStorage is deprecated` + wrong output | a stale `bitsandbytes`/`torch` combination | pin `bitsandbytes` to the version matching your CUDA/torch build |
| `ValueError: Tokenizer class LlamaTokenizer does not exist` while converting to GGUF | the source repo needs `--vocab-type`/`trust_remote_code` for its tokenizer | pass the right converter flags; convert from the original repo, not a converted one |
| `llama_quantize: failed to load model` | the input GGUF is not f16/f32 (k-quants cannot be re-quantized from a k-quant) | always convert to `--outtype f16` first, then quantize |
| `error: unknown argument '--outtype'` | a pre-2024 `convert.py` on a post-2024 model (or vice versa) | use `convert_hf_to_gguf.py` / `llama_cpp.convert_hf_to_gguf` |
| `AssertionError: padding token must be set` during AWQ/GPTQ calibration | the tokenizer has no pad token | `tok.pad_token = tok.eos_token` |
| `RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'` | you are running a 4-bit/GPTQ path on CPU | move to CUDA, or use GGUF for CPU |
| `ggml_metal_init: failed` / `CUDA error: no kernel image is available` | the runtime was not built for this GPU arch | rebuild with the right `-DCMAKE_CUDA_ARCHITECTURES` / use the matching wheel |
| `KeyError: 'q_proj'` during LoRA/QLoRA setup | `target_modules` names do not match this architecture | print `model.named_modules()` and use the real names |

---

## 12. Copy-Paste Starter Config

The default decision for a 7B-class model going to GPU serving. Copy, edit two lines, run.

```bash
#!/usr/bin/env bash
# quantize.sh — 4-bit AWQ with domain-matched calibration and a real validation gate.
# Edit MODEL and CALIB, then run. Requires: pip install autoawq transformers datasets
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"   # ← 1. your fp16 base model
CALIB="./data/domain.jsonl"                # ← 2. YOUR data: 128-512 samples, jsonl w/ "text"
OUT="./out/awq-4bit"
N_SAMPLES=256
SEQ_LEN=512

python - <<'PY'
import json, os, torch
from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL, CALIB, OUT = os.environ["MODEL"], os.environ["CALIB"], os.environ["OUT"]
N, SEQ = int(os.environ["N_SAMPLES"]), int(os.environ["SEQ_LEN"])

# ── calibration: your own data, 128-512 samples. NEVER WikiText for a domain model. ──
texts = [json.loads(l)["text"] for l in open(CALIB, encoding="utf-8")][:N]
assert len(texts) >= 128, f"need >=128 calibration samples, got {len(texts)}"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, use_cache=False)

# ── the config: W4A16, per-group 128, GEMM kernel, asymmetric zero-point ────────────
model.quantize(tok, quant_config={
        "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM",
    }, calib_data=texts, max_calib_seq_len=SEQ, max_calib_samples=len(texts),
       n_parallel_calib_samples=1)

model.save_quantized(OUT, safetensors=True)
tok.save_pretrained(OUT)                      # ← copying the tokenizer prevents a silent
                                              #   prompt-format regression
print(f"saved {OUT}")
PY

# ── smoke test: greedy, reproducible, and it checks that the model still chats ───────
python - <<'PY'
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
p = os.environ["OUT"]
tok = AutoTokenizer.from_pretrained(p)
m = AutoAWQForCausalLM.from_quantized(p, fuse_layers=True)
ids = tok.apply_chat_template([{"role": "user", "content": "Reply with valid JSON: {\"ok\": true}"}],
                              return_tensors="pt", add_generation_prompt=True).to("cuda")
out = m.generate(ids, max_new_tokens=64, do_sample=False)     # greedy = comparable across runs
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
PY

# ── the gate: mean KL, p95 KL, top-1 agreement vs the fp16 model, on YOUR data ──────
# python eval_quant.py --fp16 "$MODEL" --quant "$OUT" --texts ./data/held_out.jsonl
#   PASS if   mean_kl <= 0.10   AND   p95_kl <= 1.00   AND   top1_agreement >= 0.95
#   plus a task gate (JSON validity, IFEval, needle-in-a-haystack at max context)
echo "done. Next: run the KL gate BEFORE shipping. Perplexity alone is not a gate."
```

**Choosing differently — one-line edits:**

| Target | Change |
|---|---|
| Maximum throughput on NVIDIA | TensorRT-LLM FP8/INT8 engine instead of AWQ |
| Standard GPTQ instead | `bits=4, group_size=128, desc_act=False` via `auto-gptq` |
| CPU / Mac / edge | `llama-quantize --imatrix m.imatrix m-f16.gguf m-Q4_K_M.gguf Q4_K_M` |
| 3-bit (memory-tight) | `q_group_size=32`, and GPTQ with `desc_act=True` |
| Quick experiment only | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` |
| You will fine-tune | QLoRA (NF4 + LoRA), not this script |

---

## 13. What To Read Next

| For | Read |
|---|---|
| The full derivations, the hand-worked example, the notebook reproductions | **CS-10 — Quantization I: Fundamentals** |
| The GPTQ Hessian arithmetic, AWQ's correction, KV-cache quantization, the precision lattice, QLoRA algebra, serving flags | **CS-11 — Quantization II: Advanced Methods & Production Practice** |
| The question banks | **IQ-10** (114 questions) and **IQ-11** |
| A runnable script implementing every branch of §3 | `code/08_quantize.py` |
| The other compression axis | **CS-08 / CS-09 — Knowledge Distillation** |
| The one case where you train on a quantized base | The QLoRA module and CS-11 §4.11 |
