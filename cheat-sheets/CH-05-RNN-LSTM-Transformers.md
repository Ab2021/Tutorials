# CH-05 — RNN / LSTM → Attention Cheat Sheet

**One-line purpose:** Every equation, number and diagnostic you need to reason about why recurrence failed, why attention won, and what it costs.
**Use when:** Sizing a model, debugging a flat loss, choosing between a transformer and an SSM, or answering the "why did transformers replace LSTMs" question.
**Do NOT use when:** You need the full derivation or the notebook walkthrough — that is CS-05.

---

## 1. The 10-Second Summary

1. BPTT gradient is a **product of Jacobians**: `∂h_t/∂h_k = Π diag(1−h_j²) W_hhᵀ`, bounded by `(γρ)^{t−k}` → geometric decay or explosion.
2. Xavier init puts an RNN's spectral radius at **≈ 1.0** — the knife edge. `σ = 1/√256 = 0.0625`, `ρ = 0.0625 × 16 = 1.0`.
3. **Adam cannot fix vanishing gradients**: scale invariance is uniform, vanishing is a ratio between terms in one parameter's gradient.
4. LSTM cell path is `∂C_t/∂C_{t−1} = diag(f_t)` — diagonal, no weight matrix, bounded. **Mitigation, not a solution**: `f` must drop below 1 to forget, and the `h`-paths still carry weight matrices with `σ' ≤ 0.25`.
5. Long-range breaks for **three independent reasons**: gradient failure, information failure (fixed-state bottleneck), optimisation failure (no incentive).
6. Sequential computation is the **second fatal problem**: matvec = **1 FLOP/byte**, GEMM(n) = `n` FLOP/byte, A100 needs **≈ 153** ⇒ RNN runs at ~0.6 % MFU vs transformer ~50 % ⇒ ~100× wall clock at equal FLOPs.
7. Attention is `softmax(QKᵀ/√d_k)V` — path length **O(1)**, and the `√d_k` comes from `Var(q·k) = d_k`.
8. Attention costs `4n²d_model` FLOPs/layer and crosses the FFN at `n ≈ 2·d_model`. **FlashAttention fixes memory, not FLOPs.**
9. **Baseline reflex:** `ln 2 = 0.6931`, `ln 8000 = 8.9872`, uniform accuracy `1/V`. The notebook's `8.9872` *is* `ln 8000` — random targets.
10. In 2026: transformer by default, RNN/LSTM for constant-memory streaming and edge, hybrid SSM+attention for ≥100 k context.

---

## 2. Core Formulas

### 2.1 Recurrence forward equations

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| Vanilla RNN | `a_t = W_xh x_t + W_hh h_{t−1} + b_h`; `h_t = tanh(a_t)` | `x_t ∈ ℝ^{d_x}`, `h_t ∈ ℝ^{d_h}` | `d_h·d_x + d_h² + d_h` at `128/256` = `98 560` |
| LSTM forget gate | `f_t = σ(W_f [h_{t−1}; x_t] + b_f)` | `W_f ∈ ℝ^{d_h×(d_h+d_x)}` | `f = σ(1) = 0.731` at the `b_f = 1` init |
| LSTM input / candidate / output | `i_t = σ(W_i[·]+b_i)`; `g_t = tanh(W_g[·]+b_g)`; `o_t = σ(W_o[·]+b_o)` | `i,o ∈ (0,1)`, `g ∈ (−1,1)` | `o` gates the **gradient on-ramp** to `C` |
| LSTM cell update | `C_t = f_t ⊙ C_{t−1} + i_t ⊙ g_t` | `⊙` = element-wise | linear accumulation — no nonlinearity on the highway |
| LSTM hidden | `h_t = o_t ⊙ tanh(C_t)` | — | saturates for large `\|C_t\|` |
| LSTM / GRU params | `4·d_h·(d_h+d_x) + 4·d_h` / `3·d_h·(d_h+d_x) + 3·d_h` | 4 vs 3 matrices | at `128/256`: `394 240` vs `296 448` (−24.8 %) |
| GRU hidden | `h_t = (1−z_t) ⊙ h_{t−1} + z_t ⊙ tanh(W_h [r_t ⊙ h_{t−1}; x_t])` | `z_t, r_t` = update, reset | keep + write are forced to sum to 1 |

### 2.2 BPTT and the gradient product — **the reference derivation**

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Per-step Jacobian** | `∂h_j/∂h_{j−1} = diag(1 − h_j²) · W_hhᵀ` | `tanh'(a) = 1 − tanh²(a)` | — |
| **Full BPTT gradient** | `∂L/∂W_hh = Σ_{t=1}^{T} Σ_{k=1}^{t} [ ∂L_t/∂h_t · Π_{j=k+1}^{t} diag(1−h_j²) W_hhᵀ ] h_{k−1}ᵀ` | outer sum over outputs, inner over sources | double sum — **not** one product |
| **Submultiplicative bound** | `‖∂h_t/∂h_k‖ ≤ (γρ)^{t−k}` | `γ = max_j(1−h_j²) ≤ 1`, `ρ = σ_max(W_hh)` | `γρ = 0.9` → `0.9^50 = 5.2e-3` |
| Decay at `λ = 0.9`, `t−k = 50` | `0.9^50 = e^{50·ln 0.9} = e^{−5.268}` | — | `5.16 × 10⁻³` |
| Decay at `λ = 0.8` | `e^{50·(−0.22314)}` | — | `1.42 × 10⁻⁵` |
| Decay at `λ = 0.5` | `e^{−34.66}` | — | `8.9 × 10⁻¹⁶` — below fp32 epsilon `1.19e-7` |
| Xavier init | `Var(W) = 1/fan_in` ⇒ `σ = 1/√d_h` | Gaussian spectral radius `≈ σ√d_h` | `d_h=256`: `0.0625 × 16 = 1.0` — the knife edge |
| **LSTM highway** | `∂C_t/∂C_{t−1} = diag(f_t)` | `f,i,g` depend on `h_{t−1},x_t` — **never** on `C_{t−1}` | no `W` in the path; entries in `(0,1)` |
| LSTM `h`-path (still bad) | `∂h_t/∂h_{t−1} = diag(tanh C_t)·∂o_t/∂h_{t−1} + diag(o_t(1−tanh²C_t))·∂C_t/∂h_{t−1}` | `∂C_t/∂h_{t−1}` contains `W_f, W_i, W_g` | `σ' ≤ 0.25` ⇒ structurally an RNN Jacobian |
| LSTM read-out | `∂h_t/∂C_t = diag(o_t ⊙ (1 − tanh²(C_t)))` | — | small `o_t` blocks gradient **entry** |
| **Adam scale invariance** | `c·g/√(c²v) = g/√v` — uniform only | `ε = 1e-8` floor | cannot split `10⁻²` from `10⁻¹⁵` inside one scalar |
| Gradient clipping | `g ← g · clip/max(clip, ‖g‖)` | `clip = 1.0` typical | log the returned coefficient; target <10 % of steps clipped |

### 2.3 Attention, positions, and cost

| Concept | Formula | Symbols | Worked example |
|---|---|---|---|
| **Scaled dot-product** | `Attention(Q,K,V) = softmax(QKᵀ/√d_k) V` | `Q,K ∈ ℝ^{n×d_k}`, `V ∈ ℝ^{n×d_v}` | — |
| **`√d_k` derivation** | `Var(q·k) = Σ_{i=1}^{d_k} Var(q_i k_i) = d_k` ⇒ `sd = √d_k` | unit-variance components | `d_k = 128` → `√128 = 11.3` |
| Multi-head | `head_i = Attn(QW_Q^i, KW_K^i, VW_V^i)`; `MHA = [head_1;…;head_h]W_O` | `d_head = d_model/h` | `h=32, d_model=4096` → `d_head=128` |
| Attention params | `4 · d_model²` | Q, K, V, O projections | `4 × 4096² = 67 108 864` |
| FFN | `FFN(x) = W_2 · act(W_1 x + b_1) + b_2`; params `2·d_model·d_ff` or `3·d_model·d_ff` (SwiGLU) | `d_ff ≈ 4 d_model` (`8/3` for SwiGLU) | `3 × 4096 × 11008 = 135 266 304` |
| Residual | `y = x + F(x)` ⇒ `∂y/∂x = I + ∂F/∂x` | **addition, not concatenation** | the `I` is why depth trains |
| LayerNorm | `LN(x) = γ ⊙ (x−μ)/√(σ²+ε) + β` | per-token over features, no batch | Pre: `x+F(LN(x))`; Post: `LN(x+F(x))` |
| Pre-LN vs Post-LN | Pre keeps the identity path clean | — | Post-LN needs ~4000-step warmup |
| Sinusoidal PE | `PE(pos,2i) = sin(pos/10000^{2i/d})`; `PE(pos,2i+1) = cos(...)` | added to embeddings | relative offsets are a fixed linear map |
| **RoPE** | rotate dim pairs by `θ_i = pos·10000^{−2i/d}`; `⟨R_m q, R_n k⟩ = qᵀR_{n−m}k` | score depends only on `m−n`; ALiBi instead adds `−m·(i−j)` | Llama/Mistral/Qwen/Gemma |
| Attention FLOPs/layer | `≈ 4 n² d_model` | quadratic in `n` | `n=4096, d=4096`: `2.7e11` |
| FFN FLOPs/layer | `≈ 2 n d_model d_ff = 8 n d_model²` | linear in `n` | crosses attention at `n = 2 d_model` |
| KV cache | `2 · n_layers · n_kv_heads · d_head · bytes · tokens` | **factor of 2 = K and V** | Llama-3-8B: `2·32·8·128·2 = 131 072 B` = 128 KB/token |
| Attention intensity | `≈ n/2` FLOP/byte | bandwidth-bound below `n≈300` | FFN intensity `≈ n` |
| Roofline | `peak_FLOPs / peak_bandwidth` | A100-80GB fp16 | `312e12/2.039e12 = 153` FLOP/byte |

### 2.4 Loss baselines — the reflex numbers

| Baseline | Formula | Value |
|---|---|---|
| Balanced binary CE | `ln 2` | `0.6931` |
| Uniform over `V` classes | `ln V` | `ln 8000 = 8.9872`; `ln 32000 = 10.3735`; `ln 50257 = 10.8249` |
| Uniform sampling accuracy | `1/V` | `1/8000 = 1.25 × 10⁻⁴` |
| Perplexity / imbalanced binary | `exp(loss)` / `−[p ln p + (1−p)ln(1−p)]` | loss 8.9872 → ppl 8000; 90/10 → `0.325` |

---

## 3. Decision Tree

```
Sequence length > 8k, or recall of arbitrary distant tokens required?
├─ NO → edge/MCU (RAM < 1 MB)?          → LSTM/GRU, d_h=32–128, INT8
│       unbounded streaming, O(1) state? → SSM (Mamba-2) or causal RNN
│       everything else                  → TRANSFORMER (default)
└─ YES ├─ exact recall from far back?    → TRANSFORMER + FA2 + GQA + RoPE interp (YaRN)
       ├─ aggregate/drift/statistical?   → SSM (Mamba-2): linear in n, no KV cache
       └─ both                           → HYBRID (attention every 6–8 layers)

Loss flat and unmoving?
├─ exactly ln(V) or ln 2 from step 0 → DATA/WIRING: random targets, bad label shift,
│                                      softmax-before-CE, ignore_index missing
├─ rise then NaN at a fixed step     → DATA (corrupt batch) → fp16 overflow → reduction
├─ decaying with distance            → genuine vanishing → architectural fix, not optimizer
└─ loss falls, metric at chance      → LOSS/METRIC MISMATCH: prompt unmasked, template differs

Before blaming code, check arithmetic intensity:
├─ RNN / batch-1 decode → 1–2 FLOP/byte → MEMORY BOUND (expected, not a bug)
├─ GEMM with n ≥ 153    → compute bound → check MFU, then kernels
└─ slower than roofline predicts → profile first; suspect .item(), .cpu(), print in the loop
```

---

## 4. Hyperparameter Quick Reference

| Param | Default | Typical sweep | Effect |
|---|---|---|---|
| `d_h` (RNN/LSTM) | 256 | 128–1024 | params scale `4·d_h²`; larger = worse conditioning |
| `d_model` / `n_heads` | 4096 / 32 | 512–8192 / 8–64 | `12 d_model²` params per layer; `d_head = d_model/h` |
| `n_kv_heads` (GQA) | 8 | 1 (MQA) – `n_heads` | KV cache scales linearly; 8 → 4× saving |
| `d_ff` (SwiGLU) | `8/3 d_model` | `2–4 d_model` | 2/3 of layer params |
| LSTM forget bias | `b_f = 1.0` | fixed | starts `f = 0.731` vs `0.5` — high leverage |
| TBPTT truncation `k` | 64 | 32–200 | hard cap on learnable dependency length |
| Grad clip norm | 1.0 | 0.5–5.0 | log the returned coefficient; >10 % clipped = too low |
| Warmup Pre-LN / Post-LN | 500 / 4000 steps | 100–2000 / non-negotiable | Post-LN without it diverges |
| Full-FT `lr` vs LoRA `lr` | 2e-5 vs 2e-4 | 1e-5–5e-5 / 1e-4–3e-4 | 2e-4 on a full FT destroys the base model |
| LoRA rank / alpha | 16 / `2 × rank` | 8–64 / `1×–2×` | rank 64 ≈ full FT on behaviour tasks, overfits <5k rows |
| `max_len` / `padding` (RNN) | 200 / `'pre'` | task-dependent | post-truncation drops content; `'post'` makes `h_T` a pad artifact |
| Batch size / `β₂` (RNN) | 64 / 0.999 | 32–256 / 0.98–0.999 | intensity stays 1 FLOP/byte; lower `β₂` when gradients are heavy-tailed |

---

## 5. Copy-Paste Code Snippets

### 5.1 Minimal working example — an RNN, an LSTM and a transformer, side by side

```python
"""CH-05 reference: parameter counts, the gradient-product measurement, and the
arithmetic-intensity numbers, in one runnable file. Requires: torch>=2.1, numpy."""
import math
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

D_X, D_H, T = 128, 256, 50

# 1. PARAMETER COUNTS -- LSTM = 4 gates x (256x384 + 256) = 394,240; GRU = 3 x 256x384 + 768 = 296,448
assert sum(p.numel() for p in nn.LSTM(D_X, D_H).parameters()) == 394_240
assert sum(p.numel() for p in nn.GRU(D_X, D_H).parameters())  == 296_448

# 2. THE BPTT GRADIENT PRODUCT -- ||dL/dh_k|| vs (t-k), measured not asserted
class TanhRNN(nn.Module):
    def __init__(self, d_x, d_h):
        super().__init__()
        self.Wxh = nn.Parameter(torch.randn(d_h, d_x) / math.sqrt(d_x))   # Xavier
        self.Whh = nn.Parameter(torch.randn(d_h, d_h) / math.sqrt(d_h))   # spectral radius ~ 1.0
        self.bh  = nn.Parameter(torch.zeros(d_h))

    def forward(self, x, h0):
        hs, h = [], h0
        for t in range(x.size(1)):
            h = torch.tanh(x[:, t] @ self.Wxh.T + h @ self.Whh.T + self.bh)
            hs.append(h)
        return torch.stack(hs, dim=1)

model = TanhRNN(D_X, D_H)
h = model(torch.randn(1, T, D_X), torch.zeros(1, D_H))
h[:, -1, :].sum().backward()          # loss depends ONLY on h_T, so dL/dh_T = 1

h_det = h.detach().clone()            # replay the backward manually to read each step
grads, g = [], torch.ones(1, D_H)
for t in reversed(range(T)):
    grads.append(g.norm().item())
    g = (g * (1 - h_det[:, t, :] ** 2)) @ model.Whh   # dL/dh_{t-1} = W_hh^T diag(1-h^2) dL/dh_t
print([f"{49-k}:{grads[::-1][k]:.2e}" for k in (0, 10, 20, 30, 40, 49)])
# Expect geometric decay: (gamma*rho)^(t-k), gamma < 1, rho ~ 1.0 -> orders of
# magnitude of falloff between t-k = 0 and t-k = 49. That IS the vanishing problem.

# 3. ARITHMETIC INTENSITY -- the actual reason for the ~100x wall-clock gap
ai = lambda d, n=1, b=2: (2 * n * d * d) / (d * d * b)
print(f"matvec AI = {ai(4096, 1):.1f} | GEMM(n=4096) AI = {ai(4096, 4096):.0f} "
      f"| A100 needs {312e12/2.039e12:.0f} FLOP/byte")
```

### 5.2 Common variations

```python
# --- KV cache, exactly ---------------------------------------------------------
def kv_cache_bytes(n_layers, n_kv_heads, d_head, tokens, bytes_per=2):
    """Keys AND values -> the leading factor of 2. GQA shrinks n_kv_heads only."""
    return 2 * n_layers * n_kv_heads * d_head * bytes_per * tokens

print(kv_cache_bytes(32, 8, 128, 131_072) / 2**30, "GiB")   # Llama-3-8B @128k: 16.0
print(kv_cache_bytes(32, 32, 128, 131_072) / 2**30, "GiB")  # Llama-2-7B @128k: 64.0

# --- parameter arithmetic, exactly --------------------------------------------
def llama_params(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=32,
                 d_ff=11008, vocab=32000, tied=True):
    d_head = d_model // n_heads
    attn = 2 * d_model**2 + 2 * n_kv_heads * d_head * d_model   # Q,O + GQA-shrunk K,V
    ffn  = 3 * d_model * d_ff                                   # SwiGLU: gate, up, down
    per, embed = attn + ffn + 2 * d_model, vocab * d_model
    total = n_layers * per + embed + d_model + (0 if tied else embed)
    return total, {"ffn": ffn / per, "attn": attn / per, "embed": embed / total}

total, split = llama_params(tied=False)
print(f"{total:,}")                                # 6,738,415,616 -- Llama-2-7B, exact
print({k: f"{v:.3%}" for k, v in split.items()})   # ffn 64.2%, attn 31.9%, embed 1.9%

# --- long-context test: interpret the accuracy-vs-depth curve ------------------
#   flat at ceiling  -> context genuinely used
#   U-shape          -> "lost in the middle" (positional/optimisation failure)
#   monotone decay   -> distance failure; compare the slope to (gamma*rho)^delta
#   chance at all    -> extraction broken, or needle tokenization varies by position
# ALWAYS run the control: same positions, needle replaced by a same-length random string.
```

### 5.3 Framework-specific

```python
# --- HuggingFace: which attention kernel am I actually running? ----------------
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype="bfloat16",
    attn_implementation="flash_attention_2", device_map="auto")  # needs flash-attn
print(model.config._attn_implementation)        # verify -- do not assume

# --- HF SFT: mask the prompt so the loss measures only the answer --------------
labels = input_ids.clone()
labels[:, :prompt_len] = -100                    # ignore_index
from transformers import DataCollatorForSeq2Seq
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

# --- Keras: an LSTM classifier done correctly (padding='pre' + Masking) --------
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN, VOCAB, EMB, LATENT = 200, 10_000, 128, 256
x   = pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="post")
inp = Input(shape=(MAX_LEN,))
e   = Masking()(Embedding(VOCAB, EMB, mask_zero=True)(inp))   # skip padded steps
h   = LSTM(LATENT, return_sequences=True)(e)                  # keep every step
h   = tf.reduce_mean(tf.where(tf.expand_dims(inp > 0, -1), h, 0.0), axis=1)  # masked mean
model = Model(inp, Dense(1, activation="sigmoid")(h))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, batch_size=64, epochs=5, validation_split=0.1)   # >=5 epochs, not 1

# --- ALWAYS ship the tokenizer and preprocessing with the weights --------------
import json, pickle
json.dump(tokenizer.word_index, open("word_index.json", "w"))    # never skip this
pickle.dump({"maxlen": MAX_LEN, "vocab": VOCAB, "padding": "pre",
             "truncating": "post", "offset": 3}, open("preproc.pkl", "wb"))
model.save("model.keras")                        # .keras, not legacy .h5
```

---

## 6. CLI Commands

```bash
# Is FlashAttention actually active? -- VERIFY, do not assume
python -c "from transformers import AutoConfig as C; print(C.from_pretrained(\
'meta-llama/Llama-3.1-8B')._attn_implementation_internal)"
pip install flash-attn --no-build-isolation     # version MUST match torch+CUDA
python -c "import flash_attn; print(flash_attn.__version__)"

# Memory- or compute-bound? Profile BEFORE theorising.
ncu --set roofline --kernel-name regex:attention python bench_attn.py
nsys profile --stats=true -o prof python train.py   # look for sync gaps in the timeline

# Trainable vs frozen params -- model.summary() does NOT show this
python -c "from transformers import AutoModelForCausalLM as M; m=M.from_pretrained(\
'meta-llama/Llama-3.1-8B'); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"

# Log the clipping coefficient (clip_grad_norm_ returns the PRE-clip norm)
python -c "import torch; n=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); \
print(float(n), 'clipped' if n>1.0 else 'unchanged')"

# Migrate a legacy Keras HDF5 model off the deprecated format
python -c "import tensorflow as tf; tf.keras.models.load_model('m.h5').save('m.keras')"
```

---

## 7. VRAM / Cost Calculator

**Weights only** (`params × bytes`):

| Model | fp32 (4 B) | fp16/bf16 (2 B) | INT8 (1 B) | 4-bit NF4 (~0.55 B) |
|---|---|---|---|---|
| 1 B | 4.0 GB | 2.0 GB | 1.0 GB | 0.6 GB |
| 3 B | 12.0 GB | 6.0 GB | 3.0 GB | 1.7 GB |
| 7 B | 28.0 GB | 14.0 GB | 7.0 GB | 3.9 GB |
| 8 B | 32.0 GB | 16.0 GB | 8.0 GB | 4.4 GB |
| 13 B | 52.0 GB | 26.0 GB | 13.0 GB | 7.2 GB |
| 70 B | 280.0 GB | 140.0 GB | 70.0 GB | 38.5 GB |

**Training VRAM** = weights + grads + optimizer state + activations. Rule of thumb: **LoRA ≈ 2× frozen weights, QLoRA ≈ 0.7×**; add `0.3–1 GB` of activations per 2 k context at batch 1 with checkpointing on a 7 B model.

| Method | Bytes/param | × fp16 weights | 7 B total | 70 B total |
|---|---|---|---|---|
| Full FT, SGD momentum | `2 + 2 + 4` = 8 | 4× | 56 GB | 560 GB |
| Full FT, Adam (fp32 state) | `2 + 2 + 4 + 4` = 12 | 6× | 84 GB | 840 GB |
| Full FT, AdamW 8-bit | `2 + 2 + 1 + 1` = 6 | 3× | 42 GB | 420 GB |
| LoRA (bf16 base, fp32 adapter) | `2 + (r·2·k·4)/N` | ≈ 2× | 15–18 GB | 145–160 GB |
| QLoRA (NF4 + bf16 LoRA + paged AdamW) | `0.55 + small` | ≈ 0.6× | **6–10 GB** | **40–48 GB** |

**KV cache per token** (`2 × n_layers × n_kv_heads × d_head × bytes`):

| Model | n_layers | n_kv_heads × d_head | fp16 / token | fp16 @ 32 k | fp8 @ 128 k |
|---|---|---|---|---|---|
| Llama-2-7B | 32 | 32 × 128 | 512 KB | 16.0 GB | 32.0 GB |
| Llama-3-8B (GQA) | 32 | 8 × 128 | 128 KB | 4.0 GB | 8.0 GB |
| Llama-3-70B (GQA) | 80 | 8 × 128 | 320 KB | 10.0 GB | 20.0 GB |
| Mistral-7B (GQA) | 32 | 8 × 128 | 128 KB | 4.0 GB | 8.0 GB |

**The RNN-vs-transformer wall-clock gap** — arithmetic intensity decides it, not FLOPs:

| Workload | Intensity | A100 fp16 needs 153 FLOP/byte | MFU achieved |
|---|---|---|---|
| RNN/LSTM step (matvec) | `1` FLOP/byte | 153× short | **~0.6 %** |
| Transformer GEMM, `n = 4096` | `4096` FLOP/byte | 27× past | **~50 %** |
| Attention scores, `n = 512` | `~256` FLOP/byte | bandwidth-bound below `n≈300` | 15–35 % |
| Decode, batch 1 | `1–2` FLOP/byte | — | <1 % (hence batching) |

---

## 8. Symptom → Fix Lookup Table

| Symptom | Most likely cause | Fix |
|---|---|---|
| Loss flat at **exactly `ln(V)`** from step 0 | Random/misaligned targets; `softmax` before `CrossEntropyLoss`; missing `ignore_index` | Histogram the labels; pass **logits**; set `ignore_index=-100` on padding |
| Loss flat at **`ln 2 = 0.6931`**, accuracy 0.50 | Model predicts the base rate — underfit or untrained | Count **steps** (`N/batch × epochs`), not epochs; target ≥500 steps for a small task |
| Loss flat at `8.9872` on a decoder | `np.random.randint` targets → uniform noise, Bayes-optimal | Fix the data. Nothing is wrong with the model |
| Loss **0.6876** after "fine-tuning" with `num_words` 10000→1000 | Rare tokens remapped to `<UNK>`; frozen embedding gets a new distribution | Never reload data with a different `num_words` than the model was trained with |
| NaN at a **fixed** step | Corrupt batch (all-`-100` labels, zero-length target, NaN input) | Log the offending batch; `assert torch.isfinite(loss)`; `set_detect_anomaly(True)` |
| NaN at a **random** step | fp16 overflow (max 65 504) or LR too high | Switch to **bf16**; clip at 1.0; lower peak LR or lengthen warmup |
| `‖g‖ = 1e-7` **constant** across all layers | Scaling/wiring bug, not vanishing | Check `reduction`, init `std`, `requires_grad`; vanishing gives a *decaying profile* |
| Gradient decays with distance, long-range never learned | The `(γρ)^{t−k}` bound — genuine vanishing | Architectural fix (residuals, gating, attention, or longer TBPTT `k`) |
| Model fine at 128 tokens, fails at 512 | Learned position embeddings have no row ≥512; RoPE without scaling | Check `max_position_embeddings`; RoPE interpolation + continued pretraining |
| Inference 100× slower than FLOPs predict | Memory-bound (AI = 1) or a sync point in the loop | Profile first; remove `.item()`/`.cpu()`/`print(tensor)` from the loop; batch; CUDA graphs |
| Peak memory scales **quadratically** with `n` | FlashAttention not actually active | Set `attn_implementation="flash_attention_2"` and **verify** `config._attn_implementation` |
| Val loss **lower** than train loss, stable | Dropout active in train only; easier val split | Expected if the gap is small; check for duplicate/overlapping rows (leakage) if large |
| Loss falls smoothly, metric at chance forever | Loss measures the wrong thing (prompt unmasked); eval template mismatch | Mask the prompt with `-100`; print 20 raw generations and read them |
| Fine-tune is worse at everything after 1 epoch | LR too high for full FT (2e-4); no warmup | Full FT: `1e-5–5e-5` + `warmup_ratio ≥ 0.03`; or LoRA at `2e-4` |
| Reloaded Keras model gives garbage | Tokenizer/preprocessing not saved; `.h5` legacy format | Save `word_index` + preprocessing config with the weights; migrate to `.keras` |
| Only 47 optimizer steps in "1 epoch" | `3000 rows / batch 64 = 47` | Train ≥5 epochs, use more data, or unfreeze progressively at a lower LR |
| Attention weights uniform across all heads | Head collapse — dead/zeroed projection or LR too high | Check `logits.std()` at init; per-head entropy; raise dropout, lower LR |
| RNN `h_T` is a function of `<PAD>` | `padding='post'` with `return_sequences=False` | `padding='pre'`, or `mask_zero=True` + masked pooling |

---

## 9. Comparison Matrix

| Property | **RNN** | **LSTM** | **GRU** | **Transformer** | **SSM / Mamba** |
|---|---|---|---|---|---|
| **Parallelism over time** | None — `O(n)` serial steps | None — `O(n)` | None — `O(n)` | **Full** — one GEMM over `n` | Parallel scan, `O(log n)` depth |
| **Path length between positions** | `O(n)` | `O(n)` (highway mitigates) | `O(n)` | **`O(1)`** | `O(log n)` compute; `O(n)` information |
| **Memory at inference** | `O(d_h)` state | `O(d_h)` | `O(d_h)` | `O(n · d_model)` KV **cache** | **`O(1)`** state, no cache |
| **Long-range dependency** | Fails | Delayed failure | Delayed failure | Good (bounded by effective context) | Weak at exact recall |
| **Training cost per token** | `8 d²` FLOPs, MFU ~0.6 % | `32 d²` FLOPs, MFU ~0.6 % | `24 d²` FLOPs, MFU ~0.6 % | `4n d + 12 d²` FLOPs, MFU ~50 % | Linear in `n`, MFU moderate |
| **Inference cost per token** | `8 d²`, constant | `32 d²`, constant | `24 d²`, constant | `4 n d` attention + cache traffic, **grows with `n`** | `O(1)` per step, constant |
| **Params, `d = 256`, `d_x = 128`** | 98 560 | 394 240 (+300 %) | 296 448 (+200 %) | `12 d_model²`/layer | ~`3–4 d²`/block + conv |
| **Attention FLOPs** | — | — | — | `4 n² d_model` per layer | None (linear) |
| **KV cache, 8 B model** | 0 | 0 | 0 | 128 KB/token (GQA) | 0 |
| **Interpretability** | Gate inspection, low value | Gate inspection, low value | Gate inspection, low value | Attention maps (**weak** evidence), residual-stream probing | State inspection, active research |
| **Use in 2026** | Rare (edge, teaching) | Edge/embedded, streaming, baselines | Same, 25 % cheaper | **Default for everything** | ≥100 k context, high-concurrency streaming, hybrids |

**Fine-tuning era comparison** — why the RNN era was structurally hard:

| Blocker | RNN era | Post-attention |
|---|---|---|
| Pretrained checkpoints | None general-purpose (ELMo/ULMFiT are 2018 exceptions) | Every model ships one |
| Transfer mechanism | Features entangled with tokenizer + label space | Frozen encoder, swap the head; or adapters |
| Tokenizer reuse | None — no shared subword vocabulary | Reusable, shipped BPE/WordPiece vocabularies |
| Data needed / per-task cost | Large; retrain the whole model per task | 100–1000s of examples; adapters 20–50 MB, minutes |
| Long-range credit | Truncated BPTT; `k`-step cap on learning | Full-sequence gradient, `O(1)` path |
| Parallelism | Kernel launch per time step | GEMM; scales with GPUs |

---

## 10. Numbers To Memorize

| Number | Value |
|---|---|
| `ln 2` / `ln 8000` / `ln 32000` | `0.6931` / `8.9872` (the notebook's loss, exactly) / `10.3735` |
| `1/8000` | `1.25 × 10⁻⁴` — the notebook's accuracy, exactly |
| fp32 epsilon / `0.9^50` / `0.8^50` / `0.5^50` | `1.19e-7` / `5.2e-3` / `1.4e-5` / `8.9e-16` |
| Xavier spectral radius, `d_h=256` | `0.0625 × 16 = 1.0` |
| `σ'` max / Adam ε / LSTM forget-bias init | `0.25` / `1e-8` / `b_f = 1.0` → `f = 0.731` |
| Params at `d_h=256, d_x=128`: RNN / GRU / LSTM | `98 560` / `296 448` / `394 240` |
| Notebook model | `1 674 497` trainable; `89 s`; `47` steps; val_acc `0.5180` |
| Llama-2-7B params | `6 738 415 616` exact; FFN 64.2 %, attn 31.9 %, embed 3.89 % |
| Attention / FFN params per layer | `4 d_model²` / `8 d_model²` |
| Crossover `n` | `2 d_model` (≈8192 at `d_model=4096`) |
| Score matrix, `n=20 000`, 32 heads, fp16 | `25.6 GB` |
| KV cache: Llama-3-8B / Llama-2-7B | `128 KB` / `512 KB` per token |
| A100-80GB fp16 | `312 TFLOP/s`, `2.04 TB/s`, crossover **153 FLOP/byte** |
| Arithmetic intensity: matvec / GEMM(n) | `1` / `n` FLOP per byte |
| MFU: RNN / transformer | `~0.6 %` / `~50 %` (≈100× wall clock) |
| FlashAttention | 2–4× faster, `O(n²)→O(n)` memory, **FLOPs unchanged** |
| `√d_k` at `d_k=128` | `11.3` |
| Multi-head ablation (AILN Table 3) | single-head 24.9 vs base 25.8 BLEU (−0.9) |
| Post-LN vs Pre-LN warmup | 4000 steps vs ~100–2000 |
| ULMFiT / Jamba hybrid ratio | 18–24 % error reduction, 100 examples ≈ 100× data / ~1 attn layer per 7 Mamba |

---

## 11. Common Errors And Their Exact Messages

| Error message | Meaning | Fix |
|---|---|---|
| `NameError: name 'imdb' is not defined` | A later cell reloaded the dataset without re-running the import — the notebook's own on-camera error | Re-run the import/dataset cell, or move the loader into the same cell |
| `AttributeError: module 'tensorflow.keras.models' has no attribute 'load_model'` | Spoken as `model.load_model(...)`; the function is `keras.models.load_model` | `from tensorflow.keras.models import load_model; load_model("f.h5")` |
| `WARNING:absl:You are saving your model as an HDF5 file via 'model.save()'. This file format is considered legacy. We recommend using instead the native Keras format, e.g. 'model.save('my_model.keras')'` | `.h5` is legacy; custom objects and some config may not round-trip | `model.save("m.keras")`; keep `.h5` only for interop |
| `ValueError: Input 0 of layer "lstm" is incompatible with the layer: expected ndim=3, found ndim=2` | LSTM wants `(batch, timesteps, features)` | Insert `Embedding(...)` or `Reshape((T, 1))` |
| `ValueError: A target array with shape (N, 1) was passed for an output of shape (N, 8000, 1)` | Classification head vs a seq2seq head | Match the head: `Dense(8000, softmax)` with `(batch, T, 1)` integer targets |
| `InvalidArgumentError: logits and labels must have the same first dimension` | Wrong loss for the shape — `sparse_categorical_crossentropy` needs integer labels, class axis last | `sparse_categorical_crossentropy` + `(batch, T)` labels; or one-hot + `categorical_crossentropy` |
| `RuntimeError: CUDA out of memory` at a length that used to work | Attention score matrix (O(n²)) or KV cache | FlashAttention-2, GQA, checkpointing, fp8/fp16 cache, shorter `max_len` |
| `RuntimeError: element 0 of tensors does not require grad` | Loss left the graph — a stray `detach()`, `float()`, or `no_grad()` | Confirm `loss.requires_grad`; remove the break |
| Loss reported as `nan` with `accuracy: 0.5000` | Weights already NaN; accuracy is the argmax of garbage | Restore the last finite checkpoint; find the step with `set_detect_anomaly(True)` |
| `TypeError: unsupported operand type(s) for +: 'NoneType'` in the optimizer step | Optimizer constructed before the model's parameters existed | Construct the optimizer **after** the model |

---

## 12. Copy-Paste Starter Config

The config block, then the five instrumentation lines that make a run *readable*. Full runnable model + loop is in §5.1.

```python
# ---- CONFIG ------------------------------------------------------------------
VOCAB, D_MODEL, MAX_LEN = 8000, 128, 200      # tokenizer size, width, truncation
D_H, N_HEAD, N_LAYER    = 256, 4, 2           # recurrent width / attention shape
BATCH, EPOCHS, LR, WD   = 64, 5, 1e-3, 0.01
CLIP, WARMUP_FRAC, SEED = 1.0, 0.03, 0
PAD_ID, BOS_ID, UNK_ID  = 0, 1, 2             # Keras IMDb offsets by +3
DEVICE = "cuda"
```

```python
# ---- THE FIVE INSTRUMENTATION LINES -------------------------------------------
print(f"steps = {len(x_tr)//BATCH * EPOCHS}")                     # 1. is this even training?
print(f"baselines: ln2={math.log(2):.4f} lnV={math.log(VOCAB):.4f} "
      f"majority={max(y_tr.mean().item(), 1-y_tr.mean().item()):.4f}")   # 2. what is "no learning"?
assert torch.isfinite(loss), f"non-finite loss at ep{ep} step{i}"  # 3. fail at the FIRST bad step
g = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)       # 4. log clip frequency
if t % 50 == 0: print(f"ep{ep} step{i} loss={loss:.4f} |g|={float(g):.2f}")  # 5. one line per 50
```

```yaml
# ---- EQUIVALENT YAML (Axolotl / LLaMA-Factory style) --------------------------
base_model: meta-llama/Llama-3.1-8B-Instruct
sequence_len: 4096                 # must be <= max_position_embeddings
sample_packing: true               # raises MFU; watch the attention mask
learning_rate: 2e-4                # LoRA range; use 1e-5..5e-5 for a FULL fine-tune
lr_scheduler: cosine
warmup_ratio: 0.03                 # Post-LN needs ~4000 steps; Pre-LN ~100-2000
num_epochs: 3
micro_batch_size: 4
gradient_accumulation_steps: 16    # effective batch 64
gradient_checkpointing: true       # ~30% slower, several GB cheaper
bf16: true                         # NOT fp16 -- bf16 has fp32's exponent range
max_grad_norm: 1.0
flash_attention: true              # verify with config._attn_implementation
adapter: lora
lora_r: 16                         # 64 for harder tasks, overfits under ~5k rows
lora_alpha: 32                     # = 2 x r
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

| Output | Reading |
|---|---|
| `steps` | Under ~100 total you are warming up, not training |
| `baselines` | Any `val_loss` at or above these means nothing was learned. **Always read it.** |
| `non-finite loss` assert | Fires at the *first* bad step, while you still have a clean batch |
| `\|g\|` | >10 % of steps at the clip threshold → threshold too low or LR too high |
| val metric vs majority | At or below majority-class = no signal |

**Tuning order:** `LR` (1e-3 → 3e-4 if the loss oscillates) → `BATCH` (32 if VRAM-bound) → `EPOCHS` (→10 before concluding underfit) → `D_MODEL` (only after the pipeline is proven). Save the tokenizer beside the weights; for adapters see **CS-13 §6.8** / **CS-11 §4.11**.

---

## 13. What To Read Next

| For | Go to |
|---|---|
| Full derivations, the notebook walkthrough, 5-way comparison, cost tables | **CS-05** — `case-studies/CS-05-RNN-LSTM-to-Attention.md` |
| 98 interview questions on this material with answers and traps | **IQ-05** — `interview-questions/IQ-05-RNN-LSTM-Transformers.md` |
| Pretraining, tokenizers, the model lifecycle | **CS-01 / CH-01** |
| Transfer learning fundamentals | **CS-02** |
| Whether to fine-tune at all | **CS-04 / CH-04** |
| The first post-attention fine-tuning win (BERT) | **CS-07 / CH-07** |
| Where adapters go and why (LoRA/QLoRA) | **CS-13 §6.8** / **CS-11 §4.11** |
| Long-context continued pretraining | **CS-12** |
| Weight and KV-cache quantisation | **CS-10 / CS-11** |

**Primary sources, in reading order:** Vaswani et al. 2017 (*Attention Is All You Need*) → Bahdanau et al. 2014 (attention for NMT) → Hochreiter & Schmidhuber 1997 (LSTM) → Gers & Schmidhuber 2000 (forget-gate bias) → Cho et al. 2014 (GRU) → Pascanu et al. 2013 (vanishing/exploding gradients) → Bengio et al. 1994 (the original long-range argument) → Howard & Ruder 2018 (ULMFiT) → Su et al. 2021 (RoPE) → Press et al. 2022 (ALiBi) → Dao et al. 2022 (FlashAttention) → Gu & Dao 2023 (Mamba) → Dao & Gu 2024 (Mamba-2 / SSD) → Lieber et al. 2024 (Jamba) → Arora et al. 2023 (Zoology of SSMs and recall) → Hoffmann et al. 2022 (Chinchilla) → Kaplan et al. 2020 (scaling laws).
