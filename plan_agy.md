# 🚀 ABHISHEK'S DEFINITIVE 18-MONTH CAREER PIVOT PLAN
## Senior Data Scientist → Hardware-Aware ML / Edge AI / Physical AI Semiconductor Engineer
### Version 2.0 — Synthesized, Enhanced, Production-Grade
*June 2026 | INR-Constrained | Working Professional | Camera-Shy | Deeply Implementation-Rooted*

---

> **YOUR POSITIONING STATEMENT (Memorize This):**
> *"The ML engineer who deploys transformer-scale intelligence onto microcontroller-class silicon —
> bridging production RAG systems to ₹450 biosensing chips with zero cloud dependency."*
>
> This intersection is ULTRA RARE. Most embedded engineers can't train a transformer.
> Most ML engineers don't know what a cache miss is. You will be both.

---

## SYNTHESIS: WHAT BOTH ANALYSES AGREE ON

After cross-referencing both reasoning documents (rough_agy.md + rough_cc.md):

**From rough_cc.md (the key insight):** *"The gap is vertical, not horizontal. You need to descend the stack."*

**From rough_agy.md (the pyramid):**
```
TIER 3: AI Compiler/HW (MLIR, TVM, kernels)  → Touch, don't master yet
TIER 2: Physical AI / Edge AI (TinyML, edge)  → YOUR PRIMARY TARGET
TIER 1: Deep ML Mastery (compression, SLMs)   → 40-50% already done!
TIER 0: Your current skills (PyTorch, RAG)    → Already there
```

**KEY SYNTHESIS DECISION:** Start with the UNIFIED artifact approach (rough_cc insight) — one integrated
biosensing project that touches ALL three tiers — while building toward the 18-month pyramid (rough_agy insight).
Do NOT fragment into three separate streams in Month 1. Integrate.

---

## TIME BUDGET (NON-NEGOTIABLE REALITY)

```
Weekdays (Mon-Thu): 1.5 hrs/day × 4 = 6.0 hrs
Friday:             1.0 hrs           = 1.0 hrs
Saturday:           3.0 hrs           = 3.0 hrs
Sunday:             1.5 hrs (writing) = 1.5 hrs
──────────────────────────────────────────────
Total per week:                        11.5 hrs
Conservative (fatigue-adjusted):       10 hrs/week
Over 18 months (78 weeks):             780 hours

ALLOCATION:
  40% → Implementation (hardware + code)          312 hrs
  25% → Study (courses + papers + docs)           195 hrs
  20% → Writing (README, articles, documentation) 156 hrs
  15% → Community (OSS + LinkedIn + Reddit + X)   117 hrs
```

**RULE:** No 8-hour weekend marathons. Sustainable 2-hour max weekday sessions.
Consolidation week every 6th week — review, no new concepts.

---

## HARDWARE PROCUREMENT (PHASED, INR-OPTIMIZED)

### Phase 1 Kit — Buy Month 0 (Before Day 1), ~₹17,000

| # | Component | INR | Source | Why Essential |
|---|---|---|---|---|
| 1 | Raspberry Pi 5 (4GB) | 7,500 | Amazon.in, Robu.in | Edge inference, llama.cpp, ONNX Runtime on ARM |
| 2 | Arduino Nano 33 BLE Sense | 3,200 | Robu.in, Evelta | TinyML MCU, onboard IMU + mic + temp, Edge Impulse first-class |
| 3 | STM32 Nucleo-F401RE | 2,200 | Amazon.in | ARM Cortex-M4, CMSIS-NN, STM32CubeAI target |
| 4 | MAX30102 (PPG/SpO2) | 280 | Amazon.in | Biosensing — heartrate, oxygen |
| 5 | AD8232 ECG Module + electrodes | 450 | Amazon.in, Robu | ECG capture |
| 6 | MPU-6050 IMU module | 220 | Any electronics | Gesture/motion data |
| 7 | USB-TTL Serial Adapter | 150 | Amazon.in | STM32 debug/flash |
| 8 | Breadboard 830 tie-point | 120 | Local electronics | Prototyping |
| 9 | Jumper wire kit (M-M, M-F, F-F) | 150 | Local electronics | Connections |
| 10 | 5V/3A USB-C power supply | 600 | Amazon.in | Stable Pi 5 power |
| 11 | 16GB microSD Class 10 | 350 | Amazon.in | Pi OS |
| 12 | USB-micro cable | 100 | Mobile shop | Arduino |
| **TOTAL** | | **~15,320** | | |

### Phase 2 Optional — Month 5-6 (~₹4,000-8,000)
- Google Coral USB Edge TPU: ₹4,000-6,000 (4 TOPS INT8, USB 3.0 → Pi 5)
- OR Raspberry Pi AI Kit (Hailo-8L NPU): ~₹8,000 (13 TOPS, M.2)

### SKIP LIST (overkill, return later)
- NVIDIA Jetson Orin Nano Super: ₹22,000+ → Year 2
- FPGA development boards: Year 2+ (unless compiler track confirmed)

---

## GPU CLUSTER STRATEGY (INR-OPTIMIZED)

### Free Tier (Months 1-8, exhaust before paying)

| Platform | GPU | Hours | Best For |
|---|---|---|---|
| **Kaggle Notebooks** | T4 (16GB) | 30 hrs/week | PRIMARY — QLoRA, TFLite conversion, small model training |
| **Google Colab Free** | T4 | ~12 hrs/day | Longer runs, hyperparameter search |
| **Lightning.ai Free** | A10G | 22 hrs/month | LLM fine-tuning demos |
| **GitHub Codespaces** | CPU only | 60 hrs/month | Code editing, README, Python analysis |

### Paid Tier (Months 7-18, ~₹2,000-5,000/month)

| Platform | INR/hr | Best For | Notes |
|---|---|---|---|
| **JarvisLabs (jarvislabs.ai)** | RTX 3090: ~₹40 | Experiments, flexibility | INR billing, per-minute, India-based |
| **E2E Networks** | From ₹49 | Data compliance, reliability | India datacenter, regulatory compliant |
| **Vast.ai** | A10: ~₹100-120 | Overnight training (checkpoint) | USD billing, peer-to-peer, cheapest |
| **RunPod Community** | A100: ~₹130-150 | Multi-GPU experiments | USD spot instances |

**Total GPU spend 18 months: ~₹35,000-50,000** (less than one premium certification)

**Workflow:** Train on Kaggle (free, versioned) → Download model → Convert locally → Flash to hardware

---

# ═══════════════════════════════════════════════════════
# PART ONE: MASTER COURSE CURRICULUM (TOPIC BY TOPIC)
# Industry-Relevant, 2026-Forward, Deeply Technical
# ═══════════════════════════════════════════════════════

> All courses below are FREE unless marked with cost. Sequence matters.

---

## DOMAIN 1: SIGNAL PROCESSING FUNDAMENTALS
### Why: ECG/PPG/IMU data is your entry into edge AI. You can't skip this.

**Topic 1.1: Signals, Systems, and Sampling Theory**
- Nyquist-Shannon theorem: Why you sample ECG at 360Hz (max signal = 180Hz)
- Aliasing: What happens below Nyquist + anti-aliasing filters
- Convolution in time vs multiplication in frequency domain
- Resource: MIT OpenCourseWare 6.003 Signals & Systems (free PDF + YouTube)
- Time needed: 6 hours reading + exercises

**Topic 1.2: Digital Filters (The Workhorse)**
- FIR vs IIR filters: stability, phase response, computational cost
- Butterworth bandpass (0.5–40Hz for ECG, 0.5–4Hz for PPG)
- Notch filter at 50Hz (India powerline) and 60Hz (US standards)
- Moving average vs exponential smoothing
- Implementation:
```python
from scipy.signal import butter, sosfilt, iirnotch
import numpy as np

def ecg_filter_bank(signal, fs=360):
    # Step 1: Bandpass 0.5-40 Hz
    sos_bp = butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
    bp_out = sosfilt(sos_bp, signal)
    # Step 2: Notch at 50Hz (India power line)
    b_n, a_n = iirnotch(50, Q=30, fs=fs)
    return sosfilt(sosfilt(sos_bp, np.zeros_like(signal)), signal)  # combined

# Why SOS (second-order sections) over BA coefficients?
# SOS is numerically stable for higher-order filters — cascaded biquads
```

**Topic 1.3: Frequency Domain Analysis**
- Fast Fourier Transform (FFT): extract dominant frequency from PPG
- Short-Time Fourier Transform (STFT): time-frequency localization for ECG events
- Welch's method: power spectral density for HRV frequency-domain features
- Implementation:
```python
from scipy.signal import welch, spectrogram
import matplotlib.pyplot as plt

# LF/HF ratio — key HRV metric
rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
freqs, psd = welch(rr_intervals, fs=4.0, nperseg=64)  # 4Hz sampling for RR
lf_band = psd[(freqs >= 0.04) & (freqs < 0.15)]   # 0.04-0.15 Hz
hf_band = psd[(freqs >= 0.15) & (freqs < 0.40)]   # 0.15-0.40 Hz
lf_hf_ratio = np.sum(lf_band) / np.sum(hf_band)   # <2 = healthy autonomic balance
```

**Topic 1.4: R-Peak Detection and Cardiac Cycles**
- Pan-Tompkins algorithm: the original 1985 QRS detector (still used in production)
- Hamilton-Tompkins: improved version, lower false positive rate
- NeuroKit2 implementation and why it combines multiple algorithms
- Features to extract: RR intervals, SDNN, RMSSD, pNN50, LF/HF
- Implementation:
```python
import neurokit2 as nk

ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
_, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
hrv_time = nk.hrv_time(rpeaks['ECG_R_Peaks'], sampling_rate=fs)
hrv_freq = nk.hrv_frequency(rpeaks['ECG_R_Peaks'], sampling_rate=fs)
hrv_nonlinear = nk.hrv_nonlinear(rpeaks['ECG_R_Peaks'], sampling_rate=fs)
```

**Topic 1.5: PPG Signal Processing**
- AC/DC separation: DC = ambient light, AC = pulsatile blood volume
- SpO2 calculation: ratio-of-ratios method (R = (AC_red/DC_red)/(AC_ir/DC_ir))
- PPG quality index (signal-to-noise ratio for motion artifact detection)
- Beat-to-beat interval extraction from PPG peaks

**Resources for Domain 1:**
- MIT OCW 6.003 Signals & Systems (YouTube + PDF): FREE
- NeuroKit2 documentation and tutorials: FREE → github.com/neuropsychology/NeuroKit
- PhysioNet tutorials: physionet.org/tutorials/ — FREE
- "Biomedical Signal Processing and Signal Modeling" Clifford et al.: Free PDF
- Time investment: 3 weeks part-time

---

## DOMAIN 2: DEEP LEARNING FOR TIME SERIES (YOUR BRIDGE)
### Why: Your BERT/NLP skills translate DIRECTLY — temporal sequences are temporal sequences.

**Topic 2.1: 1D Convolutions for Sequential Data**
- Why 1D-CNNs for time series: local pattern detection (QRS morphology)
- Dilated convolutions: increase receptive field without adding parameters
- Depthwise separable 1D convolutions: MobileNet-style efficiency for signals
- Implementation:
```python
class DilatedECGNet(nn.Module):
    """Efficient ECG classifier with dilated convolutions"""
    def __init__(self, num_classes=5):
        super().__init__()
        # Dilated convs: receptive field = 1 + 2*(k-1)*dilation
        self.features = nn.Sequential(
            # dilation=1: receptive field = 5
            nn.Conv1d(1, 32, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            # dilation=2: receptive field = 13
            nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            # dilation=4: receptive field = 29
            nn.Conv1d(32, 64, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Linear(64 * 16, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))
```

**Topic 2.2: Bidirectional LSTMs for ECG/IMU**
- Why BiLSTM captures bidirectional temporal context (like BERT but for signals)
- Hidden state size vs inference latency trade-off on edge hardware
- Attention over LSTM outputs: lightweight self-attention for edge
- Your BERT intuition: `[CLS]` token → global average pooling of LSTM states

**Topic 2.3: Transformer for Time Series (2025 Edge)** ← Industry trend
- PatchTST: temporal patching (like ViT patches but on signal windows)
- TimesNet: treat 1D time series as 2D for 2D CNN processing
- TSMixer: MLP-Mixer for time series — surprisingly competitive
- Challenge: standard transformers are too large for MCU deployment
- Solution: Tiny transformers (4 heads, 2 layers, 64-dim) + INT4 quantization

**Topic 2.4: Signal Augmentation for Edge AI**
- Time masking (inspired by SpecAugment for audio)
- Amplitude scaling (±20% random) — handles sensor drift
- Gaussian noise injection (simulates ADC quantization noise)
- Temporal shifting (±50ms) — handles variable R-peak alignment
- Implementation:
```python
class ECGAugmentation(nn.Module):
    def __init__(self, noise_std=0.01, scale_range=(0.8, 1.2)):
        super().__init__()
        self.noise_std = noise_std
        self.scale_range = scale_range

    def forward(self, x):
        if self.training:
            # Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std
            # Amplitude scaling
            scale = torch.empty(x.size(0), 1, 1).uniform_(*self.scale_range).to(x.device)
            x = x * scale
            # Time masking (mask random 10% of sequence)
            mask_size = int(0.1 * x.size(-1))
            mask_start = torch.randint(0, x.size(-1) - mask_size, (1,)).item()
            x[:, :, mask_start:mask_start + mask_size] = 0.0
        return x
```

**Topic 2.5: Class Imbalance in Medical Datasets**
- AFib is rare (5% of MIT-BIH recordings) — standard cross-entropy fails
- Focal Loss: down-weight easy negatives, focus on hard positives
- Weighted sampling: oversample minority (AFib) class
- Cost-sensitive learning: asymmetric loss (false negative = worse than false positive)
- Implementation:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()
```

**Resources for Domain 2:**
- Fast.ai Practical Deep Learning (Part 1 + 2): fast.ai — FREE
- PyTorch official tutorials: pytorch.org/tutorials — FREE
- "Deep Learning for ECG Classification" arXiv surveys (multiple 2023-2025 papers)
- Kaggle competitions: "Physionet Challenge" for ECG classification — FREE
- Time investment: 4 weeks part-time

---

## DOMAIN 3: MODEL COMPRESSION (THE CORE SKILL)
### Why: This is the EXACT skill that separates Applied ML from App-Layer ML.

**Topic 3.1: Post-Training Quantization (PTQ)**
- INT8 quantization math: x_quant = round(x / scale) + zero_point
- Scale and zero-point computation: per-tensor vs per-channel
- Symmetric vs asymmetric quantization (symmetric: zero_point=0, faster on MCU)
- Calibration: why 100-1000 representative samples beats random
- Implementation:
```python
import torch.ao.quantization as quant

# Per-channel is more accurate (different scale per filter)
model.qconfig = quant.QConfig(
    activation=quant.HistogramObserver.with_args(reduce_range=True),
    weight=quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    )
)
quant.prepare(model, inplace=True)
# Calibrate
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)
quant.convert(model, inplace=True)
```

**Topic 3.2: Quantization-Aware Training (QAT)**
- Fake quantization: simulate INT8 noise during FP32 training
- Straight-Through Estimator (STE): why gradients pass through rounding
- QAT vs PTQ: when accuracy gap justifies extra training cost
- BatchNorm folding: why BN must be folded before quantization
- Scale learning: trainable scales vs fixed calibrated scales
- Implementation:
```python
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# QAT pipeline (correct sequence matters!)
model.train()
model = prepare_qat_fx(model, {"": quant.get_default_qat_qconfig('qnnpack')})

# Train with fake quantization
for epoch in range(30):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
    # Convert BN stats after warmup
    if epoch == 10:
        model.apply(torch.ao.quantization.disable_observer)
    if epoch == 12:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# Convert to real INT8
model.eval()
model_int8 = convert_fx(model)
```

**Topic 3.3: Advanced Quantization — AWQ and SmoothQuant**
- AWQ (Activation-aware Weight Quantization): protect salient channels
  - Key insight: not all weight channels are equally important
  - Calibration identifies high-activation channels → higher precision budget
- SmoothQuant: migrate quantization difficulty from activations to weights
  - Divide activations by scale, multiply weights by scale
  - Keeps both activations AND weights quantizable
- When to use: LLM fine-tuning + edge SLM deployment
- Implementation:
```python
from awq import AutoAWQForCausalLM

# AWQ for edge SLM (Qwen2.5-1.5B for edge RAG)
model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
quant_config = {
    "zero_point": True,
    "q_group_size": 128,  # per-group quantization (128 weights per group)
    "w_bit": 4,           # INT4 weights — ~2x over INT8
    "version": "GEMM"     # GEMM-optimized dequantization
}
model.quantize(tokenizer, quant_config=quant_config)
# Result: 1.5B model in ~1GB → fits in Pi 5 RAM
```

**Topic 3.4: Structured Pruning**
- Magnitude pruning: remove weights below threshold (unstructured — poor hardware speedup)
- Structured pruning: remove entire filters/channels/attention heads
- L1-norm filter pruning: filters with small L1 norm contribute least
- Implementation:
```python
import torch.nn.utils.prune as prune

# Prune 30% of filters by L1 norm
for name, module in model.named_modules():
    if isinstance(module, nn.Conv1d):
        prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)
        prune.remove(module, 'weight')  # Make pruning permanent

# Fine-tune after pruning to recover accuracy
```

**Topic 3.5: Knowledge Distillation (Teacher-Student)**
- Soft labels: teacher's logits carry inter-class similarity info
- Intermediate distillation: match hidden layer representations (layer-wise KD)
- TinyBERT-style distillation: attention map distillation
- Response distillation vs feature distillation vs relation distillation
- Implementation (from Domain 1 code — expanded):
```python
class IntermediateDistillationLoss(nn.Module):
    """Distill from intermediate feature maps + logits"""
    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.T = temperature
        self.alpha = alpha  # weight for soft KD loss
        self.beta = beta    # weight for feature distillation

    def forward(self, s_logits, t_logits, s_features, t_features, labels):
        # Soft KD loss (logit-level)
        kl = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(s_logits / self.T, dim=1),
            F.softmax(t_logits / self.T, dim=1)
        ) * (self.T ** 2)
        # Feature distillation (intermediate representation matching)
        feat_loss = F.mse_loss(s_features, t_features.detach())
        # Hard loss
        ce = F.cross_entropy(s_logits, labels)
        return self.alpha * kl + self.beta * feat_loss + (1 - self.alpha - self.beta) * ce
```

**Topic 3.6: Neural Architecture Search for Edge (NAS)**
- MCUNet: two-stage NAS → first search architecture space, then find optimal
- Once-for-All (OFA): train one supernet, specialize to any hardware post-hoc
- ProxylessNAS: gradient-based NAS with hardware latency as constraint
- Hardware-aware NAS: include actual latency measurement on target device
- Implementation:
```python
# Using MCUNet tools for automatic model design
from mcunet.tinynas import TinyNAS

nas = TinyNAS(
    target_device="stm32f411",   # Cortex-M4 @ 100MHz
    latency_budget_ms=50,         # Target inference time
    peak_memory_kb=128,           # Flash constraint
    flash_budget_kb=256           # Total flash available
)
best_architecture = nas.search(dataset=your_ecg_dataset, num_trials=200)
print(f"Best model: {best_architecture.count_parameters()} params")
```

**Resources for Domain 3:**
- MIT 6.5940/6.S965 EfficientML.ai (Song Han, MIT): efficientml.ai — FREE
  - Lectures: Pruning, Quantization, NAS, KD, on-device training
  - Assignments: implement from scratch on PyTorch
- "Deep Compression" (Song Han 2016): arXiv:1510.00149 — FREE
- "MCUNet: Tiny Deep Learning on IoT Devices" (NeurIPS 2020): FREE
- "Once-for-All" (ICLR 2020): arXiv:1908.09791 — FREE
- PyTorch Quantization docs: pytorch.org/docs/stable/quantization — FREE
- Time investment: 6 weeks part-time (most important domain)

---

## DOMAIN 4: EDGE DEPLOYMENT RUNTIMES
### Why: Knowing HOW to train for edge is different from knowing HOW to deploy.

**Topic 4.1: TensorFlow Lite Micro (TFLite Micro)**
- Architecture: no malloc, no OS, static tensor arena
- Operator support: subset of TFLite ops supported on Cortex-M
- Arena sizing: how to calculate from model topology
- CMSIS-NN integration: ARM's optimized kernel backend for Cortex-M
- Model conversion pipeline:
```python
# Full conversion pipeline
import tensorflow as tf
import numpy as np

# Step 1: Export from PyTorch to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17)

# Step 2: ONNX → TFLite INT8
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

def representative_dataset():
    for i in range(200):
        yield [np.array(X_cal[i:i+1], dtype=np.float32)]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Step 3: Convert to C array (for Arduino/STM32 embedding)
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
# Then: xxd -i model.tflite > model_data.h
```

**Topic 4.2: ONNX Runtime (Production Choice)**
- Execution providers: CPU (ARM NEON), CUDA, CoreML, DirectML, TensorRT
- Graph optimization: constant folding, operator fusion, layout optimization
- Static vs dynamic quantization in ORT
- Profiling: which operators are bottlenecks
- Implementation (Pi 5 deployment):
```python
import onnxruntime as ort

# Enable all optimizations + ARM NEON EP
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4  # Pi 5 has 4 Cortex-A76 cores
opts.inter_op_num_threads = 1

# Profile to find bottlenecks
opts.enable_profiling = True
opts.profile_file_prefix = "ort_profile"

session = ort.InferenceSession("model_int8.onnx", opts,
                                providers=['CPUExecutionProvider'])

# Warmup (mandatory — first inference is slow due to JIT)
for _ in range(10):
    session.run(None, {"input": sample})

# Benchmark 100 inferences
import time
times = []
for _ in range(100):
    t0 = time.perf_counter()
    output = session.run(None, {"input": sample})
    times.append((time.perf_counter() - t0) * 1000)
print(f"Median: {np.median(times):.2f}ms | P99: {np.percentile(times, 99):.2f}ms")
```

**Topic 4.3: Apache TVM — The Compiler Approach**
- What TVM does: compiles a model graph into hardware-specific optimized code
- Relay IR: high-level graph representation (independent of framework)
- TensorIR / TIR: low-level tensor program IR (schedule-level)
- AutoTVM vs MetaScheduler: different tuning backends
- MicroTVM: TVM's backend for bare-metal MCU deployment
- Implementation:
```python
import tvm
from tvm import relay, auto_scheduler
import onnx

# Step 1: Import ONNX model into Relay
onnx_model = onnx.load("ecg_model.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, {"input": (1, 1, 500)})

# Step 2: Target specification
target = tvm.target.Target("llvm -mcpu=cortex-a76 -mattr=+neon,+fp-armv8,+fullfp16")

# Step 3: AutoScheduler tuning (run on Pi 5 — remote RPC)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=500,   # More trials = better optimization
    measure_callbacks=[auto_scheduler.RecordToFile("pi5_tuning.json")],
    verbose=1
)
tuner.tune(tune_option)

# Step 4: Build optimized library
with auto_scheduler.ApplyHistoryBest("pi5_tuning.json"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

# Step 5: Export for Pi 5
lib.export_library("ecg_pi5.so")  # Shared object, load on Pi 5 with tvm runtime

# Typical result: 2-3x speedup vs ONNX Runtime on same hardware
```

**Topic 4.4: llama.cpp — Quantized LLM Inference**
- GGUF format: quantized model format used by llama.cpp
- Quantization levels: Q4_K_M (recommended), Q5_K_M, Q8_0
- ARM NEON optimization: SIMD vectorization for matrix-vector products
- Context management: KV cache size vs memory trade-off
- Pi 5 capabilities: ~10-15 tokens/sec for 2B model, ~6-8 tokens/sec for 3B model
- Integration with your RAG system:
```python
import subprocess
import json

class LlamaCppInference:
    def __init__(self, model_path, n_ctx=2048, n_threads=4):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.binary = "./llama.cpp/llama-cli"

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        cmd = [
            self.binary,
            "-m", self.model_path,
            "-n", str(max_tokens),
            "--ctx-size", str(self.n_ctx),
            "--threads", str(self.n_threads),
            "--temp", str(temperature),
            "--repeat-penalty", "1.1",
            "-p", prompt,
            "--no-mmap"  # Faster on Pi 5 with sufficient RAM
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Extract generated text (strip the prompt echo)
        output = result.stdout
        if prompt in output:
            output = output[output.index(prompt) + len(prompt):]
        return output.strip()
```

**Topic 4.5: Edge Impulse Platform (Industry Standard)**
- EON Tuner: automated NAS for your hardware constraints
- DSP blocks: Edge Impulse's built-in signal processing pipeline
- Cross-platform deployment: one project → Arduino, STM32, Pi, Android
- Firmware library generation: C++ library with zero dependencies
- When to use vs manual TFLite: prototyping vs production customization

**Resources for Domain 4:**
- TFLite Micro documentation: tensorflow.org/lite/microcontrollers — FREE
- ONNX Runtime optimization guide: onnxruntime.ai/docs — FREE
- Apache TVM Tutorial series: tvm.apache.org/docs/tutorial — FREE
- MicroTVM tutorial (bare-metal MCU): tvm.apache.org/docs/how_to/work_with_microtvm — FREE
- llama.cpp wiki: github.com/ggerganov/llama.cpp/wiki — FREE
- Time investment: 5 weeks part-time

---

## DOMAIN 5: EMBEDDED SYSTEMS + ARM ARCHITECTURE
### Why: You can't optimize what you don't understand at the silicon level.

**Topic 5.1: ARM Cortex-M Architecture**
- Cortex-M0/M0+: no FPU, 32MHz, <50mW (ultrawearable)
- Cortex-M4: FPU + DSP extensions, 168MHz typical, CMSIS-NN optimized
- Cortex-M7: double-precision FPU, 400+ MHz, high-end MCU
- Cortex-A76 (Pi 5): Out-of-order, NEON SIMD, cache hierarchy
- Key concepts: SRAM vs Flash, bus matrix, DMA, peripheral clock gating

**Topic 5.2: CMSIS-NN — ARM's Optimized Neural Network Kernels**
- What it is: hand-optimized SIMD implementations of NN ops for Cortex-M
- Why critical: 3-5x faster than naive C for same silicon and power
- Operators: arm_convolve_s8, arm_fully_connected_s8, arm_softmax_s8
- How TFLite Micro uses CMSIS-NN: transparent if you link correctly
- Manual integration (when you need custom ops):
```c
#include "arm_math.h"
#include "arm_nnfunctions.h"

// CMSIS-NN INT8 matrix multiply (used in fully connected layers)
arm_status arm_fully_connected_s8(
    const cmsis_nn_context* ctx,
    const cmsis_nn_fc_params* fc_params,
    const cmsis_nn_per_tensor_quant_params* quant_params,
    const cmsis_nn_dims* input_dims,
    const int8_t* input_data,
    const cmsis_nn_dims* filter_dims,
    const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims,
    const int32_t* bias_data,
    const cmsis_nn_dims* output_dims,
    int8_t* output_data
);
// Performance: processes 4 MAC ops per cycle via DSP SIMD
```

**Topic 5.3: Memory Hierarchy for Edge ML**
- Register file: fastest, 13 general-purpose registers (ARM Cortex-M)
- L1 Cache: 16-64KB, 1 cycle access (Cortex-A only)
- SRAM: 256KB-2MB typical on Cortex-M, ~3-5 cycle access
- Flash: 512KB-2MB, READ-ONLY at runtime, ~10+ cycles
- External SDRAM: 8-128MB, ~100 cycles (Cortex-A with Pi)
- Impact on ML: model weights in Flash, activations in SRAM — SRAM is the bottleneck

**Topic 5.4: Power Analysis for Wearable AI**
- Active mode vs sleep mode current (mA vs μA)
- Dynamic power: CV²f (capacitance × voltage² × frequency)
- Duty cycling: run inference every N seconds, sleep between
- Measuring with a USB ammeter or Nordic PPK2 (₹8,000)
- Real numbers:
  - STM32 Cortex-M4 @ 84MHz inference: ~30-50mA (~150-250mW)
  - STM32 Stop mode: ~5μA (650x reduction)
  - Pi 5 idle: ~600mA; Pi 5 peak inference: ~2000mA

**Topic 5.5: Embedded C and C++ for ML Engineers**
- Pointers and memory management (malloc-free on TFLite Micro)
- Fixed-point arithmetic: shift-and-add instead of float multiply
- Inline assembly for critical hot loops (optional, advanced)
- FreeRTOS tasks: structuring ML inference as a periodic task
- I2C and SPI protocols: how your sensors communicate

**Resources for Domain 5:**
- "Embedded Systems Shape the World" (UTAustinX on edX): FREE audit
- ARM Cortex-M documentation: developer.arm.com — FREE
- CMSIS-NN documentation: ARM-software/CMSIS-NN GitHub — FREE
- "The Definitive Guide to ARM Cortex-M3 and Cortex-M4" (Yiu): optional book
- STM32CubeAI tutorial: st.com/en/embedded-software/x-cube-ai.html — FREE
- Time investment: 4 weeks part-time (run alongside Domain 4)

---

## DOMAIN 6: AI COMPILER FUNDAMENTALS (TOUCH, DON'T MASTER)
### Why: Literacy in this domain earns Qualcomm/ARM interviews. Full mastery takes years.

**Topic 6.1: What Compilers Do for ML (Conceptual)**
- Graph-level optimizations: constant folding, operator fusion, dead code elimination
- Operator fusion example: Conv + BatchNorm + ReLU → single fused kernel
  - Saves: 2 memory reads (BN input/output), 2 memory writes, pipeline flush
- Layout optimization: NCHW → NHWC for ARM NEON throughput
- Kernel specialization: generate different code for batch=1 vs batch>1

**Topic 6.2: MLIR — Modular Compiler Infrastructure**
- What MLIR is: Facebook/Google's multi-level IR framework (LLVM ecosystem)
- Dialects: domain-specific operation sets (linalg, affine, tosa, stablehlo)
- Lowering: progressive transformation from high-level to hardware-specific
- Why it matters: TensorFlow, IREE, MLIR-based PyTorch all use it
- Beginner path:
  - Complete the official Toy Language tutorial: mlir.llvm.org/docs/Tutorials/
  - Understand: parsing → IR → lowering → codegen pipeline
  - Goal: be able to READ MLIR IR, not write passes

**Topic 6.3: TVM Deep Dive (Beyond Topic 4.3)**
- Relay pass infrastructure: write a simple graph optimization pass
- Schedule primitives: split, tile, vectorize, parallel, unroll
- AutoTVM vs MetaScheduler: cost model differences
- MicroTVM for MCU: compile for STM32 without OS
- RISC-V backend: targeting Mindgrove/InCore chips
- Your TVM contribution path: add tutorial → fix documentation → fix bug → add feature

**Topic 6.4: RISC-V for ML Engineers**
- Why RISC-V: open ISA, no royalties, custom extensions possible
- Vector extension (RVV): SIMD for ML workloads (equivalent to ARM NEON)
- Indian context: Mindgrove MC-01 (first commercial Indian MCU), IIT Madras SHAKTI
- Getting started with QEMU emulation:
```bash
# Install tools
sudo apt-get install qemu-user-static gcc-riscv64-linux-gnu binutils-riscv64-linux-gnu

# Cross-compile your inference code
riscv64-linux-gnu-gcc -O2 -static -march=rv64gcv \
    ecg_inference.c -lm -o ecg_riscv

# Run in QEMU
qemu-riscv64-static ./ecg_riscv

# Measure cycle counts (RISC-V performance counters)
# cycle CSR = 0xC00
```

**Resources for Domain 6:**
- MLIR Tutorial (official): mlir.llvm.org/docs/Tutorials/ — FREE
- Apache TVM Paper: arxiv.org/abs/1802.04799 — FREE
- "MLIR: A Compiler Infrastructure for the End of Moore's Law": arxiv.org/abs/2002.11054 — FREE
- LLVM discussions forum for MLIR: discourse.llvm.org — FREE
- RISC-V International training: riscv.org/technical/specifications/ — FREE
- Time investment: 3 weeks literacy level (ongoing for 18 months)

---

## DOMAIN 7: ON-DEVICE LLM AND EDGE RAG (YOUR SUPERPOWER)
### Why: Your existing RAG/LLM skills + edge hardware = nobody else doing this combination.

**Topic 7.1: SLM (Small Language Model) Landscape 2026**
- Models that run on Pi 5 (4GB RAM):
  - Gemma 3 2B (Q4_K_M, ~1.3GB): 10-15 tok/s on Pi 5
  - Qwen2.5-1.5B (Q4_K_M, ~1.0GB): 12-18 tok/s, excellent for instructions
  - Phi-3-mini-3.8B (Q4_K_M, ~2.3GB): highest quality, slower (~8 tok/s)
  - Gemma 3 1B (Q4_K_M, ~0.7GB): fastest, limited reasoning
- Models that run on HIGH-end MCU (Cortex-M55 + Ethos-U85):
  - BERT-tiny (66M → ~35KB INT8): sentence embedding
  - DistilBERT-classification (66M → ~70KB INT8): classification
  - Direct LLM: NOT yet feasible on <256KB RAM MCU (in 2026)

**Topic 7.2: Efficient RAG for Edge**
- Embedding model selection: paraphrase-MiniLM-L3 (17MB), all-MiniLM-L6 (23MB)
- FAISS for edge: IVF index vs Flat index trade-offs at small scale
- Document preprocessing: chunking strategy for small context windows
- Hybrid search: BM25 + dense embeddings for better recall
- Implementation of the Edge Medical RAG System:
```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EdgeRAGSystem:
    """Production-ready edge RAG for Pi 5 with llama.cpp backend"""

    def __init__(self, embedding_model='paraphrase-MiniLM-L3-v2', 
                 llm_model_path='gemma3-2b-q4_k_m.gguf'):
        print("Loading embedder (17MB)...")
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.doc_store = {}
        self.doc_metadata = []
        self.llm_path = llm_model_path

    def add_documents(self, documents: list[dict]):
        """Add documents with metadata: [{'content': str, 'source': str, ...}]"""
        texts = [d['content'] for d in documents]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=True)
        dim = embeddings.shape[1]
        if self.index is None:
            # IVF index for >1000 docs, Flat for smaller collections
            if len(texts) < 1000:
                self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalize)
            else:
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, min(int(len(texts)**0.5), 100))
                self.index.train(embeddings.astype('float32'))
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.doc_metadata.extend(documents)

    def retrieve(self, query: str, k: int = 3, threshold: float = 0.4) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype('float32'), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx >= 0:
                results.append({**self.doc_metadata[idx], 'score': float(score)})
        return results

    def generate_with_context(self, query: str, 
                               sensor_context: dict = None,
                               max_tokens: int = 256) -> str:
        relevant_docs = self.retrieve(query, k=3)
        context_parts = []
        if sensor_context:
            context_parts.append(f"Patient vitals: {sensor_context}")
        for doc in relevant_docs:
            context_parts.append(f"[{doc.get('source', 'Reference')}]: {doc['content']}")
        
        prompt = f"""<context>
{chr(10).join(context_parts)}
</context>

Clinical Question: {query}
Evidence-based Answer:"""
        
        import subprocess
        result = subprocess.run([
            "./llama.cpp/llama-cli",
            "-m", self.llm_path,
            "-n", str(max_tokens),
            "--threads", "4",
            "--ctx-size", "2048",
            "--temp", "0.3",        # Low temperature for medical context
            "--repeat-penalty", "1.1",
            "-p", prompt, "--log-disable"
        ], capture_output=True, text=True, timeout=180)
        return result.stdout.strip()
```

**Topic 7.3: Fine-Tuning SLMs for Edge Deployment**
- Domain adaptation: fine-tune Qwen2.5-1.5B on medical/biosensing Q&A
- LoRA for edge models: rank 4-8 sufficient for SLMs (vs rank 16+ for 7B)
- GGUF conversion: PyTorch → GGUF for llama.cpp deployment
- Inference pipeline integration with your sensors

**Resources for Domain 7:**
- llama.cpp project wiki: github.com/ggerganov/llama.cpp — FREE
- HuggingFace PEFT docs: huggingface.co/docs/peft — FREE
- MLX (Apple Metal), ExecuTorch (Meta) — alternative edge LLM runtimes
- Ollama: ollama.ai (local LLM management, easy to start)
- Time investment: 3 weeks (builds on your existing LLM skills)

---

# ═══════════════════════════════════════════════════════
# PART TWO: THE 18-MONTH PHASED IMPLEMENTATION PLAN
# ═══════════════════════════════════════════════════════

---

# PHASE 1: FOUNDATIONS (MONTHS 1-3)
## "Build the foundation. First hardware projects. First signal processing. OSS begins."

### MONTH 1 — WEEK-BY-WEEK BREAKDOWN

#### Week 1: Environment + Signal Processing Launch

**Day 1-2 (3 hrs): Full Environment Setup**
```bash
# Conda environment
conda create -n edgeai python=3.11
conda activate edgeai

# Core ML + signal processing
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow onnx onnxruntime onnx-simplifier
pip install neurokit2 biosppy scipy heartpy wfdb pyserial
pip install torchao optimum quanto neural-compressor
pip install edgeimpulse

# Data/viz/monitoring
pip install wandb streamlit plotly pandas matplotlib
pip install faiss-cpu sentence-transformers

# Dev
pip install torchinfo fvcore thop jupyter

# Arduino IDE + STM32CubeIDE: install separately
# Edge Impulse CLI: npm install -g edge-impulse-cli
# llama.cpp: git clone + make LLAMA_OPENBLAS=1
```

**Day 2-3 (3 hrs): PhysioNet Dataset Download + ECG Exploration**
- Download MIT-BIH Arrhythmia Database (physionet.org, free account)
- Download PTB-XL ECG Dataset (21,837 records)
- Implement Domain 1 Topics 1.1-1.4 (full ECG filter bank + R-peak detection)
- Commit `week1/ecg_pipeline.py` to your `edge-ml-pivot` GitHub repo

**Day 4-5 (3 hrs): PPG Pipeline + Hardware Arrival**
- Implement PPG signal processing (MAX30102 simulation using NeuroKit2)
- Connect MAX30102 to Pi 5 via I2C, capture first raw readings
- Connect Arduino Nano 33 BLE Sense via USB, flash blink sketch

**Weekend (4 hrs): LinkedIn Profile Optimization + First Post**
- Headline update, Edge AI skills added
- First LinkedIn post: Week 1 ECG visualization (attach ecg_baseline.png)
- Create Twitter/X account or pivot existing to tech focus
- Join subreddits: r/MachineLearning, r/embedded, r/RISCV, r/raspberry_pi

#### Week 2: Model Training + Quantization Basics

**Day 6-7 (3 hrs): Train 1D-CNN on ECG (Kaggle GPU)**
```python
# Kaggle Notebook header (always use this):
# Hardware target: Arduino Nano 33 BLE Sense (256KB RAM, 1MB Flash)
# Then implement TinyECGNet from plan_cc.md (TinyECGNet class)
# + your Domain 2 Topic 2.1 DilatedECGNet
# Compare both on MIT-BIH — which fits hardware constraints better?
```

**Day 8 (1.5 hrs): PTQ INT8 Baseline**
- Apply post-training quantization to trained TinyECGNet
- Measure: model size FP32 → INT8, latency, accuracy drop
- Commit benchmark table to `week2/quantization_benchmark.md`

**Day 9-10 (3 hrs): IMU Gesture Data Collection (Arduino)**
- Flash `collect_imu.ino` (from plan_cc.md Week 2)
- Collect 5 gesture classes × 50 samples each
- Save to `week2/data/gesture_data.csv`

#### Week 3-4: Hardware Deployment + Integration

**Weeks 3-4 deliver:**
- Gesture classifier deployed to Arduino Nano 33 BLE Sense
- ECG INT8 pipeline running on Pi 5 via ONNX Runtime
- Benchmark table: FP32 vs PTQ vs QAT on both hardware targets
- First OSS PR submitted (NeuroKit2 documentation)
- LinkedIn posts 2 and 3 published

**Month 1 Deliverables:**
```
[ ] GitHub repo: edge-ml-pivot (public by Week 4)
[ ] 2 trained models: TinyECGNet + GestureNet (TFLite INT8)
[ ] Arduino gesture demo working live
[ ] 4 LinkedIn posts published
[ ] 2 Twitter threads posted
[ ] 1 Reddit effortpost in r/embedded
[ ] 1 OSS PR submitted (NeuroKit2)
[ ] ECG pipeline on Pi 5 (ONNX Runtime INT8)
```

---

### MONTH 2 — STM32 Deployment + Signal Processing Depth

**Month 2 Goals:**
- Deploy PPG Anomaly model to STM32 Nucleo-F401RE via STM32CubeAI
- Implement full HRV feature extraction pipeline
- Train AFib detector (first version, FP32)
- Complete MIT 6.S965 EfficientML lectures 1-4

**Project 2.1: PPG Anomaly Detector on STM32**
- Dataset: PTB-XL subset (normal vs ST-elevation pattern)
- Model: PPGAnomalyNet (<30K params, INT8 target <30KB)
- Deploy via STM32CubeAI → measure inference at 84MHz
- Target: <50ms inference, <128KB RAM

**Project 2.2: HRV Feature Dashboard**
- Real-time HRV computation from ECG + PPG combined
- Streamlit dashboard with SDNN, RMSSD, LF/HF ratio
- Historical trend (store 24h of measurements in SQLite)

---

### MONTH 3 — Knowledge Distillation + First OSS Merge

**Month 3 Goals:**
- Implement knowledge distillation (Large ECG teacher → Tiny student)
- QAT on AFib detector first version
- FIRST OSS PR MERGED (target: NeuroKit2 or TFLite Micro)
- Article 1 published: "From Data Scientist to Edge AI Engineer: Month 1-3"
- Harvard CS249r TinyML course completed

**Project 3.1: Knowledge Distillation Study**
```python
# Teacher: DilatedECGNet (64K params, ~250KB INT8)
# Student: TinyECGNet (8K params, ~30KB INT8)
# Distillation: IntermediateDistillationLoss (Domain 3 Topic 3.5)
# Goal: student reaches within 2% of teacher AUC
# Log: wandb experiment tracking (free tier)
```

**Month 3 Deliverables:**
```
[ ] Distillation experiment: teacher vs student accuracy comparison
[ ] QAT pipeline documented end-to-end
[ ] 1 OSS PR MERGED (NeuroKit2 or TFLite Micro docs)
[ ] Article 1 published to Substack/Medium
[ ] Harvard CS249r completed
[ ] Read: MCUNet, Deep Compression, Once-for-All papers
[ ] LinkedIn followers: 100+ in edge AI niche
```

---

# PHASE 2: HARDWARE DEPTH (MONTHS 4-6)

## Month 4 — CMSIS-NN + STM32 Production Pipeline

**Topics:** Domain 5 (ARM Architecture), Domain 4 Topic 4.1 (TFLite Micro depth)

**Project 4.1: AFib Detector Version 2 on STM32**
- 1D-CNN + BiLSTM (from Domain 2 Topic 2.2)
- INT8 QAT to <64KB model size
- Deploy via STM32CubeAI, target <100ms @ 84MHz
- Measure CMSIS-NN vs naive C vs TFLite Micro default

**Study: ARM CMSIS-NN source code**
- Read: ARM-software/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
- Understand: how SIMD instructions process 4 INT8 MACs per cycle
- This understanding separates you from every "ML engineer" who just uses the framework

## Month 5 — ONNX Runtime + Multi-Vital System

**Topics:** Domain 4 Topic 4.2 (ORT depth), Domain 7 Topic 7.1 (SLM intro)

**Project 5.1: Multi-Vital Signs Monitor (Pi 5)**
```
Architecture:
AD8232 ECG + MAX30102 PPG → Pi 5 GPIO → Python pipeline
→ ONNX Runtime INT8 (AFib + anomaly) → Results
→ Streamlit real-time dashboard (HR, SpO2, AFib risk)
```

**Benchmark Study: ORT vs TFLite vs TVM (first comparison)**
- Same model, three runtimes, same Pi 5 hardware
- Metrics: cold start, warm inference, memory footprint
- Publish as LinkedIn article (your most technical piece yet)

## Month 6 — Edge Impulse + Compression Benchmark Study

**Topics:** Domain 3 Topics 3.1-3.5 (full compression suite)

**Project 6.1: Fall Detection Wearable (Edge Impulse)**
- IMU (Arduino Nano 33) → Edge Impulse Studio → EON Tuner
- Deploy: both Arduino AND Pi 5 from same project (demonstrates portability)

**Project 6.2: THE COMPRESSION BENCHMARK STUDY**
```python
# Apply 5 techniques to same base model (ECG binary classifier)
# Baseline FP32 → PTQ INT8 → QAT INT8 → Pruning 50% → Distillation → AWQ INT4
# Measure: accuracy, size_mb, latency_ms (Pi 5), latency_ms (STM32), ram_kb
# This GitHub repo will be referenced in interviews for years
```

**Phase 2 Deliverables:**
```
[ ] AFib Detector v2 on STM32 (AUC >0.92, <64KB, <100ms)
[ ] Multi-Vital Monitor live on GitHub (full README + benchmark)
[ ] Fall Detection on Edge Impulse (>90% accuracy)
[ ] Compression Benchmark Study (5 techniques, published)
[ ] 3 OSS PRs merged total
[ ] 6 LinkedIn articles published (total)
[ ] MIT 6.5940 EfficientML lectures 1-10 completed
```

---

# PHASE 3: DOMAIN DEPTH + COMPILER TOUCH (MONTHS 7-9)

## Month 7 — Complete AFib Pipeline + PhysioNet Depth

**Project 7.1: Production AFib Detector (HERO v1)**
- Dataset: PhysioNet MIT-BIH (full 48 records) + CPSC 2018 challenge data
- Model: 1D-CNN + BiLSTM (Domain 2), with attention gate
- Training: QAT from epoch 1, focal loss for class imbalance
- Evaluation: 5-fold cross-validation, confusion matrix, AUC-ROC
- Deployment: STM32 (<64KB, <100ms) + Pi 5 (ONNX Runtime, <10ms)
- GitHub: `wearable-afib-detector` (this project will get you noticed)

## Month 8 — Apache TVM Deep Dive

**Project 8.1: TVM vs ONNX Runtime Comprehensive Benchmark**
- Auto-tune for Cortex-A76 (Pi 5) AND Cortex-A53 (older hardware)
- Compare: untuned TVM vs AutoTVM vs MetaScheduler vs ONNX Runtime
- Typical result: 2-3x speedup for convolution-heavy models
- Publish as your most technical GitHub README (with Mermaid architecture diagrams)

**Project 8.2: RISC-V QEMU Experiment**
- Cross-compile ECG inference to RISC-V (RV64GC)
- Compare QEMU cycle counts: ARM vs RISC-V for same operation
- GitHub: `tinyml-riscv-qemu` (targets Mindgrove, IIT Madras SHAKTI interest)

## Month 9 — Edge Medical RAG (Your Unique Project)

**Project 9.1: Sovereign Health Edge RAG**
- Deploy Gemma 3 2B Q4_K_M on Pi 5 (llama.cpp)
- Integrate with real-time vital signs from Pi 5 pipeline
- FAISS index of medical literature (PubMed abstracts, PhysioNet docs)
- Full Streamlit dashboard: sensor → RAG → LLM → insight
- GitHub: `edge-medical-rag-pi5`
- Why unique: NO other engineer has built RAG + biosensing + edge inference in one system

**Phase 3 Deliverables:**
```
[ ] AFib Detector v3 (AUC >0.93, published benchmark)
[ ] TVM benchmark study live (comparison with ONNX Runtime)
[ ] RISC-V QEMU experiment documented
[ ] Edge Medical RAG working on Pi 5
[ ] MLIR toy tutorial completed
[ ] 5 OSS PRs merged total
[ ] 9 technical articles published
[ ] 10 technical Twitter/X threads
```

---

# PHASE 4: HERO PROJECT (MONTHS 10-12)

## THE SOVEREIGN HEALTH SYSTEM (Your Career-Defining Project)

**Full System Architecture:**
```
┌─ SENSOR LAYER ─────────────────────────────────────────┐
│  AD8232 ECG ──┐                                        │
│  MAX30102 PPG ├──► STM32 Nucleo-F401RE (ARM Cortex-M4) │
│  MPU-6050 IMU ┘    ├── AFib detector (58KB INT8)       │
│                    ├── Fall detector (22KB INT8)        │
│                    └── UART → Raspberry Pi 5            │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─ COMPUTE LAYER ────────────────────────────────────────┐
│  Raspberry Pi 5 (4GB)                                  │
│  ├── ONNX Runtime INT8 (secondary vital analysis)      │
│  ├── llama.cpp: Gemma 3 2B Q4_K_M                     │
│  │   └── FAISS RAG (medical knowledge base)            │
│  └── Streamlit: Real-time dashboard                    │
│       ├── ECG/PPG waveforms                            │
│       ├── AFib risk score + trend                      │
│       ├── SpO2 + HR chart (24h)                        │
│       └── Medical Q&A chatbot                          │
└─────────────────────────────────────────────────────────┘
```

**What makes this project un-copyable:**
1. TWO inference engines on different silicon (STM32 + Pi 5)
2. COMPLETE pipeline: raw analog electrons → LLM-powered health insight
3. ZERO cloud dependency — fully sovereign AI
4. Real PhysioNet medical dataset, real evaluation, real numbers
5. Bridges your existing RAG/LLM skills with new hardware expertise
6. Nobody else has built this exact combination

**GitHub: `sovereign-health-edge-ai`**
**Deliverables:**
- Mermaid architecture diagram (GitHub renders natively)
- Screen recording demo GIF (Streamlit dashboard, NO FACE)
- Benchmark table: all models, all hardware, all metrics
- 3-5 page technical report (PDF, arXiv-style)
- Kaggle notebook (public, for discoverability)

**Phase 4 Deliverables:**
```
[ ] Hero project complete with comprehensive README
[ ] Demo GIF created (screen recording, no face)
[ ] 8 OSS PRs merged total (LinkedIn posts for each)
[ ] 12 technical articles total
[ ] Resume updated: Edge AI section with real numbers
[ ] Applications to Tier 1 companies initiated (Sophrosyne, Mindgrove)
[ ] LinkedIn: 500+ followers in edge AI niche
[ ] Twitter: 300+ technical followers
```

---

# PHASE 5: ADVANCED + REMOTE PREP (MONTHS 13-15)

## Month 13 — AWQ + On-Device Fine-Tuning
- AWQ quantization (Domain 3 Topic 3.3) on edge SLM
- TinyTL: on-device transfer learning (fine-tune on Pi 5 itself)
- ORT Mobile: ONNX Runtime for Android deployment (expand platform reach)

## Month 14 — MLIR Contribution + NAS
- MLIR contribution: tutorial or test case (first compiler-adjacent PR)
- MCUNet NAS for your ECG dataset (find optimal architecture automatically)
- Domain 6 Topics 6.2-6.3 (MLIR + TVM passes)

## Month 15 — Remote Positioning + Resume Polish
- Resume fully optimized for physical AI roles
- Turing.com + Toptal + Crossover profiles created
- Applications to Tier 1 (if not already hired): Sophrosyne, Mindgrove, Netrasemi
- Applications to Tier 3 remote: Edge Impulse (US), embedded AI startups

---

# PHASE 6: INTERVIEW + TRANSITION (MONTHS 16-18)

## Technical Interview Deep Preparation

**The 10 Questions + Model Answers:**

**Q1: "Explain the quantization pipeline for deploying BERT on a 256KB MCU"**
→ Never force BERT on 256KB. Redesign: DistilBERT task-specific head → structured pruning → INT8 QAT → TFLite Micro. OR: replace BERT with a 1D-CNN designed from scratch for the specific task. Hardware-aware design beats model forcing.

**Q2: "PTQ vs QAT — when would you use each?"**
→ PTQ: fast (no retraining), ~1-3% accuracy drop, use when time-constrained or model is large.
→ QAT: slower (full training), <0.5% accuracy drop, use for medical/safety-critical accuracy.
→ Personal insight: For biosignal AFib detection (AUC threshold of 0.92+), QAT was the right call — 0.4% accuracy difference matters clinically.

**Q3: "What is CMSIS-NN and why does it matter?"**
→ ARM's hand-optimized neural network kernel library for Cortex-M. Uses DSP SIMD extensions to process 4 INT8 MAC operations per cycle vs 1 for naive C. Transparently accelerates TFLite Micro inference. Knowing CMSIS-NN is what separates a TinyML practitioner from someone who just runs Edge Impulse.

**Q4: "How does the TVM auto-tuner work?"**
→ Extracts computational tasks (conv2d, dense) from Relay IR. For each task, generates multiple schedule variants (different tile sizes, unroll factors, vectorization strategies). Measures actual hardware latency via runtime profiler. Uses machine learning cost model (XGBoost in AutoTVM, random forest in MetaScheduler) to guide search. Result: 2-3x speedup over default schedule for ARM Cortex-A targets.

**Q5: "Design a system that detects AFib on a wearable with no cloud"**
→ [Use your exact project numbers: 58KB model, 88ms @ 84MHz, 64KB RAM, MIT-BIH AUC 0.93]
→ This is not theory — this is your actual deployed system.

**Q6: "Operator fusion — what is it and why does it matter for edge inference?"**
→ Merging Conv + BN + ReLU into a single fused kernel. Eliminates 2 intermediate tensor reads/writes. Critical on Cortex-M where SRAM bandwidth is the bottleneck (not compute). Without fusion: 3 kernel launches, 2 memory round-trips. With fusion: 1 kernel launch, 0 intermediate memory writes.

**Q7: "What is INT8 quantization and how are scale/zero-point calculated?"**
→ Map FP32 range [min_val, max_val] to INT8 range [-128, 127].
→ scale = (max_val - min_val) / 255; zero_point = round(-min_val / scale) - 128
→ Symmetric quantization: zero_point = 0 (faster on hardware that doesn't support zero-point bias).
→ Per-channel weights are more accurate than per-tensor.

**Q8: "Explain the trade-offs between TFLite Micro, ONNX Runtime, and Apache TVM"**
→ TFLite Micro: bare-metal MCU (no OS, no malloc), static arena, CMSIS-NN backend, broadest MCU support.
→ ONNX Runtime: Linux/Android/Pi, execution provider architecture, easiest to use for ARM, good for Pi 5.
→ TVM: Requires compilation step per hardware target, highest peak performance, auto-tuning for custom hardware, compiler-level work.

---

# ═══════════════════════════════════════════════════════
# PART THREE: MASTER GITHUB REPOSITORY REFERENCE
# Curated, Starred, Categorized — 2026 Industry-Relevant
# ═══════════════════════════════════════════════════════

## CATEGORY 1: PRIMARY INFERENCE ENGINES

| Repo | Stars | Your Use | OSS Contribution Target |
|---|---|---|---|
| tensorflow/tflite-micro | 2.1k | MCU deployment (STM32, Arduino) | Month 5: doc fix, Month 9: test case |
| microsoft/onnxruntime | 14k | Pi 5 deployment, ONNX optimization | Month 8: mobile optimization |
| apache/tvm | 11k | Compiler-level optimization, auto-tuning | Month 8: Pi 5 tutorial |
| ggerganov/llama.cpp | 68k | Edge LLM on Pi 5 | Month 9: ARM benchmark addition |
| mlc-ai/mlc-llm | 20k | Multi-backend edge LLM (Metal, CUDA, Vulkan) | Month 14: documentation |
| ollama/ollama | 96k | Local LLM management, easy to start | Reference, not contribution target |

## CATEGORY 2: MODEL COMPRESSION + OPTIMIZATION

| Repo | Stars | Your Use | OSS Contribution Target |
|---|---|---|---|
| mit-han-lab/mcunet | 1.2k | Hardware-aware NAS for MCU | Month 13: add ECG experiment |
| mit-han-lab/once-for-all | 2.4k | Supernet-based NAS | Study only |
| huggingface/optimum | 2.3k | ONNX export + quantization | Month 10: add biosignal example |
| casymcc/awq | 3.8k | AWQ INT4 quantization | Month 13: implement + benchmark |
| SonySemiconductorSolutions/mct-model-optimization | 700 | Production quantization toolkit | Month 11: example addition |
| neuralmagic/sparseml | 2.1k | Structured + unstructured pruning | Month 6: tutorial contribution |
| onnx/onnxmltools | 800 | ONNX model transformation | Reference |

## CATEGORY 3: HARDWARE + EMBEDDED PLATFORMS

| Repo | Stars | Your Use |
|---|---|---|
| ARM-software/CMSIS-NN | 1.8k | Understanding optimized kernels |
| ARM-software/CMSIS_5 | 1.3k | ARM DSP library reference |
| edgeimpulse/example-standalone-inferencing | 500 | Arduino/STM32 deployment examples |
| STMicroelectronics/X-CUBE-AI | — | STM32CubeAI CLI reference |
| arduino/ArduinoTensorFlowLite | 600 | TFLite Micro on Arduino |
| micropython/micropython | 19k | MicroPython for MCUs (optional path) |

## CATEGORY 4: BIOSENSING + MEDICAL AI

| Repo | Stars | Your Use | OSS Contribution Target |
|---|---|---|---|
| neuropsychology/NeuroKit | 2.6k | ECG/PPG processing (your first PR target) | Month 3: first PR |
| AI4HealthUOL/* | Various | ECG-FM benchmarking, PPG datasets | Study + cite |
| upsidedownlabs/Rpeak | 400 | Real-time ECG web analysis | Reference |
| holtzy/ecg-tinyml | 200 | ECG TinyML deployment examples | Study |
| bmi08f/ecg-classification | 300 | MIT-BIH classification examples | Study |
| MIT-LCP/wfdb-python | 700 | PhysioNet data loading (wfdb library) | Study + use |

## CATEGORY 5: ML COMPILERS + RISC-V

| Repo | Stars | Your Use | OSS Contribution Target |
|---|---|---|---|
| llvm/llvm-project (MLIR) | 30k | MLIR literacy, toy tutorial | Month 14: tutorial contribution |
| riscv/riscv-gnu-toolchain | 3.2k | RISC-V cross-compilation | Setup only |
| MindgroveInc/MindgroveSOC | — | Target platform awareness | Watch + star |
| shaktiproject/SHAKTI-SoC | 700 | IIT Madras RISC-V SoC (contribute) | Month 13+ |

## CATEGORY 6: CURATED LISTS + LEARNING RESOURCES

| Repo | Stars | Your Use |
|---|---|---|
| crespum/edge-ai | 800 | Comprehensive edge AI resources list |
| wangxb96/Awesome-EdgeAI | 1.2k | Survey-style edge AI overview |
| mit-han-lab/efficientml.ai | 600 | MIT 6.5940 course materials |
| DataTalksClub/mlops-zoomcamp | 11k | MLOps production pipeline |
| GokuMohandas/Made-With-ML | 37k | End-to-end production ML systems |

## CATEGORY 7: LLM FINE-TUNING + PEFT (YOUR BRIDGE)

| Repo | Stars | Your Use | OSS Contribution Target |
|---|---|---|---|
| huggingface/peft | 16k | LoRA, QLoRA for SLM fine-tuning | Month 14: add biosignal example |
| huggingface/trl | 10k | RLHF, DPO, SFT training | Month 10: tool-calling notebook |
| hiyouga/LLaMA-Factory | 37k | Production fine-tuning platform | Reference |
| vllm-project/vllm | 40k | High-throughput inference server | Month 13: model card |
| microsoft/DeepSpeed | 35k | Distributed training | Study |

---

# ═══════════════════════════════════════════════════════
# PART FOUR: ONLINE PRESENCE SYSTEM (CAMERA-SHY EDITION)
# ═══════════════════════════════════════════════════════

## THE CORE PHILOSOPHY: "Authority Through Clarity"

You don't need a face. You need specificity, numbers, and code.
The engineer who writes "38ms inference on STM32 @ 84MHz" is more credible
than one who says "I work on edge AI." Hardware numbers are unfakeable authority.

## LINKEDIN SYSTEM (Primary Platform)

**Profile Architecture:**
- Banner: Circuit board aesthetic (Canva template, no photo needed)
- Headline: "Senior ML Engineer | Edge AI + TinyML | Biosensing + Open Source | India"
- About: 300 words, first-person, specific projects and numbers
- Featured: Pin 3 repos (sovereign-health, afib-detector, compression-benchmark)
- Licenses: TinyML Professional Certificate when earned

**Content Calendar (10+ posts/month from Month 10):**

| Post Type | Frequency | Format | Time |
|---|---|---|---|
| Project benchmark update | 2/month | Text + benchmark table image | 20 min |
| Technical insight | 3/month | PDF carousel (5-8 Canva slides) | 45 min |
| Paper summary | 1/month | Structured text (300 words) | 30 min |
| OSS contribution announcement | 1/month | Link to merged PR + explanation | 15 min |
| India semiconductor industry insight | 1/month | Commentary + data | 20 min |
| Personal learning log | 2/month | Short-form "learned X, surprised by Y" | 10 min |

**PDF Carousel Topics (No Face, No Camera):**
1. "5 things that happen when you quantize a neural network to INT8" (8 slides)
2. "TFLite Micro vs ONNX Runtime vs TVM: my benchmark on Pi 5"
3. "How I built a medical AI on 17,000 INR of hardware"
4. "India's chip startups are hiring ML engineers — here's what they actually want"
5. "CMSIS-NN explained: why ARM's kernels are 4x faster than your naive C"
6. "The quantization cheat sheet: PTQ vs QAT vs AWQ vs INT4 — when to use each"

**Engagement Rules:**
- Reply to 5 posts/day in edge AI, TinyML, India semiconductor niches
- Never comment "great post!" — always add a specific technical point
- Tag researchers when you implement their paper (Tag @hanlab_mit, @EdgeImpulse)
- Reply within 2 hours to any comment on your posts (boosts algorithm)

## TWITTER/X SYSTEM (Technical Network)

**Key Accounts to Follow (do immediately):**
```
@tinyMLsummit    - Foundation events and community
@EdgeImpulse     - Platform updates + community challenges
@hanlab_mit      - MIT Han Lab research (quantization, NAS)
@ApacheTVM       - TVM compiler project
@mlc_ai          - MLC-LLM, edge LLM inference
@qualcommai      - Qualcomm AI research
@Arm_Research    - ARM ML research
@embedded_ml     - Embedded ML community (follow + engage)
@physionet_news  - Medical dataset community
@jerryjliu0      - MLC-LLM creator (engage meaningfully)
@pmarca          - Tech macro (broader reach)
```

**Thread Template (Your High-Performing Format):**
```
Thread title formula: "I [did specific thing] on [specific hardware]. Here's what I learned:"
Hook: specific number ("38ms inference on a 2,200 INR STM32")
1/ The number [metric] in context [why it matters]
2/ The unexpected finding [something counterintuitive]
3/ The technical explanation [why it happens]
4/ Code snippet or benchmark table [proof]
5/ What breaks and how to fix it [honest failure log]
6/ GitHub link [credibility]
7/ Question to audience [engagement]
#EdgeAI #TinyML #EmbeddedML [3 hashtags max]
```

## REDDIT SYSTEM (Credibility Builder)

**Subreddits and Strategies:**

| Subreddit | Strategy | Frequency |
|---|---|---|
| r/MachineLearning | Long-form project write-ups (1500+ words) | 1/month |
| r/embedded | Technical answers + project references | Daily reading, 3/week answers |
| r/LocalLLaMA | Edge LLM experiments (your Pi 5 RAG system) | 2/month posts |
| r/RISCV | RISC-V TinyML experiments | 1/month post |
| r/raspberry_pi | Pi 5 ML benchmarks, biosensing projects | 2/month posts |
| r/India | India semiconductor opportunity pieces | 1/month |
| r/cscareerquestions | Share your pivot journey + advice | As relevant |

**Reddit Post Formula:**
```
Title: [P] [specific technical claim] — [hardware + result]
Example: [P] I deployed an AFib detector to a 2,200 INR STM32 — 
         58KB model, 88ms inference, AUC 0.93

Body:
1. What I built and why (2 paragraphs)
2. Technical architecture (Mermaid diagram or ASCII art)
3. Code snippets (the critical parts)
4. Benchmark results (markdown table)
5. What broke (honest failure log — Reddit loves this)
6. Key learnings
7. GitHub link
8. Open questions for the community

Never: self-promotional without technical substance
Always: invite discussion, respond to every comment
```

## TECHNICAL WRITING SYSTEM (Substack or Medium)

**Publication Schedule (1 article/month from Month 6):**

| Month | Article | Target Audience |
|---|---|---|
| 6 | "From Data Scientist to Edge AI: What I Built in 6 Months" | ML engineers considering pivot |
| 7 | "Building an AFib Detector on a 2,200 INR Chip" | Biosignal + TinyML community |
| 8 | "INT8 vs QAT vs AWQ vs Pruning: A Practitioner's Benchmark" | ML compression engineers |
| 9 | "India's Biosensing Semiconductor Boom: What ML Engineers Need" | India tech community |
| 10 | "Apache TVM vs ONNX Runtime on Pi 5: Real Numbers" | Edge AI infrastructure |
| 11 | "Running Gemma 3 on 7,500 INR Hardware: Complete Guide" | Edge LLM community |
| 12 | "The Engineer Who Bridges AI Models and Silicon: 12-Month Playbook" | Career pivots |
| 13 | "RISC-V for ML Engineers: India's Chip Future" | India semiconductor |
| 14 | "How I Got My First PR Merged to Apache TVM (Step-by-Step)" | OSS aspirants |
| 15 | "CMSIS-NN Explained: Why ARM's Kernels Are 4x Faster" | Embedded ML |
| 16 | "Signal Processing for ML Engineers: ECG, PPG, IMU Fundamentals" | Domain expansion |
| 17 | "On-Device Medical RAG: Architecture and Benchmarks" | Edge LLM + medical |
| 18 | "18 Months: Senior Data Scientist to Edge AI Engineer (Honest Log)" | Everyone |

---

# TARGET COMPANIES + APPLICATION STRATEGY

## Tier 1: India, Direct Domain Match (Apply Month 10-12)

| Company | Location | Role Type | Remote? | Why You Fit |
|---|---|---|---|---|
| Sophrosyne Technologies | Bengaluru (HSR) | ML Engineer - Biosensing SoC | Hybrid possible | AFib project directly matches their biosensing chip |
| Mindgrove Technologies | Chennai/Bengaluru | Embedded ML Engineer | Partial remote | RISC-V experiments + your TVM knowledge |
| Netrasemi | Thiruvananthapuram | Edge AI ML Engineer | Some remote for ML | ONNX Runtime + model optimization |
| Saankhya Labs | Bengaluru | Signal Processing + ML | Hybrid | Signal processing domain + ML overlap |
| Ultrahuman | Bengaluru | Wearable Health ML | Hybrid | PPG/ECG wearable experience directly maps |

## Tier 2: Strong Fit (Apply Month 12-14)

| Company | Location | Why You Fit |
|---|---|---|
| STMicroelectronics India | Bengaluru | STM32 + X-CUBE-AI (you've used these deeply) |
| Texas Instruments India | Bengaluru | Embedded AI on Sitara |
| Qualcomm India R&D | Hyderabad/Bengaluru | ONNX Runtime, QNN execution provider |
| Arm India | Bengaluru | Ethos NPU, CMSIS-NN (you've read the source) |
| NXP Semiconductors | Bengaluru | eIQ toolkit, IoT edge AI |
| Lattice Semiconductor | Bengaluru | sensAI + FPGA edge |

## Tier 3: Remote-Friendly Global (Apply Month 15-18)

| Company | Path | Why |
|---|---|---|
| Edge Impulse (US, remote-first) | SW roles | Your EdgeImpulse work is a direct signal |
| Turing.com | Remote US clients | Edge AI software contracts, USD pay |
| Toptal | Freelance vetted | Consulting on model compression |
| Crossover.com | Remote US/EU | Senior ML Engineer roles |
| Run:ai (acquired by NVIDIA) | Remote possible | MLOps for edge inference |

---

# OSS CONTRIBUTION LADDER (Tiered)

| Phase | Month | Target Repo | Contribution | Difficulty | Resume Impact |
|---|---|---|---|---|---|
| Entry | 3 | neurokit2/neurokit2 | Documentation/PPG utility | LOW | Good |
| Entry | 4 | tensorflow/tflite-micro | Doc fix from real usage | LOW | Good |
| Growth | 7 | neurokit2/neurokit2 | New biosignal algorithm | MEDIUM | Strong |
| Growth | 8 | apache/tvm | Pi 5 biosignal tutorial | MEDIUM | Strong |
| Growth | 9 | tensorflow/tflite-micro | Test case improvement | MEDIUM | Strong |
| Senior | 10 | huggingface/optimum | Edge biosignal example | MEDIUM | Strong |
| Senior | 13 | apache/tvm | MicroTVM ECG integration | HIGH | Very Strong |
| Senior | 14 | llvm/llvm-project (MLIR) | Tutorial or test case | MEDIUM-HIGH | Very Strong |
| Senior | 15 | microsoft/onnxruntime | Embedded optimization | HIGH | Very Strong |
| **Total** | **18** | **8-12 merged PRs** | | | |

---

# DECISION FRAMEWORK

| Your Confusion | The Answer |
|---|---|
| "Deep ML vs Physical AI vs Compilers — which?" | They're a pyramid. Physical AI is your PRIMARY. Compiler is a TOUCH target. Deep ML compression is the bridge. |
| "Do I need to quit my job?" | NO. 10 hrs/week × 18 months = 780 hours of deep mastery. |
| "Hardware is expensive" | ₹17,000 total kit. Less than one month of a GPU server rental. One-time cost. |
| "I'm camera shy" | Every deliverable here is code, text, screenshots, benchmark tables. Zero video. No face. |
| "Remote jobs are hard from India" | OSS PRs + technical articles + benchmarks = borderless resume. A merged TVM PR from Bengaluru = same as from Boston. |
| "The field changes too fast" | Physical constraints (RAM, Flash, Power) don't change. The skills to work within constraints compound. |

---

*Plan Version 2.0 | June 2026 | Enhanced synthesis of rough_agy.md + rough_cc.md*
*For: Abhishek Bhardwaj — Senior Data Scientist → Edge AI / Embedded ML / Semiconductor*
*Companion: rough_agy.md (reasoning) | GitHub repos: see Part Three*

> **START TOMORROW:**
> Order the hardware. Set up the conda environment. Download MIT-BIH.
> Capture your first ECG waveform. The rest is execution.
