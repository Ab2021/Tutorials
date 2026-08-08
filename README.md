# AIMET Deep Dive & 2026 Frontier Model Efficiency Knowledge Base

> **AI Model Efficiency Toolkit (AIMET) by Qualcomm Innovation Center & 2026 Frontier AI Research**
> A complete, research-grade reference covering quantization, compression, edge deployment, comparable libraries, domain-specific case studies, and 2026 breakthroughs (Google TurboQuant, Kimi K3 Attention, DeepSeek RL/MLA, Claude 4/5, Qwen3, Z.ai).

---

## 📁 Master Document Index

| # | File | Description | Key Innovations / Focus |
|---|------|-------------|-------------------------|
| 1 | [01_AIMET_Overview_and_Architecture.md](./01_AIMET_Overview_and_Architecture.md) | AIMET overview, architecture, history, installation, ecosystem | PyTorch, TF, ONNX backends |
| 2 | [02_Quantization_Fundamentals.md](./02_Quantization_Fundamentals.md) | Deep dive into quantization theory, math, precision formats | FP32→INT8/INT4, symmetric/asymmetric |
| 3 | [03_Post_Training_Quantization_PTQ.md](./03_Post_Training_Quantization_PTQ.md) | PTQ stack: CLE, Bias Correction, AdaRound, AutoQuant | Zero retraining, 8-bit optimization |
| 4 | [04_Quantization_Aware_Training_QAT.md](./04_Quantization_Aware_Training_QAT.md) | QuantSim, STE derivation, QAT with Range Learning | Fine-tuning, INT4 stabilization |
| 5 | [05_Model_Compression_Techniques.md](./05_Model_Compression_Techniques.md) | Spatial SVD, Weight SVD, Channel Pruning, Greedy Selection | FLOPs/MACs reduction, N:M sparsity |
| 6 | [06_AIMET_APIs_and_Configuration.md](./06_AIMET_APIs_and_Configuration.md) | Complete API reference, configuration JSON schemas | Programmatic Python & JSON config |
| 7 | [07_Edge_Deployment_and_Qualcomm_Hardware.md](./07_Edge_Deployment_and_Qualcomm_Hardware.md) | Snapdragon Hexagon NPU, Cloud AI 100, QAIRT, QNN | DLC compilation, mobile deployment |
| 8 | [08_Comparable_Libraries_and_Ecosystem.md](./08_Comparable_Libraries_and_Ecosystem.md) | 20 libraries compared: TensorRT, INC, TVM, AWQ, GPTQ | Cross-ecosystem feature matrix |
| 9 | [09_LLM_and_Transformer_Quantization.md](./09_LLM_and_Transformer_Quantization.md) | BERT, LLaMA, GPTQ, AWQ, SmoothQuant, KV cache | Attention & activation outlier handling |
| 10 | [10_Case_Studies_Computer_Vision.md](./10_Case_Studies_Computer_Vision.md) | Object detection, segmentation, face recognition, super-res | 8+ CV edge case studies with code |
| 11 | [11_Case_Studies_NLP_and_Audio.md](./11_Case_Studies_NLP_and_Audio.md) | BERT Q&A, ASR (Whisper), TTS, keyword spotting, Phi-2 | 9 NLP/audio edge case studies |
| 12 | [12_Case_Studies_Autonomous_Driving.md](./12_Case_Studies_Autonomous_Driving.md) | BEVFusion, lane detection, LiDAR, ADAS, ISO 26262 | Automotive NPU & safety analysis |
| 13 | [13_Case_Studies_Healthcare_and_Industry.md](./13_Case_Studies_Healthcare_and_Industry.md) | Medical CT scan, ECG wearables, surgical robotics, IoT | Regulatory & industrial edge AI |
| 14 | [14_Advanced_Topics_and_Future_Directions.md](./14_Advanced_Topics_and_Future_Directions.md) | INT4, MX formats, NAS, knowledge distillation, BitNet | 2026-2030 research roadmap |
| 15 | [15_Benchmarks_and_Performance_Analysis.md](./15_Benchmarks_and_Performance_Analysis.md) | MLPerf Inference/Mobile, accuracy-latency tables | Roofline model & ROI analysis |
| 16 | [16_2026_Google_TurboQuant_and_KV_Cache.md](./16_2026_Google_TurboQuant_and_KV_Cache.md) | **Google TurboQuant, PolarQuant, QJL, Fast-TurboQuant** | 3-bit online vector KV compression |
| 17 | [17_2026_Kimi_K3_Attention_and_MoE_Architectures.md](./17_2026_Kimi_K3_Attention_and_MoE_Architectures.md) | **Kimi K3, Kimi Delta Attention (KDA), AttnRes** | 2.8T MoE, 1M context, linear attention |
| 18 | [18_2026_DeepSeek_RL_Optimizations_and_MLA.md](./18_2026_DeepSeek_RL_Optimizations_and_MLA.md) | **DeepSeek-V3/R1/V4, GRPO RL, Multi-Head Latent Attn** | Critic-free RL, low-rank latent KV |
| 19 | [19_2026_Frontier_Models_Claude_Qwen_Zai.md](./19_2026_Frontier_Models_Claude_Qwen_Zai.md) | **Claude 4/5, Qwen3 Dual MoE, Z.ai IndexShare** | Test-time adaptive thinking, agentic AI |

---

## 🔑 Key Concepts & 2026 Frontier Breakthroughs

```
AIMET & Frontier Model Efficiency Stack (2026)
├── Model Optimization (AIMET)
│   ├── Model Preparation & BatchNorm Folding
│   ├── Post-Training Quantization (CLE, Bias Correction, AdaRound, AutoQuant)
│   ├── Quantization-Aware Training (Standard QAT, Range Learning)
│   └── SVD & Channel Pruning Compression
├── Hardware Compilation & Edge Deployment
│   ├── Qualcomm QAIRT (qairt-converter ONNX → DLC)
│   ├── Qualcomm QNN & AI Hub profiling
│   └── Snapdragon Hexagon NPU (HVX + HMX execution)
└── 2026 Frontier Innovations
    ├── Google TurboQuant: Data-oblivious 3-bit KV vector quantization (PolarQuant + QJL)
    ├── Kimi K3: Hybrid Linear Attention (KDA) + Attention Residuals (AttnRes) for 1M context
    ├── DeepSeek RL & MLA: GRPO critic-free RLHF + Multi-Head Latent Attention + Native FP8
    └── Test-Time Compute Scaling: Adaptive Thinking (Claude 4/5), Dual MoE (Qwen3), IndexShare (Z.ai)
```

---

## 🏷️ Tags
`quantization` `model-compression` `edge-ai` `post-training-quantization` `qat` `qualcomm` `snapdragon` `aimet` `google-turboquant` `kimi-k3` `kda` `attnres` `deepseek-grpo` `deepseek-mla` `claude-extended-thinking` `qwen3` `z.ai` `glm-5` `kv-cache-compression`

---

## 📅 Last Updated
August 2026 — Covering AIMET v1.31+ and 2026 Frontier Model Breakthroughs.
