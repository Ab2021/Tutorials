# Day 63: Edge Deployment & Mobile LLMs
## Core Concepts & Theory

### The Edge Frontier

**Why Edge?**
- **Privacy:** Data stays on device (Health, Finance).
- **Latency:** No network round-trip.
- **Cost:** Zero cloud inference cost.
- **Offline:** Works without internet.

**Constraints:**
- **Memory:** Phones have 8GB-16GB RAM (shared with OS).
- **Compute:** Mobile CPUs/NPUs are weaker than A100s.
- **Battery:** Heavy compute drains battery fast.
- **Thermal:** Phones throttle if they get too hot.

### 1. Model Compression for Edge

**Quantization (Aggressive):**
- **INT4:** Standard for mobile.
- **INT2 / 3-bit:** Emerging for extreme compression.
- **Activation Quantization:** Crucial for memory bandwidth.

**Pruning:**
- **Structured Pruning:** Remove heads/layers to reduce size.
- **SparseGPT:** 50% sparsity with minimal loss.

**Distillation:**
- Train a small student (1B-3B) from a large teacher (70B).
- **Phi-2 / Phi-3:** Examples of high-quality small models.

### 2. Mobile Architectures

**Small Language Models (SLMs):**
- **Size:** 1B - 7B parameters.
- **Examples:** Llama-3-8B, Phi-3-Mini (3.8B), Gemma-2B.
- **Design:** Optimized for efficiency (MQA, GQA).

**Hybrid Architectures:**
- **Cloud-Edge Hybrid:**
  - Run small model on device for simple queries.
  - Route complex queries to cloud.
  - **Router:** Runs on device.

### 3. Inference Engines for Edge

**MLC LLM (Machine Learning Compilation):**
- **Tech:** TVM-based compiler.
- **Platform:** iOS, Android, WebGPU.
- **Performance:** Near-native speed.

**Llama.cpp (GGML/GGUF):**
- **Tech:** C++ implementation, CPU optimized (AVX, ARM NEON).
- **Platform:** Cross-platform (Mac, Linux, Windows, Android).
- **Quantization:** GGUF format (k-quants).

**TensorFlow Lite / PyTorch Mobile:**
- **Tech:** Framework-specific runtimes.
- **Status:** Lagging behind Llama.cpp/MLC for LLMs.

**ExecuTorch (PyTorch Edge):**
- New PyTorch stack for edge.

### 4. WebGPU & Browser Inference

**WebLLM:**
- Run LLMs inside Chrome/Edge using WebGPU.
- **Benefit:** Zero install, privacy-preserving web apps.
- **Constraint:** Browser memory limits (usually ~4GB per tab).

### 5. NPU (Neural Processing Unit)

**Hardware Acceleration:**
- **Apple Neural Engine (ANE):** Optimized for CoreML.
- **Qualcomm Hexagon:** Android NPU.
- **Intel NPU:** AI PC.

**Optimization:**
- Offload MatMul to NPU.
- Keep CPU free for other tasks.

### 6. Battery & Thermal Management

**Power Consumption:**
- **Metric:** Joules per token.
- **Strategy:**
  - Batch processing (race to sleep).
  - Lower precision (less memory movement = less power).
  - Frame capping (limit tokens/sec).

### 7. On-Device RAG

**Concept:** RAG running entirely on phone.
- **Vector DB:** SQLite-VSS, Chroma (embedded), or LanceDB.
- **Embedding Model:** Quantized MiniLM (int8).
- **Use Case:** Search personal notes, emails, messages.

### 8. Privacy & Security

**Local Processing:**
- No data leaves device.
- **Encryption:** Model weights encrypted at rest.
- **Sandboxing:** App sandbox prevents data leakage.

### 9. Future Trends

**Speculative Decoding on Device:**
- Use a tiny draft model (100M) to speed up the 3B model.

**Personalized LoRA:**
- Train/Fine-tune LoRA adapter on device using user data.
- **Privacy:** Personalization without uploading data.

### Summary

**Edge Strategy:**
1.  **Model:** Choose **SLM (1B-3B)** like Phi-3 or Llama-3-8B.
2.  **Format:** Convert to **GGUF** (Llama.cpp) or **MLC**.
3.  **Quantization:** Use **INT4** (mandatory for memory).
4.  **Engine:** Use **MLC LLM** or **Llama.cpp**.
5.  **Hybrid:** Route hard tasks to cloud.

### Next Steps
In the Deep Dive, we will explore converting a model to GGUF format, running it with Llama.cpp, and implementing a simple WebGPU chat interface.
