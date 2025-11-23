# Day 103: Edge AI & Small Language Models (SLMs)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between GGUF and GPTQ?

**Answer:**
*   **GGUF:** CPU-centric. Optimized for Apple Silicon/Intel. Supports mixed quantization.
*   **GPTQ/AWQ:** GPU-centric. Optimized for CUDA kernels.

#### Q2: Why does "Data Quality" matter more for SLMs?

**Answer:**
Large models have enough capacity to memorize noise and still learn signal.
Small models have limited capacity. If you fill it with noise, there's no room for signal.
Phi-3 proved that "Textbook quality" data yields 10x efficiency.

#### Q3: How do you update a local model?

**Answer:**
*   **OTA (Over-The-Air):** Download a binary patch.
*   **LoRA:** Don't update the base model. Just download a new 10MB LoRA adapter.

#### Q4: What is "Apple Intelligence" architecture?

**Answer:**
Hybrid.
*   **On-Device:** 3B parameter model for email summary, smart reply.
*   **Private Cloud Compute:** For complex queries, send to Apple Silicon servers (stateless, no logging).

### Production Challenges

#### Challenge 1: Thermal Throttling

**Scenario:** Phone gets hot after 1 minute of chat. OS kills the app.
**Root Cause:** Compute intensity.
**Solution:**
*   **Quantization:** Use INT4.
*   **Duty Cycle:** Limit generation speed.

#### Challenge 2: Device Fragmentation

**Scenario:** Works on Pixel 8, crashes on Samsung S21.
**Root Cause:** Different NPUs/RAM.
**Solution:**
*   **Tiers:**
    *   High-End: Llama-3-8B.
    *   Mid-Range: Phi-3-Mini.
    *   Low-End: Cloud API only.

#### Challenge 3: Prompt Compatibility

**Scenario:** Prompt optimized for GPT-4 fails on Phi-3.
**Root Cause:** SLMs are less robust to instruction nuances.
**Solution:**
*   **DSPy:** Automated prompt optimization for the specific target model.

### System Design Scenario: Offline Translation App

**Requirement:** Translate speech in real-time without internet.
**Design:**
1.  **VAD:** Silero (Local).
2.  **ASR:** Whisper-Tiny (Quantized).
3.  **MT:** NLLB-Distilled (Neural Machine Translation).
4.  **TTS:** Piper (Fast local TTS).
5.  **Optimization:** Pipeline parallelism (Start translating sentence 1 while listening to sentence 2).

### Summary Checklist for Production
*   [ ] **Battery Test:** Measure mAh impact.
*   [ ] **Download Size:** Keep the model under 2GB (WiFi download limit psychology).
