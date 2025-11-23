# Day 63: Edge Deployment & Mobile LLMs
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What are the main constraints of running LLMs on mobile devices?

**Answer:**
- **RAM:** Most critical. Phones have 8-16GB unified memory. OS takes 2-4GB. A 7B INT4 model takes ~4-5GB. Leaving little room for other apps.
- **Thermal Throttling:** Sustained compute generates heat. Phones throttle CPU/GPU speed to cool down, degrading tokens/sec.
- **Battery:** Heavy inference drains battery rapidly.
- **Bandwidth:** Memory bandwidth on phones (e.g., 50GB/s) is much lower than A100 (2000GB/s), limiting generation speed.

#### Q2: Explain the difference between GGUF and GPTQ/AWQ.

**Answer:**
- **GGUF (Llama.cpp):** Designed for **CPU/Apple Silicon** inference. Supports "k-quants" (variable bit-width per layer). Single file format containing model + tokenizer + metadata.
- **GPTQ/AWQ:** Designed for **GPU (CUDA)** inference. Optimized for Tensor Cores.
- **Use Case:** Use GGUF for edge/mobile/Mac. Use GPTQ/AWQ for cloud GPU serving.

#### Q3: How does Speculative Decoding help on Edge devices?

**Answer:**
- Edge devices often have decent compute (NPU) but slow memory bandwidth.
- Speculative decoding allows the model to verify multiple tokens in one memory read pass.
- Since memory read is the bottleneck, this increases tokens/sec significantly without increasing total memory transfer.

#### Q4: What is the "Hybrid Cloud-Edge" pattern?

**Answer:**
- **Router:** A tiny model (or classifier) on the device analyzes the user query.
- **Local Path:** If query is simple ("Set timer", "Summarize this email"), route to local SLM (Phi-3). Zero latency, private.
- **Cloud Path:** If query is complex ("Explain quantum physics", "Generate code"), route to cloud LLM (GPT-4).
- **Benefit:** Balances privacy/cost with capability.

#### Q5: Why is INT4 quantization the standard for Edge?

**Answer:**
- **Accuracy:** LLMs retain 95%+ performance at 4-bit. Below 3-bit, performance degrades sharply.
- **Size:** 4-bit is 8x smaller than FP32. Allows running 7B models on 8GB RAM devices.
- **Hardware:** Modern NPUs/CPUs have instructions optimized for 4-bit or 8-bit integers.

---

### Production Challenges

#### Challenge 1: App Killed by OS (OOM)

**Scenario:** User switches apps while LLM is generating. OS kills the app.
**Root Cause:** LLM memory usage spiked (KV cache growth) and exceeded OS limits for background apps.
**Solution:**
- **Aggressive Unloading:** Unload model weights when app goes to background.
- **mmap:** Use memory-mapped files (GGUF) so OS can reclaim memory if needed (though this causes lag when returning).
- **Limit Context:** Strictly limit context length to prevent KV cache expansion.

#### Challenge 2: Device Overheating

**Scenario:** After 5 minutes of chatting, generation speed drops by 50%.
**Root Cause:** Thermal throttling.
**Solution:**
- **Token Limit:** Limit generation to short bursts (e.g., 200 tokens).
- **Pause:** Insert small sleeps between tokens? (Bad for UX).
- **NPU Offload:** Ensure NPU is used (more efficient than CPU/GPU).
- **Quantization:** Use lower precision (less energy).

#### Challenge 3: Slow Model Loading

**Scenario:** User opens app, has to wait 10 seconds before chatting.
**Root Cause:** Loading 4GB model from flash storage to RAM.
**Solution:**
- **mmap:** Map the file instantly (start inference while pages load).
- **Keep Alive:** Keep model loaded in a background service (if OS permits).
- **Splash Screen:** Good UX to hide loading time.

#### Challenge 4: Cross-Platform Compatibility

**Scenario:** App works on Pixel 8 but crashes on older Samsung.
**Root Cause:** Lack of NPU support or insufficient RAM on older devices.
**Solution:**
- **Device Profiling:** Check RAM/Chipset on startup.
- **Dynamic Model Selection:**
  - High-end: Load Llama-3-8B.
  - Mid-range: Load Phi-3-Mini.
  - Low-end: Disable local LLM, use cloud API.

#### Challenge 5: Updating Large Models

**Scenario:** You improved the model. Users need to download a 4GB update.
**Root Cause:** App store limits and user data plans.
**Solution:**
- **Delta Updates:** Only download changed weights (LoRA adapters).
- **WiFi Only:** Schedule downloads on WiFi.
- **Background Download:** Download silently in background.

### System Design Scenario: Offline Translation App

**Requirement:** Translate speech-to-speech offline on a phone.
**Design:**
1.  **ASR (Speech-to-Text):** Whisper-Tiny (quantized) running on CoreML/TFLite.
2.  **LLM (Translation):** Gemma-2B-IT (GGUF) running on Llama.cpp. Prompt: "Translate to Spanish: {text}".
3.  **TTS (Text-to-Speech):** Lightweight on-device TTS model.
4.  **Pipeline:** Audio -> Whisper -> LLM -> TTS -> Audio.
5.  **Optimization:** Pipelining (start translating while user is still speaking?).

### Summary Checklist for Production
- [ ] **Model:** Use **Phi-3 / Llama-3-8B**.
- [ ] **Format:** Use **GGUF** with **Q4_K_M** quantization.
- [ ] **Engine:** Use **MLC LLM** or **Llama.cpp**.
- [ ] **Fallback:** Implement **Cloud Fallback** for complex queries.
- [ ] **Profiling:** Test on **Low-end devices**.
- [ ] **Battery:** Monitor **energy usage**.
