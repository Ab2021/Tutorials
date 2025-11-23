# Day 103: Edge AI & Small Language Models (SLMs)
## Core Concepts & Theory

### The Race to the Bottom (Size)

Not everyone needs GPT-4.
**SLMs (Small Language Models)** (< 7B parameters) are enabling AI on phones, laptops, and IoT devices.
*   **Benefits:** Privacy (local), Latency (no network), Cost (zero API fees).

### 1. The Phi Series (Microsoft)

"Textbooks Are All You Need".
*   **Idea:** Train on high-quality, synthetic "textbook" data instead of noisy web data.
*   **Result:** Phi-3 (3.8B) rivals Llama-2 (70B) on reasoning tasks.
*   **Lesson:** Data Quality > Data Quantity.

### 2. Quantization & Compression

*   **Quantization:** FP16 (16-bit) -> INT4 (4-bit). 4x memory reduction.
*   **Pruning:** Removing unimportant neurons.
*   **Distillation:** Training a small Student model to mimic a large Teacher model.

### 3. Edge Hardware (NPU)

*   **Apple Neural Engine:** Optimized for CoreML.
*   **Qualcomm Hexagon:** Android AI.
*   **ExecuTorch:** PyTorch for Edge.

### 4. On-Device RAG

Running RAG on a phone.
*   **Vector DB:** SQLite-VSS or Chroma (local).
*   **Embedding:** Quantized MiniLM (20MB).
*   **Generation:** Phi-3 (2GB).

### Summary

The future is Hybrid. Heavy lifting in the cloud, privacy-sensitive/fast tasks on the Edge.
