# Day 28: Hugging Face - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Libraries, Distributed Training, and Optimization

### 1. What is the benefit of `AutoModel`?
**Answer:**
*   Abstraction. You don't need to import `BertModel` or `RoBertaModel` specifically.
*   `AutoModel.from_pretrained("path")` detects the architecture from `config.json` and loads the correct class.

### 2. How does `datasets` library handle large files?
**Answer:**
*   **Memory Mapping (Apache Arrow)**.
*   It doesn't load the full file into RAM. It maps the file on disk to virtual memory.
*   Allows random access to TB-sized datasets with minimal RAM usage.

### 3. What is "Accelerate"?
**Answer:**
*   A library to abstract away the boilerplate of distributed training.
*   Write standard PyTorch code, and Accelerate handles `DistributedDataParallel`, `FP16`, `TPU` placement.

### 4. What is the `Trainer` API?
**Answer:**
*   A high-level loop for training Transformers.
*   Handles: Gradient Accumulation, Logging (WandB), Checkpointing, Evaluation, Mixed Precision.

### 5. Explain "Gradient Accumulation".
**Answer:**
*   If batch size 32 doesn't fit in VRAM, use batch size 4 and accumulate gradients for 8 steps before calling `optimizer.step()`.
*   Simulates a larger batch size.

### 6. What is "Mixed Precision" (FP16)?
**Answer:**
*   Storing weights/activations in FP16 (Half Precision) but keeping a master copy of weights in FP32.
*   Reduces VRAM usage by 50% and speeds up training on Tensor Cores.
*   Requires Loss Scaling to prevent underflow.

### 7. What is "Data Collator"?
**Answer:**
*   A function that takes a list of samples and batches them.
*   Handles dynamic padding (pad to the longest sequence in the *batch*, not the dataset).
*   `DataCollatorForLanguageModeling` handles random masking for MLM.

### 8. How do you push a model to the Hub?
**Answer:**
*   `model.push_to_hub("my-model")`.
*   `tokenizer.push_to_hub("my-model")`.
*   Requires `huggingface-cli login`.

### 9. What is "Safetensors"?
**Answer:**
*   A new serialization format replacing `pickle` (`pytorch_model.bin`).
*   **Safe**: No code execution (Pickle is insecure).
*   **Fast**: Zero-copy loading.

### 10. What is "Pipeline"?
**Answer:**
*   High-level API for inference.
*   `pipe = pipeline("text-classification")`.
*   Handles Preprocessing (Tokenization), Model Forward, and Postprocessing (Logits $\to$ Labels).

### 11. What is "Sharded Checkpointing"?
**Answer:**
*   Splitting a large model (e.g., 50GB) into smaller chunks (`pytorch_model-00001-of-00005.bin`).
*   Prevents OOM when loading on small RAM machines.

### 12. How does `map` work in Datasets?
**Answer:**
*   Applies a function to every element.
*   `batched=True`: Passes a batch of samples (faster due to vectorization).
*   `num_proc=4`: Multiprocessing.

### 13. What is "Streaming" in Datasets?
**Answer:**
*   `load_dataset(..., streaming=True)`.
*   Returns an IterableDataset.
*   Data is downloaded/read on-the-fly. Crucial for massive web-scale datasets.

### 14. What is "Optimum"?
**Answer:**
*   Extension of Transformers for hardware optimization.
*   Supports ONNX Runtime, OpenVINO, Habana Gaudi, AWS Inferentia.

### 15. What is "BitsAndBytes"?
**Answer:**
*   Library for 8-bit and 4-bit quantization primitives.
*   Enables loading large LLMs on consumer GPUs.

### 16. How do you handle "Catastrophic Forgetting" in Fine-Tuning?
**Answer:**
*   Low Learning Rate.
*   PEFT (LoRA) - Freeze backbone, train adapters.
*   Replay Buffer (mix old data with new).

### 17. What is the difference between `save_pretrained` and `torch.save`?
**Answer:**
*   `torch.save`: Saves the raw state dict (pickle).
*   `save_pretrained`: Saves state dict + `config.json` + `tokenizer.json`. Makes it self-contained and loadable by `AutoModel`.

### 18. What is "DeepSpeed"?
**Answer:**
*   Microsoft's optimization library integrated into HF Trainer.
*   **ZeRO (Zero Redundancy Optimizer)**: Shards optimizer states, gradients, and parameters across GPUs to save memory.

### 19. What is "Timm"?
**Answer:**
*   PyTorch Image Models. The "Hugging Face" of Computer Vision (before HF expanded to CV).
*   Now integrated into HF Hub.

### 20. Why use `AutoTokenizer` instead of `BertTokenizer`?
**Answer:**
*   Portability.
*   If you switch the model checkpoint from BERT to RoBERTa, `AutoTokenizer` adapts automatically. `BertTokenizer` would fail.
