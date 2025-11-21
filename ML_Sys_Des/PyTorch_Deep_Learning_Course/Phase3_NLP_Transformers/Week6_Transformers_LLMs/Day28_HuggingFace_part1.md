# Day 28: Hugging Face - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: PEFT, BitsAndBytes, and Optimization

## 1. PEFT (Parameter-Efficient Fine-Tuning)

Fine-tuning LLaMA-7B requires 100GB+ VRAM.
**PEFT** library integrates LoRA, Prefix Tuning, P-Tuning.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# "trainable params: 4M || all params: 7B || trainable%: 0.06"
```

## 2. BitsAndBytes (Quantization)

8-bit and 4-bit training.
*   **LLM.int8()**: Outlier features in FP16, rest in INT8.
*   **QLoRA**: 4-bit NormalFloat (NF4) quantization + LoRA.
*   Allows fine-tuning LLaMA-65B on a single 48GB GPU.

## 3. Tokenizers Library (Rust)

Why Rust?
*   Python loops are too slow for tokenizing 100GB text.
*   HF Tokenizers uses Rust for parallelism.
*   `fast=True` in `AutoTokenizer` enables this.

## 4. Model Hub & Cards

*   **Model Card**: Documentation (Usage, Limitations, Bias).
*   **Versioning**: Models are git repositories. Can pin `revision="v1.0"`.
*   **Inference API**: Free tier to test models via HTTP.

## 5. Optimum

Hardware acceleration library.
*   **ONNX Runtime**: `ORTModelForSequenceClassification`.
*   **BetterTransformer**: PyTorch native fastpath (Flash Attention integration).
*   **Intel/Habana**: Optimizations for CPUs and Gaudi chips.
