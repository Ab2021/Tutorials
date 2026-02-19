# 📚 Gemma Fine-Tuning — Theory & Documentation

This `docs/` folder contains detailed theoretical explanations for every concept, method, and decision in this fine-tuning project. Written for beginners — no prior ML experience assumed.

## 📖 Reading Order (Recommended)

| # | Document | What You'll Learn |
|---|----------|-------------------|
| 1 | [01_what_is_finetuning.md](./01_what_is_finetuning.md) | Foundation: what fine-tuning is, why we do it, types of fine-tuning |
| 2 | [02_understanding_gemma.md](./02_understanding_gemma.md) | Gemma architecture, transformers, attention, how LLMs work |
| 3 | [03_lora_and_qlora.md](./03_lora_and_qlora.md) | LoRA, QLoRA, quantization — why we don't train all parameters |
| 4 | [04_quantization_deep_dive.md](./04_quantization_deep_dive.md) | 4-bit, 8-bit, NF4, double quantization — how compression works |
| 5 | [05_torch_compile_guide.md](./05_torch_compile_guide.md) | torch.compile internals, backends, Triton, graph breaks, debugging |
| 6 | [06_training_process.md](./06_training_process.md) | Training loop, loss functions, optimizers, schedulers, gradient flow |
| 7 | [07_data_pipeline.md](./07_data_pipeline.md) | Tokenization, prompt engineering, dataset preparation |
| 8 | [08_hyperparameter_guide.md](./08_hyperparameter_guide.md) | What each hyperparameter does, when to change it, decision trees |
| 9 | [09_evaluation_metrics.md](./09_evaluation_metrics.md) | ROUGE, BLEU, perplexity — measuring model quality |
| 10 | [10_troubleshooting.md](./10_troubleshooting.md) | Every error you might encounter and how to fix it |

---

> **Tip:** If you're in a hurry, read docs 1, 3, and 8 — they cover the essentials.
