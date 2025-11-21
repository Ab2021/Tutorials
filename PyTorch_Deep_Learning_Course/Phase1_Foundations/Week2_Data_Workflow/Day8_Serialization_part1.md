# Day 8: Serialization - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Safetensors, ONNX, and Versioning

## 1. The Pickle Security Problem

`torch.load` uses `pickle`.
**Vulnerability**: Pickle allows arbitrary code execution during unpickling.
If you download a `.pth` file from the internet, it could contain a malicious payload that wipes your disk when loaded.

## 2. Safetensors (Hugging Face)

A new format designed to be:
1.  **Safe**: No code execution. Pure data.
2.  **Fast**: Zero-copy memory mapping (mmap).
3.  **Lazy**: Load only specific tensors without reading the whole file.

```python
from safetensors.torch import save_file, load_file

save_file(model.state_dict(), "model.safetensors")
state_dict = load_file("model.safetensors")
```
**Standard in LLMs**: Most Llama/Mistral models now ship as safetensors.

## 3. ONNX (Open Neural Network Exchange)

Standard format for interoperability.
PyTorch -> ONNX -> TensorRT / CoreML / TFLite.

```python
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}} # Variable batch size
)
```

## 4. Model Versioning

How to handle architecture changes?
*   **Version Attribute**: Store `_version` in the module.
*   **`_load_from_state_dict`**: Override this method to implement backward compatibility logic (e.g., renaming keys, reshaping weights).

```python
def _load_from_state_dict(self, state_dict, prefix, ...):
    if 'old_layer' in state_dict:
        state_dict['new_layer'] = state_dict.pop('old_layer')
    super()._load_from_state_dict(...)
```

## 5. Distributed Checkpointing

Saving a 100B parameter model (400GB) from a single process causes OOM.
**Sharded Checkpointing**:
Each GPU saves its own part of the model to a separate file.
PyTorch `distributed.checkpoint` handles this.
