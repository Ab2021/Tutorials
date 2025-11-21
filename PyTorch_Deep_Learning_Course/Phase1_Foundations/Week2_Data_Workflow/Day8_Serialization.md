# Day 8: Serialization & Checkpointing - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Saving, Loading, and Model Portability

## 1. Theoretical Foundation: Serialization

Serialization is converting an object state into a byte stream for storage.
In PyTorch, we use `torch.save`, which wraps Python's `pickle`.

### The `state_dict`
A Python dictionary mapping:
*   **Keys**: Parameter names (strings, e.g., `layer1.weight`).
*   **Values**: Tensors.

Why save `state_dict` instead of the whole model object?
*   **Portability**: The model class code might change. Saving the object (pickle) binds you to the specific class definition.
*   **Flexibility**: You can load weights into a different architecture (e.g., Transfer Learning).

## 2. Implementation: Saving and Loading

### Basic
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel() # Must instantiate architecture first
model.load_state_dict(torch.load('model.pth'))
model.eval() # Don't forget this!
```

### Checkpointing (Training State)
To resume training, you need more than just weights. You need the Optimizer state (momentum buffers) and Scheduler state.

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## 3. Transfer Learning (Partial Loading)

Loading weights from a pre-trained model (ResNet) into a custom model.

```python
pretrained_dict = torch.load('resnet.pth')
model_dict = model.state_dict()

# 1. Filter out unnecessary keys (e.g., final FC layer)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

# 2. Overwrite entries in existing state dict
model_dict.update(pretrained_dict)

# 3. Load
model.load_state_dict(model_dict)
```

## 4. Device Handling

`torch.load` loads tensors to the device they were saved from.
If saved on GPU and loading on CPU, use `map_location`.

```python
# Load GPU model on CPU
torch.load('model.pth', map_location=torch.device('cpu'))
```

## 5. TorchScript (Serialization for C++)

For production deployment (C++, Mobile), `pickle` is not enough. We need a language-agnostic format.

```python
# Trace
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("model.pt")

# Load in C++
# torch::jit::load("model.pt");
```
