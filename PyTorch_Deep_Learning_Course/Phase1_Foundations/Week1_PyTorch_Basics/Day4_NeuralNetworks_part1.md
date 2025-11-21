# Day 4: Neural Networks - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Module Hooks, State Dicts, and JIT Scripting

## 1. The Magic of `__setattr__`

How does `nn.Module` know about the layers you assign to `self.fc1`?
It overrides `__setattr__`.
When you do `self.fc1 = nn.Linear(...)`:
1.  It checks if the value is an instance of `nn.Parameter` or `nn.Module`.
2.  If so, it registers it in `self._parameters` or `self._modules` dictionary.
3.  `model.parameters()` simply iterates over these dictionaries recursively.

**Pitfall**: If you store layers in a standard Python list, they won't be registered!
```python
# WRONG
self.layers = [nn.Linear(10, 10) for _ in range(5)] 
# PyTorch doesn't see these!

# RIGHT
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
```

## 2. Hooks: Inspecting the Black Box

Hooks allow you to run code before or after forward/backward passes of a specific layer.
Useful for:
*   **Debugging**: Checking for NaNs.
*   **Feature Extraction**: Getting output of intermediate layer (e.g., for Neural Style Transfer).
*   **Modifying Gradients**: Gradient clipping or masking.

```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation('fc1'))
model(x)
print(activations['fc1'].shape)
```

## 3. Serialization: `state_dict`

A model's state is just a Python dictionary mapping parameter names to tensors.
`model.state_dict()` -> `{'fc1.weight': tensor(...), 'fc1.bias': ...}`.

**Saving**:
`torch.save(model.state_dict(), 'model.pth')`

**Loading**:
`model.load_state_dict(torch.load('model.pth'))`
*   `strict=True` (default): Keys must match exactly.
*   `strict=False`: Ignores missing/unexpected keys (useful for Transfer Learning).

## 4. TorchScript (JIT)

Python is slow and hard to deploy (requires Python interpreter).
**TorchScript** converts your PyTorch model into a serializable, optimizable intermediate representation (IR) that can run in C++ (LibTorch).

### Tracing
Runs the model with dummy input and records operations.
```python
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model.pt")
```
*   *Limitation*: Cannot capture control flow (if/else) that depends on data.

### Scripting
Parses the Python source code (AST) to compile it.
```python
scripted_model = torch.jit.script(model)
```
*   *Pros*: Supports control flow.

## 5. `nn.Functional` vs `nn.Module`

*   `nn.Conv2d`: Holds weights. Use in `__init__`.
*   `F.conv2d`: Pure operation. Needs weights passed explicitly.

**Rule of Thumb**:
*   If it has weights (Conv, Linear, BatchNorm) -> Use `nn.Module`.
*   If it's stateless (ReLU, MaxPool, Softmax) -> Use `F.functional` (or `nn.Module` if you prefer consistency).
*   **Exception**: `Dropout` and `BatchNorm` behave differently in train/eval. Always use `nn.Module` versions (`nn.Dropout`) so they handle the `model.train()` flag correctly.
