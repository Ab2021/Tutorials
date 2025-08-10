# Day 3.3: GPU Acceleration & Memory Management - A Practical Guide

## Introduction: Unleashing the Power

Modern deep learning is computationally intensive. Training a model on millions of data points can take days or even weeks on a standard CPU. This is where the **Graphics Processing Unit (GPU)** becomes essential. GPUs are specialized hardware designed for parallel computation, capable of performing thousands of operations simultaneously. This makes them exceptionally well-suited for the matrix multiplications and tensor operations that are at the heart of deep learning.

This guide will provide a practical walkthrough of how to use GPUs in PyTorch. We will cover how to move tensors and models to the GPU, the rules you must follow for GPU operations, and how to manage GPU memory effectively.

**Today's Learning Objectives:**

1.  **Understand the `device` concept:** Learn how PyTorch abstracts hardware (CPU vs. GPU).
2.  **Master the `.to(device)` method:** The universal tool for moving any PyTorch object (tensors, models) to the correct device.
3.  **Learn the Golden Rule of GPU Computing:** Understand that all data and models in an operation must be on the same device.
4.  **Implement a GPU-aware training loop:** Write a complete training pipeline that correctly handles device placement.
5.  **Manage GPU Memory:** Learn why GPU memory is a precious resource and how to use tools like `torch.no_grad()` to manage it effectively.
6.  **Monitor GPU Usage:** Use the `nvidia-smi` command-line tool to see how much memory your model is using.

---

## Part 1: The `device` - Your Hardware Pointer

PyTorch makes it incredibly simple to switch between CPU and GPU. It does this through a `device` object. You create a `device` string that points to the hardware you want to use, and then you use this object to tell PyTorch where to put your data and your model.

### 1.1. Setting up the `device`

The standard best practice is to check if a GPU is available and set the device accordingly. This makes your code portable—it will automatically use a GPU if one is found, and fall back to the CPU otherwise.

```python
import torch

# --- The Standard Device Setup ---
# This is the boilerplate code you should have at the top of almost every PyTorch script.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Part 1: Setting up the device ---")
print(f"The selected device is: {device}")

# You can also specify a particular GPU if you have more than one
# device = torch.device("cuda:0") # First GPU
# device = torch.device("cuda:1") # Second GPU
```

### 1.2. Moving Tensors to a `device`

The `.to()` method is the universal way to move a PyTorch object. When called on a tensor, it returns a *new copy* of that tensor on the specified device.

```python
# Create a tensor on the CPU (the default)
cpu_tensor = torch.randn(2, 3)
print(f"Original tensor device: {cpu_tensor.device}")

# Move the tensor to the selected device
gpu_tensor = cpu_tensor.to(device)
print(f"New tensor device: {gpu_tensor.device}")

# Note: If the device is already the CPU, .to("cpu") returns the original tensor
# without a copy, but it's good practice to use .to(device) everywhere for consistency.
```

---

## Part 2: The Golden Rule of GPU Computing

This is the most important rule and the source of the most common error for beginners using GPUs.

**Rule:** All tensors and models involved in a single operation must reside on the **same device**.

You cannot perform an operation between a tensor on the CPU and a tensor on the GPU.

### 2.1. Demonstrating the Device Mismatch Error

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create one tensor on the CPU and one on the GPU
tensor_cpu = torch.randn(3, 4)
tensor_gpu = torch.randn(3, 4).to(device)

print("\n--- Part 2: The Golden Rule ---")
print(f"Device of tensor_cpu: {tensor_cpu.device}")
print(f"Device of tensor_gpu: {tensor_gpu.device}")

if device.type == 'cuda':
    try:
        # This will raise a RuntimeError
        result = tensor_cpu + tensor_gpu
    except RuntimeError as e:
        print(f"\nError encountered: {e}")
        print("This error occurred because the two tensors are on different devices.")

    # The Fix: Move the CPU tensor to the GPU before the operation
    tensor_cpu_on_gpu = tensor_cpu.to(device)
    result = tensor_cpu_on_gpu + tensor_gpu
    print(f"\nSuccessfully performed the operation after moving both tensors to device: {result.device}")
else:
    print("\nSkipping device mismatch error demo as no GPU is available.")
```

---

## Part 3: A Complete GPU-Aware Training Loop

Now let's apply this knowledge to a full training pipeline. The key is to move your **model** and your **data** to the `device` at the appropriate times.

**The Workflow:**
1.  Set up your `device` at the beginning of your script.
2.  Move your model to the `device` **once**, right after you create it.
3.  Inside your training loop, for each batch of data you get from your `DataLoader`, move the **data and labels** to the `device`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Setup ---
print("\n--- Part 3: GPU-Aware Training Loop ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}\n")

# --- 2. Create Dummy Data and DataLoader ---
# Let's create a simple regression dataset
num_samples = 1024
num_features = 16
X = torch.randn(num_samples, num_features)
y = torch.randn(num_samples, 1)

dataset = TensorDataset(X, y)
# num_workers can speed up data loading, pin_memory helps transfer data to GPU faster
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

# --- 3. Define a Model and Move it to the Device ---
model = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# This is the crucial step for the model.
# It moves all of the model's parameters (weights and biases) to the GPU.
model.to(device)

# --- 4. Define Loss and Optimizer ---
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. The Training Loop ---
num_epochs = 3
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    
    # The loop iterates through the DataLoader
    for batch_X, batch_y in data_loader:
        # This is the crucial step for the data.
        # Move the batch of data and labels to the same device as the model.
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Now, the rest of the training loop is standard.
        # All computations will happen on the GPU.
        
        # 1. Forward pass
        predictions = model(batch_X)
        
        # 2. Calculate loss
        loss = loss_function(predictions, batch_y)
        
        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\nTraining complete. All computations were performed on the GPU (if available).")
```

---

## Part 4: GPU Memory Management

GPU memory (VRAM) is a finite and often limited resource. A common error is `CUDA out of memory`. This happens when you try to load more tensors or a larger model onto the GPU than its memory can hold.

### 4.1. The Power of `torch.no_grad()`

During training, PyTorch needs to store not only the model's predictions but also all the intermediate values (the computation graph) required to calculate gradients during the backward pass. These intermediate values consume a lot of VRAM.

When you are doing **inference or evaluation**, you don't need gradients. The `torch.no_grad()` context manager tells PyTorch not to build the computation graph. This has two benefits:

1.  **It significantly reduces memory consumption.**
2.  **It speeds up the computation.**

**Best Practice:** Always wrap your validation and testing loops in `with torch.no_grad():`.

```python
print("\n--- Part 4: GPU Memory Management ---")

# --- Evaluation Loop with no_grad ---
model.eval() # Set the model to evaluation mode

total_error = 0
num_samples = 0

# This context manager is key!
with torch.no_grad():
    for batch_X, batch_y in data_loader:
        # Move data to the device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Get predictions
        predictions = model(batch_X)
        
        # No .backward() call is needed, and no gradients are computed.
        total_error += torch.abs(predictions - batch_y).sum().item()
        num_samples += batch_X.size(0)

mae = total_error / num_samples
print(f"Evaluation complete. Mean Absolute Error on the dataset: {mae:.4f}")
print("This was done using torch.no_grad() to save memory and increase speed.")
```

### 4.2. Other Memory Tips

*   **Batch Size:** The most common cause of `out of memory` errors is a batch size that is too large. The larger your batch size, the more data and intermediate activations need to be stored in VRAM. If you run out of memory, the first thing to try is **reducing your batch size**.
*   **Delete Unused Tensors:** Python's garbage collector and PyTorch will generally handle memory, but if you have large tensors in your main script that you no longer need, you can delete them with `del my_large_tensor`.
*   **Empty the Cache:** If you have deleted tensors but the memory is not freed (due to caching by PyTorch), you can force it with `torch.cuda.empty_cache()`. This is rarely needed in a well-structured training script but can be useful in interactive environments like Jupyter.

---

## Part 5: Monitoring with `nvidia-smi`

`nvidia-smi` (NVIDIA System Management Interface) is a command-line tool that comes with your NVIDIA drivers. It's your window into the GPU's current state.

Open a terminal (or use a `!` in a Jupyter/Colab cell) and run:

```bash
!nvidia-smi
```

**What to look for:**
*   **Fan, Temp, Perf, Pwr:Usage/Cap:** General GPU health and power usage.
*   **Memory-Usage:** This is the most important part for us. It shows `Used Memory / Total Memory` (e.g., `4505MiB / 11264MiB`). You can watch this while your model is training to see how much VRAM it's using.
*   **Processes:** Shows a list of all processes currently using the GPU. This is useful for finding other programs that might be consuming your VRAM.

You can also run it in a loop to watch the usage change over time:

```bash
# On Linux/macOS
!watch -n 1 nvidia-smi

# On Windows, there isn't a direct `watch` equivalent, but you can just run `nvidia-smi` repeatedly.
```

## Conclusion

You now know how to harness the power of GPUs to dramatically accelerate your deep learning workflows. By understanding the concept of the `device`, following the golden rule of device placement, and managing memory wisely, you can train larger and more complex models faster than ever before.

**Key Takeaways:**

1.  Always start your script by defining your `device`.
2.  Move your model to the `device` once.
3.  In your training loop, move your data batches to the `device`.
4.  **Always** wrap your evaluation/inference code in `with torch.no_grad():`.
5.  If you run out of memory, reduce your batch size.

## Self-Assessment Questions

1.  **Device Portability:** What line of code allows your script to run on a GPU if available, but fall back to the CPU if not?
2.  **The Golden Rule:** What will happen if you try to compute `model(data)` where `model` is on the GPU and `data` is on the CPU?
3.  **`no_grad`:** What are the two main benefits of using the `with torch.no_grad():` context manager?
4.  **Memory Error:** Your script crashes with a `CUDA out of memory` error. What is the very first thing you should try to fix it?
5.  **`nvidia-smi`:** You run `nvidia-smi` and see that a process named `chrome.exe` is using 1GB of your VRAM. What does this tell you, and what might you do about it if you are running out of memory?


