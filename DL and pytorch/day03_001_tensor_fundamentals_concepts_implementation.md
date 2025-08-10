# Day 3.1: Tensor Fundamentals & Concepts - A Practical Deep Dive

## Introduction: The Atom of Deep Learning

If deep learning is a universe, the **Tensor** is its fundamental particle. Everything in PyTorch—from your input data, to your model's weights, to the gradients that enable learning—is represented as a tensor. Mastering the tensor is the single most important step towards becoming proficient in PyTorch.

A tensor is a multi-dimensional array, a generalization of vectors and matrices to an arbitrary number of dimensions. But a PyTorch `Tensor` is more than just a data container; it's a super-powered array with built-in capabilities for GPU acceleration and automatic differentiation.

This guide will provide a comprehensive, practical exploration of PyTorch tensors. We will cover how to create them, their essential properties, and how to manipulate them.

**Today's Learning Objectives:**

1.  **Master Tensor Creation:** Learn the various ways to create tensors (`torch.tensor`, `torch.rand`, `torch.zeros`, etc.).
2.  **Understand Core Tensor Attributes:** Become fluent with `shape`, `dtype`, `device`, and `requires_grad`.
3.  **Grasp Data Types (`dtype`):** Understand why using the correct data type (e.g., `float32` vs. `int64`) is critical.
4.  **Learn the Basics of Indexing and Slicing:** Access and modify parts of a tensor with confidence.
5.  **Understand the CPU/GPU distinction (`device`):** Learn the concept of moving tensors between different hardware devices.
6.  **Bridge the Gap with NumPy:** See how to seamlessly convert between PyTorch Tensors and NumPy arrays.

--- 

## Part 1: Creating Tensors

There are many ways to bring a tensor into existence. Let's explore the most common ones.

```python
import torch
import numpy as np

print("--- Part 1: Tensor Creation ---")

# --- 1. From existing data (the most common way) ---
# Create a tensor from a Python list.
# PyTorch will infer the data type.
list_data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(list_data)
print(f"Tensor from list:\n{tensor_from_list}")
print(f"Data type: {tensor_from_list.dtype}\n") # Infers torch.int64

# Create a tensor from a NumPy array.
# This is a zero-copy operation on CPU, meaning they share the same memory location.
numpy_array = np.array([[5., 6.], [7., 8.]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy array:\n{tensor_from_numpy}")
print(f"Data type: {tensor_from_numpy.dtype}\n") # Infers torch.float64 from NumPy

# --- 2. Creating tensors with a specific size ---
# Create a tensor of a given shape, filled with random numbers from a uniform distribution on [0, 1).
shape = (2, 3)
rand_tensor = torch.rand(shape)
print(f"Random tensor of shape {shape}:\n{rand_tensor}\n")

# Create a tensor filled with random numbers from a standard normal distribution (mean 0, variance 1).
randn_tensor = torch.randn(shape)
print(f"Standard normal tensor of shape {shape}:\n{randn_tensor}\n")

# Create a tensor filled with zeros.
zeros_tensor = torch.zeros(shape)
print(f"Zeros tensor of shape {shape}:\n{zeros_tensor}\n")

# Create a tensor filled with ones.
ones_tensor = torch.ones(shape)
print(f"Ones tensor of shape {shape}:\n{ones_tensor}\n")

# --- 3. Creating tensors like another tensor ---
# Often, you want to create a new tensor with the same properties (shape, dtype, device)
# as an existing tensor. The `_like` methods are perfect for this.
x_data = torch.tensor([[1, 2, 3]], dtype=torch.float32)

rand_like_x = torch.rand_like(x_data) # Creates a random tensor with the same shape and dtype as x_data
print(f"Original tensor:\n{x_data}")
print(f"Random tensor like original:\n{rand_like_x}\n")

zeros_like_x = torch.zeros_like(x_data)
print(f"Zeros tensor like original:\n{zeros_like_x}\n")
```

--- 

## Part 2: The Four Essential Tensor Attributes

Every tensor has four properties that you will use constantly. They define what the tensor is, where it is, and what it's for.

*   **`tensor.shape`** (or `tensor.size()`): The dimensions of the tensor. A tuple of integers.
*   **`tensor.dtype`:** The data type of the elements in the tensor.
*   **`tensor.device`:** The device (CPU or GPU) where the tensor's data is stored.
*   **`tensor.requires_grad`:** A boolean indicating whether PyTorch should track operations on this tensor for automatic differentiation.

```python
import torch

# Create a tensor with specific properties
# We explicitly set the dtype, device, and requires_grad.
tensor = torch.randn(2, 4, dtype=torch.float32, device="cpu", requires_grad=True)

print("--- Part 2: Essential Tensor Attributes ---")
print(f"The Tensor:\n{tensor}\n")

# 1. Shape: The dimensions of the tensor
print(f"Shape: {tensor.shape}")
print(f"  - It has {tensor.dim()} dimensions.")
print(f"  - The size of the 0th dimension is {tensor.shape[0]}")
print(f"  - The size of the 1st dimension is {tensor.shape[1]}\n")

# 2. Dtype: The data type of each element
print(f"Data Type (dtype): {tensor.dtype}")
print("  - This is the standard 32-bit floating point type for deep learning.\n")

# 3. Device: Where the tensor is stored
print(f"Device: {tensor.device}")
print("  - This tensor lives on the CPU.\n")

# 4. Requires Grad: Whether to track gradients
print(f"Requires Grad: {tensor.requires_grad}")
print("  - PyTorch will build a computation graph for this tensor.")

# We can see the `grad_fn` because we performed an operation (randn) on a tensor that requires grad.
print(f"Gradient Function (grad_fn): {tensor.grad_fn}")
```

--- 

## Part 3: Data Types (`dtype`) - A Critical Detail

Using the correct data type is essential for performance and correctness.

*   **Floating Point Types (`torch.float32`, `torch.float16`, `torch.float64`):**
    *   `torch.float32` (or `torch.float`): The **default and most common** type for model weights and data. It offers a good balance of precision and memory usage.
    *   `torch.float16` (or `torch.half`): **Half-precision.** Used for mixed-precision training on modern GPUs (like NVIDIA's Tensor Core GPUs) to significantly speed up training and reduce memory usage.
    *   `torch.float64` (or `torch.double`): **Double-precision.** Rarely used in deep learning as it's often overkill and much slower.

*   **Integer Types (`torch.int64`, `torch.int32`, etc.):**
    *   `torch.int64` (or `torch.long`): The **default integer type**. Most commonly used for **labels** in classification problems.

**Type Mismatch Errors:** A common source of bugs is trying to perform an operation on tensors with different dtypes.

```python
import torch

# --- Example of a dtype mismatch ---
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
int_tensor = torch.tensor([4, 5, 6], dtype=torch.int64)

try:
    # This will raise a RuntimeError because you can't add a float and an int tensor directly.
    result = float_tensor + int_tensor
except RuntimeError as e:
    print("--- Data Type (dtype) Demonstration ---")
    print(f"Error encountered: {e}\n")

# --- The Fix: Casting ---
# You must explicitly cast one of the tensors to match the other's dtype.
int_tensor_as_float = int_tensor.to(torch.float32)
# Or more concisely: int_tensor.float()

result = float_tensor + int_tensor_as_float
print("Casting the integer tensor to float allows the operation to succeed:")
print(f"  - Original float tensor: {float_tensor}")
print(f"  - Original int tensor: {int_tensor}")
print(f"  - Int tensor cast to float: {int_tensor_as_float}")
print(f"  - Result: {result}")
print(f"  - Result dtype: {result.dtype}")
```

--- 

## Part 4: Indexing, Slicing, and Joining

Accessing and manipulating parts of tensors works very similarly to NumPy.

```python
import torch

# Create a sample tensor
# Shape: (batch_size, channels, height, width)
tensor = torch.randn(4, 3, 28, 28)

print("\n--- Part 4: Indexing, Slicing, and Joining ---")
print(f"Original tensor shape: {tensor.shape}\n")

# --- Indexing and Slicing ---
# Get the first sample in the batch
first_sample = tensor[0]
print(f"Shape of the first sample: {first_sample.shape}")

# Get the first channel of the first sample
first_channel_of_first_sample = tensor[0, 0]
print(f"Shape of the first channel of the first sample: {first_channel_of_first_sample.shape}")

# Get the top-left 14x14 patch of the first channel of the first sample
# We use : to select all elements in a dimension.
# We use start:end for slicing.
top_left_patch = tensor[0, 0, :14, :14]
print(f"Shape of the 14x14 patch: {top_left_patch.shape}\n")

# --- Joining Tensors ---
# `torch.cat` concatenates tensors along an existing dimension.
t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)

# Concatenate along dimension 0 (rows)
cat_dim0 = torch.cat([t1, t2], dim=0)
print(f"Concatenating two (2, 3) tensors along dim=0 results in shape: {cat_dim0.shape}")

# Concatenate along dimension 1 (columns)
cat_dim1 = torch.cat([t1, t2], dim=1)
print(f"Concatenating two (2, 3) tensors along dim=1 results in shape: {cat_dim1.shape}\n")

# `torch.stack` creates a *new* dimension.
t1 = torch.randn(3, 4)
t2 = torch.randn(3, 4)

# Stack the two (3, 4) tensors. This will create a new tensor of shape (2, 3, 4).
stacked_tensor = torch.stack([t1, t2], dim=0)
print(f"Stacking two (3, 4) tensors results in shape: {stacked_tensor.shape}")
```

--- 

## Part 5: The Bridge to NumPy

PyTorch and NumPy are seamlessly interoperable. You can convert a PyTorch Tensor to a NumPy array and vice-versa with ease.

**Important:** If the Tensor is on the CPU, the PyTorch Tensor and NumPy array will **share the same underlying memory location**. This means that changing the NumPy array will change the PyTorch Tensor, and vice-versa. This is a feature, not a bug, as it avoids expensive data copies.

If the Tensor is on the GPU, you must first move it to the CPU before converting it to NumPy.

```python
import torch
import numpy as np

print("\n--- Part 5: The Bridge to NumPy ---")

# --- Tensor to NumPy ---
cpu_tensor = torch.ones(5)
print(f"Original CPU Tensor: {cpu_tensor}")

numpy_array = cpu_tensor.numpy()
print(f"Converted NumPy array: {numpy_array}")

# Modify the tensor in-place
cpu_tensor.add_(1)

print("After modifying the tensor in-place...")
print(f"  - The tensor is now: {cpu_tensor}")
print(f"  - The NumPy array has also changed: {numpy_array}\n") # It changed too!

# --- NumPy to Tensor ---
numpy_array_2 = np.zeros(5)

tensor = torch.from_numpy(numpy_array_2)
print(f"Original NumPy array: {numpy_array_2}")
print(f"Converted Tensor: {tensor}")

# Modify the NumPy array
np.add(numpy_array_2, 1, out=numpy_array_2)

print("After modifying the NumPy array...")
print(f"  - The NumPy array is now: {numpy_array_2}")
print(f"  - The tensor has also changed: {tensor}\n") # It changed too!

# --- GPU Tensors ---
if torch.cuda.is_available():
    gpu_tensor = torch.ones(5, device="cuda")
    try:
        # This will raise an error
        gpu_tensor.numpy()
    except TypeError as e:
        print(f"Error trying to convert GPU tensor to NumPy: {e}")
    
    # The fix: first move to CPU, then convert
    cpu_tensor_from_gpu = gpu_tensor.cpu()
    numpy_from_gpu = cpu_tensor_from_gpu.numpy()
    print(f"Successfully converted GPU tensor to NumPy via CPU: {numpy_from_gpu}")
```

## Conclusion

The PyTorch `Tensor` is a rich and powerful data structure that serves as the foundation for everything else in the library. By understanding its core attributes (`shape`, `dtype`, `device`, `requires_grad`) and mastering the fundamental creation and manipulation routines, you have built the essential vocabulary needed to read and write PyTorch code effectively.

In the next guide, we will build on this foundation to explore the vast array of mathematical and logical operations that can be performed on these tensors.

## Self-Assessment Questions

1.  **Creation:** What is the difference between `torch.rand(2, 3)` and `torch.randn(2, 3)`?
2.  **Attributes:** You have a tensor `x`. How do you check how many dimensions it has? How do you check if it's on the GPU?
3.  **Data Types:** Your model's output is a `torch.float32` tensor of probabilities. Your true labels are in a `torch.int64` tensor. What must you do before you can calculate a loss like MSE between them? (Note: Some loss functions like `CrossEntropyLoss` handle this internally, but for many others, you need to be explicit).
4.  **`cat` vs. `stack`:** You have a list of 5 tensors, each with shape `[32, 16]`. What will be the shape of the result if you `torch.cat` them along `dim=0`? What if you `torch.stack` them along `dim=0`?
5.  **NumPy Bridge:** You have a NumPy array `my_array`. You create a tensor `my_tensor = torch.from_numpy(my_array)`. If you then execute `my_array[0] = 100`, will `my_tensor` be affected? Why or why not?

```