# Day 3.2: Tensor Operations & Manipulations - A Practical Guide

## Introduction: Making Tensors Work

In the previous guide, we learned what a tensor *is*. Now, we will learn what a tensor *does*. The power of PyTorch comes from its vast library of optimized operations that can be performed on tensors. These operations range from simple element-wise arithmetic to complex matrix multiplications and dimension manipulations.

This guide will provide a practical tour of the most common and essential tensor operations. Mastering these is key to implementing any neural network architecture, as they form the building blocks of the forward pass.

**Today's Learning Objectives:**

1.  **Perform Element-wise Operations:** Understand and use basic arithmetic, logical, and comparison operators on tensors.
2.  **Master Reduction Operations:** Learn how to aggregate tensor data using `sum`, `mean`, `max`, and `min`, and understand the critical `dim` argument.
3.  **Execute Matrix Operations:** Differentiate between element-wise multiplication (`*`) and matrix multiplication (`@`), and learn how to use `torch.matmul`.
4.  **Understand In-place Operations:** Learn about operations with a trailing underscore (`_`) and when they are (and are not) appropriate to use.
5.  **Become a Dimension-Manipulating Expert:** Master the use of `view`, `reshape`, `squeeze`, `unsqueeze`, and `permute` to change a tensor's shape to fit your model's requirements.

--- 

## Part 1: Element-wise Operations

These operations are applied independently to each element of a tensor. They require the tensors to either have the same shape or be "broadcastable" (more on this later).

```python
import torch

print("--- Part 1: Element-wise Operations ---")

tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# --- Arithmetic Operations ---
# Addition
add_result = tensor_a + tensor_b
# or: torch.add(tensor_a, tensor_b)
print(f"Addition Result:\n{add_result}\n")

# Subtraction
sub_result = tensor_a - 5 # Broadcasting a scalar
print(f"Subtraction Result (with broadcasting):\n{sub_result}\n")

# Element-wise Multiplication
# THIS IS NOT MATRIX MULTIPLICATION!
mul_result = tensor_a * tensor_b
print(f"Element-wise Multiplication Result:\n{mul_result}\n")

# Division
div_result = tensor_b / 2
print(f"Division Result:\n{div_result}\n")

# --- Comparison Operations ---
# These return a boolean tensor.
comparison_result = tensor_a > 2
print(f"Comparison Result (tensor_a > 2):\n{comparison_result}\n")

# --- Logical Operations ---
# You can use boolean tensors for indexing.
print(f"Original Tensor A:\n{tensor_a}")
print(f"Elements of A greater than 2: {tensor_a[tensor_a > 2]}\n")

# --- Other Mathematical Functions ---
# PyTorch offers a huge range of functions like torch.sin, torch.cos, torch.exp, torch.log, etc.
exp_result = torch.exp(tensor_a.float()) # exp requires a float tensor
print(f"Exponential of Tensor A:\n{exp_result}")
```

### 1.1. Broadcasting

Broadcasting allows PyTorch to perform operations on tensors of different, but compatible, sizes. The smaller tensor is "broadcast" across the larger tensor. This is what allowed us to do `tensor_a - 5` above.

**Rule:** Two tensors are broadcastable if, for each trailing dimension (checked from right to left), the dimension sizes are either equal, one of them is 1, or one of them does not exist.

```python
# --- Broadcasting Example ---
mat = torch.randn(3, 4) # Shape (3, 4)
vec = torch.randn(4)    # Shape (4)

# The vec is broadcast across the 3 rows of mat.
# It\'s as if we created a (3, 4) tensor by stacking vec 3 times.
result = mat + vec

print("\n--- Broadcasting Demo ---")
print(f"Matrix shape: {mat.shape}")
print(f"Vector shape: {vec.shape}")
print(f"Result shape after broadcasting: {result.shape}")
```

---

## Part 2: Reduction Operations

A reduction operation reduces the number of dimensions of a tensor by performing an operation across a dimension.

### 2.1. The Critical `dim` Argument

This is one of the most important concepts to master. The `dim` argument specifies the dimension **to be reduced or eliminated**.

```python
import torch

# Create a sample tensor
tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.]]) # Shape (2, 3)

print("\n--- Part 2: Reduction Operations ---")
print(f"Original Tensor:\n{tensor}\n")

# --- Reducing the whole tensor ---
# If you don\'t specify a `dim`, the operation is applied to the entire tensor.
sum_total = tensor.sum()
print(f"Total sum of all elements: {sum_total.item()}\n")

# --- Reducing along a specific dimension ---
# Reduce along dimension 0 (the rows). We are collapsing the rows.
# The result will have shape (3,).
sum_dim0 = tensor.sum(dim=0)
print(f"Sum along dim=0 (collapsing rows): {sum_dim0}")
print(f"Result shape: {sum_dim0.shape}\n")

# Reduce along dimension 1 (the columns). We are collapsing the columns.
# The result will have shape (2,).
sum_dim1 = tensor.sum(dim=1)
print(f"Sum along dim=1 (collapsing columns): {sum_dim1}")
print(f"Result shape: {sum_dim1.shape}\n")

# --- `argmax` and `argmin` ---
# These return the *indices* of the maximum/minimum value.
# This is extremely useful, for example, in getting the predicted class from a model\'s output.
model_output = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05]]) # (batch, num_classes)

# We want to find the predicted class (the index of the max value) for each sample in the batch.
# So we reduce along dimension 1.
predicted_classes = torch.argmax(model_output, dim=1)
print(f"Model Output (Logits):\n{model_output}")
print(f"Predicted classes (argmax along dim=1): {predicted_classes}")
```

---

## Part 3: Matrix Operations

This is where linear algebra comes to life.

### 3.1. Matrix Multiplication

This is the most fundamental operation in deep learning, used in every linear layer.

**Rule:** For a matrix multiplication `A @ B`, if `A` has shape `(m, n)`, then `B` must have shape `(n, p)`. The resulting matrix will have shape `(m, p)`.

```python
import torch

print("\n--- Part 3: Matrix Operations ---")

mat1 = torch.randn(3, 4) # Shape (3, 4)
mat2 = torch.randn(4, 5) # Shape (4, 5)

# Use the `@` operator for matrix multiplication
# This is the preferred, modern way.
result = mat1 @ mat2

# Alternatively, use torch.matmul()
# result_alt = torch.matmul(mat1, mat2)

print(f"Matrix 1 shape: {mat1.shape}")
print(f"Matrix 2 shape: {mat2.shape}")
print(f"Result shape after matmul: {result.shape}\n") # Expected shape (3, 5)

# Remember: Element-wise multiplication is different!
mat3 = torch.randn(3, 4)
# This would raise an error because the shapes are not the same.
# mul_result = mat1 * mat3 
```

### 3.2. Transpose

A transpose (`.T`) swaps the dimensions of a tensor.

```python
mat = torch.tensor([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)

print(f"Original Matrix (2, 3):\n{mat}")
print(f"Transposed Matrix (3, 2):\n{mat.T}")
```

---

## Part 4: In-place Operations

An in-place operation modifies the content of a tensor directly without creating a new tensor. In PyTorch, these operations are denoted by a trailing underscore, e.g., `add_()`.

**Warning:** In-place operations can be problematic for `autograd`. PyTorch needs the original tensor values to compute gradients during the backward pass. If you modify a tensor in-place, you might destroy information needed for `backward()`. It is **strongly recommended to avoid in-place operations** in neural networks, especially within the `forward` pass.

```python
import torch

print("\n--- Part 4: In-place Operations ---")

# --- Standard (out-of-place) operation ---
tensor_std = torch.ones(3)
print(f"Original tensor (standard): {tensor_std}")
print(f"ID of original tensor: {id(tensor_std)}")

result_std = tensor_std.add(torch.ones(3))
print(f"Result of standard add: {result_std}")
print(f"ID of result tensor: {id(result_std)}) # A new tensor was created
print(f"Original tensor is unchanged: {tensor_std}\n")

# --- In-place operation ---
tensor_inplace = torch.ones(3)
print(f"Original tensor (in-place): {tensor_inplace}")
print(f"ID of original tensor: {id(tensor_inplace)}")

# The add_() method modifies the tensor directly
tensor_inplace.add_(torch.ones(3))
print(f"Tensor after in-place add: {tensor_inplace}")
print(f"ID of tensor after modification: {id(tensor_inplace)}) # The ID is the same!
```

---

## Part 5: Dimension Manipulation

Often, you need to reshape a tensor to make it compatible with a particular layer. For example, after a convolutional layer, you need to flatten the output before feeding it to a linear layer.

### 5.1. `view()` and `reshape()`

Both are used to change the shape of a tensor. `view()` requires the new shape to be compatible with the original shape and memory layout. `reshape()` is more flexible and will create a copy if necessary.

**Best Practice:** Use `reshape()` unless you have a specific reason to use `view()`. It\'s more robust.

```python
import torch

print("\n--- Part 5: Dimension Manipulation ---")

# Imagine the output of a conv layer for a batch of 4 images
# Shape: (batch_size, channels, height, width)
conv_output = torch.randn(4, 16, 8, 8)

# We need to flatten this for a linear layer.
# We want to keep the batch dimension (4) and flatten the rest (16*8*8 = 1024).
# The -1 tells PyTorch to infer the correct size for that dimension.
flattened = conv_output.reshape(4, -1)

print(f"Original shape: {conv_output.shape}")
print(f"Shape after reshape(4, -1): {flattened.shape}\n")
```

### 5.2. `squeeze()` and `unsqueeze()`

These are used to remove or add dimensions of size 1.

*   `squeeze()`: Removes dimensions of size 1.
*   `unsqueeze(dim)`: Adds a dimension of size 1 at the specified position.

This is extremely common for adding a batch dimension to a single data sample.

```python
# --- unsqueeze() ---
# A single image from a dataset
single_image = torch.randn(3, 224, 224) # (channels, height, width)
print(f"Shape of a single image: {single_image.shape}")

# Most models expect a batch dimension. We add it at the beginning (dim=0).
batched_image = single_image.unsqueeze(0)
print(f"Shape after unsqueeze(0): {batched_image.shape}\n") # Now (1, 3, 224, 224)

# --- squeeze() ---
# Let\'s say a model outputs a tensor with an extra dimension
model_output = torch.randn(1, 10, 1)
print(f"Original output shape: {model_output.shape}")

squeezed_output = model_output.squeeze()
print(f"Shape after squeeze(): {squeezed_output.shape}")
```

### 5.3. `permute()`

`permute()` rearranges the dimensions of a tensor. This is often needed when a library expects a different dimension order.

For example, Matplotlib expects image data in the format `(H, W, C)` (Height, Width, Channels), but PyTorch uses `(C, H, W)`.

```python
# A PyTorch image tensor
pytor ch_image = torch.randn(3, 28, 28) # (C, H, W)

# We want to change the order to (H, W, C)
matplotlib_image = pytorch_image.permute(1, 2, 0) # The arguments are the new order of dimensions

print(f"Original PyTorch image shape (C, H, W): {pytorch_image.shape}")
print(f"Permuted image shape for Matplotlib (H, W, C): {matplotlib_image.shape}")
```

## Conclusion

You are now equipped with the fundamental tools for manipulating tensors. Neural networks are, at their core, a sequence of these operations. A forward pass is nothing more than a flow of tensors through a series of element-wise operations, matrix multiplications, and dimension manipulations.

By understanding these building blocks, you can now read and understand the code for complex architectures, debug shape-related errors with confidence, and implement your own custom models from scratch.

## Self-Assessment Questions

1.  **`*` vs. `@`:** You have two tensors, `A` and `B`, both with shape `[10, 10]`. What is the difference between the results of `A * B` and `A @ B`?
2.  **Reduction `dim`:** You have a tensor of shape `[32, 10, 128]` representing a batch of 32 sentences, where each sentence has 10 words, and each word is a 128-dimensional vector. How would you compute the average word vector for each sentence? What would be the shape of the resulting tensor?
3.  **In-place:** Why is it generally a bad idea to use an in-place operation like `x.add_(1)` inside your model\'s `forward` pass?
4.  **`unsqueeze`:** You have a single sentence represented as a tensor of shape `[15]` (15 word indices). Your model\'s embedding layer expects a batch of sentences with shape `[batch_size, seq_len]`. How would you add a batch dimension to your single sentence tensor?
5.  **`reshape`:** You have the output of a CNN layer with shape `[64, 32, 7, 7]` (batch, channels, height, width). You want to feed this to a linear layer. What shape would you `reshape` it to?

