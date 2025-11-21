# Day 1: PyTorch Tensors - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Tensors, Memory, and Optimization

### 1. What is the difference between a View and a Copy in PyTorch? Give examples.
**Answer:**
*   **View**: Shares the same underlying `Storage`. Changing the view changes the original tensor. Fast ($O(1)$).
    *   Examples: `.view()`, `.squeeze()`, `.unsqueeze()`, `.t()`, slicing `x[:]`.
*   **Copy**: Allocates new memory. Independent data. Slower ($O(N)$).
    *   Examples: `.clone()`, `.contiguous()` (if not already), `.tensor()` (factory), `x + y` (out-of-place).

### 2. Why does `x.view()` sometimes fail with a runtime error?
**Answer:**
*   `.view()` requires the tensor to be **contiguous** in memory.
*   Operations like `transpose` or `permute` change the strides such that the logical order no longer matches the physical memory order.
*   **Fix**: Call `x.contiguous().view(...)`. This forces a copy to a new contiguous layout.

### 3. Explain the concept of "Strides". How does `x.t()` work internally?
**Answer:**
*   Strides are a tuple of integers representing the number of bytes/elements to skip to move to the next index in each dimension.
*   For a row-major $3 \times 3$ matrix, strides are `(3, 1)`.
*   `x.t()` simply **swaps the strides** to `(1, 3)` and swaps the shape. It does not move a single byte of data in memory.

### 4. What is the "Channels Last" memory format and why is it preferred for CNNs?
**Answer:**
*   Standard format is NCHW. "Channels Last" is NHWC.
*   Modern hardware (NVIDIA Tensor Cores) processes data in chunks (e.g., 32 elements).
*   In CNNs, operations happen across channels (dot product of input channels with kernel channels).
*   Having channels contiguous (NHWC) allows efficient vectorization (SIMD) and better cache locality for these operations.

### 5. What is the difference between `torch.Tensor` and `torch.tensor`?
**Answer:**
*   `torch.tensor(data)`: Factory function. Infers dtype. **Copies** data. Recommended.
*   `torch.Tensor(data)`: Constructor for the default tensor type (`FloatTensor`). Can be ambiguous (e.g., `torch.Tensor(5)` creates a vector of size 5 with garbage values, not scalar 5).
*   `torch.as_tensor(data)`: Preserves memory (no copy) if data is already a tensor/ndarray.

### 6. How does PyTorch handle broadcasting? What are the rules?
**Answer:**
*   Broadcasting allows arithmetic between tensors of different shapes.
*   **Rule 1**: Align shapes from the right.
*   **Rule 2**: Dimensions are compatible if they are equal OR one of them is 1.
*   The dimension of size 1 is logically expanded (stride set to 0) to match the other.

### 7. What is `torch.compile` and how does it optimize code?
**Answer:**
*   Introduced in PyTorch 2.0.
*   It captures the PyTorch program into a graph (Dynamo).
*   It optimizes the graph (Inductor) by performing **Kernel Fusion** (combining multiple ops into one kernel to reduce memory access) and generating optimized Triton/C++ code.

### 8. What is the difference between `x.to(device)` and `x.cuda()`?
**Answer:**
*   `x.cuda()`: Hardcoded to move to default GPU.
*   `x.to(device)`: More flexible. Can handle strings (`'cuda:0'`, `'cpu'`, `'mps'`) or device objects. Also handles dtype conversion (`x.to(torch.float16)`).
*   Both trigger a Host-to-Device copy if x is on CPU.

### 9. Explain `torch.einsum`. Why might it be faster than standard ops?
**Answer:**
*   Einstein Summation allows defining ops via index strings (`ik,kj->ij`).
*   It can be faster because it allows the backend to optimize the contraction order and avoid creating intermediate tensors for every step of a complex formula.

### 10. What is the difference between `torch.cat` and `torch.stack`?
**Answer:**
*   `cat`: Concatenates along an *existing* dimension. `(3, 4)` + `(3, 4)` -> `(6, 4)` (dim 0).
*   `stack`: Concatenates along a *new* dimension. `(3, 4)` + `(3, 4)` -> `(2, 3, 4)`.

### 11. How do you debug "CUDA Out of Memory" errors?
**Answer:**
*   Reduce batch size.
*   Check for accumulated gradients (did you `zero_grad`?).
*   Check for "zombie" tensors (references kept in a list).
*   Use `torch.cuda.empty_cache()` (though this just releases cached memory to OS, doesn't fix leaks).
*   Use Gradient Checkpointing (`torch.utils.checkpoint`).

### 12. What is `torch.nn.Buffer` vs `torch.nn.Parameter`?
**Answer:**
*   **Parameter**: Learnable weights. `requires_grad=True`. Returned by `model.parameters()`. Updated by optimizer.
*   **Buffer**: Non-learnable state (e.g., BatchNorm running mean). `requires_grad=False`. Saved in `state_dict`.

### 13. Why is `x += y` (in-place) dangerous in Autograd?
**Answer:**
*   It overwrites the values in memory.
*   If the original value of `x` was needed to compute the gradient of a past operation (via chain rule), that information is lost.
*   PyTorch raises a runtime error if this happens.

### 14. What is the `dtype` hierarchy in PyTorch?
**Answer:**
*   `float32` (default).
*   `float64` (double) - higher precision, 2x memory.
*   `float16` (half) - lower precision, 0.5x memory, requires scaling.
*   `bfloat16` - truncated float32, better stability than fp16.

### 15. How does PyTorch interface with NumPy?
**Answer:**
*   `torch.from_numpy(arr)`: Zero-copy view.
*   `tensor.numpy()`: Zero-copy view (if on CPU).
*   They share the same memory block. Modifying one affects the other.

### 16. What is "Pinned Memory" (`pin_memory=True`)?
**Answer:**
*   Allocates tensor in Page-Locked (non-swappable) RAM.
*   Transfers from Pinned Memory to GPU are faster and can be asynchronous.
*   Used in `DataLoader`.

### 17. What is `torch.set_grad_enabled(False)`?
**Answer:**
*   Global switch to disable gradient tracking.
*   Similar to `torch.no_grad()` but can be toggled imperatively.

### 18. How do you handle variable length sequences?
**Answer:**
*   **Padding**: Add zeros to match max length.
*   **Masking**: Use a boolean mask to ignore padded values in Attention/Loss.
*   **Packing**: `pack_padded_sequence` for RNNs (optimizes computation by skipping pads).

### 19. What is the difference between `F.relu` and `nn.ReLU`?
**Answer:**
*   `nn.ReLU`: A Class (Module). Has state (though ReLU has none). Used in `nn.Sequential`.
*   `F.relu`: A Function (Functional). Stateless. Used in `forward` method.

### 20. How does `torch.distributed` handle tensors?
**Answer:**
*   **DDP (DistributedDataParallel)**: Replicates model on each GPU. Splits data. Syncs gradients (AllReduce) after backward pass.
*   **FSDP (FullyShardedDataParallel)**: Shards model parameters and optimizer state across GPUs.
