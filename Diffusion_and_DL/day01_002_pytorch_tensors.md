# Day 1 - Part 2: PyTorch Basics - Tensors and Operations

## 📚 Learning Objectives
By the end of this section, you will understand:
- What PyTorch tensors are and how they differ from NumPy arrays
- Tensor creation, manipulation, and mathematical operations
- GPU acceleration benefits and tensor device management
- Broadcasting rules and memory-efficient operations
- Performance considerations and optimization techniques

---

## 🔍 What is PyTorch?

### Definition and Core Philosophy
PyTorch is an open-source machine learning framework developed by Meta's AI Research lab (FAIR). It provides:
1. **Dynamic Computation Graphs**: Define-by-run execution model
2. **GPU Acceleration**: Seamless CPU-GPU operations
3. **Automatic Differentiation**: Built-in gradient computation
4. **Pythonic Design**: Intuitive, NumPy-like interface
5. **Research-Friendly**: Easy prototyping and debugging

### PyTorch Ecosystem
```
PyTorch Core ──── torch (tensor operations, autograd)
    ├── torchvision (computer vision utilities)
    ├── torchaudio (audio processing)
    ├── torchtext (natural language processing)
    ├── TorchServe (model serving)
    └── PyTorch Lightning (high-level training framework)
```

---

## 🔢 Understanding Tensors

### What are Tensors?
A **tensor** is a multi-dimensional array that serves as the fundamental data structure in PyTorch. Think of tensors as generalizations of scalars, vectors, and matrices:

```
Scalar (0D tensor): 5
Vector (1D tensor): [1, 2, 3, 4]
Matrix (2D tensor): [[1, 2], [3, 4]]
3D Tensor: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

### Tensor Properties
Every PyTorch tensor has several key properties:

1. **Data Type (dtype)**: The type of elements (float32, int64, bool, etc.)
2. **Shape**: The dimensions of the tensor
3. **Device**: Where the tensor is stored (CPU or GPU)
4. **Layout**: How data is stored in memory (dense or sparse)
5. **Requires Gradient**: Whether to compute gradients for this tensor

### Creating Tensors

#### 1. From Python Lists and NumPy Arrays
```python
import torch
import numpy as np

# From Python list
tensor_from_list = torch.tensor([1, 2, 3, 4])
print(f"From list: {tensor_from_list}")

# From NumPy array (shares memory by default)
numpy_array = np.array([1, 2, 3, 4])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"From NumPy: {tensor_from_numpy}")

# Creating a copy (doesn't share memory)
tensor_copy = torch.tensor(numpy_array)
```

#### 2. Initialization Functions
```python
# Zeros and ones
zeros = torch.zeros(3, 4)  # 3x4 tensor filled with zeros
ones = torch.ones(2, 3, 4)  # 2x3x4 tensor filled with ones
full = torch.full((2, 3), 7)  # 2x3 tensor filled with 7

# Identity matrix
identity = torch.eye(4)  # 4x4 identity matrix

# Random tensors
random_uniform = torch.rand(3, 4)  # Uniform distribution [0, 1)
random_normal = torch.randn(3, 4)  # Standard normal distribution
random_int = torch.randint(0, 10, (3, 4))  # Random integers

# Range tensors
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

#### 3. Tensor-like Creation
```python
# Create tensors with same properties as existing tensors
x = torch.randn(2, 3)
zeros_like = torch.zeros_like(x)  # Same shape, dtype, device as x
ones_like = torch.ones_like(x)
rand_like = torch.rand_like(x)
```

---

## 🆚 PyTorch vs NumPy: Key Differences

### Similarities
| Feature | PyTorch | NumPy |
|---------|---------|--------|
| Multi-dimensional arrays | ✅ | ✅ |
| Broadcasting | ✅ | ✅ |
| Mathematical operations | ✅ | ✅ |
| Indexing and slicing | ✅ | ✅ |
| Shape manipulation | ✅ | ✅ |

### Key Differences

#### 1. GPU Acceleration
```python
# NumPy - CPU only
import numpy as np
numpy_array = np.array([1, 2, 3, 4])
# No direct GPU support

# PyTorch - CPU and GPU support  
import torch
cpu_tensor = torch.tensor([1, 2, 3, 4])
gpu_tensor = cpu_tensor.cuda()  # Move to GPU
# or
gpu_tensor = torch.tensor([1, 2, 3, 4], device='cuda')
```

#### 2. Automatic Differentiation
```python
# NumPy - Manual gradient computation
def numpy_gradient():
    x = np.array([2.0])
    # Forward pass
    y = x ** 2
    # Manual backward pass
    dy_dx = 2 * x
    return y, dy_dx

# PyTorch - Automatic differentiation
def pytorch_gradient():
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward()  # Automatic gradient computation
    return y, x.grad
```

#### 3. Dynamic vs Static
```python
# NumPy operations create new arrays
a = np.array([1, 2, 3])
b = a + 1  # Creates new array

# PyTorch supports both new tensor creation and in-place operations
a = torch.tensor([1, 2, 3])
b = a + 1      # Creates new tensor
a.add_(1)      # In-place operation (modifies a)
```

### Performance Comparison
```python
import time

# Timing function
def time_operation(func, *args, iterations=1000):
    start = time.time()
    for _ in range(iterations):
        result = func(*args)
    end = time.time()
    return (end - start) / iterations

# Large matrix multiplication comparison
size = 1000
np_a = np.random.randn(size, size).astype(np.float32)
np_b = np.random.randn(size, size).astype(np.float32)

torch_a = torch.randn(size, size, dtype=torch.float32)
torch_b = torch.randn(size, size, dtype=torch.float32)

# CPU comparison
numpy_time = time_operation(np.dot, np_a, np_b)
pytorch_cpu_time = time_operation(torch.matmul, torch_a, torch_b)

print(f"NumPy time: {numpy_time:.6f}s")
print(f"PyTorch CPU time: {pytorch_cpu_time:.6f}s")
```

---

## 🧮 Tensor Operations

### 1. Mathematical Operations

#### Element-wise Operations
```python
a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
b = torch.tensor([2, 3, 4, 5], dtype=torch.float32)

# Basic arithmetic
addition = a + b           # [3, 5, 7, 9]
subtraction = a - b        # [-1, -1, -1, -1]
multiplication = a * b     # [2, 6, 12, 20]
division = a / b           # [0.5, 0.667, 0.75, 0.8]
power = a ** 2             # [1, 4, 9, 16]

# Mathematical functions
sqrt = torch.sqrt(a)       # [1, 1.414, 1.732, 2]
exp = torch.exp(a)         # [2.718, 7.389, 20.086, 54.598]
log = torch.log(a)         # [0, 0.693, 1.099, 1.386]
sin = torch.sin(a)         # [0.841, 0.909, 0.141, -0.757]
```

#### Matrix Operations
```python
# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # or A @ B, shape: (3, 5)

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)  # 10 matrices of size 3x4
batch_B = torch.randn(10, 4, 5)  # 10 matrices of size 4x5
batch_C = torch.bmm(batch_A, batch_B)  # Shape: (10, 3, 5)

# Linear algebra operations
matrix = torch.randn(4, 4)
determinant = torch.det(matrix)
inverse = torch.inverse(matrix)
eigenvals, eigenvecs = torch.eig(matrix, eigenvectors=True)
```

### 2. Reduction Operations
```python
tensor = torch.randn(3, 4, 5)

# Sum operations
total_sum = torch.sum(tensor)              # Sum all elements
dim_sum = torch.sum(tensor, dim=1)         # Sum along dimension 1
keepdim_sum = torch.sum(tensor, dim=1, keepdim=True)  # Keep dimensions

# Other reductions
mean = torch.mean(tensor)
std = torch.std(tensor)
var = torch.var(tensor)
max_val, max_indices = torch.max(tensor, dim=2)
min_val, min_indices = torch.min(tensor, dim=2)
```

### 3. Shape Manipulation
```python
x = torch.randn(2, 3, 4)

# Reshaping
reshaped = x.view(6, 4)        # Must have compatible sizes
reshaped = x.reshape(24)       # More flexible than view
flattened = x.flatten()        # Flatten to 1D

# Dimension manipulation
squeezed = x.unsqueeze(0)      # Add dimension: (1, 2, 3, 4)
squeezed = x.squeeze()         # Remove dimensions of size 1
transposed = x.transpose(0, 2) # Swap dimensions 0 and 2
permuted = x.permute(2, 0, 1)  # Reorder dimensions

# Concatenation and splitting
y = torch.randn(2, 3, 4)
concatenated = torch.cat([x, y], dim=0)    # Shape: (4, 3, 4)
stacked = torch.stack([x, y], dim=0)       # Shape: (2, 2, 3, 4)
chunks = torch.chunk(concatenated, 2, dim=0)  # Split into 2 chunks
```

---

## 📡 Broadcasting Rules

Broadcasting allows operations between tensors of different shapes, following specific rules:

### Broadcasting Rules
1. Compare dimensions from the **rightmost** (trailing) dimension
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1, OR
   - One of them doesn't exist
3. Missing dimensions are assumed to be 1

### Examples
```python
# Example 1: Compatible shapes
a = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
b = torch.randn(1, 4, 5)  # Shape: (1, 4, 5)
result = a + b            # Result shape: (3, 4, 5)

# Example 2: Scalar broadcasting
scalar = torch.tensor(5.0)
matrix = torch.randn(2, 3)
result = matrix + scalar   # Scalar is broadcast to (2, 3)

# Example 3: Vector broadcasting
vector = torch.randn(3)    # Shape: (3,)
matrix = torch.randn(2, 3) # Shape: (2, 3)
result = matrix + vector   # Vector becomes (1, 3), then (2, 3)

# Example 4: Incompatible shapes (will raise error)
try:
    a = torch.randn(3, 4)
    b = torch.randn(2, 3)
    result = a + b  # Error: shapes not compatible
except RuntimeError as e:
    print(f"Broadcasting error: {e}")
```

### Broadcasting Best Practices
```python
# Explicit broadcasting for clarity
a = torch.randn(3, 1, 5)
b = torch.randn(1, 4, 5)
# Instead of relying on implicit broadcasting
result = a + b

# Make broadcasting explicit
a_expanded = a.expand(3, 4, 5)  # Doesn't create new memory
b_expanded = b.expand(3, 4, 5)
result = a_expanded + b_expanded
```

---

## ⚡ GPU Acceleration

### Device Management
```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU")

# Create tensors on specific devices
cpu_tensor = torch.randn(1000, 1000)
gpu_tensor = torch.randn(1000, 1000, device='cuda')

# Move tensors between devices
cpu_to_gpu = cpu_tensor.to('cuda')
gpu_to_cpu = gpu_tensor.to('cpu')
```

### Performance Comparison: CPU vs GPU
```python
import time

def benchmark_matmul(size=2000, device='cpu'):
    """Benchmark matrix multiplication on specified device"""
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm-up (important for GPU)
    for _ in range(10):
        _ = torch.matmul(a, b)
    
    # Synchronization for accurate timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing
    start_time = time.time()
    for _ in range(100):
        result = torch.matmul(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    
    return avg_time

# Compare performance
cpu_time = benchmark_matmul(device='cpu')
if torch.cuda.is_available():
    gpu_time = benchmark_matmul(device='cuda')
    speedup = cpu_time / gpu_time
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"GPU speedup: {speedup:.2f}x")
```

---

## 🧠 Memory Management

### Understanding Tensor Memory
```python
# Check memory usage
tensor = torch.randn(1000, 1000)
print(f"Tensor size: {tensor.size()}")
print(f"Memory usage: {tensor.element_size() * tensor.nelement()} bytes")
print(f"Storage offset: {tensor.storage_offset()}")
print(f"Stride: {tensor.stride()}")

# Memory sharing
a = torch.randn(4, 4)
b = a.view(16)  # Shares memory with a
c = a.clone()   # Creates a copy

print(f"a and b share memory: {a.storage().data_ptr() == b.storage().data_ptr()}")
print(f"a and c share memory: {a.storage().data_ptr() == c.storage().data_ptr()}")
```

### In-place Operations
```python
# Regular operations create new tensors
a = torch.randn(3, 3)
original_id = id(a)
a = a + 1  # Creates new tensor
print(f"ID changed: {id(a) != original_id}")

# In-place operations modify existing tensor
a = torch.randn(3, 3)
original_id = id(a)
a.add_(1)  # In-place addition
print(f"ID unchanged: {id(a) == original_id}")

# Common in-place operations
a.mul_(2)        # In-place multiplication
a.div_(2)        # In-place division
a.clamp_(0, 1)   # In-place clamping
a.fill_(0)       # Fill with value
```

### Memory Optimization Tips
```python
# 1. Use in-place operations when possible
# Bad: Creates intermediate tensors
result = tensor1 + tensor2 * tensor3

# Better: Fewer temporary tensors
temp = tensor2.mul(tensor3)
result = tensor1.add(temp)

# Best: In-place when safe
tensor2.mul_(tensor3)
result = tensor1.add_(tensor2)

# 2. Clear unnecessary variables
del large_tensor  # Explicitly delete
torch.cuda.empty_cache()  # Free GPU memory cache

# 3. Use context managers for temporary operations
with torch.no_grad():  # Disable gradient computation
    result = model(input_tensor)
```

---

## 🎯 Key Questions for Self-Assessment

### Beginner Level Questions:
1. **Q**: What is the main difference between a PyTorch tensor and a NumPy array?
   **A**: PyTorch tensors support GPU acceleration and automatic differentiation, while NumPy arrays are CPU-only and don't have built-in gradient computation.

2. **Q**: How do you create a 3x4 tensor filled with zeros on GPU?
   **A**: `torch.zeros(3, 4, device='cuda')` or `torch.zeros(3, 4).cuda()`

3. **Q**: What does the `.view()` method do?
   **A**: `.view()` reshapes a tensor to a new shape while sharing the same underlying memory, requiring the new shape to be compatible with the original.

### Intermediate Level Questions:
4. **Q**: Explain the broadcasting rules in PyTorch with an example.
   **A**: Broadcasting compares dimensions from right to left. Dimensions are compatible if they're equal, one is 1, or one doesn't exist. Example: (3,1,5) + (1,4,5) → (3,4,5).

5. **Q**: What's the difference between `.view()` and `.reshape()`?
   **A**: `.view()` requires contiguous memory and shares storage, while `.reshape()` can handle non-contiguous tensors by creating a copy if necessary.

6. **Q**: How do in-place operations affect memory usage and gradients?
   **A**: In-place operations modify tensors without creating new memory, improving efficiency, but they can break gradient computation chains in autograd.

### Advanced Level Questions:
7. **Q**: Compare the performance implications of CPU vs GPU tensor operations for different workloads.
   **A**: GPU excels at parallel operations on large tensors but has overhead for small operations and data transfer. CPU is better for small tensors, control flow, and operations requiring frequent CPU-GPU transfers.

8. **Q**: Analyze the memory layout implications of different tensor operations.
   **A**: Operations like `.transpose()` create non-contiguous tensors, affecting memory access patterns. `.contiguous()` can improve performance by reorganizing memory layout for better cache locality.

---

## 🔑 Key Takeaways

1. **Tensor Fundamentals**: PyTorch tensors are multi-dimensional arrays with GPU support and automatic differentiation capabilities.

2. **GPU Acceleration**: Moving computations to GPU can provide significant speedups for large-scale operations, but requires careful memory management.

3. **Broadcasting Efficiency**: Understanding broadcasting rules enables efficient operations between tensors of different shapes without explicit reshaping.

4. **Memory Awareness**: In-place operations and proper memory management are crucial for performance optimization in large-scale applications.

5. **NumPy Compatibility**: PyTorch tensors can seamlessly convert to/from NumPy arrays, enabling integration with existing Python scientific computing workflows.

---

## 📝 Practical Exercises

### Exercise 1: Tensor Creation and Manipulation
Create a function that generates different types of tensors and analyzes their properties:

```python
def analyze_tensor(tensor, name):
    """Analyze tensor properties"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Data type: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Is contiguous: {tensor.is_contiguous()}")
    print(f"  Memory usage: {tensor.nelement() * tensor.element_size()} bytes")

# Test with different tensor types
analyze_tensor(torch.randn(3, 4), "Random Normal")
analyze_tensor(torch.eye(5, dtype=torch.int32), "Identity Matrix")
analyze_tensor(torch.arange(12).view(3, 4), "Reshaped Range")
```

### Exercise 2: Broadcasting Practice
Implement operations that demonstrate broadcasting:

```python
def broadcasting_examples():
    """Demonstrate different broadcasting scenarios"""
    # Scalar + Matrix
    scalar = torch.tensor(5.0)
    matrix = torch.randn(3, 4)
    result1 = matrix + scalar
    
    # Vector + Matrix
    vector = torch.randn(4)
    result2 = matrix + vector  # vector broadcasts to (1, 4) then (3, 4)
    
    # Matrix + Matrix with different shapes
    mat1 = torch.randn(3, 1)
    mat2 = torch.randn(1, 4)
    result3 = mat1 + mat2  # Result shape: (3, 4)
    
    return result1, result2, result3
```

---

**Next**: Continue with Day 1 - Part 3: Autograd Mechanics and Computation Graphs