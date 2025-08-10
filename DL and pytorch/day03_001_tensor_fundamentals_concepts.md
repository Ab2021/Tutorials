# Day 3.1: Tensor Fundamentals and Core Concepts

## Overview
Tensors form the fundamental building blocks of PyTorch and deep learning. Understanding tensors at a deep level—their mathematical foundations, computational properties, and implementation details—is crucial for effective PyTorch programming. This comprehensive module covers tensor concepts from basic mathematical definitions to advanced computational considerations, providing the theoretical foundation necessary for mastering PyTorch development.

## Mathematical Foundations of Tensors

### Tensor Theory and Linear Algebra

**Mathematical Definition of Tensors**
In mathematics, a tensor is a multilinear map from multiple vector spaces to the real numbers. More intuitively for deep learning practitioners, tensors are generalizations of scalars, vectors, and matrices to higher dimensions:

**Tensor Rank (Order) Hierarchy**:
- **Rank 0 (Scalar)**: Single number, no dimensions
  - Mathematical representation: Just a number like 5 or 3.14
  - PyTorch shape: `torch.Size([])`
  - Example: Loss values, learning rates, single predictions

- **Rank 1 (Vector)**: One-dimensional array of numbers
  - Mathematical representation: **v** = [v₁, v₂, ..., vₙ]
  - PyTorch shape: `torch.Size([n])`
  - Example: Feature vectors, word embeddings, bias terms

- **Rank 2 (Matrix)**: Two-dimensional array of numbers
  - Mathematical representation: **M** = [[m₁₁, m₁₂], [m₂₁, m₂₂]]
  - PyTorch shape: `torch.Size([m, n])`
  - Example: Weight matrices, images (grayscale), attention maps

- **Rank 3 (3D Tensor)**: Three-dimensional array
  - Mathematical representation: **T**[i,j,k]
  - PyTorch shape: `torch.Size([d₁, d₂, d₃])`
  - Example: RGB images, sequence embeddings, feature maps

- **Rank 4+ (Higher-order Tensors)**: Four or more dimensions
  - Mathematical representation: **T**[i,j,k,l,...]
  - PyTorch shape: `torch.Size([d₁, d₂, d₃, d₄, ...])`
  - Example: Mini-batches of images, video data, 4D convolution kernels

**Tensor Space and Vector Space Theory**
Tensors exist in tensor spaces, which are extensions of vector spaces:

**Vector Space Properties**:
- **Closure under Addition**: If **u** and **v** are tensors, then **u** + **v** is also a tensor
- **Closure under Scalar Multiplication**: If **u** is a tensor and c is a scalar, then c**u** is a tensor
- **Associativity**: (**u** + **v**) + **w** = **u** + (**v** + **w**)
- **Commutativity**: **u** + **v** = **v** + **u**
- **Zero Element**: There exists a zero tensor **0** such that **u** + **0** = **u**
- **Additive Inverse**: For every tensor **u**, there exists **-u** such that **u** + (-**u**) = **0**

**Tensor Product Spaces**:
The space of rank-n tensors over vector space V is the n-fold tensor product V⊗V⊗...⊗V. This mathematical foundation explains why tensor operations follow specific algebraic rules and why broadcasting works the way it does in PyTorch.

### Coordinate Systems and Tensor Components

**Index Notation and Einstein Summation**
Tensor operations are often expressed using index notation, which provides a compact way to describe complex operations:

**Einstein Summation Convention**:
When an index appears twice in a tensor expression, it implies summation over that index:
- **Explicit**: Σᵢ aᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
- **Einstein notation**: aᵢbᵢ (summation over i is implicit)

**Common Tensor Operations in Index Notation**:
- **Matrix multiplication**: (AB)ᵢⱼ = Aᵢₖ Bₖⱼ
- **Trace**: tr(A) = Aᵢᵢ
- **Frobenius norm**: ||A||_F = √(AᵢⱼAᵢⱼ)
- **Tensor contraction**: Cᵢⱼ = Aᵢₖₗ Bₖₗⱼ

**Tensor Coordinate Transformations**:
Understanding how tensor components transform under coordinate changes is crucial for understanding invariance properties:

**Contravariant vs Covariant Tensors**:
- **Contravariant**: Components transform opposite to coordinate transformation
- **Covariant**: Components transform the same as coordinate transformation
- **Mixed tensors**: Have both contravariant and covariant indices

This mathematical foundation explains why certain operations preserve specific properties and why some tensor operations are more natural than others.

## PyTorch Tensor Architecture

### Internal Data Structure

**Storage and Memory Layout**
PyTorch tensors are built on top of a sophisticated storage system that manages memory efficiently while providing intuitive interfaces:

**Storage Abstraction**:
```python
import torch

# Create tensor to examine internal structure
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Access underlying storage
print(f"Storage: {x.storage()}")
print(f"Storage size: {x.storage().size()}")
print(f"Storage type: {type(x.storage())}")

# Multiple tensors can share storage
y = x.view(3, 2)  # Different view of same data
print(f"Same storage: {x.storage().data_ptr() == y.storage().data_ptr()}")
```

**Storage Properties**:
- **Contiguous Memory**: Storage is always a 1D array of elements
- **Reference Counting**: Automatic memory management through reference counting
- **Type Homogeneity**: All elements in storage have the same data type
- **Device Location**: Storage is tied to a specific device (CPU, GPU, etc.)

**Tensor Metadata Structure**:
Each PyTorch tensor contains metadata that describes how to interpret the underlying storage:

**Essential Metadata Components**:
- **Shape/Size**: Dimensions of the tensor
- **Strides**: How to navigate through storage to find elements
- **Storage Offset**: Starting position in the storage
- **Data Type**: Element type (float32, int64, etc.)
- **Device**: Where the tensor resides (CPU, CUDA, etc.)
- **Gradient Information**: For automatic differentiation

**Stride System Deep Dive**:
Strides determine how tensor indices map to storage positions:

```python
# Understanding strides
x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Strides: {x.stride()}")

# Stride interpretation:
# To access element [i, j, k]:
# storage_index = i * stride[0] + j * stride[1] + k * stride[2]
# For shape (3, 4, 5): strides are typically (20, 5, 1)
# Element [1, 2, 3] is at position: 1*20 + 2*5 + 3*1 = 33

# Verify stride calculation
element_15_storage_idx = 1 * x.stride(0) + 2 * x.stride(1) + 3 * x.stride(2)
print(f"Calculated storage index: {element_15_storage_idx}")
```

**Memory Layout Patterns**:
- **Row-major (C-style)**: Default in PyTorch, last dimension varies fastest
- **Column-major (Fortran-style)**: First dimension varies fastest
- **Custom strides**: Allow for complex memory access patterns

### Data Type System

**Comprehensive Data Type Hierarchy**
PyTorch supports a rich variety of data types optimized for different use cases:

**Floating Point Types**:
- **torch.float64 (double)**: 64-bit floating point, highest precision
  - Range: ±1.7e±308, Precision: ~15-17 decimal digits
  - Use cases: Scientific computing, high-precision requirements
  - Memory: 8 bytes per element

- **torch.float32 (float)**: 32-bit floating point, standard precision
  - Range: ±3.4e±38, Precision: ~6-7 decimal digits
  - Use cases: Most deep learning applications, good balance of speed and precision
  - Memory: 4 bytes per element

- **torch.float16 (half)**: 16-bit floating point, reduced precision
  - Range: ±6.5e±4, Precision: ~3-4 decimal digits
  - Use cases: Memory-constrained environments, mixed precision training
  - Memory: 2 bytes per element

- **torch.bfloat16**: 16-bit "brain" floating point (Google's format)
  - Same range as float32, reduced precision
  - Use cases: TPU optimization, some GPU architectures
  - Better for training than float16 due to wider range

**Integer Types**:
- **torch.int64 (long)**: 64-bit signed integer
  - Range: -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
  - Use cases: Indices, large integer computations
  - Memory: 8 bytes per element

- **torch.int32 (int)**: 32-bit signed integer
  - Range: -2,147,483,648 to 2,147,483,647
  - Use cases: Most indexing operations, labels
  - Memory: 4 bytes per element

- **torch.int16 (short)**: 16-bit signed integer
- **torch.int8**: 8-bit signed integer
- **torch.uint8**: 8-bit unsigned integer (0-255)
  - Use cases: Image pixel values, quantized models

**Boolean and Complex Types**:
- **torch.bool**: Boolean values (True/False)
  - Memory: 1 byte per element
  - Use cases: Masks, conditional operations

- **torch.complex64**: 64-bit complex numbers (32-bit real + 32-bit imaginary)
- **torch.complex128**: 128-bit complex numbers (64-bit real + 64-bit imaginary)
  - Use cases: Signal processing, Fourier transforms

**Type Conversion and Precision Considerations**:
```python
# Type conversion examples
x_float64 = torch.randn(1000, dtype=torch.float64)
x_float32 = x_float64.float()  # Convert to float32
x_float16 = x_float32.half()   # Convert to float16

# Precision loss demonstration
original_value = 1.23456789012345
tensor_f64 = torch.tensor(original_value, dtype=torch.float64)
tensor_f32 = torch.tensor(original_value, dtype=torch.float32)
tensor_f16 = torch.tensor(original_value, dtype=torch.float16)

print(f"Original: {original_value}")
print(f"Float64:  {tensor_f64.item()}")
print(f"Float32:  {tensor_f32.item()}")
print(f"Float16:  {tensor_f16.item()}")

# Automatic type promotion
a = torch.tensor([1, 2, 3], dtype=torch.int32)
b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
c = a + b  # Result will be float32
print(f"Promoted type: {c.dtype}")
```

### Device Management and Tensor Location

**Device Abstraction Architecture**
PyTorch provides a unified device abstraction that allows tensors to reside on different computational devices:

**Device Types**:
- **CPU**: Central Processing Unit
  - Characteristics: Large memory capacity, lower computational throughput
  - Use cases: Data preprocessing, small models, debugging
  - Memory: System RAM (typically GBs to TBs)

- **CUDA**: NVIDIA GPU devices
  - Characteristics: High parallel throughput, limited memory
  - Use cases: Deep learning training and inference
  - Memory: GPU VRAM (typically GBs)
  - Numbering: cuda:0, cuda:1, etc. for multiple GPUs

- **MPS**: Apple Metal Performance Shaders (Apple Silicon)
  - Characteristics: Unified memory architecture
  - Use cases: Apple Silicon Mac optimization

**Device Properties and Capabilities**:
```python
# Device detection and properties
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")

# Device object creation
cpu_device = torch.device('cpu')
if torch.cuda.is_available():
    gpu_device = torch.device('cuda:0')
    
    # Check current device
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
```

**Tensor Device Operations**:
```python
# Creating tensors on specific devices
cpu_tensor = torch.randn(3, 4, device='cpu')
if torch.cuda.is_available():
    gpu_tensor = torch.randn(3, 4, device='cuda:0')
    
    # Moving tensors between devices
    cpu_to_gpu = cpu_tensor.to('cuda:0')  # Copy to GPU
    gpu_to_cpu = gpu_tensor.cpu()         # Copy to CPU
    
    # In-place device transfer
    cpu_tensor.cuda()  # Move to default CUDA device
    
    # Check tensor device
    print(f"Tensor device: {gpu_tensor.device}")
    print(f"Is CUDA tensor: {gpu_tensor.is_cuda}")

# Device context management
with torch.cuda.device(1):  # Temporarily use GPU 1
    temp_tensor = torch.randn(2, 3)  # Created on GPU 1
```

## Tensor Creation Methods

### Factory Functions and Initialization Patterns

**Basic Tensor Creation**
PyTorch provides numerous factory functions for creating tensors with different initialization patterns:

**Constant Value Initialization**:
```python
# Zeros and ones
zeros_tensor = torch.zeros(3, 4, dtype=torch.float32)
ones_tensor = torch.ones(2, 3, 5, dtype=torch.int64)
empty_tensor = torch.empty(2, 4)  # Uninitialized (faster)

# Full with specific value
full_tensor = torch.full((3, 3), fill_value=7.5)
full_like = torch.full_like(zeros_tensor, fill_value=2.0)

# Identity matrices
identity = torch.eye(4)  # 4x4 identity matrix
identity_nonsquare = torch.eye(3, 4)  # 3x4 matrix with diagonal ones
```

**Sequential and Range Creation**:
```python
# Arithmetic sequences
arange_int = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
arange_float = torch.arange(0.0, 1.0, 0.1)

# Linear spacing
linspace_tensor = torch.linspace(0, 1, steps=11)  # 11 points from 0 to 1
logspace_tensor = torch.logspace(0, 2, steps=3)   # [10^0, 10^1, 10^2]

# Geometric spacing
geometric_sequence = torch.logspace(0, 3, 4, base=2)  # Powers of 2
```

**Random Initialization Strategies**:
Random initialization is crucial for deep learning, with different distributions serving different purposes:

**Uniform Distribution**:
```python
# Uniform random in [0, 1)
uniform_01 = torch.rand(3, 4)

# Uniform random in [low, high)
uniform_range = torch.uniform(2, 5, size=(3, 4))  # [2, 5)

# Random integers
randint_tensor = torch.randint(low=0, high=10, size=(2, 3))
randint_like = torch.randint_like(zeros_tensor, low=0, high=100)
```

**Normal (Gaussian) Distribution**:
```python
# Standard normal (μ=0, σ=1)
standard_normal = torch.randn(3, 4)

# Normal with specific parameters
normal_custom = torch.normal(mean=2.0, std=1.5, size=(3, 4))

# Normal with tensor parameters
mean_tensor = torch.zeros(3, 4)
std_tensor = torch.ones(3, 4) * 0.5
normal_tensor_params = torch.normal(mean_tensor, std_tensor)
```

**Advanced Statistical Distributions**:
```python
# Bernoulli distribution
bernoulli_tensor = torch.bernoulli(torch.full((3, 4), 0.3))

# Exponential distribution
exponential_tensor = torch.exponential(torch.ones(2, 3))

# Cauchy distribution
cauchy_tensor = torch.cauchy(torch.zeros(2, 3), torch.ones(2, 3))

# Log-normal distribution
lognormal_tensor = torch.log_normal(torch.zeros(2, 3), torch.ones(2, 3))
```

### Advanced Initialization Techniques

**Xavier/Glorot Initialization**
Mathematical foundation for weight initialization in neural networks:

**Xavier Uniform**:
Weights drawn from uniform distribution with bounds: ±√(6/(fan_in + fan_out))

```python
def xavier_uniform_init(tensor, gain=1.0):
    """Xavier uniform initialization"""
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return tensor.uniform_(-bound, bound)

# Example usage
weight_matrix = torch.empty(100, 50)
xavier_uniform_init(weight_matrix)
```

**Xavier Normal**:
Weights drawn from normal distribution with std: √(2/(fan_in + fan_out))

```python
def xavier_normal_init(tensor, gain=1.0):
    """Xavier normal initialization"""
    fan_in, fan_out = calculate_fan_in_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)
```

**Kaiming/He Initialization**
Designed specifically for ReLU activations:

**Kaiming Uniform**:
```python
def kaiming_uniform_init(tensor, mode='fan_in', nonlinearity='relu'):
    """Kaiming uniform initialization"""
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity)
    bound = gain * math.sqrt(3.0 / fan)
    return tensor.uniform_(-bound, bound)
```

**Kaiming Normal**:
```python
def kaiming_normal_init(tensor, mode='fan_in', nonlinearity='relu'):
    """Kaiming normal initialization"""
    fan = calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    return tensor.normal_(0, std)
```

**Orthogonal Initialization**:
Creates orthogonal matrices, useful for RNN weights:

```python
def orthogonal_init(tensor, gain=1.0):
    """Orthogonal initialization"""
    if tensor.dim() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    
    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)
    
    if rows < cols:
        flattened.t_()
    
    # QR decomposition
    Q, R = torch.qr(flattened)
    
    # Make Q uniform
    d = R.diag()
    Q *= d.sign()
    
    if rows < cols:
        Q.t_()
    
    tensor.view_as(Q).copy_(Q)
    tensor.mul_(gain)
    return tensor
```

### Tensor Construction from Data

**From Python Data Structures**:
```python
# From lists
list_1d = [1, 2, 3, 4]
tensor_from_list = torch.tensor(list_1d)

list_2d = [[1, 2], [3, 4], [5, 6]]
tensor_from_list_2d = torch.tensor(list_2d)

# From nested structures
nested_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
tensor_3d = torch.tensor(nested_data)

# From iterables
import itertools
tensor_from_range = torch.tensor(list(range(10)))
```

**From NumPy Arrays**:
```python
import numpy as np

# NumPy to PyTorch (shares memory by default)
numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = torch.from_numpy(numpy_array)

# Verify memory sharing
numpy_array[0, 0] = 999
print(f"Tensor value changed: {tensor_from_numpy[0, 0]}")  # Should be 999

# Copy instead of sharing memory
tensor_copy = torch.tensor(numpy_array)  # Creates copy
numpy_array[0, 0] = 777
print(f"Tensor value unchanged: {tensor_copy[0, 0]}")  # Should still be 999
```

**From Other Tensors**:
```python
original_tensor = torch.randn(3, 4)

# Create similar tensors
zeros_like = torch.zeros_like(original_tensor)
ones_like = torch.ones_like(original_tensor)
randn_like = torch.randn_like(original_tensor)

# Clone tensor (copy data and grad_fn)
cloned_tensor = original_tensor.clone()

# Detach from computation graph
detached_tensor = original_tensor.detach()

# New tensor with same properties but different data
new_tensor = original_tensor.new_zeros(5, 6)
new_tensor_ones = original_tensor.new_ones(2, 3)
```

## Tensor Properties and Attributes

### Shape and Dimensionality Analysis

**Shape Inspection and Manipulation**:
```python
tensor = torch.randn(2, 3, 4, 5)

# Basic shape properties
print(f"Shape: {tensor.shape}")          # torch.Size([2, 3, 4, 5])
print(f"Size: {tensor.size()}")          # Same as shape
print(f"Dimensions: {tensor.dim()}")     # 4
print(f"Number of elements: {tensor.numel()}")  # 120

# Access individual dimensions
print(f"Batch size: {tensor.size(0)}")   # 2
print(f"Height: {tensor.size(2)}")       # 4

# Check if tensor is empty
empty_tensor = torch.empty(0, 3)
print(f"Is empty: {empty_tensor.numel() == 0}")  # True
```

**Dimensionality Properties**:
```python
# Check dimensionality
scalar = torch.tensor(5.0)
vector = torch.randn(10)
matrix = torch.randn(3, 4)
tensor_3d = torch.randn(2, 3, 4)

print(f"Scalar dimensions: {scalar.dim()}")    # 0
print(f"Vector dimensions: {vector.dim()}")    # 1
print(f"Matrix dimensions: {matrix.dim()}")    # 2
print(f"3D tensor dimensions: {tensor_3d.dim()}")  # 3

# Check specific dimensional properties
print(f"Is scalar: {scalar.dim() == 0}")
print(f"Is vector: {vector.dim() == 1}")
print(f"Is matrix: {matrix.dim() == 2}")
```

### Memory Layout and Storage Properties

**Contiguity Analysis**:
```python
# Create tensor and examine contiguity
original = torch.randn(3, 4, 5)
print(f"Original contiguous: {original.is_contiguous()}")  # True

# Transpose changes contiguity
transposed = original.transpose(1, 2)
print(f"Transposed contiguous: {transposed.is_contiguous()}")  # False

# View operations require contiguous tensors
try:
    viewed = transposed.view(-1)  # May fail
except RuntimeError as e:
    print(f"View error: {e}")
    # Make contiguous first
    contiguous_transposed = transposed.contiguous()
    viewed = contiguous_transposed.view(-1)  # Works

# Check memory layout
print(f"Original strides: {original.stride()}")
print(f"Transposed strides: {transposed.stride()}")
print(f"Contiguous strides: {contiguous_transposed.stride()}")
```

**Storage Sharing Analysis**:
```python
# Examine storage sharing
original = torch.randn(6, 4)
view_tensor = original.view(3, 8)
slice_tensor = original[1:4, :]

# Check storage sharing
print(f"Original storage ptr: {original.storage().data_ptr()}")
print(f"View storage ptr: {view_tensor.storage().data_ptr()}")
print(f"Slice storage ptr: {slice_tensor.storage().data_ptr()}")

# All should share the same storage
same_storage = (original.storage().data_ptr() == 
               view_tensor.storage().data_ptr() == 
               slice_tensor.storage().data_ptr())
print(f"Share same storage: {same_storage}")

# Check storage offset
print(f"Original offset: {original.storage_offset()}")
print(f"View offset: {view_tensor.storage_offset()}")
print(f"Slice offset: {slice_tensor.storage_offset()}")
```

### Data Type Properties and Conversion

**Type Inspection**:
```python
float_tensor = torch.randn(3, 4)
int_tensor = torch.randint(0, 10, (3, 4))
bool_tensor = torch.tensor([True, False, True])

# Check data types
print(f"Float tensor dtype: {float_tensor.dtype}")     # torch.float32
print(f"Int tensor dtype: {int_tensor.dtype}")         # torch.int64
print(f"Bool tensor dtype: {bool_tensor.dtype}")       # torch.bool

# Type checking methods
print(f"Is floating point: {float_tensor.is_floating_point()}")  # True
print(f"Is complex: {float_tensor.is_complex()}")                # False
print(f"Is signed: {int_tensor.is_signed()}")                    # True

# Default types
print(f"Default float type: {torch.get_default_dtype()}")
print(f"Default int type: {torch.tensor(1).dtype}")
```

**Advanced Type Conversion**:
```python
# Comprehensive type conversion
original = torch.randn(2, 3) * 100

# Convert to different precisions
double_precision = original.double()    # float64
single_precision = original.float()     # float32
half_precision = original.half()        # float16

# Convert to integers (truncation)
int_version = original.int()            # int32
long_version = original.long()          # int64

# Convert to boolean
bool_version = original.bool()          # Non-zero -> True, zero -> False

# Type conversion with specific parameters
converted = original.to(dtype=torch.float16, device='cpu')

# Check precision loss
print(f"Original: {original[0, 0].item():.6f}")
print(f"Half precision: {half_precision[0, 0].item():.6f}")
print(f"Precision difference: {abs(original[0, 0] - half_precision[0, 0]).item():.6f}")
```

## Key Questions for Review

### Mathematical Foundations
1. **Tensor Rank Understanding**: What is the mathematical difference between a rank-2 tensor (matrix) and a rank-3 tensor, and how does this affect operations?

2. **Einstein Summation**: How does Einstein summation notation relate to common PyTorch operations like matrix multiplication and batch operations?

3. **Vector Space Properties**: Why do tensors form a vector space, and what practical implications does this have for neural network operations?

### PyTorch Architecture
4. **Storage vs Tensor**: What is the relationship between PyTorch's storage system and tensor objects, and why does this separation exist?

5. **Stride System**: How do strides enable memory-efficient operations like transpose and slicing without data copying?

6. **Device Management**: What are the trade-offs between keeping tensors on CPU vs GPU, and when should data transfer occur?

### Data Types and Precision
7. **Numerical Precision**: How does the choice of data type (float32 vs float16 vs float64) affect both computational speed and numerical accuracy?

8. **Type Promotion**: What are PyTorch's rules for automatic type promotion during operations, and why are these rules important?

9. **Mixed Precision**: What are the benefits and challenges of using mixed precision training with float16 and float32?

### Advanced Concepts
10. **Memory Layout**: Why does memory contiguity matter for performance, and when do operations create non-contiguous tensors?

11. **Initialization Strategies**: How do different initialization methods (Xavier, Kaiming, orthogonal) affect neural network training dynamics?

12. **Tensor Sharing**: When do PyTorch operations create views vs copies, and why is this distinction important for memory management?

## Advanced Tensor Concepts

### Tensor Broadcasting Fundamentals

**Mathematical Foundation of Broadcasting**
Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions according to specific rules:

**Broadcasting Rules**:
1. **Alignment**: Dimensions are aligned from the rightmost (trailing) dimension
2. **Compatibility**: Dimensions are compatible if they are equal, or one of them is 1
3. **Expansion**: Dimensions of size 1 are stretched to match the larger dimension
4. **Addition**: Missing dimensions are assumed to be 1

**Broadcasting Examples**:
```python
# Rule demonstration
a = torch.randn(3, 1, 5)  # Shape: (3, 1, 5)
b = torch.randn(2, 4, 1)  # Shape: (2, 4, 1)

# Broadcasting alignment:
# a: (3, 1, 5)
# b: (2, 4, 1)
# Result will be: (2, 3, 4, 5) - dimensions are expanded

# This operation will fail due to incompatible dimensions
try:
    c = a + b
except RuntimeError as e:
    print(f"Broadcasting error: {e}")

# Successful broadcasting example
a = torch.randn(1, 3, 1)  # Shape: (1, 3, 1)
b = torch.randn(4, 1, 5)  # Shape: (4, 1, 5)
c = a + b                 # Result: (4, 3, 5)

print(f"Result shape: {c.shape}")
```

**Memory Implications of Broadcasting**:
Broadcasting creates virtual expansions without actually copying data, making operations memory-efficient:

```python
# Memory-efficient broadcasting
large_tensor = torch.randn(1000, 1000)
small_tensor = torch.randn(1000, 1)

# This doesn't create intermediate 1000x1000 copies
result = large_tensor + small_tensor

# Verify memory efficiency
print(f"Large tensor memory: {large_tensor.element_size() * large_tensor.numel()} bytes")
print(f"Small tensor memory: {small_tensor.element_size() * small_tensor.numel()} bytes")
print(f"Result memory: {result.element_size() * result.numel()} bytes")
```

### View System and Memory Sharing

**View vs Copy Semantics**:
Understanding when operations create views (shared memory) vs copies (separate memory) is crucial for both performance and correctness:

**Operations that Create Views**:
```python
original = torch.randn(4, 6)

# View operations (share memory)
reshaped = original.view(2, 12)
transposed = original.t()
sliced = original[1:3, :]
squeezed = original.unsqueeze(0).squeeze(0)

# Verify memory sharing
def shares_memory(a, b):
    return a.storage().data_ptr() == b.storage().data_ptr()

print(f"View shares memory: {shares_memory(original, reshaped)}")      # True
print(f"Transpose shares memory: {shares_memory(original, transposed)}")  # True
print(f"Slice shares memory: {shares_memory(original, sliced)}")       # True
```

**Operations that Create Copies**:
```python
# Copy operations (separate memory)
copied = original.clone()
contiguous = transposed.contiguous()
converted = original.float()  # If already float, this still copies

print(f"Clone shares memory: {shares_memory(original, copied)}")         # False
print(f"Contiguous shares memory: {shares_memory(transposed, contiguous)}")  # False
```

**View Safety and Constraints**:
```python
# View constraints
original = torch.randn(4, 6, 8)

# Valid view: compatible with storage layout
valid_view = original.view(2, 12, 8)  # Works

# Invalid view: incompatible with transposed storage
transposed = original.transpose(1, 2)
try:
    invalid_view = transposed.view(2, 12, 8)  # May fail
except RuntimeError as e:
    print(f"View constraint violation: {e}")
    # Solution: make contiguous first
    contiguous_first = transposed.contiguous()
    valid_view = contiguous_first.view(2, 12, 8)  # Works
```

### Tensor Indexing and Advanced Selection

**Advanced Indexing Patterns**:
PyTorch supports sophisticated indexing patterns that enable complex data selection:

**Boolean Indexing**:
```python
data = torch.randn(5, 4)
threshold = 0.5

# Boolean mask creation
mask = data > threshold
positive_values = data[mask]

print(f"Original shape: {data.shape}")
print(f"Selected values shape: {positive_values.shape}")
print(f"Number of positive values: {positive_values.numel()}")

# Advanced boolean indexing
row_mask = (data > 0).all(dim=1)  # Rows where all values are positive
selected_rows = data[row_mask]
```

**Fancy Indexing**:
```python
# Advanced index selection
data = torch.randn(10, 5)
row_indices = torch.tensor([0, 2, 4, 6])
col_indices = torch.tensor([1, 3, 0, 2])

# Select specific elements
selected_elements = data[row_indices, col_indices]

# Gather operation
gathered = torch.gather(data, dim=1, index=col_indices.unsqueeze(1).expand(-1, data.size(1)))
```

**Masked Assignment**:
```python
# Conditional assignment
data = torch.randn(3, 4)
mask = data < 0

# Replace negative values with zeros
data[mask] = 0

# More complex conditional assignment
data[data > 1] = 1.0  # Clamp to maximum value
data[data < -1] = -1.0  # Clamp to minimum value

# Masked fill operation
filled_data = data.masked_fill(mask, value=999)
```

## Performance Considerations

### Memory Access Patterns

**Cache-Friendly Access Patterns**:
Understanding memory access patterns is crucial for performance:

```python
import time

# Row-major vs column-major access
large_matrix = torch.randn(1000, 1000)

# Row-major access (cache-friendly)
start_time = time.time()
row_sum = 0
for i in range(large_matrix.size(0)):
    row_sum += large_matrix[i, :].sum()
row_major_time = time.time() - start_time

# Column-major access (cache-unfriendly)
start_time = time.time()
col_sum = 0
for j in range(large_matrix.size(1)):
    col_sum += large_matrix[:, j].sum()
column_major_time = time.time() - start_time

print(f"Row-major access time: {row_major_time:.4f}s")
print(f"Column-major access time: {column_major_time:.4f}s")
print(f"Performance ratio: {column_major_time / row_major_time:.2f}x")
```

**Contiguous Memory Benefits**:
```python
# Contiguous vs non-contiguous performance
data = torch.randn(1000, 1000)
non_contiguous = data.t()  # Transpose creates non-contiguous tensor

# Operation on contiguous tensor
start_time = time.time()
contiguous_result = data.sum()
contiguous_time = time.time() - start_time

# Operation on non-contiguous tensor
start_time = time.time()
non_contiguous_result = non_contiguous.sum()
non_contiguous_time = time.time() - start_time

print(f"Contiguous operation time: {contiguous_time:.6f}s")
print(f"Non-contiguous operation time: {non_contiguous_time:.6f}s")

# Make non-contiguous tensor contiguous for fair comparison
contiguous_version = non_contiguous.contiguous()
start_time = time.time()
contiguous_version_result = contiguous_version.sum()
contiguous_version_time = time.time() - start_time

print(f"Made-contiguous operation time: {contiguous_version_time:.6f}s")
```

### Memory Management Best Practices

**Efficient Memory Usage**:
```python
# Memory-efficient tensor operations
def memory_efficient_processing(large_data):
    """Process large tensors efficiently"""
    
    # Use in-place operations when possible
    large_data.add_(1.0)  # In-place addition
    large_data.mul_(0.5)  # In-place multiplication
    
    # Avoid intermediate tensors
    # Instead of: result = (large_data + 1) * 0.5
    # Use: large_data += 1; large_data *= 0.5
    
    # Use views instead of copies
    reshaped = large_data.view(-1)  # View, no memory copy
    
    # Clear references to free memory
    del large_data
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return reshaped

# Demonstrate memory tracking
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    
    large_tensor = torch.randn(10000, 10000, device='cuda')
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak memory usage: {peak_memory / 1e9:.2f} GB")
    
    # Process efficiently
    processed = memory_efficient_processing(large_tensor)
    final_memory = torch.cuda.memory_allocated()
    print(f"Final memory usage: {final_memory / 1e9:.2f} GB")
```

## Conclusion

Tensor fundamentals form the cornerstone of effective PyTorch programming and deep learning development. This comprehensive exploration of tensor concepts—from mathematical foundations to practical implementation details—provides the theoretical and practical knowledge necessary for advanced PyTorch development.

**Key Takeaways**:

**Mathematical Understanding**: Tensors are mathematical objects with precise algebraic properties that enable powerful operations and transformations essential for neural network computations.

**Implementation Mastery**: PyTorch's tensor system provides a sophisticated balance between performance and usability, with careful attention to memory management, device abstraction, and computational efficiency.

**Performance Optimization**: Understanding internal tensor mechanics—storage, strides, contiguity, and device management—enables writing efficient code that scales to large-scale deep learning applications.

**Design Patterns**: Effective tensor usage patterns, initialization strategies, and memory management techniques are essential for building robust and efficient deep learning systems.

The foundational knowledge established in this module provides the basis for understanding more advanced PyTorch concepts, including automatic differentiation, neural network modules, and optimization techniques. Mastery of these tensor fundamentals is essential for any serious deep learning practitioner using PyTorch.