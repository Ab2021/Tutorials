# Day 3.1: Tensor Fundamentals and Creation Methods

## Course: Comprehensive Deep Learning with PyTorch - 45-Day Masterclass
### Day 3, Part 1: Understanding PyTorch Tensors from Ground Up

---

## Overview

Tensors are the fundamental building blocks of PyTorch and deep learning. This module provides comprehensive coverage of tensor creation, properties, data types, and basic operations. Understanding tensors thoroughly is crucial for effective PyTorch development, as every computation in neural networks involves tensor operations.

## Learning Objectives

By the end of this module, you will:
- Master all tensor creation methods and their use cases
- Understand tensor properties, data types, and memory layout
- Perform basic tensor operations and indexing with confidence
- Implement tensor reshaping and dimension manipulation
- Apply tensor creation patterns for different deep learning scenarios

---

## 1. Understanding Tensors

### 1.1 Mathematical Foundation

#### What is a Tensor?

**Mathematical Definition:**
A tensor is a multi-dimensional array that generalizes scalars, vectors, and matrices to higher dimensions:

**Tensor Hierarchy:**
- **Rank 0 (Scalar):** Single number
- **Rank 1 (Vector):** 1D array of numbers
- **Rank 2 (Matrix):** 2D array of numbers  
- **Rank 3 (3D Tensor):** 3D array of numbers
- **Rank N (N-D Tensor):** N-dimensional array of numbers

**Deep Learning Context:**
```python
import torch
import numpy as np

# Examples of different tensor ranks
scalar = torch.tensor(3.14159)           # Rank 0: single value
vector = torch.tensor([1, 2, 3, 4])     # Rank 1: [4] shape
matrix = torch.tensor([[1, 2], [3, 4]]) # Rank 2: [2, 2] shape
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # Rank 3: [2, 2, 2]

print(f"Scalar: {scalar}, shape: {scalar.shape}, ndim: {scalar.ndim}")
print(f"Vector: {vector}, shape: {vector.shape}, ndim: {vector.ndim}")
print(f"Matrix: {matrix}, shape: {matrix.shape}, ndim: {matrix.ndim}")
print(f"3D Tensor: shape: {tensor_3d.shape}, ndim: {tensor_3d.ndim}")
```

**Common Deep Learning Tensor Shapes:**
```python
# Common tensor shapes in deep learning
batch_size, channels, height, width = 32, 3, 224, 224
sequence_length, vocab_size = 512, 50000
hidden_size, num_layers = 768, 12

# Image batch: [batch_size, channels, height, width]
image_batch = torch.randn(batch_size, channels, height, width)
print(f"Image batch shape: {image_batch.shape}")

# Text sequence: [batch_size, sequence_length]
text_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
print(f"Text sequence shape: {text_sequence.shape}")

# Transformer hidden states: [batch_size, sequence_length, hidden_size]
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
print(f"Hidden states shape: {hidden_states.shape}")

# Model weights: [output_features, input_features]
linear_weights = torch.randn(hidden_size, hidden_size)
print(f"Linear layer weights shape: {linear_weights.shape}")
```

#### Tensor Properties Deep Dive

**Essential Tensor Properties:**
```python
def analyze_tensor_properties(tensor, name="tensor"):
    """Comprehensive tensor property analysis"""
    print(f"\n=== {name.upper()} ANALYSIS ===")
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Size: {tensor.size()}")  # Alternative to .shape
    print(f"Number of dimensions: {tensor.ndim}")
    print(f"Number of elements: {tensor.numel()}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires gradient: {tensor.requires_grad}")
    print(f"Memory layout: {tensor.layout}")
    print(f"Is contiguous: {tensor.is_contiguous()}")
    print(f"Element size (bytes): {tensor.element_size()}")
    print(f"Storage offset: {tensor.storage_offset()}")
    print(f"Stride: {tensor.stride()}")

# Examples
tensor_2d = torch.randn(3, 4, requires_grad=True)
analyze_tensor_properties(tensor_2d, "2D tensor")

tensor_3d = torch.randn(2, 3, 4, dtype=torch.float16)
analyze_tensor_properties(tensor_3d, "3D tensor")
```

### 1.2 Data Types and Precision

#### Complete Data Type Reference

**Floating Point Types:**
```python
# Floating point tensor types
float_types = {
    'torch.float64': torch.float64,  # Double precision (64-bit)
    'torch.double': torch.double,    # Alias for float64
    'torch.float32': torch.float32,  # Single precision (32-bit) - DEFAULT
    'torch.float': torch.float,      # Alias for float32
    'torch.float16': torch.float16,  # Half precision (16-bit)
    'torch.half': torch.half,        # Alias for float16
    'torch.bfloat16': torch.bfloat16 # Brain floating point (16-bit)
}

# Demonstrate different floating point precisions
for name, dtype in float_types.items():
    tensor = torch.tensor([3.141592653589793], dtype=dtype)
    print(f"{name:15}: {tensor.item():.10f}, size: {tensor.element_size()} bytes")

# Precision comparison
original_value = 3.141592653589793
float64_tensor = torch.tensor(original_value, dtype=torch.float64)
float32_tensor = torch.tensor(original_value, dtype=torch.float32)
float16_tensor = torch.tensor(original_value, dtype=torch.float16)

print(f"\nOriginal: {original_value:.15f}")
print(f"Float64:  {float64_tensor.item():.15f}")
print(f"Float32:  {float32_tensor.item():.15f}")
print(f"Float16:  {float16_tensor.item():.15f}")
```

**Integer Types:**
```python
# Integer tensor types
int_types = {
    'torch.int64': torch.int64,    # 64-bit signed integer - DEFAULT
    'torch.long': torch.long,      # Alias for int64
    'torch.int32': torch.int32,    # 32-bit signed integer
    'torch.int': torch.int,        # Alias for int32
    'torch.int16': torch.int16,    # 16-bit signed integer
    'torch.short': torch.short,    # Alias for int16
    'torch.int8': torch.int8,      # 8-bit signed integer
    'torch.uint8': torch.uint8,    # 8-bit unsigned integer
}

# Demonstrate integer ranges
for name, dtype in int_types.items():
    info = torch.iinfo(dtype) if hasattr(torch, 'iinfo') else None
    if info:
        print(f"{name:15}: min={info.min:>12}, max={info.max:>12}, "
              f"size={torch.tensor(1, dtype=dtype).element_size()} bytes")
```

**Boolean and Complex Types:**
```python
# Boolean tensors
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(f"Boolean tensor: {bool_tensor}, dtype: {bool_tensor.dtype}")

# Complex number tensors
complex64_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
complex128_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex128)

print(f"Complex64:  {complex64_tensor}, dtype: {complex64_tensor.dtype}")
print(f"Complex128: {complex128_tensor}, dtype: {complex128_tensor.dtype}")

# Complex tensor operations
complex_ops = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(f"Real part: {complex_ops.real}")
print(f"Imaginary part: {complex_ops.imag}")
print(f"Magnitude: {complex_ops.abs()}")
print(f"Angle: {complex_ops.angle()}")
```

#### Data Type Conversion and Casting

**Type Conversion Methods:**
```python
class TypeConverter:
    """Comprehensive tensor type conversion utilities"""
    
    def __init__(self):
        self.conversion_methods = [
            ('to()', 'Universal conversion method'),
            ('type()', 'Type casting method'),
            ('float()', 'Convert to default float type'),
            ('double()', 'Convert to float64'),
            ('half()', 'Convert to float16'),
            ('int()', 'Convert to default int type'),
            ('long()', 'Convert to int64'),
            ('short()', 'Convert to int16'),
            ('byte()', 'Convert to uint8'),
            ('bool()', 'Convert to boolean')
        ]
    
    def demonstrate_conversions(self):
        """Demonstrate all conversion methods"""
        original = torch.randn(3, 4)
        print(f"Original tensor dtype: {original.dtype}")
        print(f"Original tensor: \n{original}\n")
        
        # Method 1: .to() method (most flexible)
        float16_tensor = original.to(torch.float16)
        int_tensor = original.to(torch.int32)
        bool_tensor = original.to(torch.bool)
        
        print("Using .to() method:")
        print(f"Float16: {float16_tensor.dtype}, shape: {float16_tensor.shape}")
        print(f"Int32:   {int_tensor.dtype}, shape: {int_tensor.shape}")
        print(f"Bool:    {bool_tensor.dtype}, shape: {bool_tensor.shape}")
        
        # Method 2: Type-specific methods
        print(f"\nType-specific conversions:")
        print(f"float(): {original.float().dtype}")
        print(f"double(): {original.double().dtype}")
        print(f"half(): {original.half().dtype}")
        print(f"int(): {original.int().dtype}")
        print(f"long(): {original.long().dtype}")
        print(f"bool(): {original.bool().dtype}")
        
        # Method 3: .type() method
        print(f"\nUsing .type() method:")
        print(f"type(torch.float16): {original.type(torch.float16).dtype}")
        print(f"type('torch.IntTensor'): {original.type('torch.IntTensor').dtype}")
    
    def safe_conversion_with_checks(self, tensor, target_dtype):
        """Safe conversion with range and precision checks"""
        print(f"\nSafe conversion from {tensor.dtype} to {target_dtype}:")
        
        # Check if conversion might lose information
        if tensor.dtype.is_floating_point and not target_dtype.is_floating_point:
            print("Warning: Converting from float to int - decimal part will be lost")
            print(f"Min value: {tensor.min().item():.6f}")
            print(f"Max value: {tensor.max().item():.6f}")
        
        # Perform conversion
        converted = tensor.to(target_dtype)
        
        # Check for overflow/underflow
        if target_dtype.is_floating_point:
            if torch.isfinite(converted).all():
                print("✓ No overflow/underflow detected")
            else:
                print("⚠ Overflow/underflow detected!")
        
        return converted

# Demonstrate conversions
converter = TypeConverter()
converter.demonstrate_conversions()

# Safe conversion example
test_tensor = torch.tensor([-1.7, 2.3, 100.9, -50.1])
safe_int = converter.safe_conversion_with_checks(test_tensor, torch.int32)
print(f"Original: {test_tensor}")
print(f"Converted: {safe_int}")
```

### 1.3 Tensor Creation Methods

#### From Python Data Structures

**Creating from Lists and Tuples:**
```python
class TensorCreationDemo:
    """Comprehensive tensor creation methods demonstration"""
    
    def from_python_structures(self):
        """Create tensors from Python data structures"""
        print("=== CREATING FROM PYTHON STRUCTURES ===")
        
        # From lists
        list_1d = [1, 2, 3, 4, 5]
        list_2d = [[1, 2, 3], [4, 5, 6]]
        list_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        
        tensor_1d = torch.tensor(list_1d)
        tensor_2d = torch.tensor(list_2d)
        tensor_3d = torch.tensor(list_3d)
        
        print(f"From 1D list: {tensor_1d}, shape: {tensor_1d.shape}")
        print(f"From 2D list: \n{tensor_2d}, shape: {tensor_2d.shape}")
        print(f"From 3D list shape: {tensor_3d.shape}")
        
        # From nested tuples
        tuple_data = ((1, 2, 3), (4, 5, 6))
        tensor_from_tuple = torch.tensor(tuple_data)
        print(f"From tuple: \n{tensor_from_tuple}")
        
        # Mixed data types (will be cast to common type)
        mixed_list = [1, 2.5, 3, 4.0]  # int and float
        mixed_tensor = torch.tensor(mixed_list)
        print(f"Mixed types: {mixed_tensor}, dtype: {mixed_tensor.dtype}")
        
        # Explicit dtype specification
        int_tensor = torch.tensor([1.1, 2.9, 3.7], dtype=torch.int32)
        print(f"Explicit int32: {int_tensor}")
        
        # Boolean from comparison
        bool_list = [True, False, True, False]
        bool_tensor = torch.tensor(bool_list, dtype=torch.bool)
        print(f"Boolean tensor: {bool_tensor}")
    
    def from_numpy_arrays(self):
        """Create tensors from NumPy arrays"""
        print("\n=== CREATING FROM NUMPY ARRAYS ===")
        
        # Create NumPy arrays
        np_1d = np.array([1, 2, 3, 4, 5])
        np_2d = np.random.randn(3, 4)
        np_bool = np.array([True, False, True])
        
        # Convert to PyTorch tensors
        tensor_from_np_1d = torch.from_numpy(np_1d)
        tensor_from_np_2d = torch.from_numpy(np_2d)
        tensor_from_np_bool = torch.from_numpy(np_bool)
        
        print(f"From NumPy 1D: {tensor_from_np_1d}")
        print(f"From NumPy 2D shape: {tensor_from_np_2d.shape}")
        print(f"From NumPy bool: {tensor_from_np_bool}")
        
        # Memory sharing demonstration
        print("\nMemory sharing between NumPy and PyTorch:")
        np_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        torch_tensor = torch.from_numpy(np_array)
        
        print(f"Original NumPy: {np_array}")
        print(f"PyTorch tensor: {torch_tensor}")
        
        # Modify NumPy array - PyTorch tensor also changes!
        np_array[0] = 999
        print(f"After modifying NumPy: {np_array}")
        print(f"PyTorch tensor: {torch_tensor}")
        
        # To avoid sharing memory, use .clone()
        independent_tensor = torch.from_numpy(np_array).clone()
        np_array[0] = 1
        print(f"Independent tensor: {independent_tensor}")
        print(f"NumPy after reset: {np_array}")
        
        # Alternative: torch.tensor() creates a copy
        copy_tensor = torch.tensor(np_array)
        np_array[0] = 777
        print(f"Copy tensor (unchanged): {copy_tensor}")
        print(f"NumPy array: {np_array}")

# Demonstrate creation methods
creator = TensorCreationDemo()
creator.from_python_structures()
creator.from_numpy_arrays()
```

#### Initialization Functions

**Zeros, Ones, and Empty Tensors:**
```python
class TensorInitialization:
    """Comprehensive tensor initialization methods"""
    
    def basic_initialization(self):
        """Basic tensor initialization functions"""
        print("=== BASIC INITIALIZATION ===")
        
        shape = (3, 4)
        
        # Zeros tensor
        zeros_tensor = torch.zeros(shape)
        print(f"Zeros tensor:\n{zeros_tensor}")
        
        # Ones tensor
        ones_tensor = torch.ones(shape)
        print(f"Ones tensor:\n{ones_tensor}")
        
        # Empty tensor (uninitialized memory)
        empty_tensor = torch.empty(shape)
        print(f"Empty tensor (random values):\n{empty_tensor}")
        
        # Full tensor (filled with specific value)
        full_tensor = torch.full(shape, 3.14159)
        print(f"Full tensor (π):\n{full_tensor}")
        
        # With specific dtype and device
        zeros_int = torch.zeros(shape, dtype=torch.int64)
        ones_half = torch.ones(shape, dtype=torch.float16)
        print(f"Zeros (int64): dtype={zeros_int.dtype}")
        print(f"Ones (float16): dtype={ones_half.dtype}")
    
    def like_functions(self):
        """Creating tensors with same properties as existing tensors"""
        print("\n=== 'LIKE' FUNCTIONS ===")
        
        # Reference tensor
        reference = torch.randn(2, 3, dtype=torch.float32, requires_grad=True)
        print(f"Reference tensor shape: {reference.shape}, dtype: {reference.dtype}")
        
        # Create similar tensors
        zeros_like = torch.zeros_like(reference)
        ones_like = torch.ones_like(reference)
        empty_like = torch.empty_like(reference)
        full_like = torch.full_like(reference, 2.71828)
        
        print(f"zeros_like shape: {zeros_like.shape}, dtype: {zeros_like.dtype}")
        print(f"zeros_like requires_grad: {zeros_like.requires_grad}")
        print(f"full_like sample value: {full_like[0, 0].item():.5f}")
    
    def random_initialization(self):
        """Random tensor initialization methods"""
        print("\n=== RANDOM INITIALIZATION ===")
        
        shape = (3, 4)
        
        # Uniform random [0, 1)
        rand_uniform = torch.rand(shape)
        print(f"Uniform [0,1): min={rand_uniform.min().item():.4f}, "
              f"max={rand_uniform.max().item():.4f}")
        
        # Standard normal distribution (mean=0, std=1)
        randn_normal = torch.randn(shape)
        print(f"Normal N(0,1): mean={randn_normal.mean().item():.4f}, "
              f"std={randn_normal.std().item():.4f}")
        
        # Random integers
        randint_tensor = torch.randint(0, 10, shape)
        print(f"Random integers [0,10): \n{randint_tensor}")
        
        # Random permutation
        perm_tensor = torch.randperm(10)
        print(f"Random permutation: {perm_tensor}")
        
        # Random with specific distribution parameters
        normal_custom = torch.normal(mean=5.0, std=2.0, size=shape)
        print(f"Normal N(5,2): mean={normal_custom.mean().item():.4f}")
        
        # Multinomial sampling
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        samples = torch.multinomial(weights, num_samples=10, replacement=True)
        print(f"Multinomial samples: {samples}")
    
    def advanced_initialization(self):
        """Advanced initialization patterns for deep learning"""
        print("\n=== ADVANCED INITIALIZATION ===")
        
        # Xavier/Glorot initialization
        def xavier_init(size, gain=1.0):
            fan_in, fan_out = size[1], size[0]
            std = gain * torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            return torch.normal(0, std, size)
        
        # Kaiming/He initialization
        def kaiming_init(size, mode='fan_in', nonlinearity='relu'):
            fan_in, fan_out = size[1], size[0]
            fan = fan_in if mode == 'fan_in' else fan_out
            gain = torch.nn.init.calculate_gain(nonlinearity)
            std = gain / torch.sqrt(torch.tensor(fan))
            return torch.normal(0, std, size)
        
        layer_size = (128, 256)  # (out_features, in_features)
        
        xavier_weights = xavier_init(layer_size)
        kaiming_weights = kaiming_init(layer_size, nonlinearity='relu')
        
        print(f"Xavier init: mean={xavier_weights.mean().item():.6f}, "
              f"std={xavier_weights.std().item():.6f}")
        print(f"Kaiming init: mean={kaiming_weights.mean().item():.6f}, "
              f"std={kaiming_weights.std().item():.6f}")
        
        # Using PyTorch's built-in initialization
        linear_layer = torch.nn.Linear(256, 128)
        print(f"Default Linear layer weight std: {linear_layer.weight.std().item():.6f}")
        
        # Apply Xavier initialization
        torch.nn.init.xavier_normal_(linear_layer.weight)
        print(f"After Xavier init std: {linear_layer.weight.std().item():.6f}")

# Demonstrate initialization methods
initializer = TensorInitialization()
initializer.basic_initialization()
initializer.like_functions()
initializer.random_initialization()
initializer.advanced_initialization()
```

#### Special Tensor Creation

**Identity, Diagonal, and Structured Tensors:**
```python
class SpecialTensorCreation:
    """Special tensor creation patterns and utilities"""
    
    def identity_and_diagonal(self):
        """Identity matrices and diagonal tensors"""
        print("=== IDENTITY AND DIAGONAL TENSORS ===")
        
        # Identity matrix
        identity_3x3 = torch.eye(3)
        identity_3x5 = torch.eye(3, 5)  # Rectangular identity
        print(f"3x3 Identity:\n{identity_3x3}")
        print(f"3x5 Identity:\n{identity_3x5}")
        
        # Diagonal tensor from vector
        diag_values = torch.tensor([1, 2, 3, 4])
        diag_matrix = torch.diag(diag_values)
        print(f"Diagonal matrix:\n{diag_matrix}")
        
        # Extract diagonal from matrix
        matrix = torch.randn(4, 4)
        extracted_diag = torch.diag(matrix)
        print(f"Extracted diagonal: {extracted_diag}")
        
        # Off-diagonal elements
        upper_diag = torch.diag(diag_values, diagonal=1)  # Above main diagonal
        lower_diag = torch.diag(diag_values, diagonal=-1)  # Below main diagonal
        print(f"Upper diagonal:\n{upper_diag}")
        print(f"Lower diagonal:\n{lower_diag}")
    
    def range_and_sequence_tensors(self):
        """Range, linspace, and sequence tensors"""
        print("\n=== RANGE AND SEQUENCE TENSORS ===")
        
        # Range tensors
        range_int = torch.arange(0, 10)  # [0, 1, 2, ..., 9]
        range_float = torch.arange(0, 10, 0.5)  # Step size 0.5
        range_desc = torch.arange(10, 0, -1)  # Descending
        
        print(f"Range int: {range_int}")
        print(f"Range float: {range_float}")
        print(f"Range descending: {range_desc}")
        
        # Linear space
        linspace_tensor = torch.linspace(0, 1, steps=11)  # 11 points from 0 to 1
        logspace_tensor = torch.logspace(0, 2, steps=5)   # 5 points from 10^0 to 10^2
        
        print(f"Linspace: {linspace_tensor}")
        print(f"Logspace: {logspace_tensor}")
        
        # Meshgrid for coordinate grids
        x = torch.linspace(-2, 2, 5)
        y = torch.linspace(-1, 1, 3)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        print(f"X grid:\n{X}")
        print(f"Y grid:\n{Y}")
        
        # Grid coordinates for image processing
        coords = torch.stack([X, Y], dim=0)
        print(f"Combined coordinates shape: {coords.shape}")
    
    def triangular_and_band_matrices(self):
        """Triangular and banded matrices"""
        print("\n=== TRIANGULAR AND BAND MATRICES ===")
        
        size = (4, 4)
        base_matrix = torch.randn(size)
        
        # Upper triangular
        upper_tri = torch.triu(base_matrix)
        upper_tri_diag1 = torch.triu(base_matrix, diagonal=1)  # Above main diagonal
        
        print(f"Upper triangular:\n{upper_tri}")
        print(f"Upper triangular (diagonal=1):\n{upper_tri_diag1}")
        
        # Lower triangular
        lower_tri = torch.tril(base_matrix)
        lower_tri_diag_neg1 = torch.tril(base_matrix, diagonal=-1)  # Below main diagonal
        
        print(f"Lower triangular:\n{lower_tri}")
        print(f"Lower triangular (diagonal=-1):\n{lower_tri_diag_neg1}")
        
        # Triangular matrix creation
        ones_upper = torch.triu(torch.ones(size))
        ones_lower = torch.tril(torch.ones(size))
        
        print(f"Upper triangular ones:\n{ones_upper}")
        print(f"Lower triangular ones:\n{ones_lower}")
    
    def block_and_structured_tensors(self):
        """Block and structured tensor creation"""
        print("\n=== BLOCK AND STRUCTURED TENSORS ===")
        
        # Block diagonal matrix
        def create_block_diagonal(*blocks):
            """Create block diagonal matrix from given blocks"""
            sizes = [block.shape for block in blocks]
            total_size = sum(size[0] for size in sizes)
            result = torch.zeros(total_size, total_size, dtype=blocks[0].dtype)
            
            row_start, col_start = 0, 0
            for block in blocks:
                rows, cols = block.shape
                result[row_start:row_start+rows, col_start:col_start+cols] = block
                row_start += rows
                col_start += cols
            
            return result
        
        # Create blocks
        block1 = torch.eye(2)
        block2 = torch.ones(3, 3)
        block3 = torch.diag(torch.tensor([4.0, 5.0]))
        
        block_diag = create_block_diagonal(block1, block2, block3)
        print(f"Block diagonal matrix:\n{block_diag}")
        
        # Toeplitz matrix (constant diagonals)
        def create_toeplitz(first_row, first_col):
            """Create Toeplitz matrix from first row and column"""
            n, m = len(first_col), len(first_row)
            matrix = torch.zeros(n, m, dtype=first_row.dtype)
            
            for i in range(n):
                for j in range(m):
                    if j >= i:
                        matrix[i, j] = first_row[j - i]
                    else:
                        matrix[i, j] = first_col[i - j]
            
            return matrix
        
        first_row = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        first_col = torch.tensor([1, 5, 6], dtype=torch.float32)
        toeplitz = create_toeplitz(first_row, first_col)
        print(f"Toeplitz matrix:\n{toeplitz}")
        
        # Circulant matrix (special case of Toeplitz)
        def create_circulant(first_row):
            """Create circulant matrix from first row"""
            n = len(first_row)
            matrix = torch.zeros(n, n, dtype=first_row.dtype)
            
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = first_row[(j - i) % n]
            
            return matrix
        
        circulant_row = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        circulant = create_circulant(circulant_row)
        print(f"Circulant matrix:\n{circulant}")

# Demonstrate special tensor creation
special_creator = SpecialTensorCreation()
special_creator.identity_and_diagonal()
special_creator.range_and_sequence_tensors()
special_creator.triangular_and_band_matrices()
special_creator.block_and_structured_tensors()
```

---

## 2. Tensor Properties and Metadata

### 2.1 Shape and Dimension Manipulation

#### Understanding Tensor Shapes

**Shape Analysis and Manipulation:**
```python
class ShapeAnalyzer:
    """Comprehensive tensor shape analysis and manipulation"""
    
    def __init__(self):
        self.example_tensors = self._create_example_tensors()
    
    def _create_example_tensors(self):
        """Create various tensors for shape analysis"""
        return {
            'scalar': torch.tensor(42),
            'vector': torch.randn(10),
            'matrix': torch.randn(5, 8),
            'image_batch': torch.randn(32, 3, 224, 224),
            'sequence': torch.randint(0, 1000, (16, 512)),
            'transformer_hidden': torch.randn(8, 128, 768),
            'conv_feature': torch.randn(64, 256, 14, 14)
        }
    
    def analyze_shapes(self):
        """Analyze shapes of different tensor types"""
        print("=== TENSOR SHAPE ANALYSIS ===")
        
        for name, tensor in self.example_tensors.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Size: {tensor.size()}")  # Alternative to shape
            print(f"  Dimensions: {tensor.ndim}")
            print(f"  Elements: {tensor.numel()}")
            print(f"  Memory: {tensor.numel() * tensor.element_size()} bytes")
            
            # Dimension-specific analysis
            if tensor.ndim > 0:
                print(f"  Size per dimension: {[tensor.size(i) for i in range(tensor.ndim)]}")
        
        # Shape arithmetic
        print(f"\n=== SHAPE ARITHMETIC ===")
        tensor_4d = self.example_tensors['image_batch']
        batch_size, channels, height, width = tensor_4d.shape
        
        print(f"Original shape: {tensor_4d.shape}")
        print(f"Unpacked: batch={batch_size}, channels={channels}, h={height}, w={width}")
        print(f"Total pixels per image: {channels * height * width}")
        print(f"Total pixels in batch: {batch_size * channels * height * width}")
    
    def shape_compatibility_checking(self):
        """Check shape compatibility for operations"""
        print(f"\n=== SHAPE COMPATIBILITY ===")
        
        # Matrix multiplication compatibility
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.randn(3, 5)
        
        print(f"Matrix A shape: {a.shape}")
        print(f"Matrix B shape: {b.shape}")
        print(f"Matrix C shape: {c.shape}")
        
        # Check if A @ B is possible
        if a.shape[1] == b.shape[0]:
            result_ab = a @ b
            print(f"A @ B possible: {result_ab.shape}")
        else:
            print("A @ B not compatible")
        
        # Check if A + C is possible
        if a.shape == c.shape:
            result_ac = a + c
            print(f"A + C possible: {result_ac.shape}")
        else:
            print(f"A + C not compatible: {a.shape} vs {c.shape}")
        
        # Broadcasting compatibility
        def check_broadcasting_compatibility(shape1, shape2):
            """Check if two shapes are compatible for broadcasting"""
            # Reverse shapes to check from rightmost dimension
            s1, s2 = list(reversed(shape1)), list(reversed(shape2))
            
            # Pad shorter shape with 1s
            max_len = max(len(s1), len(s2))
            s1 = [1] * (max_len - len(s1)) + s1
            s2 = [1] * (max_len - len(s2)) + s2
            
            result_shape = []
            for dim1, dim2 in zip(s1, s2):
                if dim1 == 1:
                    result_shape.append(dim2)
                elif dim2 == 1:
                    result_shape.append(dim1)
                elif dim1 == dim2:
                    result_shape.append(dim1)
                else:
                    return None  # Not compatible
            
            return tuple(reversed(result_shape))
        
        # Test broadcasting
        shapes_to_test = [
            ((3, 1), (1, 4)),
            ((2, 3, 1), (1, 4)),
            ((5, 1, 3), (1, 2, 1)),
            ((3, 4), (5, 1))  # Not compatible
        ]
        
        for shape1, shape2 in shapes_to_test:
            result_shape = check_broadcasting_compatibility(shape1, shape2)
            if result_shape:
                print(f"{shape1} + {shape2} → {result_shape}")
            else:
                print(f"{shape1} + {shape2} → Not compatible")

# Demonstrate shape analysis
shape_analyzer = ShapeAnalyzer()
shape_analyzer.analyze_shapes()
shape_analyzer.shape_compatibility_checking()
```

#### Reshaping and View Operations

**Comprehensive Reshaping Techniques:**
```python
class ReshapingMaster:
    """Master class for tensor reshaping operations"""
    
    def basic_reshaping(self):
        """Basic reshape and view operations"""
        print("=== BASIC RESHAPING ===")
        
        # Original tensor
        original = torch.randn(2, 3, 4)
        print(f"Original shape: {original.shape}, elements: {original.numel()}")
        
        # Reshape vs View
        reshaped = original.reshape(6, 4)  # New shape must have same number of elements
        viewed = original.view(6, 4)       # Creates view if possible
        
        print(f"Reshaped: {reshaped.shape}")
        print(f"Viewed: {viewed.shape}")
        
        # Check if they share memory
        print(f"Original storage address: {original.data_ptr()}")
        print(f"Reshaped storage address: {reshaped.data_ptr()}")
        print(f"Viewed storage address: {viewed.data_ptr()}")
        print(f"View shares memory: {viewed.data_ptr() == original.data_ptr()}")
        
        # Flatten operations
        flattened = original.flatten()
        flattened_from_dim1 = original.flatten(start_dim=1)  # Flatten from dimension 1
        
        print(f"Flattened: {flattened.shape}")
        print(f"Flattened from dim 1: {flattened_from_dim1.shape}")
        
        # Ravel (alias for flatten)
        raveled = original.ravel()
        print(f"Raveled: {raveled.shape}")
    
    def advanced_reshaping(self):
        """Advanced reshaping with automatic dimension inference"""
        print(f"\n=== ADVANCED RESHAPING ===")
        
        tensor_3d = torch.randn(2, 3, 4)
        
        # Use -1 for automatic dimension calculation
        auto_dim = tensor_3d.reshape(-1, 4)  # Automatically calculate first dimension
        print(f"Original: {tensor_3d.shape}")
        print(f"Auto dimension (-1, 4): {auto_dim.shape}")
        
        # Multiple automatic dimensions (only one -1 allowed)
        try:
            invalid = tensor_3d.reshape(-1, -1, 4)  # This will fail
        except RuntimeError as e:
            print(f"Multiple -1 error: {str(e)[:50]}...")
        
        # Reshape with broadcasting in mind
        tensor_for_broadcast = torch.randn(12)
        
        # Different ways to prepare for broadcasting
        column_vector = tensor_for_broadcast.reshape(-1, 1)  # 12x1
        row_vector = tensor_for_broadcast.reshape(1, -1)     # 1x12
        matrix_3x4 = tensor_for_broadcast.reshape(3, 4)      # 3x4
        
        print(f"Column vector: {column_vector.shape}")
        print(f"Row vector: {row_vector.shape}")
        print(f"Matrix 3x4: {matrix_3x4.shape}")
    
    def contiguity_and_memory_layout(self):
        """Understanding contiguity and memory layout effects"""
        print(f"\n=== CONTIGUITY AND MEMORY LAYOUT ===")
        
        # Create contiguous tensor
        original = torch.randn(2, 3, 4)
        print(f"Original contiguous: {original.is_contiguous()}")
        
        # Transpose makes non-contiguous
        transposed = original.transpose(1, 2)  # Swap dimensions 1 and 2
        print(f"Transposed contiguous: {transposed.is_contiguous()}")
        print(f"Original stride: {original.stride()}")
        print(f"Transposed stride: {transposed.stride()}")
        
        # View operations require contiguous tensor
        try:
            view_of_transposed = transposed.view(-1)
        except RuntimeError as e:
            print(f"View error: {str(e)[:50]}...")
        
        # Make contiguous and then view
        contiguous_transposed = transposed.contiguous()
        print(f"Made contiguous: {contiguous_transposed.is_contiguous()}")
        view_after_contiguous = contiguous_transposed.view(-1)
        print(f"View after contiguous: {view_after_contiguous.shape}")
        
        # Reshape vs view for non-contiguous tensors
        reshaped_transposed = transposed.reshape(-1)  # Works even if non-contiguous
        print(f"Reshape on non-contiguous: {reshaped_transposed.shape}")
        
        # Memory layout information
        def analyze_memory_layout(tensor, name):
            print(f"{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Stride: {tensor.stride()}")
            print(f"  Contiguous: {tensor.is_contiguous()}")
            print(f"  Storage offset: {tensor.storage_offset()}")
        
        analyze_memory_layout(original, "Original")
        analyze_memory_layout(transposed, "Transposed")
        analyze_memory_layout(contiguous_transposed, "Made Contiguous")
    
    def dimension_manipulation(self):
        """Advanced dimension manipulation techniques"""
        print(f"\n=== DIMENSION MANIPULATION ===")
        
        tensor_3d = torch.randn(2, 1, 4)
        
        # Squeeze (remove dimensions of size 1)
        squeezed = tensor_3d.squeeze()  # Remove all size-1 dimensions
        squeezed_dim1 = tensor_3d.squeeze(1)  # Remove specific dimension if size 1
        
        print(f"Original: {tensor_3d.shape}")
        print(f"Squeezed all: {squeezed.shape}")
        print(f"Squeezed dim 1: {squeezed_dim1.shape}")
        
        # Unsqueeze (add dimensions of size 1)
        unsqueezed = squeezed.unsqueeze(0)  # Add dimension at position 0
        unsqueezed_end = squeezed.unsqueeze(-1)  # Add dimension at end
        
        print(f"Unsqueezed at 0: {unsqueezed.shape}")
        print(f"Unsqueezed at end: {unsqueezed_end.shape}")
        
        # Multiple unsqueeze operations
        multiple_unsqueeze = squeezed.unsqueeze(0).unsqueeze(-1).unsqueeze(2)
        print(f"Multiple unsqueeze: {multiple_unsqueeze.shape}")
        
        # Expand (repeat dimensions without copying data)
        small_tensor = torch.randn(1, 3, 1)
        expanded = small_tensor.expand(2, 3, 4)  # Expand specific dimensions
        expanded_as = small_tensor.expand_as(torch.randn(2, 3, 4))  # Expand to match another tensor
        
        print(f"Small tensor: {small_tensor.shape}")
        print(f"Expanded: {expanded.shape}")
        print(f"Expanded as: {expanded_as.shape}")
        print(f"Shares memory: {expanded.data_ptr() == small_tensor.data_ptr()}")
        
        # Repeat (actually copy data)
        repeated = small_tensor.repeat(2, 1, 4)  # Repeat along each dimension
        print(f"Repeated: {repeated.shape}")
        print(f"Shares memory with original: {repeated.data_ptr() == small_tensor.data_ptr()}")

# Demonstrate reshaping techniques
reshaper = ReshapingMaster()
reshaper.basic_reshaping()
reshaper.advanced_reshaping()
reshaper.contiguity_and_memory_layout()
reshaper.dimension_manipulation()
```

### 2.2 Device Management

#### CPU vs GPU Tensors

**Device Operations and Management:**
```python
class DeviceManager:
    """Comprehensive device management for tensors"""
    
    def __init__(self):
        self.device_info = self._get_device_info()
    
    def _get_device_info(self):
        """Gather comprehensive device information"""
        info = {
            'cpu_available': True,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'cuda_devices': []
        }
        
        if info['cuda_available']:
            for i in range(info['cuda_device_count']):
                device_props = torch.cuda.get_device_properties(i)
                info['cuda_devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count
                })
        
        return info
    
    def display_device_info(self):
        """Display comprehensive device information"""
        print("=== DEVICE INFORMATION ===")
        print(f"CPU Available: {self.device_info['cpu_available']}")
        print(f"CUDA Available: {self.device_info['cuda_available']}")
        
        if self.device_info['cuda_available']:
            print(f"CUDA Devices: {self.device_info['cuda_device_count']}")
            print(f"Current Device: {self.device_info['current_device']}")
            
            for device in self.device_info['cuda_devices']:
                print(f"  Device {device['id']}: {device['name']}")
                print(f"    Memory: {device['total_memory'] / 1e9:.1f} GB")
                print(f"    Compute Capability: {device['major']}.{device['minor']}")
                print(f"    Multiprocessors: {device['multi_processor_count']}")
        else:
            print("No CUDA devices available")
    
    def device_creation_and_movement(self):
        """Demonstrate tensor creation and movement between devices"""
        print(f"\n=== DEVICE CREATION AND MOVEMENT ===")
        
        # Create tensors on different devices
        cpu_tensor = torch.randn(1000, 1000)
        print(f"CPU tensor device: {cpu_tensor.device}")
        print(f"CPU tensor memory: {cpu_tensor.numel() * cpu_tensor.element_size() / 1e6:.1f} MB")
        
        if self.device_info['cuda_available']:
            # Create tensor directly on GPU
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"GPU tensor device: {gpu_tensor.device}")
            
            # Move CPU tensor to GPU
            cpu_to_gpu = cpu_tensor.to('cuda')
            print(f"Moved to GPU device: {cpu_to_gpu.device}")
            
            # Move back to CPU
            gpu_to_cpu = gpu_tensor.to('cpu')
            print(f"Moved to CPU device: {gpu_to_cpu.device}")
            
            # Device object usage
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            device_tensor = torch.randn(100, 100, device=device)
            print(f"Device object tensor: {device_tensor.device}")
            
            # Multiple GPU handling
            if self.device_info['cuda_device_count'] > 1:
                gpu1_tensor = torch.randn(100, 100, device='cuda:1')
                print(f"GPU 1 tensor: {gpu1_tensor.device}")
        else:
            print("No CUDA available - using CPU only")
    
    def memory_management(self):
        """GPU memory management techniques"""
        print(f"\n=== GPU MEMORY MANAGEMENT ===")
        
        if not self.device_info['cuda_available']:
            print("No CUDA available - skipping GPU memory management")
            return
        
        # Check initial memory
        if torch.cuda.is_available():
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
            print(f"Initial GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.3f} GB")
        
        # Allocate some GPU memory
        large_tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            large_tensors.append(tensor)
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"After tensor {i+1}: allocated={allocated:.3f}GB, cached={cached:.3f}GB")
        
        # Delete tensors and check memory
        del large_tensors
        print(f"After deletion: allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"After deletion: cached={torch.cuda.memory_reserved() / 1e9:.3f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        print(f"After cache clear: allocated={torch.cuda.memory_allocated() / 1e9:.3f} GB")
        print(f"After cache clear: cached={torch.cuda.memory_reserved() / 1e9:.3f} GB")
        
        # Memory context manager
        class GPUMemoryTracker:
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    self.start_allocated = torch.cuda.memory_allocated()
                    self.start_cached = torch.cuda.memory_reserved()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if torch.cuda.is_available():
                    self.end_allocated = torch.cuda.memory_allocated()
                    self.end_cached = torch.cuda.memory_reserved()
                    self.peak_allocated = torch.cuda.max_memory_allocated()
                    
                    print(f"Memory usage summary:")
                    print(f"  Start allocated: {self.start_allocated / 1e9:.3f} GB")
                    print(f"  End allocated: {self.end_allocated / 1e9:.3f} GB")
                    print(f"  Peak allocated: {self.peak_allocated / 1e9:.3f} GB")
                    print(f"  Net change: {(self.end_allocated - self.start_allocated) / 1e9:.3f} GB")
        
        # Use memory tracker
        with GPUMemoryTracker():
            temp_tensor = torch.randn(2000, 2000, device='cuda')
            temp_result = temp_tensor @ temp_tensor.T
            del temp_tensor, temp_result
    
    def efficient_device_patterns(self):
        """Efficient patterns for device management"""
        print(f"\n=== EFFICIENT DEVICE PATTERNS ===")
        
        # Pattern 1: Device-agnostic code
        def device_agnostic_function(input_tensor, weight_tensor):
            """Function that works on any device"""
            # Ensure both tensors are on the same device
            device = input_tensor.device
            weight_tensor = weight_tensor.to(device)
            
            # Perform computation
            output = torch.mm(input_tensor, weight_tensor)
            return output
        
        # Test device-agnostic function
        cpu_input = torch.randn(100, 200)
        cpu_weight = torch.randn(200, 50)
        
        result_cpu = device_agnostic_function(cpu_input, cpu_weight)
        print(f"CPU result device: {result_cpu.device}")
        
        if self.device_info['cuda_available']:
            gpu_input = cpu_input.to('cuda')
            result_gpu = device_agnostic_function(gpu_input, cpu_weight)
            print(f"GPU result device: {result_gpu.device}")
        
        # Pattern 2: Automatic device selection
        class AutoDevice:
            def __init__(self, prefer_gpu=True):
                self.device = self._select_device(prefer_gpu)
            
            def _select_device(self, prefer_gpu):
                if prefer_gpu and torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    return torch.device('cpu')
            
            def to_device(self, tensor):
                return tensor.to(self.device)
            
            def __str__(self):
                return str(self.device)
        
        auto_device = AutoDevice()
        print(f"Auto-selected device: {auto_device}")
        
        # Pattern 3: Batch device transfer
        def batch_to_device(tensors, device):
            """Efficiently move multiple tensors to device"""
            return [tensor.to(device, non_blocking=True) for tensor in tensors]
        
        # Create multiple tensors
        tensor_list = [torch.randn(100, 100) for _ in range(5)]
        
        if self.device_info['cuda_available']:
            # Transfer batch to GPU
            gpu_tensors = batch_to_device(tensor_list, 'cuda')
            print(f"Batch transferred to: {gpu_tensors[0].device}")

# Demonstrate device management
device_manager = DeviceManager()
device_manager.display_device_info()
device_manager.device_creation_and_movement()
device_manager.memory_management()
device_manager.efficient_device_patterns()
```

---

## 3. Basic Tensor Operations

### 3.1 Element-wise Operations

#### Arithmetic Operations

**Complete Arithmetic Operations Coverage:**
```python
class ArithmeticOperations:
    """Comprehensive arithmetic operations on tensors"""
    
    def basic_arithmetic(self):
        """Basic arithmetic operations"""
        print("=== BASIC ARITHMETIC OPERATIONS ===")
        
        # Create sample tensors
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([2.0, 3.0, 4.0, 5.0])
        scalar = 2.5
        
        print(f"Tensor A: {a}")
        print(f"Tensor B: {b}")
        print(f"Scalar: {scalar}")
        
        # Addition
        add_result = a + b
        add_scalar = a + scalar
        add_inplace = a.clone().add_(b)  # In-place operation
        
        print(f"\nAddition:")
        print(f"A + B = {add_result}")
        print(f"A + scalar = {add_scalar}")
        print(f"A.add_(B) = {add_inplace}")
        
        # Subtraction
        sub_result = a - b
        sub_scalar = a - scalar
        rsub_scalar = torch.rsub(a, scalar)  # scalar - A
        
        print(f"\nSubtraction:")
        print(f"A - B = {sub_result}")
        print(f"A - scalar = {sub_scalar}")
        print(f"scalar - A = {rsub_scalar}")
        
        # Multiplication
        mul_result = a * b
        mul_scalar = a * scalar
        
        print(f"\nMultiplication:")
        print(f"A * B = {mul_result}")
        print(f"A * scalar = {mul_scalar}")
        
        # Division
        div_result = a / b
        div_scalar = a / scalar
        floor_div = a // b  # Floor division
        
        print(f"\nDivision:")
        print(f"A / B = {div_result}")
        print(f"A / scalar = {div_scalar}")
        print(f"A // B = {floor_div}")
        
        # Modulo
        mod_result = a % b
        print(f"A % B = {mod_result}")
        
        # Power operations
        pow_result = a ** 2
        pow_tensor = torch.pow(a, b)
        sqrt_result = torch.sqrt(a)
        
        print(f"\nPower operations:")
        print(f"A ** 2 = {pow_result}")
        print(f"A ** B = {pow_tensor}")
        print(f"sqrt(A) = {sqrt_result}")
    
    def advanced_arithmetic(self):
        """Advanced arithmetic operations"""
        print(f"\n=== ADVANCED ARITHMETIC ===")
        
        # Complex arithmetic
        a = torch.tensor([1.0, -2.0, 3.0, -4.0])
        
        # Absolute value and sign
        abs_result = torch.abs(a)
        sign_result = torch.sign(a)
        
        print(f"Original: {a}")
        print(f"Absolute: {abs_result}")
        print(f"Sign: {sign_result}")
        
        # Rounding operations
        float_tensor = torch.tensor([1.2, 2.7, -3.1, -4.9])
        
        floor_result = torch.floor(float_tensor)
        ceil_result = torch.ceil(float_tensor)
        round_result = torch.round(float_tensor)
        trunc_result = torch.trunc(float_tensor)
        
        print(f"\nFloat tensor: {float_tensor}")
        print(f"Floor: {floor_result}")
        print(f"Ceil: {ceil_result}")
        print(f"Round: {round_result}")
        print(f"Trunc: {trunc_result}")
        
        # Fractional and integer parts
        frac_result = torch.frac(float_tensor)
        
        print(f"Fractional part: {frac_result}")
        
        # Reciprocal
        positive_tensor = torch.tensor([1.0, 2.0, 4.0, 0.5])
        reciprocal = torch.reciprocal(positive_tensor)
        
        print(f"\nPositive tensor: {positive_tensor}")
        print(f"Reciprocal: {reciprocal}")
    
    def mathematical_functions(self):
        """Mathematical functions on tensors"""
        print(f"\n=== MATHEMATICAL FUNCTIONS ===")
        
        # Exponential and logarithmic
        x = torch.linspace(0.1, 2.0, 5)
        
        exp_result = torch.exp(x)
        log_result = torch.log(x)
        log10_result = torch.log10(x)
        log2_result = torch.log2(x)
        
        print(f"Input: {x}")
        print(f"exp(x): {exp_result}")
        print(f"log(x): {log_result}")
        print(f"log10(x): {log10_result}")
        print(f"log2(x): {log2_result}")
        
        # Trigonometric functions
        angles = torch.linspace(0, torch.pi, 5)
        
        sin_result = torch.sin(angles)
        cos_result = torch.cos(angles)
        tan_result = torch.tan(angles)
        
        print(f"\nAngles: {angles}")
        print(f"sin: {sin_result}")
        print(f"cos: {cos_result}")
        print(f"tan: {tan_result}")
        
        # Inverse trigonometric
        values = torch.linspace(-0.9, 0.9, 5)
        
        asin_result = torch.asin(values)
        acos_result = torch.acos(values)
        atan_result = torch.atan(values)
        
        print(f"\nValues: {values}")
        print(f"asin: {asin_result}")
        print(f"acos: {acos_result}")
        print(f"atan: {atan_result}")
        
        # Hyperbolic functions
        sinh_result = torch.sinh(x)
        cosh_result = torch.cosh(x)
        tanh_result = torch.tanh(x)
        
        print(f"\nHyperbolic functions for x={x}:")
        print(f"sinh: {sinh_result}")
        print(f"cosh: {cosh_result}")
        print(f"tanh: {tanh_result}")
    
    def error_functions_and_special(self):
        """Error functions and special mathematical functions"""
        print(f"\n=== ERROR FUNCTIONS AND SPECIAL ===")
        
        x = torch.linspace(-2, 2, 5)
        
        # Error function
        erf_result = torch.erf(x)
        erfc_result = torch.erfc(x)  # Complementary error function
        
        print(f"Input: {x}")
        print(f"erf(x): {erf_result}")
        print(f"erfc(x): {erfc_result}")
        
        # Gamma function
        positive_x = torch.linspace(0.5, 3.0, 5)
        gamma_result = torch.gamma(positive_x)
        lgamma_result = torch.lgamma(positive_x)  # Log gamma
        
        print(f"\nPositive input: {positive_x}")
        print(f"gamma(x): {gamma_result}")
        print(f"lgamma(x): {lgamma_result}")
        
        # Digamma function (derivative of log gamma)
        digamma_result = torch.digamma(positive_x)
        print(f"digamma(x): {digamma_result}")

# Demonstrate arithmetic operations
arithmetic_ops = ArithmeticOperations()
arithmetic_ops.basic_arithmetic()
arithmetic_ops.advanced_arithmetic()
arithmetic_ops.mathematical_functions()
arithmetic_ops.error_functions_and_special()
```

#### Comparison and Logical Operations

**Comprehensive Comparison Operations:**
```python
class ComparisonAndLogical:
    """Comparison and logical operations on tensors"""
    
    def comparison_operations(self):
        """All comparison operations"""
        print("=== COMPARISON OPERATIONS ===")
        
        a = torch.tensor([1, 2, 3, 4, 5])
        b = torch.tensor([2, 2, 1, 5, 3])
        scalar = 3
        
        print(f"Tensor A: {a}")
        print(f"Tensor B: {b}")
        print(f"Scalar: {scalar}")
        
        # Element-wise comparisons
        eq_result = a == b  # Equal
        ne_result = a != b  # Not equal
        gt_result = a > b   # Greater than
        ge_result = a >= b  # Greater than or equal
        lt_result = a < b   # Less than
        le_result = a <= b  # Less than or equal
        
        print(f"\nElement-wise comparisons:")
        print(f"A == B: {eq_result}")
        print(f"A != B: {ne_result}")
        print(f"A > B:  {gt_result}")
        print(f"A >= B: {ge_result}")
        print(f"A < B:  {lt_result}")
        print(f"A <= B: {le_result}")
        
        # Comparisons with scalar
        gt_scalar = a > scalar
        eq_scalar = a == scalar
        
        print(f"\nComparisons with scalar {scalar}:")
        print(f"A > {scalar}: {gt_scalar}")
        print(f"A == {scalar}: {eq_scalar}")
        
        # Functional forms
        equal_func = torch.eq(a, b)
        greater_func = torch.gt(a, b)
        less_func = torch.lt(a, b)
        
        print(f"\nFunctional forms:")
        print(f"torch.eq(A, B): {equal_func}")
        print(f"torch.gt(A, B): {greater_func}")
        print(f"torch.lt(A, B): {less_func}")
        
        # NaN handling
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0, float('nan')])
        isnan_result = torch.isnan(nan_tensor)
        isfinite_result = torch.isfinite(nan_tensor)
        isinf_result = torch.isinf(torch.tensor([1.0, float('inf'), -float('inf'), 0.0]))
        
        print(f"\nNaN/Inf handling:")
        print(f"Tensor with NaN: {nan_tensor}")
        print(f"isnan: {isnan_result}")
        print(f"isfinite: {isfinite_result}")
        print(f"isinf: {isinf_result}")
    
    def logical_operations(self):
        """Logical operations on boolean tensors"""
        print(f"\n=== LOGICAL OPERATIONS ===")
        
        # Create boolean tensors
        a = torch.tensor([True, False, True, False])
        b = torch.tensor([True, True, False, False])
        
        print(f"Boolean A: {a}")
        print(f"Boolean B: {b}")
        
        # Logical operations
        and_result = a & b  # Logical AND
        or_result = a | b   # Logical OR
        xor_result = a ^ b  # Logical XOR
        not_result = ~a     # Logical NOT
        
        print(f"\nLogical operations:")
        print(f"A & B: {and_result}")
        print(f"A | B: {or_result}")
        print(f"A ^ B: {xor_result}")
        print(f"~A: {not_result}")
        
        # Functional forms
        logical_and = torch.logical_and(a, b)
        logical_or = torch.logical_or(a, b)
        logical_xor = torch.logical_xor(a, b)
        logical_not = torch.logical_not(a)
        
        print(f"\nFunctional forms:")
        print(f"logical_and: {logical_and}")
        print(f"logical_or: {logical_or}")
        print(f"logical_xor: {logical_xor}")
        print(f"logical_not: {logical_not}")
        
        # Short-circuit evaluation patterns
        tensor_values = torch.tensor([1, 0, 3, 0, 5])
        
        # Convert to boolean (non-zero is True)
        bool_values = tensor_values.bool()
        print(f"\nTensor values: {tensor_values}")
        print(f"As boolean: {bool_values}")
        
        # Any and all operations
        any_result = torch.any(bool_values)
        all_result = torch.all(bool_values)
        
        print(f"Any True: {any_result}")
        print(f"All True: {all_result}")
        
        # Any/all along dimensions
        matrix_bool = torch.tensor([[True, False, True],
                                   [False, True, False],
                                   [True, True, True]])
        
        any_dim0 = torch.any(matrix_bool, dim=0)
        any_dim1 = torch.any(matrix_bool, dim=1)
        all_dim0 = torch.all(matrix_bool, dim=0)
        all_dim1 = torch.all(matrix_bool, dim=1)
        
        print(f"\nMatrix boolean:\n{matrix_bool}")
        print(f"Any along dim 0: {any_dim0}")
        print(f"Any along dim 1: {any_dim1}")
        print(f"All along dim 0: {all_dim0}")
        print(f"All along dim 1: {all_dim1}")
    
    def advanced_logical_patterns(self):
        """Advanced logical patterns and applications"""
        print(f"\n=== ADVANCED LOGICAL PATTERNS ===")
        
        # Masking patterns
        data = torch.randn(10)
        positive_mask = data > 0
        negative_mask = data < 0
        near_zero_mask = torch.abs(data) < 0.5
        
        print(f"Data: {data}")
        print(f"Positive mask: {positive_mask}")
        print(f"Near zero mask: {near_zero_mask}")
        
        # Apply masks
        positive_values = data[positive_mask]
        values_near_zero = data[near_zero_mask]
        
        print(f"Positive values: {positive_values}")
        print(f"Values near zero: {values_near_zero}")
        
        # Conditional selection
        condition = data > 0
        result = torch.where(condition, data, torch.zeros_like(data))
        print(f"Conditional result: {result}")
        
        # Multiple conditions
        x = torch.randn(10)
        
        # Classify values: negative (-1), small positive (0), large positive (1)
        classification = torch.where(x < 0, -1,
                                   torch.where(x < 1, 0, 1))
        
        print(f"\nClassification input: {x}")
        print(f"Classification result: {classification}")
        
        # Count operations
        positive_count = (x > 0).sum()
        negative_count = (x < 0).sum()
        zero_count = (x == 0).sum()
        
        print(f"Positive count: {positive_count}")
        print(f"Negative count: {negative_count}")
        print(f"Zero count: {zero_count}")

# Demonstrate comparison and logical operations
comp_logical = ComparisonAndLogical()
comp_logical.comparison_operations()
comp_logical.logical_operations()
comp_logical.advanced_logical_patterns()
```

### 3.2 Indexing and Slicing

#### Basic Indexing Patterns

**Comprehensive Indexing Techniques:**
```python
class IndexingMaster:
    """Master class for tensor indexing and slicing"""
    
    def basic_indexing(self):
        """Basic indexing operations"""
        print("=== BASIC INDEXING ===")
        
        # 1D tensor indexing
        tensor_1d = torch.tensor([10, 20, 30, 40, 50])
        
        print(f"1D tensor: {tensor_1d}")
        print(f"First element: {tensor_1d[0]}")
        print(f"Last element: {tensor_1d[-1]}")
        print(f"Second to last: {tensor_1d[-2]}")
        
        # 2D tensor indexing
        tensor_2d = torch.tensor([[1, 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12]])
        
        print(f"\n2D tensor:\n{tensor_2d}")
        print(f"Element [1, 2]: {tensor_2d[1, 2]}")
        print(f"Element [0, -1]: {tensor_2d[0, -1]}")
        print(f"First row: {tensor_2d[0]}")
        print(f"Second column: {tensor_2d[:, 1]}")
        
        # 3D tensor indexing
        tensor_3d = torch.randn(2, 3, 4)
        
        print(f"\n3D tensor shape: {tensor_3d.shape}")
        print(f"Element [0, 1, 2]: {tensor_3d[0, 1, 2]}")
        print(f"First matrix:\n{tensor_3d[0]}")
        print(f"Middle row of each matrix:\n{tensor_3d[:, 1, :]}")
    
    def slicing_operations(self):
        """Comprehensive slicing operations"""
        print(f"\n=== SLICING OPERATIONS ===")
        
        # 1D slicing
        tensor_1d = torch.arange(10)
        
        print(f"Original: {tensor_1d}")
        print(f"First 5: {tensor_1d[:5]}")
        print(f"Last 3: {tensor_1d[-3:]}")
        print(f"Every 2nd: {tensor_1d[::2]}")
        print(f"Reverse: {tensor_1d[::-1]}")
        print(f"Middle slice: {tensor_1d[2:7]}")
        
        # 2D slicing
        tensor_2d = torch.arange(20).reshape(4, 5)
        
        print(f"\n2D tensor:\n{tensor_2d}")
        print(f"Top-left 2x3:\n{tensor_2d[:2, :3]}")
        print(f"Bottom-right 2x2:\n{tensor_2d[-2:, -2:]}")
        print(f"Every other row:\n{tensor_2d[::2, :]}")
        print(f"Every other column:\n{tensor_2d[:, ::2]}")
        print(f"Border elements:\n{tensor_2d[[0, -1], :]}")
        
        # Advanced slicing patterns
        print(f"\nAdvanced slicing:")
        print(f"Diagonal elements: {torch.diag(tensor_2d[:4, :4])}")
        
        # Negative step slicing
        print(f"Reversed rows:\n{tensor_2d[::-1, :]}")
        print(f"Reversed columns:\n{tensor_2d[:, ::-1]}")
        print(f"Fully reversed:\n{tensor_2d[::-1, ::-1]}")
    
    def advanced_indexing(self):
        """Advanced indexing with boolean masks and fancy indexing"""
        print(f"\n=== ADVANCED INDEXING ===")
        
        # Boolean indexing
        data = torch.randn(10)
        print(f"Original data: {data}")
        
        # Boolean masks
        positive_mask = data > 0
        large_mask = torch.abs(data) > 1
        
        print(f"Positive mask: {positive_mask}")
        print(f"Positive values: {data[positive_mask]}")
        print(f"Large magnitude values: {data[large_mask]}")
        
        # Combined conditions
        combined_mask = (data > 0) & (torch.abs(data) < 1)
        print(f"Small positive values: {data[combined_mask]}")
        
        # Fancy indexing with integer tensors
        tensor_2d = torch.arange(20).reshape(4, 5)
        
        # Select specific rows
        row_indices = torch.tensor([0, 2, 3])
        selected_rows = tensor_2d[row_indices]
        print(f"\nOriginal 2D:\n{tensor_2d}")
        print(f"Selected rows [0,2,3]:\n{selected_rows}")
        
        # Select specific elements
        row_idx = torch.tensor([0, 1, 2, 3])
        col_idx = torch.tensor([1, 2, 3, 4])
        diagonal_like = tensor_2d[row_idx, col_idx]
        print(f"Diagonal-like elements: {diagonal_like}")
        
        # Advanced boolean indexing on 2D
        large_elements_mask = tensor_2d > 10
        large_elements = tensor_2d[large_elements_mask]
        print(f"Elements > 10: {large_elements}")
        
        # Replace elements using boolean indexing
        modified = tensor_2d.clone()
        modified[modified < 5] = -1
        print(f"Modified (< 5 → -1):\n{modified}")
    
    def conditional_indexing(self):
        """Conditional indexing and selection operations"""
        print(f"\n=== CONDITIONAL INDEXING ===")
        
        # torch.where for conditional selection
        x = torch.randn(3, 4)
        print(f"Original tensor:\n{x}")
        
        # Replace negative values with zeros
        result1 = torch.where(x > 0, x, torch.zeros_like(x))
        print(f"Replace negative with zeros:\n{result1}")
        
        # Clamp positive values to 0.5
        result2 = torch.where(x > 0.5, 0.5, x)
        print(f"Clamp positive to 0.5:\n{result2}")
        
        # Multiple conditions
        condition1 = x > 0
        condition2 = x < 0.5
        combined_condition = condition1 & condition2
        
        result3 = torch.where(combined_condition, x, torch.full_like(x, -999))
        print(f"Keep only small positive:\n{result3}")
        
        # torch.masked_select for extraction
        positive_values = torch.masked_select(x, x > 0)
        print(f"All positive values: {positive_values}")
        
        # nonzero indices
        nonzero_indices = torch.nonzero(x > 0, as_tuple=False)
        nonzero_tuple = torch.nonzero(x > 0, as_tuple=True)
        
        print(f"Nonzero indices:\n{nonzero_indices}")
        print(f"Nonzero as tuple: {nonzero_tuple}")
        
        # Find specific patterns
        target_value = 0.0
        close_to_target = torch.abs(x - target_value) < 0.1
        print(f"Close to zero:\n{close_to_target}")
        
        # Topk and sorting-based selection
        flat_x = x.flatten()
        top3_values, top3_indices = torch.topk(flat_x, 3)
        bottom3_values, bottom3_indices = torch.topk(flat_x, 3, largest=False)
        
        print(f"Top 3 values: {top3_values}")
        print(f"Top 3 indices: {top3_indices}")
        print(f"Bottom 3 values: {bottom3_values}")
        print(f"Bottom 3 indices: {bottom3_indices}")

# Demonstrate indexing techniques
indexer = IndexingMaster()
indexer.basic_indexing()
indexer.slicing_operations()
indexer.advanced_indexing()
indexer.conditional_indexing()
```

---

## 4. Key Questions and Answers

### Beginner Level Questions

**Q1: What's the difference between a Python list and a PyTorch tensor?**
**A:** Key differences include:
- **Data type homogeneity:** Tensors store elements of the same data type, lists can store mixed types
- **Performance:** Tensors use optimized C++ backend, much faster for numerical operations
- **GPU support:** Tensors can be moved to GPU for parallel computation, lists cannot
- **Broadcasting:** Tensors support automatic broadcasting for operations between different shapes
- **Gradient computation:** Tensors can track gradients for automatic differentiation
- **Memory layout:** Tensors have contiguous memory layout optimized for vectorized operations

**Q2: How do I check if a tensor operation will work before running it?**
**A:** Several ways to verify compatibility:
```python
# Shape compatibility for operations
def check_operation_compatibility(a, b, operation):
    if operation == 'add':
        # Check broadcasting compatibility
        return a.shape == b.shape or can_broadcast(a.shape, b.shape)
    elif operation == 'matmul':
        # Check matrix multiplication compatibility
        return a.shape[-1] == b.shape[-2]
    
# Check device compatibility
def same_device(a, b):
    return a.device == b.device

# Check data type compatibility
def compatible_dtypes(a, b):
    return a.dtype == b.dtype or torch.can_cast(a.dtype, b.dtype)
```

**Q3: What does it mean for a tensor to be "contiguous" and why does it matter?**
**A:** Contiguity refers to memory layout:
- **Contiguous:** Elements are stored sequentially in memory following row-major (C-style) order
- **Non-contiguous:** Elements are stored with gaps or in different order (e.g., after transpose)
- **Why it matters:** Some operations (like `.view()`) require contiguous tensors for efficiency
- **Solution:** Use `.contiguous()` to create a contiguous copy when needed

**Q4: How do I create a tensor with specific initialization for deep learning?**
**A:** Common initialization patterns:
```python
# Xavier/Glorot initialization
weight = torch.randn(out_features, in_features)
torch.nn.init.xavier_normal_(weight)

# Kaiming/He initialization (for ReLU)
weight = torch.randn(out_features, in_features)
torch.nn.init.kaiming_normal_(weight, nonlinearity='relu')

# Zero initialization for biases
bias = torch.zeros(out_features)

# Custom initialization
weight = torch.randn(shape) * 0.01  # Small random values
```

### Intermediate Level Questions

**Q5: How does broadcasting work in PyTorch and what are its rules?**
**A:** Broadcasting rules (applied from rightmost dimension):
1. **Dimension alignment:** Start from the trailing dimensions and work forward
2. **Size compatibility:** Dimensions are compatible if they are equal, or one of them is 1
3. **Dimension extension:** If tensors have different number of dimensions, the smaller one is padded with 1s on the left

```python
# Examples
a = torch.randn(3, 1, 4)  # Shape: [3, 1, 4]
b = torch.randn(1, 5, 1)  # Shape: [1, 5, 1]
# Result shape: [3, 5, 4]

# Broadcasting allows efficient operations without copying data
result = a + b  # No actual copying, just different indexing
```

**Q6: What's the difference between `.reshape()`, `.view()`, and `.flatten()`?**
**A:** Different tensor reshaping methods:
- **`.view()`:** Creates a new tensor that shares storage with original. Requires contiguous tensor. Fails if reshaping isn't possible without copying data.
- **`.reshape()`:** More flexible than view. Returns a view if possible, otherwise returns a copy. Always succeeds if the total number of elements matches.
- **`.flatten()`:** Specific case of reshaping to 1D. Can specify start and end dimensions to flatten.

```python
# Practical differences
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)  # Makes non-contiguous

# view() fails on non-contiguous
try:
    z = y.view(-1)  # Error!
except:
    z = y.contiguous().view(-1)  # Works after making contiguous

# reshape() handles non-contiguous automatically
z = y.reshape(-1)  # Always works
```

**Q7: How do I efficiently move tensors between CPU and GPU?**
**A:** Efficient device management strategies:
```python
# Best practices for device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Non-blocking transfer (when possible)
cpu_tensor = torch.randn(1000, 1000)
gpu_tensor = cpu_tensor.to(device, non_blocking=True)

# Batch transfers
def move_batch_to_device(tensors, device):
    return [t.to(device, non_blocking=True) for t in tensors]

# Keep frequently used tensors on GPU
model = model.to(device)  # Move model once
for batch in dataloader:
    batch = batch.to(device)  # Move data as needed
```

### Advanced Level Questions

**Q8: How can I create custom tensor operations that are memory-efficient?**
**A:** Advanced memory-efficient patterns:
```python
# In-place operations when possible
def efficient_activation(x):
    # Instead of: return torch.relu(x) + 1
    x = torch.relu_(x)  # In-place ReLU
    x.add_(1)           # In-place addition
    return x

# Memory-efficient tensor creation
def create_large_tensor_efficiently(shape, init_value=0):
    # Instead of creating then filling
    # tensor = torch.empty(shape).fill_(init_value)
    
    # Direct creation with value
    return torch.full(shape, init_value)

# Use context managers for temporary computations
def memory_efficient_computation(x):
    with torch.no_grad():  # Disable gradient computation
        temp = x.pow(2).sum(dim=-1, keepdim=True)
        return x / temp.sqrt()
```

**Q9: How do I implement custom tensor indexing patterns for specialized use cases?**
**A:** Advanced indexing techniques:
```python
def advanced_indexing_patterns(tensor):
    # Gather operation (advanced indexing)
    indices = torch.tensor([[0, 1], [2, 0]])
    gathered = torch.gather(tensor, 1, indices)
    
    # Scatter operation (inverse of gather)
    scattered = torch.zeros_like(tensor)
    scattered.scatter_(1, indices, gathered)
    
    # Advanced boolean indexing
    def conditional_replace(x, condition_func, replacement_func):
        mask = condition_func(x)
        x[mask] = replacement_func(x[mask])
        return x
    
    # Multi-dimensional advanced indexing
    def extract_diagonal_blocks(matrix, block_size):
        n = matrix.size(0)
        blocks = []
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            blocks.append(matrix[i:end_i, i:end_i])
        return blocks
```

**Q10: What are the performance implications of different tensor operations?**
**A:** Performance characteristics of tensor operations:

**Memory bandwidth bound operations:**
- Element-wise operations (add, multiply, etc.)
- Broadcasting operations
- Most activation functions

**Compute bound operations:**
- Matrix multiplication
- Convolutions
- Complex mathematical functions

**Memory layout sensitive operations:**
- Transpose operations
- Reshaping and viewing
- Strided operations

```python
# Performance optimization strategies
def optimize_tensor_operations():
    # 1. Minimize data movement
    x = torch.randn(1000, 1000, device='cuda')
    # Good: keep intermediate results on GPU
    y = x.pow(2).sum(dim=1).sqrt()
    
    # 2. Use in-place operations when safe
    x.pow_(2)  # Modifies x in place
    
    # 3. Batch operations when possible
    # Instead of: [torch.matmul(a, b) for a, b in zip(As, Bs)]
    # Use: torch.bmm(torch.stack(As), torch.stack(Bs))
    
    # 4. Use appropriate data types
    # float16 for inference, float32 for training
    # int64 only when necessary (int32 often sufficient)
```

---

## 5. Tricky Questions for Deep Understanding

### Memory and Performance Paradoxes

**Q1: Why might a "simpler" tensor operation sometimes be slower than a complex one?**
**A:** This reveals the complexity of modern hardware optimization:

**Memory hierarchy effects:**
```python
# Counter-intuitive example
large_tensor = torch.randn(10000, 10000)

# Simple operation but memory-bound
simple_result = large_tensor + 1  # Reads entire tensor, adds 1, writes back

# Complex operation but compute-efficient
complex_result = torch.mm(large_tensor[:1000, :1000], 
                         large_tensor[:1000, :1000].T)  # Matrix multiply on subset

# The matrix multiplication might be faster because:
# 1. Better cache utilization on smaller data
# 2. GPU compute units fully utilized
# 3. Memory access patterns optimized for matrix ops
```

**Hidden factors:**
- **Memory bandwidth:** Simple ops often memory-bound, not compute-bound
- **Cache efficiency:** Smaller working sets may run faster
- **Vectorization:** Some complex ops better utilize SIMD/GPU parallelism
- **Kernel fusion:** Multiple simple ops may require multiple memory passes

**Q2: How can creating more tensor objects sometimes use less memory?**
**A:** This paradox highlights the difference between storage and views:

```python
# Paradox example
large_data = torch.randn(10000, 10000)  # 400MB of data

# Scenario 1: Keep reference to full tensor
subset1 = large_data[:100, :100]  # Still references full 400MB storage
result1 = subset1 * 2
# Total memory: 400MB (original) + 0.04MB (result) = 400.04MB

# Scenario 2: Clone subset (more objects, less memory)
subset2 = large_data[:100, :100].clone()  # Only 0.04MB storage
del large_data  # Original can be freed
result2 = subset2 * 2
# Total memory: 0.04MB + 0.04MB = 0.08MB
```

**Key insights:**
- **Storage vs. tensor objects:** Multiple tensor objects can share storage
- **Memory lifecycle:** Views keep entire storage alive
- **Cloning strategy:** Sometimes copying small parts and releasing large storage saves memory

### Broadcasting and Shape Paradoxes

**Q3: Why do some "mathematically equivalent" broadcasting operations have different performance?**
**A:** Broadcasting implementation affects performance:

```python
# These are mathematically equivalent but perform differently
a = torch.randn(1000, 1)      # Column vector
b = torch.randn(1, 1000)      # Row vector

# Operation 1: a + b (broadcast both)
result1 = a + b  # Creates 1000x1000 result

# Operation 2: Explicit expansion
a_expanded = a.expand(1000, 1000)
b_expanded = b.expand(1000, 1000)
result2 = a_expanded + b_expanded

# Operation 3: Using outer product concept
result3 = a.mm(torch.ones(1, 1000)) + torch.ones(1000, 1).mm(b)
```

**Performance differences:**
- **Memory access patterns:** Different broadcasting directions affect cache performance
- **Kernel selection:** Different shapes may trigger different optimized kernels
- **Memory allocation:** Temporary storage patterns differ between approaches

**Q4: How can understanding tensor strides unlock significant optimizations?**
**A:** Strides reveal the true memory layout and enable optimizations:

```python
def analyze_stride_performance():
    # Create a large matrix
    matrix = torch.randn(1000, 1000)
    
    # Different ways to access the same data
    print(f"Original stride: {matrix.stride()}")  # (1000, 1)
    
    # Row-major access (efficient)
    row_sum = matrix.sum(dim=1)  # Sum along rows (stride 1 dimension)
    
    # Column-major access (less efficient)  
    col_sum = matrix.sum(dim=0)  # Sum along columns (stride 1000 dimension)
    
    # Transposed version
    matrix_t = matrix.t()
    print(f"Transposed stride: {matrix_t.stride()}")  # (1, 1000)
    
    # Now column access on transposed is efficient
    col_sum_t = matrix_t.sum(dim=0)  # Equivalent to original row sum
    
    # Custom strided operations
    # Extract every other element in a memory-efficient way
    strided_view = matrix[::2, ::2]  # Creates view with different strides
    print(f"Strided view stride: {strided_view.stride()}")
```

### Data Type and Precision Paradoxes

**Q5: When might lower precision arithmetic give more accurate results?**
**A:** This counterintuitive situation occurs in several scenarios:

**Numerical stability:**
```python
def precision_paradox_example():
    # Example: Computing softmax with very large values
    large_values = torch.tensor([1000.0, 1001.0, 1002.0])
    
    # Standard float32 computation
    exp_values_f32 = torch.exp(large_values)  # May overflow to inf
    softmax_f32 = exp_values_f32 / exp_values_f32.sum()
    
    # Mixed precision approach
    # Subtract max for stability (this is what real softmax does)
    stable_values = large_values - large_values.max()
    exp_stable = torch.exp(stable_values)
    softmax_stable = exp_stable / exp_stable.sum()
    
    print(f"Unstable softmax: {softmax_f32}")
    print(f"Stable softmax: {softmax_stable}")
```

**Scenarios where lower precision helps:**
- **Regularization effect:** Lower precision adds noise that can prevent overfitting
- **Numerical stability:** Some algorithms are more stable with limited precision
- **Hardware optimization:** Lower precision enables faster, more parallel computation
- **Memory bandwidth:** Less memory traffic allows more compute per unit time

**Q6: How do tensor creation methods affect subsequent operation performance?**
**A:** Creation method determines memory layout and affects all future operations:

```python
def creation_performance_impact():
    # Method 1: From list (potentially non-optimal layout)
    from_list = torch.tensor([[i + j for j in range(1000)] for i in range(1000)])
    
    # Method 2: Direct allocation then filling
    direct = torch.zeros(1000, 1000)
    for i in range(1000):
        direct[i] = torch.arange(i, i + 1000)
    
    # Method 3: Vectorized creation
    vectorized = torch.arange(0, 1000)[:, None] + torch.arange(0, 1000)[None, :]
    
    # Performance test: matrix multiplication
    import time
    
    def time_operation(tensor, name):
        start = time.time()
        result = torch.mm(tensor.float(), tensor.float().t())
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        print(f"{name}: {end - start:.4f}s")
        return result
    
    # Test performance (vectorized creation often fastest for subsequent ops)
    time_operation(from_list.float(), "From list")
    time_operation(direct, "Direct allocation")
    time_operation(vectorized.float(), "Vectorized")
```

---

## Summary and Best Practices

### Tensor Creation Decision Tree

**For beginners:**
1. **From Python data:** Use `torch.tensor()` for small data, lists, and prototyping
2. **Initialization:** Use `torch.zeros()`, `torch.ones()`, `torch.randn()` for clean initialization
3. **Device placement:** Always specify device at creation time when possible
4. **Data types:** Use default types unless you have specific precision requirements

### Performance Optimization Guidelines

**Memory efficiency:**
- Use in-place operations when safe (`add_()`, `mul_()`, etc.)
- Be aware of storage sharing vs. copying
- Use appropriate data types (float16 for inference, int32 instead of int64 when possible)
- Consider memory layout (contiguous vs. non-contiguous)

**Computational efficiency:**
- Batch operations when possible
- Minimize device transfers
- Use broadcasting instead of explicit expansion
- Understand when operations are memory-bound vs. compute-bound

### Common Pitfalls and Solutions

**Shape-related issues:**
- Always check tensor shapes before operations
- Understand broadcasting rules thoroughly
- Use `.view()` vs. `.reshape()` appropriately
- Be careful with dimension ordering in different frameworks

**Device and memory issues:**
- Ensure all tensors in an operation are on the same device
- Monitor GPU memory usage in training loops
- Use `.detach()` and `torch.no_grad()` to prevent unwanted gradient computation
- Clear variables and call `torch.cuda.empty_cache()` when running out of GPU memory

Understanding tensors thoroughly is fundamental to effective PyTorch usage. These concepts form the foundation for all neural network operations, data processing, and optimization algorithms you'll encounter in deep learning.

---

## Next Steps

In the next module, we'll dive into advanced tensor operations, mathematical functions, linear algebra operations, and broadcasting patterns that are essential for implementing neural network computations and custom deep learning algorithms.