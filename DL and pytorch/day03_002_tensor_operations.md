# Day 3.2: Advanced Tensor Operations and Mathematical Functions

## Course: Comprehensive Deep Learning with PyTorch - 45-Day Masterclass
### Day 3, Part 2: Mathematical Operations, Broadcasting, and Linear Algebra

---

## Overview

This module covers advanced tensor operations that form the computational backbone of deep learning. We'll explore mathematical functions, linear algebra operations, broadcasting mechanics, reduction operations, and advanced manipulation techniques essential for implementing neural networks and deep learning algorithms.

## Learning Objectives

By the end of this module, you will:
- Master broadcasting rules and implement efficient tensor operations
- Perform comprehensive linear algebra operations for deep learning
- Understand and apply reduction operations across different dimensions
- Implement advanced tensor manipulation techniques
- Optimize tensor operations for performance and memory efficiency

---

## 1. Broadcasting Deep Dive

### 1.1 Broadcasting Mechanics and Rules

#### Understanding Broadcasting Rules

**Comprehensive Broadcasting Analysis:**
```python
import torch
import numpy as np

class BroadcastingMaster:
    """Deep dive into PyTorch broadcasting mechanics"""
    
    def __init__(self):
        self.examples = self._create_broadcast_examples()
    
    def _create_broadcast_examples(self):
        """Create comprehensive broadcasting examples"""
        return {
            'scalar_vector': (torch.tensor(5), torch.randn(4)),
            'vector_matrix': (torch.randn(4), torch.randn(3, 4)),
            'matrix_3d': (torch.randn(3, 4), torch.randn(2, 3, 4)),
            'complex_shapes': (torch.randn(1, 3, 1), torch.randn(2, 1, 4)),
            'edge_cases': (torch.randn(1, 1), torch.randn(5, 4, 3, 2))
        }
    
    def demonstrate_broadcasting_rules(self):
        """Demonstrate broadcasting rules with detailed analysis"""
        print("=== BROADCASTING RULES DEMONSTRATION ===")
        
        def analyze_broadcast(a, b, operation_name):
            """Analyze a broadcasting operation step by step"""
            print(f"\n{operation_name}:")
            print(f"  Tensor A shape: {a.shape}")
            print(f"  Tensor B shape: {b.shape}")
            
            # Check if broadcasting is possible
            try:
                result = a + b
                print(f"  Result shape: {result.shape}")
                print(f"  Broadcasting: ✓ Success")
                
                # Analyze which dimensions were broadcasted
                self._analyze_broadcast_pattern(a.shape, b.shape, result.shape)
                
            except RuntimeError as e:
                print(f"  Broadcasting: ✗ Failed - {str(e)[:50]}...")
        
        # Test all examples
        for name, (a, b) in self.examples.items():
            analyze_broadcast(a, b, name)
    
    def _analyze_broadcast_pattern(self, shape_a, shape_b, result_shape):
        """Analyze which dimensions were broadcasted"""
        print(f"  Broadcasting analysis:")
        
        # Reverse shapes for right-to-left analysis
        rev_a = list(reversed(shape_a))
        rev_b = list(reversed(shape_b))
        rev_result = list(reversed(result_shape))
        
        # Pad shorter shapes with 1s
        max_len = len(rev_result)
        rev_a += [1] * (max_len - len(rev_a))
        rev_b += [1] * (max_len - len(rev_b))
        
        for i, (dim_a, dim_b, dim_result) in enumerate(zip(rev_a, rev_b, rev_result)):
            dimension_name = f"dim -{i+1}" if i < len(rev_result) else f"dim {len(rev_result)-i-1}"
            if dim_a == 1 and dim_b != 1:
                print(f"    {dimension_name}: A broadcasted from {dim_a} to {dim_result}")
            elif dim_a != 1 and dim_b == 1:
                print(f"    {dimension_name}: B broadcasted from {dim_b} to {dim_result}")
            elif dim_a == dim_b:
                print(f"    {dimension_name}: No broadcasting needed ({dim_a})")
    
    def manual_broadcasting_implementation(self):
        """Implement manual broadcasting to understand the mechanism"""
        print(f"\n=== MANUAL BROADCASTING IMPLEMENTATION ===")
        
        def manual_broadcast_shapes(shape_a, shape_b):
            """Manually compute broadcast result shape"""
            # Reverse for right-to-left processing
            rev_a = list(reversed(shape_a))
            rev_b = list(reversed(shape_b))
            
            # Determine result length
            result_len = max(len(rev_a), len(rev_b))
            
            # Pad with 1s
            rev_a += [1] * (result_len - len(rev_a))
            rev_b += [1] * (result_len - len(rev_b))
            
            # Compute result shape
            result_shape = []
            for dim_a, dim_b in zip(rev_a, rev_b):
                if dim_a == 1:
                    result_shape.append(dim_b)
                elif dim_b == 1:
                    result_shape.append(dim_a)
                elif dim_a == dim_b:
                    result_shape.append(dim_a)
                else:
                    raise ValueError(f"Cannot broadcast dimensions {dim_a} and {dim_b}")
            
            return tuple(reversed(result_shape))
        
        # Test manual implementation
        test_cases = [
            ((3, 1, 4), (1, 5, 1)),
            ((2, 1), (3, 1, 4)),
            ((5, 4), (1,)),
            ((3, 4, 5), (4, 1)),
        ]
        
        for shape_a, shape_b in test_cases:
            try:
                manual_result = manual_broadcast_shapes(shape_a, shape_b)
                
                # Verify with PyTorch
                a = torch.randn(shape_a)
                b = torch.randn(shape_b)
                torch_result = (a + b).shape
                
                match = manual_result == torch_result
                print(f"Shapes {shape_a} + {shape_b}:")
                print(f"  Manual: {manual_result}")
                print(f"  PyTorch: {torch_result}")
                print(f"  Match: {'✓' if match else '✗'}")
                
            except ValueError as e:
                print(f"Shapes {shape_a} + {shape_b}: {e}")
    
    def broadcasting_performance_analysis(self):
        """Analyze performance implications of broadcasting"""
        print(f"\n=== BROADCASTING PERFORMANCE ANALYSIS ===")
        
        import time
        
        # Create test tensors
        large_matrix = torch.randn(1000, 1000)
        row_vector = torch.randn(1, 1000)
        col_vector = torch.randn(1000, 1)
        scalar = torch.randn(1)
        
        def time_operation(func, name, iterations=100):
            """Time a broadcasting operation"""
            start = time.time()
            for _ in range(iterations):
                result = func()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end = time.time()
            avg_time = (end - start) / iterations * 1000  # ms
            print(f"  {name}: {avg_time:.3f}ms average")
            return result
        
        print("Performance comparison of different broadcasting patterns:")
        
        # Test different broadcasting scenarios
        time_operation(lambda: large_matrix + scalar, "Matrix + Scalar")
        time_operation(lambda: large_matrix + row_vector, "Matrix + Row Vector")
        time_operation(lambda: large_matrix + col_vector, "Matrix + Column Vector")
        time_operation(lambda: large_matrix + large_matrix, "Matrix + Matrix (no broadcast)")
        
        # Memory usage comparison
        print(f"\nMemory usage analysis:")
        print(f"  Large matrix: {large_matrix.numel() * large_matrix.element_size() / 1e6:.1f} MB")
        print(f"  Row vector: {row_vector.numel() * row_vector.element_size() / 1e6:.3f} MB")
        print(f"  Broadcasting saves: {(large_matrix.numel() - row_vector.numel()) * row_vector.element_size() / 1e6:.1f} MB")

# Demonstrate broadcasting
broadcaster = BroadcastingMaster()
broadcaster.demonstrate_broadcasting_rules()
broadcaster.manual_broadcasting_implementation()
broadcaster.broadcasting_performance_analysis()
```

#### Advanced Broadcasting Patterns

**Complex Broadcasting Scenarios:**
```python
class AdvancedBroadcastingPatterns:
    """Advanced broadcasting patterns for deep learning"""
    
    def neural_network_broadcasting(self):
        """Broadcasting patterns common in neural networks"""
        print("=== NEURAL NETWORK BROADCASTING PATTERNS ===")
        
        # Batch processing pattern
        batch_size, input_dim, hidden_dim = 32, 128, 256
        
        # Input batch
        inputs = torch.randn(batch_size, input_dim)
        print(f"Input batch shape: {inputs.shape}")
        
        # Weight matrix
        weights = torch.randn(input_dim, hidden_dim)
        print(f"Weight matrix shape: {weights.shape}")
        
        # Bias vector
        bias = torch.randn(hidden_dim)
        print(f"Bias vector shape: {bias.shape}")
        
        # Forward pass computation
        # Linear transformation: batch_inputs @ weights
        linear_output = torch.mm(inputs, weights)  # [batch_size, hidden_dim]
        print(f"Linear output shape: {linear_output.shape}")
        
        # Add bias (broadcasting)
        output_with_bias = linear_output + bias  # bias broadcasts to [batch_size, hidden_dim]
        print(f"Output with bias shape: {output_with_bias.shape}")
        
        # Alternative: einsum for clarity
        einsum_output = torch.einsum('bi,ih->bh', inputs, weights) + bias
        print(f"Einsum output shape: {einsum_output.shape}")
        
        # Batch normalization pattern
        # Statistics computed per feature across batch
        mean = output_with_bias.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        var = output_with_bias.var(dim=0, keepdim=True)    # [1, hidden_dim]
        
        # Normalization (broadcasting)
        normalized = (output_with_bias - mean) / (var + 1e-5).sqrt()
        print(f"Normalized shape: {normalized.shape}")
        
    def attention_broadcasting(self):
        """Broadcasting in attention mechanisms"""
        print(f"\n=== ATTENTION MECHANISM BROADCASTING ===")
        
        batch_size, seq_len, d_model = 16, 128, 512
        
        # Query, Key, Value tensors
        queries = torch.randn(batch_size, seq_len, d_model)
        keys = torch.randn(batch_size, seq_len, d_model)
        values = torch.randn(batch_size, seq_len, d_model)
        
        print(f"Q, K, V shapes: {queries.shape}")
        
        # Scaled dot-product attention
        # Attention scores: Q @ K.T
        scores = torch.bmm(queries, keys.transpose(-2, -1))  # [B, seq_len, seq_len]
        scores = scores / (d_model ** 0.5)  # Scale (broadcasting)
        
        print(f"Attention scores shape: {scores.shape}")
        
        # Softmax along last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_output = torch.bmm(attention_weights, values)  # [B, seq_len, d_model]
        
        print(f"Attended output shape: {attended_output.shape}")
        
        # Multi-head attention broadcasting
        num_heads = 8
        head_dim = d_model // num_heads
        
        # Reshape for multi-head
        q_heads = queries.view(batch_size, seq_len, num_heads, head_dim)
        q_heads = q_heads.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        
        k_heads = keys.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v_heads = values.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        print(f"Multi-head Q shape: {q_heads.shape}")
        
        # Multi-head attention computation
        head_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        head_scores = head_scores / (head_dim ** 0.5)  # Broadcasting scale
        
        head_attention = torch.softmax(head_scores, dim=-1)
        head_output = torch.matmul(head_attention, v_heads)
        
        # Reshape back
        multi_head_output = head_output.transpose(1, 2).contiguous()
        multi_head_output = multi_head_output.view(batch_size, seq_len, d_model)
        
        print(f"Multi-head output shape: {multi_head_output.shape}")
    
    def convolutional_broadcasting(self):
        """Broadcasting patterns in convolutional operations"""
        print(f"\n=== CONVOLUTIONAL BROADCASTING ===")
        
        batch_size, channels, height, width = 16, 64, 32, 32
        
        # Feature maps
        feature_maps = torch.randn(batch_size, channels, height, width)
        print(f"Feature maps shape: {feature_maps.shape}")
        
        # Channel-wise operations (broadcasting)
        # Different per channel
        channel_scales = torch.randn(1, channels, 1, 1)
        scaled_features = feature_maps * channel_scales  # Broadcast to all spatial locations
        
        print(f"Channel scales shape: {channel_scales.shape}")
        print(f"Scaled features shape: {scaled_features.shape}")
        
        # Spatial broadcasting
        # Add positional encoding
        height_encoding = torch.arange(height).float().view(1, 1, height, 1)
        width_encoding = torch.arange(width).float().view(1, 1, 1, width)
        
        position_encoded = (feature_maps + 
                          height_encoding + width_encoding)  # Multiple broadcasts
        
        print(f"Position encoded shape: {position_encoded.shape}")
        
        # Global operations with broadcasting
        # Global average pooling equivalent
        global_mean = feature_maps.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # Broadcast back to spatial dimensions
        mean_subtracted = feature_maps - global_mean
        
        print(f"Global mean shape: {global_mean.shape}")
        print(f"Mean subtracted shape: {mean_subtracted.shape}")
    
    def advanced_indexing_with_broadcasting(self):
        """Combine advanced indexing with broadcasting"""
        print(f"\n=== ADVANCED INDEXING + BROADCASTING ===")
        
        # Create sample data
        data = torch.randn(100, 50)  # [samples, features]
        
        # Select specific samples and features
        sample_indices = torch.tensor([0, 5, 10, 15, 20])
        feature_indices = torch.tensor([1, 3, 5, 7])
        
        # Advanced indexing creates [5, 4] tensor
        selected_data = data[sample_indices][:, feature_indices]
        print(f"Selected data shape: {selected_data.shape}")
        
        # Broadcasting with selection
        # Add different bias to each selected feature
        feature_bias = torch.tensor([0.1, 0.2, 0.3, 0.4])  # [4]
        biased_data = selected_data + feature_bias  # Broadcast to [5, 4]
        
        print(f"Biased data shape: {biased_data.shape}")
        
        # Complex indexing with broadcasting
        # Create attention-like pattern
        batch_size, seq_len = 8, 16
        attention_tensor = torch.randn(batch_size, seq_len, seq_len)
        
        # Create causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Apply mask (broadcasting)
        masked_attention = attention_tensor.masked_fill(causal_mask, float('-inf'))
        
        print(f"Attention shape: {attention_tensor.shape}")
        print(f"Mask shape: {causal_mask.shape}")
        print(f"Masked attention shape: {masked_attention.shape}")

# Demonstrate advanced broadcasting
advanced_broadcaster = AdvancedBroadcastingPatterns()
advanced_broadcaster.neural_network_broadcasting()
advanced_broadcaster.attention_broadcasting()
advanced_broadcaster.convolutional_broadcasting()
advanced_broadcaster.advanced_indexing_with_broadcasting()
```

### 1.2 Broadcasting Optimization Techniques

#### Memory-Efficient Broadcasting

**Optimized Broadcasting Patterns:**
```python
class BroadcastingOptimization:
    """Memory and performance optimization for broadcasting operations"""
    
    def memory_efficient_broadcasting(self):
        """Demonstrate memory-efficient broadcasting techniques"""
        print("=== MEMORY-EFFICIENT BROADCASTING ===")
        
        # Problem: Large tensor operations with broadcasting
        large_shape = (1000, 1000)
        small_shape = (1, 1000)
        
        large_tensor = torch.randn(large_shape)
        small_tensor = torch.randn(small_shape)
        
        print(f"Large tensor memory: {large_tensor.numel() * large_tensor.element_size() / 1e6:.1f} MB")
        print(f"Small tensor memory: {small_tensor.numel() * small_tensor.element_size() / 1e6:.1f} MB")
        
        # Method 1: Direct broadcasting (memory efficient)
        result1 = large_tensor + small_tensor
        print(f"Method 1 - Direct broadcast result shape: {result1.shape}")
        
        # Method 2: Explicit expansion (memory inefficient)
        expanded_small = small_tensor.expand_as(large_tensor)
        print(f"Expanded tensor shares memory: {expanded_small.data_ptr() == small_tensor.data_ptr()}")
        
        # Method 3: Using expand vs repeat
        repeated_small = small_tensor.repeat(large_shape[0], 1)
        print(f"Repeated tensor shares memory: {repeated_small.data_ptr() == small_tensor.data_ptr()}")
        print(f"Repeated tensor memory: {repeated_small.numel() * repeated_small.element_size() / 1e6:.1f} MB")
        
        # Best practice: Use broadcasting directly
        # PyTorch optimizes this internally
        def efficient_broadcast_operation(large, small):
            """Memory-efficient broadcasting operation"""
            return large + small  # Let PyTorch handle broadcasting
        
        def inefficient_broadcast_operation(large, small):
            """Memory-inefficient manual expansion"""
            return large + small.expand_as(large)
        
        # Both give same result, but first is more memory efficient
        efficient_result = efficient_broadcast_operation(large_tensor, small_tensor)
        inefficient_result = inefficient_broadcast_operation(large_tensor, small_tensor)
        
        print(f"Results equal: {torch.allclose(efficient_result, inefficient_result)}")
    
    def in_place_broadcasting(self):
        """In-place broadcasting operations for memory efficiency"""
        print(f"\n=== IN-PLACE BROADCASTING ===")
        
        # Original tensors
        matrix = torch.randn(100, 50)
        bias = torch.randn(50)
        
        print(f"Original matrix id: {id(matrix)}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Bias shape: {bias.shape}")
        
        # In-place addition with broadcasting
        matrix.add_(bias)  # Modifies matrix in-place
        
        print(f"After in-place add id: {id(matrix)}")
        print(f"Matrix shape after: {matrix.shape}")
        
        # Demonstrate memory savings
        def compare_memory_usage():
            """Compare memory usage of in-place vs out-of-place operations"""
            large_matrix = torch.randn(1000, 1000)
            small_vector = torch.randn(1000)
            
            # Method 1: Out-of-place (creates new tensor)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                result_new = large_matrix + small_vector
                peak_memory_new = torch.cuda.max_memory_allocated()
            
            # Method 2: In-place (modifies existing tensor)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                large_matrix_copy = large_matrix.clone()
                large_matrix_copy.add_(small_vector)
                peak_memory_inplace = torch.cuda.max_memory_allocated()
                
                print(f"Out-of-place peak memory: {peak_memory_new / 1e6:.1f} MB")
                print(f"In-place peak memory: {peak_memory_inplace / 1e6:.1f} MB")
                print(f"Memory saved: {(peak_memory_new - peak_memory_inplace) / 1e6:.1f} MB")
        
        if torch.cuda.is_available():
            compare_memory_usage()
        else:
            print("CUDA not available - skipping GPU memory comparison")
    
    def chunked_broadcasting(self):
        """Handle very large broadcasting operations in chunks"""
        print(f"\n=== CHUNKED BROADCASTING ===")
        
        def chunked_broadcast_add(large_tensor, small_tensor, chunk_size=1000):
            """Add small tensor to large tensor in chunks to save memory"""
            result = torch.empty_like(large_tensor)
            
            for i in range(0, large_tensor.size(0), chunk_size):
                end_idx = min(i + chunk_size, large_tensor.size(0))
                chunk = large_tensor[i:end_idx]
                result[i:end_idx] = chunk + small_tensor
            
            return result
        
        # Example with very large tensors
        try:
            # This might use too much memory if done all at once
            very_large = torch.randn(10000, 1000)
            small = torch.randn(1, 1000)
            
            print(f"Large tensor shape: {very_large.shape}")
            print(f"Small tensor shape: {small.shape}")
            
            # Process in chunks
            chunked_result = chunked_broadcast_add(very_large, small, chunk_size=2000)
            
            # Verify correctness (on a small subset)
            direct_result = very_large[:100] + small
            chunked_subset = chunked_result[:100]
            
            print(f"Chunked processing correct: {torch.allclose(direct_result, chunked_subset)}")
            
        except RuntimeError as e:
            print(f"Memory error (expected for very large tensors): {str(e)[:50]}...")
    
    def vectorized_broadcasting_patterns(self):
        """Vectorized operations that leverage broadcasting"""
        print(f"\n=== VECTORIZED BROADCASTING PATTERNS ===")
        
        # Pattern 1: Distance computation
        def pairwise_distances(points1, points2):
            """Compute pairwise distances using broadcasting"""
            # points1: [N, D], points2: [M, D]
            # Result: [N, M]
            
            # Expand dimensions for broadcasting
            p1_expanded = points1.unsqueeze(1)  # [N, 1, D]
            p2_expanded = points2.unsqueeze(0)  # [1, M, D]
            
            # Compute squared distances
            distances = ((p1_expanded - p2_expanded) ** 2).sum(dim=2)
            return distances.sqrt()
        
        # Test distance computation
        points_a = torch.randn(100, 3)  # 100 3D points
        points_b = torch.randn(50, 3)   # 50 3D points
        
        distances = pairwise_distances(points_a, points_b)
        print(f"Distance matrix shape: {distances.shape}")
        
        # Pattern 2: Gaussian RBF kernel
        def gaussian_rbf_kernel(x1, x2, bandwidth=1.0):
            """Compute Gaussian RBF kernel matrix"""
            distances_sq = pairwise_distances(x1, x2) ** 2
            return torch.exp(-distances_sq / (2 * bandwidth ** 2))
        
        kernel_matrix = gaussian_rbf_kernel(points_a, points_b)
        print(f"Kernel matrix shape: {kernel_matrix.shape}")
        
        # Pattern 3: Outer product operations
        def batch_outer_product(vectors_a, vectors_b):
            """Compute batch of outer products using broadcasting"""
            # vectors_a: [B, N], vectors_b: [B, M]
            # Result: [B, N, M]
            
            a_expanded = vectors_a.unsqueeze(-1)  # [B, N, 1]
            b_expanded = vectors_b.unsqueeze(-2)  # [B, 1, M]
            
            return a_expanded * b_expanded  # [B, N, M]
        
        # Test batch outer product
        batch_a = torch.randn(32, 10)
        batch_b = torch.randn(32, 8)
        
        outer_products = batch_outer_product(batch_a, batch_b)
        print(f"Batch outer products shape: {outer_products.shape}")
        
        # Verify with manual computation
        manual_outer = torch.bmm(batch_a.unsqueeze(-1), batch_b.unsqueeze(-2))
        print(f"Manual vs broadcast outer product equal: {torch.allclose(outer_products, manual_outer)}")

# Demonstrate broadcasting optimization
optimizer = BroadcastingOptimization()
optimizer.memory_efficient_broadcasting()
optimizer.in_place_broadcasting()
optimizer.chunked_broadcasting()
optimizer.vectorized_broadcasting_patterns()
```

---

## 2. Linear Algebra Operations

### 2.1 Matrix Operations

#### Basic Matrix Operations

**Comprehensive Matrix Operations:**
```python
class LinearAlgebraOperations:
    """Comprehensive linear algebra operations in PyTorch"""
    
    def basic_matrix_operations(self):
        """Basic matrix operations essential for deep learning"""
        print("=== BASIC MATRIX OPERATIONS ===")
        
        # Create sample matrices
        A = torch.randn(4, 3)
        B = torch.randn(3, 5)
        C = torch.randn(4, 4)
        
        print(f"Matrix A shape: {A.shape}")
        print(f"Matrix B shape: {B.shape}")
        print(f"Matrix C shape: {C.shape}")
        
        # Matrix multiplication
        AB = torch.mm(A, B)  # or A @ B
        print(f"A @ B shape: {AB.shape}")
        
        # Batch matrix multiplication
        batch_A = torch.randn(10, 4, 3)
        batch_B = torch.randn(10, 3, 5)
        batch_AB = torch.bmm(batch_A, batch_B)
        print(f"Batch A @ B shape: {batch_AB.shape}")
        
        # Alternative: einsum for complex operations
        einsum_AB = torch.einsum('bij,bjk->bik', batch_A, batch_B)
        print(f"Einsum batch multiplication equal: {torch.allclose(batch_AB, einsum_AB)}")
        
        # Transpose operations
        A_T = A.t()  # 2D transpose
        A_T_alt = A.transpose(0, 1)  # Explicit dimension specification
        
        print(f"A transpose shape: {A_T.shape}")
        print(f"Transpose methods equal: {torch.allclose(A_T, A_T_alt)}")
        
        # Multi-dimensional transpose
        tensor_3d = torch.randn(2, 3, 4)
        transposed_3d = tensor_3d.transpose(1, 2)  # Swap dimensions 1 and 2
        print(f"3D tensor transpose: {tensor_3d.shape} -> {transposed_3d.shape}")
        
        # Matrix power
        matrix_squared = torch.mm(C, C)  # Manual
        matrix_squared_alt = torch.matrix_power(C, 2)  # Built-in
        print(f"Matrix power methods equal: {torch.allclose(matrix_squared, matrix_squared_alt)}")
    
    def matrix_decompositions(self):
        """Matrix decompositions for numerical analysis"""
        print(f"\n=== MATRIX DECOMPOSITIONS ===")
        
        # Create test matrix
        n = 5
        A = torch.randn(n, n)
        symmetric_A = A + A.t()  # Make symmetric for eigendecomposition
        
        print(f"Original matrix shape: {A.shape}")
        
        # SVD - Singular Value Decomposition
        U, S, Vh = torch.svd(A)
        print(f"SVD shapes - U: {U.shape}, S: {S.shape}, V: {Vh.shape}")
        
        # Verify SVD reconstruction
        A_reconstructed = torch.mm(torch.mm(U, torch.diag(S)), Vh.t())
        svd_error = torch.norm(A - A_reconstructed)
        print(f"SVD reconstruction error: {svd_error.item():.2e}")
        
        # Eigendecomposition (for symmetric matrices)
        eigenvalues, eigenvectors = torch.eig(symmetric_A, eigenvectors=True)
        print(f"Eigenvalues shape: {eigenvalues.shape}")
        print(f"Eigenvectors shape: {eigenvectors.shape}")
        
        # QR decomposition
        Q, R = torch.qr(A)
        print(f"QR shapes - Q: {Q.shape}, R: {R.shape}")
        
        # Verify QR reconstruction
        QR_reconstructed = torch.mm(Q, R)
        qr_error = torch.norm(A - QR_reconstructed)
        print(f"QR reconstruction error: {qr_error.item():.2e}")
        
        # Check Q orthogonality
        Q_orthogonality = torch.mm(Q.t(), Q)
        identity_error = torch.norm(Q_orthogonality - torch.eye(n))
        print(f"Q orthogonality error: {identity_error.item():.2e}")
        
        # Cholesky decomposition (for positive definite matrices)
        # Create positive definite matrix
        positive_definite = torch.mm(A, A.t()) + torch.eye(n) * 0.1
        
        try:
            L = torch.cholesky(positive_definite)
            print(f"Cholesky factor shape: {L.shape}")
            
            # Verify Cholesky reconstruction
            chol_reconstructed = torch.mm(L, L.t())
            chol_error = torch.norm(positive_definite - chol_reconstructed)
            print(f"Cholesky reconstruction error: {chol_error.item():.2e}")
            
        except RuntimeError as e:
            print(f"Cholesky failed (matrix not positive definite): {str(e)[:50]}...")
    
    def matrix_norms_and_properties(self):
        """Matrix norms and properties computation"""
        print(f"\n=== MATRIX NORMS AND PROPERTIES ===")
        
        # Create test matrices
        A = torch.randn(4, 5)
        square_matrix = torch.randn(4, 4)
        
        # Vector norms (when matrix is flattened)
        frobenius_norm = torch.norm(A, 'fro')  # Frobenius norm
        l2_norm = torch.norm(A)  # Default is Frobenius for matrices
        l1_norm = torch.norm(A, p=1)
        inf_norm = torch.norm(A, p=float('inf'))
        
        print(f"Matrix A shape: {A.shape}")
        print(f"Frobenius norm: {frobenius_norm.item():.4f}")
        print(f"L2 norm (same as Frobenius): {l2_norm.item():.4f}")
        print(f"L1 norm: {l1_norm.item():.4f}")
        print(f"Infinity norm: {inf_norm.item():.4f}")
        
        # Matrix norms along dimensions
        row_norms = torch.norm(A, dim=1)  # L2 norm of each row
        col_norms = torch.norm(A, dim=0)  # L2 norm of each column
        
        print(f"Row norms shape: {row_norms.shape}")
        print(f"Column norms shape: {col_norms.shape}")
        
        # Matrix properties
        determinant = torch.det(square_matrix)
        trace = torch.trace(square_matrix)
        
        print(f"Square matrix determinant: {determinant.item():.4f}")
        print(f"Square matrix trace: {trace.item():.4f}")
        
        # Matrix rank (using SVD)
        _, S, _ = torch.svd(A)
        rank = torch.sum(S > 1e-7).item()  # Numerical rank with threshold
        print(f"Matrix rank: {rank}")
        
        # Condition number
        condition_number = torch.max(S) / torch.min(S)
        print(f"Condition number: {condition_number.item():.4f}")
        
        # Matrix inverse (for square matrices)
        if square_matrix.shape[0] == square_matrix.shape[1]:
            try:
                inverse = torch.inverse(square_matrix)
                print(f"Inverse shape: {inverse.shape}")
                
                # Verify inverse
                identity_check = torch.mm(square_matrix, inverse)
                inverse_error = torch.norm(identity_check - torch.eye(square_matrix.shape[0]))
                print(f"Inverse verification error: {inverse_error.item():.2e}")
                
            except RuntimeError as e:
                print(f"Matrix inversion failed: {str(e)[:50]}...")
        
        # Pseudoinverse for non-square matrices
        pseudoinverse = torch.pinverse(A)
        print(f"Pseudoinverse shape: {pseudoinverse.shape}")
    
    def solve_linear_systems(self):
        """Solving linear systems Ax = b"""
        print(f"\n=== SOLVING LINEAR SYSTEMS ===")
        
        # Create well-conditioned system
        n = 5
        A = torch.randn(n, n)
        # Make A well-conditioned by adding to diagonal
        A = A + torch.eye(n) * 2
        b = torch.randn(n, 3)  # Multiple right-hand sides
        
        print(f"System matrix A shape: {A.shape}")
        print(f"Right-hand side b shape: {b.shape}")
        
        # Method 1: Direct solution using torch.solve
        try:
            x, _ = torch.solve(b, A)  # Deprecated in newer versions
            print(f"Solution x shape: {x.shape}")
        except AttributeError:
            # For newer PyTorch versions
            x = torch.linalg.solve(A, b)
            print(f"Solution x shape: {x.shape}")
        
        # Verify solution
        residual = torch.mm(A, x) - b
        residual_norm = torch.norm(residual)
        print(f"Residual norm: {residual_norm.item():.2e}")
        
        # Method 2: Using LU decomposition
        LU, pivots = torch.lu(A)
        x_lu = torch.lu_solve(b, LU, pivots)
        
        lu_residual = torch.mm(A, x_lu) - b
        lu_residual_norm = torch.norm(lu_residual)
        print(f"LU solution residual norm: {lu_residual_norm.item():.2e}")
        
        # Method 3: Using Cholesky for positive definite systems
        # Create positive definite system
        A_pd = torch.mm(A.t(), A) + torch.eye(n) * 0.1
        b_pd = torch.randn(n, 1)
        
        try:
            L = torch.cholesky(A_pd)
            x_chol = torch.cholesky_solve(b_pd, L)
            
            chol_residual = torch.mm(A_pd, x_chol) - b_pd
            chol_residual_norm = torch.norm(chol_residual)
            print(f"Cholesky solution residual norm: {chol_residual_norm.item():.2e}")
            
        except RuntimeError as e:
            print(f"Cholesky solve failed: {str(e)[:50]}...")
        
        # Method 4: Least squares for overdetermined systems
        # Create overdetermined system (more equations than unknowns)
        m, n = 10, 5
        A_over = torch.randn(m, n)
        b_over = torch.randn(m, 1)
        
        # Least squares solution
        x_lstsq = torch.lstsq(b_over, A_over).solution[:n]
        
        lstsq_residual = torch.mm(A_over, x_lstsq) - b_over
        lstsq_residual_norm = torch.norm(lstsq_residual)
        print(f"Least squares residual norm: {lstsq_residual_norm.item():.4f}")

# Demonstrate linear algebra operations
linalg_ops = LinearAlgebraOperations()
linalg_ops.basic_matrix_operations()
linalg_ops.matrix_decompositions()
linalg_ops.matrix_norms_and_properties()
linalg_ops.solve_linear_systems()
```

### 2.2 Advanced Linear Algebra for Deep Learning

#### Neural Network Linear Operations

**Optimized Linear Operations for Neural Networks:**
```python
class DeepLearningLinearOps:
    """Linear algebra operations optimized for deep learning"""
    
    def efficient_linear_layers(self):
        """Efficient implementation of linear layers"""
        print("=== EFFICIENT LINEAR LAYER OPERATIONS ===")
        
        batch_size, input_dim, hidden_dim, output_dim = 64, 512, 256, 128
        
        # Input batch
        X = torch.randn(batch_size, input_dim)
        
        # Weight matrices
        W1 = torch.randn(input_dim, hidden_dim)
        b1 = torch.randn(hidden_dim)
        W2 = torch.randn(hidden_dim, output_dim)
        b2 = torch.randn(output_dim)
        
        print(f"Input shape: {X.shape}")
        print(f"W1 shape: {W1.shape}, b1 shape: {b1.shape}")
        print(f"W2 shape: {W2.shape}, b2 shape: {b2.shape}")
        
        # Method 1: Sequential operations
        h1 = torch.mm(X, W1) + b1
        h1_relu = torch.relu(h1)
        output = torch.mm(h1_relu, W2) + b2
        
        print(f"Output shape: {output.shape}")
        
        # Method 2: Using einsum for clarity
        h1_einsum = torch.einsum('bi,ih->bh', X, W1) + b1
        h1_relu_einsum = torch.relu(h1_einsum)
        output_einsum = torch.einsum('bh,ho->bo', h1_relu_einsum, W2) + b2
        
        print(f"Einsum output equal: {torch.allclose(output, output_einsum)}")
        
        # Method 3: Batch matrix multiplication for multiple samples
        # Reshape for bmm if needed
        X_batch = X.unsqueeze(0)  # [1, batch_size, input_dim]
        W1_batch = W1.unsqueeze(0)  # [1, input_dim, hidden_dim]
        
        # This is less efficient for this case, but useful for other scenarios
        h1_bmm = torch.bmm(X_batch, W1_batch).squeeze(0) + b1
        
        print(f"BMM result equal: {torch.allclose(h1, h1_bmm)}")
    
    def attention_linear_operations(self):
        """Linear operations in attention mechanisms"""
        print(f"\n=== ATTENTION LINEAR OPERATIONS ===")
        
        batch_size, seq_len, d_model = 32, 128, 512
        num_heads = 8
        d_k = d_model // num_heads
        
        # Input sequence
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Linear projection matrices for Q, K, V
        W_q = torch.randn(d_model, d_model)
        W_k = torch.randn(d_model, d_model)
        W_v = torch.randn(d_model, d_model)
        
        print(f"Input shape: {x.shape}")
        print(f"Weight matrices shape: {W_q.shape}")
        
        # Method 1: Standard linear projections
        Q = torch.einsum('bld,dm->blm', x, W_q)
        K = torch.einsum('bld,dm->blm', x, W_k)
        V = torch.einsum('bld,dm->blm', x, W_v)
        
        print(f"Q, K, V shapes: {Q.shape}")
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        
        print(f"Multi-head Q shape: {Q.shape}")
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        print(f"Attention output shape: {attention_output.shape}")
        
        # Method 2: Fused multi-head linear projection
        # More efficient: single large matrix multiplication
        W_qkv = torch.cat([W_q, W_k, W_v], dim=1)  # [d_model, 3*d_model]
        QKV = torch.einsum('bld,dm->blm', x, W_qkv)
        
        # Split into Q, K, V
        Q_fused, K_fused, V_fused = QKV.chunk(3, dim=-1)
        
        print(f"Fused QKV computation equal: {torch.allclose(Q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1), Q_fused)}")
    
    def convolutional_linear_equivalence(self):
        """Demonstrate equivalence between conv and linear operations"""
        print(f"\n=== CONV-LINEAR EQUIVALENCE ===")
        
        batch_size, in_channels, height, width = 16, 64, 8, 8
        out_channels = 128
        
        # Input feature maps
        x = torch.randn(batch_size, in_channels, height, width)
        
        # Method 1: 1x1 Convolution
        conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        conv_output = conv1x1(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Conv1x1 output shape: {conv_output.shape}")
        
        # Method 2: Equivalent linear operation
        # Reshape input to [batch_size * height * width, in_channels]
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(-1, in_channels)
        
        # Use conv weight as linear weight
        linear_weight = conv1x1.weight.squeeze(-1).squeeze(-1).t()  # [in_channels, out_channels]
        linear_output = torch.mm(x_reshaped, linear_weight)
        
        # Reshape back to feature map format
        linear_output = linear_output.view(batch_size, height, width, out_channels)
        linear_output = linear_output.permute(0, 3, 1, 2)
        
        print(f"Linear equivalent output shape: {linear_output.shape}")
        print(f"Conv and linear outputs equal: {torch.allclose(conv_output, linear_output, atol=1e-6)}")
        
        # Method 3: Using einsum for clarity
        einsum_output = torch.einsum('bchw,oc->bohw', x, conv1x1.weight.squeeze(-1).squeeze(-1))
        print(f"Einsum output equal: {torch.allclose(conv_output, einsum_output, atol=1e-6)}")
    
    def batch_linear_operations(self):
        """Efficient batch linear operations"""
        print(f"\n=== BATCH LINEAR OPERATIONS ===")
        
        # Scenario: Different linear transformations for each sample in batch
        batch_size, input_dim, output_dim = 32, 128, 64
        
        # Input batch
        X = torch.randn(batch_size, input_dim)
        
        # Different weight matrix for each sample
        W_batch = torch.randn(batch_size, input_dim, output_dim)
        b_batch = torch.randn(batch_size, output_dim)
        
        print(f"Input shape: {X.shape}")
        print(f"Batch weights shape: {W_batch.shape}")
        print(f"Batch bias shape: {b_batch.shape}")
        
        # Method 1: Loop (inefficient)
        output_loop = torch.zeros(batch_size, output_dim)
        for i in range(batch_size):
            output_loop[i] = torch.mv(W_batch[i], X[i]) + b_batch[i]
        
        # Method 2: Batch matrix-vector multiplication
        X_expanded = X.unsqueeze(-1)  # [batch_size, input_dim, 1]
        output_bmv = torch.bmm(W_batch.transpose(-2, -1), X_expanded).squeeze(-1) + b_batch
        
        print(f"Batch output shape: {output_bmv.shape}")
        print(f"Loop and batch methods equal: {torch.allclose(output_loop, output_bmv)}")
        
        # Method 3: Using einsum
        output_einsum = torch.einsum('bi,bio->bo', X, W_batch) + b_batch
        print(f"Einsum method equal: {torch.allclose(output_bmv, output_einsum)}")
        
        # Performance comparison
        import time
        
        def time_method(method, name, iterations=100):
            start = time.time()
            for _ in range(iterations):
                result = method()
            end = time.time()
            avg_time = (end - start) / iterations * 1000
            print(f"  {name}: {avg_time:.3f}ms average")
        
        print(f"Performance comparison:")
        time_method(lambda: torch.bmm(W_batch.transpose(-2, -1), X.unsqueeze(-1)).squeeze(-1) + b_batch, "Batch BMM")
        time_method(lambda: torch.einsum('bi,bio->bo', X, W_batch) + b_batch, "Einsum")

# Demonstrate deep learning linear operations
dl_linalg = DeepLearningLinearOps()
dl_linalg.efficient_linear_layers()
dl_linalg.attention_linear_operations()
dl_linalg.convolutional_linear_equivalence()
dl_linalg.batch_linear_operations()
```

---

## 3. Reduction Operations

### 3.1 Statistical Reductions

#### Comprehensive Statistical Operations

**Statistical Reduction Operations:**
```python
class StatisticalReductions:
    """Comprehensive statistical reduction operations"""
    
    def basic_statistical_operations(self):
        """Basic statistical operations across dimensions"""
        print("=== BASIC STATISTICAL OPERATIONS ===")
        
        # Create sample data
        data_2d = torch.randn(5, 8)
        data_3d = torch.randn(4, 5, 6)
        
        print(f"2D data shape: {data_2d.shape}")
        print(f"3D data shape: {data_3d.shape}")
        
        # Sum operations
        total_sum = torch.sum(data_2d)
        sum_dim0 = torch.sum(data_2d, dim=0)  # Sum over rows
        sum_dim1 = torch.sum(data_2d, dim=1)  # Sum over columns
        sum_keepdim = torch.sum(data_2d, dim=0, keepdim=True)
        
        print(f"Total sum: {total_sum.item():.4f}")
        print(f"Sum dim 0 shape: {sum_dim0.shape}")
        print(f"Sum dim 1 shape: {sum_dim1.shape}")
        print(f"Sum keepdim shape: {sum_keepdim.shape}")
        
        # Mean operations
        mean_all = torch.mean(data_2d)
        mean_dim0 = torch.mean(data_2d, dim=0)
        mean_dims = torch.mean(data_3d, dim=[0, 2])  # Mean over multiple dims
        
        print(f"Mean all: {mean_all.item():.4f}")
        print(f"Mean dim 0 shape: {mean_dim0.shape}")
        print(f"Mean dims [0,2] shape: {mean_dims.shape}")
        
        # Variance and standard deviation
        var_all = torch.var(data_2d)
        std_all = torch.std(data_2d)
        var_unbiased = torch.var(data_2d, unbiased=True)  # Bessel's correction
        
        print(f"Variance: {var_all.item():.4f}")
        print(f"Std deviation: {std_all.item():.4f}")
        print(f"Unbiased variance: {var_unbiased.item():.4f}")
        
        # Variance along dimensions
        var_dim0 = torch.var(data_2d, dim=0, unbiased=False)
        std_dim1 = torch.std(data_2d, dim=1, keepdim=True)
        
        print(f"Variance dim 0 shape: {var_dim0.shape}")
        print(f"Std dim 1 keepdim shape: {std_dim1.shape}")
    
    def advanced_statistical_operations(self):
        """Advanced statistical operations"""
        print(f"\n=== ADVANCED STATISTICAL OPERATIONS ===")
        
        data = torch.randn(100, 50)
        
        # Percentiles and quantiles
        median = torch.median(data)
        q25 = torch.quantile(data, 0.25)
        q75 = torch.quantile(data, 0.75)
        
        print(f"Median: {median.values.item():.4f}")
        print(f"25th percentile: {q25.item():.4f}")
        print(f"75th percentile: {q75.item():.4f}")
        
        # Multiple quantiles at once
        quantiles = torch.quantile(data, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        print(f"Multiple quantiles: {quantiles}")
        
        # Min/Max operations with indices
        min_val, min_idx = torch.min(data, dim=1)
        max_val, max_idx = torch.max(data, dim=1)
        
        print(f"Min values shape: {min_val.shape}")
        print(f"Min indices shape: {min_idx.shape}")
        
        # Argmin/Argmax
        argmin_global = torch.argmin(data)
        argmax_dim0 = torch.argmax(data, dim=0)
        
        print(f"Global argmin: {argmin_global.item()}")
        print(f"Argmax dim 0 shape: {argmax_dim0.shape}")
        
        # Mode (most frequent value)
        int_data = torch.randint(0, 10, (20, 15))
        mode_val, mode_idx = torch.mode(int_data, dim=1)
        
        print(f"Mode values shape: {mode_val.shape}")
        print(f"Mode indices shape: {mode_idx.shape}")
    
    def covariance_and_correlation(self):
        """Covariance and correlation operations"""
        print(f"\n=== COVARIANCE AND CORRELATION ===")
        
        # Generate correlated data
        n_samples, n_features = 1000, 5
        
        # Create correlation structure
        true_corr = torch.tensor([
            [1.0, 0.8, 0.3, -0.2, 0.0],
            [0.8, 1.0, 0.4, -0.1, 0.1],
            [0.3, 0.4, 1.0, 0.0, 0.5],
            [-0.2, -0.1, 0.0, 1.0, -0.3],
            [0.0, 0.1, 0.5, -0.3, 1.0]
        ])
        
        # Generate data with desired correlation
        L = torch.linalg.cholesky(true_corr)
        uncorr_data = torch.randn(n_samples, n_features)
        corr_data = torch.mm(uncorr_data, L.t())
        
        print(f"Generated data shape: {corr_data.shape}")
        
        # Compute sample correlation matrix
        # Center the data
        centered_data = corr_data - corr_data.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = torch.mm(centered_data.t(), centered_data) / (n_samples - 1)
        
        # Compute correlation matrix
        std_devs = torch.sqrt(torch.diag(cov_matrix))
        corr_matrix = cov_matrix / torch.outer(std_devs, std_devs)
        
        print(f"Sample correlation matrix:")
        print(corr_matrix)
        
        # Compare with true correlation
        corr_error = torch.norm(corr_matrix - true_corr)
        print(f"Correlation estimation error: {corr_error.item():.4f}")
        
        # Pairwise correlations
        def pairwise_correlation(x, y):
            """Compute correlation between two vectors"""
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            
            numerator = torch.sum(x_centered * y_centered)
            denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
            
            return numerator / denominator
        
        # Test pairwise correlation
        pair_corr = pairwise_correlation(corr_data[:, 0], corr_data[:, 1])
        matrix_corr = corr_matrix[0, 1]
        
        print(f"Pairwise correlation: {pair_corr.item():.4f}")
        print(f"Matrix correlation: {matrix_corr.item():.4f}")
        print(f"Correlation methods agree: {torch.allclose(pair_corr, matrix_corr)}")
    
    def advanced_moments_and_distributions(self):
        """Advanced moments and distribution statistics"""
        print(f"\n=== ADVANCED MOMENTS AND DISTRIBUTIONS ===")
        
        # Generate data from different distributions
        normal_data = torch.randn(10000)
        uniform_data = torch.rand(10000) * 2 - 1  # Uniform[-1, 1]
        exponential_data = torch.exponential(torch.ones(10000))
        
        def compute_moments(data, name):
            """Compute first four moments of data"""
            mean = torch.mean(data)
            var = torch.var(data, unbiased=True)
            std = torch.sqrt(var)
            
            # Standardize data for higher moments
            standardized = (data - mean) / std
            
            # Third moment (skewness)
            skewness = torch.mean(standardized ** 3)
            
            # Fourth moment (kurtosis)
            kurtosis = torch.mean(standardized ** 4) - 3  # Excess kurtosis
            
            print(f"{name} distribution:")
            print(f"  Mean: {mean.item():.4f}")
            print(f"  Std: {std.item():.4f}")
            print(f"  Skewness: {skewness.item():.4f}")
            print(f"  Excess Kurtosis: {kurtosis.item():.4f}")
        
        compute_moments(normal_data, "Normal")
        compute_moments(uniform_data, "Uniform")
        compute_moments(exponential_data, "Exponential")
        
        # Robust statistics (resistant to outliers)
        # Add outliers to normal data
        normal_with_outliers = torch.cat([normal_data, torch.tensor([10.0, -15.0, 20.0])])
        
        # Compare mean vs median
        mean_with_outliers = torch.mean(normal_with_outliers)
        median_with_outliers = torch.median(normal_with_outliers).values
        
        print(f"\nWith outliers:")
        print(f"  Mean: {mean_with_outliers.item():.4f}")
        print(f"  Median: {median_with_outliers.item():.4f}")
        
        # Median Absolute Deviation (MAD) - robust scale estimator
        mad = torch.median(torch.abs(normal_with_outliers - median_with_outliers)).values
        
        print(f"  MAD: {mad.item():.4f}")
        print(f"  Std: {torch.std(normal_with_outliers).item():.4f}")

# Demonstrate statistical reductions
stats = StatisticalReductions()
stats.basic_statistical_operations()
stats.advanced_statistical_operations()
stats.covariance_and_correlation()
stats.advanced_moments_and_distributions()
```

### 3.2 Specialized Reductions for Deep Learning

#### Neural Network Specific Reductions

**Reductions Optimized for Deep Learning:**
```python
class DeepLearningReductions:
    """Specialized reduction operations for deep learning"""
    
    def loss_function_reductions(self):
        """Reductions commonly used in loss functions"""
        print("=== LOSS FUNCTION REDUCTIONS ===")
        
        batch_size, num_classes = 64, 10
        
        # Generate sample predictions and targets
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        print(f"Logits shape: {logits.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Cross-entropy loss components
        log_softmax = torch.log_softmax(logits, dim=1)
        softmax_probs = torch.softmax(logits, dim=1)
        
        # Manual cross-entropy computation
        # Select log probabilities for correct classes
        log_probs_correct = log_softmax[range(batch_size), targets]
        
        # Different reduction strategies
        ce_sum = -torch.sum(log_probs_correct)  # Sum reduction
        ce_mean = -torch.mean(log_probs_correct)  # Mean reduction
        ce_none = -log_probs_correct  # No reduction
        
        print(f"Cross-entropy sum: {ce_sum.item():.4f}")
        print(f"Cross-entropy mean: {ce_mean.item():.4f}")
        print(f"Cross-entropy per sample shape: {ce_none.shape}")
        
        # Compare with PyTorch's built-in
        ce_builtin_mean = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        ce_builtin_sum = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
        
        print(f"Built-in CE mean: {ce_builtin_mean.item():.4f}")
        print(f"Built-in CE sum: {ce_builtin_sum.item():.4f}")
        print(f"Manual vs built-in mean: {torch.allclose(ce_mean, ce_builtin_mean)}")
        
        # Focal loss reduction (for imbalanced datasets)
        def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction='mean'):
            """Focal loss with different reductions"""
            ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)  # Probability of correct class
            focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
            
            if reduction == 'mean':
                return torch.mean(focal_loss_val)
            elif reduction == 'sum':
                return torch.sum(focal_loss_val)
            else:
                return focal_loss_val
        
        focal_mean = focal_loss(logits, targets, reduction='mean')
        focal_sum = focal_loss(logits, targets, reduction='sum')
        
        print(f"Focal loss mean: {focal_mean.item():.4f}")
        print(f"Focal loss sum: {focal_sum.item():.4f}")
    
    def batch_normalization_reductions(self):
        """Reductions for batch normalization statistics"""
        print(f"\n=== BATCH NORMALIZATION REDUCTIONS ===")
        
        batch_size, channels, height, width = 32, 64, 16, 16
        
        # Feature maps from convolutional layer
        feature_maps = torch.randn(batch_size, channels, height, width)
        
        print(f"Feature maps shape: {feature_maps.shape}")
        
        # Batch normalization statistics
        # Mean and variance across batch and spatial dimensions
        bn_mean = torch.mean(feature_maps, dim=[0, 2, 3], keepdim=True)
        bn_var = torch.var(feature_maps, dim=[0, 2, 3], keepdim=True, unbiased=False)
        
        print(f"BN mean shape: {bn_mean.shape}")
        print(f"BN variance shape: {bn_var.shape}")
        
        # Alternative: flatten spatial dimensions first
        flattened = feature_maps.view(batch_size, channels, -1)
        bn_mean_alt = torch.mean(flattened, dim=[0, 2], keepdim=True)
        bn_var_alt = torch.var(flattened, dim=[0, 2], keepdim=True, unbiased=False)
        
        # Reshape back
        bn_mean_alt = bn_mean_alt.view(1, channels, 1, 1)
        bn_var_alt = bn_var_alt.view(1, channels, 1, 1)
        
        print(f"Alternative BN mean equal: {torch.allclose(bn_mean, bn_mean_alt)}")
        print(f"Alternative BN var equal: {torch.allclose(bn_var, bn_var_alt)}")
        
        # Layer normalization statistics (different reduction)
        ln_mean = torch.mean(feature_maps, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(feature_maps, dim=[1, 2, 3], keepdim=True, unbiased=False)
        
        print(f"LN mean shape: {ln_mean.shape}")
        print(f"LN variance shape: {ln_var.shape}")
        
        # Group normalization (groups of channels)
        num_groups = 8
        groups_per_channel = channels // num_groups
        
        # Reshape for group normalization
        grouped = feature_maps.view(batch_size, num_groups, groups_per_channel, height, width)
        gn_mean = torch.mean(grouped, dim=[2, 3, 4], keepdim=True)
        gn_var = torch.var(grouped, dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        # Reshape back
        gn_mean = gn_mean.view(batch_size, num_groups, 1, 1, 1).expand(-1, -1, groups_per_channel, -1, -1)
        gn_mean = gn_mean.contiguous().view(batch_size, channels, 1, 1)
        
        print(f"GN mean shape: {gn_mean.shape}")
        
        # Instance normalization (per instance, per channel)
        in_mean = torch.mean(feature_maps, dim=[2, 3], keepdim=True)
        in_var = torch.var(feature_maps, dim=[2, 3], keepdim=True, unbiased=False)
        
        print(f"IN mean shape: {in_mean.shape}")
        print(f"IN variance shape: {in_var.shape}")
    
    def attention_reductions(self):
        """Reductions in attention mechanisms"""
        print(f"\n=== ATTENTION MECHANISM REDUCTIONS ===")
        
        batch_size, seq_len, d_model = 16, 32, 256
        
        # Attention scores and values
        attention_scores = torch.randn(batch_size, seq_len, seq_len)
        values = torch.randn(batch_size, seq_len, d_model)
        
        print(f"Attention scores shape: {attention_scores.shape}")
        print(f"Values shape: {values.shape}")
        
        # Softmax attention (normalization across sequence dimension)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted average reduction
        attended_output = torch.bmm(attention_weights, values)
        
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Attended output shape: {attended_output.shape}")
        
        # Global attention (average over all positions)
        global_context = torch.mean(values, dim=1, keepdim=True)  # [B, 1, D]
        
        print(f"Global context shape: {global_context.shape}")
        
        # Max pooling attention (max over sequence)
        max_pooled, _ = torch.max(values, dim=1, keepdim=True)
        
        print(f"Max pooled shape: {max_pooled.shape}")
        
        # Learnable global attention
        # Attention weights for global pooling
        global_attention_scores = torch.randn(batch_size, 1, seq_len)
        global_attention_weights = torch.softmax(global_attention_scores, dim=-1)
        
        # Weighted global context
        weighted_global = torch.bmm(global_attention_weights, values)
        
        print(f"Weighted global context shape: {weighted_global.shape}")
        
        # Multi-scale attention reductions
        # Local attention (sliding window)
        window_size = 5
        
        # Create sliding window masks
        local_masks = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            local_masks[i, start:end] = 1
        
        # Apply local mask to attention scores
        masked_scores = attention_scores + (1 - local_masks) * (-1e9)
        local_attention_weights = torch.softmax(masked_scores, dim=-1)
        local_attended = torch.bmm(local_attention_weights, values)
        
        print(f"Local attended output shape: {local_attended.shape}")
    
    def regularization_reductions(self):
        """Reductions used in regularization techniques"""
        print(f"\n=== REGULARIZATION REDUCTIONS ===")
        
        # Model parameters (simulated)
        layer1_weights = torch.randn(256, 128)
        layer2_weights = torch.randn(128, 64)
        layer3_weights = torch.randn(64, 10)
        
        all_weights = [layer1_weights, layer2_weights, layer3_weights]
        
        # L1 regularization (sum of absolute values)
        l1_penalty = sum(torch.sum(torch.abs(w)) for w in all_weights)
        
        print(f"L1 penalty: {l1_penalty.item():.4f}")
        
        # L2 regularization (sum of squared values)
        l2_penalty = sum(torch.sum(w ** 2) for w in all_weights)
        
        print(f"L2 penalty: {l2_penalty.item():.4f}")
        
        # Frobenius norm penalty (matrix-specific)
        frobenius_penalties = [torch.norm(w, 'fro') ** 2 for w in all_weights]
        total_frobenius = sum(frobenius_penalties)
        
        print(f"Frobenius penalty: {total_frobenius.item():.4f}")
        print(f"L2 equals Frobenius: {torch.allclose(l2_penalty, total_frobenius)}")
        
        # Spectral norm penalty (largest singular value)
        spectral_norms = [torch.norm(w, 2) for w in all_weights]  # Spectral norm
        max_spectral_norm = max(spectral_norms)
        
        print(f"Max spectral norm: {max_spectral_norm.item():.4f}")
        
        # Group sparsity penalty
        def group_lasso_penalty(weight_matrix, group_size=8):
            """Group Lasso penalty for structured sparsity"""
            # Reshape into groups
            num_groups = weight_matrix.size(0) // group_size
            grouped_weights = weight_matrix[:num_groups * group_size].view(num_groups, group_size, -1)
            
            # L2 norm within each group, then L1 norm across groups
            group_norms = torch.norm(grouped_weights, p=2, dim=[1, 2])
            return torch.sum(group_norms)
        
        group_penalties = [group_lasso_penalty(w) for w in all_weights if w.size(0) >= 8]
        total_group_penalty = sum(group_penalties) if group_penalties else torch.tensor(0.0)
        
        print(f"Group Lasso penalty: {total_group_penalty.item():.4f}")
        
        # Dropout simulation (random reduction)
        activations = torch.randn(32, 128)  # Batch of activations
        dropout_rate = 0.5
        
        # Training mode dropout
        dropout_mask = torch.bernoulli(torch.full_like(activations, 1 - dropout_rate))
        dropped_activations = activations * dropout_mask / (1 - dropout_rate)
        
        # Verify scaling
        original_mean = torch.mean(activations)
        dropped_mean = torch.mean(dropped_activations)
        
        print(f"Original activation mean: {original_mean.item():.4f}")
        print(f"Dropped activation mean: {dropped_mean.item():.4f}")
        print(f"Means approximately equal: {torch.allclose(original_mean, dropped_mean, atol=0.1)}")

# Demonstrate deep learning reductions
dl_reductions = DeepLearningReductions()
dl_reductions.loss_function_reductions()
dl_reductions.batch_normalization_reductions()
dl_reductions.attention_reductions()
dl_reductions.regularization_reductions()
```

---

## 4. Advanced Tensor Manipulation

### 4.1 Complex Tensor Operations

#### Advanced Manipulation Techniques

**Complex Tensor Manipulation Operations:**
```python
class AdvancedTensorManipulation:
    """Advanced tensor manipulation techniques"""
    
    def tensor_splitting_and_concatenation(self):
        """Advanced splitting and concatenation operations"""
        print("=== TENSOR SPLITTING AND CONCATENATION ===")
        
        # Create sample tensors
        tensor_large = torch.randn(12, 8, 6)
        tensor_a = torch.randn(3, 8, 6)
        tensor_b = torch.randn(5, 8, 6)
        tensor_c = torch.randn(4, 8, 6)
        
        print(f"Large tensor shape: {tensor_large.shape}")
        
        # Splitting operations
        # Split into equal parts
        split_equal = torch.split(tensor_large, 4, dim=0)  # 3 tensors of size 4
        print(f"Equal split: {[t.shape for t in split_equal]}")
        
        # Split into unequal parts
        split_unequal = torch.split(tensor_large, [3, 5, 4], dim=0)
        print(f"Unequal split: {[t.shape for t in split_unequal]}")
        
        # Chunk operation (similar to split but specify number of chunks)
        chunks = torch.chunk(tensor_large, 3, dim=0)
        print(f"Chunks: {[t.shape for t in chunks]}")
        
        # Concatenation
        concatenated = torch.cat([tensor_a, tensor_b, tensor_c], dim=0)
        print(f"Concatenated shape: {concatenated.shape}")
        
        # Stack (adds new dimension)
        tensor_2d_a = torch.randn(4, 5)
        tensor_2d_b = torch.randn(4, 5)
        tensor_2d_c = torch.randn(4, 5)
        
        stacked_0 = torch.stack([tensor_2d_a, tensor_2d_b, tensor_2d_c], dim=0)
        stacked_1 = torch.stack([tensor_2d_a, tensor_2d_b, tensor_2d_c], dim=1)
        stacked_2 = torch.stack([tensor_2d_a, tensor_2d_b, tensor_2d_c], dim=2)
        
        print(f"Original 2D shape: {tensor_2d_a.shape}")
        print(f"Stacked dim 0: {stacked_0.shape}")
        print(f"Stacked dim 1: {stacked_1.shape}")
        print(f"Stacked dim 2: {stacked_2.shape}")
        
        # Unbind (opposite of stack)
        unbound = torch.unbind(stacked_0, dim=0)
        print(f"Unbound: {[t.shape for t in unbound]}")
        
        # Verify roundtrip
        print(f"Roundtrip successful: {torch.allclose(unbound[0], tensor_2d_a)}")
    
    def advanced_indexing_operations(self):
        """Advanced indexing and selection operations"""
        print(f"\n=== ADVANCED INDEXING OPERATIONS ===")
        
        # Create sample data
        data = torch.randn(100, 50, 30)
        
        # Gather operation
        indices_dim1 = torch.randint(0, 50, (100, 20))  # Select 20 out of 50 features
        gathered = torch.gather(data, 1, indices_dim1.unsqueeze(-1).expand(-1, -1, 30))
        
        print(f"Original shape: {data.shape}")
        print(f"Indices shape: {indices_dim1.shape}")
        print(f"Gathered shape: {gathered.shape}")
        
        # Scatter operation (inverse of gather)
        scattered = torch.zeros_like(data)
        scattered.scatter_(1, indices_dim1.unsqueeze(-1).expand(-1, -1, 30), gathered)
        
        # Verify partial reconstruction
        verification = torch.gather(scattered, 1, indices_dim1.unsqueeze(-1).expand(-1, -1, 30))
        print(f"Scatter-gather roundtrip: {torch.allclose(gathered, verification)}")
        
        # Index select operation
        selected_samples = torch.index_select(data, 0, torch.tensor([0, 5, 10, 15]))
        selected_features = torch.index_select(data, 1, torch.tensor([1, 3, 5, 7, 9]))
        
        print(f"Selected samples shape: {selected_samples.shape}")
        print(f"Selected features shape: {selected_features.shape}")
        
        # Take operation (flattened indexing)
        flat_indices = torch.tensor([0, 100, 500, 1000, 5000])
        taken_elements = torch.take(data, flat_indices)
        print(f"Taken elements shape: {taken_elements.shape}")
        
        # Advanced boolean indexing
        # Multi-condition selection
        condition1 = torch.abs(data) > 1.0
        condition2 = data > 0
        combined_condition = condition1 & condition2
        
        selected_elements = data[combined_condition]
        print(f"Multi-condition selected elements: {selected_elements.numel()}")
        
        # Masked selection with replacement
        mask = torch.rand_like(data) > 0.7  # Keep ~30% of elements
        masked_data = torch.where(mask, data, torch.zeros_like(data))
        
        print(f"Mask keeps {mask.float().mean().item():.1%} of elements")
    
    def tensor_permutation_operations(self):
        """Advanced tensor permutation and reordering"""
        print(f"\n=== TENSOR PERMUTATION OPERATIONS ===")
        
        # Create sample tensor
        tensor_4d = torch.randn(2, 3, 4, 5)
        
        print(f"Original shape: {tensor_4d.shape}")
        
        # Permute dimensions
        permuted = tensor_4d.permute(3, 1, 0, 2)  # New order: [5, 3, 2, 4]
        print(f"Permuted shape: {permuted.shape}")
        
        # Multiple permutations
        # Common transformations for different data layouts
        
        # NCHW to NHWC (batch, channels, height, width -> batch, height, width, channels)
        nchw_tensor = torch.randn(16, 32, 64, 64)
        nhwc_tensor = nchw_tensor.permute(0, 2, 3, 1)
        
        print(f"NCHW shape: {nchw_tensor.shape}")
        print(f"NHWC shape: {nhwc_tensor.shape}")
        
        # Sequence processing: (batch, seq, features) -> (seq, batch, features)
        batch_first = torch.randn(32, 128, 256)
        seq_first = batch_first.permute(1, 0, 2)
        
        print(f"Batch first: {batch_first.shape}")
        print(f"Sequence first: {seq_first.shape}")
        
        # Advanced permutation patterns
        # Matrix batch transpose
        matrix_batch = torch.randn(10, 5, 7)  # Batch of matrices
        transposed_batch = matrix_batch.transpose(-2, -1)  # Transpose last two dims
        
        print(f"Matrix batch: {matrix_batch.shape}")
        print(f"Transposed batch: {transposed_batch.shape}")
        
        # Roll operation (circular shift)
        sequence = torch.arange(10)
        rolled_right = torch.roll(sequence, 3)  # Shift right by 3
        rolled_left = torch.roll(sequence, -2)  # Shift left by 2
        
        print(f"Original sequence: {sequence}")
        print(f"Rolled right by 3: {rolled_right}")
        print(f"Rolled left by 2: {rolled_left}")
        
        # Multi-dimensional roll
        matrix_2d = torch.arange(12).view(3, 4)
        rolled_2d = torch.roll(matrix_2d, shifts=(1, -1), dims=(0, 1))
        
        print(f"Original 2D:\n{matrix_2d}")
        print(f"Rolled 2D:\n{rolled_2d}")
    
    def tensor_padding_operations(self):
        """Comprehensive padding operations"""
        print(f"\n=== TENSOR PADDING OPERATIONS ===")
        
        # Create sample tensor
        tensor_2d = torch.arange(12).view(3, 4).float()
        tensor_3d = torch.randn(2, 3, 4)
        
        print(f"Original 2D tensor:\n{tensor_2d}")
        
        # Constant padding
        padded_constant = torch.nn.functional.pad(tensor_2d, (1, 2, 1, 1), mode='constant', value=0)
        print(f"Constant padded shape: {padded_constant.shape}")
        print(f"Constant padded:\n{padded_constant}")
        
        # Reflection padding
        padded_reflect = torch.nn.functional.pad(tensor_2d, (1, 1, 1, 1), mode='reflect')
        print(f"Reflection padded:\n{padded_reflect}")
        
        # Replication padding (extend border values)
        padded_replicate = torch.nn.functional.pad(tensor_2d, (1, 1, 1, 1), mode='replicate')
        print(f"Replication padded:\n{padded_replicate}")
        
        # Circular padding
        padded_circular = torch.nn.functional.pad(tensor_2d, (1, 1, 1, 1), mode='circular')
        print(f"Circular padded:\n{padded_circular}")
        
        # Complex padding for 3D tensors
        # Pad format: (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right, ...)
        complex_pad = torch.nn.functional.pad(tensor_3d, (2, 1, 1, 2, 0, 1), mode='constant', value=-1)
        
        print(f"3D original shape: {tensor_3d.shape}")
        print(f"3D padded shape: {complex_pad.shape}")
        
        # Asymmetric padding
        asymmetric = torch.nn.functional.pad(tensor_2d, (0, 3, 2, 0), mode='constant', value=99)
        print(f"Asymmetric padded shape: {asymmetric.shape}")
        print(f"Asymmetric padded:\n{asymmetric}")
        
        # Padding for sequence processing
        # Variable length sequences
        sequences = [
            torch.randn(5, 10),   # Length 5
            torch.randn(8, 10),   # Length 8
            torch.randn(3, 10),   # Length 3
        ]
        
        # Pad to same length
        max_length = max(seq.size(0) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            pad_length = max_length - seq.size(0)
            padded_seq = torch.nn.functional.pad(seq, (0, 0, 0, pad_length), mode='constant', value=0)
            padded_sequences.append(padded_seq)
        
        # Stack into batch
        batch_sequences = torch.stack(padded_sequences)
        
        print(f"Original sequence lengths: {[seq.size(0) for seq in sequences]}")
        print(f"Padded batch shape: {batch_sequences.shape}")

# Demonstrate advanced tensor manipulation
advanced_manip = AdvancedTensorManipulation()
advanced_manip.tensor_splitting_and_concatenation()
advanced_manip.advanced_indexing_operations()
advanced_manip.tensor_permutation_operations()
advanced_manip.tensor_padding_operations()
```

---

## 5. Key Questions and Answers

### Beginner Level Questions

**Q1: What's the difference between `torch.sum()` and `torch.mean()` when dealing with different dimensions?**
**A:** Both functions can operate on different dimensions but have different purposes:
- **`torch.sum()`:** Adds all values along specified dimensions
- **`torch.mean()`:** Computes average along specified dimensions

```python
tensor = torch.randn(3, 4)
sum_all = torch.sum(tensor)        # Single value: sum of all elements
sum_dim0 = torch.sum(tensor, dim=0) # Shape [4]: sum along rows
mean_dim1 = torch.mean(tensor, dim=1) # Shape [3]: average along columns
```

**Q2: How does broadcasting work with mathematical operations?**
**A:** Broadcasting allows operations between tensors of different shapes by automatically expanding smaller tensors:
- **Rules:** Dimensions are aligned from right to left
- **Compatible:** Dimensions must be equal or one must be 1
- **Expansion:** Size-1 dimensions are "virtually" expanded

```python
a = torch.randn(3, 1, 4)  # Shape [3, 1, 4]
b = torch.randn(1, 5, 1)  # Shape [1, 5, 1] 
result = a + b             # Result shape [3, 5, 4]
```

**Q3: What's the difference between `torch.mm()`, `torch.bmm()`, and `@` operator?**
**A:** Different matrix multiplication operations:
- **`torch.mm()`:** 2D matrix multiplication only
- **`torch.bmm()`:** Batch matrix multiplication (3D tensors)
- **`@` operator:** General matrix multiplication (works with 2D and higher)

**Q4: How do I efficiently compute pairwise distances between points?**
**A:** Use broadcasting to vectorize the computation:
```python
def pairwise_distances(x, y):
    # x: [N, D], y: [M, D]
    # Expand for broadcasting: [N, 1, D] and [1, M, D]
    x_expanded = x.unsqueeze(1)
    y_expanded = y.unsqueeze(0) 
    distances = torch.norm(x_expanded - y_expanded, dim=2)
    return distances  # Shape [N, M]
```

### Intermediate Level Questions

**Q5: How do I implement custom reduction operations efficiently?**
**A:** Use PyTorch's built-in functions and combine them creatively:
```python
def custom_reduction(tensor, dim, operation='geometric_mean'):
    if operation == 'geometric_mean':
        # Geometric mean: nth root of product
        log_values = torch.log(torch.abs(tensor) + 1e-8)
        log_mean = torch.mean(log_values, dim=dim)
        return torch.exp(log_mean)
    elif operation == 'harmonic_mean':
        # Harmonic mean: n / sum(1/x)
        reciprocals = 1.0 / (tensor + 1e-8)
        return tensor.size(dim) / torch.sum(reciprocals, dim=dim)
```

**Q6: What's the most efficient way to apply different operations to different parts of a tensor?**
**A:** Use masking and `torch.where()` for conditional operations:
```python
def conditional_operations(tensor):
    # Different operations based on value ranges
    result = torch.zeros_like(tensor)
    
    # Method 1: Multiple where operations
    positive_mask = tensor > 0
    negative_mask = tensor < 0
    
    result = torch.where(positive_mask, torch.sqrt(tensor), result)
    result = torch.where(negative_mask, -torch.sqrt(-tensor), result)
    
    return result
```

**Q7: How do I efficiently implement attention mechanisms using tensor operations?**
**A:** Use einsum and broadcasting for clean, efficient attention:
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [batch, seq_len, d_model]
    d_k = q.size(-1)
    
    # Compute attention scores
    scores = torch.einsum('bqd,bkd->bqk', q, k) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax and apply to values
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.einsum('bqk,bvd->bqd', attention_weights, v)
    
    return output, attention_weights
```

### Advanced Level Questions

**Q8: How can I implement memory-efficient operations for very large tensors?**
**A:** Use chunking and gradient checkpointing:
```python
class MemoryEfficientOperation:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
    
    def chunked_operation(self, large_tensor, operation_func):
        """Apply operation in chunks to save memory"""
        results = []
        
        for i in range(0, large_tensor.size(0), self.chunk_size):
            chunk = large_tensor[i:i + self.chunk_size]
            chunk_result = operation_func(chunk)
            results.append(chunk_result)
        
        return torch.cat(results, dim=0)
    
    def checkpoint_operation(self, tensor, expensive_func):
        """Use gradient checkpointing for memory efficiency"""
        from torch.utils.checkpoint import checkpoint
        return checkpoint(expensive_func, tensor)
```

**Q9: How do I optimize tensor operations for specific hardware (CPU vs GPU)?**
**A:** Consider hardware characteristics in your implementation:
```python
def hardware_optimized_operation(tensor):
    if tensor.is_cuda:
        # GPU optimization: prefer parallel operations
        # Use larger chunks, minimize kernel launches
        return torch.nn.functional.conv2d(tensor, weight, stride=2)
    else:
        # CPU optimization: prefer cache-friendly operations
        # Use smaller chunks, optimize for memory hierarchy
        return manual_convolution_cpu_optimized(tensor, weight)

def choose_algorithm_by_size(tensor):
    """Choose algorithm based on tensor size"""
    if tensor.numel() < 1000:
        return small_tensor_algorithm(tensor)
    elif tensor.numel() < 1000000:
        return medium_tensor_algorithm(tensor)
    else:
        return large_tensor_algorithm(tensor)
```

**Q10: How do I implement custom broadcasting rules for specialized operations?**
**A:** Create functions that explicitly handle broadcasting:
```python
def custom_broadcast_operation(a, b, operation):
    """Custom operation with explicit broadcasting control"""
    
    # Determine output shape
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    
    # Pad shapes to same length
    max_dims = max(len(a_shape), len(b_shape))
    a_shape = [1] * (max_dims - len(a_shape)) + a_shape
    b_shape = [1] * (max_dims - len(b_shape)) + b_shape
    
    # Compute result shape
    result_shape = []
    for dim_a, dim_b in zip(a_shape, b_shape):
        if dim_a == 1:
            result_shape.append(dim_b)
        elif dim_b == 1:
            result_shape.append(dim_a)
        elif dim_a == dim_b:
            result_shape.append(dim_a)
        else:
            raise ValueError(f"Cannot broadcast {dim_a} and {dim_b}")
    
    # Expand tensors
    a_expanded = a.expand(result_shape)
    b_expanded = b.expand(result_shape)
    
    # Apply operation
    return operation(a_expanded, b_expanded)
```

---

## Summary and Advanced Patterns

### Performance Optimization Guidelines

**Memory Efficiency:**
- Use in-place operations when possible (`add_()`, `mul_()`, etc.)
- Leverage broadcasting instead of explicit tensor expansion
- Consider chunking for very large operations
- Use appropriate data types (float16 for inference, int32 vs int64)

**Computational Efficiency:**
- Batch operations to maximize parallelism
- Use optimized linear algebra operations (`torch.mm`, `torch.bmm`)
- Leverage einsum for complex tensor contractions
- Choose algorithms based on tensor sizes and hardware

### Common Patterns in Deep Learning

**Neural Network Operations:**
- Linear transformations with broadcasting for bias addition
- Batch normalization statistics computation
- Attention mechanism implementations
- Loss function reductions

**Data Processing:**
- Tensor reshaping for different data layouts
- Advanced indexing for data selection
- Padding operations for sequence processing
- Statistical computations for normalization

Understanding advanced tensor operations is crucial for implementing efficient deep learning algorithms. These operations form the building blocks for neural networks, optimization algorithms, and data preprocessing pipelines.

---

## Next Steps

In the next module, we'll explore GPU acceleration and memory management techniques that are essential for training large-scale deep learning models efficiently.