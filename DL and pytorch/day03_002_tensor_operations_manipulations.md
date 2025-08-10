# Day 3.2: Advanced Tensor Operations and Mathematical Manipulations

## Overview
Tensor operations form the computational core of deep learning algorithms. Beyond basic arithmetic, PyTorch provides a comprehensive suite of mathematical operations, linear algebra functions, statistical computations, and advanced manipulation techniques. This module provides exhaustive coverage of tensor operations, from elementary mathematical functions to sophisticated linear algebra operations, with emphasis on both theoretical understanding and practical implementation patterns.

## Mathematical Operations

### Element-wise Arithmetic Operations

**Basic Arithmetic Operations**
Element-wise operations apply functions to corresponding elements of tensors, following broadcasting rules:

**Fundamental Arithmetic**:
```python
import torch
import math

# Create sample tensors
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])

# Addition (multiple equivalent forms)
add_result1 = a + b                    # Operator overloading
add_result2 = torch.add(a, b)          # Function call
add_result3 = a.add(b)                 # Method call
a.add_(b)                              # In-place operation

# Subtraction
sub_result = a - b                     # Element-wise subtraction
sub_result = torch.sub(a, b)           # Function form

# Multiplication  
mul_result = a * b                     # Element-wise multiplication
mul_result = torch.mul(a, b)           # Function form

# Division
div_result = a / b                     # Element-wise division
div_result = torch.div(a, b)           # Function form

# Floor division
floor_div = a // b                     # Floor division
floor_div = torch.floor_divide(a, b)   # Function form

# Remainder/Modulo
remainder = a % b                      # Modulo operation
remainder = torch.remainder(a, b)      # Function form
```

**Power and Root Operations**:
```python
# Power operations
squared = a ** 2                       # Square using operator
squared = torch.pow(a, 2)              # Power function
sqrt_result = torch.sqrt(a)            # Square root
cube_root = torch.pow(a, 1/3)          # Cube root

# Exponential and logarithmic
exp_result = torch.exp(a)              # e^x
exp2_result = torch.exp2(a)            # 2^x
log_result = torch.log(a)              # Natural logarithm
log10_result = torch.log10(a)          # Base-10 logarithm
log2_result = torch.log2(a)            # Base-2 logarithm

# Advanced exponential functions
expm1_result = torch.expm1(a)          # exp(x) - 1 (numerically stable)
log1p_result = torch.log1p(a)          # log(1 + x) (numerically stable)

# Demonstration of numerical stability
small_values = torch.tensor([1e-8, 1e-10, 1e-12])
unstable = torch.log(1 + small_values) - small_values  # May lose precision
stable = torch.log1p(small_values) - small_values      # Numerically stable
print(f"Unstable computation: {unstable}")
print(f"Stable computation: {stable}")
```

### Trigonometric and Hyperbolic Functions

**Standard Trigonometric Functions**:
```python
# Create angle tensor (in radians)
angles = torch.tensor([0, math.pi/6, math.pi/4, math.pi/3, math.pi/2])

# Basic trigonometric functions
sin_values = torch.sin(angles)         # Sine
cos_values = torch.cos(angles)         # Cosine  
tan_values = torch.tan(angles)         # Tangent

# Verify trigonometric identity: sin²(x) + cos²(x) = 1
identity_check = sin_values**2 + cos_values**2
print(f"sin²(x) + cos²(x): {identity_check}")  # Should be all ones

# Inverse trigonometric functions
values = torch.tensor([0.0, 0.5, 0.707, 0.866, 1.0])
asin_values = torch.asin(values)       # Arcsine
acos_values = torch.acos(values)       # Arccosine
atan_values = torch.atan(values)       # Arctangent

# Two-argument arctangent (useful for angle computation)
y_coords = torch.tensor([1.0, 1.0, 0.0, -1.0])
x_coords = torch.tensor([1.0, 0.0, 1.0, -1.0])
angles_atan2 = torch.atan2(y_coords, x_coords)
```

**Hyperbolic Functions**:
```python
# Hyperbolic functions (important for activation functions)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

sinh_values = torch.sinh(x)            # Hyperbolic sine
cosh_values = torch.cosh(x)            # Hyperbolic cosine
tanh_values = torch.tanh(x)            # Hyperbolic tangent

# Inverse hyperbolic functions
asinh_values = torch.asinh(sinh_values)
acosh_values = torch.acosh(cosh_values)  # Note: cosh(x) >= 1 always
atanh_values = torch.atanh(tanh_values)  # Note: |tanh(x)| < 1 always

# Verify hyperbolic identity: cosh²(x) - sinh²(x) = 1
hyperbolic_identity = cosh_values**2 - sinh_values**2
print(f"cosh²(x) - sinh²(x): {hyperbolic_identity}")  # Should be all ones

# Relationship between hyperbolic and exponential functions
sinh_manual = (torch.exp(x) - torch.exp(-x)) / 2
cosh_manual = (torch.exp(x) + torch.exp(-x)) / 2
tanh_manual = sinh_manual / cosh_manual

print(f"sinh difference: {torch.abs(sinh_values - sinh_manual).max()}")
print(f"cosh difference: {torch.abs(cosh_values - cosh_manual).max()}")
print(f"tanh difference: {torch.abs(tanh_values - tanh_manual).max()}")
```

### Rounding and Discretization Operations

**Rounding Functions**:
```python
# Sample data with decimal values
decimal_data = torch.tensor([-2.7, -1.3, -0.5, 0.0, 0.5, 1.3, 2.7])

# Different rounding strategies
floor_result = torch.floor(decimal_data)      # Round down to nearest integer
ceil_result = torch.ceil(decimal_data)        # Round up to nearest integer
round_result = torch.round(decimal_data)      # Round to nearest integer
trunc_result = torch.trunc(decimal_data)      # Truncate decimal part

print(f"Original:  {decimal_data}")
print(f"Floor:     {floor_result}")
print(f"Ceil:      {ceil_result}")
print(f"Round:     {round_result}")
print(f"Truncate:  {trunc_result}")

# Fractional part extraction
frac_result = torch.frac(decimal_data)        # Fractional part
print(f"Fractional: {frac_result}")

# Rounding to specific decimal places
precise_data = torch.tensor([3.14159, 2.71828, 1.41421])
rounded_2_places = torch.round(precise_data * 100) / 100
print(f"Rounded to 2 decimal places: {rounded_2_places}")
```

**Sign and Absolute Value Operations**:
```python
# Sign-related operations
mixed_data = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])

abs_values = torch.abs(mixed_data)           # Absolute value
sign_values = torch.sign(mixed_data)         # Sign (-1, 0, or 1)
sgn_values = torch.sgn(mixed_data)           # Sign (handles complex numbers)

print(f"Original: {mixed_data}")
print(f"Absolute: {abs_values}")
print(f"Sign:     {sign_values}")

# Sign-based operations
negative_mask = torch.signbit(mixed_data)    # Boolean mask for negative values
positive_only = torch.where(negative_mask, torch.zeros_like(mixed_data), mixed_data)
print(f"Positive only: {positive_only}")

# Copysign: magnitude of first, sign of second
magnitude = torch.tensor([1.0, 2.0, 3.0])
sign_source = torch.tensor([-1.0, 1.0, -1.0])
copysign_result = torch.copysign(magnitude, sign_source)
print(f"Copysign result: {copysign_result}")  # [-1.0, 2.0, -3.0]
```

## Linear Algebra Operations

### Matrix Operations

**Matrix Multiplication Variants**:
```python
# Different types of matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.randn(3, 5)

# Standard matrix multiplication
matmul_result = torch.matmul(A, B)        # General matrix multiplication
matmul_result = A @ B                     # Operator form (Python 3.5+)

# Batch matrix multiplication
batch_A = torch.randn(10, 3, 4)          # Batch of 3x4 matrices
batch_B = torch.randn(10, 4, 5)          # Batch of 4x5 matrices
batch_result = torch.bmm(batch_A, batch_B)  # Batch matrix multiplication

# More general batch matrix multiplication
batch_A_broadcast = torch.randn(1, 3, 4)      # Broadcasts across batch
batch_B_multi = torch.randn(10, 4, 5)         # Multiple batches
general_batch = torch.matmul(batch_A_broadcast, batch_B_multi)

# Matrix-vector multiplication
vector = torch.randn(4)
mv_result = torch.mv(A, vector)               # Matrix-vector product

# Outer product
vec1 = torch.randn(3)
vec2 = torch.randn(5)
outer_result = torch.outer(vec1, vec2)        # Outer product
```

**Advanced Matrix Operations**:
```python
# Matrix properties and transformations
square_matrix = torch.randn(4, 4)

# Transpose operations
transpose_result = square_matrix.t()          # 2D transpose
transpose_result = torch.transpose(square_matrix, 0, 1)  # General transpose

# For higher dimensional tensors
tensor_3d = torch.randn(2, 3, 4)
transposed_3d = torch.transpose(tensor_3d, 1, 2)  # Swap dimensions 1 and 2

# Matrix trace (sum of diagonal elements)
trace_result = torch.trace(square_matrix)

# Diagonal operations
diagonal_elements = torch.diag(square_matrix)     # Extract diagonal
diagonal_matrix = torch.diag(diagonal_elements)   # Create diagonal matrix

# Matrix determinant
det_result = torch.det(square_matrix)

# Matrix rank (using SVD)
rank_result = torch.linalg.matrix_rank(square_matrix)

print(f"Matrix shape: {square_matrix.shape}")
print(f"Trace: {trace_result}")
print(f"Determinant: {det_result}")
print(f"Rank: {rank_result}")
```

### Decompositions and Factorizations

**Singular Value Decomposition (SVD)**:
```python
# SVD: A = U * Σ * V^T
matrix_for_svd = torch.randn(5, 3)

# Full SVD
U, S, Vt = torch.linalg.svd(matrix_for_svd, full_matrices=True)
print(f"Original shape: {matrix_for_svd.shape}")
print(f"U shape: {U.shape}")      # Left singular vectors
print(f"S shape: {S.shape}")      # Singular values
print(f"Vt shape: {Vt.shape}")    # Right singular vectors (transposed)

# Verify reconstruction
reconstructed = U @ torch.diag_embed(S) @ Vt
reconstruction_error = torch.norm(matrix_for_svd - reconstructed[:, :3])
print(f"Reconstruction error: {reconstruction_error}")

# Reduced SVD (economy size)
U_reduced, S_reduced, Vt_reduced = torch.linalg.svd(matrix_for_svd, full_matrices=False)
print(f"Reduced U shape: {U_reduced.shape}")
print(f"Reduced Vt shape: {Vt_reduced.shape}")

# SVD applications
def low_rank_approximation(matrix, rank):
    """Create low-rank approximation using SVD"""
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    # Keep only top 'rank' singular values
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    Vt_truncated = Vt[:rank, :]
    return U_truncated @ torch.diag(S_truncated) @ Vt_truncated

# Example of low-rank approximation
original_matrix = torch.randn(10, 8)
rank_3_approx = low_rank_approximation(original_matrix, 3)
approximation_error = torch.norm(original_matrix - rank_3_approx)
print(f"Rank-3 approximation error: {approximation_error}")
```

**QR Decomposition**:
```python
# QR decomposition: A = Q * R
matrix_for_qr = torch.randn(6, 4)

# QR decomposition
Q, R = torch.linalg.qr(matrix_for_qr, mode='reduced')
print(f"Original shape: {matrix_for_qr.shape}")
print(f"Q shape: {Q.shape}")      # Orthogonal matrix
print(f"R shape: {R.shape}")      # Upper triangular matrix

# Verify orthogonality of Q
orthogonality_check = Q.t() @ Q
identity_error = torch.norm(orthogonality_check - torch.eye(Q.shape[1]))
print(f"Q orthogonality error: {identity_error}")

# Verify reconstruction
qr_reconstructed = Q @ R
qr_reconstruction_error = torch.norm(matrix_for_qr - qr_reconstructed)
print(f"QR reconstruction error: {qr_reconstruction_error}")

# Applications of QR decomposition
def solve_linear_least_squares(A, b):
    """Solve least squares problem using QR decomposition"""
    Q, R = torch.linalg.qr(A)
    # Solve R * x = Q^T * b
    Qtb = Q.t() @ b
    x = torch.linalg.solve_triangular(R, Qtb, upper=True)
    return x

# Example least squares problem
A_ls = torch.randn(10, 3)  # Overdetermined system
b_ls = torch.randn(10)
x_solution = solve_linear_least_squares(A_ls, b_ls)
residual = torch.norm(A_ls @ x_solution - b_ls)
print(f"Least squares residual: {residual}")
```

**Eigenvalue Decomposition**:
```python
# Eigenvalue decomposition for symmetric matrices
symmetric_matrix = torch.randn(4, 4)
symmetric_matrix = (symmetric_matrix + symmetric_matrix.t()) / 2  # Make symmetric

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors shape: {eigenvectors.shape}")

# Verify eigenvalue equation: A * v = λ * v
for i in range(len(eigenvalues)):
    Av = symmetric_matrix @ eigenvectors[:, i]
    lambda_v = eigenvalues[i] * eigenvectors[:, i]
    error = torch.norm(Av - lambda_v)
    print(f"Eigenvalue {i} error: {error}")

# Reconstruct matrix using eigendecomposition
reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
reconstruction_error = torch.norm(symmetric_matrix - reconstructed)
print(f"Eigendecomposition reconstruction error: {reconstruction_error}")

# For general (non-symmetric) matrices
general_matrix = torch.randn(4, 4)
eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(general_matrix)
print(f"Complex eigenvalues shape: {eigenvalues_complex.shape}")
print(f"Some eigenvalues: {eigenvalues_complex[:3]}")
```

### System Solving and Matrix Inversion

**Linear System Solving**:
```python
# Solve Ax = b for various cases
A_square = torch.randn(4, 4)
b_vector = torch.randn(4)

# Standard linear system solving
x_solution = torch.linalg.solve(A_square, b_vector)

# Verify solution
verification = A_square @ x_solution
solution_error = torch.norm(b_vector - verification)
print(f"Linear system solution error: {solution_error}")

# Multiple right-hand sides
B_multiple = torch.randn(4, 3)  # 3 different b vectors
X_multiple = torch.linalg.solve(A_square, B_multiple)

# Triangular system solving (more efficient when applicable)
upper_triangular = torch.triu(torch.randn(4, 4))  # Upper triangular
lower_triangular = torch.tril(torch.randn(4, 4))  # Lower triangular

x_upper = torch.linalg.solve_triangular(upper_triangular, b_vector, upper=True)
x_lower = torch.linalg.solve_triangular(lower_triangular, b_vector, upper=False)
```

**Matrix Inversion and Pseudoinverse**:
```python
# Matrix inversion
invertible_matrix = torch.randn(4, 4)
# Ensure matrix is well-conditioned by adding to diagonal
invertible_matrix += torch.eye(4) * 0.1

# Compute inverse
matrix_inverse = torch.linalg.inv(invertible_matrix)

# Verify inversion
identity_check = invertible_matrix @ matrix_inverse
identity_error = torch.norm(identity_check - torch.eye(4))
print(f"Matrix inversion error: {identity_error}")

# Pseudoinverse for non-square matrices
rectangular_matrix = torch.randn(6, 4)
pseudoinverse = torch.linalg.pinv(rectangular_matrix)
print(f"Original shape: {rectangular_matrix.shape}")
print(f"Pseudoinverse shape: {pseudoinverse.shape}")

# Moore-Penrose pseudoinverse properties
# For overdetermined systems: pinv(A) = (A^T A)^(-1) A^T
manual_pinv = torch.linalg.inv(rectangular_matrix.t() @ rectangular_matrix) @ rectangular_matrix.t()
pinv_difference = torch.norm(pseudoinverse - manual_pinv)
print(f"Pseudoinverse computation difference: {pinv_difference}")
```

## Statistical Operations

### Descriptive Statistics

**Central Tendency Measures**:
```python
# Create sample data with different distributions
normal_data = torch.randn(1000, 10) * 2 + 5    # Normal distribution
uniform_data = torch.rand(1000, 10) * 10       # Uniform distribution
exponential_data = torch.exponential(torch.ones(1000, 10))

def comprehensive_statistics(data, name):
    """Compute comprehensive statistics for tensor data"""
    print(f"\n{name} Statistics:")
    print(f"Shape: {data.shape}")
    
    # Central tendency
    mean_val = torch.mean(data, dim=0)
    median_val = torch.median(data, dim=0).values
    mode_val = torch.mode(data, dim=0).values if data.dtype == torch.long else "N/A (continuous data)"
    
    print(f"Mean: {mean_val.mean():.4f} ± {mean_val.std():.4f}")
    print(f"Median: {median_val.mean():.4f} ± {median_val.std():.4f}")
    
    # Variability measures
    var_val = torch.var(data, dim=0)
    std_val = torch.std(data, dim=0)
    
    print(f"Variance: {var_val.mean():.4f}")
    print(f"Standard deviation: {std_val.mean():.4f}")
    
    # Range and quantiles
    min_val = torch.min(data, dim=0).values
    max_val = torch.max(data, dim=0).values
    range_val = max_val - min_val
    
    print(f"Range: {range_val.mean():.4f}")
    print(f"Min: {min_val.mean():.4f}")
    print(f"Max: {max_val.mean():.4f}")
    
    # Quantiles
    q25 = torch.quantile(data, 0.25, dim=0)
    q75 = torch.quantile(data, 0.75, dim=0)
    iqr = q75 - q25
    
    print(f"IQR: {iqr.mean():.4f}")
    print(f"25th percentile: {q25.mean():.4f}")
    print(f"75th percentile: {q75.mean():.4f}")

# Analyze different distributions
comprehensive_statistics(normal_data, "Normal")
comprehensive_statistics(uniform_data, "Uniform")
comprehensive_statistics(exponential_data, "Exponential")
```

**Advanced Statistical Measures**:
```python
# Higher-order moments and distribution shape
def distribution_shape_analysis(data, name):
    """Analyze distribution shape characteristics"""
    print(f"\n{name} Distribution Shape:")
    
    # Flatten for overall analysis
    flat_data = data.flatten()
    
    # Central moments
    mean = torch.mean(flat_data)
    centered = flat_data - mean
    
    # Second moment (variance)
    second_moment = torch.mean(centered ** 2)
    
    # Third moment (skewness-related)
    third_moment = torch.mean(centered ** 3)
    skewness = third_moment / (second_moment ** 1.5)
    
    # Fourth moment (kurtosis-related)
    fourth_moment = torch.mean(centered ** 4)
    kurtosis = fourth_moment / (second_moment ** 2) - 3  # Excess kurtosis
    
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    
    # Interpretations
    if abs(skewness) < 0.5:
        skew_interp = "approximately symmetric"
    elif skewness > 0.5:
        skew_interp = "right-skewed (positive skew)"
    else:
        skew_interp = "left-skewed (negative skew)"
    
    if abs(kurtosis) < 0.5:
        kurt_interp = "approximately normal kurtosis"
    elif kurtosis > 0.5:
        kurt_interp = "heavy-tailed (leptokurtic)"
    else:
        kurt_interp = "light-tailed (platykurtic)"
    
    print(f"Interpretation: {skew_interp}, {kurt_interp}")

distribution_shape_analysis(normal_data, "Normal")
distribution_shape_analysis(uniform_data, "Uniform")
distribution_shape_analysis(exponential_data, "Exponential")
```

### Correlation and Covariance

**Correlation Analysis**:
```python
# Generate correlated data
n_samples = 1000
n_features = 5

# Create correlation structure
correlation_matrix = torch.eye(n_features)
correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.8  # Strong positive
correlation_matrix[2, 3] = correlation_matrix[3, 2] = -0.6  # Moderate negative
correlation_matrix[1, 4] = correlation_matrix[4, 1] = 0.3   # Weak positive

# Generate correlated data using Cholesky decomposition
L = torch.linalg.cholesky(correlation_matrix)
uncorrelated = torch.randn(n_samples, n_features)
correlated_data = uncorrelated @ L.t()

# Compute sample correlation matrix
def correlation_matrix_from_data(data):
    """Compute correlation matrix from data"""
    # Center the data
    centered = data - torch.mean(data, dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov_matrix = (centered.t() @ centered) / (data.shape[0] - 1)
    
    # Convert to correlation matrix
    std_devs = torch.sqrt(torch.diag(cov_matrix))
    correlation = cov_matrix / torch.outer(std_devs, std_devs)
    
    return correlation

computed_correlation = correlation_matrix_from_data(correlated_data)
print("Original correlation matrix:")
print(correlation_matrix)
print("\nComputed correlation matrix:")
print(computed_correlation)
print("\nDifference:")
print(torch.abs(correlation_matrix - computed_correlation))

# Built-in correlation function
torch_correlation = torch.corrcoef(correlated_data.t())
print("\nPyTorch corrcoef result:")
print(torch_correlation)
```

**Covariance Analysis**:
```python
# Covariance matrix computation and analysis
def covariance_analysis(data, feature_names=None):
    """Comprehensive covariance analysis"""
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
    
    # Compute covariance matrix
    cov_matrix = torch.cov(data.t())
    
    print("Covariance Matrix:")
    print(cov_matrix)
    
    # Eigenanalysis of covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print(f"\nEigenvalues: {eigenvalues}")
    
    # Explained variance ratios
    total_variance = torch.sum(eigenvalues)
    explained_ratios = eigenvalues / total_variance
    cumulative_explained = torch.cumsum(explained_ratios, dim=0)
    
    print("\nPrincipal Component Analysis:")
    for i, (eigenval, ratio, cumulative) in enumerate(zip(eigenvalues, explained_ratios, cumulative_explained)):
        print(f"PC{i+1}: λ={eigenval:.4f}, explains {ratio:.3%}, cumulative {cumulative:.3%}")
    
    # Condition number (ratio of largest to smallest eigenvalue)
    condition_number = eigenvalues[0] / eigenvalues[-1]
    print(f"\nCondition number: {condition_number:.4f}")
    
    if condition_number > 30:
        print("Warning: High condition number indicates multicollinearity")
    
    return cov_matrix, eigenvalues, eigenvectors

# Analyze our correlated data
cov_matrix, eigenvals, eigenvecs = covariance_analysis(correlated_data)
```

### Distribution Operations

**Probability Distributions and Sampling**:
```python
# Working with probability distributions
from torch.distributions import Normal, Uniform, Exponential, Bernoulli, MultivariateNormal

# Normal distribution operations
normal_dist = Normal(loc=0.0, scale=1.0)
normal_samples = normal_dist.sample((1000,))

# Probability density function
x_values = torch.linspace(-3, 3, 100)
pdf_values = normal_dist.log_prob(x_values).exp()  # PDF values

# Cumulative distribution function
cdf_values = normal_dist.cdf(x_values)

print(f"Normal samples mean: {normal_samples.mean():.4f}")
print(f"Normal samples std: {normal_samples.std():.4f}")

# Multivariate normal distribution
mean_vector = torch.zeros(3)
covariance_matrix = torch.tensor([[1.0, 0.5, 0.2],
                                  [0.5, 2.0, -0.3],
                                  [0.2, -0.3, 1.5]])

mvn_dist = MultivariateNormal(mean_vector, covariance_matrix)
mvn_samples = mvn_dist.sample((500,))

# Verify sample statistics match distribution parameters
sample_mean = torch.mean(mvn_samples, dim=0)
sample_cov = torch.cov(mvn_samples.t())

print(f"\nMultivariate Normal:")
print(f"True mean: {mean_vector}")
print(f"Sample mean: {sample_mean}")
print(f"Mean difference: {torch.norm(mean_vector - sample_mean):.4f}")
print(f"\nCovariance difference norm: {torch.norm(covariance_matrix - sample_cov):.4f}")
```

**Hypothesis Testing and Statistical Tests**:
```python
# Statistical hypothesis testing utilities
def two_sample_t_test(sample1, sample2, alpha=0.05):
    """Perform two-sample t-test for equal means"""
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = torch.mean(sample1), torch.mean(sample2)
    var1, var2 = torch.var(sample1, unbiased=True), torch.var(sample2, unbiased=True)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # Standard error
    se = torch.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # t-statistic
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    print(f"Two-sample t-test results:")
    print(f"Sample 1 mean: {mean1:.4f}, Sample 2 mean: {mean2:.4f}")
    print(f"Difference: {mean1 - mean2:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"Degrees of freedom: {df}")
    
    # Critical value (approximation for large df)
    if df > 30:
        critical_value = 1.96 if alpha == 0.05 else 2.58  # Normal approximation
    else:
        # Would need t-distribution implementation for exact critical values
        critical_value = 2.0  # Rough approximation
    
    reject_null = abs(t_stat) > critical_value
    print(f"Critical value (±): {critical_value:.4f}")
    print(f"Reject null hypothesis: {reject_null}")
    
    return t_stat, reject_null

# Example hypothesis test
group1 = torch.randn(50) + 1.0  # Mean ≈ 1
group2 = torch.randn(50) + 1.5  # Mean ≈ 1.5

t_stat, significant = two_sample_t_test(group1, group2)

# One-sample tests
def one_sample_t_test(sample, mu0, alpha=0.05):
    """One-sample t-test against hypothesized mean"""
    n = len(sample)
    sample_mean = torch.mean(sample)
    sample_std = torch.std(sample, unbiased=True)
    
    # t-statistic
    t_stat = (sample_mean - mu0) / (sample_std / torch.sqrt(torch.tensor(float(n))))
    
    print(f"\nOne-sample t-test (H0: μ = {mu0}):")
    print(f"Sample mean: {sample_mean:.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    
    # Simple critical value (normal approximation)
    critical_value = 1.96 if alpha == 0.05 else 2.58
    reject_null = abs(t_stat) > critical_value
    print(f"Reject null hypothesis: {reject_null}")
    
    return t_stat, reject_null

# Test if group1 mean is significantly different from 0
one_sample_t_test(group1, 0.0)
```

## Reduction Operations

### Aggregation Functions

**Sum and Product Operations**:
```python
# Create multi-dimensional data for reduction examples
data_3d = torch.randn(4, 5, 6)
print(f"Original shape: {data_3d.shape}")

# Sum operations along different dimensions
sum_all = torch.sum(data_3d)                    # Sum all elements (scalar)
sum_dim0 = torch.sum(data_3d, dim=0)            # Sum along dimension 0
sum_dim1 = torch.sum(data_3d, dim=1)            # Sum along dimension 1
sum_dim2 = torch.sum(data_3d, dim=2)            # Sum along dimension 2

print(f"Sum all elements: {sum_all} (shape: {sum_all.shape})")
print(f"Sum dim 0: shape {sum_dim0.shape}")
print(f"Sum dim 1: shape {sum_dim1.shape}")
print(f"Sum dim 2: shape {sum_dim2.shape}")

# Multiple dimensions
sum_dims_01 = torch.sum(data_3d, dim=[0, 1])    # Sum along dims 0 and 1
sum_dims_12 = torch.sum(data_3d, dim=[1, 2])    # Sum along dims 1 and 2

print(f"Sum dims [0,1]: shape {sum_dims_01.shape}")
print(f"Sum dims [1,2]: shape {sum_dims_12.shape}")

# Keepdim parameter
sum_keepdim = torch.sum(data_3d, dim=1, keepdim=True)
print(f"Sum dim 1 keepdim: shape {sum_keepdim.shape}")

# Product operations
prod_all = torch.prod(data_3d)                  # Product of all elements
prod_dim0 = torch.prod(data_3d, dim=0)          # Product along dimension 0

# Cumulative operations
cumsum_result = torch.cumsum(data_3d, dim=2)    # Cumulative sum along dim 2
cumprod_result = torch.cumprod(torch.abs(data_3d), dim=1)  # Cumulative product
```

**Mean and Variance Operations**:
```python
# Statistical reduction operations
batch_data = torch.randn(32, 10, 20)  # Batch of 32 samples, each 10x20

# Mean operations
mean_all = torch.mean(batch_data)                    # Overall mean
mean_batch = torch.mean(batch_data, dim=0)           # Mean across batch
mean_spatial = torch.mean(batch_data, dim=[1, 2])    # Mean across spatial dims

print(f"Overall mean: {mean_all:.4f}")
print(f"Batch mean shape: {mean_batch.shape}")
print(f"Spatial mean shape: {mean_spatial.shape}")

# Variance and standard deviation
var_batch = torch.var(batch_data, dim=0, unbiased=True)     # Unbiased variance
std_batch = torch.std(batch_data, dim=0, unbiased=True)     # Unbiased std

# Along multiple dimensions with unbiased correction
var_spatial = torch.var(batch_data, dim=[1, 2], unbiased=True)
std_spatial = torch.std(batch_data, dim=[1, 2], unbiased=True)

print(f"Spatial variance shape: {var_spatial.shape}")
print(f"Spatial std shape: {std_spatial.shape}")

# Mean and variance simultaneously (more efficient)
mean_and_var = torch.var_mean(batch_data, dim=0, unbiased=True)
var_result, mean_result = mean_and_var

print(f"Combined operation - Mean shape: {mean_result.shape}, Var shape: {var_result.shape}")
```

### Min/Max and Argmin/Argmax Operations

**Extrema Finding**:
```python
# Create data with known extrema for verification
structured_data = torch.tensor([
    [[1, 8, 3], [4, 2, 9]],
    [[7, 5, 6], [1, 3, 4]]
])
print(f"Structured data shape: {structured_data.shape}")
print("Structured data:")
print(structured_data)

# Min and max operations
min_all = torch.min(structured_data)
max_all = torch.max(structured_data)
print(f"\nGlobal min: {min_all}, max: {max_all}")

# Min/max along specific dimensions
min_dim0, argmin_dim0 = torch.min(structured_data, dim=0)
max_dim0, argmax_dim0 = torch.max(structured_data, dim=0)

print(f"\nMin along dim 0:")
print(min_dim0)
print(f"Argmin along dim 0:")
print(argmin_dim0)

print(f"\nMax along dim 0:")
print(max_dim0)
print(f"Argmax along dim 0:")
print(argmax_dim0)

# Verification of argmin/argmax results
print("\nVerification:")
for i in range(structured_data.shape[1]):
    for j in range(structured_data.shape[2]):
        argmin_idx = argmin_dim0[i, j].item()
        argmax_idx = argmax_dim0[i, j].item()
        
        min_val = structured_data[argmin_idx, i, j]
        max_val = structured_data[argmax_idx, i, j]
        
        print(f"Position ({i},{j}): min={min_val} at index {argmin_idx}, max={max_val} at index {argmax_idx}")

# Global argmin/argmax (flattened indices)
global_argmin = torch.argmin(structured_data)
global_argmax = torch.argmax(structured_data)

# Convert flat indices to multi-dimensional indices
def unravel_index(index, shape):
    """Convert flat index to multi-dimensional indices"""
    indices = []
    for dim_size in reversed(shape):
        indices.append(index % dim_size)
        index = index // dim_size
    return list(reversed(indices))

global_argmin_coords = unravel_index(global_argmin.item(), structured_data.shape)
global_argmax_coords = unravel_index(global_argmax.item(), structured_data.shape)

print(f"\nGlobal argmin: {global_argmin} -> coordinates {global_argmin_coords}")
print(f"Global argmax: {global_argmax} -> coordinates {global_argmax_coords}")
print(f"Values: min={structured_data[tuple(global_argmin_coords)]}, max={structured_data[tuple(global_argmax_coords)]}")
```

**Top-k and Sorting Operations**:
```python
# Top-k operations for finding multiple extrema
random_data = torch.randn(5, 8)
k = 3

# Top-k largest values
topk_values, topk_indices = torch.topk(random_data, k, dim=1, largest=True)
print(f"Original data shape: {random_data.shape}")
print(f"Top-{k} values shape: {topk_values.shape}")
print(f"Top-{k} indices shape: {topk_indices.shape}")

print(f"\nFirst row data: {random_data[0]}")
print(f"Top-{k} values: {topk_values[0]}")
print(f"Top-{k} indices: {topk_indices[0]}")

# Verify top-k results
first_row = random_data[0]
top_indices = topk_indices[0]
print("Verification:")
for i, idx in enumerate(top_indices):
    print(f"  Rank {i+1}: value {first_row[idx]:.4f} at index {idx}")

# Bottom-k (smallest values)
bottomk_values, bottomk_indices = torch.topk(random_data, k, dim=1, largest=False)
print(f"\nBottom-{k} values: {bottomk_values[0]}")
print(f"Bottom-{k} indices: {bottomk_indices[0]}")

# Full sorting
sorted_values, sorted_indices = torch.sort(random_data, dim=1, descending=True)
print(f"\nFull sort shapes - values: {sorted_values.shape}, indices: {sorted_indices.shape}")

# Verify that top-k matches first k elements of full sort
topk_matches_sort = torch.allclose(topk_values, sorted_values[:, :k])
indices_match = torch.equal(topk_indices, sorted_indices[:, :k])
print(f"Top-k values match sort: {topk_matches_sort}")
print(f"Top-k indices match sort: {indices_match}")

# k-th order statistics
def kth_order_statistic(data, k, dim=-1):
    """Find the k-th order statistic along a dimension"""
    # k-th smallest element (0-indexed)
    kth_value = torch.kthvalue(data, k + 1, dim=dim)  # kthvalue uses 1-indexing
    return kth_value.values, kth_value.indices

# Find median (middle element)
median_k = random_data.shape[1] // 2
median_values, median_indices = kth_order_statistic(random_data, median_k, dim=1)
print(f"\nMedian values (k={median_k}): {median_values}")

# Compare with torch.median
torch_median = torch.median(random_data, dim=1)
median_close = torch.allclose(median_values, torch_median.values)
print(f"Median matches torch.median: {median_close}")
```

## Key Questions for Review

### Mathematical Operations
1. **Numerical Stability**: Why are functions like `torch.log1p` and `torch.expm1` important for numerical stability, and when should they be used?

2. **Broadcasting in Operations**: How does broadcasting affect the computational complexity and memory usage of element-wise operations?

3. **Trigonometric Identities**: How can you verify trigonometric identities using PyTorch operations, and why might small numerical errors occur?

### Linear Algebra
4. **Matrix Decompositions**: When should you use SVD vs QR vs eigendecomposition, and what are the computational trade-offs?

5. **Condition Numbers**: What does a high condition number indicate about a matrix, and how does it affect numerical computations?

6. **Pseudoinverse Applications**: In what scenarios is the pseudoinverse more appropriate than regular matrix inversion?

### Statistical Operations
7. **Biased vs Unbiased Estimators**: Why does PyTorch provide both biased and unbiased versions of variance and standard deviation, and when should each be used?

8. **Correlation vs Covariance**: What is the relationship between correlation and covariance matrices, and how do they differ in interpretation?

9. **Principal Component Analysis**: How do eigenvalues and eigenvectors of the covariance matrix relate to principal components and explained variance?

### Reduction Operations
10. **Dimension Preservation**: When should you use the `keepdim` parameter in reduction operations, and how does it affect subsequent operations?

11. **Multiple Dimension Reduction**: What are the implications of reducing along multiple dimensions simultaneously vs sequentially?

12. **Memory Efficiency**: How do reduction operations affect memory usage, especially for large tensors?

## Advanced Tensor Manipulation Patterns

### Shape Manipulation and Broadcasting

**Advanced Broadcasting Techniques**:
```python
# Complex broadcasting scenarios
def demonstrate_advanced_broadcasting():
    """Demonstrate sophisticated broadcasting patterns"""
    
    # Multi-dimensional broadcasting
    batch_size, seq_len, embed_dim = 32, 128, 512
    
    # Embeddings: (batch_size, seq_len, embed_dim)
    embeddings = torch.randn(batch_size, seq_len, embed_dim)
    
    # Position encodings: (1, seq_len, embed_dim) - broadcasts across batches
    pos_encodings = torch.randn(1, seq_len, embed_dim)
    
    # Layer norm scaling: (embed_dim,) - broadcasts across batch and sequence
    layer_norm_scale = torch.randn(embed_dim)
    
    # Attention mask: (batch_size, 1, seq_len, seq_len) - broadcasts across heads
    attention_mask = torch.randn(batch_size, 1, seq_len, seq_len)
    
    # Complex broadcasting operation
    # Add position encodings (broadcasts across batch)
    embedded_with_pos = embeddings + pos_encodings
    
    # Apply layer norm scaling (broadcasts across batch and sequence)
    normalized = embedded_with_pos * layer_norm_scale.unsqueeze(0).unsqueeze(0)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Position encodings shape: {pos_encodings.shape}")
    print(f"Layer norm scale shape: {layer_norm_scale.shape}")
    print(f"Result shape: {normalized.shape}")
    
    # Verify broadcasting rules
    print(f"Broadcasting successful: {normalized.shape == embeddings.shape}")
    
    return normalized

advanced_broadcast_result = demonstrate_advanced_broadcasting()

# Einstein summation for complex operations
def einstein_summation_examples():
    """Demonstrate Einstein summation notation in PyTorch"""
    
    # Batch matrix multiplication using einsum
    A = torch.randn(32, 10, 15)  # Batch of 10x15 matrices
    B = torch.randn(32, 15, 20)  # Batch of 15x20 matrices
    
    # Traditional batch matrix multiplication
    C_bmm = torch.bmm(A, B)
    
    # Einstein summation equivalent
    C_einsum = torch.einsum('bij,bjk->bik', A, B)
    
    print(f"Batch matmul shapes: A{A.shape} @ B{B.shape} = C{C_bmm.shape}")
    print(f"Results match: {torch.allclose(C_bmm, C_einsum)}")
    
    # More complex: bilinear attention
    queries = torch.randn(32, 128, 64)    # (batch, seq_len, d_model)
    keys = torch.randn(32, 128, 64)       # (batch, seq_len, d_model)  
    bilinear_weight = torch.randn(64, 64) # (d_model, d_model)
    
    # Bilinear attention: Q * W * K^T
    attention_scores = torch.einsum('bqd,de,bke->bqk', queries, bilinear_weight, keys)
    print(f"Bilinear attention shape: {attention_scores.shape}")
    
    # Verify with manual computation
    manual_scores = queries @ bilinear_weight @ keys.transpose(1, 2)
    print(f"Manual computation matches: {torch.allclose(attention_scores, manual_scores)}")
    
    # Tensor contraction example
    tensor_4d = torch.randn(5, 6, 7, 8)
    
    # Contract dimensions 1 and 2
    contracted = torch.einsum('ijkl,ikml->ijml', tensor_4d, tensor_4d)
    print(f"Tensor contraction result shape: {contracted.shape}")

einstein_summation_examples()
```

### Memory-Efficient Operations

**In-place Operation Patterns**:
```python
class MemoryEfficientOperations:
    """Collection of memory-efficient tensor operation patterns"""
    
    @staticmethod
    def efficient_normalization(tensor, epsilon=1e-8):
        """Memory-efficient layer normalization"""
        # Compute statistics
        mean = torch.mean(tensor, dim=-1, keepdim=True)
        var = torch.var(tensor, dim=-1, keepdim=True, unbiased=False)
        
        # In-place operations to save memory
        tensor.sub_(mean)  # Subtract mean in-place
        tensor.div_(torch.sqrt(var + epsilon))  # Divide by std in-place
        
        return tensor
    
    @staticmethod
    def efficient_softmax(tensor, dim=-1):
        """Numerically stable and memory-efficient softmax"""
        # Subtract max for numerical stability (in-place)
        max_vals = torch.max(tensor, dim=dim, keepdim=True)[0]
        tensor.sub_(max_vals)
        
        # Compute exponential in-place
        tensor.exp_()
        
        # Normalize in-place
        sum_vals = torch.sum(tensor, dim=dim, keepdim=True)
        tensor.div_(sum_vals)
        
        return tensor
    
    @staticmethod
    def efficient_gradient_clipping(parameters, max_norm):
        """Memory-efficient gradient clipping"""
        # Compute total norm of gradients
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            for param in parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    @staticmethod
    def efficient_moving_average(current_avg, new_value, momentum):
        """Efficient exponential moving average update"""
        # In-place update: avg = momentum * avg + (1 - momentum) * new_value
        current_avg.mul_(momentum).add_(new_value, alpha=(1 - momentum))
        return current_avg

# Demonstrate memory-efficient operations
def memory_efficiency_demo():
    """Demonstrate memory efficiency techniques"""
    
    # Test efficient normalization
    test_tensor = torch.randn(1000, 512)
    original_tensor = test_tensor.clone()
    
    # Track memory before operation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        test_tensor = test_tensor.cuda()
        original_tensor = original_tensor.cuda()
    
    # Apply efficient normalization
    normalized = MemoryEfficientOperations.efficient_normalization(test_tensor.clone())
    
    # Verify correctness
    expected_mean = torch.mean(normalized, dim=-1)
    expected_std = torch.std(normalized, dim=-1)
    
    print(f"Normalized tensor mean (should be ~0): {expected_mean.abs().mean():.6f}")
    print(f"Normalized tensor std (should be ~1): {(expected_std - 1).abs().mean():.6f}")
    
    # Test efficient softmax
    logits = torch.randn(64, 1000)
    if torch.cuda.is_available():
        logits = logits.cuda()
    
    softmax_result = MemoryEfficientOperations.efficient_softmax(logits.clone())
    
    # Verify softmax properties
    row_sums = torch.sum(softmax_result, dim=-1)
    all_positive = torch.all(softmax_result >= 0)
    
    print(f"Softmax row sums (should be 1): {row_sums.mean():.6f} ± {row_sums.std():.6f}")
    print(f"All values positive: {all_positive}")

memory_efficiency_demo()
```

### Advanced Indexing and Selection

**Complex Indexing Patterns**:
```python
def advanced_indexing_patterns():
    """Demonstrate sophisticated indexing and selection patterns"""
    
    # Multi-dimensional advanced indexing
    batch_size, seq_len, vocab_size = 32, 128, 50000
    
    # Logits from language model
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Token indices for each position
    token_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Extract logits for actual tokens (for loss computation)
    batch_indices = torch.arange(batch_size).unsqueeze(1)  # (32, 1)
    seq_indices = torch.arange(seq_len).unsqueeze(0)       # (1, 128)
    
    # Advanced indexing to select specific logits
    selected_logits = logits[batch_indices, seq_indices, token_indices]
    print(f"Selected logits shape: {selected_logits.shape}")
    
    # Alternative using gather
    gathered_logits = torch.gather(logits, dim=2, index=token_indices.unsqueeze(2)).squeeze(2)
    print(f"Gathered logits shape: {gathered_logits.shape}")
    print(f"Results match: {torch.allclose(selected_logits, gathered_logits)}")
    
    # Scatter operations for one-hot encoding
    num_classes = 10
    class_indices = torch.randint(0, num_classes, (100,))
    one_hot = torch.zeros(100, num_classes)
    one_hot.scatter_(1, class_indices.unsqueeze(1), 1)
    
    # Verify one-hot encoding
    print(f"One-hot sum per row: {one_hot.sum(dim=1).unique()}")  # Should be all 1s
    print(f"Correct class selected: {torch.all(one_hot[torch.arange(100), class_indices] == 1)}")
    
    # Advanced boolean indexing
    data_matrix = torch.randn(1000, 100)
    
    # Complex boolean conditions
    outlier_threshold = 2.0
    outlier_mask = (torch.abs(data_matrix) > outlier_threshold).any(dim=1)
    
    # Select rows without outliers
    clean_data = data_matrix[~outlier_mask]
    print(f"Original data shape: {data_matrix.shape}")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Removed {outlier_mask.sum()} outlier rows")
    
    # Conditional selection with where
    clipped_data = torch.where(
        torch.abs(data_matrix) > outlier_threshold,
        torch.sign(data_matrix) * outlier_threshold,
        data_matrix
    )
    
    # Verify clipping
    max_abs_value = torch.abs(clipped_data).max()
    print(f"Maximum absolute value after clipping: {max_abs_value:.4f}")
    print(f"Clipping successful: {max_abs_value <= outlier_threshold}")

advanced_indexing_patterns()
```

## Conclusion

Advanced tensor operations form the computational foundation of deep learning algorithms. This comprehensive exploration covers the mathematical, statistical, and linear algebra operations that enable sophisticated neural network computations. Understanding these operations deeply—from basic element-wise arithmetic to complex linear algebra decompositions—is essential for implementing efficient and numerically stable deep learning systems.

**Key Takeaways**:

**Mathematical Foundation**: Element-wise operations, trigonometric functions, and advanced mathematical operations provide the computational primitives for neural network forward and backward passes.

**Linear Algebra Mastery**: Matrix operations, decompositions, and system solving techniques are fundamental to understanding and implementing advanced neural network architectures and optimization algorithms.

**Statistical Computing**: Statistical operations enable data analysis, hypothesis testing, and the implementation of statistical learning algorithms.

**Reduction Operations**: Aggregation functions are crucial for computing loss functions, gradients, and summary statistics across different tensor dimensions.

**Performance Optimization**: Understanding memory-efficient operation patterns, broadcasting rules, and advanced indexing techniques is essential for building scalable deep learning systems.

**Numerical Stability**: Proper use of numerically stable algorithms and understanding of floating-point arithmetic limitations prevents common numerical issues in deep learning training.

The operations covered in this module provide the computational building blocks for implementing state-of-the-art deep learning algorithms, from basic neural networks to sophisticated architectures like transformers and graph neural networks. Mastery of these tensor operations is fundamental to becoming an effective deep learning practitioner and researcher.