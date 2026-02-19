# 2. Linear Algebra for Machine Learning — The Math That Powers LoRA

## Table of Contents
- [Why Linear Algebra?](#why-linear-algebra)
- [Vectors: The Building Blocks](#vectors-the-building-blocks)
- [Dot Product: Measuring Similarity](#dot-product-measuring-similarity)
- [Matrices: Transformations](#matrices-transformations)
- [Matrix Multiplication: The Core Operation](#matrix-multiplication-the-core-operation)
- [Transpose, Inverse, and Identity](#transpose-inverse-and-identity)
- [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
- [Rank of a Matrix](#rank-of-a-matrix)
- [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
- [Low-Rank Approximation: Why LoRA Works](#low-rank-approximation-why-lora-works)
- [Norms: Measuring Size](#norms-measuring-size)
- [Practical: Matrix Operations in PyTorch](#practical-matrix-operations-in-pytorch)

---

## Why Linear Algebra?

Every operation in a neural network is a **linear algebra operation** under the hood:

```
Embedding lookup:    row selection from a matrix
Self-attention:      matrix multiplication + softmax
Feed-forward layer:  matrix multiplication + activation
LoRA adapter:        low-rank matrix factorization
Quantization:        scaling and rounding of matrix elements
```

Understanding linear algebra = understanding EXACTLY what your model is doing at every step.

---

## Vectors: The Building Blocks

### What Is a Vector?

A vector is an ordered list of numbers. In ML, vectors represent **everything**:

```
Token embedding:     [0.12, -0.34, 0.56, 0.78, ...]  (2048 dimensions for Gemma)
Gradient vector:     [∂L/∂w₁, ∂L/∂w₂, ∂L/∂w₃, ...]  (one gradient per weight)
Probability vector:  [0.02, 0.05, 0.01, 0.30, ...]    (one per vocab token)
```

### Vector Spaces

```
A "vector space" is the set of all possible vectors of a given dimension.

ℝ² = all 2D vectors: [x, y]
  Example: word embeddings projected to 2D:
    "king"  → [0.7, 0.9]
    "queen" → [0.8, 0.95]
    "man"   → [0.5, 0.4]
    "woman" → [0.6, 0.45]
  
  Notice: king - man + woman ≈ queen
    [0.7, 0.9] - [0.5, 0.4] + [0.6, 0.45] = [0.8, 0.95] ✓
  
  This works because embeddings capture SEMANTIC RELATIONSHIPS
  as geometric relationships (directions in vector space).
```

---

## Dot Product: Measuring Similarity

### Definition

```
For vectors a = [a₁, a₂, ..., aₙ] and b = [b₁, b₂, ..., bₙ]:

  a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σᵢ aᵢbᵢ

Geometric interpretation:
  a · b = |a| × |b| × cos(θ)

Where θ is the angle between vectors:
  θ = 0°   → cos(θ) = 1  → maximum dot product (same direction)
  θ = 90°  → cos(θ) = 0  → zero dot product (orthogonal/unrelated)
  θ = 180° → cos(θ) = -1 → negative dot product (opposite direction)
```

### Why Dot Product Matters for Attention

```
In self-attention, we compute dot products between queries and keys:

  attention_score = Q · K

  High score → tokens are RELEVANT to each other
  Low score  → tokens are UNRELATED
  
Example:
  Q("sat") = [0.5, 0.3, 0.8]   (query: "what did something sit?")
  K("cat") = [0.4, 0.2, 0.9]   (key: "I am a cat")
  K("the") = [0.1, 0.1, 0.0]   (key: "I am a determiner")
  
  Q·K("cat") = 0.5×0.4 + 0.3×0.2 + 0.8×0.9 = 0.20 + 0.06 + 0.72 = 0.98 HIGH
  Q·K("the") = 0.5×0.1 + 0.3×0.1 + 0.8×0.0 = 0.05 + 0.03 + 0.00 = 0.08 LOW
  
  "sat" pays much more attention to "cat" than to "the" — correct!
```

### Cosine Similarity

```
cos_sim(a, b) = (a · b) / (|a| × |b|)

Normalized dot product — ranges from -1 to 1:
  1.0 = identical direction
  0.0 = unrelated
  -1.0 = opposite

Used for finding similar embeddings:
  cos_sim("king", "queen") ≈ 0.8   (similar)
  cos_sim("king", "apple") ≈ 0.1   (unrelated)
```

---

## Matrices: Transformations

### What Is a Matrix?

A matrix is a 2D grid of numbers. In neural networks, matrices represent **linear transformations** — they transform one vector into another.

```
W = [[w₁₁, w₁₂, w₁₃],     ← row 1 (output dimension 1)
     [w₂₁, w₂₂, w₂₃],     ← row 2 (output dimension 2)
     [w₃₁, w₃₂, w₃₃],     ← row 3 (output dimension 3)
     [w₄₁, w₄₂, w₄₃]]     ← row 4 (output dimension 4)
                             
Shape: (4, 3) = 4 rows × 3 columns
Reads as: "transforms a 3D vector into a 4D vector"
```

### Matrices as Transformations

```
Every matrix multiplication is a LINEAR TRANSFORMATION:
  rotation, scaling, projection, or combination thereof.

Example: 2D rotation by angle θ:
  R = [[cos(θ), -sin(θ)],
       [sin(θ),  cos(θ)]]

  R × [1, 0] = [cos(θ), sin(θ)]  ← rotates the x-axis

In neural networks:
  W × x = y
  "Transform input x into output y using learned transformation W"
  
  Each row of W defines one output feature.
  W[i] · x = y[i]  (output feature i = dot product of row i with input)
```

---

## Matrix Multiplication: The Core Operation

### Algorithm

```
C = A × B    where A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ → C ∈ ℝᵐˣᵖ

Rule: Inner dimensions must match! (n = n)

C[i][j] = Σₖ A[i][k] × B[k][j]   (dot product of row i of A with column j of B)
```

### Worked Example

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

C[0][0] = 1×5 + 2×7 = 5 + 14 = 19
C[0][1] = 1×6 + 2×8 = 6 + 16 = 22
C[1][0] = 3×5 + 4×7 = 15 + 28 = 43
C[1][1] = 3×6 + 4×8 = 18 + 32 = 50

C = [[19, 22],
     [43, 50]]
```

### Computational Cost

```
For A ∈ ℝᵐˣⁿ × B ∈ ℝⁿˣᵖ:

  Multiplications: m × n × p
  Additions: m × (n-1) × p

For a Gemma attention layer (q_proj):
  A = input = (batch_size × seq_len × 2048)
  B = W_q = (2048 × 2048)
  
  Multiplications per token: 2048 × 2048 = 4.2 million
  For 512 tokens: 2.1 billion multiplications — just for ONE projection!
  
  This is why GPUs (with Tensor Cores) are essential.
```

---

## Transpose, Inverse, and Identity

### Transpose (Aᵀ)

```
A = [[1, 2, 3],       Aᵀ = [[1, 4],
     [4, 5, 6]]              [2, 5],
                              [3, 6]]
                              
Shape: (2,3) → (3,2)
Rule: Aᵀ[i][j] = A[j][i]   (flip rows and columns)

In attention: Kᵀ is used to compute Q × Kᵀ (attention scores)
```

### Identity Matrix (I)

```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

A × I = A  (identity changes nothing)

Analogous to multiplying by 1 in regular arithmetic.

In LoRA: at initialization, LoRA output = 0
  So: W_effective = W + 0 = W × I (effectively)
```

---

## Eigenvalues and Eigenvectors

### Definition

```
An eigenvector v of matrix A satisfies:
  A × v = λ × v

Where λ (lambda) is the eigenvalue.

In plain English:
  "When we transform v by A, the result points in the SAME DIRECTION
   as v, just scaled by factor λ."
```

### Practical Example

```
A = [[2, 1],
     [0, 3]]

Eigenvector v₁ = [1, 0]: A × [1, 0] = [2, 0] = 2 × [1, 0]  → λ₁ = 2
Eigenvector v₂ = [1, 1]: A × [1, 1] = [3, 3] = 3 × [1, 1]  → λ₂ = 3

These vectors define the "natural directions" of transformation A.
```

### Why Eigenvalues Matter for LoRA

```
When fine-tuning, we're creating a weight update ΔW.
ΔW has eigenvalues that tell us HOW MUCH the model changes
in each "direction" of weight space.

Key insight from the LoRA paper:
  The eigenvalue spectrum of ΔW is STEEP — most eigenvalues are near 0!
  
  Eigenvalues of ΔW: [32.4, 8.7, 2.1, 0.8, 0.3, 0.1, 0.05, ...]
  
  The top few eigenvalues capture most of the change.
  The rest are negligible.
  
  This means ΔW has LOW RANK — it can be approximated by a much
  smaller matrix product A × B (where rank r = number of significant
  eigenvalues).
```

---

## Rank of a Matrix

### Definition

```
The RANK of a matrix = number of linearly independent rows (or columns).

Rank 1 matrix:
  [[2, 4, 6],     All rows are multiples of [1, 2, 3]
   [1, 2, 3],     Rank = 1 (only ONE independent direction)
   [3, 6, 9]]

Rank 2 matrix:
  [[1, 0, 1],     Row 1 and Row 2 are independent
   [0, 1, 1],     Row 3 = Row 1 + Row 2 (dependent)
   [1, 1, 2]]     Rank = 2

Full rank matrix (rank 3):
  [[1, 0, 0],     
   [0, 1, 0],     All rows are independent
   [0, 0, 1]]     Rank = 3
```

### Low-Rank = Compressible

```
A full-rank d×d matrix has d independent directions.
  Storing it requires: d × d numbers = d² parameters

A rank-r matrix (r << d) has only r independent directions.
  It can be written as: A × B where A ∈ ℝᵈˣʳ, B ∈ ℝʳˣᵈ
  Storing it requires: d × r + r × d = 2dr parameters

Example with d=2048, r=16:
  Full rank: 2048² = 4,194,304 parameters
  Rank 16:   2 × 2048 × 16 = 65,536 parameters
  Compression: 64× fewer parameters!

THIS IS EXACTLY WHAT LoRA EXPLOITS!
```

---

## Singular Value Decomposition (SVD)

### Definition

Any matrix M can be decomposed as:

```
M = U × Σ × Vᵀ

Where:
  M ∈ ℝᵐˣⁿ     — original matrix
  U ∈ ℝᵐˣᵐ     — left singular vectors (orthogonal)
  Σ ∈ ℝᵐˣⁿ     — diagonal matrix of singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
  Vᵀ ∈ ℝⁿˣⁿ    — right singular vectors (orthogonal)
```

### Worked Example

```
M = [[3, 0],
     [0, 2],
     [0, 0]]

SVD:
  U = [[1, 0, 0],     Σ = [[3, 0],     Vᵀ = [[1, 0],
       [0, 1, 0],          [0, 2],           [0, 1]]
       [0, 0, 1]]          [0, 0]]

Singular values: σ₁ = 3, σ₂ = 2

These singular values tell us:
  Direction 1 (σ₁=3): Most important — largest magnitude change
  Direction 2 (σ₂=2): Less important
```

### Low-Rank Approximation via SVD

```
Take only the top r singular values:

M ≈ U[:,:r] × Σ[:r,:r] × Vᵀ[:r,:]

For a weight update matrix ΔW with singular values:
  [54.2, 12.3, 3.1, 0.8, 0.2, 0.05, 0.01, ...]

Taking top r=3:
  ΔW ≈ U[:,:3] × diag(54.2, 12.3, 3.1) × V[:,:3]ᵀ

Captures (54.2² + 12.3² + 3.1²) / (total sum of σ²) ≈ 99.7% of the variance!
Only 3 directions capture virtually ALL the information.

This mathematical fact is the foundation of LoRA:
  Instead of learning ΔW (full d×d matrix),
  learn A (d×r) and B (r×d) directly.
  A×B ≈ best rank-r approximation of what ΔW would have been.
```

---

## Low-Rank Approximation: Why LoRA Works

### The Complete Picture

```
Pre-trained weights:  W₀ ∈ ℝ²⁰⁴⁸ˣ²⁰⁴⁸  (frozen, 4.2M params)
Full fine-tune would learn: ΔW ∈ ℝ²⁰⁴⁸ˣ²⁰⁴⁸  (4.2M params)
LoRA learns: A ∈ ℝ²⁰⁴⁸ˣ¹⁶, B ∈ ℝ¹⁶ˣ²⁰⁴⁸  (65K params)

Updated weight: W = W₀ + A×B
           where A×B is a rank-16 approximation of ΔW

Why this works:
  1. ΔW is empirically low-rank (LoRA paper proves this)
  2. Fine-tuning for a specific task only needs a few "directions" of change
  3. The pre-trained model already knows language — we're just STEERING it
```

### How Many "Directions" Does Fine-Tuning Need?

```
From the LoRA paper (experiments on GPT-3):

Rank | Relative Performance
─────┼─────────────────────
  1  │  94% of full fine-tuning
  4  │  97% of full fine-tuning
  8  │  99% of full fine-tuning
 16  │  99.5% of full fine-tuning  ← our choice
 64  │  99.9% of full fine-tuning

Even rank=1 captures most of the fine-tuning signal!
Rank=16 is virtually identical to full fine-tuning.
```

---

## Norms: Measuring Size

### L1 Norm (Manhattan Distance)

```
||x||₁ = |x₁| + |x₂| + ... + |xₙ|

Example: x = [3, -4, 1]
||x||₁ = 3 + 4 + 1 = 8

Used for: Sparse regularization (L1 regularization/Lasso)
```

### L2 Norm (Euclidean Distance) — Most Common

```
||x||₂ = √(x₁² + x₂² + ... + xₙ²)

Example: x = [3, -4, 1]
||x||₂ = √(9 + 16 + 1) = √26 = 5.10

Used for:
  - Gradient clipping (max_grad_norm = 1.0)
  - Weight decay (L2 regularization)
  - RMSNorm in Gemma
```

### Frobenius Norm (for Matrices)

```
||A||_F = √(Σᵢⱼ Aᵢⱼ²)

The L2 norm generalized to matrices.

Used for: Measuring total magnitude of weight changes
  ||ΔW||_F tells us "how much did the weights change overall?"
```

---

## Practical: Matrix Operations in PyTorch

```python
import torch

# ===== VECTORS =====
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Dot product (similarity in attention)
dot = torch.dot(a, b)  # 1×4 + 2×5 + 3×6 = 32
print(f"Dot product: {dot}")  # 32.0

# Cosine similarity
cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
print(f"Cosine similarity: {cos_sim}")  # 0.974

# L2 norm (used in gradient clipping)
norm = torch.norm(a)  # √(1+4+9) = √14 = 3.74
print(f"L2 norm: {norm}")  # 3.742

# ===== MATRICES =====
W = torch.randn(2048, 2048)  # Simulating a Gemma linear layer

# Matrix multiplication (the core operation)
x = torch.randn(1, 2048)     # One token embedding
y = x @ W                     # Linear transformation
print(f"Output shape: {y.shape}")  # [1, 2048]

# ===== LoRA SIMULATION =====
d, r = 2048, 16
A = torch.randn(d, r)  # Down-projection
B = torch.zeros(r, d)  # Up-projection (init to zero!)

# LoRA output (initially zero!)
lora_output = x @ A @ B
print(f"LoRA output norm (init): {lora_output.norm():.4f}")  # ~0.0

# After training, B will have learned values:
B_trained = torch.randn(r, d) * 0.01
lora_output = x @ A @ B_trained
print(f"LoRA output norm (trained): {lora_output.norm():.4f}")

# The rank of A @ B
AB = A @ B_trained  # Shape: (2048, 2048) but rank 16!
print(f"Full matrix has {d*d} = {d*d:,} elements")
print(f"LoRA stores {d*r + r*d} = {d*r + r*d:,} elements")
print(f"Compression ratio: {(d*d)/(d*r + r*d):.0f}x")

# ===== SVD (to verify low-rank) =====
U, S, Vh = torch.linalg.svd(AB)
print(f"Singular values: {S[:20]}")  # Only first 16 are non-zero!
print(f"Number of non-zero singular values: {(S > 1e-6).sum()}")  # 16
```
