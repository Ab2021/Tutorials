# 1. Neural Networks from Scratch — First Principles to Deep Learning

## Table of Contents
- [What Is a Neural Network?](#what-is-a-neural-network)
- [The Single Neuron (Perceptron)](#the-single-neuron-perceptron)
- [Activation Functions Deep Dive](#activation-functions-deep-dive)
- [Multi-Layer Perceptron](#multi-layer-perceptron)
- [Forward Pass: Complete Worked Example](#forward-pass-complete-worked-example)
- [Loss Functions from First Principles](#loss-functions-from-first-principles)
- [Backpropagation: Full Calculus Derivation](#backpropagation-full-calculus-derivation)
- [Gradient Descent: Walking Downhill](#gradient-descent-walking-downhill)
- [Universal Approximation Theorem](#universal-approximation-theorem)
- [From MLP to Transformer: The Bridge](#from-mlp-to-transformer-the-bridge)

---

## What Is a Neural Network?

A neural network is a mathematical function that maps inputs to outputs through a series of **learned transformations**. Each transformation is parameterized by **weights** (numbers) that are adjusted during training to make the function produce desired outputs.

### The Biological Inspiration (and Why It's Misleading)

```
Biological Neuron:                  Artificial Neuron:
                                    
  dendrites →                       inputs (x₁, x₂, ...) →
  soma (cell body) →                weighted sum + bias →
  axon →                            activation function →
  synapses →                        output

The analogy is LOOSE. Modern neural networks are really just
matrix multiplications and nonlinear functions. They're
mathematical objects, not brain simulations.
```

---

## The Single Neuron (Perceptron)

### Mathematical Definition

```
Given:
  Inputs:  x = [x₁, x₂, ..., xₙ]     (n-dimensional input vector)
  Weights: w = [w₁, w₂, ..., wₙ]     (n learnable parameters)
  Bias:    b                            (1 learnable parameter)
  
The neuron computes:
  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b   (weighted sum)
  z = wᵀx + b                          (dot product notation)
  y = σ(z)                              (apply activation function σ)
```

### Practical Example: Product Rating Predictor

```
Inputs (features of a product review):
  x₁ = 0.8   (sentiment score, 0=negative, 1=positive)
  x₂ = 0.3   (review length, normalized)
  x₃ = 1.0   (verified purchase? yes=1, no=0)

Weights (learned during training):
  w₁ = 2.5   (sentiment matters a lot)
  w₂ = 0.1   (review length matters less)
  w₃ = 0.5   (verified purchase matters somewhat)
  b  = -1.0  (bias/offset)

Weighted sum:
  z = 2.5(0.8) + 0.1(0.3) + 0.5(1.0) + (-1.0)
  z = 2.0 + 0.03 + 0.5 - 1.0
  z = 1.53

Apply sigmoid activation:
  y = σ(1.53) = 1/(1 + e^(-1.53)) = 0.822

Output: 0.822 → Model predicts ~82% chance of positive review
```

### Why Bias Matters

```
Without bias (b=0):
  The decision boundary MUST pass through the origin.
  y = σ(2.5x₁ + 0.1x₂ + 0.5x₃)
  When all inputs are 0, output is always 0.5 (neutral)

With bias:
  The decision boundary can shift anywhere.
  Bias = "what's the default prediction when all inputs are zero?"
```

---

## Activation Functions Deep Dive

Activation functions introduce **nonlinearity**. Without them, stacking layers would just be one big linear transformation (useless for complex tasks).

### Sigmoid

```
σ(z) = 1 / (1 + e^(-z))

Output range: (0, 1)
Derivative: σ'(z) = σ(z) × (1 - σ(z))

Graph:
  1.0 ─────────────────────────────────────  ____━━━━━━━
  0.5 ─────────────────────────────  ___╱───
  0.0 ━━━━━━━━____─────────────────
      -6   -4   -2    0    2    4    6

Practical example:
  z = -3.0 → σ(-3) = 0.047  (strong negative → near 0)
  z =  0.0 → σ(0)  = 0.500  (neutral → exactly 0.5)
  z =  3.0 → σ(3)  = 0.953  (strong positive → near 1)

Problems:
  ❌ Vanishing gradients: for |z| > 4, gradient ≈ 0
     σ'(-5) = 0.0067 → gradient almost disappears!
     Multiplied across many layers → gradient = ~0 → no learning
  ❌ Not zero-centered (outputs always positive)
  ❌ Expensive (exponential computation)

When to use: Output layer for binary classification ONLY
```

### Tanh

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

Output range: (-1, 1)
Derivative: tanh'(z) = 1 - tanh²(z)

Practical example:
  z = -2.0 → tanh(-2) = -0.964
  z =  0.0 → tanh(0)  =  0.000
  z =  2.0 → tanh(2)  =  0.964

Advantages over sigmoid:
  ✅ Zero-centered (outputs can be negative)
  ✅ Stronger gradients (derivative max = 1.0 vs 0.25 for sigmoid)

Problems:
  ❌ Still has vanishing gradient for |z| > 3

When to use: Sometimes in RNNs. Rarely in transformers.
```

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)

Output range: [0, ∞)
Derivative: ReLU'(z) = 1 if z > 0, else 0

Graph:
  ▲
  │        ╱
  │       ╱
  │      ╱
  │     ╱
  │____╱________________▶
      0

Practical example:
  z = -5.0 → ReLU(-5) = 0     (negative → zero)
  z =  0.0 → ReLU(0)  = 0
  z =  3.7 → ReLU(3.7) = 3.7  (positive → pass through)

Advantages:
  ✅ No vanishing gradient for positive values
  ✅ Extremely fast (just max(0, z))
  ✅ Sparse activation (many zeros → efficient)

Problems:
  ❌ "Dying ReLU": if z < 0 always, gradient = 0, neuron never recovers
  ❌ Not zero-centered

When to use: Hidden layers in most networks. Standard choice.
```

### GELU (Gaussian Error Linear Unit) — Used by Gemma

```
GELU(z) = z × Φ(z)
where Φ(z) is the cumulative distribution function of standard normal

Approximation: GELU(z) ≈ 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))

Graph:
  ▲
  │         ╱
  │        ╱
  │      _╱
  │   __╱
  │__╱__________________▶
     0

Practical example:
  z = -2.0 → GELU(-2) = -0.045  (slightly negative — not hard zero!)
  z =  0.0 → GELU(0)  =  0.000
  z =  2.0 → GELU(2)  =  1.955  (slightly less than 2)

Key difference from ReLU:
  ReLU(-1) = 0      (hard cutoff)
  GELU(-1) = -0.159 (soft, allows small negative values)

Advantages:
  ✅ Smooth (differentiable everywhere, unlike ReLU's corner at 0)
  ✅ No dying neuron problem (small gradients even for negative z)
  ✅ Empirically better for transformers

GEMMA USES: GeGLU = Gated GELU (GELU combined with a gating mechanism)
  GeGLU(x) = GELU(xW₁) ⊙ (xW₂)
  The gate learns WHICH features to activate
```

### SiLU / Swish — Also Common in Modern LLMs

```
SiLU(z) = z × σ(z) = z / (1 + e^(-z))

Very similar to GELU but analytically simpler.
Used in LLaMA, Mistral.

Practical:
  z = -2.0 → SiLU(-2) = -0.238
  z =  0.0 → SiLU(0)  =  0.000
  z =  2.0 → SiLU(2)  =  1.762
```

### Activation Function Summary

| Function | Range | Gradient Issue | Speed | Used In |
|----------|-------|---------------|-------|---------|
| Sigmoid | (0,1) | Vanishing | Slow | Output only |
| Tanh | (-1,1) | Vanishing | Medium | RNNs |
| ReLU | [0,∞) | Dying neurons | Fast | CNNs, MLPs |
| **GELU** | **(-∞,∞)** | **None** | **Medium** | **Transformers (Gemma)** |
| SiLU | (-∞,∞) | None | Medium | LLaMA, Mistral |

---

## Multi-Layer Perceptron

### Architecture

```
Input Layer      Hidden Layer 1     Hidden Layer 2     Output
(3 neurons)      (4 neurons)        (4 neurons)        (1 neuron)

  x₁ ─────┬──→ h₁⁽¹⁾ ────┬──→ h₁⁽²⁾ ────┬──→ ŷ
           │              │              │
  x₂ ─────┼──→ h₂⁽¹⁾ ────┼──→ h₂⁽²⁾ ────┤
           │              │              │
  x₃ ─────┼──→ h₃⁽¹⁾ ────┼──→ h₃⁽²⁾ ────┤
           │              │              │
           └──→ h₄⁽¹⁾ ────┘──→ h₄⁽²⁾ ────┘

Each arrow represents a WEIGHT (learned parameter).
Total connections: (3×4) + (4×4) + (4×1) = 12 + 16 + 4 = 32 weights
Plus biases: 4 + 4 + 1 = 9
Total parameters: 41
```

### Matrix Form

```
Layer 1:
  z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾        W⁽¹⁾ ∈ ℝ⁴ˣ³, b⁽¹⁾ ∈ ℝ⁴
  h⁽¹⁾ = GELU(z⁽¹⁾)

Layer 2:
  z⁽²⁾ = W⁽²⁾h⁽¹⁾ + b⁽²⁾     W⁽²⁾ ∈ ℝ⁴ˣ⁴, b⁽²⁾ ∈ ℝ⁴
  h⁽²⁾ = GELU(z⁽²⁾)

Output:
  z⁽³⁾ = W⁽³⁾h⁽²⁾ + b⁽³⁾     W⁽³⁾ ∈ ℝ¹ˣ⁴, b⁽³⁾ ∈ ℝ¹
  ŷ = σ(z⁽³⁾)                  sigmoid for binary output
```

---

## Forward Pass: Complete Worked Example

Let's compute a full forward pass with actual numbers.

```
NETWORK: 2 inputs → 3 hidden (ReLU) → 1 output (sigmoid)

Inputs: x = [1.0, 0.5]

Weights and biases (pretend these are learned):
  W⁽¹⁾ = [[0.3, -0.2],    b⁽¹⁾ = [0.1, -0.1, 0.0]
           [0.5,  0.4],
           [-0.1, 0.6]]
           
  W⁽²⁾ = [[0.2, -0.3, 0.7]]  b⁽²⁾ = [-0.2]

STEP 1: Hidden layer
  z₁⁽¹⁾ = 0.3(1.0) + (-0.2)(0.5) + 0.1 = 0.3 - 0.1 + 0.1 = 0.30
  z₂⁽¹⁾ = 0.5(1.0) + 0.4(0.5) + (-0.1) = 0.5 + 0.2 - 0.1 = 0.60
  z₃⁽¹⁾ = (-0.1)(1.0) + 0.6(0.5) + 0.0 = -0.1 + 0.3 + 0.0 = 0.20

  h⁽¹⁾ = ReLU([0.30, 0.60, 0.20]) = [0.30, 0.60, 0.20]
  (All positive, so ReLU doesn't change anything)

STEP 2: Output layer
  z⁽²⁾ = 0.2(0.30) + (-0.3)(0.60) + 0.7(0.20) + (-0.2)
        = 0.06 - 0.18 + 0.14 - 0.2
        = -0.18

  ŷ = σ(-0.18) = 1/(1 + e^0.18) = 1/1.197 = 0.455

OUTPUT: 0.455 (model predicts ~45.5% probability)
```

---

## Loss Functions from First Principles

### Binary Cross-Entropy (BCE)

```
For a single example with true label y ∈ {0, 1} and prediction ŷ ∈ (0, 1):

  L = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

When y = 1 (positive example):
  L = -log(ŷ)
  If ŷ = 0.99 → L = -log(0.99) = 0.01  ← confident and correct, low loss
  If ŷ = 0.50 → L = -log(0.50) = 0.69  ← uncertain, medium loss
  If ŷ = 0.01 → L = -log(0.01) = 4.61  ← confident and WRONG, high loss!

When y = 0 (negative example):
  L = -log(1-ŷ)
  If ŷ = 0.01 → L = -log(0.99) = 0.01  ← correct
  If ŷ = 0.99 → L = -log(0.01) = 4.61  ← wrong!
```

### Cross-Entropy for Language Modeling (Our Case)

```
At each position t, the model outputs a probability distribution
over ALL V tokens in the vocabulary.

For vocabulary of size V = 256,128:
  Model outputs: p = [p₁, p₂, ..., p₂₅₆₁₂₈]  where Σpᵢ = 1

The true next token is token index k.
One-hot encoding: y = [0, 0, ..., 1, ..., 0]  (1 at position k)

Cross-entropy loss:
  L = -Σᵢ yᵢ log(pᵢ) = -log(pₖ)

Just the negative log of the probability assigned to the CORRECT token!

Over the full sequence of T tokens:
  L = -(1/T) Σᵗ log(p(xₜ | x₁, ..., xₜ₋₁))

"The average negative log-probability of each true token."
```

---

## Backpropagation: Full Calculus Derivation

### The Chain Rule

The chain rule from calculus is THE fundamental tool of backpropagation:

```
If y = f(g(x)), then:  dy/dx = (dy/dg) × (dg/dx)

Or: "The rate of change of y with respect to x equals
     the rate of change of y with respect to g
     TIMES the rate of change of g with respect to x."
```

### Full Backprop for Our 2-3-1 Network

```
Forward pass (from earlier):
  z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾       →  [0.30, 0.60, 0.20]
  h⁽¹⁾ = ReLU(z⁽¹⁾)          →  [0.30, 0.60, 0.20]
  z⁽²⁾ = W⁽²⁾h⁽¹⁾ + b⁽²⁾    →  -0.18
  ŷ = σ(z⁽²⁾)                 →  0.455

Suppose true label y = 1:
  L = -log(ŷ) = -log(0.455) = 0.787

BACKWARD PASS (compute all gradients):

Step 1: ∂L/∂ŷ
  L = -log(ŷ)
  ∂L/∂ŷ = -1/ŷ = -1/0.455 = -2.198

Step 2: ∂L/∂z⁽²⁾ (through sigmoid)
  ŷ = σ(z⁽²⁾), so ∂ŷ/∂z⁽²⁾ = σ(z⁽²⁾)(1 - σ(z⁽²⁾)) = 0.455 × 0.545 = 0.248
  ∂L/∂z⁽²⁾ = ∂L/∂ŷ × ∂ŷ/∂z⁽²⁾ = -2.198 × 0.248 = -0.545

  Shortcut: For cross-entropy + sigmoid, ∂L/∂z⁽²⁾ = ŷ - y = 0.455 - 1 = -0.545 ✓

Step 3: ∂L/∂W⁽²⁾ (output weights)
  z⁽²⁾ = W⁽²⁾h⁽¹⁾ + b⁽²⁾
  ∂L/∂W⁽²⁾ = ∂L/∂z⁽²⁾ × h⁽¹⁾ᵀ = -0.545 × [0.30, 0.60, 0.20]
           = [-0.164, -0.327, -0.109]
  
  ∂L/∂b⁽²⁾ = ∂L/∂z⁽²⁾ = -0.545

Step 4: ∂L/∂h⁽¹⁾ (propagate backwards)
  ∂L/∂h⁽¹⁾ = W⁽²⁾ᵀ × ∂L/∂z⁽²⁾ = [0.2, -0.3, 0.7]ᵀ × (-0.545)
           = [-0.109, 0.164, -0.382]

Step 5: ∂L/∂z⁽¹⁾ (through ReLU)
  ReLU'(z) = 1 if z > 0, else 0
  All z⁽¹⁾ values are > 0, so:
  ∂L/∂z⁽¹⁾ = ∂L/∂h⁽¹⁾ ⊙ ReLU'(z⁽¹⁾) = [-0.109, 0.164, -0.382] ⊙ [1, 1, 1]
           = [-0.109, 0.164, -0.382]

Step 6: ∂L/∂W⁽¹⁾ (hidden weights)
  ∂L/∂W⁽¹⁾ = ∂L/∂z⁽¹⁾ × xᵀ
  
  = [-0.109]         [1.0, 0.5]ᵀ     [[-0.109, -0.055],
    [ 0.164]    ×                  =   [ 0.164,  0.082],
    [-0.382]                           [-0.382, -0.191]]

WEIGHT UPDATE (SGD with lr=0.1):
  W⁽²⁾_new = W⁽²⁾ - 0.1 × ∂L/∂W⁽²⁾
           = [0.2, -0.3, 0.7] - 0.1 × [-0.164, -0.327, -0.109]
           = [0.216, -0.267, 0.711]

The weights shifted to make ŷ closer to y=1!
```

---

## Gradient Descent: Walking Downhill

### The Loss Landscape

```
Imagine the loss as a hilly terrain. Each point represents
a specific set of weight values, and the height is the loss:

  Loss
   ▲
   │    ╱╲
   │   ╱  ╲      ╱╲
   │  ╱    ╲    ╱  ╲
   │ ╱      ╲__╱    ╲___        ← local minimum
   │╱                    ╲___   ← global minimum
   └──────────────────────────▶ Weights

Gradient descent: Start at a random point, compute the slope
(gradient), and step in the DOWNHILL direction.
```

### Variants

```
BATCH Gradient Descent:
  Compute gradient using ALL training examples.
  Update once per epoch.
  ✅ Stable, exact gradient
  ❌ Very slow for large datasets
  ❌ Needs entire dataset in memory

STOCHASTIC Gradient Descent (SGD):
  Compute gradient using ONE random example.
  Update after every example.
  ✅ Very fast updates
  ❌ Very noisy gradient (zigzag path)
  
MINI-BATCH Gradient Descent (our approach):
  Compute gradient using a BATCH of examples (e.g., 4-16).
  Update after each batch.
  ✅ Good balance: stable enough, fast enough
  ✅ GPU parallelism (process batch simultaneously)
  This is what everyone uses in practice.
```

---

## Universal Approximation Theorem

### The Theorem

> A neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision, given sufficient neurons and a suitable activation function.

### What This Means

```
In plain English:
  A neural network CAN, in theory, learn ANY function.
  Want to map product reviews → recommendations? A neural network can do it.
  
But "can" ≠ "easily will":
  - May need MANY neurons (exponentially many for some functions)
  - Finding the right weights via gradient descent is not guaranteed
  - Depth (multiple layers) is much more efficient than width for complex functions
  
This is why we use DEEP networks (transformers with 18 layers)
instead of wide, shallow ones.
```

---

## From MLP to Transformer: The Bridge

### Why MLPs Aren't Enough for Language

```
MLP problem: Fixed-size input.
  "The cat sat on the mat" → must be exactly N numbers
  "Hello" → must ALSO be exactly N numbers
  
  How do you handle variable-length text?

MLP problem: No notion of ORDER.
  MLP([cat, sat, mat]) = MLP([mat, cat, sat])
  But "The cat sat on the mat" ≠ "The mat sat on the cat"!

MLP problem: No concept of CONTEXT.
  Each input is processed independently.
  But the meaning of "bank" depends on context:
    "river bank" vs "bank account"
```

### Transformers Solve All Three

```
1. VARIABLE LENGTH → Sequence processing
   Process any number of tokens, one per position.

2. ORDER → Positional encoding
   Add position information: token = content + position.

3. CONTEXT → Self-attention
   Each token can "look at" all other tokens to understand context.
   "bank" attends to "river" → learns it means riverbank.
```

This is why the architecture progressed:
```
MLP (1960s) → RNN (1990s) → LSTM (1997) → Transformer (2017) → GPT/Gemma (2023)

Each step added: variable length → memory → long memory → parallel attention
```
