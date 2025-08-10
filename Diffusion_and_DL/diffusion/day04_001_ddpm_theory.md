# Day 4 - Part 1: Denoising Diffusion Probabilistic Models (DDPM) Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of DDPM and its probabilistic formulation
- Theoretical analysis of forward and reverse diffusion processes
- Mathematical principles of variational lower bound (VLB) optimization
- Information-theoretic perspectives on noise scheduling and training objectives
- Theoretical frameworks for reparameterization tricks and loss simplification
- Mathematical modeling of DDPM vs score-based equivalence

---

## 🎯 DDPM Mathematical Framework

### Probabilistic Formulation

#### Forward Diffusion Process
**Mathematical Definition**:
```
Markov Chain Formulation:
q(x₁:T | x₀) = ∏ᵗ₌₁ᵀ q(xₜ | xₜ₋₁)

Forward Transition:
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
where βₜ ∈ (0,1) is noise schedule

Reparameterization:
xₜ = √(1-βₜ)xₜ₋₁ + √βₜ εₜ
where εₜ ~ N(0,I)

Cumulative Product:
αₜ = 1 - βₜ
ᾱₜ = ∏ₛ₌₁ᵗ αₛ

Direct Sampling:
q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)
Closed-form sampling at any timestep
```

**Information-Theoretic Analysis**:
```
Mutual Information Decay:
I(x₀; xₜ) = H(x₀) - H(x₀|xₜ)
Decreases monotonically with t
Rate controlled by noise schedule βₜ

Signal-to-Noise Ratio:
SNR(t) = ᾱₜ/(1-ᾱₜ)
Measures information preservation
High SNR: structure preserved
Low SNR: approaching pure noise

Entropy Evolution:
H(xₜ) increases with t
Differential entropy for continuous distributions
Final entropy H(xₜ) ≈ H(N(0,I)) for large T

KL Divergence to Prior:
KL(q(xₜ|x₀) || p(xₜ)) → 0 as T → ∞
Forward process approaches chosen prior
Typically p(xₜ) = N(0,I)
```

#### Reverse Diffusion Process
**Mathematical Formulation**:
```
Reverse Markov Chain:
p_θ(x₀:T) = p(xₜ) ∏ᵗ₌₁ᵀ p_θ(xₜ₋₁ | xₜ)

Reverse Transition:
p_θ(xₜ₋₁ | xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))
Parameterized by neural network

Gaussian Assumption:
Reverse transitions are Gaussian
Motivated by forward process properties
Tractable for small step sizes

Mean Parameterization:
μ_θ(xₜ,t) = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ))ε_θ(xₜ,t))
Predict noise ε_θ instead of mean directly
Equivalent parameterizations available

Variance Schedule:
Σ_θ(xₜ,t) = σₜ²I
Can be learned or fixed
Common choice: σₜ² = βₜ or σₜ² = β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) βₜ
```

### Variational Lower Bound

#### Mathematical Derivation
**Evidence Lower Bound (ELBO)**:
```
Log-Likelihood Bound:
log p_θ(x₀) ≥ E_q[-log q(x₁:T|x₀) + log p_θ(x₀:T)]

ELBO Decomposition:
L = E_q[log p_θ(x₀|x₁) - KL(q(xₜ|x₀) || p(xₜ)) 
    - Σᵗ₌₂ᵀ KL(q(xₜ₋₁|xₜ,x₀) || p_θ(xₜ₋₁|xₜ))]

Term Interpretation:
L₀: Reconstruction term
Lₜ: Transition matching terms (t = 1,...,T-1)
Lₜ: Prior matching term

Negative ELBO:
L = L₀ + L₁ + ... + Lₜ₋₁ + Lₜ
Minimize negative ELBO for training
```

**Simplified Loss Derivation**:
```
Posterior Distribution:
q(xₜ₋₁|xₜ,x₀) = N(xₜ₋₁; μ̃ₜ(xₜ,x₀), β̃ₜI)
where μ̃ₜ(xₜ,x₀) = (√ᾱₜ₋₁βₜx₀ + √αₜ(1-ᾱₜ₋₁)xₜ)/(1-ᾱₜ)

KL Divergence:
KL(q(xₜ₋₁|xₜ,x₀) || p_θ(xₜ₋₁|xₜ)) = ½σₜ²||μ̃ₜ - μ_θ||²

Reparameterization:
xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε
where ε ~ N(0,I)

Simplified Objective:
L_simple = E_t,x₀,ε[||ε - ε_θ(xₜ,t)||²]
Predict noise instead of mean
Equivalent to weighted VLB under certain conditions
```

#### Theoretical Analysis
**Optimality and Consistency**:
```
VLB Tightness:
Gap between log p_θ(x₀) and ELBO
Depends on approximation quality of q(x₁:T|x₀)
Forward process design affects tightness

Asymptotic Behavior:
As T → ∞ and βₜ → 0:
Forward process approaches SDE
Reverse process approaches score-based sampling
DDPM converges to score-based models

Optimal Reverse Process:
p*(xₜ₋₁|xₜ) ∝ q(xₜ|xₜ₋₁)p*(xₜ₋₁)
Bayes' theorem gives optimal reverse
Neural network approximates this optimal process

Parameterization Equivalence:
Different parameterizations (ε, μ, x₀) equivalent
Choice affects optimization and numerical stability
Noise prediction often most stable
```

### Noise Schedule Analysis

#### Mathematical Principles
**Schedule Design Criteria**:
```
Coverage Requirement:
βₜ chosen to ensure q(xₜ|x₀) ≈ N(0,I) for large T
Sufficient noise to erase structure

Information Preservation:
Balance between noise addition and structure retention
Too fast: information lost too quickly
Too slow: insufficient exploration at end

Mathematical Constraints:
0 < βₜ < 1 for all t
Monotonicity: often βₜ ≤ βₜ₊₁
Boundary conditions: β₁ small, βₜ moderate
```

**Common Schedules**:
```
Linear Schedule:
βₜ = β₁ + (βₜ - β₁)(t-1)/(T-1)
Simple and interpretable
May not be optimal for all data types

Cosine Schedule:
αₜ = cos²((t/T + s)/(1 + s) × π/2)
where s is small offset
Slower noise addition initially
Better preservation of coarse structure

Learned Schedules:
Optimize noise schedule parameters
Data-dependent adaptation
Computational overhead vs performance gain

Theoretical Optimality:
No universally optimal schedule
Depends on data distribution characteristics
Information-theoretic criteria for design
```

#### Signal-to-Noise Analysis
**SNR Evolution**:
```
SNR Definition:
SNR(t) = ᾱₜ/(1-ᾱₜ) = E[||√ᾱₜ x₀||²]/E[||(1-ᾱₜ)ε||²]

Log-SNR Parameterization:
log SNR(t) = log ᾱₜ - log(1-ᾱₜ)
More stable for optimization
Linear in log-space often better

SNR-Based Schedules:
Choose log SNR(t) function directly
Convert to βₜ via relationship
More principled than direct βₜ specification

Critical SNR Values:
High SNR (> 1): structure dominates
Medium SNR (≈ 1): balanced signal/noise
Low SNR (< 1): noise dominates
Transition points affect generation quality
```

### Connection to Score-Based Models

#### Mathematical Equivalence
**Score Function Connection**:
```
DDPM Score Relation:
ε_θ(xₜ,t) = -√(1-ᾱₜ) s_θ(xₜ,t)
where s_θ(xₜ,t) = ∇log p(xₜ)

Reverse Process Mean:
μ_θ(xₜ,t) = (1/√αₜ)(xₜ - (βₜ/√(1-ᾱₜ))ε_θ(xₜ,t))
= (1/√αₜ)(xₜ + βₜs_θ(xₜ,t))

Langevin Connection:
DDPM reverse step ≈ Langevin MCMC step
xₜ₋₁ = xₜ + ½βₜs_θ(xₜ,t) + √βₜ z
Discrete approximation to continuous Langevin

SDE Limit:
As T → ∞, βₜ → 0, βₜT → constant:
DDPM approaches continuous SDE
dx = -½β(t)x dt + √β(t) dW
```

**Training Objective Equivalence**:
```
Score Matching Loss:
J_SM = E_t[λ(t)E_{x₀,ε}[||s_θ(xₜ,t) - ∇log q(xₜ|x₀)||²]]

DDPM Loss:
L_DDPM = E_t,x₀,ε[||ε - ε_θ(xₜ,t)||²]

Equivalence:
∇log q(xₜ|x₀) = -ε/√(1-ᾱₜ)
Therefore: s_θ = -ε_θ/√(1-ᾱₜ)
DDPM loss ≡ score matching with λ(t) = (1-ᾱₜ)

Weight Function:
λ(t) = (1-ᾱₜ) gives more weight to low noise
Balances different timesteps
Can be modified for different emphasis
```

### Advanced Theoretical Analysis

#### Information-Theoretic Perspective
**Rate-Distortion Analysis**:
```
Forward Process as Encoder:
Encoder: x₀ → x₁:T (adds noise)
Rate: I(x₀; x₁:T) (information preserved)
Distortion: E[||x₀ - x̂₀||²] (reconstruction error)

Reverse Process as Decoder:
Decoder: x₁:T → x̂₀ (removes noise)
Optimal decoder minimizes distortion given rate
DDPM approximates optimal decoder

Rate-Distortion Curve:
R(D) = min I(x₀; x₁:T) subject to E[||x₀ - x̂₀||²] ≤ D
DDPM operates on this curve
Different noise schedules give different operating points

Information Bottleneck:
Forward process creates information bottleneck
Reverse process recovers information
Similar to VAE but with hierarchical structure
```

**Generalization Bounds**:
```
Sample Complexity:
Number of samples needed for accurate density modeling
Depends on data distribution complexity
DDPM may have better sample complexity than GANs

Approximation Error:
Neural network approximation quality
Universal approximation theorems apply
Depth and width requirements for accuracy

Generalization Error:
Test vs training performance gap
Depends on model complexity and sample size
DDPM training objective may provide implicit regularization

Convergence Guarantees:
Under appropriate conditions:
Generated distribution approaches data distribution
Rate depends on approximation quality and schedule choice
```

---

## 🎯 Advanced Understanding Questions

### DDPM Mathematical Theory:
1. **Q**: Analyze the mathematical relationship between the DDPM variational lower bound and the simplified training objective, deriving conditions under which they are equivalent.
   **A**: Mathematical relationship: VLB = Σₜ Lₜ where Lₜ are KL divergences between forward posterior and reverse model. Simplified objective: E[||ε - ε_θ||²] ignores weighting and constants. Equivalence conditions: (1) optimal variance schedule σₜ² = β̃ₜ, (2) specific weighting λ(t) = (1-ᾱₜ), (3) small step size approximation. Analysis: simplified objective corresponds to weighted VLB with emphasis on low-noise timesteps. Deviation from VLB: ignores prior matching term L_T, uses uniform weighting across timesteps. Practical implication: simplified objective often works better despite theoretical suboptimality, suggesting VLB may not be tight bound for finite T.

2. **Q**: Develop a theoretical framework for analyzing the impact of different noise schedules on DDPM training dynamics and generation quality.
   **A**: Framework components: (1) signal-to-noise ratio evolution SNR(t) = ᾱₜ/(1-ᾱₜ), (2) information preservation I(x₀; xₜ), (3) score estimation difficulty. Mathematical analysis: linear schedule gives uniform information removal, cosine schedule preserves structure longer. Training dynamics: early timesteps (high noise) easier to learn, late timesteps (low noise) require fine details. Generation quality: smooth SNR transitions prevent artifacts, sufficient noise ensures mode coverage. Optimal schedules: balance information preservation with computational efficiency. Theoretical insight: schedule should match data complexity - complex structures need slower noise addition, simple patterns can handle faster schedules.

3. **Q**: Compare the mathematical foundations of DDPM and score-based diffusion models, analyzing their theoretical equivalence and practical differences.
   **A**: Mathematical equivalence: DDPM predicts noise ε_θ, score models predict score s_θ = -ε_θ/√(1-ᾱₜ). Both minimize same objective up to weighting function λ(t). Training objectives equivalent when λ(t) = (1-ᾱₜ). Practical differences: DDPM uses fixed timesteps, score models often continuous time. DDPM emphasizes low-noise regions, score models can weight differently. Implementation: DDPM more structured (discrete steps), score models more flexible (adaptive sampling). Convergence: both converge to same distribution under appropriate conditions. Theoretical insight: choice between formulations affects optimization dynamics and implementation convenience but not fundamental capabilities.

### Noise Schedule Theory:
4. **Q**: Analyze the mathematical principles behind optimal noise schedule design, deriving theoretical bounds on information preservation and generation quality.
   **A**: Optimal schedule principles: balance information preservation with exploration capability. Mathematical framework: I(x₀; xₜ) = H(x₀) - H(ε)/2 log(2πe(1-ᾱₜ)) for Gaussian case. Information bound: I(x₀; xₜ) ≥ 0 with equality when xₜ ~ N(0,I). Generation quality: depends on score estimation accuracy across noise levels. Theoretical bounds: reconstruction error ≤ Σₜ approximation_error(t) × information_weight(t). Optimal schedules: minimize total error subject to computational constraints. Design principles: slower noise addition for complex structures, ensure sufficient exploration, smooth transitions between scales. Key insight: no universally optimal schedule, must adapt to data characteristics and computational budget.

5. **Q**: Develop a mathematical theory for the approximation quality of the Gaussian assumption in DDPM reverse transitions, analyzing when this assumption holds and fails.
   **A**: Gaussian assumption: p_θ(xₜ₋₁|xₜ) = N(μ_θ, σₜ²I). Validity conditions: (1) small step size βₜ, (2) smooth data distribution, (3) sufficient diffusion time. Mathematical analysis: central limit theorem suggests Gaussianity for small steps, but mode-splitting can occur. Failure modes: multimodal posteriors when step size too large, non-Gaussian tails for heavy-tailed data, discrete data structures. Approximation quality: KL divergence between true posterior and Gaussian approximation. Theoretical bounds: error O(βₜ²) for smooth distributions under small step assumption. Practical implications: smaller steps improve accuracy but increase computational cost, alternative distributions (mixtures, flows) can improve approximation for complex cases.

6. **Q**: Compare the information-theoretic properties of different DDPM parameterizations (noise prediction, mean prediction, data prediction) and their impact on training dynamics.
   **A**: Information-theoretic comparison: all parameterizations equivalent in terms of mutual information but differ in optimization landscape. Noise prediction: ε_θ directly predicts corruption, simple gradient structure. Mean prediction: μ_θ predicts reverse mean, requires careful scaling. Data prediction: x̂₀_θ predicts clean data, interpretable but potentially unstable. Training dynamics: noise prediction often most stable due to consistent scale across timesteps. Gradient analysis: different parameterizations have different sensitivity to approximation errors. Numerical stability: noise prediction less sensitive to extreme values, data prediction can amplify errors. Theoretical insight: parameterization choice affects optimization efficiency and numerical stability but not fundamental model capacity or asymptotic performance.

### Advanced Applications:
7. **Q**: Design a mathematical framework for analyzing the sample complexity and generalization bounds of DDPM across different data distributions and model architectures.
   **A**: Framework components: (1) data distribution complexity (smoothness, support size), (2) neural network approximation capacity, (3) finite sample effects. Sample complexity: depends on effective dimension of data distribution and score function complexity class. Mathematical bounds: O(d_eff/ε²) samples for ε-accurate generation under smoothness assumptions. Architecture impact: deeper networks better for complex distributions but require more samples. Generalization analysis: uniform convergence over score function class, depends on Rademacher complexity. Data-dependent bounds: smoother distributions require fewer samples, multimodal distributions increase complexity. Theoretical guarantee: under appropriate regularity conditions, DDPM achieves minimax optimal sample complexity for density estimation. Key insight: sample complexity scales with intrinsic rather than ambient dimension when data has low-dimensional structure.

8. **Q**: Develop a unified mathematical theory connecting DDPM to other generative models (VAEs, GANs, flows), identifying fundamental relationships and trade-offs.
   **A**: Unified framework: all generative models minimize divergences between data and model distributions but use different metrics and optimization strategies. DDPM minimizes weighted score matching objective (Fisher divergence variant). VAEs minimize reverse KL divergence via ELBO. GANs minimize JS divergence via adversarial training. Flows minimize forward KL via change of variables. Mathematical connections: all relate to optimal transport in different metrics, DDPM can be viewed as hierarchical VAE with specific encoder structure. Trade-offs: DDPM (stable training, slow sampling), VAEs (fast sampling, blurry images), GANs (sharp images, unstable training), flows (exact likelihood, architectural constraints). Fundamental relationships: choice of divergence determines model properties, optimization method affects training dynamics. Theoretical insight: no single model dominates all scenarios, choice should match application requirements and constraints.

---

## 🔑 Key DDPM Theory Principles

1. **Hierarchical Diffusion**: DDPM uses a hierarchical approach with many small noise steps, enabling stable training and high-quality generation through gradual structure removal and reconstruction.

2. **Variational Foundation**: Despite using simplified training objectives, DDPM has principled probabilistic foundations through variational lower bound optimization of the evidence.

3. **Score-Based Equivalence**: DDPM is mathematically equivalent to score-based diffusion models, with different parameterizations affecting optimization dynamics but not fundamental capabilities.

4. **Information-Theoretic Design**: Noise schedules should be designed based on information-theoretic principles, balancing structure preservation with sufficient exploration for mode coverage.

5. **Gaussian Approximation**: The Gaussian assumption for reverse transitions is reasonable for small steps but can be limiting for complex distributions, suggesting potential improvements through more flexible transition models.

---

**Next**: Continue with Day 5 - Implementing DDPM from Scratch Theory