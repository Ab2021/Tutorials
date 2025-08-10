# Day 3 - Part 1: Denoising Score Matching (DSM) Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of score matching and its connection to energy-based models
- Theoretical analysis of denoising score matching and its advantages over basic score matching
- Mathematical principles of noise-conditional score networks and multi-scale training
- Information-theoretic perspectives on denoising and reconstruction objectives
- Theoretical frameworks for annealed Langevin dynamics and sampling procedures
- Mathematical modeling of score function approximation and generalization bounds

---

## 🎯 Score Matching Mathematical Theory

### Fundamental Score Matching Framework

#### Score Function and Fisher Divergence
**Mathematical Foundation**:
```
Score Function Definition:
s(x) = ∇_x log p(x)
Logarithmic gradient of probability density
Points toward regions of higher probability

Fisher Divergence:
D_F(p || q) = ½E_p[||∇log p(x) - ∇log q(x)||²]
Measures difference between score functions
Avoids explicit density normalization
Related to Fisher Information Matrix

Score Matching Objective:
J(θ) = ½E_p_data[||s_θ(x) - ∇log p_data(x)||²]
Learn score function without knowing p_data
Requires integration by parts transformation

Transformed Objective (Hyvärinens trick):
J(θ) = E_p_data[trace(∇s_θ(x)) + ½||s_θ(x)||²] + const
Eliminates unknown score ∇log p_data(x)
Requires boundary conditions: lim_{||x||→∞} p_data(x)s_θ(x) = 0
```

**Information-Theoretic Interpretation**:
```
Connection to Maximum Likelihood:
Score matching ≈ MLE for exponential families
Consistent estimator under regularity conditions
Fisher-optimal when model class contains true distribution

Relationship to Energy-Based Models:
p_θ(x) = exp(-E_θ(x))/Z_θ
Score: s_θ(x) = -∇E_θ(x)
Avoids intractable partition function Z_θ

Statistical Efficiency:
Score matching achieves √n convergence rate
Comparable to MLE under smoothness assumptions
Robust to model misspecification
Lower sample complexity than adversarial methods
```

#### Practical Challenges of Basic Score Matching
**Computational Issues**:
```
Trace Computation:
trace(∇s_θ(x)) = Σᵢ ∂s_θ,i/∂xᵢ
Requires computing diagonal of Hessian
O(d) forward passes for d-dimensional data
Hutchinson estimator: trace(A) ≈ ε^T A ε for ε ~ N(0,I)

Boundary Conditions:
Assumption: lim_{||x||→∞} p_data(x)s_θ(x) = 0
Often violated in practice
Boundary effects corrupt score estimates
Particularly problematic for unbounded support

High-Dimensional Challenges:
Curse of dimensionality affects score estimation
Requires large networks for accurate approximation
Sample complexity scales with dimension
Boundary issues amplified in high dimensions

Numerical Stability:
Score functions can have large gradients
Optimization challenges near data boundaries
Gradient explosion in low-density regions
Requires careful initialization and regularization
```

### Denoising Score Matching Theory

#### Mathematical Formulation
**Noise-Perturbed Distributions**:
```
Perturbed Data Distribution:
p_σ(x) = ∫ p_data(y) q_σ(x|y) dy
where q_σ(x|y) = N(x; y, σ²I)

Noise-Conditional Score:
s_σ(x) = ∇_x log p_σ(x)
Score of noise-perturbed distribution
Smoother than original score function

Analytical Expression:
For Gaussian noise: ∇_x log q_σ(x|y) = -(x-y)/σ²
Exact gradient available during training
No boundary conditions required

Multi-Scale Framework:
{σ₁, σ₂, ..., σ_L} with σ₁ > σ₂ > ... > σ_L
Cover different scales of data structure
Annealed training from coarse to fine
```

**Denoising Score Matching Objective**:
```
DSM Loss Function:
L_DSM(θ) = ½Σₗ λ(σₗ) E_p_data E_q_σₗ[||s_θ(x,σₗ) - ∇log q_σₗ(x|y)||²]

Explicit Form:
L_DSM(θ) = ½Σₗ λ(σₗ) E_p_data E_ε~N(0,I)[||s_θ(y + σₗε, σₗ) + ε/σₗ||²]

Weight Function λ(σ):
Balances different noise levels
Common choices: λ(σ) = σ², λ(σ) = 1, λ(σ) = 1/σ²
Affects relative importance of scales

Mathematical Properties:
- No boundary conditions required
- Tractable exact gradients
- Consistent estimator as σ → 0
- Smooth optimization landscape
```

#### Theoretical Analysis
**Consistency and Convergence**:
```
Convergence as σ → 0:
lim_{σ→0} s_σ(x) = s(x) = ∇log p_data(x)
DSM recovers true score in limit
Rate depends on smoothness of p_data

Approximation Error:
||s_σ(x) - s(x)|| = O(σ²) for smooth densities
Bias-variance trade-off in noise level selection
Smaller σ: less bias, more variance
Larger σ: more bias, less variance

Sample Complexity:
Õ(d/ε²) samples for ε-accurate score estimation
Dimension dependence through covering numbers
Improved constants compared to basic score matching
Denoising provides implicit regularization

Generalization Bounds:
With probability 1-δ:
||s_θ - s_σ||²_L² ≤ training_error + O(√(log(1/δ)/n))
Depends on function class complexity
Neural network approximation theory
```

**Information-Theoretic Analysis**:
```
Mutual Information Perspective:
I(Y; Y + σε) = H(Y) - H(σε) + H(Y,ε) - H(Y + σε)
Information preserved under noise corruption
Decreases as σ increases

Rate-Distortion Connection:
Denoising autoencoder connection
R(D) = min I(Y; Ỹ) subject to E[||Y - Ỹ||²] ≤ D
Score matching learns optimal reconstruction
Minimax relationship between compression and distortion

Score Fisher Information:
I_s(p_σ) = E_p_σ[||∇log p_σ(x)||²]
Measures sensitivity of log-likelihood
Related to convergence rates
Finite for smoothed distributions
```

### Multi-Scale Score Networks

#### Noise Schedule Design
**Mathematical Principles**:
```
Geometric Progression:
σᵢ = σ_max · (σ_min/σ_max)^{(i-1)/(L-1)}
Covers wide range of scales efficiently
Logarithmic spacing optimal for many distributions

Coverage Analysis:
Largest scale σ_max ≈ max_x ||x - E[x]||
Smallest scale σ_min ≈ data_precision
Ensures coverage of all data scales
Number of scales L ≈ log(σ_max/σ_min)

Adaptive Schedules:
Data-dependent noise level selection
Estimate local density smoothness
Allocate more levels where needed
Information-theoretic criteria for optimization

Theoretical Optimality:
Minimax optimal under certain conditions
Relates to approximation theory
Trade-off between bias and variance
Computational budget allocation
```

**Conditioning Mechanisms**:
```
Noise-Conditional Architecture:
s_θ(x, σ) = f_θ(x, g(σ))
where g(σ) encodes noise level

Embedding Strategies:
- Positional encoding: g(σ) = [sin(log σ), cos(log σ)]
- Learned embedding: g(σ) via lookup table
- Feature modulation: FiLM layers

Scale Equivariance:
s_θ(αx, ασ) = s_θ(x, σ) for scale transformations
Architectural constraints for invariance
Improves generalization across scales

Mathematical Properties:
Network capacity allocation across scales
Feature sharing vs scale-specific parameters
Optimization dynamics with multiple objectives
Gradient flow analysis across scales
```

#### Annealed Langevin Dynamics

**Sampling Procedure**:
```
Multi-Scale Sampling:
Initialize: x₀ ~ N(0, σ₁²I)
For i = 1, ..., L:
  For t = 1, ..., T:
    x_t = x_{t-1} + ε_i s_θ(x_{t-1}, σᵢ) + √(2ε_i) z_t
    where z_t ~ N(0,I)
  Use final x_T as initialization for next scale

Step Size Selection:
ε_i ∝ σᵢ²/d (dimension-dependent scaling)
Ensures stable sampling across scales
Adaptive step sizes based on score magnitude

Temperature Annealing:
Gradual reduction in effective temperature
σᵢ controls exploration vs exploitation
Connection to simulated annealing
```

**Convergence Theory**:
```
Mixing Time Analysis:
Each scale has characteristic mixing time
τ_mix,i ∝ (radius/σᵢ)² for convex distributions
Coarse scales: fast mixing, global exploration
Fine scales: slow mixing, local refinement

Overall Convergence:
Total error bounded by sum over scales
Approximation error: score network accuracy
Sampling error: finite-time MCMC
Discretization error: finite step size

Geometric Convergence:
Under log-concavity and smoothness:
||p_t - p_∞||_TV ≤ Ce^{-t/τ}
Rate depends on condition number
Multi-scale improves effective conditioning

Practical Considerations:
Trade-off between accuracy and computation
Early stopping criteria
Adaptive number of steps per scale
Temperature scheduling optimization
```

### Advanced Theoretical Analysis

#### Manifold Score Matching
**Low-Dimensional Structure**:
```
Data Manifold Hypothesis:
Data lies on unknown manifold M ⊂ ℝᵈ
Intrinsic dimension d_int << d
Score function tangent to manifold

Manifold Denoising:
Noise perpendicular to manifold
s_θ(x, σ) approximates projection to tangent space
Connection to principal component analysis
Spectral methods for manifold learning

Geometric Score Matching:
Incorporate manifold structure in loss
Tangent space regularization
Curvature-aware sampling procedures
Riemannian Langevin dynamics

Sample Complexity:
Depends on intrinsic dimension d_int
Exponential improvement over ambient dimension
Manifold learning improves score estimation
Adaptive methods for unknown manifolds
```

**Differential Geometry Connection**:
```
Riemannian Manifolds:
Metric tensor g_ij for distance measurement
Connection Γ^k_{ij} for parallel transport
Curvature tensor R^l_{ijk} for geometric properties

Score on Manifolds:
Riemannian gradient: ∇_M log p = g^{-1}∇log p
Natural gradient in data geometry
Faster convergence than Euclidean gradient

Langevin on Manifolds:
dx = (∇_M log p + ½trace(∇g))dt + dW_M
Additional drift term from metric
Preserves manifold structure during sampling
```

#### Neural Network Approximation Theory
**Universal Approximation**:
```
Score Function Approximation:
Neural networks can approximate continuous score functions
Depth and width requirements for accuracy
ReLU networks: piecewise linear approximation
Smooth activations: better for score estimation

Approximation Rates:
For s ∈ C^k(bounded domain):
||s - s_θ||_∞ ≤ C · width^{-k/d}
Curse of dimensionality in worst case
Better rates under structural assumptions

Optimization Landscape:
Score matching has benign optimization properties
No spurious local minima under overparameterization
Gradient descent convergence guarantees
Connection to neural tangent kernel theory

Generalization Analysis:
Rademacher complexity bounds
Depends on network architecture and data distribution
Score estimation smoother than density estimation
Implicit regularization from denoising
```

---

## 🎯 Advanced Understanding Questions

### Score Matching Theory:
1. **Q**: Analyze the mathematical relationship between Fisher divergence minimization and maximum likelihood estimation, deriving conditions under which they are equivalent.
   **A**: Mathematical relationship: Fisher divergence D_F(p||q) = ½E_p[||∇log p - ∇log q||²] related to MLE through score functions. Equivalence conditions: (1) exponential family models p_θ(x) = exp(θᵀt(x) - A(θ)), (2) sufficient statistics t(x) with finite Fisher information. Under these conditions, score matching becomes ∇A(θ) = E_p[t(x)], identical to MLE moment matching. Analysis: score matching achieves MLE optimality when model class includes true distribution. Key insight: score matching provides MLE benefits without computing intractable normalizing constants, making it practical for energy-based models.

2. **Q**: Develop a theoretical framework for analyzing the bias-variance trade-off in denoising score matching across different noise levels and data distributions.
   **A**: Framework components: bias from noise smoothing, variance from finite samples. Mathematical analysis: bias ||E[s_θ(x,σ)] - ∇log p_data(x)|| ≤ Cσ² for smooth p_data. Variance: Var[s_θ(x,σ)] ∝ 1/(nσ²) for n samples. Optimal noise level: σ* minimizes bias² + variance ∝ n^{-1/(4+d)} showing curse of dimensionality. Data distribution effects: heavy tails require larger σ, sharp modes require smaller σ. Trade-off optimization: adaptive noise schedules based on local data characteristics. Theoretical insight: multi-scale approach necessary to handle diverse data structures efficiently.

3. **Q**: Compare the mathematical foundations of different score matching variants (basic, denoising, sliced, spectral) and analyze their computational and statistical trade-offs.
   **A**: Mathematical comparison: Basic score matching minimizes Fisher divergence exactly but requires boundary conditions and trace computation. Denoising perturbs data with noise, avoiding boundary issues but introducing bias. Sliced projects to random directions, reducing computation but losing information. Spectral uses frequency domain, efficient for stationary processes. Trade-offs: basic (accurate but expensive), denoising (practical with controlled bias), sliced (fast but approximate), spectral (efficient for specific data types). Statistical properties: all consistent under appropriate conditions, differ in finite-sample behavior. Computational complexity: basic O(d²), denoising O(d), sliced O(d), spectral O(d log d). Optimal choice depends on data characteristics and computational constraints.

### Denoising and Multi-Scale Theory:
4. **Q**: Analyze the mathematical principles behind optimal noise schedule design in multi-scale denoising score matching, deriving theoretical bounds on approximation quality.
   **A**: Optimal noise schedule principles: cover data scales efficiently while maintaining approximation quality. Mathematical framework: approximation error ||s_σ(x) - s(x)|| ≤ Cσ²||∇²log p(x)|| for smooth densities. Coverage requirement: σ_max ≥ data_diameter, σ_min ≤ noise_floor. Geometric progression optimal: minimizes total approximation error under computational budget. Theoretical bounds: total error ≤ Σᵢ(bias²ᵢ + varianceᵢ) with optimal allocation giving n^{-1/(4+d)} rate. Adaptive schedules: data-dependent noise selection based on local smoothness estimation. Key insight: logarithmic spacing balances bias-variance trade-off across scales while ensuring computational efficiency.

5. **Q**: Develop a mathematical theory for the convergence properties of annealed Langevin dynamics, considering both approximation and sampling errors.
   **A**: Convergence theory components: (1) approximation error from score network, (2) sampling error from finite-time MCMC, (3) discretization error from finite step size. Mathematical analysis: total error bounded by ||p_generated - p_data||_TV ≤ C₁ε_approx + C₂ε_sampling + C₃ε_discretization. Approximation: depends on network capacity and training quality. Sampling: geometric convergence under log-concavity with rate ∝ condition number. Discretization: error O(step_size) for first-order schemes. Multi-scale improvement: effective condition number reduced by noise, faster mixing at coarse scales. Theoretical guarantee: under regularity conditions, generated samples converge to data distribution with rate depending on approximation quality and computational budget.

6. **Q**: Compare the information-theoretic properties of different conditioning mechanisms in noise-conditional score networks, analyzing their impact on generation quality.
   **A**: Information-theoretic analysis: conditioning mechanism determines how noise level information flows through network. Concatenation: simple but may not preserve scale structure. FiLM (Feature-wise Linear Modulation): multiplicative conditioning preserves relative feature magnitudes. Attention: adaptive importance weighting across features. Analysis: FiLM preserves scale equivariance better than concatenation, attention provides most flexibility but highest complexity. Generation quality: proper conditioning essential for multi-scale consistency, poor conditioning causes artifacts at scale transitions. Mathematical framework: mutual information I(features; noise_level) measures conditioning effectiveness. Optimal choice: FiLM for natural images (scale structure important), attention for complex structured data, concatenation for simple baselines.

### Advanced Applications:
7. **Q**: Design a mathematical framework for denoising score matching on non-Euclidean manifolds, addressing the challenges of intrinsic geometry and curvature.
   **A**: Framework components: (1) Riemannian manifold structure (metric, connection, curvature), (2) manifold-aware noise model, (3) geometric score matching objective. Mathematical formulation: score s_M(x) = ∇_M log p(x) using Riemannian gradient. Noise model: Brownian motion on manifold with heat kernel q_t(x|y). Geometric DSM: minimize E[||s_θ(x,t) - ∇_M log q_t(x|y)||²_g] where ||·||_g is Riemannian norm. Challenges: curvature affects noise distribution, parallel transport needed for gradients, computational complexity increases. Solutions: local coordinate patches, finite element methods, neural network approximation of geometric quantities. Theoretical guarantee: convergence to manifold score function under appropriate regularity conditions.

8. **Q**: Develop a unified mathematical theory connecting denoising score matching to variational autoencoders and optimal transport, identifying fundamental relationships and practical implications.
   **A**: Unified theory: all three frameworks minimize divergences between distributions but use different metrics. DSM minimizes Fisher divergence through score matching. VAEs minimize reverse KL divergence through evidence lower bound. Optimal transport minimizes Wasserstein distance through transport maps. Mathematical connections: score-based dynamics are Wasserstein gradient flows, VAE encoder approximates optimal transport map, all relate to information geometry. Practical implications: DSM provides stable training without adversarial dynamics, VAEs enable encoder-decoder structure, optimal transport gives principled distance metrics. Fundamental relationships: all can be viewed as different discretizations of the same underlying continuous optimization problem in probability space. Key insight: method choice should match problem structure and computational constraints.

---

## 🔑 Key Denoising Score Matching Principles

1. **Noise as Regularization**: Adding noise to data serves as implicit regularization, smoothing distributions and making score estimation more tractable while avoiding boundary condition issues.

2. **Multi-Scale Information**: Different noise levels capture different scales of data structure, from global geometry (high noise) to fine details (low noise), requiring multi-scale training approaches.

3. **Fisher Divergence Optimization**: Score matching minimizes Fisher divergence between model and data distributions, providing a principled alternative to maximum likelihood that avoids normalizing constants.

4. **Annealed Sampling**: Gradual noise reduction during sampling enables global exploration at high noise levels and local refinement at low noise levels, improving convergence and sample quality.

5. **Approximation-Sampling Duality**: Generation quality depends on both score function approximation accuracy and sampling procedure effectiveness, requiring joint optimization of network training and sampling algorithms.

---

**Next**: Continue with Day 4 - Denoising Diffusion Probabilistic Models (DDPM) Theory