# Day 2 - Part 1: Mathematical Foundations of Diffusion Models Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of stochastic processes and their application to diffusion models
- Theoretical analysis of Brownian motion, Gaussian processes, and their properties
- Mathematical principles of Fokker-Planck and Langevin equations in generative modeling
- Information-theoretic perspectives on stochastic differential equations (SDEs)
- Theoretical frameworks for score-based models and their connection to diffusion
- Mathematical modeling of probability flow and neural differential equations

---

## 🌊 Stochastic Processes Theory

### Mathematical Foundation of Random Processes

#### Brownian Motion and Wiener Processes
**Mathematical Definition**:
```
Brownian Motion W(t):
1. W(0) = 0 almost surely
2. Independent increments: W(t) - W(s) ⊥ W(u) - W(v) for non-overlapping intervals
3. Gaussian increments: W(t) - W(s) ~ N(0, t-s) for t > s
4. Continuous paths: W(t) is continuous in t

Mathematical Properties:
- E[W(t)] = 0 (zero mean)
- Var[W(t)] = t (variance grows linearly)
- Cov[W(s), W(t)] = min(s,t) (covariance structure)
- Non-differentiable everywhere (rough paths)

Stochastic Integral:
∫₀ᵗ f(s,ω) dW(s) via Itô construction
Requires special integration theory
Not pathwise Riemann-Stieltjes integrable

Connection to Diffusion:
Brownian motion drives noise in SDEs
Provides mathematical foundation for random perturbations
Central to stochastic calculus framework
```

**Itô Calculus and Stochastic Integration**:
```
Itô's Formula:
For f(x,t) and SDE dx = μ(x,t)dt + σ(x,t)dW:
df = (∂f/∂t + μ∂f/∂x + ½σ²∂²f/∂x²)dt + σ(∂f/∂x)dW

Key Difference from Ordinary Calculus:
(dW)² = dt (Itô isometry)
Second-order terms cannot be neglected
Chain rule includes extra ½σ²∂²f/∂x² term

Quadratic Variation:
[W,W]ₜ = t (deterministic)
[X,X]ₜ for continuous martingale X
Essential for stochastic integration theory

Applications to Diffusion:
- Change of variables in SDEs
- Deriving Fokker-Planck equations
- Converting between SDE formulations
- Computing expectations and variances
```

#### Gaussian Processes and Reproducing Kernel Hilbert Spaces
**Mathematical Framework**:
```
Gaussian Process Definition:
G = {X(t) : t ∈ T} where any finite collection
{X(t₁), ..., X(tₙ)} ~ MultiNormal

Characterization:
Completely specified by:
- Mean function: m(t) = E[X(t)]
- Covariance function: k(s,t) = Cov[X(s), X(t)]

Reproducing Kernel Hilbert Space (RKHS):
H_k = completion of span{k(·,t) : t ∈ T}
Inner product: ⟨f,g⟩_H = ⟨f, k(·,t)⟩_H = f(t)
Reproducing property enables function evaluation

Connection to Score Functions:
Score function s(x) = ∇log p(x) lies in RKHS
Enables regularization and theoretical analysis
Kernel methods for score estimation
RKHS norms control function complexity
```

**Stationarity and Spectral Theory**:
```
Stationary Processes:
k(s,t) = k(t-s) (translation invariant)
Mean and covariance structure unchanging over time

Spectral Representation:
X(t) = ∫ e^{iωt} dZ(ω)
where Z(ω) is complex random measure
Fourier analysis of random processes

Power Spectral Density:
S(ω) = ∫ k(τ)e^{-iωτ} dτ
Frequency domain characterization
Whittle likelihood for parameter estimation

Applications:
- Designing noise schedules in diffusion
- Understanding frequency content of data
- Spectral methods for score estimation
- Optimal filtering and smoothing
```

### Stochastic Differential Equations

#### Fokker-Planck Equation Theory
**Mathematical Derivation**:
```
SDE: dx = μ(x,t)dt + σ(x,t)dW

Forward Kolmogorov (Fokker-Planck) Equation:
∂p/∂t = -∇·(μp) + ½∇²(σ²p)
= -∇·(μp) + ½σ²∇²p (for constant σ)

Physical Interpretation:
- Drift term: -∇·(μp) (deterministic flow)
- Diffusion term: ½σ²∇²p (random spreading)
- Probability conservation: ∫p(x,t)dx = 1

Boundary Conditions:
- Natural boundaries: p(±∞,t) = 0
- Reflecting boundaries: ∂p/∂n = 0
- Absorbing boundaries: p = 0
- Periodic boundaries: p(0,t) = p(L,t)

Stationary Distribution:
∂p/∂t = 0 gives stationary solution p_∞(x)
Detailed balance condition for reversibility
Connection to equilibrium statistical mechanics
```

**Backward Kolmogorov Equation**:
```
Backward Equation:
∂u/∂t + μ∇u + ½σ²∇²u = 0
u(x,T) = f(x) (terminal condition)

Solution Representation:
u(x,t) = E[f(X_T) | X_t = x]
Expected value of terminal payoff
Connection to optimal control theory

Feynman-Kac Formula:
u(x,t) = E[∫ᵗᵀ g(X_s,s)ds + f(X_T) | X_t = x]
Relates PDEs to stochastic integrals
Fundamental in mathematical finance
Useful for expectation computations

Applications in Diffusion:
- Computing marginal probabilities
- Designing optimal noise schedules
- Understanding information flow
- Connecting forward and reverse processes
```

#### Langevin Dynamics and Sampling
**Mathematical Framework**:
```
Langevin SDE:
dx = ∇log p(x)dt + √(2)dW
Converges to stationary distribution p(x)

Overdamped Langevin:
dx = -∇U(x)dt + √(2kT)dW
where U(x) = -log p(x) is potential
Temperature T controls exploration

Underdamped Langevin:
dx = v dt
dv = -γv dt - ∇U(x)dt + √(2γkT)dW
Includes momentum for faster mixing

Convergence Theory:
Exponential convergence under log-concavity
Rate depends on condition number κ
Mixing time: O(κlog(1/ε)) for ε accuracy
```

**Discretization and Numerical Methods**:
```
Euler-Maruyama Scheme:
x_{n+1} = x_n + μ(x_n,t_n)Δt + σ(x_n,t_n)√(Δt)Z_n
where Z_n ~ N(0,I)

Strong vs Weak Convergence:
Strong: E[|X_T - X_T^N|] → 0 (pathwise)
Weak: E[f(X_T)] → E[f(X_T^N)] (distributional)
Different rates and requirements

Higher-Order Schemes:
Milstein: includes (dW)² terms
Runge-Kutta: multistage methods
Stochastic Taylor expansions
Order of convergence analysis

Practical Considerations:
- Step size selection for stability
- Computational cost vs accuracy
- Geometric properties preservation
- Adaptive step size control
```

### Score-Based Models Mathematical Theory

#### Score Function Properties
**Mathematical Definition and Properties**:
```
Score Function:
s(x) = ∇_x log p(x)
Points toward higher probability regions
Independent of normalization constant

Fisher Information Matrix:
I(θ) = E[∇log p(x;θ) ∇log p(x;θ)ᵀ]
= -E[∇²log p(x;θ)]
Measures sensitivity to parameter changes

Score Matching Objective:
J(θ) = ½E_p[||s_θ(x) - ∇log p(x)||²]
= E_p[trace(∇s_θ(x)) + ½||s_θ(x)||²] + const
Integration by parts eliminates ground truth score

Denoising Score Matching:
J_σ(θ) = ½E_p E_q(x̃|x)[||s_θ(x̃,σ) - ∇log q(x̃|x)||²]
where q(x̃|x) = N(x̃; x, σ²I)
Avoids boundary terms and simplifies computation
```

**Information-Theoretic Analysis**:
```
Mutual Information:
I(X; X̃) where X̃ = X + σε, ε ~ N(0,I)
Measures information preserved under noise
Decreases as σ increases

Rate-Distortion Function:
R(D) = min I(X; X̃) subject to E[||X - X̃||²] ≤ D
Optimal compression-distortion trade-off
Connection to denoising autoencoders

Score Estimation Error:
||s_θ(x,σ) - ∇log p_σ(x)||²
Affects sampling quality and convergence
Depends on network capacity and training data

Generalization Bounds:
Sample complexity for score estimation
Depends on data dimension and smoothness
High-dimensional curse affects convergence rates
```

#### Annealed Langevin Dynamics
**Multi-Scale Framework**:
```
Noise Schedule:
σ₁ > σ₂ > ... > σ_L with σ_L ≪ data_scale ≪ σ₁
Geometric progression: σᵢ = σ₁(σ_L/σ₁)^{(i-1)/(L-1)}

Perturbed Data Distribution:
p_σ(x) = ∫ p_data(y) N(x; y, σ²I) dy
Smoothed version of data distribution
Removes sharp modes and discontinuities

Annealed Sampling:
Start with x₀ ~ N(0, σ₁²I)
For i = 1, ..., L:
  Run Langevin MCMC targeting p_σᵢ(x)
  Initialize from previous level
Gradually refine samples

Mathematical Justification:
Large σ: global exploration, easy mixing
Small σ: local refinement, detailed structure
Smooth interpolation between noise and data
```

**Convergence Analysis**:
```
Mixing Time Analysis:
τ_mix ∝ κ/log(σ²) where κ is condition number
Better conditioning at high noise levels
Faster mixing enables global exploration

Multi-Scale Convergence:
Overall error bounded by sum of level errors
Early levels: coarse structure (fast convergence)
Late levels: fine details (slow convergence)
Trade-off between levels and mixing time

Approximation vs Sampling Error:
Approximation: score network accuracy
Sampling: finite-time MCMC error
Both contribute to generation quality
Optimal balance depends on computational budget

Theoretical Guarantees:
Under regularity conditions:
Generated samples approach data distribution
Convergence rate depends on score estimation error
Polynomial dependence on dimension and condition number
```

---

## 🔀 Neural Stochastic Differential Equations

### Mathematical Framework

#### Neural SDE Theory
**Parameterization of SDEs**:
```
Neural SDE:
dx = f_θ(x,t)dt + g_φ(x,t)dW
where f_θ, g_φ are neural networks

Drift and Diffusion Learning:
f_θ: deterministic component (neural ODE)
g_φ: stochastic component (noise adaptation)
Joint optimization over θ, φ

Universal Approximation:
Neural networks can approximate any continuous drift/diffusion
Depth and width requirements for approximation quality
Regularity conditions for SDE existence and uniqueness

Training Objective:
Minimize KL divergence between data and model distributions
Equivalent to maximum likelihood for observed data
Requires efficient likelihood computation or approximation
```

**Probability Flow ODE**:
```
Deterministic ODE:
dx = [f(x,t) - ½g(t)²∇_x log p_t(x)]dt
Same marginal distributions as original SDE
Removes stochasticity while preserving probability evolution

Neural ODE Connection:
Continuous-depth neural networks
Backpropagation through ODE solvers
Memory-efficient gradient computation
Regularization through ODE dynamics

Sampling via ODE:
Deterministic sampling from Gaussian noise
Faster than stochastic sampling methods
Exact likelihood computation possible
Trade-off: determinism vs stochasticity
```

#### Continuous Normalizing Flows
**Mathematical Foundation**:
```
Flow Equation:
∂p/∂t + ∇·(vp) = 0
Continuity equation for probability transport
v(x,t): velocity field (neural network)

Change of Variables:
log p₁(x₁) = log p₀(x₀) - ∫₀¹ ∇·v(x(t),t) dt
Trace of Jacobian integrated over path
Enables exact likelihood computation

Hutchinson's Trace Estimator:
trace(∇·v) = E_ε[εᵀ∇·v(x)ε] where ε ~ N(0,I)
Unbiased estimator with O(1) complexity
Enables scalable likelihood computation
Variance depends on smoothness of v

Optimal Transport Connection:
Find velocity field minimizing transport cost
Connection to Wasserstein distances
Regularization through kinetic energy
Monge-Ampère equation solutions
```

### Advanced Theoretical Analysis

#### Information Geometry of Score Functions
**Geometric Perspective**:
```
Score Function as Vector Field:
s(x) = ∇log p(x) defines vector field on data manifold
Geometric flow toward high-probability regions
Connection to gradient flows and optimization

Riemannian Metric:
Fisher information metric: g_ij = E[∂log p/∂θᵢ ∂log p/∂θⱼ]
Natural geometry for probability distributions
Geodesics are efficient paths in parameter space

Wasserstein Geometry:
Optimal transport distances between distributions
2-Wasserstein: W₂²(p,q) = inf E[||X-Y||²] over couplings
Connection to score-based dynamics
Gradient flows in Wasserstein space

Natural Gradients:
Preconditioned by Fisher information matrix
Faster convergence in probability space
Connection to mirror descent algorithms
Practical implementation challenges
```

**Manifold Hypothesis and Intrinsic Dimension**:
```
Data Manifold Structure:
Real data lies on low-dimensional manifold M ⊂ ℝᵈ
Intrinsic dimension d_int ≪ d (ambient dimension)
Score function tangent to manifold

Manifold Score Matching:
Project score to tangent space of manifold
Requires manifold structure knowledge or estimation
Connection to principal component analysis
Spectral methods for dimension reduction

Curvature Effects:
Gaussian curvature affects diffusion behavior
Mean curvature influences drift terms
Ricci curvature controls convergence rates
Geometric deep learning connections

Sample Complexity:
Depends on intrinsic rather than ambient dimension
Curse of dimensionality partially alleviated
Manifold learning improves generalization
Adaptive methods for unknown manifolds
```

---

## 🎯 Advanced Understanding Questions

### Stochastic Process Theory:
1. **Q**: Analyze the mathematical relationship between different types of stochastic processes (Brownian motion, Lévy processes, jump diffusions) and their suitability for different generative modeling scenarios.
   **A**: Mathematical comparison: Brownian motion provides continuous Gaussian increments ideal for smooth data generation. Lévy processes include jumps enabling discontinuous generation suitable for discrete data or mode switching. Jump diffusions combine continuous drift with discrete jumps, modeling data with both smooth evolution and sudden changes. Suitability analysis: Brownian motion best for natural images (smooth textures), Lévy processes for discrete structures (text, graphs), jump diffusions for mixed continuous-discrete data. Mathematical trade-offs: continuity vs expressiveness, computational tractability vs modeling flexibility. Theoretical insight: process choice should match data structure and desired generation properties.

2. **Q**: Develop a theoretical framework for analyzing the convergence properties of different discretization schemes for SDEs in the context of diffusion models.
   **A**: Convergence framework: strong convergence (pathwise) vs weak convergence (distributional) with different rates and applications. Euler-Maruyama: strong order 0.5, weak order 1.0, simple but potentially unstable. Milstein: includes second-order corrections, strong order 1.0 for scalar SDEs. Runge-Kutta: higher-order schemes with better stability. Analysis factors: step size, noise coefficient magnitude, drift term properties. Diffusion-specific considerations: score function smoothness affects convergence, adaptive step sizing based on noise level, geometric properties preservation. Theoretical guarantee: under appropriate smoothness conditions, discretization error can be made arbitrarily small with sufficient computational cost.

3. **Q**: Compare the mathematical foundations of Fokker-Planck equations and Langevin dynamics, analyzing their role in understanding diffusion model training and sampling.
   **A**: Mathematical relationship: Fokker-Planck describes probability evolution, Langevin provides sampling mechanism. Both arise from same underlying SDE but focus on different aspects. Fokker-Planck: ∂p/∂t = -∇·(μp) + ½∇²(σ²p) governs distribution evolution. Langevin: dx = μ(x,t)dt + σ(x,t)dW generates samples from evolving distribution. Training connection: score matching approximates gradients of log-probability appearing in Langevin dynamics. Sampling connection: reverse process implements Langevin sampling with learned score. Theoretical insight: both frameworks essential for understanding diffusion models from complementary perspectives (distribution evolution vs particle dynamics).

### Score-Based Model Theory:
4. **Q**: Analyze the mathematical trade-offs between different score matching objectives (basic, denoising, sliced) and their impact on generation quality and computational efficiency.
   **A**: Mathematical comparison: basic score matching requires boundary conditions and integration by parts, computationally expensive. Denoising score matching uses perturbed data, avoids boundary terms, computationally efficient. Sliced score matching projects onto random directions, reduces computational cost but may lose information. Trade-offs: accuracy vs efficiency, boundary condition requirements vs approximation quality. Generation impact: denoising enables multi-scale learning, sliced reduces computational cost but may affect fine details. Theoretical analysis: all methods consistent under appropriate conditions, differ in finite-sample behavior and computational requirements. Optimal choice depends on data characteristics and computational constraints.

5. **Q**: Develop a mathematical theory for the information content and reconstruction capabilities of score functions at different noise levels in diffusion models.
   **A**: Information theory framework: I(X₀; Xₜ) = H(X₀) - H(X₀|Xₜ) measures preserved information at noise level t. Low noise: high information content, detailed structure preserved, difficult global exploration. High noise: low information content, global structure only, easy exploration. Score function accuracy requirements vary: high noise tolerates approximation errors, low noise requires high precision. Reconstruction capability: depends on both information content and score estimation quality. Mathematical bound: reconstruction error bounded by score estimation error and remaining information content. Optimal strategy: balance noise levels to maintain sufficient information while enabling tractable score estimation.

6. **Q**: Compare the theoretical properties of continuous-time and discrete-time diffusion formulations, analyzing their mathematical equivalence and practical implications.
   **A**: Mathematical equivalence: discrete diffusion is finite-difference approximation of continuous SDE. Continuous: mathematically elegant, enables theoretical analysis, connects to differential equations. Discrete: computationally practical, finite-step guarantees, simpler implementation. Theoretical properties: both learn same score function, differ in numerical integration. Convergence analysis: discrete approximation error decreases with smaller time steps. Practical implications: continuous enables adaptive solvers and higher-order methods, discrete simplifies training and inference. Key insight: formulation choice affects computational efficiency and theoretical analysis but both approaches fundamentally equivalent for sufficient time resolution.

### Advanced Applications:
7. **Q**: Design a mathematical framework for analyzing the sample complexity and generalization bounds of score-based generative models on high-dimensional manifolds.
   **A**: Framework components: (1) manifold structure characterization (intrinsic dimension, curvature), (2) score function approximation on manifolds, (3) generalization from training to test data. Sample complexity: depends on intrinsic dimension d_int rather than ambient dimension d. Mathematical bound: O(d_int log(n)/n) convergence rate under smoothness assumptions. Manifold-specific considerations: curvature affects mixing time, tangent space approximation errors, geodesic distance metrics. Generalization analysis: uniform convergence over function class, stability of score estimation, robustness to manifold perturbations. Theoretical guarantee: under appropriate regularity conditions, score-based models achieve minimax optimal rates for manifold-supported distributions.

8. **Q**: Develop a unified mathematical theory connecting diffusion models, optimal transport, and variational inference, identifying fundamental relationships and trade-offs.
   **A**: Unified framework: all three minimize divergences between distributions but use different metrics and optimization methods. Optimal transport: minimizes Wasserstein distance through transport maps. Variational inference: minimizes KL divergence through approximate posteriors. Diffusion models: minimize score matching objective related to Fisher divergence. Mathematical connections: all can be viewed as gradient flows in appropriate spaces (Wasserstein, KL, Fisher). Trade-offs: computational tractability vs theoretical optimality, approximation quality vs practical feasibility. Fundamental relationships: score-based dynamics correspond to Wasserstein gradient flows, variational objectives relate to information geometry. Theoretical insight: choice of method depends on problem structure, computational constraints, and desired theoretical guarantees.

---

## 🔑 Key Mathematical Foundations Principles

1. **Stochastic Process Foundation**: Diffusion models are fundamentally based on stochastic processes, with Brownian motion and SDEs providing the mathematical framework for understanding noise addition and removal.

2. **Score Function Centrality**: The score function ∇log p(x) is central to diffusion models, providing a tractable way to learn distributions without normalizing constants while enabling principled sampling via Langevin dynamics.

3. **Multi-Scale Information Theory**: Different noise levels preserve different amounts of information about the original data, with high noise enabling global exploration and low noise preserving fine details.

4. **Continuous-Discrete Duality**: Both continuous-time (SDE) and discrete-time formulations are mathematically valid, with the choice affecting computational efficiency and theoretical analysis but not fundamental capabilities.

5. **Convergence Guarantees**: Under appropriate regularity conditions, score-based diffusion models provide theoretical guarantees for convergence to the data distribution, with rates depending on dimension, smoothness, and approximation quality.

---

**Next**: Continue with Day 3 - Denoising Score Matching (DSM) Theory