# Day 6 - Part 1: Sampling Techniques in Diffusion Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of deterministic vs stochastic sampling in diffusion models
- Theoretical analysis of DDIM and accelerated sampling techniques
- Mathematical principles of conditional sampling and guidance mechanisms
- Information-theoretic perspectives on sampling quality and diversity trade-offs
- Theoretical frameworks for adaptive sampling and step size optimization
- Mathematical modeling of sampling trajectory optimization and convergence analysis

---

## 🎯 Theoretical Foundation of Diffusion Sampling

### Mathematical Framework of Reverse Process

#### Ancestral Sampling (DDPM)
**Mathematical Formulation**:
```
DDPM Reverse Process:
x_{t-1} = μ_θ(x_t, t) + σ_t z_t
where z_t ~ N(0, I) and σ_t² = β_t

Mean Prediction:
μ_θ(x_t, t) = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t))

Variance Schedule:
σ_t² = β_t (original DDPM)
σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) β_t (optimal for VLB)

Stochastic Properties:
- Maintains reverse process variance
- Each step introduces fresh randomness
- Enables diverse sample generation
- T steps required for complete generation
```

**Information-Theoretic Analysis**:
```
Entropy Evolution:
H(x_{t-1} | x_t) = ½ log(2πeσ_t²)
Higher variance → higher sampling entropy
Stochasticity enables mode exploration

Mutual Information:
I(x_0; x_t) decreases with t
Ancestral sampling preserves information structure
Random walk in probability space

Sampling Distribution:
p_sample(x_0) ≈ p_data(x_0) as T → ∞
Convergence guaranteed under regularity conditions
Quality depends on score estimation accuracy

Trajectory Diversity:
Different random seeds yield different samples
Stochasticity essential for generation diversity
Trade-off: diversity vs computational cost
```

#### Deterministic Sampling Theory
**Mathematical Foundation**:
```
Probability Flow ODE:
dx = [f(x,t) - ½g(t)²s_θ(x,t)]dt
Deterministic trajectory with same marginals as SDE

Neural ODE Connection:
x_0 = ODESolve(x_T, score_function, [T,0])
Continuous-time formulation
Memory-efficient backpropagation through solver

Mathematical Properties:
- Unique trajectory for given initial condition
- Preserves marginal distributions
- Enables exact likelihood computation
- Faster sampling with adaptive solvers

Discretization:
Euler method: x_{t-1} = x_t + h·[f(x_t,t) - ½g(t)²s_θ(x_t,t)]
Higher-order methods: Runge-Kutta, Adams-Bashforth
Step size h affects accuracy vs speed trade-off
```

### DDIM: Denoising Diffusion Implicit Models

#### Mathematical Derivation
**Non-Markovian Process**:
```
DDIM Forward Process:
q_σ(x_{1:T} | x_0) where σ controls stochasticity
Generalizes DDPM (σ = 1) to deterministic (σ = 0)

Marginal Consistency:
q_σ(x_t | x_0) = q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
Same marginals as DDPM regardless of σ

Reverse Process:
x_{t-1} = √ᾱ_{t-1} (x_t - √(1-ᾱ_t)ε_θ(x_t,t))/√ᾱ_t 
         + √(1-ᾱ_{t-1}-σ_t²) ε_θ(x_t,t) + σ_t ε_t

Deterministic Case (σ = 0):
x_{t-1} = √ᾱ_{t-1} (x_t - √(1-ᾱ_t)ε_θ(x_t,t))/√ᾱ_t 
         + √(1-ᾱ_{t-1}) ε_θ(x_t,t)
```

**Accelerated Sampling**:
```
Subsequence Sampling:
Use subset S ⊂ {1,2,...,T} with |S| << T
Maintain same noise schedule relationships
Significantly fewer steps (e.g., 50 vs 1000)

Mathematical Justification:
Deterministic trajectory is smooth
Dense sampling not necessary for good approximation
Adaptive step sizes based on local curvature

Error Analysis:
Discretization error ∝ step_size²
Large steps may accumulate errors
Trade-off between speed and quality
Empirically robust for many applications
```

#### Theoretical Analysis
**Convergence Properties**:
```
Deterministic Convergence:
For σ = 0, same initial condition → same output
Enables reproducible generation
Invertible mapping (theoretically)

Stochastic Interpolation:
σ ∈ [0,1] interpolates between deterministic and stochastic
Controllable diversity level
σ = 0: fast, deterministic
σ = 1: slow, diverse (equivalent to DDPM)

Stability Analysis:
DDIM often more stable than DDPM
Less accumulation of approximation errors
Better for few-step sampling
Deterministic nature aids debugging

Approximation Quality:
Error bounds depend on score estimation accuracy
Smoother trajectories → better approximation
Fewer discretization artifacts
Consistent quality across different step counts
```

### Advanced Sampling Algorithms

#### Higher-Order Methods
**Runge-Kutta Integration**:
```
RK4 for Probability Flow ODE:
k1 = h·[f(x_t,t) - ½g(t)²s_θ(x_t,t)]
k2 = h·[f(x_t+k1/2,t+h/2) - ½g(t+h/2)²s_θ(x_t+k1/2,t+h/2)]
k3 = h·[f(x_t+k2/2,t+h/2) - ½g(t+h/2)²s_θ(x_t+k2/2,t+h/2)]
k4 = h·[f(x_t+k3,t+h) - ½g(t+h)²s_θ(x_t+k3,t+h)]
x_{t-1} = x_t + (k1 + 2k2 + 2k3 + k4)/6

Mathematical Benefits:
O(h⁴) accuracy vs O(h) for Euler
Better error control for large steps
Adaptive step size possible
Higher computational cost per step

Stability Properties:
Larger stability region than Euler
Less sensitive to stiff equations
Better preservation of invariants
Improved long-term accuracy
```

**Exponential Integrators**:
```
Mathematical Formulation:
Exploit exponential structure of linear terms
Exact integration of linear components
Numerical integration only for nonlinear terms

Error Analysis:
Reduced error accumulation
Better conditioning for stiff problems
Specialized for particular equation structures
Computational overhead for matrix exponentials

Applications:
Suitable for certain noise schedules
Potentially faster convergence
Research area for diffusion models
Not yet widely adopted
```

#### Adaptive Sampling Strategies
**Mathematical Framework**:
```
Error Estimation:
Local truncation error: τ = ||x_predicted - x_exact||
Richardson extrapolation for error estimates
Embedded methods (e.g., Dormand-Prince)

Step Size Control:
h_new = h_old × (tolerance/τ)^{1/(p+1)}
where p is method order
Automatic adaptation to solution smoothness

Timestep Selection:
Concentrate steps where score changes rapidly
Sparse sampling in smooth regions
Information-theoretic criteria for step placement

Mathematical Benefits:
Guaranteed error bounds
Optimal computational efficiency
Robust to different problem characteristics
Minimal user parameter tuning
```

### Conditional Sampling Theory

#### Mathematical Framework
**Conditional Generation**:
```
Conditional Distribution:
p(x_0 | c) where c is conditioning information
Examples: class labels, text, images

Conditional Score:
s_θ(x_t, t, c) = ∇_{x_t} log p_t(x_t | c)
Learns conditional score function
Network takes additional conditioning input

Conditional Reverse Process:
x_{t-1} = μ_θ(x_t, t, c) + σ_t z_t
Mean depends on conditioning
Variance schedule unchanged

Training Objective:
L = E_{t,x_0,c,ε}[||ε - ε_θ(x_t, t, c)||²]
Include conditioning in noise prediction
Classifier-free training possible
```

#### Classifier Guidance
**Mathematical Formulation**:
```
Guided Score:
s_guided(x_t, t, c) = s(x_t, t) + ω∇_{x_t} log p(c | x_t)
Combine unconditional score with classifier gradient
ω controls guidance strength

Classifier Training:
Train classifier p_φ(c | x_t) on noisy images
Different noise levels → different classifiers
Or single robust classifier across noise levels

Guided Sampling:
x_{t-1} = μ_θ(x_t, t) + ω σ_t²∇_{x_t} log p_φ(c | x_t) + σ_t z_t
Additional term pulls toward desired class
Stronger guidance → better conditioning, less diversity

Mathematical Analysis:
Guidance changes effective distribution
May exit data manifold for large ω
Quality-diversity trade-off
Requires separate classifier training
```

#### Classifier-Free Guidance
**Mathematical Theory**:
```
Unconditional Score Decomposition:
s(x_t, t, c) = s(x_t, t) + [s(x_t, t, c) - s(x_t, t)]
= s(x_t, t) + score_difference

Classifier-Free Score:
s_cfg(x_t, t, c) = s(x_t, t) + ω[s(x_t, t, c) - s(x_t, t)]
= (1 + ω)s(x_t, t, c) - ωs(x_t, t)

Training Strategy:
Randomly drop conditioning with probability p
Learn both conditional and unconditional scores
Single model for both components

Mathematical Benefits:
No separate classifier needed
Better calibration than classifier guidance
Controllable guidance strength
End-to-end training
```

### Sampling Quality Analysis

#### Mathematical Metrics
**Sample Quality Measures**:
```
Frechet Inception Distance (FID):
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
where μ, Σ are mean and covariance of Inception features

Inception Score (IS):
IS = exp(E[KL(p(y|x) || p(y))])
Measures quality and diversity
Higher IS indicates better samples

LPIPS (Learned Perceptual Image Patch Similarity):
Perceptual distance between images
Based on deep network features
Better correlation with human judgment

Precision and Recall:
Precision: fraction of generated samples in real manifold
Recall: fraction of real manifold covered by generated samples
Separate quality and diversity measures
```

**Sampling Efficiency Analysis**:
```
Computational Cost:
T × (forward_pass + backward_pass) for T steps
DDIM enables T/k speedup for k < T
Quality degradation vs speedup trade-off

Sample Diversity:
Measured by pairwise distances in feature space
Stochastic sampling → higher diversity
Deterministic sampling → lower diversity
Conditioning affects diversity

Convergence Rate:
How quickly samples approach target distribution
Depends on score estimation accuracy
Number of steps affects convergence
Different algorithms have different rates

Trade-off Analysis:
Speed vs Quality: fewer steps, lower quality
Diversity vs Fidelity: stochastic vs deterministic
Memory vs Computation: caching vs recomputation
```

---

## 🎯 Advanced Understanding Questions

### Sampling Algorithm Theory:
1. **Q**: Analyze the mathematical relationship between deterministic (DDIM) and stochastic (DDPM) sampling, deriving conditions under which they produce equivalent distributions.
   **A**: Mathematical relationship: both sampling methods share same marginal distributions q(x_t|x_0) but differ in conditional distributions q(x_{t-1}|x_t,x_0). DDIM uses deterministic trajectory through probability space, DDPM uses stochastic walk. Equivalence conditions: (1) perfect score estimation, (2) infinite sampling steps, (3) appropriate noise schedule. Analysis: DDIM trajectory is deterministic integral curve, DDPM adds stochastic perturbations. Key insight: both methods sample from same target distribution but explore it differently - DDIM through single trajectory, DDPM through random walk.

2. **Q**: Develop a theoretical framework for analyzing the error accumulation in accelerated sampling methods (DDIM, higher-order integrators) and derive optimal step size selection strategies.
   **A**: Framework components: (1) local truncation error per step, (2) global error accumulation, (3) stability analysis. Mathematical analysis: Euler methods have O(h) local error, global error O(h×T) = O(1) for fixed total time. Higher-order methods: local error O(h^p), global error O(h^{p-1}). Optimal step size: balance between few large steps (fast but inaccurate) vs many small steps (slow but accurate). Error accumulation: depends on score estimation accuracy and trajectory smoothness. Strategy: adaptive step sizing based on local error estimates and curvature of probability flow ODE.

3. **Q**: Compare the mathematical foundations of different guidance mechanisms (classifier guidance, classifier-free guidance) and analyze their impact on sample quality and computational efficiency.
   **A**: Mathematical comparison: classifier guidance adds ω∇log p(c|x_t) term, classifier-free uses weighted combination of conditional/unconditional scores. Classifier guidance requires separate classifier training, classifier-free learns both in single model. Impact analysis: both improve conditioning strength but reduce diversity. Computational efficiency: classifier guidance needs additional forward pass per step, classifier-free needs two score evaluations. Quality analysis: classifier-free often better calibrated, avoids adversarial examples from imperfect classifiers. Theoretical insight: both methods modify effective sampling distribution to emphasize conditional modes.

### Advanced Sampling Methods:
4. **Q**: Analyze the mathematical principles behind higher-order numerical integration methods for diffusion sampling and their stability properties for different noise schedules.
   **A**: Mathematical principles: higher-order methods (RK4, Adams-Bashforth) achieve better accuracy by using multiple function evaluations per step. Stability analysis: larger stability regions allow bigger step sizes, critical for accelerated sampling. Noise schedule impact: steep schedules require more stable integrators, shallow schedules work with simple methods. Error analysis: O(h^p) local error for p-th order method, but constant factors matter for practical performance. Stability properties: some methods have better damping of high-frequency errors. Theoretical insight: method choice should match noise schedule characteristics and desired speed-accuracy trade-off.

5. **Q**: Develop a mathematical theory for adaptive timestep selection in diffusion sampling that optimizes computational efficiency while maintaining sample quality.
   **A**: Theory components: (1) local error estimation through embedded methods, (2) step size control algorithms, (3) quality-efficiency trade-offs. Mathematical framework: minimize total computational cost subject to error tolerance constraints. Adaptive strategy: h_new = h_old × (tol/error)^{1/(p+1)} where p is method order. Error estimation: compare predictions from different order methods. Quality maintenance: ensure global error remains below threshold. Efficiency optimization: concentrate steps where score function changes rapidly, use large steps in smooth regions. Theoretical guarantee: adaptive methods achieve desired accuracy with near-optimal computational cost.

6. **Q**: Compare the information-theoretic properties of different sampling trajectories (stochastic vs deterministic) and their impact on generation diversity and mode coverage.
   **A**: Information-theoretic analysis: stochastic sampling maintains entropy throughout process, deterministic sampling follows single trajectory with decreasing entropy. Diversity impact: stochastic enables exploration of multiple modes, deterministic produces single sample per initial condition. Mode coverage: stochastic better for multi-modal distributions, deterministic may miss modes. Mathematical framework: entropy evolution H(x_t) during sampling, mutual information I(x_0; x_T) preservation. Trade-offs: diversity vs speed, exploration vs exploitation. Theoretical insight: choice depends on application needs - stochastic for diverse generation, deterministic for consistent reproduction.

### Conditional Sampling Theory:
7. **Q**: Design a mathematical framework for analyzing the trade-offs between guidance strength and sample diversity in conditional diffusion models.
   **A**: Framework components: (1) guidance strength parameter ω, (2) sample diversity metrics, (3) conditioning fidelity measures. Mathematical analysis: stronger guidance improves conditioning but reduces diversity through distribution narrowing. Trade-off curve: plot diversity vs conditioning strength, find optimal operating point. Diversity measures: pairwise distances in feature space, entropy of generated distribution. Conditioning fidelity: classification accuracy, text-image alignment scores. Theoretical model: guidance modifies effective temperature of distribution, ω controls temperature. Optimal guidance: maximize conditioning subject to diversity constraints, or vice versa depending on application priorities.

8. **Q**: Develop a unified mathematical theory connecting different sampling acceleration techniques and identifying fundamental limits on sampling speed vs quality trade-offs.
   **A**: Unified theory: all acceleration methods reduce sampling steps through better numerical integration or trajectory optimization. Fundamental limits: governed by condition number of reverse process, smoothness of score function, approximation quality. Mathematical framework: sample quality bounded by approximation error + discretization error + numerical error. Speed limits: minimum steps required depends on noise schedule and score accuracy. Trade-off analysis: pareto frontier between speed and quality, no method can dominate on both simultaneously. Theoretical bounds: under smoothness assumptions, quality degrades as O(1/√steps) for optimal methods. Key insight: acceleration success depends on exploiting structure in specific problem instances.

---

## 🔑 Key Sampling Techniques Principles

1. **Deterministic vs Stochastic Duality**: DDIM and DDPM represent complementary approaches to sampling - deterministic for speed and consistency, stochastic for diversity and robustness.

2. **Acceleration Through Structure**: Faster sampling exploits the smoothness of probability flow ODEs and approximate integrability of the reverse diffusion process.

3. **Guidance Mechanisms**: Both classifier and classifier-free guidance work by modifying the effective score function to emphasize conditional modes while reducing sample diversity.

4. **Quality-Speed Trade-offs**: Fundamental trade-offs exist between sampling speed, sample quality, and generation diversity, with optimal choices depending on application requirements.

5. **Numerical Integration Theory**: Advanced ODE solvers and adaptive methods can significantly improve sampling efficiency while maintaining mathematical rigor and error control.

---

**Next**: Continue with Day 7 - Advanced Architectures Theory