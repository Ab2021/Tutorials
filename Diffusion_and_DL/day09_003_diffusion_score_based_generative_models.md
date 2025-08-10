# Day 9 - Part 3: Diffusion Models and Score-Based Generative Modeling Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of diffusion processes and stochastic differential equations
- Score-based generative modeling theory and Langevin dynamics
- Denoising diffusion probabilistic models (DDPM) mathematical framework
- Variance preserving and variance exploding diffusion processes
- Advanced sampling techniques: DDIM, DPM-Solver, and classifier guidance
- Theoretical connections to energy-based models and optimal transport

---

## 🌊 Stochastic Differential Equations and Diffusion Theory

### Mathematical Foundation of Diffusion Processes

#### Forward Diffusion Process
**Stochastic Differential Equation (SDE) Framework**:
```
Forward SDE:
dx = f(x,t)dt + g(t)dw

Where:
- x(t): data at time t
- f(x,t): drift coefficient
- g(t): diffusion coefficient  
- w: Wiener process (Brownian motion)
- t ∈ [0,T]: time index

Variance Preserving (VP) SDE:
dx = -½β(t)x dt + √β(t) dw
β(t): noise schedule

Variance Exploding (VE) SDE:
dx = √(dσ²(t)/dt) dw
σ(t): noise scale schedule
```

**Solution Properties**:
```
Probability Density Evolution:
∂p_t/∂t = -∇·(f(x,t)p_t) + ½g²(t)∇²p_t (Fokker-Planck equation)

For VP SDE:
q(x_t|x_0) = N(x_t; √α̃_t x_0, (1-α̃_t)I)
Where α̃_t = exp(-½∫₀ᵗ β(s)ds)

For VE SDE:
q(x_t|x_0) = N(x_t; x_0, σ²(t)I)

Mathematical Properties:
- Markov property: future depends only on present
- Gaussian transitions for linear SDEs
- Continuous-time generalization of discrete diffusion
- Analytical tractability for certain forms
```

#### Reverse-Time SDE Theory
**Time Reversal Theorem**:
```
Reverse SDE:
dx = [f(x,t) - g²(t)∇_x log p_t(x)]dt + g(t)dw̄

Where:
- ∇_x log p_t(x): score function
- w̄: reverse-time Wiener process

For VP SDE:
dx = [-½β(t)x - β(t)∇_x log p_t(x)]dt + √β(t)dw̄

For VE SDE:
dx = -g²(t)∇_x log p_t(x)dt + g(t)dw̄

Key Insight:
Reverse process requires score function ∇_x log p_t(x)
Score estimation enables generative modeling
Forward process destroys structure, reverse process recreates it
```

**Probability Flow ODE**:
```
Deterministic Sampling:
dx = [f(x,t) - ½g²(t)∇_x log p_t(x)]dt

Properties:
- Same marginal distributions as reverse SDE
- Deterministic trajectories
- Enables fast sampling
- Connection to normalizing flows

Mathematical Benefits:
- Exact likelihood computation possible
- Stable numerical integration
- Controllable sampling process
- Theoretical connection to optimal transport
```

### Score Function Theory

#### Score Matching Framework
**Fisher Divergence Minimization**:
```
Score Function:
s(x) = ∇_x log p(x)

Fisher Divergence:
J(s) = ½E_p[||s(x) - ∇_x log p(x)||²]

Score Matching Objective:
min_θ E_p[||s_θ(x) - ∇_x log p(x)||²]

Challenge:
True score ∇_x log p(x) unknown
Need alternative formulation for training
```

**Denoising Score Matching**:
```
Noisy Data Distribution:
q_σ(x̃|x) = N(x̃; x, σ²I)
q_σ(x̃) = ∫ q_σ(x̃|x)p(x)dx

Denoising Objective:
min_θ E_x~p E_x̃~q_σ(·|x)[||s_θ(x̃) - ∇_x̃ log q_σ(x̃|x)||²]

Explicit Score:
∇_x̃ log q_σ(x̃|x) = -(x̃ - x)/σ²

Practical Objective:
min_θ E_x~p E_ε~N(0,I)[||s_θ(x + σε) + ε/σ||²]

Benefits:
- Tractable training objective
- No need for true score
- Works with any noise level σ
- Theoretical equivalence to score matching
```

#### Annealed Langevin Dynamics
**Multi-Scale Score Estimation**:
```
Noise Schedule:
σ₁ > σ₂ > ... > σ_L, where σ_L ≈ 0
Train score model for each noise level

Annealed Sampling:
For l = 1 to L:
    For i = 1 to T_l:
        x ← x + α_l s_θ(x, σ_l) + √(2α_l) z
    Where z ~ N(0,I)

Mathematical Justification:
Large σ: covers entire data distribution
Small σ: refines samples near data manifold
Annealing escapes local minima
Convergence to data distribution
```

**Langevin MCMC Theory**:
```
Langevin Dynamics:
dx = ∇_x log p(x)dt + √2 dw

Discrete Approximation:
x_{k+1} = x_k + α∇_x log p(x_k) + √(2α) z_k

Where α is step size, z_k ~ N(0,I)

Convergence Analysis:
Under regularity conditions:
- Chain converges to stationary distribution p(x)
- Convergence rate O(1/α) + O(α)
- Optimal step size α* balances discretization vs mixing
- Score errors affect convergence rate
```

---

## 🎯 Denoising Diffusion Probabilistic Models

### DDPM Mathematical Framework

#### Forward Process Definition
**Discrete Diffusion Chain**:
```
Forward Process:
q(x₁,...,x_T|x₀) = ∏ᵢ₌₁ᵀ q(xᵢ|xᵢ₋₁)

Gaussian Transition:
q(xᵢ|xᵢ₋₁) = N(xᵢ; √(1-βᵢ)xᵢ₋₁, βᵢI)

Where βᵢ ∈ (0,1) is noise schedule

Closed Form:
q(xᵢ|x₀) = N(xᵢ; √ᾱᵢx₀, (1-ᾱᵢ)I)
Where ᾱᵢ = ∏ⱼ₌₁ⁱ(1-βⱼ)

Reparameterization:
xᵢ = √ᾱᵢ x₀ + √(1-ᾱᵢ) ε
Where ε ~ N(0,I)
```

#### Reverse Process and Training
**Reverse Process Approximation**:
```
Approximate Reverse:
p_θ(x₀,...,x_{T-1}|x_T) = p(x_T) ∏ᵢ₌₁ᵀ p_θ(xᵢ₋₁|xᵢ)

Gaussian Reverse Transition:
p_θ(xᵢ₋₁|xᵢ) = N(xᵢ₋₁; μ_θ(xᵢ,i), Σ_θ(xᵢ,i))

Theoretical Justification:
For small βᵢ, reverse transitions are approximately Gaussian
Optimal reverse mean μ* computable in closed form
Neural network approximates this optimal reverse process
```

**Variational Bound**:
```
DDPM ELBO:
L = E_q[-log p_θ(x₀|x₁)] + ∑ᵢ₌₂ᵀ E_q[KL(q(xᵢ₋₁|xᵢ,x₀) || p_θ(xᵢ₋₁|xᵢ))] + KL(q(x_T|x₀) || p(x_T))

Simplified Objective:
L_simple = E_t,x₀,ε[||ε - ε_θ(√ᾱₜ x₀ + √(1-ᾱₜ) ε, t)||²]

Where:
- t ~ Uniform(1,T)
- ε ~ N(0,I)
- ε_θ: neural network predicting noise

Mathematical Equivalence:
L_simple corresponds to denoising score matching
Score s_θ(x,t) = -ε_θ(x,t)/√(1-ᾱₜ)
Simplified objective easier to optimize
```

#### Noise Schedule Theory
**β Schedule Design**:
```
Linear Schedule:
βᵢ = β₁ + (βₜ - β₁)(i-1)/(T-1)
Simple but may not be optimal

Cosine Schedule:
ᾱₜ = cos²(π/2 · (t/T + s)/(1 + s))
Where s is small offset
Better preservation of structure

Mathematical Analysis:
Schedule affects:
- Training dynamics
- Sample quality  
- Number of steps needed
- Numerical stability

Optimal Schedule:
Depends on data distribution and model capacity
Fast schedules: fewer steps, lower quality
Slow schedules: more steps, higher quality
```

### Advanced Diffusion Formulations

#### Variance Preserving vs Variance Exploding
**VP Diffusion Mathematical Properties**:
```
VP SDE: dx = -½β(t)x dt + √β(t) dw

Properties:
- Preserves signal-to-noise ratio structure
- ᾱₜ controls signal preservation
- Natural for image data
- Bounded variance at all times

Score Parametrization:
ε_θ(x,t) predicts noise
s_θ(x,t) = -ε_θ(x,t)/√(1-ᾱₜ)

Mathematical Benefits:
- Stable training dynamics
- Predictable noise levels
- Good empirical performance
- Theoretical guarantees
```

**VE Diffusion Analysis**:
```
VE SDE: dx = √(dσ²(t)/dt) dw

Properties:
- Variance grows without bound
- Signal preserved exactly at t=0
- Different noise paradigm
- Unbounded variance as t→∞

Score Parametrization:
s_θ(x,t) = σ(t)ε_θ(x,t)
Direct score prediction

Theoretical Differences:
- Different limiting behavior
- Alternative training dynamics
- May suit different data types
- Unified framework possible
```

#### Continuous-Time Formulation
**SDE-Based Generative Models**:
```
General SDE Framework:
dx = f(x,t)dt + g(t)dw

Score-Based Model:
ε_θ(x,t) approximates score function
s_θ(x,t) = ∇_x log p_t(x)

Reverse SDE:
dx = [f(x,t) - g²(t)s_θ(x,t)]dt + g(t)dw̄

Probability Flow ODE:
dx = [f(x,t) - ½g²(t)s_θ(x,t)]dt

Benefits:
- Continuous-time framework
- Flexible noise schedules
- Theoretical rigor
- Multiple sampling strategies
```

---

## 🚀 Advanced Sampling and Acceleration

### DDIM and Deterministic Sampling

#### Non-Markovian Reverse Process
**DDIM Mathematical Framework**:
```
DDIM Forward Process:
q_σ(x_{1:T}|x_0) = q_σ(x_T|x_0) ∏ᵢ₌₂ᵀ q_σ(xᵢ₋₁|xᵢ,x₀)

Where:
q_σ(xᵢ₋₁|xᵢ,x₀) = N(xᵢ₋₁; √ᾱᵢ₋₁x₀ + √(1-ᾱᵢ₋₁-σᵢ²)·(xᵢ-√ᾱᵢx₀)/√(1-ᾱᵢ), σᵢ²I)

Properties:
- Same marginals q(xᵢ|x₀) as DDPM
- Different joint distribution q(x₁:T|x₀)
- σᵢ controls stochasticity
- σᵢ = 0: deterministic process
```

**Deterministic Sampling Formula**:
```
DDIM Update:
x_{i-1} = √ᾱᵢ₋₁ · x₀ + √(1-ᾱᵢ₋₁) · ε_θ(xᵢ,i)

Where x₀ predicted from:
x₀ = (xᵢ - √(1-ᾱᵢ)ε_θ(xᵢ,i))/√ᾱᵢ

Mathematical Properties:
- Deterministic mapping from noise to sample
- Allows fast sampling with fewer steps
- Invertible process enables interpolation
- Connection to probability flow ODE

Acceleration Benefits:
- 10-50× fewer sampling steps
- Maintains sample quality
- Enables latent space interpolation
- Faster inference for applications
```

#### DPM-Solver Theory
**High-Order ODE Solvers**:
```
Probability Flow ODE:
dx/dt = -½g²(t)s_θ(x,t)

DPM-Solver Approach:
Use high-order numerical methods
Leverage structure of diffusion ODE
Adaptive step size control

Mathematical Framework:
Taylor expansion around solution
Higher-order approximation
O(h^k) local error for k-th order method

Benefits:
- Fewer function evaluations
- Better numerical stability
- Adaptive integration
- Theoretical convergence guarantees
```

**Exponential Integrator Methods**:
```
Linear Part Separation:
dx/dt = -λ(t)x + g(x,t)

Where λ(t) corresponds to linear noise
g(x,t) contains nonlinear score term

Exponential Integrator:
Exact integration of linear part
Approximation of nonlinear part
Better stability properties

Mathematical Advantages:
- Handles stiff equations better
- Preserves exponential decay
- Higher accuracy for diffusion ODEs
- Theoretical error bounds
```

### Classifier Guidance and Conditioning

#### Mathematical Theory of Guidance
**Conditional Score Functions**:
```
Conditional Generation:
p(x|y) ∝ p(x)p(y|x)

Conditional Score:
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)

Classifier Guidance:
s_guided(x,t,y) = s_θ(x,t) + γ∇_x log p_φ(y|x_t)

Where:
- s_θ: unconditional score model
- p_φ: classifier
- γ: guidance strength
- x_t: noisy sample at time t
```

**Classifier-Free Guidance**:
```
Joint Training:
Train model on both conditional and unconditional data
p_θ(x|y) and p_θ(x) with same parameters

Guidance Formula:
s_guided(x,t,y) = s_θ(x,t) + γ(s_θ(x,t,y) - s_θ(x,t))
                = (1+γ)s_θ(x,t,y) - γs_θ(x,t)

Mathematical Properties:
- No separate classifier needed
- Controllable guidance strength
- Better sample-condition alignment
- Theoretical connection to importance sampling

Benefits:
- Higher quality conditional samples
- Better text-image alignment
- No classifier training required
- More stable than classifier guidance
```

#### Information-Theoretic Analysis
**Mutual Information in Guidance**:
```
Guidance Effect:
Classifier guidance increases I(X;Y)
Trade-off with sample diversity
Higher γ → better conditioning, lower diversity

Mathematical Framework:
Guidance modifies sampling distribution
p_guided(x|y) ∝ p(x|y)^(1+γ)
Sharpens conditional distribution

Rate-Distortion Perspective:
Guidance controls rate-distortion trade-off
Higher guidance → lower entropy, better fidelity
Optimal γ depends on application requirements
```

---

## 🔄 Connections to Other Generative Models

### Score-Based Models and Energy Functions

#### Energy-Based Model Connection
**Score-Energy Relationship**:
```
Energy Function:
p(x) = exp(-E(x))/Z
Where Z is partition function

Score Function:
∇_x log p(x) = -∇_x E(x)

Score Model as EBM:
Training score model ≡ learning energy function
E_θ(x) = -∫ s_θ(x)^T dx + constant

Benefits:
- Unified theoretical framework
- Energy interpretation of score
- Connection to physics
- Principled optimization landscape
```

**Contrastive Divergence Connection**:
```
CD Objective:
L_CD = E_data[E_θ(x)] - E_model[E_θ(x)]

Score Matching Connection:
Minimizing Fisher divergence ≡ maximizing likelihood
Score matching avoids partition function
More stable than CD training

Mathematical Equivalence:
Both approaches learn energy function
Different training objectives
Score matching more tractable
```

### Optimal Transport Theory

#### Probability Flow and Transport
**Optimal Transport Framework**:
```
Transport Problem:
min_T ∫ c(x,T(x)) p_data(x) dx
Subject to: T#p_data = p_noise

Where T# denotes pushforward measure

Diffusion Connection:
Probability flow ODE implements transport
Continuous transport path
Optimal under quadratic cost c(x,y) = ||x-y||²/2

Mathematical Properties:
- Continuous transport map
- Minimal transport cost
- Straight-line paths in probability space
- Connection to Wasserstein distance
```

**Schrödinger Bridge Formulation**:
```
Schrödinger Bridge:
Find stochastic process minimizing relative entropy
Subject to boundary conditions
p_0 = p_data, p_T = p_noise

Entropic Optimal Transport:
Regularized transport problem
Adds entropy regularization term
Solution is diffusion process

Mathematical Result:
Schrödinger bridge = optimal diffusion
Connects transport and diffusion theory
Provides principled framework
```

### Flow-Based Model Connections

#### Continuous Normalizing Flows
**Neural ODE Framework**:
```
Continuous Flow:
dx/dt = f_θ(x,t)

Change of Variables:
log p_T(x_T) = log p_0(x_0) - ∫₀ᵀ Tr(∇_x f_θ(x,t)) dt

Diffusion as Flow:
Probability flow ODE is normalizing flow
Deterministic continuous transformation
Exact likelihood computation possible

Mathematical Connection:
Both define continuous transformations
Diffusion: stochastic → deterministic
Flow: deterministic invertible maps
Complementary approaches
```

**Rectified Flows**:
```
Straight-Line Flows:
Transport along straight lines in data space
Minimize transport cost directly
Connection to optimal transport

Mathematical Formulation:
x_t = (1-t)x_0 + tx_1
Where x_0 ~ p_data, x_1 ~ p_noise

Benefits:
- Simpler transport paths
- Faster sampling
- Better numerical properties
- Theoretical optimality
```

---

## 🎯 Advanced Understanding Questions

### Diffusion Process Theory:
1. **Q**: Analyze the mathematical relationship between forward diffusion noise schedules and reverse process approximation quality, deriving optimal schedule design principles.
   **A**: Forward noise schedule β(t) affects reverse process approximation through signal-to-noise evolution. Mathematical analysis: smaller β(t) → better reverse Gaussian approximation but requires more timesteps. Optimal schedule balances approximation error vs computational cost. Key insights: (1) schedule affects local approximation quality, (2) cosine schedules preserve low-frequency information better, (3) adaptive schedules can optimize for specific metrics. Theoretical framework: approximation error bounds depend on schedule smoothness and noise level progression.

2. **Q**: Compare variance preserving (VP) and variance exploding (VE) diffusion formulations and analyze their theoretical advantages for different types of data distributions.
   **A**: VP preserves signal variance through scaling: Var(x_t) = constant, while VE grows variance: Var(x_t) → ∞. Mathematical differences: VP maintains bounded dynamics, VE allows unbounded growth. Theoretical advantages: VP better for bounded data (images), VE better for unbounded data (point clouds). Key insight: choice affects training stability, sampling quality, and theoretical guarantees. Unified framework possible through appropriate parameterization.

3. **Q**: Develop a theoretical framework connecting score-based models to energy-based models and analyze the implications for training and sampling.
   **A**: Framework: score function s(x) = -∇E(x) where E(x) is energy function. Training score model equivalent to learning energy landscape. Implications: (1) score matching avoids partition function computation, (2) sampling via Langevin dynamics, (3) energy interpretation provides geometric intuition. Theoretical benefits: score matching more stable than contrastive divergence, direct optimization of Fisher divergence, principled sampling through gradient flow.

### Advanced Sampling Theory:
4. **Q**: Analyze the mathematical foundations of DDIM and derive conditions under which deterministic sampling preserves generation quality while reducing computational cost.
   **A**: DDIM modifies reverse process while preserving marginal distributions q(x_t|x_0). Mathematical analysis: deterministic limit (σ→0) creates invertible mapping, enables fast sampling. Quality preservation conditions: (1) sufficient model capacity, (2) appropriate timestep selection, (3) proper noise prediction accuracy. Theoretical guarantee: DDIM converges to same distribution as DDPM under perfect score estimation. Computational benefit: O(10-50×) speedup with minimal quality loss.

5. **Q**: Compare different high-order ODE solvers for probability flow ODEs and analyze their numerical stability and convergence properties for diffusion models.
   **A**: Comparison of Euler, Heun, DPM-Solver methods for dx/dt = -½g²(t)s_θ(x,t). Mathematical analysis: higher-order methods achieve O(h^k) local error vs O(h) for Euler. Stability analysis: stiffness from large g²(t) requires special treatment. Convergence properties: exponential integrators handle linear part exactly, better for diffusion structure. Key insight: method choice depends on noise schedule, model accuracy, and computational budget.

6. **Q**: Develop a mathematical analysis of classifier guidance and classifier-free guidance, comparing their theoretical properties and practical trade-offs.
   **A**: Classifier guidance: s_guided = s_θ + γ∇log p_φ(y|x), requires separate classifier. Classifier-free: s_guided = (1+γ)s_θ(x,y) - γs_θ(x), single model. Mathematical comparison: both modify score function to increase I(X;Y). Theoretical properties: classifier-free more stable (no classifier training), better aligned with generation process. Trade-offs: classifier-free requires joint training, classifier guidance more flexible for different conditions.

### Model Connections:
7. **Q**: Analyze the theoretical connections between diffusion models, normalizing flows, and optimal transport, developing a unified mathematical framework.
   **A**: Unified framework through continuous transport perspective. Mathematical connections: (1) probability flow ODE is normalizing flow, (2) diffusion implements Schrödinger bridge, (3) optimal transport provides geometric foundation. Key insights: all methods transport between distributions with different trade-offs. Framework components: continuous-time dynamics, transport cost minimization, entropy regularization. Theoretical unification enables hybrid methods combining advantages of each approach.

8. **Q**: Design a comprehensive theoretical analysis of diffusion model training dynamics and derive conditions for stable convergence and optimal sample quality.
   **A**: Training dynamics analysis through score function approximation error and its propagation. Mathematical framework: decompose total error into approximation error, optimization error, and discretization error. Stability conditions: (1) appropriate learning rate scheduling, (2) noise schedule design, (3) model capacity matching data complexity. Convergence theory: under Lipschitz conditions and bounded approximation error, training converges to optimal score function. Optimal quality conditions: sufficient model capacity, appropriate training duration, proper regularization.

---

## 🔑 Key Diffusion Model Principles

1. **SDE Foundation**: Diffusion models are based on stochastic differential equations that define continuous-time processes for data corruption and restoration.

2. **Score Function Learning**: The core insight is learning score functions ∇log p(x) through denoising, enabling tractable generative modeling without partition functions.

3. **Reverse Process Design**: Theoretical guarantee that reverse-time SDE generates samples from data distribution, with practical approximation through neural networks.

4. **Sampling Flexibility**: Multiple sampling strategies (stochastic SDE, deterministic ODE, accelerated methods) provide trade-offs between quality and speed.

5. **Guidance Mechanisms**: Classifier and classifier-free guidance enable controllable generation while maintaining theoretical foundations through score modification.

---

**Next**: Continue with Day 9 - Part 4: Advanced Generative Architectures and Energy-Based Models