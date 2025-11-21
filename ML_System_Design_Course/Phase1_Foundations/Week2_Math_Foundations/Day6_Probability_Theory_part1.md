# Day 6 (Part 1): Advanced Probability & Bayesian Inference

> **Phase**: 6 - Deep Dive
> **Topic**: Beyond Coin Flips
> **Focus**: MCMC, Variational Inference, and Conjugate Priors
> **Reading Time**: 60 mins

---

## 1. Bayesian Inference at Scale

Calculating the posterior $P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$ is hard because the denominator (Evidence) involves an integral that is often intractable.

### 1.1 Markov Chain Monte Carlo (MCMC)
*   **Idea**: Construct a Markov Chain whose stationary distribution is the target posterior. Wander around the parameter space collecting samples.
*   **Metropolis-Hastings**: The classic algorithm. Propose a jump. Accept/Reject based on probability ratio.
*   **NUTS (No-U-Turn Sampler)**: Used in PyMC3/Stan. Uses gradients (Hamiltonian Monte Carlo) to explore efficient paths, avoiding random walks.

### 1.2 Variational Inference (VI)
*   **Idea**: Turn integration into optimization.
*   **Method**: Propose a simple family of distributions $Q(\theta)$ (e.g., Gaussian). Minimize the KL Divergence between $Q(\theta)$ and the true posterior $P(\theta|D)$.
*   **Pros**: Much faster than MCMC. Scales to large datasets (Stochastic VI).
*   **Cons**: Biased (underestimates variance).

### 1.3 Conjugate Priors
*   **Magic**: If Prior is Beta and Likelihood is Binomial, Posterior is Beta. No integration needed!
*   **Table**:
    *   Beta -> Binomial
    *   Dirichlet -> Multinomial
    *   Gamma -> Poisson
    *   Normal -> Normal (mean)

---

## 2. Advanced Distributions

### 2.1 Dirichlet Distribution
*   **Definition**: A distribution over distributions.
*   **Use Case**: Latent Dirichlet Allocation (LDA) for topic modeling. Modeling uncertainty in classification probabilities.

### 2.2 Poisson Process
*   **Definition**: Modeling events occurring continuously and independently at a constant average rate.
*   **Use Case**: Modeling arrival of requests to a server.

---

## 3. Tricky Interview Questions

### Q1: What is the "Reparameterization Trick" in VAEs?
> **Answer**: We need to differentiate through a random sampling node $z \sim N(\mu, \sigma)$. We can't backprop through randomness.
> *   **Trick**: Rewrite $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim N(0, 1)$.
> *   Now randomness is external ($\epsilon$). We can compute gradients w.r.t $\mu$ and $\sigma$.

### Q2: Explain the difference between MLE and MAP.
> **Answer**:
> *   **MLE (Maximum Likelihood)**: $\text{argmax}_\theta P(D|\theta)$. Assumes uniform prior. Prone to overfitting on small data.
> *   **MAP (Maximum A Posteriori)**: $\text{argmax}_\theta P(D|\theta)P(\theta)$. Includes prior. Acts as regularization (e.g., L2 regularization corresponds to a Gaussian Prior).

### Q3: Why does MCMC take "Burn-in" samples?
> **Answer**: The chain starts at a random point (maybe low probability). It takes time to converge to the stationary distribution (the high probability region). Samples collected during this warm-up phase are biased and should be discarded.

---

## 4. Practical Edge Case: Numerical Underflow
*   **Problem**: Multiplying thousands of probabilities ($0.5^{1000}$) results in 0.
*   **Solution**: Always work in Log-Space. $\log(A \cdot B) = \log A + \log B$. Summing logs is stable.

