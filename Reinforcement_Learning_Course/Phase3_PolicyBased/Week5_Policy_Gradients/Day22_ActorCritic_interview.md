# Day 22 Interview Questions: Actor-Critic

## Q1: What are the roles of the Actor and the Critic?
**Answer:**
*   **Actor:** Learns the policy $\pi_\theta(a|s)$. It decides which action to take.
*   **Critic:** Learns the value function $V_\phi(s)$ or $Q_\phi(s, a)$. It evaluates how good the Actor's choices are.
*   The Critic provides feedback (the Advantage) to guide the Actor's updates.

## Q2: Why is Actor-Critic better than pure Policy Gradient (REINFORCE)?
**Answer:**
REINFORCE uses the full Monte Carlo return $G_t$, which has high variance because it depends on the entire trajectory.
Actor-Critic uses the TD error (bootstrapping) as the advantage:
$$ A(s, a) = r + \gamma V(s') - V(s) $$
This reduces variance significantly because it only depends on one step of reward, not the entire episode. The Critic's estimate of $V(s')$ replaces the noisy future return.

## Q3: What is the difference between A2C and A3C?
**Answer:**
*   **A3C (Asynchronous):** Multiple agents update a shared global network asynchronously. Each agent computes gradients and applies them immediately without waiting for others.
*   **A2C (Synchronous):** Multiple agents collect experience in parallel, but gradient updates are synchronized (wait for all workers before updating).
*   A2C is more stable and GPU-efficient. A3C can be faster on CPUs but is noisier.

## Q4: What is Generalized Advantage Estimation (GAE)?
**Answer:**
GAE is a way to compute the advantage that balances bias and variance by mixing n-step returns.
$$ A^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l} $$
where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
*   $\lambda = 0$: 1-step TD (low variance, high bias).
*   $\lambda = 1$: Monte Carlo (high variance, low bias).
*   Typically $\lambda \approx 0.95$ for a good tradeoff.

## Q5: Why do we use Entropy Regularization?
**Answer:**
Without entropy regularization, the policy can converge prematurely to a deterministic policy (all probability on one action), which stops exploration.
Adding entropy $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ to the loss encourages the policy to remain stochastic, maintaining exploration.
The coefficient is usually small (e.g., 0.01) and often annealed over time.

## Q6: What loss functions are used in Actor-Critic?
**Answer:**
*   **Critic Loss:** Mean Squared Error between the predicted value and the target (TD target).
    $$ L_{critic} = (V_\phi(s) - (r + \gamma V_\phi(s')))^2 $$
*   **Actor Loss:** Policy Gradient weighted by the advantage (detached from Critic gradients).
    $$ L_{actor} = -\log \pi_\theta(a|s) \cdot A(s, a).detach() $$
*   **Entropy Loss:** Entropy of the policy.
    $$ L_{entropy} = -H(\pi) $$
*   Total: $L = L_{actor} + c_1 L_{critic} + c_2 L_{entropy}$
