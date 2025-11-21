# Day 27 Interview Questions: SAC

## Q1: What is Maximum Entropy RL?
**Answer:**
Maximum Entropy RL modifies the objective to maximize both reward and entropy:
$$ J = \mathbb{E}\left[\sum (r_t + \alpha H(\pi(\cdot|s_t)))\right] $$
where $H(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ is the entropy.
This encourages the policy to:
1. **Explore:** High entropy means diverse actions.
2. **Be robust:** Learn multiple ways to solve the task.
3. **Avoid premature convergence:** Stay stochastic longer.

## Q2: How does SAC differ from DDPG and TD3?
**Answer:**
*   **DDPG/TD3:** Deterministic policies $a = \mu(s)$. Add noise for exploration.
*   **SAC:** Stochastic policy $\pi(a|s)$ (Gaussian). Exploration via entropy maximization.
*   SAC automatically balances exploration/exploitation via the temperature $\alpha$, making it more sample-efficient and robust.

## Q3: What is the Reparameterization Trick?
**Answer:**
To backpropagate through a stochastic policy, we need gradients w.r.t. the policy parameters.
Standard sampling $a \sim \mathcal{N}(\mu, \sigma)$ is not differentiable.
**Reparameterization:**
$$ a = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
Now $a$ is a deterministic function of $\mu, \sigma, \epsilon$, and we can differentiate through $\mu$ and $\sigma$.

## Q4: How does SAC automatically tune the temperature $\alpha$?
**Answer:**
SAC treats $\alpha$ as a learnable parameter trained to match a target entropy:
$$ \max_\alpha \mathbb{E}[\alpha (H(\pi) - H_{target})] $$
*   If $H(\pi) > H_{target}$: Reduce $\alpha$ to encourage less exploration.
*   If $H(\pi) < H_{target}$: Increase $\alpha$ to encourage more exploration.
*   **Target Entropy:** Typically $H_{target} = -\dim(\mathcal{A})$.
This automates the exploration schedule without manual tuning.

## Q5: Why does SAC use two critics?
**Answer:**
Like TD3, SAC uses **Clipped Double Q-Learning** to reduce overestimation bias:
$$ Q_{target} = r + \gamma (\min(Q'_1(s', a'), Q'_2(s', a')) - \alpha \log \pi(a'|s')) $$
The minimum of two independent estimates provides a pessimistic (conservative) target, preventing the Q-function from overestimating values.

## Q6: What is the squashing function and why is it needed?
**Answer:**
Actions are sampled from an unbounded Gaussian $\mathcal{N}(\mu, \sigma)$, but environments often require bounded actions (e.g., $[-1, 1]$).
SAC applies $\tanh$ to squash:
$$ a = \tanh(\mu + \sigma \epsilon) $$
**Important:** The log-probability must be corrected for this transformation:
$$ \log \pi(a|s) = \log \mathcal{N}(a'|\mu, \sigma) - \sum \log(1 - \tanh^2(a'_i)) $$
This ensures the entropy calculation is correct.
