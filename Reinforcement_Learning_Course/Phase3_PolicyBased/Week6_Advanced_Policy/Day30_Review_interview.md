# Day 30 Interview Questions: Phase 3 Review

## Q1: What are the main advantages of Policy Gradients over Value-Based methods?
**Answer:**
1. **Continuous Actions:** PG can handle continuous action spaces naturally (output Gaussian). Value-based methods struggle (need to discretize or use NAF).
2. **Stochastic Policies:** PG learns stochastic policies, which can be optimal in partially observable environments.
3. **Better Convergence:** PG can converge to a locally optimal policy, while value-based methods can oscillate.
4. **Direct Optimization:** PG directly optimizes the objective (expected return), not a proxy (value function).

## Q2: Why is PPO so popular?
**Answer:**
*   **Simple:** First-order, easy to implement.
*   **Stable:** Clipped objective prevents destructive updates.
*   **Sample-Efficient:** Can reuse data for multiple epochs.
*   **General:** Works on discrete and continuous actions, Atari to robotics.
*   **Robust:** Less sensitive to hyperparameters than TRPO or A2C.

## Q3: When would you use DDPG/TD3 vs. SAC?
**Answer:**
*   **TD3:** When you need a deterministic policy or want simplicity. Uses added noise for exploration.
*   **SAC:** When sample efficiency and robustness are critical. Stochastic policy with entropy maximization for automatic exploration.
*   **Practical:** SAC is the modern default for continuous control, but TD3 is simpler and faster.

## Q4: What is the credit assignment problem in MARL?
**Answer:**
In cooperative multi-agent RL, all agents receive the same global reward.
**Problem:** Which agent's actions contributed to the reward?
**Solutions:**
*   **Value Decomposition (VDN/QMIX):** Decompose global Q into individual agent Q-values.
*   **Counterfactual Baselines:** Compare reward with/without agent $i$.
*   **Centralized Critic:** Use global information during training (CTDE).

## Q5: What is the key idea behind Meta-RL?
**Answer:**
**Learning to Learn:** Train on a distribution of tasks to enable fast adaptation to new tasks.
**MAML:** Learn an initialization that can be quickly fine-tuned.
**RL²:** Use an RNN to encode the task implicitly.
**Goal:** Few-shot learning—adapt to a new task with minimal data.

## Q6: How do you diagnose if your policy gradient is not learning?
**Answer:**
1. **Check Rewards:** Are they improving? If flat, check environment setup.
2. **Check Value Function:** Is the critic learning? Low explained variance suggests issues.
3. **Check KL Divergence:** Is it too large (unstable updates) or too small (no learning)?
4. **Check Entropy:** Is it decreasing over time? If stuck at high entropy, the policy isn't converging.
5. **Check Gradients:** Are they vanishing (too small) or exploding (too large)?
6. **Simplify:** Test on a simpler environment first (CartPole).

## Q7: What are the three tricks in TD3?
**Answer:**
1. **Clipped Double Q-Learning:** Use two critics, take the minimum for the target to reduce overestimation.
2. **Delayed Policy Updates:** Update the actor less frequently than the critics (e.g., every 2 updates).
3. **Target Policy Smoothing:** Add clipped noise to the target action to make Q-values robust.

## Q8: What is GAE and why is it useful?
**Answer:**
**Generalized Advantage Estimation** interpolates between 1-step TD (low variance, high bias) and Monte Carlo (high variance, low bias):
$$ A^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l} $$
*   $\lambda = 0$: Pure TD.
*   $\lambda = 1$: Pure MC.
*   Typical: $\lambda = 0.95$ for a good bias-variance tradeoff.
GAE significantly reduces variance in policy gradients, leading to faster and more stable learning.
