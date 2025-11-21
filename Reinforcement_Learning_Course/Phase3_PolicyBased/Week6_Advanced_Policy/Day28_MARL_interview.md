# Day 28 Interview Questions: Multi-Agent RL

## Q1: What makes Multi-Agent RL harder than single-agent RL?
**Answer:**
1. **Non-Stationarity:** From each agent's perspective, the environment is changing because other agents are also learning.
2. **Credit Assignment:** In cooperative settings, it's hard to determine which agent's actions led to success or failure.
3. **Scalability:** The joint action space grows exponentially with the number of agents.
4. **Coordination:** Agents need to coordinate their actions, which requires communication or implicit understanding.

## Q2: What is CTDE (Centralized Training, Decentralized Execution)?
**Answer:**
CTDE is a paradigm in cooperative MARL:
*   **Centralized Training:** During training, use global state information and all agents' actions to learn. This allows faster, more stable learning.
*   **Decentralized Execution:** During execution, each agent uses only its local observations to act. This is more practical due to communication constraints.
Examples: QMIX, MADDPG.

## Q3: How does QMIX improve upon VDN?
**Answer:**
*   **VDN:** Decomposes $Q_{total} = \sum Q_i$. Too restrictive (only additive).
*   **QMIX:** Uses a **mixing network** with non-negative weights to combine $Q_i$ values.
    $$ Q_{total} = f_{mix}(Q_1, ..., Q_n; s) $$
*   **Monotonicity Constraint:** $\frac{\partial Q_{total}}{\partial Q_i} \geq 0$ ensures that maximizing individual $Q_i$ also maximizes $Q_{total}$.
*   This allows richer value factorization while preserving argmax consistency.

## Q4: What is the credit assignment problem in MARL?
**Answer:**
In cooperative MARL, all agents receive the same global reward.
**Problem:** How do we determine which agent's actions contributed to the reward?
**Solutions:**
*   **Value Decomposition (VDN/QMIX):** Decompose the global Q-value into individual contributions.
*   **Counterfactual Baselines:** Compare the reward with and without agent $i$.
*   **Difference Rewards:** Compute $r_i = r_{team} - r_{team \setminus i}$.

## Q5: What is MADDPG?
**Answer:**
**Multi-Agent DDPG** extends DDPG to multi-agent settings using CTDE:
*   **Centralized Critic:** $Q_i(o_1, ..., o_n, a_1, ..., a_n)$ sees all observations and actions during training.
*   **Decentralized Actor:** $\mu_i(o_i)$ uses only local observation during execution.
*   Each agent learns its own actor-critic pair, but the critic uses global information for more stable training.

## Q6: What are the main approaches to communication in MARL?
**Answer:**
*   **CommNet:** All agents share hidden states at each timestep (fully connected communication).
*   **TarMAC:** Targeted communication (agents learn who to communicate with via attention mechanisms).
*   **IC3Net:** Agents learn when and what to communicate (gating mechanism).
*   **Challenges:** Communication bandwidth, learning meaningful messages, scalability.
