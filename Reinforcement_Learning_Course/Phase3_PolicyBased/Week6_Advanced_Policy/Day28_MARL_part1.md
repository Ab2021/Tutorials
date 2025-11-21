# Day 28 Deep Dive: MADDPG and Communication

## 1. MADDPG: Multi-Agent DDPG
For continuous control in multi-agent settings, **MADDPG** extends DDPG:
*   **Centralized Critic:** $Q_i(o_1, ..., o_n, a_1, ..., a_n)$ sees all observations and actions.
*   **Decentralized Actor:** $\mu_i(o_i)$ only sees local observation.
*   **Training:** Use centralized critic to train each actor.
*   **Execution:** Each agent uses only its local actor.

## 2. Communication in MARL
Agents can improve coordination by **communicating**:
*   **CommNet:** All agents share hidden states at each timestep.
*   **TarMAC:** Targeted multi-agent communication (agents learn who to communicate with).
*   **IC3Net:** Agents learn when and what to communicate.

## 3. Credit Assignment Problem
In cooperative MARL, how do we assign credit to individual agents?
*   **Global Reward:** All agents receive the same team reward. Hard to tell who contributed.
*   **VDN/QMIX:** Decompose the value function to assign implicit credit.
*   **Counterfactual Baseline:** Compare the team reward with and without agent $i$.

## 4. Competitive MARL: Self-Play
In competitive settings (e.g., AlphaGo), agents play against themselves:
*   Agent plays against past versions of itself.
*   This creates a curriculum of increasingly difficult opponents.
*   **Problem:** Can lead to cyclic strategies (rock-paper-scissors).
*   **Solution:** Maintain a population of diverse agents.
