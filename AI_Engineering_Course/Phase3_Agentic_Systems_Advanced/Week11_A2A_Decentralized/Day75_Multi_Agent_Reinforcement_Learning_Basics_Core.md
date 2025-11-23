# Day 75: Multi-Agent Reinforcement Learning (MARL) Basics
## Core Concepts & Theory

### From One to Many

Single-Agent RL (AlphaGo) is about "Me vs Environment".
Multi-Agent RL (OpenAI Five) is about "Me vs You" or "Us vs Them".
The environment is no longer static; it contains other agents who are learning and changing.

### 1. Cooperative vs Competitive

*   **Cooperative:** All agents share the same Reward Function. (e.g., Warehouse robots). Goal: Maximize Team Score.
*   **Competitive (Zero-Sum):** My gain is your loss. (e.g., Chess, Poker). Goal: Beat Opponent.
*   **Mixed-Sum:** Complex incentives. (e.g., Traffic, Prisoner's Dilemma).

### 2. The Non-Stationarity Problem

In Single-Agent RL, if I take action A in state S, the probability of reaching state S' is fixed.
In MARL, the transition depends on *what the other agents do*.
Since other agents are learning, the "rules of the game" seem to change constantly from my perspective.
*   *Result:* Standard Q-Learning fails because the Q-values never converge.

### 3. Centralized Training, Decentralized Execution (CTDE)

The dominant paradigm in MARL.
*   **Training:** The "Coach" sees everything (God View). It trains the agents using global information.
*   **Execution:** The "Player" only sees its local view. It acts based on its own policy.
*   *Algorithm:* MAPPO (Multi-Agent PPO), QMIX.

### 4. Nash Equilibrium

The goal of MARL is rarely "Optimal Policy" (because it depends on the opponent).
The goal is **Nash Equilibrium**: A strategy where no player has an incentive to deviate, assuming others hold their strategies constant.

### 5. Self-Play

How do you train an agent to be the best in the world?
You make it play against itself.
*   **Curriculum:** As the agent gets better, its opponent (itself) gets better.
*   **Explosion:** This leads to exponential skill improvement (AlphaZero).

### Summary

MARL is the frontier of AI. It deals with the complexity of social interaction, strategy, and adaptation. It is essential for autonomous driving, finance, and RTS games.
