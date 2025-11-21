# Day 18 Deep Dive: Designing Hierarchies

## 1. The Challenge of Sub-goals
In h-DQN, we assume the goals are predefined (e.g., "Reach (x, y)").
In general HRL, discovering *what* the sub-goals should be is the hardest problem.
*   **Bottleneck States:** Sub-goals are often "bottlenecks" in the state space (e.g., doorways, elevators).
*   **Eigen-options:** Using graph Laplacian to find natural clusters in the state transition graph.

## 2. Intrinsic Rewards
The Controller is trained with Intrinsic Rewards.
$$ R_{int}(s, g, s') = \mathbb{I}[s' \in g] $$
*   This reward is **dense** (the controller gets feedback every time it reaches a subgoal).
*   The Meta-Controller gets **sparse** reward (only when the final task is done).
*   This decoupling allows the Controller to learn basic skills (navigation) even if the Meta-Controller is still exploring randomly.

## 3. The Options Framework
Sutton, Precup, & Singh (1999) formalized HRL using **Options**.
An Option $\omega$ consists of:
1.  **Initiation Set $I_\omega$:** States where the option can start.
2.  **Policy $\pi_\omega$:** The behavior within the option.
3.  **Termination Condition $\beta_\omega$:** Probability of ending the option.
h-DQN is a specific implementation where options are "Reach Goal $g$".

## 4. Feudal Reinforcement Learning
Another HRL architecture where a "Manager" sets tasks for a "Worker".
*   **Manager:** Operates at low temporal resolution.
*   **Worker:** Operates at high temporal resolution.
*   **Directional Goals:** The Manager can output a direction vector $d$, and the Worker is rewarded for moving in that direction: $r_{int} = d^T (s_{t+1} - s_t)$.
