# Day 18 Interview Questions: Hierarchical RL

## Q1: What is the main motivation for Hierarchical RL?
**Answer:**
To solve the problem of **Temporal Abstraction** and **Sparse Rewards** in long-horizon tasks.
Standard RL operates at the level of atomic actions (e.g., 10ms motor torques). Planning a trip to London at this resolution involves millions of steps, making credit assignment impossible. HRL breaks the task into high-level sub-goals (e.g., "Go to airport"), reducing the effective horizon.

## Q2: Explain the difference between the Meta-Controller and the Controller in h-DQN.
**Answer:**
*   **Meta-Controller:** High-level agent. Takes state $s$, outputs a goal $g$. Operates on a slow timescale. Maximizes extrinsic reward.
*   **Controller:** Low-level agent. Takes state $s$ and goal $g$, outputs action $a$. Operates on a fast timescale. Maximizes intrinsic reward (reaching $g$).

## Q3: What is an "Option" in the Options Framework?
**Answer:**
An Option is a generalization of a primitive action. It is a temporally extended course of action.
It is defined by a tuple $(I, \pi, \beta)$:
*   $I$: Initiation set (where can I start this option?).
*   $\pi$: Policy (what do I do?).
*   $\beta$: Termination condition (when am I done?).
Example: "Open Door" is an option. $I$ = near door. $\pi$ = hand movements. $\beta$ = door is open.

## Q4: Why is HRL considered more sample efficient for Transfer Learning?
**Answer:**
Because the low-level skills (Controller policies) are often reusable.
If an agent learns how to "Walk" (Controller) to reach a "Flag" (Meta-Controller goal), and we change the task to "Walk" to a "Ball", the Controller's policy can be reused without retraining. Only the Meta-Controller needs to learn the new high-level strategy.

## Q5: What is the "Non-Stationarity" problem in HRL?
**Answer:**
The Meta-Controller is learning to select goals based on the Controller's performance.
However, the Controller is *also* learning and changing its behavior simultaneously.
From the Meta-Controller's perspective, the "transition dynamics" of the high-level MDP (i.e., probability of reaching goal $g$) are changing over time. This non-stationarity makes training unstable.
