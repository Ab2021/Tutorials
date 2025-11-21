# Day 19 Interview Questions: Inverse RL

## Q1: What is the fundamental difference between Reinforcement Learning and Inverse Reinforcement Learning?
**Answer:**
*   **RL:** Given Reward Function $\to$ Find Optimal Policy.
*   **IRL:** Given Optimal Policy (Demonstrations) $\to$ Find Reward Function.

## Q2: Why is Behavioral Cloning (BC) considered "brittle"?
**Answer:**
BC is essentially supervised learning on state-action pairs. It assumes independent and identically distributed (i.i.d.) data.
However, in RL, the agent's actions determine the next state (distribution shift).
If the agent makes a small error, it enters a state slightly different from the expert's trajectory. Since it has no training data there, it makes a bigger error. These errors compound quadratically over time ($O(T^2)$), leading to failure. This is known as **Covariate Shift**.

## Q3: Explain the concept of "Reward Ambiguity" in IRL.
**Answer:**
For any set of demonstrations, there are infinite reward functions that could explain them.
*   The trivial solution $R(s) = 0$ (or any constant) makes *any* policy optimal.
*   To solve this, we need constraints or priors (e.g., Maximum Entropy principle) to pick the most "reasonable" reward function that explains the data without assuming extra structure.

## Q4: How does GAIL relate to GANs?
**Answer:**
GAIL (Generative Adversarial Imitation Learning) uses the GAN framework.
*   The **Policy** acts as the **Generator** (producing state-action pairs).
*   A **Discriminator** tries to distinguish expert pairs from agent pairs.
*   The output of the Discriminator is used as a **Reward Signal** for the Policy.
This allows performing IRL and RL simultaneously in a scalable way.

## Q5: When would you use IRL instead of manually designing a reward function?
**Answer:**
1.  **Complex Tasks:** When the task is easy to do but hard to specify (e.g., "driving safely", "walking naturally").
2.  **Human-AI Interaction:** Learning user preferences.
3.  **Sim-to-Real:** Transferring a policy from a simulator (where we know rewards) to the real world (where we might want to fine-tune based on human demos).
