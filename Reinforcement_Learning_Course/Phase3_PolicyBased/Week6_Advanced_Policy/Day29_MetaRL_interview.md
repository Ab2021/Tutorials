# Day 29 Interview Questions: Meta-RL

## Q1: What is Meta-Reinforcement Learning?
**Answer:**
Meta-RL (Learning to Learn) trains an agent on a distribution of tasks so it can quickly adapt to new tasks with minimal data.
**Goal:** Few-shot learning—learn a new task using only a few episodes.
**Example:** A robot trained on opening various doors can quickly learn to open a new type of door.

## Q2: How does MAML work?
**Answer:**
MAML learns an initialization $\theta_0$ that can be quickly fine-tuned to any task.
**Training:**
1. **Inner Loop:** For each task, take gradient steps to adapt: $\theta' = \theta - \alpha \nabla \mathcal{L}_{task}(\theta)$.
2. **Outer Loop:** Update $\theta$ based on performance after adaptation: $\theta \leftarrow \theta - \beta \nabla \mathcal{L}_{task}(\theta')$.
**Key Insight:** We're optimizing for a good starting point, not a good final policy.

## Q3: Why does MAML require second-order derivatives?
**Answer:**
The outer loop updates $\theta$ based on $\mathcal{L}(\theta')$, where $\theta'$ is a function of $\theta$ (from the inner loop).
$$ \nabla_\theta \mathcal{L}(\theta') = \frac{d\mathcal{L}}{d\theta'} \frac{d\theta'}{d\theta} $$
The term $\frac{d\theta'}{d\theta}$ involves the Hessian (second derivative).
**FOMAML** approximates this by ignoring $\frac{d\theta'}{d\theta}$, making it much faster with minimal performance loss.

## Q4: What is RL²?
**Answer:**
**RL² (Fast RL via Slow RL)** uses an RNN to implicitly encode the task.
*   The RNN's hidden state acts as a "memory" of the task.
*   Episodes are structured as: $(task_1, task_2, ..., task_n)$ where each task is a sequence of episodes.
*   The agent learns to adapt by updating its hidden state based on experience.
*   **Advantage:** No explicit inner/outer loop. The RNN learns to learn implicitly.

## Q5: What are the applications of Meta-RL?
**Answer:**
*   **Few-Shot Imitation:** Learn a new behavior from 1-5 demonstrations.
*   **Sim-to-Real Transfer:** Train on diverse simulated tasks, adapt quickly to the real robot.
*   **Continual Learning:** Adapt to changing environments over time.
*   **Personalization:** Adapt policies to individual users (e.g., personalized healthcare).

## Q6: What is the difference between MAML and Reptile?
**Answer:**
*   **MAML:** Uses first- or second-order gradients. Inner loop adapts, outer loop meta-updates.
*   **Reptile:** Simpler. Take multiple SGD steps on the task, then move toward the adapted parameters: $\theta \leftarrow \theta + \epsilon(\theta' - \theta)$.
*   Reptile is faster and easier to implement, with similar performance to MAML.
