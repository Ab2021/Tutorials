# Day 29: Meta-Reinforcement Learning

## 1. The Generalization Problem
Standard RL excels at a single task but struggles when transferred to new tasks.
**Meta-RL (Learning to Learn):** Train an agent on a distribution of tasks so it can quickly adapt to new tasks.
*   **Goal:** Few-shot adaptation (learn a new task with minimal data).
*   **Example:** A robot learning to open different types of doors.

## 2. Problem Formulation
A **task distribution** $p(\mathcal{T})$ where each task $\mathcal{T}_i$ is an MDP.
*   **Meta-Training:** Train on tasks sampled from $p(\mathcal{T})$.
*   **Meta-Testing:** Adapt to a new task $\mathcal{T}_{new} \sim p(\mathcal{T})$ with few samples.
*   **Metric:** Performance after $K$ episodes on the new task.

## 3. MAML: Model-Agnostic Meta-Learning
**Key Idea:** Learn an initialization $\theta_0$ that can be quickly fine-tuned to any task.
1.  **Inner Loop (Adaptation):** For each task $\mathcal{T}_i$, take one or few gradient steps:
    $$ \theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta) $$
2.  **Outer Loop (Meta-Update):** Update $\theta$ based on the performance after adaptation:
    $$ \theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}(\theta_i') $$
*   This requires **second-order derivatives** (backprop through the inner gradient step).

## 4. RL²: Recurrent Learning to Learn
Use an **RNN** to encode the task:
*   The RNN's hidden state acts as a "memory" of the task.
*   The agent learns to adapt by updating its hidden state based on experience.
*   **Advantage:** No explicit meta-update. The RNN implicitly learns to learn.
*   **Training:** Standard RL (PPO, A2C) on episodes that span multiple tasks.

## 5. Code Sketch: MAML for RL
```python
import torch

def maml_update(policy, task_distribution, inner_steps=1, inner_lr=0.01, outer_lr=0.001):
    meta_optimizer = torch.optim.Adam(policy.parameters(), lr=outer_lr)
    
    for meta_iteration in range(1000):
        tasks = task_distribution.sample(n_tasks=5)
        meta_loss = 0
        
        for task in tasks:
            # Inner loop: Adapt to task
            task_policy = copy.deepcopy(policy)
            task_optimizer = torch.optim.SGD(task_policy.parameters(), lr=inner_lr)
            
            for _ in range(inner_steps):
                # Collect data from task
                states, actions, rewards = collect_episode(task, task_policy)
                loss = compute_policy_loss(task_policy, states, actions, rewards)
                
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
            
            # Outer loop: Evaluate adapted policy
            states_test, actions_test, rewards_test = collect_episode(task, task_policy)
            meta_loss += compute_policy_loss(task_policy, states_test, actions_test, rewards_test)
        
        # Meta-update
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
```

## 6. Applications
*   **Few-Shot Imitation Learning:** Learn to mimic a new behavior from a single demonstration.
*   **Sim-to-Real Transfer:** Train in simulation on diverse tasks, adapt quickly to the real world.
*   **Continual Learning:** Lifelong learning where the agent adapts to changing environments.

### Key Takeaways
*   Meta-RL enables fast adaptation to new tasks.
*   MAML learns a good initialization for fine-tuning.
*   RL² uses an RNN to implicitly encode the task.
