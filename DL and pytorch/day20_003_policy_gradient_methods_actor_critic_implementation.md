# Day 20.3: Policy Gradient Methods & Actor-Critic - A Practical Guide

## Introduction: Learning the Policy Directly

In the previous guide, we explored value-based methods like DQN, which learn a value function and then derive a policy from it (e.g., by choosing the action with the highest Q-value). **Policy Gradient (PG)** methods represent a fundamentally different approach: they learn a **parameterized policy directly**, without needing to learn a value function first.

Instead of learning the *value* of actions, we directly learn the probability of taking each action in each state. We then adjust the parameters of our policy network based on the rewards we receive.

This guide provides a practical introduction to the core ideas behind policy gradient methods, the classic REINFORCE algorithm, and the powerful **Actor-Critic** architecture that combines the best of both policy-based and value-based methods.

**Today's Learning Objectives:**

1.  **Understand the Core Idea of Policy Gradients:** Grasp how we can directly optimize a policy by encouraging actions that lead to good outcomes.
2.  **Learn the REINFORCE Algorithm:** See the fundamental policy gradient algorithm in action.
3.  **Grasp the Concept of an Actor-Critic Model:** Understand how combining a policy network (the Actor) and a value network (the Critic) can lead to more stable and efficient learning.
4.  **Implement a Simple Actor-Critic Model:** Build and train an Actor-Critic agent to solve the CartPole environment.

--- 

## Part 1: The Policy Gradient Theorem

**The Goal:** We want to adjust our policy's parameters, `θ`, to maximize the expected return. The Policy Gradient Theorem provides a way to do this. It tells us that the gradient of the expected return is:

`∇_θ J(θ) = E[ ∇_θ log(π_θ(A_t|S_t)) * G_t ]`

Let's break this down:
*   `π_θ(A_t|S_t)`: Our policy network. It takes state `S_t` and outputs the probability of taking action `A_t`.
*   `log(π_θ(A_t|S_t))`: The log-probability of the action we actually took.
*   `∇_θ log(...)`: The gradient of this log-probability. This tells us which direction to change our parameters `θ` to make that action more or less likely in the future.
*   `G_t`: The **return** (the total discounted reward) we received after taking that action.

**The Intuition:**
*   If `G_t` is **high** (we got a good outcome), we want to make the action we took more likely. We "push up" the probability `π_θ(A_t|S_t)` by moving our parameters `θ` in the direction of the gradient `∇_θ log(...)`.
*   If `G_t` is **low** (we got a bad outcome), we want to make the action we took less likely. We "push down" the probability by moving `θ` in the opposite direction of the gradient.

This is the core of the **REINFORCE** algorithm.

**Problem with REINFORCE:** The return `G_t` can have very high variance. A good action might be followed by a string of bad luck, leading to a low return, and vice-versa. This makes the gradient estimates very noisy and the training process unstable.

--- 

## Part 2: The Actor-Critic Architecture

**The Solution:** To reduce this variance, we introduce a **baseline**. Instead of multiplying the gradient by the raw return `G_t`, we multiply it by `G_t - b(S_t)`, where `b(S_t)` is a baseline that depends only on the state. A natural choice for this baseline is the **state-value function, `V(s)`**.

The term `A(s, a) = Q(s, a) - V(s)` is called the **Advantage Function**. It tells us how much better taking action `a` is compared to the average action from state `s`.

This leads to the **Actor-Critic** architecture:

1.  **The Actor:** This is our **policy network**. It controls how the agent behaves. It takes a state and outputs a probability distribution over actions. `π_θ(a|s)`.

2.  **The Critic:** This is our **value network**. It estimates the state-value function, `V_φ(s)`. Its job is to critique the actions taken by the Actor. `φ` represents the Critic's parameters.

**The Training Process:**
*   The Actor takes an action `a` in state `s`.
*   We observe the next state `s'` and the reward `r`.
*   **The Critic evaluates the action:** It computes the TD Error: `δ = r + γ*V_φ(s') - V_φ(s)`. This TD error is a low-variance estimate of the Advantage function.
*   **The Critic updates its own weights `φ`** to make its value estimate `V_φ(s)` better (e.g., by minimizing the TD Error squared).
*   **The Actor updates its policy parameters `θ`** using this TD error as the signal: `∇_θ log(π_θ(a|s)) * δ`. If the TD error is positive (the action was better than expected), the Actor makes that action more likely. If it's negative, it makes it less likely.

![Actor-Critic](https://i.imgur.com/Y1Yl3oX.png)

--- 

## Part 3: Implementing an Actor-Critic for CartPole

Let's build a simple Actor-Critic agent to solve CartPole. This example is adapted from the official PyTorch documentation.

```python
import gymnasium as gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

print("--- Part 3: Implementing an Actor-Critic for CartPole ---")

# --- 1. Setup ---
env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. The Actor-Critic Model ---
# We use a single network with two output heads.
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128) # 4 observations in CartPole
        
        # The Actor head: outputs action probabilities
        self.action_head = nn.Linear(128, 2) # 2 actions: left or right
        # The Critic head: outputs a single state value
        self.value_head = nn.Linear(128, 1)
        
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # Actor: returns action logits
        action_logits = self.action_head(x)
        # Critic: returns state value
        state_values = self.value_head(x)
        return F.softmax(action_logits, dim=-1), state_values

# --- 3. The Training Loop ---
model = Policy().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
ep = np.finfo(np.float32).eps.item() # Small constant for numerical stability

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs, state_value = model(state)
    
    # Create a categorical distribution and sample an action
    m = Categorical(probs)
    action = m.sample()
    
    # Save the log-probability of the action and the state value
    model.saved_actions.append((m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    policy_losses = []
    value_losses = []
    returns = []
    
    # Calculate the discounted returns G_t for each step in the episode
    for r in model.rewards[::-1]:
        R = r + 0.99 * R # gamma = 0.99
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    # Calculate Actor and Critic losses
    for (log_prob, value), R in zip(model.saved_actions, returns):
        # Advantage = Return - Value (our TD Error estimate)
        advantage = R - value.item()
        
        # Actor loss (Policy Gradient)
        # We use -log_prob because we want to perform gradient *ascent*
        policy_losses.append(-log_prob * advantage)
        
        # Critic loss (MSE between value and actual return)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
        
    optimizer.zero_grad()
    # Sum the losses and perform a single backprop step
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    
    # Clear the saved actions and rewards for the next episode
    del model.saved_actions[:]
    del model.rewards[:]

# --- 4. Main Loop ---
running_reward = 10

for i_episode in count(1):
    state, _ = env.reset()
    ep_reward = 0
    for t in range(1, 10000):
        action = select_action(state)
        state, reward, done, _, _ = env.step(action)
        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break
            
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    finish_episode()
    
    if i_episode % 10 == 0:
        print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        
    # CartPole is considered solved if the average reward is > 195
    if running_reward > env.spec.reward_threshold:
        print(f"\nSolved! Running reward is now {running_reward} and the last episode received {ep_reward} reward!")
        break
```

## Conclusion

Policy Gradient methods offer a powerful and direct way to solve reinforcement learning problems. By parameterizing the policy itself and updating it to encourage actions that lead to higher returns, we can tackle a wide range of tasks.

**Key Takeaways:**

1.  **Direct Policy Optimization:** Unlike value-based methods, policy gradient methods directly learn the parameters of the agent's policy.
2.  **REINFORCE and High Variance:** The basic REINFORCE algorithm works but suffers from high variance in its gradient estimates, making it unstable.
3.  **Actor-Critic for Stability:** The Actor-Critic architecture is the modern standard. It combines the best of both worlds:
    *   The **Actor** (policy network) learns what actions to take.
    *   The **Critic** (value network) learns how to evaluate those actions, providing a low-variance learning signal (the advantage or TD error) to the Actor.
4.  **A Powerful Combination:** This combination allows for more stable and efficient learning than either value-based or simple policy-based methods alone.

Actor-Critic methods form the foundation of many state-of-the-art deep reinforcement learning algorithms, including A2C/A3C and more advanced methods like TRPO, PPO, and SAC.

## Self-Assessment Questions

1.  **Policy Gradient vs. Value-based:** What is the fundamental difference in what a policy gradient method learns compared to a Q-learning method?
2.  **The REINFORCE Rule:** In the REINFORCE algorithm, if an action leads to a large positive return, how are the policy parameters updated?
3.  **Actor vs. Critic:** In an Actor-Critic model, what is the job of the Actor? What is the job of the Critic?
4.  **The Advantage Function:** What does the advantage `A(s, a)` represent intuitively?
5.  **Training Signal:** In our Actor-Critic implementation, what value is used as the learning signal (i.e., what the log-probability is multiplied by) for updating the Actor?
