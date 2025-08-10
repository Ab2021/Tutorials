# Day 20.2: Deep Q-Networks & Value-Based Methods - A Practical Guide

## Introduction: Learning the Value of Actions

How does an agent learn an optimal policy? One of the main families of algorithms in Reinforcement Learning, **value-based methods**, does this indirectly. Instead of learning a policy `π(a|s)` directly, it learns a **value function**. A value function estimates how good it is to be in a particular state, or how good it is to take a particular action in a particular state.

If the agent knows the value of every possible action it can take from its current state, it can simply choose the action with the highest value. This is the core idea behind **Q-Learning** and its deep learning extension, the **Deep Q-Network (DQN)**.

This guide provides a practical introduction to Q-learning and shows how to build a simple DQN to solve a classic RL environment.

**Today's Learning Objectives:**

1.  **Understand Value Functions:** Learn the difference between the state-value function `V(s)` and the action-value function `Q(s, a)`.
2.  **Grasp the Bellman Equation:** See the fundamental recursive relationship that defines value functions.
3.  **Learn the Q-Learning Algorithm:** Understand how the Q-update rule allows an agent to learn the optimal Q-function through trial and error.
4.  **Explore Deep Q-Networks (DQN):** See how a neural network can be used to approximate the Q-function for problems with large state spaces.
5.  **Understand Experience Replay and Target Networks:** Learn about the two key innovations that made DQN training stable.
6.  **Implement a DQN:** Build and train a simple DQN to solve the CartPole environment.

---

## Part 1: Value Functions and the Bellman Equation

*   **State-Value Function `V(s)`:** The expected return when starting in state `s` and following a policy `π` thereafter. "How good is it to be in this state?"

*   **Action-Value Function `Q(s, a)`:** The expected return when starting in state `s`, taking action `a`, and then following a policy `π` thereafter. "How good is it to take this action in this state?" This is what we will focus on.

**The Bellman Equation:**
This is the foundational equation of RL. It expresses the value of a state in terms of the values of its successor states. For the Q-function, it is:

`Q(s, a) = E[R_{t+1} + γ * max_{a'} Q(S_{t+1}, a')]`

In words: "The value of taking action `a` in state `s` is the expected immediate reward `R_{t+1}` plus the discounted value of the **best possible action** `a'` you can take in the next state `S_{t+1}`."

This recursive relationship is the key. If we knew the optimal Q-function `Q*`, the optimal policy would be to always choose the action that maximizes it: `π*(s) = argmax_a Q*(s, a)`.

---

## Part 2: The Q-Learning Algorithm

Q-learning is an algorithm that iteratively approximates the optimal Q-function, `Q*`, directly from experience, without needing a model of the environment.

**The Update Rule:**
At each step, after taking action `A_t` in state `S_t` and observing the reward `R_{t+1}` and next state `S_{t+1}`, we update our estimate of `Q(S_t, A_t)`:

`Q(S_t, A_t) <- Q(S_t, A_t) + α * [ (R_{t+1} + γ * max_a Q(S_{t+1}, a)) - Q(S_t, A_t) ]`

Let's break this down:
*   `α`: The learning rate.
*   `(R_{t+1} + γ * max_a Q(S_{t+1}, a))`: This is our **target value**. It's our new, better estimate of the Q-value based on the reward we just got and our current estimate of the value of the next state.
*   `[...]`: This whole term is the **Temporal Difference (TD) Error**. It's the difference between our new target and our old estimate. We are nudging our old estimate in the direction of the new target.

---

## Part 3: Deep Q-Networks (DQN)

**The Problem:** For environments with a small number of states and actions, we can represent the Q-function as a simple table (a Q-table). But for environments with large or continuous state spaces (like from image pixels), this is impossible.

**The Solution:** We use a **neural network** to approximate the Q-function. This network takes the state `s` as input and outputs a vector of Q-values, one for each possible action `a`.

`Q(s, a; θ) ≈ Q*(s, a)` (where `θ` are the network's weights).

**The Training Process:** We can train this network using a loss function based on the TD Error from the Bellman equation.

`Loss = ( (R_{t+1} + γ * max_a Q(S_{t+1}, a; θ)) - Q(S_t, A_t; θ) )^2`

This looks like a standard supervised learning problem where:
*   **Input:** `S_t`
*   **Prediction:** `Q(S_t, A_t; θ)`
*   **Target:** `R_{t+1} + γ * max_a Q(S_{t+1}, a; θ)`

However, training this naively is very unstable. The DQN paper introduced two crucial innovations:

1.  **Experience Replay:** We don't train the network on experiences as they happen. Instead, we store a large number of past transitions `(s, a, r, s')` in a memory buffer called a **replay buffer**. During training, we sample **random mini-batches** from this buffer. This breaks the correlation between consecutive samples, making training more stable and data-efficient.

2.  **Fixed Q-Targets (Target Network):** Notice that the **target** in our loss function depends on the same network we are trying to update. This is like a dog chasing its own tail and leads to instability. To fix this, we use **two** neural networks:
    *   A **policy network** (`Q(s, a; θ)`) that we are actively training.
    *   A **target network** (`Q(s, a; θ⁻)`) whose weights are a delayed copy of the policy network's weights. The target network is used *only* to calculate the target value. Its weights are frozen for a period and then periodically updated with the weights from the policy network. This provides a stable, consistent target for the policy network to learn towards.

---

## Part 4: Implementing a DQN for CartPole

Let's put it all together to solve the classic `CartPole-v1` environment from OpenAI Gym. The goal is to balance a pole on a cart by moving the cart left or right.

### 4.1. Setup and Replay Memory

```python
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("--- Part 4: Implementing a DQN for CartPole ---")

# --- 1. Setup Environment and Device ---
env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Experience Replay Buffer ---
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

### 4.2. The DQN Model and Training Loop

```python
# --- 3. The DQN Architecture ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 4. Training Setup ---
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9 # Epsilon for epsilon-greedy exploration
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005 # Update rate for the target network
LR = 1e-4

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    # Epsilon-greedy action selection
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    # The core training step
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- 5. The Main Training Loop ---
num_episodes = 100 # For a real run, use more (e.g., 600)
episode_durations = []

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 20 == 0:
                print(f"Episode {i_episode+1}/{num_episodes} finished after {t+1} timesteps.")
            break

print('\nComplete')
# (Plotting code would go here)
```

## Conclusion

Deep Q-Networks were a major breakthrough, demonstrating that a deep neural network could be trained with reinforcement learning to master complex tasks directly from high-dimensional sensory input (in the original paper, Atari game pixels).

**Key Takeaways:**

1.  **Q-function vs. V-function:** What is the main difference between the state-value function `V(s)` and the action-value function `Q(s, a)`?
2.  **TD Error:** In the Q-learning update rule, what does the Temporal Difference (TD) Error represent?
3.  **Experience Replay:** What are the two main benefits of using an experience replay buffer?
4.  **Target Network:** Why is a separate target network used in the DQN algorithm?
5.  **Epsilon-Greedy:** In our implementation, we used an epsilon-greedy strategy for `select_action`. What is the purpose of the "epsilon" part of this strategy?
