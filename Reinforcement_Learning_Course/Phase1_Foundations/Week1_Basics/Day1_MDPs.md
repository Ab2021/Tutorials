# Day 1: Introduction to Reinforcement Learning & MDPs

## 1. What is Reinforcement Learning?
Reinforcement Learning (RL) is a computational approach to learning from interaction. Unlike Supervised Learning (where we have labeled data) or Unsupervised Learning (where we find patterns in unlabeled data), RL involves an **agent** learning to make decisions by interacting with an **environment** to maximize a cumulative **reward**.

### The Agent-Environment Loop
1.  **Observation ($S_t$):** The agent perceives the state of the environment.
2.  **Action ($A_t$):** The agent decides to take an action based on the state.
3.  **Reward ($R_{t+1}$):** The environment provides feedback (scalar reward).
4.  **Next State ($S_{t+1}$):** The environment transitions to a new state.

## 2. Markov Decision Processes (MDPs)
An MDP is the mathematical framework used to describe the environment in RL. It is defined by the tuple $(S, A, P, R, \gamma)$:

*   **$S$ (State Space):** The set of all possible states.
*   **$A$ (Action Space):** The set of all possible actions.
*   **$P$ (Transition Probability):** $P(s'|s, a) = \mathbb{P}[S_{t+1}=s' | S_t=s, A_t=a]$. This defines the dynamics of the world.
*   **$R$ (Reward Function):** $R(s, a, s') = \mathbb{E}[R_{t+1} | S_t=s, A_t=a, S_{t+1}=s']$.
*   **$\gamma$ (Discount Factor):** $\gamma \in [0, 1]$. It determines the present value of future rewards.

### The Markov Property
"The future is independent of the past given the present."
Mathematically: $\mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_t, S_{t-1}, ..., S_0]$.
This means the current state $S_t$ captures all relevant information from the history.

## 3. Returns and Goals
The goal of the agent is to maximize the **expected return** ($G_t$).
The return is the sum of discounted future rewards:
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

*   If $\gamma = 0$: The agent is **myopic** (cares only about immediate reward).
*   If $\gamma \to 1$: The agent is **far-sighted** (cares about long-term cumulative reward).

## 4. Code Example: A Simple GridWorld MDP
Let's implement a basic GridWorld environment from scratch to understand $S, A, P, R$.

```python
import numpy as np

class GridWorld:
    def __init__(self, size=3):
        self.size = size
        self.state = (0, 0) # Start at top-left
        self.goal = (size-1, size-1) # Goal at bottom-right
        # Actions: 0: Up, 1: Right, 2: Down, 3: Left
        self.action_space = [0, 1, 2, 3]
    
    def step(self, action):
        x, y = self.state
        
        # Transition logic (Deterministic for now)
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Right
            y = min(self.size - 1, y + 1)
        elif action == 2: # Down
            x = min(self.size - 1, x + 1)
        elif action == 3: # Left
            y = max(0, y - 1)
            
        self.state = (x, y)
        
        # Reward function
        if self.state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
            
        return self.state, reward, done
    
    def reset(self):
        self.state = (0, 0)
        return self.state

# Interaction Loop
env = GridWorld()
state = env.reset()
done = False
steps = 0

print(f"Start State: {state}")
while not done:
    action = np.random.choice(env.action_space) # Random policy
    next_state, reward, done = env.step(action)
    print(f"Step {steps+1}: Action {action} -> State {next_state}, Reward {reward}")
    steps += 1

print("Goal Reached!")
```

### Key Takeaways
*   RL is about learning from interaction (trial and error).
*   MDPs formalize the problem.
*   The Agent maximizes discounted return.
