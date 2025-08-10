# Day 20.1: Reinforcement Learning Fundamentals & MDP Theory - A Practical Guide

## Introduction: Learning from Interaction

So far, we have focused on **Supervised Learning**, where a model learns from a labeled dataset. **Reinforcement Learning (RL)** is a completely different paradigm. It is about learning to make optimal decisions by **interacting with an environment**.

An **agent** (our model) exists in an **environment**. At each step, the agent observes the environment's **state**, takes an **action**, and receives a **reward** (or penalty) and the new state. The goal of the agent is not to predict a label, but to learn a **policy**—a strategy for choosing actions—that maximizes its cumulative reward over time.

This is the same way humans and animals learn: through trial and error. This paradigm is incredibly powerful and is used to solve problems involving sequential decision-making, from playing games like Go and StarCraft to controlling robotic arms and optimizing chemical reactions.

This guide will introduce the fundamental concepts of RL and the mathematical framework used to formalize it: the **Markov Decision Process (MDP)**.

**Today's Learning Objectives:**

1.  **Understand the Core Components of RL:** Learn the definitions of Agent, Environment, State, Action, and Reward.
2.  **Grasp the Agent-Environment Loop:** See how these components interact in a continuous loop.
3.  **Learn about the Markov Decision Process (MDP):** Understand the mathematical formulation of the RL problem, including the Markov property.
4.  **Implement a Simple Environment:** Build a basic grid-world environment from scratch to make the concepts concrete.
5.  **Define a Policy:** Understand the difference between a deterministic and a stochastic policy.
6.  **Calculate Cumulative Reward (Return):** Learn about the discounted return and the role of the discount factor, gamma.

--- 

## Part 1: The Agent-Environment Loop

Reinforcement Learning can be described by a simple loop:

1.  At time `t`, the **Agent** observes the current **State** `S_t` of the **Environment**.
2.  Based on `S_t`, the Agent chooses an **Action** `A_t` according to its **Policy** `pi`.
3.  The Environment receives the action `A_t`.
4.  The Environment transitions to a new **State** `S_{t+1}` and gives the Agent a **Reward** `R_{t+1}`.
5.  The loop repeats from the new state `S_{t+1}`.

![RL Loop](https://i.imgur.com/t5n6Q3X.png)

*   **Agent:** The learner and decision-maker. In deep RL, this is our neural network.
*   **Environment:** The world the agent exists in and interacts with. It defines the rules.
*   **State (S):** A complete description of the environment at a particular moment.
*   **Action (A):** A choice the agent can make.
*   **Reward (R):** A scalar feedback signal. It tells the agent how good or bad its last action was. The agent's goal is to maximize the total reward.
*   **Policy (π):** The agent's strategy or behavior. It's a function that maps a state to an action: `π(A|S)`. It defines which action to take in a given state.

--- 

## Part 2: The Markov Decision Process (MDP)

An MDP is the mathematical framework we use to describe the environment in an RL problem. An MDP is defined by a tuple `(S, A, P, R, γ)`:

*   `S`: A finite set of states.
*   `A`: A finite set of actions.
*   `P`: The **state transition probability function**, `P(s' | s, a)`. This defines the dynamics of the environment. It's the probability of transitioning to state `s'` if you are in state `s` and take action `a`.
*   `R`: The **reward function**, `R(s, a, s')`. This defines the reward you get for taking action `a` in state `s` and ending up in state `s'`.
*   `γ` (gamma): The **discount factor**, a number between 0 and 1.

### The Markov Property

The key assumption in an MDP is the **Markov Property**: "The future is independent of the past, given the present."

This means that the state `S_t` must contain all the information needed to make an optimal decision. The probability of transitioning to `S_{t+1}` depends *only* on `S_t` and `A_t`, not on the entire history of states and actions that came before.

### The Goal: Maximizing the Return

The agent's goal is to maximize not just the immediate reward, but the **cumulative future reward**, which is called the **return (G_t)**.

`G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...`

To prevent this sum from being infinite and to make future rewards less valuable than immediate ones (just like in finance), we introduce the **discount factor (γ)**.

`G_t = R_{t+1} + γ*R_{t+2} + γ^2*R_{t+3} + ...`

*   If `γ = 0`, the agent is **myopic** and only cares about the immediate reward.
*   If `γ` is close to 1, the agent is **farsighted** and cares about rewards far into the future.

--- 

## Part 3: Implementing a Simple Grid World Environment

To make these concepts concrete, let's build a simple text-based grid world. This is a classic RL problem.

*   **States:** The agent's (x, y) position on a grid.
*   **Actions:** Move Up, Down, Left, or Right.
*   **Rewards:** The agent gets a large positive reward for reaching a goal state, a large negative reward for falling into a hole, and a small negative reward for every other step (to encourage it to finish quickly).

```python
import numpy as np

print("--- Part 3: Implementing a Grid World Environment ---")

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.world = np.zeros((size, size))
        
        # Define special states
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.hole_pos = (size // 2, size // 2)
        
        # Define rewards
        self.goal_reward = 10
        self.hole_penalty = -10
        self.step_penalty = -0.1
        
        # Agent's current position
        self.agent_pos = self.start_pos
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = ['Up', 'Down', 'Left', 'Right']

    def reset(self):
        """Resets the environment to the starting state."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """
        Performs one step in the environment.
        Returns: (next_state, reward, done)
        """
        # --- State Transition Logic (Dynamics) ---
        x, y = self.agent_pos
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.size - 1, y + 1)
        
        self.agent_pos = (x, y)
        next_state = self.agent_pos
        
        # --- Reward Logic ---
        done = False
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        elif self.agent_pos == self.hole_pos:
            reward = self.hole_penalty
            done = True
        else:
            reward = self.step_penalty
            
        return next_state, reward, done

    def render(self):
        """Prints the current state of the world."""
        grid = np.full((self.size, self.size), '.')
        grid[self.goal_pos] = 'G'
        grid[self.hole_pos] = 'H'
        grid[self.agent_pos] = 'A'
        print("\n".join("".join(row) for row in grid))
        print("-" * self.size)

# --- Let's simulate an episode with a random policy ---
env = GridWorld()
state = env.reset()
done = False
total_reward = 0

print("Simulating an episode with a random agent:")
env.render()

for step_num in range(10):
    if done:
        break
    # Our policy is to choose a random action
    action = np.random.randint(0, 4)
    
    next_state, reward, done = env.step(action)
    
    print(f"Step {step_num+1}: Action='{env.actions[action]}', Reward={reward:.1f}")
    env.render()
    
    total_reward += reward
    state = next_state

print(f"\nEpisode finished. Total reward: {total_reward:.2f}")
```

--- 

## Part 4: The Policy

The agent's brain is its **policy**, `π(a|s)`. It tells the agent what to do in any given state.

*   **Deterministic Policy:** For a given state, it always returns the same action. `a = π(s)`.
*   **Stochastic Policy:** For a given state, it returns a **probability distribution** over all possible actions. `π(a|s) = P[A_t = a | S_t = s]`. The agent then samples an action from this distribution. Stochastic policies are more common in deep RL as they allow for exploration.

In our simulation above, our policy was a uniform random distribution: `π(a|s) = 0.25` for all actions `a` and states `s`.

The goal of all RL algorithms is to find an **optimal policy**, `π*`, that maximizes the expected discounted return.

## Conclusion

Reinforcement Learning is a powerful framework for solving sequential decision-making problems. By framing a problem in terms of an agent interacting with an environment to maximize a cumulative reward signal, we can begin to tackle complex, real-world challenges.

**Key Takeaways:**

1.  **The Agent-Environment Loop:** RL is defined by the continuous interaction between an agent and its environment through states, actions, and rewards.
2.  **The Goal is to Maximize Return:** The agent's objective is to learn a policy that maximizes the long-term discounted cumulative reward (the return).
3.  **MDPs Formalize the Problem:** The Markov Decision Process is the mathematical language we use to describe the environment, its dynamics (`P`), and its rewards (`R`).
4.  **The Markov Property is Key:** We assume that the current state provides all the necessary information to make an optimal decision.

With this foundational understanding of the RL problem, we are now ready to explore the algorithms that are used to solve it. In the next guides, we will see how to use neural networks to learn the optimal policy, starting with value-based methods like Q-learning.

## Self-Assessment Questions

1.  **RL Components:** What are the five core components of a Reinforcement Learning problem?
2.  **The Goal:** What is the ultimate objective of an RL agent?
3.  **Markov Property:** In your own words, what is the Markov Property?
4.  **Discount Factor (gamma):** What would be the effect of setting the discount factor `γ` to 0.99 versus 0.1?
5.  **Policy:** What is the difference between a deterministic and a stochastic policy?
