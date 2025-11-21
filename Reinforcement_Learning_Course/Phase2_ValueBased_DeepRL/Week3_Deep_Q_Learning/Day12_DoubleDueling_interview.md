# Day 12 Interview Questions: Double & Dueling DQN

## Q1: Why does Q-Learning suffer from overestimation bias?
**Answer:**
Because of the maximization step ($\max_a Q(s, a)$) in the target calculation.
If the Q-value estimates contain random noise, taking the maximum tends to select the highest noise (positive error) rather than the true highest value. This bias propagates through the Bellman equation, leading to inflated Q-values.

## Q2: How does Double DQN fix overestimation?
**Answer:**
It separates the **selection** of the best action from the **evaluation** of that action.
*   **Selection:** Uses the *online* network to find the greedy action: $a^* = \arg\max Q_{online}(s', a)$.
*   **Evaluation:** Uses the *target* network to estimate the value of that action: $Q_{target}(s', a^*)$.
Since the two networks are likely to have different noise patterns, it's unlikely that the action maximizing the online network also has a positively biased value in the target network.

## Q3: What is the main motivation for Dueling DQN?
**Answer:**
To allow the agent to learn the state value $V(s)$ independently of the action advantages $A(s, a)$.
In many states, the choice of action doesn't affect the outcome much (e.g., waiting for a monster to appear). In standard DQN, the agent has to learn $Q(s, a)$ for every action separately to realize the state is bad. In Dueling DQN, it learns $V(s)$ once, which generalizes to all actions, improving sample efficiency.

## Q4: Why do we subtract the mean from the Advantage stream in Dueling DQN?
**Answer:**
To ensure **identifiability**.
Without this constraint, the equation $Q = V + A$ has infinite solutions (we can add a constant to V and subtract it from A). Subtracting the mean forces $V(s)$ to represent the average Q-value in that state, making the decomposition unique and stable.

## Q5: Is Double DQN always better than DQN?
**Answer:**
In practice, almost always yes. It costs nothing (no extra parameters, just a slight change in the update equation) and provides significant stability improvements. It is considered a standard default in value-based RL.
