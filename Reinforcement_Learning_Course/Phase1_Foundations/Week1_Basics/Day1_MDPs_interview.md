# Day 1 Interview Questions: MDPs & RL Basics

## Q1: What is the Markov Property and why is it important?
**Answer:**
The Markov Property states that "the future is independent of the past given the present."
$$ \mathbb{P}[S_{t+1} | S_t] = \mathbb{P}[S_{t+1} | S_t, S_{t-1}, ..., S_0] $$
**Importance:** It simplifies the problem significantly. If the state is Markovian, the agent doesn't need to remember the entire history of the episode to make an optimal decision; it only needs to know the current state $S_t$. This allows us to use stationary policies $\pi(a|s)$.

## Q2: What is the difference between Reinforcement Learning and Supervised Learning?
**Answer:**
1.  **Feedback:** SL has "instructive" feedback (labels telling you the correct answer). RL has "evaluative" feedback (rewards telling you how good an action was, but not what the best action was).
2.  **Time:** SL usually assumes i.i.d. data. In RL, data is sequential and highly correlated (state $S_{t+1}$ depends on $S_t$).
3.  **Agency:** In SL, the model doesn't influence the data distribution. In RL, the agent's actions determine the data it sees next (exploration vs. exploitation).

## Q3: Why do we use a discount factor $\gamma$?
**Answer:**
1.  **Mathematical Convergence:** It ensures the sum of rewards (return) is finite in infinite horizon settings.
2.  **Preference:** It models the preference for immediate rewards over delayed rewards.
3.  **Uncertainty:** It accounts for the uncertainty of the environment or the possibility of the episode terminating unexpectedly.

## Q4: What happens if the environment is NOT Markovian?
**Answer:**
If the current observation $O_t$ doesn't capture all relevant information (e.g., a single frame of a moving ball doesn't show velocity), the environment is a POMDP.
**Solutions:**
1.  **State Augmentation:** Stack multiple frames (e.g., 4 frames in Atari) to capture velocity/direction, making the state effectively Markovian.
2.  **Recurrent Neural Networks (RNNs):** Use an LSTM/GRU to maintain a hidden state that summarizes the history.

## Q5: Can a reward be negative?
**Answer:**
Yes. Negative rewards are often used as "punishments" or "costs".
*   **Example:** A robot hitting a wall might get -10.
*   **Time Penalty:** Often, a reward of -1 is given for every time step to encourage the agent to finish the task as quickly as possible (e.g., solving a maze).

## Q6: What is the difference between a Policy and a Plan?
**Answer:**
*   **Policy ($\pi$):** A universal rule that maps *any* state to an action (or distribution over actions). It tells the agent what to do in every possible situation.
*   **Plan:** A specific sequence of actions from a *specific* start state (e.g., "Go Left, then Up, then Right"). A plan breaks if something unexpected happens (stochastic transition), whereas a policy handles it.
