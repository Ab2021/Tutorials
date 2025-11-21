# Day 16 Interview Questions: Recurrent RL (DRQN)

## Q1: What is a POMDP? Give an example.
**Answer:**
Partially Observable Markov Decision Process.
It is an MDP where the agent cannot directly observe the full state $S_t$. Instead, it receives an observation $O_t$ which is a function of the state (often noisy or incomplete).
*   **Example:** First-Person Shooter game. You only see what's in front of you. You don't know if an enemy is behind you unless you remember seeing them earlier. The "State" includes the enemy's position, but the "Observation" does not.

## Q2: Why does standard DQN fail in POMDPs?
**Answer:**
DQN learns a mapping $Q(s, a)$. It assumes that the input $s$ contains all necessary information to make an optimal decision (Markov Property).
In a POMDP, $O_t$ is ambiguous. Two different states (e.g., "Enemy behind" vs "No enemy") might produce the exact same observation (empty corridor). DQN will try to learn an average value for this observation, which is sub-optimal for both situations.

## Q3: How does DRQN solve the POMDP problem?
**Answer:**
By adding a recurrent layer (LSTM/GRU), DRQN maintains an internal hidden state $h_t$.
$$ h_t = f(h_{t-1}, o_t) $$
This hidden state integrates information over time, effectively reconstructing the true state $S_t$ from the history of observations. The Q-function $Q(h_t, a)$ is then conditioned on this history.

## Q4: What is the "Burn-In" strategy in training Recurrent RL?
**Answer:**
When training on short random sequences from a replay buffer, initializing the hidden state to zero is incorrect (because the sequence actually happened in the middle of an episode).
**Burn-In** involves running the network for a few steps (e.g., 40 steps) *before* the training sequence starts, without updating weights. This allows the hidden state to evolve from zero to a meaningful representation of the history before the gradient updates begin.

## Q5: Can we just stack frames instead of using an LSTM?
**Answer:**
Frame stacking (e.g., 4 frames) helps with short-term dependencies (like velocity or acceleration). However, it cannot handle **long-term dependencies** (e.g., remembering a key location seen 1000 steps ago). LSTMs (or Transformers) are needed for long-term memory.
