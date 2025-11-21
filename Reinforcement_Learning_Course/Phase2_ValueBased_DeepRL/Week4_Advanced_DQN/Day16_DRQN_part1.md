# Day 16 Deep Dive: Training Recurrent Agents

## 1. Random Replay vs. Sequential Replay
In standard DQN, we sample random transitions. In DRQN, we have two choices:
1.  **Bootstrapped Sequential Updates:**
    *   Sample a random episode.
    *   Start from the beginning ($h_0 = 0$).
    *   Run through the entire episode to update.
    *   **Pros:** Mathematically correct hidden state.
    *   **Cons:** Violates i.i.d. assumption (highly correlated gradients), inefficient for long episodes.
2.  **Random Sequence Replay:**
    *   Sample a random sequence of length $L$ (e.g., 8 steps) from an episode.
    *   Initialize $h_{start} = 0$ (Zero Start).
    *   **Pros:** Breaks correlations.
    *   **Cons:** The hidden state at the start of the sequence is wrong (it assumes no history, but there *was* history).

## 2. Burn-In Period
To fix the "Zero Start" problem in Random Sequence Replay:
*   Sample a sequence of length $L_{burn} + L_{train}$.
*   Run the LSTM for the first $L_{burn}$ steps **without** calculating loss or updating weights. This allows the hidden state $h$ to "warm up" and capture some history.
*   Train on the remaining $L_{train}$ steps.

## 3. Hidden State Management
When interacting with the environment (inference), we must carry the hidden state forward.
*   `state, hidden = env.reset(), net.init_hidden()`
*   Loop:
    *   `action, hidden = net(state, hidden)`
    *   `next_state, reward, done = env.step(action)`
    *   If `done`: `hidden = net.init_hidden()` (Reset hidden state for new episode).

## 4. Attention Mechanisms
Modern alternatives to LSTMs in RL include **Transformers** (Decision Transformer, GTrXL).
*   **Benefit:** Can handle much longer contexts than LSTMs.
*   **Cost:** Computationally expensive ($O(N^2)$ attention).
