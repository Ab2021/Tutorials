# Day 16: Recurrent RL (DRQN)

## 1. Partially Observable MDPs (POMDPs)
Standard DQN assumes the state is fully observable (Markov Property).
In many games (e.g., Doom, FPS) or real-world tasks, a single frame doesn't tell the whole story (e.g., "Am I moving left or right?").
*   **Frame Stacking:** We can stack 4 frames to capture velocity. This is a hack.
*   **Recurrent Neural Networks:** A more elegant solution is to use an LSTM/GRU to maintain a **hidden state** $h_t$ that summarizes the entire history.

## 2. Deep Recurrent Q-Network (DRQN)
DRQN replaces the first fully connected layer of DQN with an LSTM layer.
*   **Input:** Sequence of frames $o_t$.
*   **Internal State:** $h_t = \text{LSTM}(o_t, h_{t-1})$.
*   **Output:** $Q(h_t, a)$.
*   **Benefit:** Can solve POMDPs where information is hidden for long periods (e.g., "I saw a key 100 steps ago").

## 3. Training Challenges
Training DRQN is harder than DQN.
*   **Sequential Updates:** We can't just sample random transitions $(s, a, r, s')$. We need **sequences** (traces) of experience to train the LSTM.
*   **Replay Buffer:** Stores episodes, not transitions.
*   **Sampling:** Sample a random episode, then sample a sequence of length $L$ (e.g., 8 steps) from it.

## 4. Code Example: DRQN Architecture
```python
import torch
import torch.nn as nn

class DRQN(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature Extractor (CNN or Linear)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        
        # Q-Value Head
        self.fc = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        features = self.feature_extractor(x)
        
        # LSTM forward
        # out shape: (batch_size, seq_len, hidden_dim)
        out, hidden = self.lstm(features, hidden)
        
        # Q-values for each step in sequence
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

# Usage
net = DRQN(input_dim=10, action_dim=4)
dummy_input = torch.randn(32, 8, 10) # Batch 32, Seq 8, Dim 10
hidden = net.init_hidden(32)
q_vals, new_hidden = net(dummy_input, hidden)
print("Q-Values Shape:", q_vals.shape) # (32, 8, 4)
```

### Key Takeaways
*   DRQN handles partial observability.
*   Requires training on sequences.
*   Crucial for First-Person Shooters and Robotics.
