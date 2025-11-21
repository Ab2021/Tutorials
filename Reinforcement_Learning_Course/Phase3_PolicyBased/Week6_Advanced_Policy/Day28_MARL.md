# Day 28: Multi-Agent Reinforcement Learning (MARL)

## 1. From Single to Multi-Agent
So far, we've considered a single agent interacting with a stationary environment.
**Multi-Agent RL (MARL):** Multiple agents interact with each other in a shared environment.
*   **Example:** Autonomous vehicles, robot teams, game playing (e.g., StarCraft, Dota).
*   **Challenge:** The environment is **non-stationary** from each agent's perspective (other agents are learning too).

## 2. Types of MARL Settings
*   **Cooperative:** All agents share a common goal (maximize team reward).
    *   Example: Robot warehouse, multi-robot exploration.
*   **Competitive:** Agents have opposing goals (zero-sum game).
    *   Example: Chess, Go, Poker.
*   **Mixed:** Some agents cooperate, some compete.
    *   Example: Multi-team sports, market competition with alliances.

## 3. Centralized Training, Decentralized Execution (CTDE)
A key paradigm in cooperative MARL:
*   **Training (Centralized):** Use global state information, all agents' actions, etc.
*   **Execution (Decentralized):** Each agent uses only its local observations.
*   **Why?** Centralized training leverages full information for faster learning. Decentralized execution is more practical (communication constraints).

## 4. Independent Q-Learning (IQL)
The simplest baseline: Each agent learns independently using Q-Learning.
*   Each agent treats other agents as part of the environment.
*   **Problem:** The environment is non-stationary (others are learning).
*   **Result:** Often fails to converge.

## 5. Value Decomposition Networks (VDN)
For cooperative settings, decompose the team Q-value:
$$ Q_{total}(s, a_1, ..., a_n) = \sum_{i=1}^n Q_i(o_i, a_i) $$
*   Each agent has a local Q-function $Q_i$ based on its observation $o_i$.
*   The sum equals the global Q-value.
*   **Training:** Use the global reward to train $Q_{total}$.
*   **Execution:** Each agent uses $\arg\max_{a_i} Q_i(o_i, a_i)$.

## 6. QMIX: Monotonic Value Factorization
VDN's additivity is too restrictive.
**QMIX** uses a **mixing network**:
$$ Q_{total}(s, a_1, ..., a_n) = f_{mix}(Q_1, ..., Q_n; s) $$
*   $f_{mix}$ is a neural network with **non-negative weights** (ensures monotonicity).
*   **Monotonicity:** $\frac{\partial Q_{total}}{\partial Q_i} \geq 0$.
*   This allows richer value factorization while preserving the argmax consistency.

## 7. Code Sketch: QMIX Architecture
```python
import torch
import torch.nn as nn

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, obs):
        return self.net(obs)

class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super().__init__()
        self.n_agents = n_agents
        # Hypernetworks to generate weights (ensure non-negative)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, 1))
    
    def forward(self, agent_qs, state):
        # agent_qs: [batch, n_agents]
        # state: [batch, state_dim]
        batch_size = agent_qs.size(0)
        
        # Generate weights (non-negative via abs)
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, 32)
        b1 = self.hyper_b1(state).view(batch_size, 1, 32)
        
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, 32, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, 1)
```

### Key Takeaways
*   MARL is challenging due to non-stationarity and credit assignment.
*   CTDE is a powerful paradigm for cooperative tasks.
*   QMIX is state-of-the-art for cooperative MARL.
