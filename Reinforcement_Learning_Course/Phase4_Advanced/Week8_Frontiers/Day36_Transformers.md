# Day 36: Transformers in Reinforcement Learning

## 1. Why Transformers for RL?
**Transformers** excel at sequence modeling and have revolutionized NLP.
**RL applications:**
*   **Long Context:** Model long episode histories.
*   **Multi-Modal:** Combine vision, language, and actions.
*   **Transfer Learning:** Pretrain on diverse tasks, fine-tune on specific ones.

## 2. Decision Transformer (Revisited)
Treats RL as **conditional sequence generation**:
*   Input: $(R_t, s_t, a_t, R_{t+1}, s_{t+1}, ...)$ where $R_t$ is return-to-go.
*   Output: Action $a_t$.
*   At test time, condition on desired return $R^*$.

**Advantages:**
*   No bootstrapping (offline-friendly).
*   Leverages pretrained Transformers.
*   Can handle multi-task learning naturally.

## 3. Trajectory Transformer
Models the entire trajectory distribution:
$$ p(s_0, a_0, r_0, s_1, a_1, r_1, ...) $$
*   Can generate entire trajectories.
*   Useful for planning and imitation.

## 4. GATO: Generalist Agent
**GATO** (DeepMind, 2022) is a single Transformer that can:
*   Play Atari games.
*   Caption images.
*   Chat.
*   Control a real robot arm.

**Key Idea:** All tasks are tokenized sequences. The model learns a universal policy.

## 5. Multi-Game Decision Transformer
Train on data from **multiple games** simultaneously:
*   The model learns to condition on the game/task.
*   Enables zero-shot transfer to new games (with in-context learning).

## 6. Code Sketch: Decision Transformer
```python
import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        self.embed_timestep = nn.Embedding(1000, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4), 
            num_layers=n_layers
        )
        
        self.predict_action = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, returns, states, actions, timesteps):
        # Embed each modality
        return_embeddings = self.embed_return(returns)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        
        # Interleave: (R, s, a, R, s, a, ...)
        # Simplified: just use states for now
        embeddings = state_embeddings + return_embeddings + time_embeddings
        
        # Transformer
        out = self.transformer(embeddings)
        
        # Predict actions
        action_preds = self.predict_action(out)
        return action_preds
```

### Key Takeaways
*   Transformers bring powerful sequence modeling to RL.
*   Decision Transformers treat RL as supervised learning on  sequences.
*   GATO demonstrates the potential of generalist agents.
*   Multi-task training enables transfer and few-shot adaptation.
