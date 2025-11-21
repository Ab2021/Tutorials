# Day 33: MuZero - Planning with Learned Models

## 1. AlphaZero Recap
**AlphaZero** (Silver et al., 2017) mastered Go, Chess, and Shogi using:
*   **Monte Carlo Tree Search (MCTS):** Plan by simulating games using the **true rules**.
*   **Self-Play:** Generate training data by playing against itself.
*   **Neural Network:** Predicts policy $p(a|s)$ and value $v(s)$.

**Limitation:** Requires knowing the game rules (can only play perfect-information board games).

## 2. MuZero: Learning the Rules
**MuZero** (Schrittwieser et al., 2020) learns a model for **planning only**, not for understanding:
*   **Representation Function:** $h_\theta(o_1, ..., o_t) \rightarrow s^0$ (encode history).
*   **Dynamics Function:** $g_\theta(s^k, a) \rightarrow (r^k, s^{k+1})$ (predict reward and next latent state).
*   **Prediction Function:** $f_\theta(s^k) \rightarrow (p^k, v^k)$ (predict policy and value).

**Key Insight:** The latent states $s^k$ don't need to represent the true state. They only need to be useful for **planning**.

## 3. MCTS with Learned Model
1.  Encode observations: $s^0 = h(o_1, ..., o_t)$.
2.  Run MCTS using the learned dynamics $g$ and prediction $f$.
3.  Select action based on visit counts.
4.  Update the model using self-play data.

## 4. Achievements
*   **Atari:** Matches or exceeds AlphaZero performance without knowing game rules.
*   **Go:** Matches AlphaZero (which has perfect knowledge).
*   **Real-World:** Applied to video compression (YouTube).

## 5. Code Sketch: MuZero Components
```python
class MuZeroNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = RepresentationNet()  # h: obs -> s
        self.dynamics = DynamicsNet()              # g: (s, a) -> (r, s')
        self.prediction = PredictionNet()          # f: s -> (p, v)
    
    def initial_inference(self, observation):
        # Encode observation to latent
        state = self.representation(observation)
        policy_logits, value = self.prediction(state)
        return state, policy_logits, value
    
    def recurrent_inference(self, hidden_state, action):
        # Predict next state and reward
        next_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_state)
        return next_state, reward, policy_logits, value

def  mcts_search(network, state, num_simulations=50):
    root = Node(state)
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        
        # Selection: traverse tree using UCB
        while node.expanded():
            action, node = select_child(node)
            search_path.append(node)
        
        # Expansion
        parent = search_path[-2]
        action = search_path[-1].action
        next_state, reward, policy, value = network.recurrent_inference(parent.state, action)
        node.expand(next_state, reward, policy)
        
        # Backpropagation
        backpropagate(search_path, value)
    
    # Return action with highest visit count
    return select_action(root)
```

### Key Takeaways
*   MuZero learns a model for planning, not for understanding.
*   The model predicts rewards and useful latent states for MCTS.
*   Achieves superhuman performance on Atari and board games without knowing rules.
