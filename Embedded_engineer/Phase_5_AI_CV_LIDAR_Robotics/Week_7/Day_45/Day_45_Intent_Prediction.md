# Day 45: Intent Prediction (Pedestrian Trajectory)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> A robot shouldn't just avoid where you *are*. It should avoid where you *will be*.
> - **Focus:** Social Force Model, Kalman Filtering (CV Model), and LSTM Trajectory Prediction.
> - **Code:** Predicting pedestrian paths and planning a robot path that yields courteously.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Model** pedestrian motion using Constant Velocity (CV) and Social Force Models.
2.  **Train** a simple Recurrent Neural Network (LSTM) to predict future coordinates $(x_{t+k}, y_{t+k})$.
3.  **Integrate** predictions into the Local Planner (TEB/DWA) as dynamic obstacles.
4.  **Evaluate** "Social Compliance": Did the robot scare the human?

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install torch numpy matplotlib
```

### Prior Knowledge
- Kalman Filters (Day 10).
- Deep Learning (Day 1-4).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Physics-Based Models

1.  **Constant Velocity (CV):** Good for 0.5s horizon.
    *   $x_{t+1} = x_t + v_x \Delta t$.
2.  **Social Force Model (Helbing 1995):** Humans are particles.
    *   $F_{attr}$: Attracted to Goal.
    *   $F_{rep}$: Repelled by other humans (Personal Space Bubble).
    *   $F_{wall}$: Repelled by walls.
    *   Good for crowds, but ignores "Intent" (e.g., stopping to look at phone).

### 🔹 Part 2: Data-Driven Models (Social-LSTM)

Alahi et al. (CVPR 2016).
*   **Input:** Past trajectory of person $i$ and neighbors $j$.
*   **Social Pooling:** Sharing hidden states between nearby LSTMs to capture interaction.
*   **Output:** Bivariate Gaussian distribution of future position $(x, y)$.

### 🔹 Part 3: Intent Clues

Trajectory is not enough.
*   **Head Orientation:** Where are they looking? (Gaze).
*   **Body Pose:** Meaning to turn?
*   **Context:** Standing at a bus stop vs walking in a corridor.

---

## 💻 Implementation: LSTM Trajectory Predictor

We will implement a vanilla LSTM to predict the next 5 steps based on the past 5 steps.

### 🛠️ Project Structure
```text
day45_intent/
├── src/
│   ├── dataset.py
│   ├── lstm_model.py
│   └── train.py
└── visualize_pred.py
```

### 👨‍💻 Model (`src/lstm_model.py`)

```python
import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2) # Output: dx, dy
        
    def forward(self, x):
        # x shape: (Batch, SeqLen, 2)
        # Output: (Batch, SeqLen, Hidden)
        out, _ = self.lstm(x)
        
        # We only care about the last output for prediction sequence
        # But here we train Many-to-Many for simplicity (Predict next step at every step)
        pred = self.fc(out) 
        return pred

    def predict_future(self, past, n_steps=10):
        # Autoregressive inference
        preds = []
        current = past
        h = None
        
        for _ in range(n_steps):
            # Run LSTM on current
            out, h = self.lstm(current, h)
            next_step = self.fc(out[:, -1, :]).unsqueeze(1) # Predict next delta (Batch, 1, 2)
            
            preds.append(next_step)
            current = next_step # Feed prediction as input
            
        return torch.cat(preds, dim=1)
```

### 👨‍💻 Training Loop (`src/train.py`)

Toy Dataset: Synthesis of Sinusoidal paths (people walking in curves).

```python
import torch
import torch.optim as optim
import numpy as np
from src.lstm_model import TrajectoryLSTM

# Generate Data
def generate_paths(n_samples=1000, len_seq=20):
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, len_seq)
        x = t
        y = np.sin(t) + np.random.normal(0, 0.1, len_seq)
        path = np.stack([x, y], axis=1)
        # Convert to deltas (velocity) to make it stationary
        deltas = path[1:] - path[:-1]
        data.append(deltas)
    return torch.tensor(data, dtype=torch.float32)

def train():
    data = generate_paths()
    # Split: Input=0..9, Target=1..10
    X = data[:, :-1, :]
    Y = data[:, 1:, :] # Teacher forcing
    
    model = TrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), "model.pth")
    return model

if __name__ == "__main__":
    train()
```

### 👨‍💻 Visualization (`visualize_pred.py`)

Visualizing the "Fan" of possible futures compared to Ground Truth.

---

## 🔬 Lab Exercise: The "Chicken" Game

### 1. Lab Objectives
- Robot and Simulated Human move towards each other in a corridor.
- **Scenario A (Reactive):** Robot waits until distance < 1m, then stops. (Scary).
- **Scenario B (Predictive):** Robot predicts collision in 3s. Robot slows down and deviates *early* (5m away).
- **Result:** Scenario B is perceived as "Polite" and "Intelligent".

---

## 🚀 Project: "Perception-Aware Planner"

**Goal:** Modify the Costmap.
1.  **Input:** Predicted trajectory of human for $t=0..3s$.
2.  **Costmap Layer:** Inflate cost along the *predicted* path, not just current position.
    *   Current Pos: Cost = 254 (Lethal).
    *   Predicted Pos ($t=2$): Cost = 128 (Warning).
3.  **Result:** Global Planner naturally routes *behind* the walking person, anticipating their motion.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Freezing Robot"
*   **Symptom:** Human trajectory uncertainty covers the whole corridor. Costmap is blocked. Robot cannot find a path.
*   **Cause:** "The Frozen Robot Problem" (Trautman 2010). If we assume humans keep moving regardless of us, no path exists.
*   **Fix:** **Cooperative Planning**. Assume the human will *also* react to avoid the robot. (Interaction-aware models).

#### 2. "Drifting Predictions"
*   **Symptom:** LSTM predicts person will walk through a wall.
*   **Fix:** Add Scene Constraints (Scene-LSTM). Mask out predictions that lie in static obstacles (Occupancy Grid).

---

## ⚡ Optimization: Kalman Filter vs LSTM

*   **KF:** ~5 microseconds to predict. Linear only. Good for 0.5s.
*   **LSTM:** ~5 milliseconds. Nonlinear. Good for >2s.
*   **Strategy:** Use KF for immediate collision avoidance (Local Planner). Use LSTM for long-term route planning.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Constant Velocity works for walking. Does it work for stopping?
    *   **A:** No. It overshoots. Need a "Starting/Stopping" state machine.
2.  **Q:** What is "Social Force"?
    *   **A:** A virtual repulsive force field around humans that models their desire to maintain personal space.
3.  **Q:** How does Gaze detection help?
    *   **A:** If a person is looking at the robot, they have seen it. We can be aggressive. If they are looking away (phone), we must be extra cautious (beep).

### Challenge Task
> **Task:** Group Detection.
> 1. Two people walking side-by-side (Distance < 0.5m, Velocity matched).
> 2. Treat them as a single "Group Object".
> 3. Do not plan a path *between* them. That is rude.

---

## 📚 Further Reading
- **Social Force Model:** Helbing & Molnar.
- **Social LSTM:** Alahi et al. (CVPR 2016).
- **Trautman:** "Unfreezing the Robot".

---

**Day 45 Complete**
