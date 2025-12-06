# Day 152: Behavior Prediction (Trajectory Prediction)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 22: Autonomous Driving Stack

---

> **📝 Content Creator Instructions:**
> Don't hit where they are. Hit where they will be.
> - **Focus:** Prediction Horizon (3-5s), Physics-based vs Maneuver-based prediction, Map-Aware Prediction (Target Lanes), and Multi-Modal Trajectories.
> - **Code:** A Python script `traj_predictor.py` that takes a tracked vehicle state and a Lane Map, and outputs 3 probabilistic trajectories (Keep Lane, Turn Left, Turn Right).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** Short-term (Physics) vs Long-term (Intent) prediction.
2.  **Utilize** High-Definition (HD) Maps to constrain predictions (Cars usually follow lanes).
3.  **Generate** Multi-Modal predictions: A car approaching an intersection might go Straight (60%) or Turn Right (40%).
4.  **Implement** a Lane-Based Trajectory Generator.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib
```

### Prior Knowledge
- Kalman Filters (Week 4).
- Frenet Coordinates ($s, d$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Prediction Horizon

*   **0-1s:** Constant Velocity (CV) / Constant Acceleration (CA). Physics dominates.
*   **1-3s:** Maneuver Intention. "Are they changing lanes?"
*   **3-8s:** Interaction / Route. "Are they exiting the highway?"

### 🔹 Part 2: Map-Aware Prediction

Cars don't move randomly. They follow Topology.
1.  **Snap to Lane:** Find nearest centerline.
2.  **Project:** Extrapolate along centerline arc length ($s$).
3.  **Modes:**
    *   **Keep Lane:** Follow current center.
    *   **Cut-in:** Move from Lane A to Lane B.

### 🔹 Part 3: Representation

Output is not one path, but a distribution.
*   **Gaussian:** $\mathcal{N}(\mu_t, \Sigma_t)$ for each timestep.
*   **Trajectory Set:** $K$ distinct paths with weights $w_k$.

---

## 💻 Implementation: The Oracle

We simulate a car on a highway with 3 lanes. We predict its future 3 seconds.

### 🛠️ Project Structure
```text
day152_prediction/
├── src/
│   ├── traj_predictor.py
└── output/
    ├── prediction_viz.png
```

### 👨‍💻 Trajectory Predictor (`src/traj_predictor.py`)

```python
import numpy as np
import matplotlib.pyplot as plt

class Lane:
    def __init__(self, id, y_center):
        self.id = id
        self.y = y_center # Assume straight highway for simplicity
        
    def get_coords(self, x_start, x_end):
        x = np.linspace(x_start, x_end, 100)
        y = np.ones_like(x) * self.y
        return x, y

class Vehicle:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy # Lateral velocity hints at intent
        
    def get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy])

class Predictor:
    def __init__(self, lanes):
        self.lanes = lanes
        self.horizon = 3.0 # seconds
        self.dt = 0.1
        self.steps = int(self.horizon / self.dt)
        
    def predict(self, vehicle):
        predictions = [] # List of (Trajectory, Probability, Label)
        
        # 1. Physics Prediction (Baseline)
        # x = x + vx*t, y = y + vy*t
        t = np.linspace(0, self.horizon, self.steps)
        path_phys_x = vehicle.x + vehicle.vx * t
        path_phys_y = vehicle.y + vehicle.vy * t
        predictions.append( (np.column_stack((path_phys_x, path_phys_y)), 0.1, "Physics") )
        
        # 2. Map-Based Prediction
        # Find current lane
        nearest_lane = min(self.lanes, key=lambda l: abs(l.y - vehicle.y))
        dist_to_center = vehicle.y - nearest_lane.y
        
        # Check Lateral Velocity
        # If vy > 0.5, likely changing Left (if lane exists)
        # If vy < -0.5, likely changing Right
        
        # Mode A: Keep Lane
        # Smoothly return to center of nearest_lane
        # Simple Logic: y(t) = lane_y + (current_y - lane_y) * exp(-k*t)
        path_kl_x = vehicle.x + vehicle.vx * t
        path_kl_y = nearest_lane.y + dist_to_center * np.exp(-1.0 * t)
        
        prob_kl = 0.8
        if abs(vehicle.vy) > 0.5: prob_kl = 0.2 # Dropped prob if moving sideways
        
        predictions.append( (np.column_stack((path_kl_x, path_kl_y)), prob_kl, f"Keep Lane {nearest_lane.id}") )
        
        # Mode B: Lane Change
        # If lateral velocity is significant, predict smooth transition to next lane
        target_lane = None
        if vehicle.vy > 0.5: # Moving Left (Positive Y)
            # Find lane above
            candidates = [l for l in self.lanes if l.y > nearest_lane.y]
            if candidates: target_lane = candidates[0]
                
        elif vehicle.vy < -0.5: # Moving Right
            candidates = [l for l in self.lanes if l.y < nearest_lane.y]
            if candidates: target_lane = candidates[0]
            
        if target_lane:
            # Logistic curve transition to target_lane.y
            # For simplicity, linear interpolation of lateral shift based on current vy
            # Or better: Minimum Jerk Trajectory to target
            
            # Simple simulation: continue lateral vel until target reached, then straighten
            time_to_reach = abs(target_lane.y - vehicle.y) / abs(vehicle.vy)
            
            path_lc_y = []
            current_y = vehicle.y
            for _ in range(self.steps):
                if hasattr(target_lane, 'y'): # valid
                    dy = target_lane.y - current_y
                    if abs(dy) < 0.1: 
                        sim_vy = 0 # Arrived
                    else:
                        sim_vy = vehicle.vy # Constant closure
                    
                    current_y += sim_vy * self.dt
                    path_lc_y.append(current_y)
                    
            path_lc_x = vehicle.x + vehicle.vx * t
            
            predictions.append( (np.column_stack((path_lc_x, path_lc_y)), 0.7, f"Switch to {target_lane.id}") )
            
        return predictions

def main():
    # Setup Highway
    # Lane 0: y=0, Lane 1: y=4, Lane 2: y=8
    lanes = [Lane(0, 0), Lane(1, 4), Lane(2, 8)]
    
    # 1. Car Driving Straight in Lane 1
    car_straight = Vehicle(x=0, y=4.2, vx=20, vy=0.1)
    
    # 2. Car Changing Lane (1 -> 2)
    car_changing = Vehicle(x=0, y=5.0, vx=20, vy=1.5) # Fast lateral move
    
    predictor = Predictor(lanes)
    
    # Predict
    preds_s = predictor.predict(car_straight)
    preds_c = predictor.predict(car_changing)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    # Draw Lanes
    for l in lanes:
        lx, ly = l.get_coords(-10, 80)
        plt.plot(lx, ly, 'k--', alpha=0.5)
        plt.text(75, l.y + 0.5, f"Lane {l.id}")
        
    def plot_veh(preds, car, label):
        plt.plot(car.x, car.y, 'ko', markersize=8)
        for traj, prob, name in preds:
            alpha = prob if prob < 1.0 else 1.0
            width = 3 if prob > 0.5 else 1
            if "Switch" in name and prob > 0.5: color='r'
            elif "Keep" in name and prob > 0.5: color='g'
            else: color='gray'
            
            plt.plot(traj[:,0], traj[:,1], color=color, linewidth=width, alpha=0.7, label=f"{label}: {name} ({prob:.1f})")
            
    plot_veh(preds_s, car_straight, "Car A")
    plot_veh(preds_c, car_changing, "Car B")
    
    plt.title("Multi-Modal Trajectory Prediction (3s Horizon)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.ylim(-2, 12)
    plt.grid()
    plt.legend()
    plt.savefig("output/prediction_viz.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Intersection"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:**
    *   Car A (Low $v_y$) predicts "Keep Lane 1".
    *   Car B (High $v_y$) predicts "Switch to Lane 2".
- **Modify:** Add Intersection Logic.
- **Scenario:** Map has branching lanes. Lane 1 splits into Lane 1A (Straight) and Lane 1B (Right Turn).
- **Task:** Create a prediction that branches based on Blinkers (if detectable) or historical deceleration. If car slows down near junction, Prob(Turn) increases.

---

## 🚀 Project: "Social LSTM"

**Goal:** Predict pedestrians.
1.  **Architecture:** LSTM for each person. Pooling layer connects them.
2.  **Concept:** People avoid each other.
3.  **Result:** If Person A walks towards Person B, the network predicts they will curve to avoid collision. Standard constant velocity predicts a crash.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Waggling Prediction"
*   **Cause:** Noisy $v_y$. Small steering correction looks like a lane change.
*   **Fix:** Smoothing or Hysteresis. Require sustained lateral velocity > threshold for 0.5s.

#### 2. "Ghosting"
*   **Cause:** Predicting a path through a wall/obstacle.
*   **Fix:** Post-process predictions with a Static Map Check. If Traj hits a curb, Probability = 0.

---

## ⚡ Optimization: VectorNet

State-of-the-art Prediction (Waymo).
*   **Input:** Vectorized Map (Polylines) + Agent Trajectories.
*   **Graph Neural Net:** Processes context and interactions efficiently.
*   **Output:** Multi-modal trajectories.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just one prediction?
    *   **A:** Because the future is uncertain. If you predict "Definitely Straight" and they turn, you might crash. Safe planning requires accounting for *all* likely futures.
2.  **Q:** Inputs to prediction?
    *   **A:** Current State ($x,y,v,a,yaw$), Map (Polylines, Traffic Lights), Head Pose (Driver looking left?), Blinkers.
3.  **Q:** Target-based vs Path-based?
    *   **A:** Target-based predicts a Goal ($x,y$) and fits a curve. Path-based predicts $(x_t, y_t)$ steps. Target is better for long term.

### Challenge Task
> **Task:** Frenet Conversion.
> 1. Convert inputs to Frenet Frame $(s, d)$.
> 2. Predict $d(t)$ (Lateral offset).
> 3. Much easier to predict "Keep Lane" ($d \approx 0$) in Frenet than in Cartesian xy.

---

## 📚 Further Reading
- **VectorNet:** "Encoding HD Maps and Agent Dynamics from Vectorized Representation".
- **TNT:** "Target-driven Trajectory Prediction".

---

**Day 152 Complete**
