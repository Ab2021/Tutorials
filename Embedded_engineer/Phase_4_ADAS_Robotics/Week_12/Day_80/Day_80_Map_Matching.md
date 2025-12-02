# Day 80: Map Matching Algorithms
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 80 Focus:**
> Your GPS says you are in the middle of a building. But you know you are on a road. **Map Matching** is the art of correcting noisy GPS data by using the constraints of the road network. It's not just "snap to nearest"; it's about finding the most likely *path*.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Identify** the failure modes of "Nearest Neighbor" matching (Parallel roads, Intersections).
2.  **Formulate** Map Matching as a Hidden Markov Model (HMM) problem.
3.  **Define** Emission Probabilities (Distance) and Transition Probabilities (Topology).
4.  **Implement** the Viterbi Algorithm to find the optimal sequence of road segments.
5.  **Visualize** the raw GPS trace vs the matched road path.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Probability:** Bayes' Theorem.
-   **Graph Theory:** Nodes and Edges.
-   **Dynamic Programming:** Viterbi Algorithm.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `networkx` (for graph topology).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Problem with "Snap to Nearest"

Imagine a highway (Road A) running parallel to a service road (Road B), 10 meters apart.
-   GPS error is 5-10 meters.
-   If you just snap to the nearest line, your position will jump back and forth between Highway and Service Road.
-   **Solution:** Use history. If you were on the Highway 1 second ago, you are probably still on the Highway (unless there is an exit).

### 🔹 Part 2: Hidden Markov Model (HMM)

-   **Hidden States ($S$):** The actual road segment you are on.
-   **Observations ($O$):** The noisy GPS points.
-   **Emission Probability $P(O_t | S_t)$:** How likely is this GPS point if I am on Road $S$?
    -   Based on distance: $P \propto \exp(-dist^2 / 2\sigma^2)$.
-   **Transition Probability $P(S_t | S_{t-1})$:** How likely is it to move from Road $A$ to Road $B$?
    -   If connected: High probability.
    -   If disconnected: Zero probability.
    -   Also considers speed and heading consistency.

### 🔹 Part 3: The Viterbi Algorithm

Finds the sequence of states $S_{1:T}$ that maximizes the joint probability:
$$ \text{argmax} \prod P(O_t | S_t) P(S_t | S_{t-1}) $$
-   Dynamic Programming approach.
-   Complexity: $O(T \times N^2)$ where $N$ is number of candidate roads near each point.

---

## 💻 Implementation: HMM Map Matcher

**Scenario:**
-   **Map:** Two parallel roads (Y=0 and Y=10).
-   **GPS:** Noisy trace moving along Y=0.
-   **Goal:** Correctly match to Road 0, avoiding jumps to Road 1.

### 🛠️ Setup
Create `week12_day80` and `map_matcher.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day80
cd ~/ros2_ws/src/week12_day80
touch map_matcher.py
```

### 👨‍💻 Code: Viterbi Map Matcher

```python
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Constants ---
SIGMA_GPS = 5.0 # GPS Noise std dev (meters)
BETA = 0.1 # Transition probability decay (1/meters)

class RoadSegment:
    def __init__(self, id, x1, y1, x2, y2):
        self.id = id
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        
    def distance(self, p):
        # Point to Segment distance
        d = self.p2 - self.p1
        if np.dot(d, d) == 0: return np.linalg.norm(p - self.p1)
        t = np.dot(p - self.p1, d) / np.dot(d, d)
        t = max(0, min(1, t))
        proj = self.p1 + t * d
        return np.linalg.norm(p - proj)

    def project(self, p):
        d = self.p2 - self.p1
        t = np.dot(p - self.p1, d) / np.dot(d, d)
        t = max(0, min(1, t))
        return self.p1 + t * d

class HMMMatcher:
    def __init__(self, roads):
        self.roads = roads
        
    def match(self, gps_trace):
        # gps_trace: List of (x, y)
        T = len(gps_trace)
        N = len(self.roads)
        
        # Viterbi Tables
        # V[t, i]: Max probability of ending at state i at time t
        V = np.zeros((T, N))
        # Path[t, i]: The previous state that led to V[t, i]
        Path = np.zeros((T, N), dtype=int)
        
        # 1. Initialization (t=0)
        for i in range(N):
            dist = self.roads[i].distance(gps_trace[0])
            emission = self.gaussian(dist, SIGMA_GPS)
            V[0, i] = emission # Assume uniform prior
            
        # 2. Recursion
        for t in range(1, T):
            for j in range(N): # Current State
                max_prob = -1.0
                best_prev = -1
                
                dist = self.roads[j].distance(gps_trace[t])
                emission = self.gaussian(dist, SIGMA_GPS)
                
                for i in range(N): # Previous State
                    # Transition Prob
                    # Simplified: Based on distance between projections
                    # Real HMM uses graph connectivity (Dijkstra)
                    
                    # Project points
                    p_prev = self.roads[i].project(gps_trace[t-1])
                    p_curr = self.roads[j].project(gps_trace[t])
                    
                    # Distance traveled on map
                    dist_map = np.linalg.norm(p_curr - p_prev)
                    
                    # Distance traveled by GPS (Great Circle)
                    dist_gps = np.linalg.norm(gps_trace[t] - gps_trace[t-1])
                    
                    # Transition prob: Difference in distance
                    # P(j|i) ~ exp(-|d_map - d_gps|)
                    transition = math.exp(-abs(dist_map - dist_gps) * BETA)
                    
                    prob = V[t-1, i] * transition * emission
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = i
                        
                V[t, j] = max_prob
                Path[t, j] = best_prev
                
        # 3. Termination
        best_last_state = np.argmax(V[T-1, :])
        
        # 4. Backtracking
        states = [best_last_state]
        for t in range(T-1, 0, -1):
            prev = Path[t, states[-1]]
            states.append(prev)
            
        return states[::-1] # Reverse

    def gaussian(self, x, sigma):
        return math.exp(-0.5 * (x / sigma)**2)

def main():
    # Map: Parallel Roads
    # Road 0: y=0 (Highway)
    # Road 1: y=15 (Service Road) - 15m away
    roads = [
        RoadSegment(0, 0, 0, 100, 0),
        RoadSegment(1, 0, 15, 100, 15)
    ]
    
    matcher = HMMMatcher(roads)
    
    # Simulation: Driving on Road 0
    gps_trace = []
    true_path = []
    
    for x in range(0, 100, 5):
        # True pos
        true_path.append([x, 0])
        
        # GPS Noise (Large Y error)
        # Sometimes jumps closer to Road 1
        noise_y = np.random.normal(0, 5.0)
        # Add a bias towards Road 1 to test robustness
        if 30 < x < 70: noise_y += 6.0 
        
        gps_trace.append(np.array([x, noise_y]))
        
    # Run Matcher
    matched_indices = matcher.match(gps_trace)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    # Draw Roads
    for r in roads:
        plt.plot([r.p1[0], r.p2[0]], [r.p1[1], r.p2[1]], 'k-', linewidth=2, label=f'Road {r.id}')
        
    # Draw GPS
    gps_x = [p[0] for p in gps_trace]
    gps_y = [p[1] for p in gps_trace]
    plt.plot(gps_x, gps_y, 'g.--', label='GPS Trace')
    
    # Draw Matched
    matched_x = []
    matched_y = []
    for i, idx in enumerate(matched_indices):
        # Project GPS onto matched road
        proj = roads[idx].project(gps_trace[i])
        matched_x.append(proj[0])
        matched_y.append(proj[1])
        
    plt.plot(matched_x, matched_y, 'r.-', label='HMM Match')
    
    plt.legend()
    plt.ylim(-20, 40)
    plt.title("HMM Map Matching")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Parallel Road

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   GPS trace (Green) wanders between $y=-5$ and $y=10$.
    -   Between $x=30$ and $x=70$, the GPS is actually closer to Road 1 ($y=15$) than Road 0 ($y=0$) due to the bias.
    -   **Nearest Neighbor:** Would snap to Road 1.
    -   **HMM (Red):** Stays on Road 0!
    -   *Why?* Because switching to Road 1 would require a "teleportation" (Transition Probability is low because the graph isn't connected or the speed doesn't match).
3.  **Experiment:**
    -   Increase `SIGMA_GPS`. If noise is huge, HMM might eventually jump.
    -   Decrease `BETA`. If transition penalty is low, it jumps easier.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Numerical Underflow
**Symptom:** Probabilities become 0.0.
**Cause:** Multiplying many small probabilities ($10^{-5} \times 10^{-5} \dots$).
**Solution:** Work in **Log Space**. Add log-probabilities instead of multiplying. $\text{argmax} \sum (\log P_{emit} + \log P_{trans})$.

#### 2. Lag
**Symptom:** Matcher is one step behind.
**Cause:** Viterbi is an offline algorithm (needs future data).
**Solution:** For real-time, use a **Particle Filter** or a sliding window Viterbi (latency of 2-3 seconds).

---

## ⚡ Optimization & Best Practices

### 1. Candidate Selection
Don't evaluate *all* roads for every point.
-   Use a Quadtree (Day 79) to find roads within 50m radius.
-   Only run Viterbi on these $N$ candidates.

### 2. Graph Routing
For Transition Probability, use $A^*$ distance on the road graph.
-   If Road A and Road B are close geometrically but require a 5km detour to drive between them (e.g., overpass), $P(A \to B)$ should be 0.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does Nearest Neighbor fail on parallel roads?
    *   **A:** It ignores history and topology. It only looks at the current instant.
2.  **Q:** What is the "Emission Probability"?
    *   **A:** The likelihood of observing a GPS point given a road segment (usually Gaussian of distance).
3.  **Q:** Can HMM handle U-turns?
    *   **A:** Yes, if the graph topology allows it. But it might penalize heading changes if heading consistency is part of the cost.

### Challenge Task
**Task:** Heading Weight.
1.  Add heading to the GPS trace.
2.  Add a term to Emission Probability: $\exp(-|\theta_{gps} - \theta_{road}|)$.
3.  This helps distinguish roads going in opposite directions.

---

## 📚 Further Reading & References
-   [Hidden Markov Map Matching (Newson & Krumm)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-through-noise-and-sparseness/)
-   [Valhalla Map Matching Engine](https://github.com/valhalla/valhalla)

---

**Day 80 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
