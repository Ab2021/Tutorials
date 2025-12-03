# Day 138: Behavior Planning (Finite State Machines)
## Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making

---

> **📝 Day 138 Focus:**
> MPC drives the car. A* finds the route. But who decides *when* to change lanes? **Behavior Planning** is the high-level logic. We use **Finite State Machines (FSM)** to manage states like "Lane Keep", "Prepare Lane Change", and "Execute Lane Change".

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** States, Transitions, and Guards in an FSM.
2.  **Design** a hierarchical FSM for Highway Driving.
3.  **Implement** state transition logic based on sensor inputs (e.g., "Car ahead is slow").
4.  **Simulate** a multi-lane traffic scenario.
5.  **Critique** the limitations of FSMs (Scalability).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Logic:** If-Else, State Diagrams.
-   **Day 136:** Frenet Coordinates ($s, d$).

### Hardware Requirements
-   **None:** Simulation based.

### Software Stack
-   **Python:** `enum`, `random`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Hierarchy

1.  **Route Planner:** A to B (Map level).
2.  **Behavior Planner:** "Change Lane", "Overtake" (Decision level).
3.  **Trajectory Planner:** "Move to $d=4$ in 5s" (Geometry level).
4.  **Controller:** Steering/Throttle (Actuation level).

### 🔹 Part 2: The States

Common states for Highway Driving:
-   **KL (Keep Lane):** Maintain speed and center.
-   **PLCL (Prep Lane Change Left):** Check blind spot, match speed of left lane.
-   **LCL (Lane Change Left):** Execute the maneuver.
-   **PLCR / LCR:** Same for Right.

### 🔹 Part 3: Cost Functions

How to decide? Calculate cost for each feasible state.
$$ J = w_v (v_{target} - v_{curr}) + w_d (dist_{obstacle}) + w_c (comfort) $$
-   If "Keep Lane" cost > "Lane Change" cost, then Transition.

---

## 💻 Implementation: Highway FSM

**Scenario:**
-   3 Lanes (0, 1, 2).
-   Ego Car in Lane 1.
-   Slow car ahead.
-   Task: Decide to switch to Lane 0 or 2.

### 🛠️ Setup
Create `week20_day138` and `behavior_fsm.py`.

```bash
mkdir -p ~/ros2_ws/src/week20_day138
cd ~/ros2_ws/src/week20_day138
touch behavior_fsm.py
```

### 👨‍💻 Code: State Machine

```python
import time
from enum import Enum
import random

class State(Enum):
    KEEP_LANE = 0
    PREP_LANE_CHANGE_LEFT = 1
    LANE_CHANGE_LEFT = 2
    PREP_LANE_CHANGE_RIGHT = 3
    LANE_CHANGE_RIGHT = 4

class Vehicle:
    def __init__(self, id, lane, s, v):
        self.id = id
        self.lane = lane # 0, 1, 2
        self.s = s # Longitudinal pos
        self.v = v # Speed
        self.state = State.KEEP_LANE
        self.target_lane = lane

    def update(self, dt, traffic):
        # Simple Physics
        self.s += self.v * dt
        
        # Behavior Logic
        self.choose_next_state(traffic)
        
        # Execute State (Mock)
        if self.state == State.LANE_CHANGE_LEFT:
            self.lane -= 0.1 # Gradual change
            if abs(self.lane - self.target_lane) < 0.1:
                self.lane = self.target_lane
                self.state = State.KEEP_LANE
                
        elif self.state == State.LANE_CHANGE_RIGHT:
            self.lane += 0.1
            if abs(self.lane - self.target_lane) < 0.1:
                self.lane = self.target_lane
                self.state = State.KEEP_LANE

    def choose_next_state(self, traffic):
        # 1. Get Kinematics of nearby cars
        car_ahead = self.get_car_ahead(traffic, self.lane)
        car_left = self.get_car_ahead(traffic, self.lane - 1)
        car_right = self.get_car_ahead(traffic, self.lane + 1)
        
        # 2. Calculate Costs
        cost_kl = 0
        if car_ahead:
            dist = car_ahead.s - self.s
            if dist < 20: cost_kl += 100 # Too close!
            if car_ahead.v < self.v: cost_kl += 50 # Slow car
            
        cost_lcl = 1000 # High default
        if self.lane > 0:
            cost_lcl = 10 # Base cost for effort
            if car_left:
                if car_left.s - self.s < 20: cost_lcl += 500 # Blocked
                
        cost_lcr = 1000
        if self.lane < 2:
            cost_lcr = 10
            if car_right:
                if car_right.s - self.s < 20: cost_lcr += 500
                
        # 3. Transition
        if self.state == State.KEEP_LANE:
            if cost_lcl < cost_kl and cost_lcl < cost_lcr:
                print(f"Deciding LCL (Cost: {cost_lcl} vs KL: {cost_kl})")
                self.state = State.LANE_CHANGE_LEFT
                self.target_lane = self.lane - 1
            elif cost_lcr < cost_kl:
                print(f"Deciding LCR (Cost: {cost_lcr} vs KL: {cost_kl})")
                self.state = State.LANE_CHANGE_RIGHT
                self.target_lane = self.lane + 1
            else:
                # Brake if stuck
                if cost_kl > 50: self.v *= 0.95

    def get_car_ahead(self, traffic, lane):
        closest = None
        min_dist = 9999
        for car in traffic:
            if car.id == self.id: continue
            if abs(car.lane - lane) < 0.5 and car.s > self.s:
                dist = car.s - self.s
                if dist < min_dist:
                    min_dist = dist
                    closest = car
        return closest

def main():
    # Ego Car
    ego = Vehicle(0, 1, 0, 30) # Lane 1, 30 m/s
    
    # Traffic
    traffic = [
        ego,
        Vehicle(1, 1, 50, 20), # Slow car ahead in Lane 1
        Vehicle(2, 0, 50, 35), # Fast car in Lane 0 (Blocking Left)
        Vehicle(3, 2, 100, 30) # Far car in Lane 2 (Free Right)
    ]
    
    print("Starting Simulation...")
    for t in range(50):
        print(f"T={t}: Ego Lane={ego.lane:.1f}, S={ego.s:.1f}, V={ego.v:.1f}, State={ego.state.name}")
        
        for car in traffic:
            car.update(0.1, traffic)
            
        time.sleep(0.1)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Overtake

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   Ego approaches Car 1 (Slow). `cost_kl` rises.
    -   Ego checks Left (Car 2 is there). `cost_lcl` is high.
    -   Ego checks Right (Lane 2 is free). `cost_lcr` is low.
    -   Ego switches to `LANE_CHANGE_RIGHT`.
3.  **Experiment:**
    -   Remove Car 2.
    -   **Result:** Ego might prefer Left (standard overtaking side) if costs are tuned to prefer Left.
    -   **Lesson:** Tuning weights ($w$) determines the "personality" of the car (Aggressive vs Conservative).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flickering
**Symptom:** Car switches Left, then immediately Right, then Left.
**Cause:** Costs are very close. Noise flips the decision.
**Solution:** Hysteresis. Add a "switching cost" (e.g., +5) to prevent changing decisions unless the benefit is significant.

#### 2. Blind Spot
**Symptom:** Car merges into another car.
**Cause:** `get_car_ahead` only checks *ahead*.
**Solution:** Must also check `get_car_behind`. If a car is approaching fast from behind in the target lane, do not merge.

---

## ⚡ Optimization & Best Practices

### 1. Behavior Trees (BT)
FSMs get messy ($N^2$ transitions).
-   **Behavior Trees:** Modular, hierarchical logic.
-   Nodes: Sequence, Selector, Condition, Action.
-   Easier to maintain and debug for complex behaviors.

### 2. POMDP (Partially Observable Markov Decision Process)
FSM assumes we know everything.
-   **POMDP:** Handles uncertainty (e.g., "Is that pedestrian going to cross?").
-   Probabilistic decision making.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is a "Guard" in FSM?
    *   **A:** The condition that must be met to transition from State A to State B (e.g., `speed < 10` to go to `STOP`).
2.  **Q:** Why do we need a "Prepare" state (PLCL)?
    *   **A:** To signal intent (Blinker) and adjust speed/position *before* committing to the dangerous maneuver.
3.  **Q:** How does Behavior Planning interact with Trajectory Planning?
    *   **A:** Behavior Planner outputs constraints (Target Lane, Target Speed, Time). Trajectory Planner generates the curve that satisfies them.

### Challenge Task
**Task:** Emergency Stop.
1.  Add a state `EMERGENCY_STOP`.
2.  Transition if `dist < 5m`.
3.  Action: `v = 0` (Max braking).
4.  Simulate a car cutting in front abruptly.

---

## 📚 Further Reading & References
-   [Behavior Trees in Robotics](https://arxiv.org/abs/1709.00084)
-   [Udacity Highway Driving Project](https://github.com/udacity/CarND-Path-Planning-Project)

---

**Day 138 Complete** | Phase 4: ADAS & Robotics Systems | Week 20: Path Planning & Decision Making
