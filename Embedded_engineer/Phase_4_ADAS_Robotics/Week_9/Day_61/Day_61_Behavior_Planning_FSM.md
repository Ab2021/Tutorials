# Day 61: Behavior Planning (Finite State Machines)
## Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation

---

> **📝 Day 61 Focus:**
> A* finds the path. MPC steers the wheel. But who decides *when* to change lanes? Who decides to stop for a pedestrian? This is **Behavior Planning**. Today, we build the "Captain" of the ship using **Finite State Machines (FSM)**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Structure** the Planning Hierarchy: Mission -> Behavior -> Local.
2.  **Design** a Finite State Machine (FSM) for highway driving.
3.  **Define** States (Lane Keep, Change Left/Right) and Transitions (Gap Available, Slow Car Ahead).
4.  **Implement** a Python FSM that makes high-level decisions based on sensor inputs.
5.  **Critique** the limitations of FSMs (Explosion of states) and preview Behavior Trees.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Logic:** If-Else statements.
-   **State Machines:** Basic concept (State, Event, Transition).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Planning Hierarchy

1.  **Mission Planner (Global):** "Go from New York to LA." (A* on Road Network). Output: List of Roads.
2.  **Behavior Planner (Decision):** "Overtake this slow truck." (FSM). Output: Target Lane / Maneuver.
3.  **Local Planner (Trajectory):** "Steer 5 degrees left to enter left lane." (MPC/Min Jerk). Output: Steering/Throttle.

### 🔹 Part 2: Finite State Machines (FSM)

A system that can be in exactly one of a finite number of states at any given time.
-   **States:** `LANE_KEEP`, `PREPARE_LANE_CHANGE_LEFT`, `LANE_CHANGE_LEFT`.
-   **Inputs:** Sensor data (Car ahead distance, Left lane gap).
-   **Transitions:** Logic rules.
    -   *If* `LANE_KEEP` *and* `Car Ahead < 30m` *and* `Left Lane Free`: Transition to `PREPARE_LANE_CHANGE_LEFT`.

### 🔹 Part 3: Cost Functions for Behavior

How do we decide? We calculate a **Cost** for each possible state.
-   $J_{lane\_keep} = w_1 \times \text{collision\_risk} + w_2 \times \text{efficiency}$.
-   $J_{change\_left} = w_1 \times \text{collision\_risk} + w_2 \times \text{efficiency} + w_3 \times \text{comfort}$.
-   Choose the state with the lowest cost.

---

## 💻 Implementation: Highway FSM

**Scenario:**
-   **Ego Car:** Driving on a 3-lane highway.
-   **Traffic:** Slow cars ahead.
-   **Goal:** Maintain high speed without crashing.

### 🛠️ Setup
Create `week9_day61` and `behavior_fsm.py`.

```bash
mkdir -p ~/ros2_ws/src/week9_day61
cd ~/ros2_ws/src/week9_day61
touch behavior_fsm.py
```

### 👨‍💻 Code: FSM Implementation

```python
import time
import random

# --- Constants ---
LANE_WIDTH = 4.0
SAFE_DISTANCE = 30.0 # meters
DESIRED_SPEED = 30.0 # m/s (~108 km/h)

class State:
    LANE_KEEP = "LANE_KEEP"
    PREP_LEFT = "PREP_LEFT"
    PREP_RIGHT = "PREP_RIGHT"
    CHANGE_LEFT = "CHANGE_LEFT"
    CHANGE_RIGHT = "CHANGE_RIGHT"

class Vehicle:
    def __init__(self, id, lane, s, v):
        self.id = id
        self.lane = lane # 0, 1, 2 (Left to Right)
        self.s = s # Longitudinal position
        self.v = v # Speed

class BehaviorPlanner:
    def __init__(self):
        self.state = State.LANE_KEEP
        self.lane = 1 # Start in middle lane
        self.s = 0.0
        self.v = 20.0
        
    def update(self, sensor_fusion):
        # sensor_fusion: List of Vehicle objects
        
        print(f"Current State: {self.state} | Lane: {self.lane} | S: {self.s:.1f} | V: {self.v:.1f}")
        
        # 1. Perception Analysis
        car_ahead = self.get_vehicle_ahead(sensor_fusion, self.lane)
        car_left = self.get_vehicle_ahead(sensor_fusion, self.lane - 1)
        car_right = self.get_vehicle_ahead(sensor_fusion, self.lane + 1)
        
        dist_ahead = car_ahead.s - self.s if car_ahead else 999.0
        
        # 2. State Transitions
        if self.state == State.LANE_KEEP:
            if dist_ahead < SAFE_DISTANCE:
                print(f"  -> Slow car ahead ({dist_ahead:.1f}m). Considering change.")
                # Check Left
                if self.lane > 0 and self.is_lane_safe(sensor_fusion, self.lane - 1):
                    self.state = State.PREP_LEFT
                # Check Right
                elif self.lane < 2 and self.is_lane_safe(sensor_fusion, self.lane + 1):
                    self.state = State.PREP_RIGHT
                else:
                    print("  -> Stuck. Braking.")
                    self.v -= 1.0 # Brake
            else:
                if self.v < DESIRED_SPEED:
                    self.v += 0.5 # Accelerate
                    
        elif self.state == State.PREP_LEFT:
            if self.is_lane_safe(sensor_fusion, self.lane - 1):
                print("  -> Left lane clear. Executing change.")
                self.state = State.CHANGE_LEFT
            else:
                print("  -> Left lane blocked. Abort.")
                self.state = State.LANE_KEEP
                
        elif self.state == State.PREP_RIGHT:
            if self.is_lane_safe(sensor_fusion, self.lane + 1):
                print("  -> Right lane clear. Executing change.")
                self.state = State.CHANGE_RIGHT
            else:
                print("  -> Right lane blocked. Abort.")
                self.state = State.LANE_KEEP
                
        elif self.state == State.CHANGE_LEFT:
            # Simulate maneuver (instant for FSM logic, usually takes time)
            self.lane -= 1
            self.state = State.LANE_KEEP
            
        elif self.state == State.CHANGE_RIGHT:
            self.lane += 1
            self.state = State.LANE_KEEP
            
        # Update Position
        self.s += self.v * 0.1 # dt = 0.1

    def get_vehicle_ahead(self, vehicles, lane):
        closest_dist = 999.0
        closest_veh = None
        for veh in vehicles:
            if veh.lane == lane and veh.s > self.s:
                dist = veh.s - self.s
                if dist < closest_dist:
                    closest_dist = dist
                    closest_veh = veh
        return closest_veh

    def is_lane_safe(self, vehicles, lane):
        # Check for cars in target lane within safety buffer
        for veh in vehicles:
            if veh.lane == lane:
                if abs(veh.s - self.s) < SAFE_DISTANCE:
                    return False
        return True

def main():
    ego = BehaviorPlanner()
    
    # Traffic
    traffic = [
        Vehicle(1, 1, 50.0, 15.0), # Slow car in middle lane
        Vehicle(2, 0, 40.0, 25.0), # Car in left lane (blocking initially)
        Vehicle(3, 2, 100.0, 20.0) # Car far away in right lane
    ]
    
    # Simulation Loop
    for t in range(100):
        print(f"\n--- Step {t} ---")
        
        # Update Traffic
        for veh in traffic:
            veh.s += veh.v * 0.1
            
        # Update Ego
        ego.update(traffic)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Overtake

### Lab Objectives
1.  Run the simulation.
2.  **Observation:**
    -   Ego starts in Lane 1. Approaches Vehicle 1 (Slow).
    -   State changes `LANE_KEEP` -> `PREP_LEFT`.
    -   But Vehicle 2 is blocking Left Lane. State reverts to `LANE_KEEP` (or waits).
    -   Once Vehicle 2 passes, Ego switches to `CHANGE_LEFT` -> `LANE_KEEP` (Lane 0).
    -   Ego accelerates to 30 m/s.
3.  **Experiment:**
    -   Move Vehicle 2 to Lane 2.
    -   *Result:* Ego should immediately switch left.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Flickering States
**Symptom:** Switching between `PREP_LEFT` and `LANE_KEEP` every cycle.
**Cause:** Hysteresis is missing. The threshold for entering a state should be harder than leaving it.
**Solution:** Add a timer. "Must be safe for 2 seconds before switching."

#### 2. Deadlock
**Symptom:** Stuck behind a slow car forever.
**Cause:** Logic doesn't check *all* options (e.g., didn't check Right lane if Left was blocked).
**Solution:** Implement a cost-based selector that evaluates all lanes simultaneously.

---

## ⚡ Optimization & Best Practices

### 1. Behavior Trees (BT)
FSMs get messy ($N^2$ transitions).
**Behavior Trees:**
-   Tree structure: Root -> Selector -> Sequence -> Action.
-   Modular and reusable.
-   Example: `Sequence(CheckSafety, ChangeLane)`.

### 2. Prediction
Don't just look at where cars are *now*.
-   Use a Constant Velocity model to predict where they will be in 5 seconds.
-   "Gap Acceptance": Will the gap still be there when I arrive?

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the output of the Behavior Planner?
    *   **A:** A high-level decision or constraint for the Local Planner. E.g., "Target Lane = 0", "Target Speed = 25 m/s".
2.  **Q:** Why do we need a `PREPARE` state?
    *   **A:** To signal intent (Turn Signal) and adjust speed to match the target lane gap before moving laterally.
3.  **Q:** How does this relate to MPC?
    *   **A:** The Behavior Planner tells the MPC *what* to do (Reference Path). The MPC figures out *how* to do it (Steering/Throttle).

### Challenge Task
**Task:** Cost Function Implementation.
1.  Instead of hardcoded `if/else`, calculate a cost for each lane.
2.  `Cost = 1000 * Collision + 10 * (DesiredSpeed - LaneSpeed) + 5 * LaneChangePenalty`.
3.  Pick the lane with min cost.

---

## 📚 Further Reading & References
-   [Udacity Self-Driving Car: Behavior Planning](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)
-   [Behavior Trees in Robotics](https://arxiv.org/abs/1709.00084)

---

**Day 61 Complete** | Phase 4: ADAS & Robotics Systems | Week 9: Path Planning & Navigation
