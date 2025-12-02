# Day 31: Behavior Planning (State Machines)
## Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making

---

> **📝 Day 31 Focus:**
> We can plan a path (Global) and generate a trajectory (Local). But *should* we change lanes? *Should* we stop for that pedestrian? This is **Behavior Planning**. Today, we give the robot a brain using **Finite State Machines (FSM)** to handle high-level decision making.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the role of the Behavior Planner in the ADAS stack.
2.  **Design** a Finite State Machine (FSM) for highway driving.
3.  **Implement** States, Transitions, and Guard Conditions in Python.
4.  **Analyze** the limitations of simple FSMs (State Explosion).
5.  **Simulate** a scenario where the car decides to overtake a slow vehicle.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Logic:** Boolean algebra (AND, OR, NOT).
-   **Programming:** Classes and Objects.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `statemachine` (optional), or pure Python classes.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Decision Hierarchy

1.  **Mission Planner:** "Go from Home to Office." (Route).
2.  **Behavior Planner:** "Overtake this slow truck." (Decision).
3.  **Local Planner:** "Steer 5 degrees left, accelerate to 50mph." (Trajectory).
4.  **Controller:** "Apply 12V to steering motor." (Actuation).

### 🔹 Part 2: Finite State Machines (FSM)

An FSM consists of:
-   **States:** A finite set of conditions (e.g., `LaneKeep`, `LaneChangeLeft`, `Stop`).
-   **Transitions:** Rules to move from one state to another.
-   **Inputs:** Sensor data (e.g., `CarAheadDistance`, `Speed`).

#### 2.1 Example: Adaptive Cruise Control (ACC)
-   **State: Cruise**
    -   If `Distance < SafeDist`: Transition to **Follow**.
-   **State: Follow**
    -   If `Distance > SafeDist`: Transition to **Cruise**.
    -   If `Speed < 0`: Transition to **Stop**.

### 🔹 Part 3: Handling Complexity

Simple FSMs get messy quickly ($N$ states $\to N^2$ transitions).
**Hierarchical FSM (HFSM):**
-   **Superstate:** `DriveHighway`
    -   **Substates:** `LaneKeep`, `LaneChange`.
-   **Superstate:** `Park`
    -   **Substates:** `SearchSpot`, `Reverse`, `Align`.

This encapsulates logic and keeps the diagram clean.

---

## 💻 Implementation: Highway FSM

We will implement a Python class `HighwayFSM` that manages the behavior of an autonomous car.

### 🛠️ Setup
Create `week5_day31` and `behavior_fsm.py`.

```bash
mkdir -p ~/ros2_ws/src/week5_day31
cd ~/ros2_ws/src/week5_day31
touch behavior_fsm.py
```

### 👨‍💻 Code: State Machine Implementation

```python
import time
import random

# Enum for States
class State:
    LANE_KEEP = "LANE_KEEP"
    PREP_LANE_CHANGE_LEFT = "PREP_LANE_CHANGE_LEFT"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    PREP_LANE_CHANGE_RIGHT = "PREP_LANE_CHANGE_RIGHT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"

class Vehicle:
    def __init__(self, lane_id=1, speed=20):
        self.lane_id = lane_id # 0=Left, 1=Center, 2=Right
        self.speed = speed
        self.state = State.LANE_KEEP
        
    def update(self, sensor_data):
        # sensor_data: { 'car_ahead_dist': float, 'left_lane_free': bool, 'right_lane_free': bool }
        
        print(f"\n[Current State: {self.state}] | Lane: {self.lane_id} | Speed: {self.speed}")
        
        if self.state == State.LANE_KEEP:
            self.handle_lane_keep(sensor_data)
            
        elif self.state == State.PREP_LANE_CHANGE_LEFT:
            self.handle_prep_left(sensor_data)
            
        elif self.state == State.LANE_CHANGE_LEFT:
            self.handle_change_left()
            
        # ... (Right lane logic omitted for brevity, symmetric to Left)

    def handle_lane_keep(self, data):
        dist = data['car_ahead_dist']
        
        if dist > 30:
            print("Road clear. Cruising.")
            self.speed = 30
        else:
            print(f"Car ahead ({dist}m). Slowing down.")
            self.speed = 20
            
            # Decision: Overtake?
            if data['left_lane_free'] and self.lane_id > 0:
                print("Deciding to overtake Left.")
                self.transition(State.PREP_LANE_CHANGE_LEFT)
            else:
                print("Cannot overtake. Following.")

    def handle_prep_left(self, data):
        # Check blind spot, turn on blinker
        print("Blinker ON (Left). Checking blind spot...")
        
        if data['left_lane_free']:
            print("Blind spot clear.")
            self.transition(State.LANE_CHANGE_LEFT)
        else:
            print("Car in blind spot! Abort.")
            self.transition(State.LANE_KEEP)

    def handle_change_left(self):
        # Execute trajectory
        print("Executing Lane Change Trajectory...")
        # Simulate time passing
        self.lane_id -= 1
        print(f"Lane Change Complete. Now in Lane {self.lane_id}.")
        self.transition(State.LANE_KEEP)

    def transition(self, new_state):
        print(f"Transitioning: {self.state} -> {new_state}")
        self.state = new_state

def run_simulation():
    ego_car = Vehicle(lane_id=1) # Start in Center Lane
    
    # Simulate a scenario
    # 1. Clear road
    # 2. Car appears ahead
    # 3. Left lane free -> Overtake
    
    scenarios = [
        {'car_ahead_dist': 100, 'left_lane_free': True},
        {'car_ahead_dist': 100, 'left_lane_free': True},
        {'car_ahead_dist': 20, 'left_lane_free': True}, # Car ahead!
        {'car_ahead_dist': 20, 'left_lane_free': True}, # Prep
        {'car_ahead_dist': 20, 'left_lane_free': True}, # Change
        {'car_ahead_dist': 100, 'left_lane_free': True}, # Done
    ]
    
    for i, data in enumerate(scenarios):
        print(f"--- Step {i} ---")
        ego_car.update(data)
        time.sleep(1)

if __name__ == "__main__":
    run_simulation()
```

---

## 🔬 Lab Exercise: Hysteresis

### Lab Objectives
1.  Run the script.
2.  **Problem:** If `car_ahead_dist` fluctuates between 29.9 and 30.1, the car might jitter between `Cruise` and `Follow`.
3.  **Solution:** Add **Hysteresis**.
    -   Transition to `Follow` if `dist < 30`.
    -   Transition to `Cruise` only if `dist > 35`.
4.  **Task:** Implement this logic in `handle_lane_keep`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Deadlock
**Symptom:** Car gets stuck in a state (e.g., `PrepLaneChange`) forever.
**Cause:** Missing transition condition (e.g., what if the lane never becomes free?).
**Solution:** Add a timeout. If `Prep` takes > 10s, abort to `LaneKeep`.

#### 2. Oscillation
**Symptom:** Car changes Left, then immediately changes Right.
**Cause:** Cost function for "Best Lane" is unstable.
**Solution:** Add a cost penalty for recent lane changes.

---

## ⚡ Optimization & Best Practices

### 1. Behavior Trees (BT)
FSMs are hard to maintain for complex behaviors.
**Behavior Trees** are more modular.
-   **Nodes:** Sequence, Selector, Action.
-   **Logic:** "Try to Overtake. If fail, Try to Follow. If fail, Emergency Stop."
-   Used by Navigation2 (ROS 2).

### 2. Cost-Based Planning
Instead of hard rules (If A then B), generate multiple behaviors (Keep, Left, Right) and score them.
-   $Cost = w_1 \times \text{Safety} + w_2 \times \text{Speed} + w_3 \times \text{Comfort}$.
-   Pick the lowest cost. This is what the Frenet Planner (Day 30) does implicitly.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is a "Guard Condition"?
    *   **A:** A boolean check that must be true to allow a transition (e.g., `IsLaneFree == True`).
2.  **Q:** Why do we need a `PrepLaneChange` state?
    *   **A:** To signal intent (blinker) and check safety (blind spot) *before* committing to the trajectory.
3.  **Q:** What happens if the FSM has no transition for the current input?
    *   **A:** Undefined behavior (Crash). FSMs must be exhaustive.

### Challenge Task
**Task:** Emergency Stop.
1.  Add a global check in `update()`:
    -   If `sensor_data['collision_imminent']` is True:
    -   Override everything and transition to `EMERGENCY_STOP`.
2.  In `EMERGENCY_STOP`, set speed to 0 and refuse to transition out until reset.

---

## 📚 Further Reading & References
-   [Udacity Self-Driving Car Nanodegree - Behavior Planning](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)
-   [Behavior Trees in Robotics (Colledanchise)](https://arxiv.org/abs/1709.00084)

---

**Day 31 Complete** | Phase 4: ADAS & Robotics Systems | Week 5: Path Planning & Decision Making
