# Day 75: Behavior Trees (BT.CPP)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> State Machines are spaghetti. Behavior Trees are modular.
> - **Focus:** The BT.CPP library, XML structure, Custom Action Nodes (C++), and designing a complex Navigation Recovery Strategy.
> - **Code:** A custom BT Node `CheckBattery` that forces the robot to return home if voltage < 20%.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Read and Write** Behavior Tree XMLs for Nav2.
2.  **Explain** Control Flow Nodes: Sequence `->`, Fallback `?`, Parallell, and Decorators (Inverter, Retry).
3.  **Create** a custom C++ BT Action Node and register it with `bt_navigator`.
4.  **Design** a "Patrol with Battery Monitor" behavior.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-behaviortree-cpp-v3
sudo apt install gromit # Visualization (Groot) - optional
```

### Prior Knowledge
- Finite State Machines (FSM).
- C++ Factory Pattern.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Behavior Trees?

*   **FSM Problem:** $N$ states have $N^2$ transitions. Adding a state requires modifying all others.
*   **BT Solution:** Tree structure.
    *   **Tick:** Signal travels from Root down.
    *   **Return Status:** SUCCESS, FAILURE, RUNNING.
    *   **Modularity:** Sub-trees can be reused (e.g., "Open Door" sub-tree).

### 🔹 Part 2: Node Types

1.  **Sequence (`->`):** Run children in order. If one FAILS, return FAILURE. (AND logic).
2.  **Fallback (`?`):** Run children in order. If one SUCCEEDS, return SUCCESS. (OR logic / Retry).
    *   *Usage:* "Try Plan A. If fail, Try Plan B."
3.  **Action:** Leaf node. Does work (Move, Speak).
4.  **Condition:** Leaf node. Check state (IsBatteryLow?). Returns SUCCESS/FAILURE immediately.
5.  **Decorator:** Modifies child result. `Inverter`, `RetryUntilSuccessful`.

### 🔹 Part 3: Nav2 Specifics

Nav2 uses `BehaviorTree.CPP V3`.
*   `ComputePathToPose` (Action)
*   `FollowPath` (Action)
*   `PipelineSequence`: Runs children, but if child 2 fails, child 1 is re-ticked. (Reactive).

---

## 💻 Implementation: Custom Battery Check Node

We will write a C++ Condition Node.

### 🛠️ Project Structure
```text
day75_bt/
├── include/day75_bt/
│   └── check_battery.hpp
├── src/
│   └── check_battery.cpp
├── behavior_trees/
│   └── patrol_w_battery.xml
└── CMakeLists.txt
```

### 👨‍💻 Header (`check_battery.hpp`)

```cpp
#ifndef CHECK_BATTERY_HPP
#define CHECK_BATTERY_HPP

#include "behaviortree_cpp_v3/condition_node.h"
#include "rclcpp/rclcpp.hpp"

namespace day75_bt {

class CheckBattery : public BT::ConditionNode {
public:
  CheckBattery(const std::string& name, const BT::NodeConfiguration& config);

  static BT::PortsList providedPorts() {
    return { BT::InputPort<float>("min_voltage") };
  }

  BT::NodeStatus tick() override;
  
private:
  rclcpp::Node::SharedPtr node_;
};

}
#endif
```

### 👨‍💻 Implementation (`check_battery.cpp`)

```cpp
#include "day75_bt/check_battery.hpp"

namespace day75_bt {

CheckBattery::CheckBattery(const std::string& name, const BT::NodeConfiguration& config)
  : BT::ConditionNode(name, config)
{
    // Need a way to get ROS node. usually passed via blackboard or global singleton in Nav2 plugins
    // For simplicity, we assume we subscribe to /battery_state
}

BT::NodeStatus CheckBattery::tick() {
    float min_voltage;
    if (!getInput("min_voltage", min_voltage)) {
        throw BT::RuntimeError("Missing parameter [min_voltage]");
    }

    // Mock Battery Reading
    // In real Nav2 plugin, use node_->create_subscription...
    float current_voltage = 24.0; // Retrieve from ROS topic
    
    if (current_voltage > min_voltage) {
        return BT::NodeStatus::SUCCESS;
    } else {
        return BT::NodeStatus::FAILURE; 
    }
}

}
```

### 👨‍💻 Behavior Tree XML (`patrol_w_battery.xml`)

Logic:
1.  Check Battery. If Low (Fail), Go Home.
2.  Else, Patrol.

```xml
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Fallback name="RootFallback">
            <!-- 1. Safety Logic: Sequence -->
            <Sequence name="BatterySafety">
                <Inverter>
                    <CheckBattery min_voltage="20.0"/> <!-- Success if High -->
                </Inverter>
                <!-- If Battery High -> Success -> Invert -> Fail -> Fallback to Patrol -->
                <!-- If Battery Low -> Fail -> Invert -> Success -> Sequence Continues -->
                
                <NavigateToPose pose="0 0 0" behaviour="go_home"/>
            </Sequence>

            <!-- 2. Patrol Logic -->
            <Sequence name="Patrol">
                 <NavigateToPose pose="10 0 0" behaviour="waypoint_1"/>
                 <Wait duration="5.0"/>
                 <NavigateToPose pose="0 10 0" behaviour="waypoint_2"/>
            </Sequence>
        </Fallback>
    </BehaviorTree>
</root>
```

*Correction on Logic:*
Standard Check:
```xml
<Sequence>
   <CheckBattery/> <!-- Returns Success if OK -->
   <Patrol/>
</Sequence>
```
If Battery OK, Patrol.
If Battery Fails, Sequence Fails. Root should handle it?
Usually we use `Reaction` logic:
`Fallback(Sequence(BatteryLow?, GoHome), Patrol)`

---

## 🔬 Lab Exercise: "Groot Visualization"

### 1. Lab Objectives
- Run Nav2 with `enable_groot_monitoring:=true` (exposed on ZMQ port).
- Launch `Groot` (GUI). Connect.
- **Task:** Send goal.
- **Observation:** Watch the tree light up Green (Success) and Orange (Running) in real-time.
- **Action:** Block the robot. Watch "Navigate" fail, falls back to "ClearCostmap", then "Spin", then "Backup". The **Recovery Subtree** in action.

---

## 🚀 Project: "The Persistent Cleaner"

**Goal:** Clean 4 rooms. Resume if interrupted.
1.  **Blackboard:** Store `cleaned_rooms = [False, False, False, False]`.
2.  **BT Structure:**
    *   Sequence:
        *   `CleanRoom1` (Decorated with `Retry(3)`).
        *   `CleanRoom2`.
        *   ...
3.  **Interruption:** If User hits E-Stop, BT halts.
4.  **Resume:** When restarted, the Blackboard variable must be persistent (or re-checked).
5.  **Implementation:** Create a custom Decorator `SkipIfDone` that checks blackboard.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Tick Error"
*   **Symptom:** BT crashes with `Port not found`.
*   **Cause:** XML Argument (`min_voltage`) does not match C++ `getInput`.
*   **Fix:** Check spelling in XML vs `providedPorts()`.

#### 2. "Robot Stuck in Loop"
*   **Symptom:** Retrying forever.
*   **Cause:** `RetryUntilSuccessful` has num_attempts=-1 (Infinite).
*   **Fix:** Always set a limit (e.g., 5 attempts).

---

## ⚡ Optimization: Asynchronous Actions

Running `ComputePathToPose` blocks the tree?
*   No. Navigation Actions return `RUNNING`.
*   The Tree Ticks at 100Hz.
*   While `Navigate` is RUNNING, the tree can check other conditions in a `Parallel` node (e.g., "Person Detected?").
*   This allows **Preemption** (Cancel Navigation if Person Detected).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Sequence vs ReactiveSequence?
    *   **A:** Sequence executes A (Success) -> B. Once A is done, it never checks A again. ReactiveSequence checks A *every tick* while B is running. If A fails later, B is halted.
2.  **Q:** What is the Blackboard?
    *   **A:** Shared memory key-value store for the BT. Used to pass data (e.g., "Goal Pose") between nodes.
3.  **Q:** Difference between Condition and Action?
    *   **A:** Conditions are instantaneous (Check variable). Actions take time (Move robot) and return RUNNING.

### Challenge Task
> **Task:** Follow Target with Timeout.
> 1. Use `PipelineSequence`.
> 2. Node 1: `DetectPerson` (Returns Pose).
> 3. Node 2: `NavigateToPose`.
> 4. Decorate with `KeepRunningUntilFailure`? No, use `TimeLimit` decorator.

---

## 📚 Further Reading
- **BehaviorTree.CPP V3 Docs:** Michele Colledanchise.
- **Nav2 BT XML Guide:** Default trees explained.

---

**Day 75 Complete**
