# Day 65: MoveIt Task Constructor (MTC)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> "Go pick up that cup." sounds simple.
> In Robotics, it's: `MoveToPreGrasp` -> `Approach` -> `CloseGripper` -> `AttachObject` -> `Lift` -> `Retreat`.
> - **Focus:** Hierarchical Task Networks (HTN), MTC Stages types (Generator, Propagator, Connector), and Serial/Parallel Containers.
> - **Code:** A robust C++ MTC pipeline for a Pick & Place operation with backtracking.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Decompose** complex manipulation tasks into atomic MTC stages.
2.  **Implement** the "Pick and Place" pipeline using `CurrentState`, `MoveTo`, `GenerateGraspPose`, and `ModifyPlanningScene`.
3.  **Debug** MTC solutions using the Rviz `MotionPlanningTasks` panel.
4.  **Handle** Failures: Why did the "Connect" stage fail? (Reachability vs Collision).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Manipulator with Gripper (Frank Emika Panda is standard for MTC tutorials).

### Software Environment
```bash
sudo apt install ros-humble-moveit-task-constructor-core
sudo apt install ros-humble-moveit-task-constructor-loader
sudo apt install ros-humble-moveit-task-constructor-visualization
```

### Prior Knowledge
- Day 64 (MoveIt Core).
- C++ Lambda functions (std::function).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Linear Pipeline vs. MTC

*   **MoveGroup (Linear):**
    1. Plan to Pose A. Execute.
    2. Close Gripper. Execute.
    3. Plan to Pose B. Fail? (Collision).
    *   **Problem:** If Step 3 fails, we acted on Step 1 & 2. We are stuck in a bad state.
*   **MTC (Global Search):**
    *   Constructs a **Task Graph**.
    *   Solves *all* steps in simulation first.
    *   Finds a continuous valid trajectory through all stages (A -> B -> C).
    *   Only Executes if the *entire* chain is valid.

### 🔹 Part 2: Stage Types

1.  **Generator (Source):** Creates states.
    *   `CurrentState`: Starts from where the robot is.
    *   `GeneratePose`: Samples poses (e.g., Grasp Candidates).
2.  **Propagator (Forward/Backward):**
    *   `MoveTo`: Kinematic motion (Joint space).
    *   `MoveRelative`: Cartesian motion (e.g., Approach 10cm along Z).
3.  **Connector:**
    *   `Connect`: Bridges two states using a planner (RRTConnect). Connects "Pre-Grasp" to "Home".
4.  **Modifiers:**
    *   `ModifyPlanningScene`: Attaches/Detaches objects (Simulates grasping).

### 🔹 Part 3: Containers

*   **Serial Container:** A -> B -> C. (Sequence).
*   **Parallel Container:** Alternatives. "Try Grasp A, Grasp B, Grasp C". If Grasp A fails IK, try B. This is the power of MTC.

---

## 💻 Implementation: The MTC Pick Pipeline

We will create a specialized class `PickPlaceTask`.

### 🛠️ Project Structure
```text
day65_mtc/
├── include/
│   └── pick_place_task.h
├── src/
│   ├── pick_place_task.cpp
│   └── main.cpp
├── launch/
│   └── mtc_demo.launch.py
└── CMakeLists.txt
```

### 👨‍💻 Header (`include/pick_place_task.h`)

```cpp
#pragma once
#include <rclcpp/rclcpp.hpp>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/task_constructor/solvers.h>

namespace mtc = moveit::task_constructor;

class PickPlaceTask {
public:
    PickPlaceTask(const std::string& task_name);
    bool init(const rclcpp::Node::SharedPtr& node);
    bool plan();
    bool execute();

private:
    mtc::Task task_;
    mtc::solvers::PipelinePlannerPtr sampling_planner_; // OMPL
    mtc::solvers::CartesianPathPtr cartesian_planner_; // Linear

    void setupTask();
};
```

### 👨‍💻 Implementation (`src/pick_place_task.cpp`)

This is where the magic happens. Note the "Reverse" thinking often needed.

```cpp
#include "pick_place_task.h"

PickPlaceTask::PickPlaceTask(const std::string& task_name) : task_(task_name) {}

bool PickPlaceTask::init(const rclcpp::Node::SharedPtr& node) {
    task_.loadRobotModel(node);
    
    // Solvers
    sampling_planner_ = std::make_shared<mtc::solvers::PipelinePlanner>(node);
    cartesian_planner_ = std::make_shared<mtc::solvers::CartesianPath>();
    cartesian_planner_->setMaxVelocityScaling(1.0);
    cartesian_planner_->setMaxAccelerationScaling(1.0);
    cartesian_planner_->setStepSize(.01);

    setupTask();
    return true;
}

void PickPlaceTask::setupTask() {
    mtc::Stage* current_state_ptr = nullptr; // Pointer to track stages

    // 1. Current State
    auto stage_state_current = std::make_unique<mtc::stages::CurrentState>("current");
    current_state_ptr = stage_state_current.get();
    task_.add(std::move(stage_state_current));

    // 2. Open Hand
    auto stage_open_hand = std::make_unique<mtc::stages::MoveTo>("open hand", sampling_planner_);
    stage_open_hand->setGroup("hand");
    stage_open_hand->setGoal("open");
    task_.add(std::move(stage_open_hand));

    // 3. Connect to Pick (Move from Home to Pre-Grasp)
    auto stage_connect_to_pick = std::make_unique<mtc::stages::Connect>(
        "connect to pick", 
        mtc::stages::Connect::GroupPlannerVector{{"panda_arm", sampling_planner_}}
    );
    stage_connect_to_pick->properties().configureInitFrom(mtc::Stage::PARENT);
    task_.add(std::move(stage_connect_to_pick));

    // --- PICK CONTAINER (SERIAL) ---
    {
        auto grasp = std::make_unique<mtc::SerialContainer>("pick object");
        
        // 4. Approach Object (Cartesian)
        auto stage_approach = std::make_unique<mtc::stages::MoveRelative>("approach object", cartesian_planner_);
        stage_approach->properties().set("marker_ns", "approach_object");
        stage_approach->properties().set("link", "panda_hand");
        stage_approach->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = "panda_link0";
        vec.vector.z = -0.15; // Move Down 15cm
        stage_approach->setDirection(vec);
        grasp->add(std::move(stage_approach));

        // 5. Generate Grasp Pose (Generator)
        // Here we ideally use a Sampler. For simplicity, we assume we know the pose.
        // In reality, this is where "GraspNet" output feeds in.
        auto stage_gen_grasp = std::make_unique<mtc::stages::GenerateGraspPose>("generate grasp pose");
        stage_gen_grasp->properties().configureInitFrom(mtc::Stage::PARENT);
        stage_gen_grasp->setPreGraspPose("open");
        stage_gen_grasp->setObject("cube");
        stage_gen_grasp->setAngleDelta(M_PI / 12); // Try rotations
        stage_gen_grasp->setMonitoredStage(current_state_ptr); // Compute IK for grasp
        
        // ... (This part requires a ComputeIK stage wrapping the generator)
        // MTC syntax is verbose for Grasp generation.
        // Usually: GeneratePose -> ComputeIK -> AllowCollision -> CloseGripper -> Attach
        
        // Let's implement the "Touch" logic:
        // Allow Collision (Hand touches Object)
        // Modify Planning Scene (Attach Object)
    }
}
```

*Note:* The code above is simplified. A full MTC Pick implementation is ~300 lines. The key concept is that you define the *logic constraints* (Approach this much, Grasp this way), and the solver fills in the joint angles.

### 👨‍💻 Launch & Visualize

```python
# launch/mtc.launch.py
# Must launch move_group + Rviz with MTC Panel
```

---

## 🔬 Lab Exercise: The "Parallel" Grasp

### 1. Lab Objectives
- The robot tries to pick up a mug.
- **Top Grasp:** Approach from +Z.
- **Side Grasp:** Approach from +X.
- **Task:** Create a `ParallelContainer` ("Alternatives"). Add both grasp strategies.
- **Outcome:** If the mug is under a shelf (Top blocked), MTC should automatically prune the Top Grasp branch and select the Side Grasp.
- **Visualization:** See the failed solutions in Red and successful in Green in Rviz.

---

## 🚀 Project: "Multi-Object Stacking"

**Goal:** Stack 3 Blocks (Red, Green, Blue).
1.  **Logic:** Loop 3 times.
2.  **Height:** Target Z increases by `block_height` each time.
3.  **Constraint:** When moving Red block, DO NOT collide with Green block (already placed).
4.  **MTC:** Since MTC plans globally, it knows where the Green block *will be* in the future? No, MTC is usually "One Task per Plan".
    *   **Solution:** Plan Task 1 (Pick Red). Execute. Update Scene. Plan Task 2 (Pick Green).
    *   **Advanced:** "Task Graph" across multiple objects (very expensive).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "No Solutions Found"
*   **Cause:** IK Failure is #1. The target grasp pose is out of reach.
*   **Fix:** Use `GeneratePose` with `setAngleDelta(0.1)` to try rotating the gripper around the Z-axis of the object. Often a 45-degree rotation makes it reachable.

#### 2. "Collision in Approach"
*   **Cause:** "Approach" stage starts *in collision* because the gripper is too close.
*   **Fix:** Use `AllowCollision` stage *before* the Approach? No, usually `AllowCollision` is for the *Touch* moment. Check the "Approach Distance".

---

## ⚡ Optimization: Bi-Directional Planning

Standard MTC builds from Start $\to$ Goal.
*   **Bi-Directional:** Some MTC stages act as "forward" and "backward" propagators.
*   **Pick:** We usually plan *backwards* from the Grasp.
    *   Grasp Pose is fixed.
    *   Retreat is defined relative to Grasp.
    *   Approach is defined relative to Grasp.
    *   We *Connect* Start to Approach.
*   This is physically more robust than trying to "aim" the start trajectory to land exactly on the object.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is MTC better than a script?
    *   **A:** Reversibility and Global Optimality. A script executes blindly. MTC checks if the *Retreat* is possible before even starting the *Approach*.
2.  **Q:** What is the `ModifyPlanningScene` stage?
    *   **A:** Ideally used to Attach/Detach objects. It changes the collision matrix (e.g., "Ignore collision between Hand and Object").
3.  **Q:** Serial vs Parallel Container?
    *   **A:** Serial = AND (All must succeed). Parallel = OR (One must succeed).

### Challenge Task
> **Task:** The "Handover".
> 1. Robot A picks object. Move to center.
> 2. Robot B moves to center.
> 3. Robot A releases. Robot B grasps.
> 4. **MTC:** This requires a multi-robot MoveGroup? No, synchronization is hard. Hard to model in single MTC task unless using a unified `group="both_arms"`.

---

## 📚 Further Reading
- **MoveIt Tutorials:** "MTC Pick and Place".
- **PickNik MTC:** Deep Dive videos.

---

**Day 65 Complete**
