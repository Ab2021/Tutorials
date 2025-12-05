# Day 67: Dexterous Manipulation (BiDexHand)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> Parallel grippers are simple. Hands are complex (20+ DOF).
> - **Focus:** Multi-finger kinematics, The "Grasp Matrix", BiDexHand dataset, and Simulation of Shadow/Allegro Hands.
> - **Code:** A controller for an Allegro Hand (16 DOF) to switch between Power, Pinch, and Key grasps.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Model** the kinematics of a multi-finger hand (Serial chains attached to a Palm).
2.  **Calculate** the Grasp Matrix $G$ relating Contact Forces to Object Wrench.
3.  **Implement** "Synergy" control to reduce 20 DOF to 2-3 usable components (PCA).
4.  ** Simulate** an Allegro Hand in Gazebo and perform in-hand reorientation.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None (Simulation recommended). Real Shadow Hands cost $50k+.
- Allegro Hand (Simulated).

### Software Environment
```bash
sudo apt install ros-humble-ros2-control
# Clone Allegro Hand ROS2 description
```

### Prior Knowledge
- Jacobian Transpose (Day 50).
- Eigen Library (C++).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Curse of Dimensionality

A human hand has ~27 Degrees of Freedom.
*   **Planning:** RRT* in 7-DOF (Arm) is fast. RRT* in 27-DOF (Hand+Arm) is exponential time.
*   **Solution: Eigengrasps (Synergies).**
    *   Observation: When we close a fist, all fingers move together.
    *   PCA on human data shows 2 Principal Components explain 80% of variance.
    *   Control Input: $u \in \mathbb{R}^2 \to q \in \mathbb{R}^{20}$.

### 🔹 Part 2: The Grasp Matrix ($G$)

For an object to be stable, the hand must resist *any* external wrench ($W_{ext}$).
$$ G \cdot f_c = -W_{ext} $$
*   $G$: Grasp Matrix ($6 \times m$). Maps contact forces ($f_c$) to Object Frame.
*   **Force Closure:** Can the grasp resist wrenches in all directions? (Is $G$ full rank and strictly positive coefficients possible?)

### 🔹 Part 3: BiDexHand & Learning

*   **BiDexHand:** A dataset of bimanual (two-handed) dexterous manipulation.
*   **Challenge:** Getting the fingers to *roll* over the object (In-hand manipulation) without dropping it.
*   **Reinforcement Learning:** Essential here. Hard-coding rolling contacts is mathematically painful. RL agents learn it via PPO.

---

## 💻 Implementation: Hand Controller (Synergies)

We will implement a "Synergy Controller" for a 16-DOF Allegro Hand (4 fingers x 4 joints).

### 🛠️ Project Structure
```text
day67_dexterous/
├── config/
│   └── hand.yaml
├── urdf/
│   └── allegro.urdf.xacro
├── src/
│   └── hand_controller.cpp
└── include/
    └── synergies.h
```

### 👨‍💻 Synergy Mapping (`include/synergies.h`)

We define a linear map $q = S \cdot z + q_{mean}$.

```cpp
#pragma once
#include <Eigen/Dense>
#include <vector>

class HandSynergies {
public:
    HandSynergies() {
        // 16 Joints. 2 Synergies.
        measure_matrix_ = Eigen::MatrixXd::Zero(16, 2);
        mean_pose_ = Eigen::VectorXd::Zero(16);
        
        // Define Synergy 1: "Close Fist" (All fingers curl)
        // Indices 0-3 (Index), 4-7 (Middle), 8-11 (Ring), 12-15 (Thumb)
        for(int i=0; i<12; ++i) measure_matrix_(i, 0) = 1.0; 
        
        // Define Synergy 2: "Spread" (Abduction)
        // Joints 0, 4, 8 are usually abduction/adduction
        measure_matrix_(0, 1) = -0.5;
        measure_matrix_(4, 1) = 0.0;
        measure_matrix_(8, 1) = 0.5;
    }
    
    Eigen::VectorXd map(double s1, double s2) {
        Eigen::Vector2d input(s1, s2);
        return (measure_matrix_ * input) + mean_pose_;
    }

private:
    Eigen::MatrixXd measure_matrix_;
    Eigen::VectorXd mean_pose_;
};
```

### 👨‍💻 Controller Node (`src/hand_controller.cpp`)

ROS 2 Node subscribed to `Joy` (Gamepad) to control the hand.

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include "synergies.h"

class HandController : public rclcpp::Node {
public:
    HandController() : Node("hand_controller") {
        pub_joints_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
        sub_joy_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10, std::bind(&HandController::joyCb, this, std::placeholders::_1));
            
        synergies_ = std::make_unique<HandSynergies>();
    }

    void joyCb(const sensor_msgs::msg::Joy::SharedPtr msg) {
        // Map Joystick Axes to Synergies
        double s1 = msg->axes[1]; // Forward/Back = Open/Close
        double s2 = msg->axes[0]; // Left/Right = Spread
        
        Eigen::VectorXd q = synergies_->map(s1, s2);
        
        auto joint_msg = sensor_msgs::msg::JointState();
        joint_msg.header.stamp = this->now();
        
        std::vector<std::string> names = {
            "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
            "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
            "ring_joint_0",   "ring_joint_1",   "ring_joint_2",   "ring_joint_3",
            "thumb_joint_0",  "thumb_joint_1",  "thumb_joint_2",  "thumb_joint_3"
        };
        
        joint_msg.name = names;
        joint_msg.position.resize(16);
        for(int i=0; i<16; ++i) joint_msg.position[i] = q[i];
        
        pub_joints_->publish(joint_msg);
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joints_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_joy_;
    std::unique_ptr<HandSynergies> synergies_;
};
```

---

## 🔬 Lab Exercise: The "Pen Spin"

### 1. Lab Objectives
- Spawn a Cylinder (Pen) in Gazebo floating in the air (use `0G` gravity initially).
- Control the hand to make contacts.
- **Task:** Rotate the pen 90 degrees using the Finger Gaiting technique (Index hold, Middle push).
- **Difficulty:** Extreme. Requires synchronized impedance control.
- **Simplified:** Just achieve a stable 3-finger grasp.

---

## 🚀 Project: "Rock Paper Scissors"

**Goal:** Gesture Recognition + Actuation.
1.  **Input:** Camera sees User Hand (MediaPipe from Day 44).
2.  **Logic:** User plays "Rock". Robot logic chooses "Paper" (Cheat mode).
3.  **Actuation:** Robot Hand forms "Paper" (Open Palm).
4.  **Timing:** Must happen in < 500ms.
5.  **Challenge:** Mapping "Paper" to Joint Angles quickly using the Synergy class.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Self Collision"
*   **Symptom:** Fingers explode in simulation.
*   **Cause:** Fingers intersecting each other. The collision mesh is slightly larger than visual mesh.
*   **Fix:** **Self-Collision Matrix (SRDF)**. Disable collision checking between adjacent links (Finger1 Link1 and Finger1 Link2).

#### 2. "Object slips"
*   **Symptom:** Grasp looks good, but object falls.
*   **Cause:** Simulation friction is too low. Or "Torsional Friction" (Spinning) is unmodeled in ODE.
*   **Fix:** Use `min_depth` contact parameter in Gazebo or increase friction coeff $\mu > 1.0$ (Sticky).

---

## ⚡ Optimization: Tactile Feedback

Visual feedback is too slow for slip detection.
*   **Tactile Sensors:** BioTac, GelSight.
*   **Simulation:** Use `ContactSensor` plugin on finger tips.
*   **Control Loop:** If `ShearForce > Threshold`, Increase `NormalForce`. Reflex loop (1kHz).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Power Grasp vs Precision Grasp?
    *   **A:** Power: Palm contact, high stability (Hammer). Precision: Fingertip contact, high dexterity (Pen).
2.  **Q:** What is "Finger Gaiting"?
    *   **A:** Regrasping the object *without* letting go. Like walking your fingers up a pencil.
3.  **Q:** Why are underactuated hands popular (e.g., Robotiq)?
    *   **A:** 1 Motor drives 3 joints via springs/tendons. The hand "mechanically adapts" to the object shape. Simple control, robust grasping.

### Challenge Task
> **Task:** The Rubik's Cube.
> 1. Just *holding* it with one hand while the other arm rotates a face.
> 2. This is a dual-arm, dexterous manipulation task.
> 3. State-of-the-Art (OpenAI) required months of training. Don't expect to solve it in a day, but set up the URDFs.

---

## 📚 Further Reading
- **BiDexHand Benchmark:** CVPR 2024.
- **Hand Synergies:** "Postural Synergies for Robotic Grasping".

---

**Day 67 Complete**
