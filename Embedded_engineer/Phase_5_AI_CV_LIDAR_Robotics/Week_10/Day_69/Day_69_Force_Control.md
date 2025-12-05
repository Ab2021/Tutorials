# Day 69: Force & Compliance Control (CRISP)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> Don't fight the wall. Be the water.
> - **Focus:** Impedance vs Admittance Control, Force Sensing (F/T Sensors), and Contact-Rich tasks (Peg-in-Hole).
> - **Code:** A custom `ros2_control` Controller implementing Cartesian Impedance: $F = K(x_d - x) + D(\dot{x}_d - \dot{x})$.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Distinguish** between Stiff (Position) and Soft (Impedance) control modes.
2.  **Implement** a Cartesian Impedance Controller in C++ for `ros2_control`.
3.  **Perform** a "Peg-in-Hole" insertion using force feedback to align forces.
4.  **Tune** Stiffness ($K$) and Damping ($D$) matrices for stable interaction.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Robot with Torque Sensors (Panda) OR External F/T Sensor (ATI).
- Or Simulation (Gazebo supports F/T sensors).

### Software Environment
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```

### Prior Knowledge
- Mass-Spring-Damper Systems.
- Jacobians ($F = J^T \tau$).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Philosophy of Contact

If a robot in Position Control ($K_p=10000$) hits a wall, $Force \to \infty$. Damage occurs.
*   **Compliance:** The robot acts like a spring.
*   **Impedance Control:** Output is Force/Torque.
    $$ \tau_{cmd} = J^T(q) \left[ K_x (x_{des} - x) + D_x (\dot{x}_{des} - \dot{x}) \right] + G(q) $$
    *   If $x$ moves away from $x_{des}$, push back with force proportional to $K_x$.
    *   If $K_x$ is low ("Soft"), the robot yields to obstacles.
*   **Admittance Control:** Output is Position/Velocity.
    *   Used on robots *without* torque control (Industrial UR5).
    *   Measure Force $F_{ext}$.
    *   Compute virtual motion: $\Delta x = \frac{F_{ext}}{K}$.
    *   Command new position $x' = x + \Delta x$.

### 🔹 Part 2: CRISP (Constraint-based Robust Impedance)

A framework (or concept) for varying impedance online.
*   **Free Space:** High Stiffness (Accurate tracking).
*   **Contact:** Low Stiffness (Safe interaction).
*   **Insertion:** High Stiffness in Z (Push), Low Stiffness in X/Y (Align hole).

### 🔹 Part 3: Peg-in-Hole Strategy

1.  **Search:** Spiral motion in X/Y to find hole. Force $F_z$ is high (pushing).
2.  **Detection:** If $F_z$ drops suddenly, we found the hole.
3.  **Insertion:** Push in Z. Since X/Y stiffness is low, the peg "centers" itself (Chamfer effect).

---

## 💻 Implementation: Cartesian Impedance Controller

We will write a C++ Controller Plugin.

### 🛠️ Project Structure
```text
day69_force/
├── src/
│   └── impedance_controller.cpp
├── include/
│   └── impedance_controller.hpp
└── plugin_description.xml
```

### 👨‍💻 Controller Logic (`src/impedance_controller.cpp`)

Simplified Snippet for `ros2_control` `update()` loop.

```cpp
#include "impedance_controller.hpp"

controller_interface::return_type ImpedanceController::update(
    const rclcpp::Time & time, const rclcpp::Duration & period)
{
    // 1. Get State (q, dq)
    Eigen::VectorXd q = get_positions();
    Eigen::VectorXd dq = get_velocities();
    
    // 2. Compute Kinematics (x, J)
    Eigen::MatrixXd J = robot_model_->getJacobian(q);
    Eigen::Affine3d T = robot_model_->getFK(q);
    Eigen::Vector3d x = T.translation();
    
    // 3. Compute Orientation Error (Quaternion -> Axis Angle)
    Eigen::Quaterniond q_curr(T.rotation());
    Eigen::Quaterniond q_des(target_pose_.rotation());
    // Orientation error calculation...
    Eigen::Vector3d or_err = ...;

    // 4. Compute Impedance Law
    // F = Kp * (x_des - x) - Kd * (dq_cartesian)
    Eigen::Vector3d pos_err = target_pose_.translation() - x;
    
    Eigen::Vector6d F_cart;
    F_cart.head(3) = K_pos_ * pos_err - D_pos_ * (J.topRows(3) * dq);
    F_cart.tail(3) = K_ori_ * or_err - D_ori_ * (J.bottomRows(3) * dq);
    
    // 5. Map to Joint Torques
    // tau = J^T * F + Gravity + Coriolis
    Eigen::VectorXd tau_cmd = J.transpose() * F_cart + robot_model_->getGravity(q);
    
    // 6. Send to Interfaces
    for(size_t i=0; i<n_joints_; ++i) {
        effort_interfaces_[i].set_value(tau_cmd[i]);
    }
    
    return controller_interface::return_type::OK;
}
```

### 👨‍💻 Usage (ROS 2)

Switch controllers from `joint_trajectory_controller` to `impedance_controller`.

```bash
ros2 control switch_controllers --stop joint_trajectory_controller --start impedance_controller
```

---

## 🔬 Lab Exercise: " The Scale"

### 1. Lab Objectives
- Robot holds a stationary pose.
- Push the robot hand by hand.
- **Observation:** It acts like a spring. Pushing 1cm requires $K$ Force.
- **Task:** Implement a "Weight Scale".
    1. Place object on hand.
    2. Hand sags by $\Delta x$.
    3. Mass $m = (K \cdot \Delta x) / g$.
- **Validation:** Compare with true mass.

---

## 🚀 Project: "Assembly Station"

**Goal:** Insert a cylindrical battery into a tight slot.
1.  **Approach:** Move to `z = slot_height + 2cm`.
2.  **Stiffness:** Set $K_x, K_y = 50 N/m$ (Soft), $K_z = 500 N/m$ (Stiff), $K_{rot} = 10 Nm/rad$.
3.  **Search:** Execute Circular Spiral pattern `(r=0.5cm)`.
4.  **Insert:** While searching, apply constant Force $F_z = -5N$.
5.  **Effect:** The soft X/Y axes allow the battery to "slide" into the hole when aligned. Once $Z$ velocity increases, stop Spiraling.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Unstable Vibrations"
*   **Symptom:** Robot shakes violently.
*   **Cause:** Damping $D$ is too low relative to Stiffness $K$. Or Sampling Rate is too low Loop Delay.
*   **Fix:** Increase Damping. Rule of thumb: $D = 2\sqrt{K}$. Also ensure real-time kernel ($1kHz$ loop).

#### 2. "Gravity Sag"
*   **Symptom:** Robot droops under its own weight.
*   **Cause:** Gravity Compensation term `G(q)` is inaccurate (Wrong Mass in URDF).
*   **Fix:** Perform System ID (Day 62) to correct URDF dynamics.

---

## ⚡ Optimization: Null-Space Stiffness

We have 7 joints but controlling 6-DOF pose. 1 Redundant DOF.
*   **Null-Space Control:** We can control the "Elbow" configuration without moving the hand.
*   $\tau = J^T F + (I - J^T J^\#)\tau_{null}$.
*   Use $\tau_{null}$ to keep the elbow up or away from singularities.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not use Force Control everywhere?
    *   **A:** Position control is more precise in free space (0.1mm accuracy). Force control depends on dynamics model accuracy (2-3mm accuracy unless calibrated).
2.  **Q:** Impedance vs Admittance regarding Hardware?
    *   **A:** Impedance requires Torque Source (Current control). Admittance works on Position Source (Industrial Arms) but has lower bandwidth.
3.  **Q:** What is a "Wrench"?
    *   **A:** A 6D vector combining Force (3D) and Torque (3D).

### Challenge Task
> **Task:** Surface wiping.
> 1. Robot must wipe a whiteboard.
> 2. Maintain constant Normal Force $F_n = 5N$ against the board.
> 3. Move mainly in Tangential direction.
> 4. If the board is curved, the robot must adapt automatically (Hybrid Force/Position Control).

---

## 📚 Further Reading
- **Modern Robotics:** Chapter on Force Control.
- **Hogan, N.:** "Impedance Control: An Approach to Manipulation".

---

**Day 69 Complete**
