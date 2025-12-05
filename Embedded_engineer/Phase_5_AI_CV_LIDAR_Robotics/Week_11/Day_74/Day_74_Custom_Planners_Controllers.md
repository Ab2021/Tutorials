# Day 74: Custom Planners & Controllers (MPPI)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> DWB is old. MPPI is the future.
> - **Focus:** Writing Custom Nav2 Plugins (Planner/Controller), MPPI (Model Predictive Path Integral) theory, and GPU-accelerated control.
> - **Code:** A basic C++ "Pure Pursuit" Controller plugin from scratch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Develop** a custom Controller Plugin inheriting `nav2_core::Controller`.
2.  **Understand** the MPPI algorithm: Stochastic sampling, Cost evaluation, and Control update.
3.  **Configure** MPPI for distinct behaviors (Ackermann steering, Omni-directional).
4.  **Compare** DWB (Critic-based) vs MPPI (Optimization-based).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (Desired for MPPI, though CPU version exists).

### Software Environment
```bash
sudo apt install ros-humble-nav2-mppi-controller
```

### Prior Knowledge
- Model Predictive Control (MPC) conceptual understanding (Day 23).
- C++ Inheritance.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Controller Interface

All Nav2 controllers must implement `computeVelocityCommands`.
*   **Input:** Current Pose, Velocity, Goal, and the Path (Global Plan).
*   **Output:** Twist (`cmd_vel`).
*   **Frequency:** Loops at 20Hz-50Hz.

### 🔹 Part 2: MPPI (Model Predictive Path Integral)

How does it work?
1.  **Sample:** Generate 1000 random noise sequences for control inputs ($\Delta u$).
2.  **Rollout:** Simulate robot forward using a kinematic model for each sequence.
3.  **Score:** Evaluate cost of each trajectory (Collision, Path Distance, Speed).
4.  **Weight:** Convert cost to probability ($P \propto \exp(-Cost/\lambda)$).
5.  **Update:** New control $u_{new} = \sum P_i \cdot u_i$.
*   **Key:** No gradients needed! Works with non-differentiable costmaps (binary obstacles).

### 🔹 Part 3: MPPI vs DWB

*   **DWB (Dynamic Window Approach extended):**
    *   Samples a *single step* (velocity). Checks if that velocity leads to a crash using a projected arc.
    *   Greedy. Gets stuck in local minima (U-shaped traps).
*   **MPPI:**
    *   Samples *trajectories* (Horizon $T=2s$).
    *   Sees the U-trap ahead and steers around it.
    *   Computationally heavy.

---

## 💻 Implementation: Custom "Pure Pursuit" Plugin

We will implement the classic Pure Pursuit algorithm as a Nav2 plugin.
Logic: Find point on path at lookahead distance $L$. Steer towards it.

### 🛠️ Project Structure
```text
day74_control/
├── include/day74_control/
│   └── simple_pure_pursuit.hpp
├── src/
│   └── simple_pure_pursuit.cpp
├── plugin_description.xml
└── CMakeLists.txt
```

### 👨‍💻 Header (`simple_pure_pursuit.hpp`)

```cpp
#ifndef SIMPLE_PURE_PURSUIT_HPP
#define SIMPLE_PURE_PURSUIT_HPP

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_loader.hpp"

namespace day74_control {

class SimplePurePursuit : public nav2_core::Controller {
public:
  SimplePurePursuit() = default;
  ~SimplePurePursuit() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setPlan(const nav_msgs::msg::Path & path) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  nav_msgs::msg::Path global_plan_;
  double lookahead_dist_;
  double max_speed_;
  rclcpp::Logger logger_{rclcpp::get_logger("SimplePurePursuit")};
};

} # namespace
#endif
```

### 👨‍💻 Implementation (`simple_pure_pursuit.cpp`)

```cpp
#include "day74_control/simple_pure_pursuit.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(day74_control::SimplePurePursuit, nav2_core::Controller)

namespace day74_control {

void SimplePurePursuit::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> &,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> &) 
{
  auto node = parent.lock();
  lookahead_dist_ = 0.5; // Parameterize this!
  max_speed_ = 0.5;
  logger_ = node->get_logger();
}

void SimplePurePursuit::activate() {}
void SimplePurePursuit::deactivate() {}
void SimplePurePursuit::cleanup() {}
void SimplePurePursuit::setPlan(const nav_msgs::msg::Path & path) { global_plan_ = path; }
void SimplePurePursuit::setSpeedLimit(const double & speed_limit, const bool & percentage) {}

geometry_msgs::msg::TwistStamped SimplePurePursuit::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist &,
  nav2_core::GoalChecker *) 
{
  // 1. Find target point
  // Simple logic: return first point > lookahead distance away
  geometry_msgs::msg::PoseStamped target_pose;
  bool found = false;
  
  for(const auto & p : global_plan_.poses) {
    double dist = std::hypot(p.pose.position.x - pose.pose.position.x, 
                             p.pose.position.y - pose.pose.position.y);
    if(dist > lookahead_dist_) {
        target_pose = p;
        found = true;
        break;
    }
  }
  
  if(!found && !global_plan_.poses.empty()) target_pose = global_plan_.poses.back();

  // 2. Compute curvature (Pure Pursuit Law)
  // curvature = 2 * y / L^2 (in vehicle frame)
  // Need to transform target_pose to robot frame
  double dx = target_pose.pose.position.x - pose.pose.position.x;
  double dy = target_pose.pose.position.y - pose.pose.position.y;
  
  double yaw = tf2::getYaw(pose.pose.orientation);
  double x_local = dx * cos(yaw) + dy * sin(yaw);
  double y_local = -dx * sin(yaw) + dy * cos(yaw);
  
  double curv = 2.0 * y_local / (lookahead_dist_ * lookahead_dist_);
  
  // 3. Command
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = pose.header.stamp;
  cmd.twist.linear.x = max_speed_;
  cmd.twist.angular.z = curv * max_speed_;
  
  return cmd;
}

}
```

### 👨‍💻 MPPI Configuration (`nav2_params.yaml`)

To use MPPI instead of our simple one:

```yaml
ControllerServer:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 56
      model_dt: 0.05
      batch_size: 2000 # Samples
      vx_std: 0.2
      wz_std: 0.4
      vx_max: 0.5
      wz_max: 1.0
      critics: ["ConstraintCritic", "CostCritic", "GoalCritic", "PathAlignCritic"]
      ConstraintCritic:
         enabled: true
         cost_power: 1
         cost_weight: 4.0
      CostCritic: # Obstacles
         enabled: true
         cost_weight: 3.81
         inflation_radius: 0.55
```

---

## 🔬 Lab Exercise: "The Dynamic Crowd"

### 1. Lab Objectives
- Spawn 5 "Pedestrians" (Moving Cylinders) in Gazebo using `actor` plugin.
- **Task:** Navigate through the crowd.
- **Controller A (DWB):** Will likely stop and wait (Oscillate).
- **Controller B (MPPI):** Will weave through gaps smoothly.
- **Analysis:** Observe the Local Plan (Trajectory Cloud) in Rviz. DWB shows 1 arc. MPPI shows 2000 arcs (Cloud).

---

## 🚀 Project: "Drifting Robot"

**Goal:** High-Speed Navigation control.
1.  **Sim:** Accel Max 2.0 m/s2. Speed Max 2.0 m/s.
2.  **Model:** Ackermann Vehicle (Car-like).
3.  **Task:** Navigate a hairpin turn.
4.  **MPPI Config:** Enable `slip_factor` in the motion model (if available) or tune `wz_std` to allow aggressive sampling.
5.  **Result:** The controller should anticipate the turn and steer early (Out-In-Out racing line emergence due to cost optimization).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "MPPI CPU Overload"
*   **Symptom:** Controller rate drops to 2Hz.
*   **Cause:** `batch_size` too high for CPU. 2000 samples requires ~8 cores or GPU.
*   **Fix:** Reduce `batch_size` to 500. Reduce `time_steps`.

#### 2. "Robot spinning in place"
*   **Symptom:** Planner returns path, Controller spins.
*   **Cause:** `PathAlign` critic weight too high? Or Start Pose of robot has $180^\circ$ error vs Map.
*   **Fix:** Check Localization. Lower `PathAlign` weight.

---

## ⚡ Optimization: Initial Trajectory

MPPI samples Random Noise.
*   **Warm Start:** Use the *previous* optimal control sequence as the mean for the next step's noise.
*   **Shift:** $u_{init}[t] = u_{prev}[t+1]$.
*   Ensures continuity and faster convergence.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Pure Pursuit vs MPC?
    *   **A:** Pure Pursuit is geometric (lookahead). MPC is optimization (minimizes cost function over horizon). MPC handles dynamics/constraints better.
2.  **Q:** What is a "Critic" in MPPI?
    *   **A:** A cost function component. E.g., `ObstacleCritic` computes cost based on proximity to walls. `GoalCritic` based on distance to goal.
3.  **Q:** Can MPPI handle "Jack-knifing" trailers?
    *   **A:** Yes, if the Kinematic Model in the rollout step includes the trailer dynamics.

### Challenge Task
> **Task:** Reversing Controller.
> 1. Modify the Pure Pursuit plugin.
> 2. Allow `linear.x` to be negative.
> 3. If target is *behind* the robot, set `linear.x = -speed` and invert steering logic.

---

## 📚 Further Reading
- **MPPI Paper:** "Information Theoretic Model Predictive Control".
- **Nav2 MPPI Configuration Guide:** Tunable parameters.

---

**Day 74 Complete**
