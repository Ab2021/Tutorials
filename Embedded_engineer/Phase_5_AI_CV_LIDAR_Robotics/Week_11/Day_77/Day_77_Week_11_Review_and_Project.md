# Day 77: Week 11 Review & Capstone Project
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> We went from "Moving A to B" to "Enterprise Grade Navigation".
> - **Goal:** Integrate Custom Layers, MPPI Control, Lifecycle management, and Behavior Trees into a robust fleet-ready stack.
> - **Code:** A `warehouse_amr.launch.py` that brings up the full customized stack.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a Production-Grade Nav2 Stack (Not just the default `nav2_bringup`).
2.  **Harmonize** Local and Global Planners (Grid Search + MPPI).
3.  **Validate** Robustness: Run the robot for 1 hour in a dynamic environment without getting stuck.

---

## 📚 Week 11 Review: The Navigation Stack

| Day | Topic | Key Lesson | Tool |
|-----|-------|------------|------|
| **71** | **Nav2 Architecture** | Lifecycle Nodes & Action Servers | `lifecycle_manager` |
| **72** | **SLAM Toolbox** | GraphSLAM & Map Serialization | `async_slam_toolbox` |
| **73** | **Costmap Layers** | Voxels & Keepout Zones | `VoxelLayer`, `CostmapFilters` |
| **74** | **Custom Controllers** | MPPI (Trajectory Optimization) | `MPPIController` |
| **75** | **Behavior Trees** | Logic & Recovery Strategies | `BehaviorTree.CPP` |
| **76** | **Outdoor GPS** | Coordinates & EKF Fusion | `navsat_transform`, `robot_localization` |

### The "Stack" Diagram
```mermaid
graph TD
    A[Lidar/Depth] --> B[Voxel Layer]
    C[GPS] --> D[Robot Localization (EKF)]
    D --> E[TF (map->odom)]
    B --> F[Costmap 2D]
    F --> G[Global Planner (Smac)]
    F --> H[Local Planner (MPPI)]
    G --> H
    H --> I[Twist (cmd_vel)]
    J[Behavior Tree] --> G
    J --> H
```

---

## 🚀 Weekly Capstone: "The Warehouse Autonomy Stack"

**Scenario:** A large 100x100m warehouse with Human Workers (Dynamic) and Forbidden Zones (Forklift Charging).
**Robot:** Differential Drive AMR (Autonomous Mobile Robot).
**Mission:** "Deliver Pallet to Dock 4".

### 🛠️ Project Structure
```text
week11_capstone/
├── config/
│   ├── amr_nav2_params.yaml (MPPI + Smac + Layers)
│   └── behavior_tree.xml (Recovery + Battery)
├── maps/
│   └── warehouse_mask.pgm (Keepout Filter)
├── launch/
│   └── deployment.launch.py
└── src/
    └── task_dispatcher.py
```

### 👨‍💻 Configuration Highlights

**1. Global Planner (Smac Hybrid-A*):**
*   Allows Reversing (Ackermann or Diff Drive).
*   Path Smoothing enabled.
*   `motion_model_for_search: DIFF_DRIVE`.

**2. Local Planner (MPPI):**
*   `critics: [Constraint, Obstacle, PathAlign, PathFollow]`.
*   High `Obstacle` weight to avoid Humans.
*   `batch_size: 1000`.

**3. Costmap Layers:**
*   Global: `StaticLayer`, `InflationLayer`, `KeepoutFilter`.
*   Local: `ObstacleLayer` (Lidar), `VoxelLayer` (Depth Cam), `InflationLayer`.

**4. Behavior Tree:**
*   **Root:** `RecoveryNode` (Retry 6 times).
*   **Main:** `PipelineSequence`.
    *   `RateController` (1Hz): `ComputePathToPose`.
    *   `RateController` (20Hz): `FollowPath`.
*   **Recovery:** `RoundRobin`.
    *   `ClearCostmap`.
    *   `Spin`.
    *   `Wait`.
    *   `BackUp`.

### 👨‍💻 Launch File (`launch/deployment.launch.py`)

A "Single Command" launch file.

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    nav2_dir = get_package_share_directory('nav2_bringup')
    my_dir = get_package_share_directory('week11_capstone')
    
    # 1. Map & Params
    map_file = os.path.join(my_dir, 'maps', 'warehouse.yaml')
    params_file = os.path.join(my_dir, 'config', 'amr_nav2_params.yaml')

    # 2. Main Nav2 Launch
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'map': map_file,
            'params_file': params_file,
            'use_sim_time': 'true',
            'autostart': 'true'
        }.items()
    )

    return LaunchDescription([nav2_cmd])
```

---

## 📝 Self-Assessment Quiz

1.  **Architecture:**
    *   Why use `LifecycleNodes` in production?
    *   **A:** Deterministic startup. We ensure the Map is loaded *before* the planner starts planning. Prevents race conditions.
2.  **Control:**
    *   Limits of MPPI?
    *   **A:** High Compute (GPU preferred). Stochastic nature means output can "jitter" slightly (requires smoothing).
3.  **GPS:**
    *   Common failure mode in Urban Canyons?
    *   **A:** Multipath reflection. GPS jumps 20m. EKF rejects it if `Mahalanobis Distance` is high (Outlier rejection).

---

## ⏭️ Look Ahead: Week 12
We move to the frontier of AI Robotics.
**Week 12: Robot Learning (RL & Imitation).**
*   Programming robots manually (Weeks 1-11) is hard.
*   Can we teach them by demonstration? (Imitation Learning).
*   Can they learn by trial and error? (Reinforcement Learning).
*   Diffusion Policies, DAgger, and Sim-to-Real.

---

**Week 11 Complete**
