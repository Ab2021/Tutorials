# Day 72: SLAM Toolbox & Lifelong Mapping
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> GMapping is dead. AMCL is just localization.
> - **Focus:** SLAM Toolbox (GraphSLAM), Lifelong Mapping (Map Persistence), Loop Closure, and Map Merging.
> - **Code:** A fully configured `mapper_params.yaml` tuned for large environments (>100m).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** SLAM Toolbox for Asynchronous Online Mapping.
2.  **Serialize** (Save) and **Deserialize** (Load) Pose Graphs for continued mapping.
3.  **Tune** Loop Closure parameters to prevent "Map Jumping".
4.  **Execute** Lifelong Mapping: Run for days, updating the map without memory explosion.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Lidar (2D is fine, e.g., RPLIDAR).
- Odometry (must be decent, <5% drift).

### Software Environment
```bash
sudo apt install ros-humble-slam-toolbox
```

### Prior Knowledge
- GraphSLAM Theory (Nodes, Edges, Optimization).
- TF2 (odom -> base_link).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: GMapping vs SLAM Toolbox

*   **GMapping (Particle Filter):**
    *   Maintains $N$ hypothesis maps.
    *   Problem: Memory explosion ($N \times MapSize$). Failed on large maps.
*   **SLAM Toolbox (Karto + GraphSLAM):**
    *   **Frontend:** Scan Matching (Correlative Scan Matcher). Builds Nodes/Edges.
    *   **Backend:** Solver (Ceres/G2O) optimizes the Pose Graph.
    *   Result: One map. Efficient. Loop Closure warps the whole graph to fix drift.

### 🔹 Part 2: Lifelong Mapping

Normal SLAM fails after 2 hours because the graph grows infinitely.
*   **Node Pruning:** Remove nodes that don't add new information (conceptually).
*   **Area Trimming:** Keep only the "Working Set" of the graph active in optimization.
*   **Serialization:** Save the *Graph* (not just the Image). Loading an Image (`map_server`) only gives occupancy. Loading a Graph allows you to *modify* the map later.

### 🔹 Part 3: Loop Closure

The robot returns to a known spot.
*   **Detection:** Scan Matching against *historical* nodes.
*   **Verification:** check consistency.
*   **Optimization:** $\min \sum (z_{ij} - h(x_i, x_j))^2$.
*   **Effect:** The map "snaps" together.

---

## 💻 Implementation: Production Configuration

We don't write C++ today. We master the Configuration (YAML).

### 🛠️ Project Structure
```text
day72_slam/
├── config/
│   └── mapper_params_online_async.yaml
├── launch/
│   └── lifelong_slam.launch.py
└── maps/
    └── office_save.posegraph
```

### 👨‍💻 Configuration (`config/mapper_params_online_async.yaml`)

Tuned for a warehouse environment.

```yaml
slam_toolbox:
  ros__parameters:

    # 1. Mode Setup
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT
    
    # 2. Keyframe Generation (When to add a node)
    minimum_time_interval: 0.5
    minimum_travel_distance: 0.5 # 0.5m
    minimum_travel_heading: 0.5  # ~30 deg
    
    # 3. Scan Matching (Frontend)
    # Search window for matching scans
    scan_buffer_size: 10
    scan_buffer_maximum_scan_distance: 10.0
    link_match_minimum_response_fine: 0.1  
    link_scan_maximum_distance: 1.5
    
    # 4. Loop Closure (Backend)
    do_loop_closing: true
    loop_search_space_dimension: 8.0 # Search 8m radius
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_coarse: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    # 5. Lifelong / Memory
    enable_interactive_mode: true # Allow Rviz interaction
    resolution: 0.05
    max_laser_range: 20.0 # Lidar max range
```

### 👨‍💻 Launch (`launch/lifelong_slam.launch.py`)

Invokes the node with params.

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value='config/mapper_params_online_async.yaml'),
            
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[params_file]
        )
    ])
```

---

## 🔬 Lab Exercise: "The Snap"

### 1. Lab Objectives
- Drive robot in a large loop (corridor).
- **Observe:** Drift accumulates. The start and end locations don't align in the map.
- **Action:** Drive *past* the start point again.
- **Event:** Loop Closure triggers.
- **Result:** The map visibly "Snaps" or warps. The corridor straightens out.
- **Task:** Verify `map` frame jumps relative to `odom` frame. (TF tree update).

---

## 🚀 Project: "Resume Mapping"

**Goal:** Map Floor 1 today. Shutdown. Map Floor 1 Extension tomorrow.
1.  **Map:** Drive and create `floor1`.
2.  **Service Call:** `ros2 service call /slam_toolbox/serialize_map ... "filename: floor1"`.
3.  **Restart:** Kill node. Restart.
4.  **Service Call:** `ros2 service call /slam_toolbox/deserialize_map ... "filename: floor1"`.
5.  **Continue:** Drive into unmapped area.
6.  **Result:** New area is stitched seamlessly to `floor1`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Map is smearing" (Double Walls)
*   **Symptom:** Walls appear thick or double.
*   **Cause:** Odometry is terrible (slipping wheels). Or Lidar TF is wrong (Rotation offset).
*   **Fix:** Calibrate wheel odometry using `basic_intertial_navigation` tests. Or Enable `use_scan_matching_odometry` (Experimental, uses Lidar to fix Odom).

#### 2. "Loop Closure messed up the map"
*   **Symptom:** Rooms overlap after loop closure. False Positive match.
*   **Cause:** Corridor A looks identical to Corridor B (Perceptual Aliasing).
*   **Fix:** Increase `loop_match_minimum_response_fine` (Make it stricter).

---

## ⚡ Optimization: Map Merging

Scenario: 2 Robots mapping separately.
1.  Robot A saves `map_A.posegraph`.
2.  Robot B saves `map_B.posegraph`.
3.  **Merge Tool:** (Provided by SLAM Toolbox).
    *   Find relative transform between A and B (requires common landmark or manual alignment).
    *   Fuse Graphs.
    *   Output `merged_map`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Difference between `.pgm` (Map Server) and `.posegraph` (SLAM Toolbox)?
    *   **A:** `.pgm` is a static image (Occupancy Grid). It is "dead". `.posegraph` is the raw data (Nodes+Edges). It is "alive" and editable.
2.  **Q:** What is "Ceres Solver"?
    *   **A:** A non-linear least squares solver library by Google. Used to optimize the graph.
3.  **Q:** Why not just use AMCL?
    *   **A:** AMCL requires a *pre-existing* map. It cannot handle changes (moving furniture). SLAM updates the map live.

### Challenge Task
> **Task:** Localization Mode.
> 1. Use SLAM Toolbox in `localization` mode.
> 2. It loads a map, keeps the graph static, but runs scan matching to localize using the graph.
> 3. Does it consume less CPU than AMCL? Compare.

---

## 📚 Further Reading
- **Steve Macenski's Paper:** "SLAM Toolbox: SLAM for the dynamic world".
- **Google Cartographer:** Alternative GraphSLAM (more complex, high CPU).

---

**Day 72 Complete**
