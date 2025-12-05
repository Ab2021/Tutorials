# Day 73: Costmap Layers (Prohibition/Voxel)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 11: Navigation 2 (Nav2) Mastery

---

> **📝 Content Creator Instructions:**
> The world is not 2D. The costmap shouldn't be either.
> - **Focus:** 2D vs 3D Layers, Keeper Zones (Prohibition Layer), Voxel Grid (Raytracing 3D), and writing a Custom C++ Costmap Layer.
> - **Code:** A custom `GradientLayer` plugin that adds soft costs based on Wi-Fi signal strength (or any scalar field).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** the `VoxelLayer` to use PointCloud2 data for 3D obstacle avoidance (Overhangs).
2.  **Define** "Keep Out Zones" using the `KeepoutFilter` (Virtual Walls).
3.  **Implement** a custom Costmap2D Plugin in C++ (`nav2_costmap_2d::Layer`).
4.  **Debug** Ghost Obstacles using Raytracing parameters.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Depth Camera or 3D Lidar (Velodyne).

### Software Environment
```bash
sudo apt install ros-humble-nav2-voxel-grid
sudo apt install ros-humble-nav2-simple-commander
```

### Prior Knowledge
- Occupancy Grids (0=Free, 100=Occupied, 255=Unknown).
- Inflation Radius.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Layered Costmap Architecture

The Master Costmap is a summation of layers.
1.  **Static Layer:** The Map (Walls).
2.  **Obstacle Layer (2D):** Lidar hits.
3.  **Voxel Layer (3D):** Point Cloud hits. Handles Table legs vs Table tops.
4.  **Inflation Layer:** Expands lethal obstacles by robot radius + safety buffer.
5.  **Prohibition Layer (Costmap Filters):** User-defined zones (e.g., "Don't go near the server rack").

### 🔹 Part 2: Voxel Grid Logic

*   **Marking:** Sensor hit at $(x,y,z)$ marks voxel as Occupied.
*   **Raytracing (Clearing):** If sensor passes through $(x,y,z)$ and sees nothing behind it, clear the voxel.
*   **3D to 2D Projection:** The Master Costmap is 2D.
    *   If *any* voxel in the vertical column $(x,y, z_{min}:z_{max})$ is occupied, the 2D cell $(x,y)$ is marked Lethal.
    *   This allows the robot to drive *under* a bridge (clear voxels at robot height) but avoid the bridge columns.

### 🔹 Part 3: Costmap Filters

*   **Mask:** An image file (PGM) where black pixels = Special Zone.
*   **Keepout Filter:** Black pixels = LETHAL_OBSTACLE.
*   **Speed Limit Filter:** Black pixels = Max Velocity 0.5 m/s.

---

## 💻 Implementation: Custom Gradient Layer

We will create a C++ plugin that reads a user prompt or topic and applies a gradient cost. E.g., "Avoid the center of the room".

### 🛠️ Project Structure
```text
day73_layers/
├── include/day73_layers/
│   └── gradient_layer.hpp
├── src/
│   └── gradient_layer.cpp
├── plugin_description.xml
└── CMakeLists.txt
```

### 👨‍💻 Header (`gradient_layer.hpp`)

```cpp
#ifndef GRADIENT_LAYER_HPP
#define GRADIENT_LAYER_HPP

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"

namespace day73_layers
{
class GradientLayer : public nav2_costmap_2d::Layer
{
public:
  GradientLayer();
  virtual void onInitialize();
  virtual void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y);

  virtual void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j);

  virtual void reset() { return; }

private:
  double last_robot_x_, last_robot_y_;
  int center_cost_;
};
}
#endif
```

### 👨‍💻 Implementation (`gradient_layer.cpp`)

Adds a "Hill" of cost in the center of the map $(0,0)$.

```cpp
#include "day73_layers/gradient_layer.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(day73_layers::GradientLayer, nav2_costmap_2d::Layer)

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

namespace day73_layers
{

GradientLayer::GradientLayer() {}

void GradientLayer::onInitialize()
{
  center_cost_ = 100; // Configurable param in real usage
  current_ = true;
}

void GradientLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  // We touch the whole map potentially, so update entire bounds? 
  // For efficiency, usually we only update locally. 
  // Here we update a 10x10m area around 0,0
  *min_x = -5.0; *min_y = -5.0;
  *max_x = 5.0;  *max_y = 5.0;
}

void GradientLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;

  unsigned char * master_array = master_grid.getCharMap();
  unsigned int size_x = master_grid.getSizeInCellsX();
  
  // Convert 0,0 world coords to map coords
  unsigned int center_mx, center_my;
  master_grid.worldToMap(0.0, 0.0, center_mx, center_my);

  for (int j = min_j; j < max_j; j++) {
    for (int i = min_i; i < max_i; i++) {
        
      int index = master_grid.getIndex(i, j);
      
      // Distance from center (0,0) in cells
      double dist = sqrt(pow(i - (int)center_mx, 2) + pow(j - (int)center_my, 2));
      
      // Create a gradient: High cost at center, fading out at 50 cells distance
      int gradient = 0;
      if (dist < 50.0) {
        gradient = (int)(250 * (1.0 - (dist / 50.0)));
      }
      
      // Add to Master Grid (max logic)
      int old_cost = master_array[index];
      if (old_cost == NO_INFORMATION) continue; // Don't overwrite unknown with cost
      if (old_cost == LETHAL_OBSTACLE) continue; // Don't lower lethal

      int new_cost = std::max(old_cost, gradient);
      master_array[index] = (unsigned char)new_cost;
    }
  }
}

} // namespace
```

### 👨‍💻 Config (`nav2_params.yaml`)

Register the plugin.

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["voxel_layer", "inflation_layer", "gradient_layer"]
      gradient_layer:
        plugin: "day73_layers::GradientLayer"
        enabled: true
```

---

## 🔬 Lab Exercise: "The Invisible Wall"

### 1. Lab Objectives
- Open GIMP/Paint. Create a map image `keepout.pgm`. Draw a black line across a doorway.
- Load this as a `Costmap Filter` (Keepout Filter).
- **Task:** Send a navigation goal *through* the doorway.
- **Observation:** The planner (A*) should route *around* the building (if possible) or fail. It treats the invisible line as a wall.
- **Application:** Use this to block off "Wet Floor" areas dynamically.

---

## 🚀 Project: "3D Overhang Avoidance"

**Goal:** Navigate under a table but avoid a low-hanging lamp.
1.  **Setup:** Gazebo world with a Table ($z=0.8m$) and a Hanging Lamp ($z=0.4m$). Robot Height: 0.3m.
2.  **Voxel Layer:**
    *   `z_voxels`: 16.
    *   `z_resolution`: 0.1m.
    *   `mark_threshold`: 0 (Single hit marks it).
3.  **Test:**
    *   Drive under Table? Yes. (Voxels at $z=0.8$ are Occupied, but projection allows passing if robot is $0.3m$).
    *   Drive under Lamp? No. (Voxels at $z=0.4$ Occupied. Robot collision check fails).
    *   **Wait:** Voxel Layer projects to 2D. So *both* Table and Lamp appear as obstacles in 2D costmap?
    *   **Correction:** We configure `max_obstacle_height: 2.0`. But for the 2D projection, we typically project *everything*.
    *   **Advanced:** Use `3D Costmap` (experimental) or check collision in 3D.
    *   **Std Nav2:** We usually exclude points $> robot\_height$ from the costmap clearing/marking to allow passing under things.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Ghost Obstacles" (Clearing problem)
*   **Symptom:** Dynamic object moves away, but costmap stays Lethal.
*   **Cause:** Raytracing failed. Usually because `max_obstacle_height` < Sensor Height. If sensor says "Clear at 5m", but `obstacle_height` limit ignores it, pixels aren't cleared.
*   **Fix:** Ensure `raytrace_max_range` > `obstacle_max_range`.

#### 2. "Plugin not found"
*   **Symptom:** `Failed to load plugin 'day73_layers::GradientLayer'`.
*   **Cause:** Did not source `install/setup.bash`. Or `plugin_description.xml` missing export.
*   **Fix:** Check `colbuild` output and XML paths.

---

## ⚡ Optimization: Probabilistic Voxel Grid

Instead of Binary (Hit/Miss), store a probability.
*   `Hit`: $P = P + \log(odds)$.
*   `Miss`: $P = P - \log(odds)$.
*   Reduces noise from sparse Lidar points.
*   Implemented in `spatio_temporal_voxel_layer` (STVL) - A standard alternative to `voxel_layer`.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use Inflation Layer?
    *   **A:** To create a gradient ensuring the robot stays away from walls. The center of the robot follows the "valley" of low cost.
2.  **Q:** Difference between Keepout Filter and Static Layer?
    *   **A:** Static Layer is the permanent map. Keepout Filter is a "temporary" mask overlay (e.g., Construction Zone) that can be swapped easily.
3.  **Q:** What happens if `updateCosts` is too slow?
    *   **A:** The Control Loop frequency drops. Robot stutters. Costmap updates must be fast (< 50ms).

### Challenge Task
> **Task:** Speed Limit Zone.
> 1. Use the filter to define a zone `speed_limit.pgm`.
> 2. Set max speed to 0.2 m/s in this zone.
> 3. Observe robot slowing down (School Zone behavior) and speeding up outside.

---

## 📚 Further Reading
- **Nav2 Documentation:** "Costmap 2D Configuration".
- **STVL:** "Spatio-Temporal Voxel Layer".

---

**Day 73 Complete**
