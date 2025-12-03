# Day 163: Environment Mapping
## Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking

---

> **📝 Day 163 Focus:**
> You can't park if you don't know where the spots are. Today, we build the **HD Map** of the parking lot. Unlike highway maps, parking maps need **Slot Geometry** (position, orientation, ID) and **Drivable Aisles**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Generate** an Occupancy Grid Map using SLAM (Gmapping/Cartographer).
2.  **Annotate** Parking Slots (Semantic Map) on top of the grid.
3.  **Define** the Lanelet2 format for parking lots.
4.  **Implement** a Map Server to publish map and slots.
5.  **Handle** Multi-floor mapping (Z-axis).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 132:** HD Maps (Lanelet2).
-   **Day 131:** Graph SLAM.

### Hardware Requirements
-   **Lidar/Camera:** For data collection (Simulated).

### Software Stack
-   **ROS 2:** `nav2_map_server`, `slam_toolbox`.
-   **Tools:** `JOSM` (for Lanelet2 editing).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Base Map (Geometric)

-   **Occupancy Grid:** A 2D array where 0=Free, 100=Occupied, -1=Unknown.
-   **Source:** Lidar SLAM (Gmapping) or Camera SLAM (ORB-SLAM).
-   **Resolution:** High resolution needed (5cm) to capture pillars and curbs accurately.

### 🔹 Part 2: The Semantic Map (Topological)

-   **Parking Slot:** Defined by 4 corner points + Entry point.
-   **ID:** "A-101".
-   **Type:** "Standard", "Handicap", "Compact".
-   **Aisle:** The path connecting slots. Directionality (One-way).

### 🔹 Part 3: Lanelet2 for Parking

-   **Lanelet:** Represents a drivable strip (Aisle).
-   **Area:** Represents a parking spot.
-   **Regulatory Element:** "No Entry", "Speed Limit 5 km/h".

---

## 💻 Implementation: Mapping the Lot

**Scenario:**
-   Drive the car manually in CARLA (Town04 Parking Garage).
-   Record Lidar data.
-   Generate a Map.
-   Annotate spots.

### 🛠️ Setup
Create `week24_capstone/maps`.

### 👨‍💻 Code: Map Server Node

This node publishes the Grid Map and the Parking Spot List.

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import yaml
import os

class ParkingMapServer(Node):
    def __init__(self):
        super().__init__('parking_map_server')
        
        # Pubs
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', 1, True) # Latched
        self.pub_slots = self.create_publisher(MarkerArray, '/parking_slots', 1, True)
        
        # Load Map
        self.load_map()
        self.load_slots()

    def load_map(self):
        # In a real system, use nav2_map_server
        # Here we mock a simple 100x100 grid
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = 0.1
        grid.info.width = 100
        grid.info.height = 100
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = 0.0
        grid.data = [0] * (100*100) # Empty map
        
        # Add walls (Mock)
        for i in range(100):
            grid.data[i] = 100 # Bottom wall
            grid.data[9900+i] = 100 # Top wall
            
        self.pub_map.publish(grid)
        self.get_logger().info("Map Published")

    def load_slots(self):
        # Load from YAML
        # slots:
        #   - id: 1
        #     corners: [[10, 10], [12.5, 10], [12.5, 15], [10, 15]]
        
        markers = MarkerArray()
        
        # Mock Slot 1
        m = Marker()
        m.header.frame_id = "map"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.1
        m.color.a = 1.0
        m.color.g = 1.0
        
        # 2.5m x 5m spot
        p1 = Point(x=10.0, y=10.0, z=0.0)
        p2 = Point(x=12.5, y=10.0, z=0.0)
        p3 = Point(x=12.5, y=15.0, z=0.0)
        p4 = Point(x=10.0, y=15.0, z=0.0)
        
        m.points = [p1, p2, p3, p4, p1] # Loop
        markers.markers.append(m)
        
        self.pub_slots.publish(markers)
        self.get_logger().info("Slots Published")

def main(args=None):
    rclpy.init(args=args)
    node = ParkingMapServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

### 📄 Slot Config (`maps/slots.yaml`)

```yaml
slots:
  - id: 1
    type: standard
    pose: [11.25, 12.5, 1.57] # Center x, y, theta
    corners: [[10, 10], [12.5, 10], [12.5, 15], [10, 15]]
  - id: 2
    type: standard
    pose: [13.75, 12.5, 1.57]
    corners: [[12.5, 10], [15, 10], [15, 15], [12.5, 15]]
```

---

## 🔬 Lab Exercise: Mapping in CARLA

### Lab Objectives
1.  **Launch SLAM:**
    ```bash
    ros2 launch slam_toolbox online_async_launch.py
    ```
2.  **Drive:**
    -   Drive around the parking lot in CARLA.
    -   Ensure loop closure (return to start).
3.  **Save Map:**
    ```bash
    ros2 run nav2_map_server map_saver_cli -f my_parking_lot
    ```
4.  **Annotate:**
    -   Open the map image (`.pgm`) in GIMP/Photoshop.
    -   Find pixel coordinates of parking lines.
    -   Convert to World coordinates (using resolution/origin).
    -   Write `slots.yaml`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Map Drift
**Symptom:** Walls are double/blurry.
**Cause:** Poor Odometry (Wheel slip on concrete) or lack of features (Long corridors).
**Solution:** Use **IMU** to constrain rotation. Use **Graph SLAM** (Day 131) to optimize the loop.

#### 2. Dynamic Obstacles
**Symptom:** Map contains "ghost" cars that moved away.
**Cause:** SLAM assumes static world.
**Solution:** Post-process the map to remove transient obstacles, or use a "Life-long Mapping" algorithm.

---

## ⚡ Optimization & Best Practices

### 1. Multi-Floor Mapping
-   Standard 2D grids fail with ramps.
-   **Solution:** Use **2.5D Height Maps** or **3D Octomaps**.
-   Or, treat each floor as a separate 2D map and switch maps when the altimeter changes.

### 2. Vector Maps
-   Instead of Grid (Pixels), use Vectors (Lines).
-   Much smaller file size.
-   Easier for the Planner (Snap to line).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why do we need a Semantic Map for parking?
    *   **A:** The Occupancy Grid only tells us "Free Space". It doesn't tell us "This is a valid parking spot" vs "This is a driveway".
2.  **Q:** What is the standard size of a parking spot?
    *   **A:** Approx 2.5m x 5.0m.
3.  **Q:** How do we handle GPS in a garage?
    *   **A:** We don't. We rely on Lidar/Visual SLAM against the pre-built map.

### Challenge Task
**Task:** Slot Occupancy.
1.  Write a node that subscribes to Lidar.
2.  Check if any points fall inside the `corners` of a slot defined in `slots.yaml`.
3.  Publish `/parking_slots_status` (Free/Occupied).

---

## 📚 Further Reading & References
-   [Lanelet2 Format](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)
-   [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox)

---

**Day 163 Complete** | Phase 4: ADAS & Robotics Systems | Week 24: Capstone Project - Autonomous Valet Parking
