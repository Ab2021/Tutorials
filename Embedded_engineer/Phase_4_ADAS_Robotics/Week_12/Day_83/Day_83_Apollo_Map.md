# Day 83: Apollo HD Map
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 83 Focus:**
> Baidu Apollo is the Android of Autonomous Driving. Its HD Map format is a modification of OpenDRIVE, optimized for real-time querying. Today, we dive into the **Apollo Map**, understanding its Protobuf structure and how the Planning module uses it.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Analyze** the Apollo Map structure (Protobuf based).
2.  **Differentiate** between `base_map`, `routing_map`, and `sim_map`.
3.  **Explain** the concept of "Overlaps" (Logical connections between objects).
4.  **Generate** a simple Apollo-compatible map file.
5.  **Visualize** the map using a Python script (simulating Dreamview).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 78:** OpenDRIVE.
-   **Protobuf:** Google Protocol Buffers.

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `protobuf`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Apollo Map Format (`map.proto`)

Apollo uses **Protocol Buffers** (.pb) instead of XML (.xodr) for speed and size.
Key components:
-   **Header:** Version, Projection (UTM zone).
-   **Lane:** Central curve, Left/Right boundaries, Predecessor/Successor.
-   **Junction:** Polygon defining the intersection area.
-   **Signal:** Traffic lights.
-   **StopSign:** Stop locations.
-   **Overlap:** A critical concept. If a Lane crosses a Crosswalk, they have an "Overlap". This links the geometry (Lane) to the rule (Yield to Pedestrian).

### 🔹 Part 2: Map Types

1.  **Base Map (`base_map.bin`):** The full geometric and semantic map. Used by Perception and Localization.
2.  **Routing Map (`routing_map.bin`):** A topology graph (Node/Edge) extracted from the Base Map. Used by A* Routing.
3.  **Sim Map (`sim_map.bin`):** Lightweight version for Dreamview visualization.

### 🔹 Part 3: The Coordinate System

Apollo uses **UTM (Universal Transverse Mercator)**.
-   All coordinates are absolute $(x, y, z)$ in meters.
-   Unlike OpenDRIVE (which uses reference line $s, t$), Apollo stores the absolute geometry of lane boundaries.
-   This makes "Point-in-Polygon" checks faster.

---

## 💻 Implementation: Apollo Map Generator

**Scenario:**
-   **Task:** Create a valid Apollo Map Protobuf message containing one Lane and one Stop Sign.
-   **Note:** We will simulate the Protobuf classes since compiling the actual Apollo proto files requires the full Bazel build system.

### 🛠️ Setup
Create `week12_day83` and `apollo_map_gen.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day83
cd ~/ros2_ws/src/week12_day83
touch apollo_map_gen.py
```

### 👨‍💻 Code: Apollo Map Builder (Python Simulation)

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Mocking Apollo Protobuf Classes ---
# In a real Apollo env, you would import modules.map.proto.map_pb2
class Point:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class CurveSegment:
    def __init__(self, points):
        self.line_segment = points # List of Point

class Curve:
    def __init__(self):
        self.segment = [] # List of CurveSegment

class LaneBoundaryType:
    DOTTED_YELLOW = 1
    SOLID_YELLOW = 2

class LaneBoundary:
    def __init__(self):
        self.curve = Curve()
        self.type = LaneBoundaryType.DOTTED_YELLOW

class Lane:
    def __init__(self, id):
        self.id = id
        self.central_curve = Curve()
        self.left_boundary = LaneBoundary()
        self.right_boundary = LaneBoundary()
        self.length = 0.0
        self.speed_limit = 0.0
        self.overlap_id = []

class StopSign:
    def __init__(self, id):
        self.id = id
        self.stop_line = [] # List of Curve
        self.overlap_id = []

class Map:
    def __init__(self):
        self.lane = []
        self.stop_sign = []

# --- Map Generator ---
class ApolloMapBuilder:
    def __init__(self):
        self.map = Map()
        
    def add_lane(self, id, start_pt, end_pt, width=3.5):
        lane = Lane(id)
        
        # Central Curve (Line)
        p1 = Point(start_pt[0], start_pt[1])
        p2 = Point(end_pt[0], end_pt[1])
        lane.central_curve.segment.append(CurveSegment([p1, p2]))
        
        # Length
        lane.length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        
        # Boundaries (Offset)
        # Vector along lane
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        heading = np.arctan2(dy, dx)
        
        # Normal vector (Left)
        nx = -np.sin(heading)
        ny = np.cos(heading)
        
        # Left Boundary
        l1 = Point(p1.x + nx * width/2, p1.y + ny * width/2)
        l2 = Point(p2.x + nx * width/2, p2.y + ny * width/2)
        lane.left_boundary.curve.segment.append(CurveSegment([l1, l2]))
        
        # Right Boundary
        r1 = Point(p1.x - nx * width/2, p1.y - ny * width/2)
        r2 = Point(p2.x - nx * width/2, p2.y - ny * width/2)
        lane.right_boundary.curve.segment.append(CurveSegment([r1, r2]))
        
        self.map.lane.append(lane)
        return lane
        
    def add_stop_sign(self, id, x, y, heading, width=3.5):
        sign = StopSign(id)
        
        # Stop Line (Perpendicular to heading)
        nx = -np.sin(heading)
        ny = np.cos(heading)
        
        # Line centered at x,y
        p1 = Point(x + nx * width/2, y + ny * width/2)
        p2 = Point(x - nx * width/2, y - ny * width/2)
        
        c = Curve()
        c.segment.append(CurveSegment([p1, p2]))
        sign.stop_line.append(c)
        
        self.map.stop_sign.append(sign)
        return sign

    def visualize(self):
        plt.figure(figsize=(10, 6))
        
        # Draw Lanes
        for lane in self.map.lane:
            # Center
            seg = lane.central_curve.segment[0].line_segment
            plt.plot([seg[0].x, seg[1].x], [seg[0].y, seg[1].y], 'g--', label='Center')
            
            # Left
            seg = lane.left_boundary.curve.segment[0].line_segment
            plt.plot([seg[0].x, seg[1].x], [seg[0].y, seg[1].y], 'k-', label='Boundary')
            
            # Right
            seg = lane.right_boundary.curve.segment[0].line_segment
            plt.plot([seg[0].x, seg[1].x], [seg[0].y, seg[1].y], 'k-')
            
        # Draw Stop Signs
        for sign in self.map.stop_sign:
            # Stop Line
            seg = sign.stop_line[0].segment[0].line_segment
            plt.plot([seg[0].x, seg[1].x], [seg[0].y, seg[1].y], 'r-', linewidth=3, label='Stop Line')
            plt.text(seg[0].x, seg[0].y, "STOP", color='r', fontsize=12, weight='bold')
            
        plt.title("Apollo HD Map Visualization")
        plt.xlabel("UTM Easting (m)")
        plt.ylabel("UTM Northing (m)")
        plt.axis('equal')
        plt.grid()
        plt.show()

def main():
    builder = ApolloMapBuilder()
    
    # Create a Road Network
    # Lane 1: (0,0) to (50, 0)
    builder.add_lane("lane_1", (0, 0), (50, 0))
    
    # Lane 2: (50,0) to (100, 20) - Turn
    builder.add_lane("lane_2", (50, 0), (100, 20))
    
    # Stop Sign at end of Lane 1
    builder.add_stop_sign("stop_1", 48, 0, 0.0)
    
    print("Generated Apollo Map with 2 Lanes and 1 Stop Sign.")
    builder.visualize()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Overlap

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** You see the lanes and the stop line.
3.  **Missing Link:** The car on `lane_1` doesn't know it needs to stop for `stop_1`.
4.  **Experiment:**
    -   Create an **Overlap**.
    -   `overlap = Overlap(id="overlap_1")`
    -   `overlap.object.append(LaneOverlap("lane_1"))`
    -   `overlap.object.append(StopSignOverlap("stop_1"))`
    -   Add this ID to `lane.overlap_id` and `stop_sign.overlap_id`.
    -   **Result:** Now the Planner knows: "When I am on Lane 1, I am affected by Stop Sign 1".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Protobuf Version Mismatch
**Symptom:** `DecodeError`.
**Cause:** Apollo uses specific Protobuf versions.
**Solution:** Always use the Docker container provided by Apollo to generate maps.

#### 2. Discontinuous Lanes
**Symptom:** Routing fails.
**Cause:** The end point of Lane 1 is $(50.0, 0.0)$ but start of Lane 2 is $(50.01, 0.0)$.
**Solution:** Snap points. Apollo requires strict geometric continuity (< 1cm gap) for topology generation.

---

## ⚡ Optimization & Best Practices

### 1. Smoothing
Raw map data (from survey) is noisy.
-   **Spline Smoothing:** Fit a Cubic Spline to the lane points to ensure smooth curvature (kappa).
-   Jerky curvature = Jerky steering.

### 2. Map Validation
Before using a map, run validation checks:
-   Are all lanes connected?
-   Do all traffic lights have stop lines?
-   Are there any self-intersections?

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why does Apollo use Protobuf?
    *   **A:** It's binary (smaller size), strongly typed (safer), and fast to serialize/deserialize compared to XML/JSON.
2.  **Q:** What is the difference between `Base Map` and `Routing Map`?
    *   **A:** Base Map has geometry (curves). Routing Map has topology (graph). Routing Map is derived from Base Map.
3.  **Q:** What is an "Overlap"?
    *   **A:** A logical association between two map elements (e.g., Lane and Signal) that share a spatial region.

### Challenge Task
**Task:** Junction Generation.
1.  Define a `Junction` polygon that covers the area where Lane 1 meets Lane 2.
2.  Add `Junction` to the map.
3.  Add Overlaps between Lanes and Junction.

---

## 📚 Further Reading & References
-   [Apollo Map Data Structure](https://github.com/ApolloAuto/apollo/blob/master/modules/map/proto/map.proto)
-   [LGSVL Simulator (Supports Apollo Maps)](https://www.lgsvlsimulator.com/)

---

**Day 83 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
