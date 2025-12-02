# Day 79: Map Layers (Lanes, Signs, Signals)
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 79 Focus:**
> A map is like an onion; it has layers. The **Geometric Layer** keeps you on the road. The **Semantic Layer** tells you the rules. The **Dynamic Layer** tells you about traffic. Today, we build a query engine to extract intelligence from the map.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Decompose** an HD Map into functional layers: Geometric, Semantic, and Topological.
2.  **Model** static elements: Traffic Signs, Signals, and Pole-like objects.
3.  **Implement** a Spatial Index (Quadtree) for fast map queries ($O(\log N)$).
4.  **Query** the map for attributes: "What is the speed limit at my current location?"
5.  **Visualize** map layers using different colors and markers.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 78:** HD Map Format.
-   **Data Structures:** Trees (Quadtree/R-Tree).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Layered Map Model

1.  **Geometric Layer (Base):**
    -   Raw point clouds, mesh of the road surface, curb lines.
    -   Used for: Localization (Lidar matching).
2.  **Topological Layer (Graph):**
    -   Lane connectivity (Predecessor/Successor).
    -   Used for: Global Routing (A*).
3.  **Semantic Layer (Attributes):**
    -   Lane types (Bus, HOV), Speed limits, Turn restrictions.
    -   Used for: Behavior Planning.
4.  **Landmark Layer (Features):**
    -   Traffic signs, Poles, Traffic lights.
    -   Used for: Visual Localization and Perception validation.

### 🔹 Part 2: Spatial Indexing

A map might cover a whole city.
Linear search (`for lane in all_lanes`) is too slow ($O(N)$).
**Quadtree:**
-   Recursively divide 2D space into 4 quadrants.
-   Store objects in leaf nodes.
-   Query: "Find all objects in this box". Complexity: $O(\log N)$.

---

## 💻 Implementation: Map Query Engine

**Scenario:**
-   **Map:** A collection of Lanes and Signs scattered in 2D space.
-   **Task:** Find the nearest lane and speed limit for a given robot position $(x, y)$.

### 🛠️ Setup
Create `week12_day79` and `map_query.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day79
cd ~/ros2_ws/src/week12_day79
touch map_query.py
```

### 👨‍💻 Code: Quadtree Map

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Map Elements ---
class LaneSegment:
    def __init__(self, id, x1, y1, x2, y2, speed_limit):
        self.id = id
        self.p1 = np.array([x1, y1])
        self.p2 = np.array([x2, y2])
        self.speed_limit = speed_limit
        
    def get_bounds(self):
        min_x = min(self.p1[0], self.p2[0])
        max_x = max(self.p1[0], self.p2[0])
        min_y = min(self.p1[1], self.p2[1])
        max_y = max(self.p1[1], self.p2[1])
        return (min_x, min_y, max_x, max_y)

    def distance(self, p):
        # Distance from point p to line segment p1-p2
        p1 = self.p1
        p2 = self.p2
        d = p2 - p1
        if np.dot(d, d) == 0: return np.linalg.norm(p - p1)
        t = np.dot(p - p1, d) / np.dot(d, d)
        t = max(0, min(1, t))
        projection = p1 + t * d
        return np.linalg.norm(p - projection)

class TrafficSign:
    def __init__(self, id, x, y, type):
        self.id = id
        self.pos = np.array([x, y])
        self.type = type # STOP, YIELD, LIMIT_50
        
    def get_bounds(self):
        # Point object, small bounds
        return (self.pos[0]-1, self.pos[1]-1, self.pos[0]+1, self.pos[1]+1)

# --- Quadtree ---
class Quadtree:
    def __init__(self, bounds, capacity=4):
        self.bounds = bounds # (min_x, min_y, max_x, max_y)
        self.capacity = capacity
        self.objects = []
        self.divided = False
        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None
        
    def insert(self, obj):
        if not self.intersects(self.bounds, obj.get_bounds()):
            return False
            
        if len(self.objects) < self.capacity:
            self.objects.append(obj)
            return True
            
        if not self.divided:
            self.subdivide()
            
        if self.nw.insert(obj): return True
        if self.ne.insert(obj): return True
        if self.sw.insert(obj): return True
        if self.se.insert(obj): return True
        return False
        
    def subdivide(self):
        x1, y1, x2, y2 = self.bounds
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        self.nw = Quadtree((x1, mid_y, mid_x, y2), self.capacity)
        self.ne = Quadtree((mid_x, mid_y, x2, y2), self.capacity)
        self.sw = Quadtree((x1, y1, mid_x, mid_y), self.capacity)
        self.se = Quadtree((mid_x, y1, x2, mid_y), self.capacity)
        self.divided = True
        
    def query(self, range_bounds, found):
        if not self.intersects(self.bounds, range_bounds):
            return
            
        for obj in self.objects:
            if self.intersects(obj.get_bounds(), range_bounds):
                found.append(obj)
                
        if self.divided:
            self.nw.query(range_bounds, found)
            self.ne.query(range_bounds, found)
            self.sw.query(range_bounds, found)
            self.se.query(range_bounds, found)
            
    def intersects(self, b1, b2):
        # b: (min_x, min_y, max_x, max_y)
        return not (b2[0] > b1[2] or b2[2] < b1[0] or 
                    b2[1] > b1[3] or b2[3] < b1[1])

    def draw(self, ax):
        rect = patches.Rectangle((self.bounds[0], self.bounds[1]), 
                                 self.bounds[2]-self.bounds[0], 
                                 self.bounds[3]-self.bounds[1], 
                                 linewidth=1, edgecolor='gray', facecolor='none')
        ax.add_patch(rect)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.sw.draw(ax)
            self.se.draw(ax)

def main():
    # 1. Create Map
    qt = Quadtree((0, 0, 100, 100))
    
    # Lanes
    lanes = []
    for i in range(10):
        # Horizontal lanes
        l = LaneSegment(i, 0, i*10, 100, i*10, 30 + i*5)
        qt.insert(l)
        lanes.append(l)
        
    # Signs
    signs = []
    s1 = TrafficSign(100, 50, 50, "STOP")
    qt.insert(s1)
    signs.append(s1)
    
    # 2. Query
    robot_pos = np.array([55, 42]) # Near Lane 4 (y=40)
    search_radius = 5.0
    search_bounds = (robot_pos[0]-search_radius, robot_pos[1]-search_radius,
                     robot_pos[0]+search_radius, robot_pos[1]+search_radius)
    
    found_objects = []
    qt.query(search_bounds, found_objects)
    
    print(f"Robot at {robot_pos}")
    print(f"Found {len(found_objects)} objects in range:")
    
    nearest_lane = None
    min_dist = float('inf')
    
    for obj in found_objects:
        if isinstance(obj, LaneSegment):
            d = obj.distance(robot_pos)
            print(f" - Lane {obj.id}: Dist={d:.2f}m, Limit={obj.speed_limit} km/h")
            if d < min_dist:
                min_dist = d
                nearest_lane = obj
        elif isinstance(obj, TrafficSign):
            print(f" - Sign {obj.type} at {obj.pos}")
            
    if nearest_lane:
        print(f"-> Snapped to Lane {nearest_lane.id}. Speed Limit: {nearest_lane.speed_limit}")
        
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    qt.draw(ax)
    
    for l in lanes:
        ax.plot([l.p1[0], l.p2[0]], [l.p1[1], l.p2[1]], 'b-')
    for s in signs:
        ax.plot(s.pos[0], s.pos[1], 'rs')
        
    ax.plot(robot_pos[0], robot_pos[1], 'go', label='Robot')
    
    # Draw Search Box
    rect = patches.Rectangle((search_bounds[0], search_bounds[1]), 
                             search_bounds[2]-search_bounds[0], 
                             search_bounds[3]-search_bounds[1], 
                             linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Quadtree Map Query")
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Speed Trap

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** The robot at $(55, 42)$ finds Lane 4 ($y=40$) and Lane 5 ($y=50$) in the search box.
3.  **Experiment:**
    -   Move robot to $(50, 50)$.
    -   It should find the `STOP` sign.
    -   **Logic:** If `STOP` sign found within 10m, the planner should prepare to stop.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Boundary Errors
**Symptom:** Objects on the edge of a quadrant are missed.
**Cause:** Floating point comparison errors or strict inequality.
**Solution:** Use a small epsilon or overlapping boundaries.

#### 2. Unbalanced Tree
**Symptom:** Query is slow.
**Cause:** All objects are in one corner. The tree becomes a linked list.
**Solution:** Use **R-Tree** (Dynamic bounding boxes) instead of Quadtree (Fixed grid) for better balance with sparse data.

---

## ⚡ Optimization & Best Practices

### 1. R-Tree
For road networks (long, thin lines), Quadtrees are inefficient (lines cross many nodes).
-   **R-Tree:** Groups objects by proximity into bounding boxes. Much better for spatial queries of lanes.
-   Python: `rtree` library.

### 2. Layered Query
Don't query everything.
-   `get_nearest_lane(pos)` -> Geometric Layer.
-   `get_speed_limit(lane_id)` -> Semantic Layer (Hash Map lookup, $O(1)$).
-   Don't search for speed limits spatially if they are linked to lanes!

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Geometric and Semantic layers?
    *   **A:** Geometric = Where is it? (Coordinates). Semantic = What is it? (Rules/Type).
2.  **Q:** Why use a Spatial Index?
    *   **A:** To avoid checking every object in the map. It reduces complexity from $O(N)$ to $O(\log N)$.
3.  **Q:** What is a "Landmark" in mapping?
    *   **A:** A distinct, static feature (Sign, Pole) used for localization.

### Challenge Task
**Task:** 3D Quadtree (Octree).
1.  Extend the Quadtree to 3 dimensions (Octree).
2.  Divide space into 8 octants.
3.  Useful for 3D Point Cloud maps (LiDAR).

---

## 📚 Further Reading & References
-   [Introduction to Spatial Indexing](https://en.wikipedia.org/wiki/Spatial_database)
-   [R-Tree Algorithm](https://en.wikipedia.org/wiki/R-tree)

---

**Day 79 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
