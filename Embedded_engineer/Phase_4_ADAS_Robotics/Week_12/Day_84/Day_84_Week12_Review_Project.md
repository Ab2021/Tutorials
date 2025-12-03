# Day 84: Week 12 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 84 Focus:**
> We have built the map, parsed it, and learned how to update it. Now, we use it. Today's project is to build a **Map-Based Navigation System** for a Campus Shuttle. The shuttle must follow a specific route, stop at stations, and obey stop signs, all driven by the HD Map.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Integrate** Map Parsing, Localization, and Planning into a navigation stack.
2.  **Implement** a Global Router (A*) on the lane graph.
3.  **Execute** traffic rules (Stop Signs) using Map Overlaps.
4.  **Simulate** a full mission: "Go from Dorm to Library".
5.  **Handle** map matching errors gracefully.

---

## 📚 Week 12 Review

### 1. HD Map Formats
-   **OpenDRIVE (.xodr):** XML-based, reference lines + offsets. Standard.
-   **Lanelet2 (.osm):** Lane segments + regulatory elements. Flexible.
-   **Apollo (.pb):** Protobuf-based, optimized for performance.

### 2. Map Layers
-   **Geometric:** Where is the curb?
-   **Semantic:** Is this a bus lane?
-   **Topological:** Which lane connects to which?

### 3. Map Matching
-   **Problem:** GPS is noisy.
-   **Solution:** HMM (Hidden Markov Model) finds the most likely sequence of road segments.

### 4. Semantic Mapping
-   **LiDAR + Deep Learning:** Building maps labeled with "Road", "Building", "Pole".
-   **Dynamic Removal:** Filtering out cars/pedestrians.

---

## 🛠️ Capstone Project: Campus Shuttle Navigation

**Goal:** Navigate a virtual shuttle on a campus map.
**Map:** A loop road with 2 Stops and 1 Intersection.
**Mission:** Start at Stop A, Drive to Stop B, Wait 5s, Return to Stop A.

### Package Structure
Create `week12_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week12_project
cd ~/ros2_ws/src/week12_project
touch shuttle_nav.py
```

### 👨‍💻 Code: The Shuttle Navigator

```python
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# --- Map Data ---
class MapNode:
    def __init__(self, id, x, y, type="ROAD"):
        self.id = id
        self.x = x
        self.y = y
        self.type = type # ROAD, STOP_SIGN, STATION
        self.neighbors = [] # List of (Node, Cost)

class CampusMap:
    def __init__(self):
        self.nodes = {}
        self.build_map()
        
    def build_map(self):
        # Simple Loop: 0 -> 1 -> 2 -> 3 -> 0
        # Node 1: Station A
        # Node 3: Station B
        # Node 2: Stop Sign
        
        coords = [
            (0, 0), (50, 0), (100, 50), (50, 100), (0, 50)
        ]
        
        # Create Nodes
        for i, (x, y) in enumerate(coords):
            self.nodes[i] = MapNode(i, x, y)
            
        # Annotate
        self.nodes[1].type = "STATION_A"
        self.nodes[2].type = "STOP_SIGN"
        self.nodes[3].type = "STATION_B"
        
        # Connect (Loop)
        self.connect(0, 1)
        self.connect(1, 2)
        self.connect(2, 3)
        self.connect(3, 4)
        self.connect(4, 0)
        
    def connect(self, id1, id2):
        n1 = self.nodes[id1]
        n2 = self.nodes[id2]
        dist = np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        n1.neighbors.append((n2, dist))

    def get_nearest_node(self, x, y):
        min_dist = float('inf')
        best_node = None
        for n in self.nodes.values():
            d = np.sqrt((n.x - x)**2 + (n.y - y)**2)
            if d < min_dist:
                min_dist = d
                best_node = n
        return best_node

# --- Navigation Stack ---
class Shuttle:
    def __init__(self, map):
        self.map = map
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.state = "IDLE" # IDLE, DRIVING, WAITING
        self.plan = [] # List of Nodes
        self.target_idx = 0
        
    def set_mission(self, start_id, end_id):
        # Simplified Routing (Just follow the list since it's a loop)
        # In real app: Use A*
        self.plan = []
        curr = self.map.nodes[start_id]
        while curr.id != end_id:
            self.plan.append(curr)
            curr = curr.neighbors[0][0] # Assume single path
        self.plan.append(self.map.nodes[end_id])
        
        self.x = self.plan[0].x
        self.y = self.plan[0].y
        self.target_idx = 1
        self.state = "DRIVING"
        print(f"Mission Set: {start_id} -> {end_id}")
        
    def update(self, dt):
        if self.state == "IDLE":
            return
            
        if self.state == "WAITING":
            # Logic handled in main loop (sleep)
            return

        # Driving Logic
        target_node = self.plan[self.target_idx]
        dx = target_node.x - self.x
        dy = target_node.y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Check for Arrival at Node
        if dist < 1.0:
            print(f"Arrived at Node {target_node.id} ({target_node.type})")
            
            # Handle Node Types
            if target_node.type == "STOP_SIGN":
                print(">> STOP SIGN! Waiting 3s...")
                self.state = "WAITING"
                return "STOP"
                
            if "STATION" in target_node.type:
                print(f">> STATION {target_node.type}! Boarding Passengers (5s)...")
                self.state = "WAITING"
                return "STATION"
            
            # Next Waypoint
            self.target_idx += 1
            if self.target_idx >= len(self.plan):
                print("Mission Complete!")
                self.state = "IDLE"
                return "DONE"
            target_node = self.plan[self.target_idx] # Update target
            
        # Move towards target
        heading = math.atan2(dy, dx)
        self.speed = 10.0 # m/s
        self.x += self.speed * math.cos(heading) * dt
        self.y += self.speed * math.sin(heading) * dt
        return "MOVING"

def main():
    campus_map = CampusMap()
    shuttle = Shuttle(campus_map)
    
    # Mission: Station A (1) to Station B (3)
    shuttle.set_mission(1, 3)
    
    # Simulation Loop
    dt = 0.1
    history_x = []
    history_y = []
    
    plt.figure(figsize=(8, 8))
    
    while shuttle.state != "IDLE":
        status = shuttle.update(dt)
        
        history_x.append(shuttle.x)
        history_y.append(shuttle.y)
        
        if status == "STOP":
            plt.pause(1.0) # Simulating wait
            shuttle.state = "DRIVING"
            shuttle.target_idx += 1 # Move past stop sign
            
        elif status == "STATION":
            plt.pause(1.0)
            shuttle.state = "IDLE" # End of mission for this demo
            
        # Viz
        plt.cla()
        # Draw Map
        for n in campus_map.nodes.values():
            plt.plot(n.x, n.y, 'ko')
            plt.text(n.x, n.y+2, f"{n.id}:{n.type}")
            for nb, _ in n.neighbors:
                plt.plot([n.x, nb.x], [n.y, nb.y], 'k-', alpha=0.3)
                
        # Draw Shuttle
        plt.plot(shuttle.x, shuttle.y, 'bs', markersize=10, label='Shuttle')
        plt.plot(history_x, history_y, 'b--', alpha=0.5)
        
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)
        plt.title("Campus Shuttle Navigation")
        plt.legend()
        plt.pause(0.01)
        
    plt.show()

if __name__ == "__main__":
    main()
```

---

## 🧪 Verification & Testing

### 1. The Stop Sign Test
-   **Observation:** The shuttle reaches Node 2 (STOP_SIGN).
-   **Result:** It prints "STOP SIGN!", pauses, and then continues.
-   **Failure Mode:** If the map didn't have the `STOP_SIGN` type, the shuttle would blow through the intersection.

### 2. The Station Stop
-   **Observation:** The shuttle reaches Node 3 (STATION_B).
-   **Result:** It stops and ends the mission.

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Map Formats
1.  **Q:** Which format is best for on-board real-time usage?
    *   **A:** Apollo (.pb) or NDS. XML/JSON are too slow to parse.
2.  **Q:** What is the difference between a Node and a Lane?
    *   **A:** A Node is a topological point (Graph). A Lane is a geometric curve (Geometry).

### Section 2: Navigation
3.  **Q:** How does the shuttle know where to stop?
    *   **A:** It queries the Map Node Type. In a real HD Map, it would check for an "Overlap" with a Stop Line.
4.  **Q:** What happens if the GPS fails?
    *   **A:** Map Matching fails. The shuttle must perform an "Emergency Stop" or switch to "Visual Lane Following".

---

## 🏆 Conclusion

Congratulations on completing Week 12!
-   You have mastered **HD Maps**.
-   You know how to read them, update them, and drive using them.

**Next Week:** We enter the world of **V2X (Vehicle-to-Everything)**. The map tells you about static things. V2X tells you about dynamic things that you can't see (e.g., a car around the corner).

---

**Day 84 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
