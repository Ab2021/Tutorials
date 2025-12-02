# Day 78: HD Map Format (OpenDRIVE, Lanelet2)
## Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching

---

> **📝 Day 78 Focus:**
> A GPS dot on a blank screen is useless. The car needs to know: "Is this a left-turn lane?", "What is the speed limit here?", "Where is the traffic light?". **HD Maps** provide this centimeter-level semantic information. Today, we learn the language of maps.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between SD Maps (Navigation) and HD Maps (Perception/Planning).
2.  **Analyze** the structure of an OpenDRIVE (xodr) file: Roads, Lanes, Junctions.
3.  **Explain** the Lanelet2 format: Primitives (Points, Linestrings) and Regulatory Elements.
4.  **Parse** an OpenDRIVE XML file in Python to extract lane geometry.
5.  **Visualize** the road network geometry.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** Basic tag structure.
-   **Geometry:** Cubic Splines (used in OpenDRIVE).

### Hardware Requirements
-   **None:** Pure algorithm day.

### Software Stack
-   **Python:** `lxml`, `matplotlib`, `numpy`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: SD vs HD Maps

-   **SD Map (Google Maps):**
    -   Topological graph (Nodes = Intersections, Edges = Roads).
    -   Accuracy: ~5-10 meters.
    -   Usage: Global Routing ("Turn right in 500m").
-   **HD Map (Autonomous Driving):**
    -   Geometric description (Lane borders, Curb height).
    -   Semantic attributes (Lane type, Speed limit, Traffic light ID).
    -   Accuracy: < 10 centimeters.
    -   Usage: Local Planning, Localization, Perception validation.

### 🔹 Part 2: OpenDRIVE (.xodr)

The industry standard (ASAM).
-   **Reference Line:** The "spine" of the road (Geometry: Line, Arc, Spiral, Poly3).
-   **Lanes:** Defined as offsets from the Reference Line (Polynomial).
    -   Lane 0: Center (Reference).
    -   Lane -1, -2: Right side.
    -   Lane +1, +2: Left side.
-   **Junctions:** Connecting roads.

### 🔹 Part 3: Lanelet2 (.osm)

Used by Autoware. Based on OpenStreetMap.
-   **Lanelet:** Atomic lane segment defined by Left and Right boundaries (Linestrings).
-   **Regulatory Element:** Traffic rules linked to Lanelets (e.g., Stop Sign, Traffic Light).
-   **Area:** Parking lots, Drivable space.

---

## 💻 Implementation: OpenDRIVE Parser

**Scenario:**
-   **Input:** A simplified OpenDRIVE XML snippet.
-   **Output:** Visualization of the road geometry (Reference line + Lane borders).

### 🛠️ Setup
Create `week12_day78` and `xodr_parser.py`.

```bash
mkdir -p ~/ros2_ws/src/week12_day78
cd ~/ros2_ws/src/week12_day78
touch xodr_parser.py
```

### 👨‍💻 Code: Simple XODR Parser

```python
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# --- Sample OpenDRIVE XML (Embedded for simplicity) ---
XODR_DATA = """
<OpenDRIVE>
    <road name="ExampleRoad" length="100.0" id="1" junction="-1">
        <planView>
            <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="100.0">
                <line/>
            </geometry>
        </planView>
        <lanes>
            <laneSection s="0.0">
                <left>
                    <lane id="1" type="driving" level="false">
                        <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                </left>
                <center>
                    <lane id="0" type="none" level="false"/>
                </center>
                <right>
                    <lane id="-1" type="driving" level="false">
                        <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
                    </lane>
                </right>
            </laneSection>
        </lanes>
    </road>
</OpenDRIVE>
"""

class RoadGeometry:
    def __init__(self, s, x, y, hdg, length, type):
        self.s = float(s)
        self.x = float(x)
        self.y = float(y)
        self.hdg = float(hdg)
        self.length = float(length)
        self.type = type # line, arc, spiral, poly3

    def calc_position(self, s_local):
        # Simplified: Only handles Line geometry
        if self.type == 'line':
            x = self.x + s_local * np.cos(self.hdg)
            y = self.y + s_local * np.sin(self.hdg)
            return x, y, self.hdg
        return 0, 0, 0

class LaneWidth:
    def __init__(self, a, b, c, d):
        # Polynomial: width = a + b*ds + c*ds^2 + d*ds^3
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        
    def get_width(self, ds):
        return self.a + self.b*ds + self.c*ds**2 + self.d*ds**3

class OpenDriveParser:
    def __init__(self, xml_string):
        self.root = ET.fromstring(xml_string)
        self.roads = []
        self.parse()
        
    def parse(self):
        for road in self.root.findall('road'):
            r_data = {'id': road.get('id'), 'geometries': [], 'lanes': []}
            
            # Parse Plan View (Reference Line)
            plan_view = road.find('planView')
            for geo in plan_view.findall('geometry'):
                g_type = 'unknown'
                if geo.find('line') is not None: g_type = 'line'
                
                r_data['geometries'].append(RoadGeometry(
                    geo.get('s'), geo.get('x'), geo.get('y'), 
                    geo.get('hdg'), geo.get('length'), g_type
                ))
                
            # Parse Lanes (Simplified: Only first laneSection)
            lanes = road.find('lanes')
            section = lanes.find('laneSection')
            
            # Left Lanes
            left = section.find('left')
            if left:
                for lane in left.findall('lane'):
                    width = lane.find('width')
                    w_poly = LaneWidth(width.get('a'), width.get('b'), width.get('c'), width.get('d'))
                    r_data['lanes'].append({'id': int(lane.get('id')), 'width': w_poly})
                    
            # Right Lanes
            right = section.find('right')
            if right:
                for lane in right.findall('lane'):
                    width = lane.find('width')
                    w_poly = LaneWidth(width.get('a'), width.get('b'), width.get('c'), width.get('d'))
                    r_data['lanes'].append({'id': int(lane.get('id')), 'width': w_poly})
            
            self.roads.append(r_data)

    def plot(self):
        plt.figure(figsize=(10, 5))
        
        for road in self.roads:
            # Generate points along reference line
            geo = road['geometries'][0] # Assume 1 geometry for demo
            s_vals = np.linspace(0, geo.length, 100)
            
            ref_x = []
            ref_y = []
            
            # For each lane
            lane_points = {}
            for lane in road['lanes']:
                lane_points[lane['id']] = {'x': [], 'y': []}
            
            for s in s_vals:
                rx, ry, rh = geo.calc_position(s)
                ref_x.append(rx)
                ref_y.append(ry)
                
                # Calculate Lane Borders
                # Normal vector (Left)
                nx = -np.sin(rh)
                ny = np.cos(rh)
                
                for lane in road['lanes']:
                    w = lane['width'].get_width(s)
                    lid = lane['id']
                    
                    # Offset depends on ID sign
                    # ID > 0 (Left): Offset = +Width/2 (Center of lane)
                    # Wait, usually border is defined.
                    # Simplified: Plot Center of Lane
                    offset = 0
                    if lid > 0: offset = w/2 + (lid-1)*w # Simplified assumption
                    elif lid < 0: offset = -w/2 + (lid+1)*w
                    
                    lx = rx + nx * offset
                    ly = ry + ny * offset
                    lane_points[lid]['x'].append(lx)
                    lane_points[lid]['y'].append(ly)
            
            # Plot Reference
            plt.plot(ref_x, ref_y, 'k--', label='Reference Line')
            
            # Plot Lanes
            for lid, pts in lane_points.items():
                plt.plot(pts['x'], pts['y'], label=f'Lane {lid}')
                
        plt.title("OpenDRIVE Visualization")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.show()

def main():
    parser = OpenDriveParser(XODR_DATA)
    parser.plot()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Curve

### Lab Objectives
1.  Run the simulation.
2.  **Observation:** A straight road with 2 lanes (Left and Right).
3.  **Experiment:**
    -   Change the geometry type to `arc` (requires implementing `calc_position` for Arc).
    -   Arc formula:
        $$ x = x_0 + \frac{1}{\kappa} (\sin(\theta_0 + \kappa s) - \sin(\theta_0)) $$
        $$ y = y_0 - \frac{1}{\kappa} (\cos(\theta_0 + \kappa s) - \cos(\theta_0)) $$
    -   Set `curvature` ($\kappa$) to 0.05.
    -   **Result:** The road curves to the left. The lanes follow parallel curves.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Coordinate Systems
**Symptom:** Lanes overlap or go wrong way.
**Cause:** OpenDRIVE uses a specific Track Coordinate System ($s, t$). $t$ is positive to the left.
**Solution:** Ensure your normal vector calculation respects the Left-Hand Rule or Right-Hand Rule as defined by the standard.

#### 2. Polynomial Oscillations
**Symptom:** Lane width wobbles.
**Cause:** High-order polynomials (Cubic) can oscillate if coefficients are bad.
**Solution:** Visualize the width profile $w(s)$ separately to sanity check.

---

## ⚡ Optimization & Best Practices

### 1. Spatial Indexing
Parsing XML is slow. Searching for "Which lane am I in?" by iterating all lanes is $O(N)$.
-   **Grid Map / R-Tree:** Convert the vector map into a spatial index.
-   Query: `get_lanes_in_rect(x, y, w, h)`.

### 2. Lanelet2 for Planning
OpenDRIVE is great for simulation (Road creation). Lanelet2 is better for Planning (Graph search).
-   **Convert** OpenDRIVE to Lanelet2 for the navigation stack.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the "Reference Line"?
    *   **A:** The central geometry curve from which all lanes are defined as offsets.
2.  **Q:** Why do we need "Lane Sections"?
    *   **A:** Because the number of lanes changes (e.g., 2 lanes become 3). A new section starts whenever the lane configuration changes.
3.  **Q:** What is the difference between `s` and `t` coordinates?
    *   **A:** $s$ is along the road (longitudinal). $t$ is perpendicular to the road (lateral). This is the Frenet Frame!

### Challenge Task
**Task:** Lane Width Change.
1.  Modify the XML `width` element.
2.  Set `a="3.5" b="-0.02"`.
3.  Observe the lane narrowing as $s$ increases ($w = 3.5 - 0.02s$).

---

## 📚 Further Reading & References
-   [OpenDRIVE Standard](https://www.asam.net/standards/detail/opendrive/)
-   [Lanelet2 Library](https://github.com/fzi-forschungszentrum-informatik/lanelet2)

---

**Day 78 Complete** | Phase 4: ADAS & Robotics Systems | Week 12: HD Maps & Map Matching
