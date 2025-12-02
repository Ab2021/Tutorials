# Day 25: HD Maps & Vector Maps (Lanelet2)
## Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)

---

> **📝 Day 25 Focus:**
> A robot knows *where* it is (Localization). Now it needs to know *what* is around it. Is this a left-turn lane? What is the speed limit? Where is the traffic light? **High Definition (HD) Maps** provide this semantic layer. Today, we master **Lanelet2**, the industry-standard format for autonomous driving maps.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Differentiate** between Standard Maps (Google Maps) and HD Maps (Lanelet2/OpenDRIVE).
2.  **Deconstruct** the Lanelet2 data model: Points, Linestrings, Lanelets, and Regulatory Elements.
3.  **Parse** an OSM (OpenStreetMap) file containing Lanelet2 data using Python.
4.  **Query** the map: "Which lane am I in?" and "What is the speed limit here?".
5.  **Visualize** an HD Map using Python and Matplotlib.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **XML:** Basic understanding of tags and attributes (OSM is XML-based).
-   **Geometry:** Polygons, Line Segments.
-   **Coordinate Systems:** UTM (Universal Transverse Mercator) vs WGS84 (Lat/Lon).

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `lanelet2` (if available via apt/conda), or `xml.etree.ElementTree` for raw parsing. We will write a raw parser to ensure compatibility.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is an HD Map?

Standard maps (Navigation) have meter-level accuracy and topological links (Road A connects to Road B).
**HD Maps** have centimeter-level accuracy and detailed semantics.

#### 1.1 Layers of an HD Map
1.  **Geometric Layer:** Raw point cloud / mesh of the world.
2.  **Semantic Layer:** Vector features (Lane lines, Stop lines, Traffic signs).
3.  **Topological Layer:** Graph connectivity (Lane 1 merges into Lane 2).

#### 1.2 Formats
-   **OpenDRIVE:** Complex, CAD-based standard (ASAM). Used by simulators (CARLA).
-   **Lanelet2:** Lightweight, OSM-based standard. Used by Autoware and ROS 2.

---

### 🔹 Part 2: The Lanelet2 Data Model

Lanelet2 is built on top of the OpenStreetMap (OSM) data structure.

#### 2.1 Primitives
1.  **Point:** 3D coordinate (ID, Lat, Lon, Elev).
2.  **LineString:** Ordered list of Points. Can represent a curb, a lane divider, or a stop line.
    -   *Attributes:* `type=line_thin`, `subtype=dashed`.
3.  **Polygon:** Closed LineString. Represents areas (Parking lots, Buildings).

#### 2.2 The Lanelet
The atomic unit of drivable space.
-   Defined by **Left Bound** (LineString) and **Right Bound** (LineString).
-   **Direction:** Traffic flows parallel to the bounds.
-   **Adjacency:** Lanelets can share bounds (Lane change possible).

#### 2.3 Regulatory Elements
Rules that apply to Lanelets.
-   **Traffic Light:** Links a Lanelet to a LineString (Stop Line) and a Physical Light.
-   **Speed Limit:** Attribute of the Lanelet.
-   **Right of Way:** Defines priority at intersections.

---

### 🔹 Part 3: Coordinate Systems

OSM uses **WGS84** (Latitude, Longitude).
Robots use **Cartesian** (x, y in meters).
We must project WGS84 to a local Cartesian frame (usually **UTM** or a local tangent plane).

---

## 💻 Implementation: Lanelet2 Parser & Visualizer

We will write a Python script to parse a `.osm` file (Lanelet2 format) and visualize the lanes.

### 🛠️ Setup
Create `week4_day25` and `hd_map_parser.py`.
Download a sample map (e.g., `town01.osm` from CARLA or create a dummy one). We will create a dummy one in code.

```bash
mkdir -p ~/ros2_ws/src/week4_day25
cd ~/ros2_ws/src/week4_day25
touch hd_map_parser.py map_data.osm
```

### 👨‍💻 Code: Creating a Dummy Map (XML)

First, let's create `map_data.osm` content.

```xml
<?xml version='1.0' encoding='UTF-8'?>
<osm version='0.6' generator='JOSM'>
  <!-- Points -->
  <node id='1' lat='0.00000' lon='0.00000' />
  <node id='2' lat='0.00010' lon='0.00000' />
  <node id='3' lat='0.00000' lon='0.00004' />
  <node id='4' lat='0.00010' lon='0.00004' />
  
  <node id='5' lat='0.00000' lon='0.00008' />
  <node id='6' lat='0.00010' lon='0.00008' />

  <!-- LineStrings (Lane Boundaries) -->
  <way id='101'> <!-- Left Bound of Lane 1 -->
    <nd ref='1' />
    <nd ref='2' />
    <tag k='type' v='line_thin' />
    <tag k='subtype' v='solid' />
  </way>
  
  <way id='102'> <!-- Right Bound of Lane 1 / Left of Lane 2 -->
    <nd ref='3' />
    <nd ref='4' />
    <tag k='type' v='line_thin' />
    <tag k='subtype' v='dashed' />
  </way>
  
  <way id='103'> <!-- Right Bound of Lane 2 -->
    <nd ref='5' />
    <nd ref='6' />
    <tag k='type' v='line_thin' />
    <tag k='subtype' v='solid' />
  </way>

  <!-- Lanelets -->
  <relation id='1001'> <!-- Lane 1 -->
    <member type='way' ref='101' role='left' />
    <member type='way' ref='102' role='right' />
    <tag k='type' v='lanelet' />
    <tag k='subtype' v='road' />
    <tag k='speed_limit' v='50' />
  </relation>
  
  <relation id='1002'> <!-- Lane 2 -->
    <member type='way' ref='102' role='left' />
    <member type='way' ref='103' role='right' />
    <tag k='type' v='lanelet' />
    <tag k='subtype' v='road' />
    <tag k='speed_limit' v='50' />
  </relation>
</osm>
```

### 👨‍💻 Code: The Parser

```python
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# Simple Lat/Lon to Meters (Approximation for small areas)
# 1 deg lat ~= 111,000 m
# 1 deg lon ~= 111,000 * cos(lat) m
ORIGIN_LAT = 0.0
ORIGIN_LON = 0.0
METERS_PER_DEG = 111000.0

def project(lat, lon):
    y = (lat - ORIGIN_LAT) * METERS_PER_DEG
    x = (lon - ORIGIN_LON) * METERS_PER_DEG * np.cos(np.radians(ORIGIN_LAT))
    return x, y

class LaneletMap:
    def __init__(self, filename):
        self.nodes = {} # id -> (x, y)
        self.ways = {}  # id -> [node_ids]
        self.lanelets = [] # list of dicts
        
        self.parse(filename)

    def parse(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # 1. Parse Nodes (Points)
        for node in root.findall('node'):
            nid = int(node.get('id'))
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            self.nodes[nid] = project(lat, lon)
            
        # 2. Parse Ways (LineStrings)
        for way in root.findall('way'):
            wid = int(way.get('id'))
            refs = []
            for nd in way.findall('nd'):
                refs.append(int(nd.get('ref')))
            self.ways[wid] = refs
            
        # 3. Parse Relations (Lanelets)
        for rel in root.findall('relation'):
            if rel.find("tag[@k='type'][@v='lanelet']") is not None:
                lanelet = {'id': int(rel.get('id')), 'left': [], 'right': []}
                
                # Get Bounds
                for member in rel.findall('member'):
                    role = member.get('role')
                    ref = int(member.get('ref'))
                    if role == 'left':
                        lanelet['left'] = self.ways[ref]
                    elif role == 'right':
                        lanelet['right'] = self.ways[ref]
                        
                # Get Attributes
                for tag in rel.findall('tag'):
                    lanelet[tag.get('k')] = tag.get('v')
                    
                self.lanelets.append(lanelet)

    def get_lane_centerline(self, lanelet):
        # Average left and right bounds
        left_nodes = [self.nodes[n] for n in lanelet['left']]
        right_nodes = [self.nodes[n] for n in lanelet['right']]
        
        # Assume same number of points for simplicity (Real implementation needs interpolation)
        centerline = []
        min_len = min(len(left_nodes), len(right_nodes))
        for i in range(min_len):
            cx = (left_nodes[i][0] + right_nodes[i][0]) / 2.0
            cy = (left_nodes[i][1] + right_nodes[i][1]) / 2.0
            centerline.append((cx, cy))
            
        return centerline

    def plot(self):
        plt.figure(figsize=(8, 8))
        
        for lanelet in self.lanelets:
            # Plot Left Bound (Blue)
            lx = [self.nodes[n][0] for n in lanelet['left']]
            ly = [self.nodes[n][1] for n in lanelet['left']]
            plt.plot(lx, ly, 'b-', linewidth=2, label='Boundary')
            
            # Plot Right Bound (Blue)
            rx = [self.nodes[n][0] for n in lanelet['right']]
            ry = [self.nodes[n][1] for n in lanelet['right']]
            plt.plot(rx, ry, 'b-', linewidth=2)
            
            # Plot Centerline (Red Dashed)
            center = self.get_lane_centerline(lanelet)
            cx = [p[0] for p in center]
            cy = [p[1] for p in center]
            plt.plot(cx, cy, 'r--', linewidth=1, label='Centerline')
            
            # Label Speed Limit
            if 'speed_limit' in lanelet:
                mid = center[len(center)//2]
                plt.text(mid[0], mid[1], f"{lanelet['speed_limit']} km/h", color='green')

        plt.title("Lanelet2 HD Map Visualization")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.axis('equal')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    m = LaneletMap('map_data.osm')
    m.plot()
```

---

## 🔬 Lab Exercise: Map Query

### Lab Objectives
1.  Run the parser and visualizer.
2.  **Task:** Implement a function `find_nearest_lane(x, y)`.
    -   Input: Robot position $(x, y)$.
    -   Output: ID of the Lanelet the robot is currently in.
    -   *Hint:* Use `shapely.geometry.Polygon` to check if point is inside the polygon formed by Left and Right bounds.

### Code Snippet (Shapely)
```python
from shapely.geometry import Point, Polygon

def find_nearest_lane(self, x, y):
    p = Point(x, y)
    for lanelet in self.lanelets:
        # Construct Polygon from Left + Reversed(Right)
        left_pts = [self.nodes[n] for n in lanelet['left']]
        right_pts = [self.nodes[n] for n in lanelet['right']]
        poly_pts = left_pts + right_pts[::-1]
        
        poly = Polygon(poly_pts)
        if poly.contains(p):
            return lanelet['id']
    return None
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Lat/Lon Distortion
**Symptom:** Map looks stretched.
**Cause:** Using simple scaling for projection.
**Solution:** Use `utm` library (`pip install utm`) for proper projection.

#### 2. Disconnected Lanes
**Symptom:** Gaps between lanes.
**Cause:** In Lanelet2, adjacent lanes MUST share the same LineString ID. If they use different LineStrings (even with same coordinates), they are topologically disconnected.

#### 3. XML Parsing Errors
**Cause:** Malformed OSM file.
**Solution:** Use JOSM (Java OpenStreetMap Editor) to view and validate the `.osm` file visually.

---

## ⚡ Optimization & Best Practices

### 1. R-Tree Indexing
Checking every lanelet for `contains(point)` is $O(N)$.
-   Use an **R-Tree** (Spatial Index) to quickly find candidate lanelets near the point. $O(\log N)$.

### 2. Routing Graph
Lanelet2 maps can be converted into a directed graph.
-   Nodes: Lanelets.
-   Edges: Adjacency (Lane change) or Successor (Next lane).
-   Use **Dijkstra/A*** on this graph for Global Path Planning.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between a Way and a Relation in OSM?
    *   **A:** A Way is a list of points (geometry). A Relation groups Ways to form logical structures (Lanelets).
2.  **Q:** Why do we need HD Maps if we have cameras?
    *   **A:** Redundancy and Lookahead. Maps tell you about the curve 1km away, or the lane lines covered by snow.
3.  **Q:** What is a "Regulatory Element"?
    *   **A:** A rule (Stop sign, Traffic light) tied to a specific part of the map.

### Challenge Task
**Task:** Traffic Light Association.
1.  Add a traffic light node to the XML.
2.  Add a Regulatory Element relation linking the Lanelet to the Traffic Light.
3.  Parse and visualize the link (draw a line from lane to light).

---

## 📚 Further Reading & References
-   [Lanelet2 Paper (Poggenhans et al.)](https://ieeexplore.ieee.org/document/8569777)
-   [JOSM Editor](https://josm.openstreetmap.de/) - The best tool for editing HD Maps manually.

---

**Day 25 Complete** | Phase 4: ADAS & Robotics Systems | Week 4: Localization & Mapping (SLAM)
