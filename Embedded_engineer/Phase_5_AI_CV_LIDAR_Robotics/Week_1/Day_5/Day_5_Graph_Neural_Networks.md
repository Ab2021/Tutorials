# Day 5: Graph Neural Networks for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> Robots don't just see pixels; they see relationships. "The cup is *on* the table *near* the robot."
> - **Focus:** GNNs, Scene Graphs, Message Passing, and Spatial Reasoning.
> - **Code:** Implementation of Graph Convolution Layer (GCN) and 3D Scene Graph construction.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Conceptualize** a robot's environment as a Graph (Nodes = Objects, Edges = Relations).
2.  **Derive** the Message Passing algorithm for Graph Neural Networks (GNNs).
3.  **Implement** a Graph Convolutional Network (GCN) and Graph Attention Network (GAT) in PyTorch Geometric.
4.  **Construct** a 3D Scene Graph from RGB-D data (Nodes with attributes).
5.  **Reason** about spatial relationships (e.g., "Find the mug *behind* the laptop") using Graph Traversal.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU recommended (GNNs can be memory intensive).

### Software Environment
```bash
# PyTorch Geometric (PyG) Installation
# Note: Instructions vary by CUDA version. Check pytorch-geometric.com
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install networkx  # Graph manipulation
```

### Prior Knowledge
- Adjacency Matrices.
- Feature Vectors.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Graphs in Robotics

Standard Deep Learning (CNNs, RNNs) operates on Euclidean data (Grids, Sequences).
Robotics data is often **Non-Euclidean**:
1.  **Scene Graphs:** Objects and their relationships.
2.  **Road Networks:** Intersections and lanes (HD Maps).
3.  **Multi-Agent Swarms:** Communication links between robots.

#### 1.1 The Graph Structure
A Graph $G = (V, E)$.
*   **Nodes ($V$):** Entities (e.g., "Table", "Car", "Robot"). Each node $v$ has a feature vector $h_v$ (e.g., Bounding Box, Color, Class ID).
*   **Edges ($E$):** Relationships (e.g., "On", "Near", "Approaching"). Edges can also have features $e_{uv}$ (e.g., Distance, Relative Velocity).

#### 1.2 Message Passing Paradigm
The core engine of GNNs.
To update a node's representation $h_v^{(k)}$, we aggregate information from its neighbors $\mathcal{N}(v)$.

$$ h_v^{(k+1)} = \phi \left( h_v^{(k)}, \bigoplus_{u \in \mathcal{N}(v)} \psi(h_u^{(k)}, h_v^{(k)}, e_{uv}) \right) $$

where:
*   $\psi$: Message function (e.g., MLP).
*   $\bigoplus$: Aggregation function (Sum, Max, Mean).
*   $\phi$: Update function (e.g., MLP).

#### 1.3 Graph Convolutional Network (GCN)
Simplified message passing (Kipf & Welling, 2017).
$$ H^{(k+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(k)} W^{(k)}) $$
*   Intuitively: Weighted average of neighbors' features.
*   *Limitation:* Isotropic weighting (all neighbors are equal).

#### 1.4 Graph Attention Network (GAT)
Adds attention coefficients $\alpha_{uv}$ to edges.
$$ h_v' = \sigma \left( \sum_{u \in \mathcal{N}(v)} \alpha_{uv} W h_u \right) $$
*   *Robotics use:* In multi-agent path planning, a robot should pay more attention to the neighbor on a collision course than the one moving away.

### 🔹 Part 2: 3D Scene Graphs

A **3D Scene Graph** is a structured representation of the environment.
*   **Level 1 (Metric):** Point Cloud / Mesh.
*   **Level 2 (Objects):** Instance Segmented Objects (Nodes).
*   **Level 3 (Relations):** Spatial edges ("support", "proximity") and Semantic edges ("same_category").
*   **Level 4 (Rooms/Zones):** "Kitchen", "Loop".

**Hydra (MIT)** is a state-of-the-art Spatial Perception System that builds Scene Graphs in real-time.

---

## 💻 Implementation: Building a GNN from Scratch

We will use `torch_geometric` to implement a GCN for scene classification.

### 🛠️ Project Structure
```text
day5_gnn/
├── dataset/
│   └── visual_genome_mini/
├── models/
│   ├── gcn.py
│   └── gat.py
├── scene_graph_builder.py
└── train_reasoner.py
```

### 👨‍💻 Code Implementation

#### 1. Graph Convolution Layer (`models/gcn.py`)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 2)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Normalize node features.
        return norm.view(-1, 1) * x_j
```

#### 2. Scene Graph Reasoner Network

A network that takes a scene graph and predicts the relationship between two queried nodes.

```python
class SceneGraphReasoner(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.classifier = nn.Linear(64 * 2, num_classes) # Input: concat of two nodes

    def forward(self, data, node_idx_a, node_idx_b):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        # Extract features for the two nodes we are querying
        feat_a = x[node_idx_a]
        feat_b = x[node_idx_b]
        
        # Concatenate and classify their relationship
        combined = torch.cat([feat_a, feat_b], dim=1)
        return self.classifier(combined)
```

---

## 🔬 Lab Exercise: Robot Spatial Reasoning

### 1. Lab Objectives
- Simulate object detections and bounding boxes.
- Construct a graph where Edge = Euclidean Distance < threshold.
- Task: Find "The closest 'cup' to the 'laptop'".

### 2. Step-by-Step Guide

#### Phase A: Scene Generation

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Simulate Objects [x, y, z, class_id]
objects = [
    {'id': 0, 'name': 'robot',  'pos': np.array([0, 0, 0])},
    {'id': 1, 'name': 'table',  'pos': np.array([2, 0, 0.5])},
    {'id': 2, 'name': 'laptop', 'pos': np.array([2, 0.2, 1.0])}, # On table
    {'id': 3, 'name': 'cup',    'pos': np.array([2, -0.3, 1.0])}, # On table
    {'id': 4, 'name': 'floor',  'pos': np.array([0, 0, -1.0])},
]

# Build Graph
G = nx.Graph()

# Add Nodes
for obj in objects:
    G.add_node(obj['id'], name=obj['name'], pos=obj['pos'])

# Add Geometric Edges (Proximity)
threshold = 1.0 # meters
for i in range(len(objects)):
    for j in range(i+1, len(objects)):
        dist = np.linalg.norm(objects[i]['pos'] - objects[j]['pos'])
        if dist < threshold:
            G.add_edge(i, j, weight=dist, type='near')

# Visualize
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'name')
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_labels(G, pos, labels)
plt.show()
```

#### Phase B: Reasoning Query
"Find 'cup' such that distance(cup, laptop) is minimized."

```python
def find_nearest(target_name, reference_id, graph):
    ref_pos = graph.nodes[reference_id]['pos']
    nearest_dist = float('inf')
    nearest_id = None
    
    for node in graph.nodes():
        if graph.nodes[node]['name'] == target_name:
            dist = np.linalg.norm(graph.nodes[node]['pos'] - ref_pos)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = node
                
    return nearest_id, nearest_dist

# Find laptop ID
laptop_id = [n for n in G.nodes if G.nodes[n]['name'] == 'laptop'][0]
cup_id, dist = find_nearest('cup', laptop_id, G)

print(f"Closest cup to laptop is ID {cup_id} at distance {dist:.2f}m")
```

### 3. Expected Output
- Graph visualization showing connections between table, laptop, and cup.
- Console output identifying the correct cup.

---

## 🚀 Project: Semantic Scene Graph Generation

**Goal:** Take an RGB-D image, run Detectron2 (detection), project to 3D, and build a semantic graph.

### 1. High-Level Logic

```python
class SceneGraphBuilder:
    def __init__(self):
        self.detector = Detectron2Wrapper() # Returns boxes, classes, masks
        self.graph = nx.Graph()
        
    def process_frame(self, rgb, depth, camera_intrinsics):
        # 1. Detect Objects
        detections = self.detector(rgb)
        
        # 2. Estimate 3D centroids
        for det in detections:
            u, v = det['center']
            z = depth[v, u] # Simple; usually needs median of mask
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # 3. Add to Graph (with data association/tracking)
            self.update_graph(det['class'], [x, y, z])
            
        # 4. Compute Relationships
        self.compute_edges()
        
    def compute_edges(self):
        # Predicates: "On", "Support"
        for node_a in self.graph.nodes:
            for node_b in self.graph.nodes:
                if self.is_supporting(node_a, node_b):
                    self.graph.add_edge(node_a, node_b, relation='supporting')
                    
    def is_supporting(self, a, b):
        # Heuristic: A is directly below B and overlap in XY
        # ... geometry math ...
        return True
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Oversmoothing
*   **Symptom:** In deep GNNs (many layers), all node features converge to the same value.
*   **Cause:** Repeated aggregation creates a "washing out" effect.
*   **Fix:** Use fewer layers (2-3 usually enough for logical hops) or use Skip Connections (ResGCN).

#### 2. Scalability
*   **Symptom:** OOM on large graphs (e.g., dense point cloud graphs).
*   **Fix:** Use **GraphSAGE** (Neighbor Sampling) instead of full GCN. Only sample a fixed number of neighbors per node.

---

## ⚡ Optimization: Graph Attention

In robotics, noise is common. A "Ghost" detection should not influence the state of the "Robot" node.

**Implementation Task:**
Replace `GCNConv` with `GATConv` (from `torch_geometric`).
Observe how attention weights $\alpha_{uv}$ approximate 0 for outliers (ghost objects), effectively filtering them out of the message passing step.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** How does a GNN differ from a CNN?
    *   **A:** A CNN assumes a fixed grid neighborhood (up, down, left, right). A GNN operates on arbitrary neighborhood sizes defined by an adjacency matrix.
2.  **Q:** What is "Message Passing"?
    *   **A:** The process of a node gathering feature vectors from its neighbors, aggregating them, and updating its own state.
3.  **Q:** Why are Scene Graphs useful for "Long-Horizon Planning"?
    *   **A:** They abstract low-level pixels into high-level concepts ("Kitchen", "Fridge"). A planner can say "Go to Fridge", which is a node traversal, rather than pixel-level trajectory optimization.

### Challenge Task
> **Task:** Implement a "Temporal Scene Graph".
> 1. Nodes have a time dimension $T$.
> 2. Track objects across frames.
> 3. Add temporal edges: $Node_A(t) \to Node_A(t+1)$.
> 4. Use a GNN to predict the future position of Node A (Trajectory Prediction).

---

## 📚 Further Reading
- **GCN:** Kipf & Welling (ICLR 2017).
- **GAT:** Velickovic et al. (ICLR 2018).
- **3D Scene Graph:** Armeni et al. (ICCV 2019).
- **Hydra:** Hughes et al. (RSS 2022).

---

**Day 5 Complete**
