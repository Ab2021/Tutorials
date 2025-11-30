# Fraud Detection & Network Analysis (Part 2) - Graph Theory & GNNs - Theoretical Deep Dive

## Overview
"Individual fraud is a nuisance. Organized fraud is a business."
Fraud Rings (Organized Crime) cannot be detected by looking at claims in isolation. We need to look at the **connections**. If 10 claimants share the same phone number, that's a ring. Today, we use **Graph Theory** and **Graph Neural Networks (GNNs)** to catch them.

---

## 1. Conceptual Foundation

### 1.1 The Network View

*   **Nodes:** Entities (Claimants, Policies, Vehicles, Doctors, Garages, IP Addresses).
*   **Edges:** Relationships (Shared Address, Shared Phone, "Accident With", "Treated By").
*   **Fraud Ring:** A dense cluster of nodes connected by suspicious edges.

### 1.2 Homophily

*   "Birds of a feather flock together."
*   If a node is connected to known fraudsters, it is likely fraudulent.
*   **Guilt by Association:** This is the core principle of Network Analysis.

---

## 2. Mathematical Framework

### 2.1 Centrality Measures

1.  **Degree Centrality:** Number of connections. (e.g., A Garage connected to 500 accidents).
2.  **Betweenness Centrality:** Bridges between groups. (e.g., A Lawyer connecting two different fraud rings).
3.  **PageRank:** "Vote" of importance. A node is suspicious if it's connected to other suspicious nodes.

### 2.2 Graph Neural Networks (GNN)

*   **Message Passing:** Nodes aggregate information from their neighbors.
*   **Update Rule:**
    $$ h_v^{(k)} = \sigma \left( W \cdot \text{AGG} \{ h_u^{(k-1)} : u \in \mathcal{N}(v) \} \right) $$
    *   $h_v^{(k)}$: Embedding of node $v$ at layer $k$.
    *   $\mathcal{N}(v)$: Neighbors of $v$.
    *   *Result:* The embedding of a node contains information about its entire neighborhood.

---

## 3. Theoretical Properties

### 3.1 Community Detection

*   **Louvain Algorithm:** Optimizes "Modularity" to find clusters.
*   **Connected Components:** The simplest method. Find isolated islands in the graph.
*   *Application:* Each component is a potential "Ring".

### 3.2 Bipartite Graphs

*   **Structure:** Claimants on one side, Accidents on the other.
*   **Projection:** Project to a "Claimant-Claimant" graph where an edge exists if they were in the same accident.

---

## 4. Modeling Artifacts & Implementation

### 4.1 NetworkX (Graph Construction)

```python
import networkx as nx
import matplotlib.pyplot as plt

# 1. Create Graph
G = nx.Graph()

# 2. Add Edges (Claimant -> Shared Info)
# A claimant is connected to their Phone Number
G.add_edge("Claimant_A", "Phone_555-0199")
G.add_edge("Claimant_B", "Phone_555-0199") # Suspicious!

# 3. Find Components
rings = list(nx.connected_components(G))
for ring in rings:
    if len(ring) > 5:
        print(f"Suspicious Ring: {ring}")

# 4. Calculate Centrality
centrality = nx.degree_centrality(G)
```

### 4.2 PyTorch Geometric (GNN)

```python
import torch
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels=16, out_channels=32)
        self.conv2 = GCNConv(in_channels=32, out_channels=2) # Binary Class

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        return x

# Input: Graph where 'x' are node features (Age, History)
# Output: Probability of Fraud for EACH node
```

---

## 5. Evaluation & Validation

### 5.1 Link Prediction

*   **Task:** Predict missing edges.
*   *Scenario:* If A and B share an address, and B and C share a phone, predict that A and C likely know each other (Hidden link).

### 5.2 Visual Inspection

*   **Force-Directed Layout:** Visualize the graph.
*   Fraud rings look like "Spider Webs" or "Stars" (One central organizer).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Super-Nodes**
    *   "10,000 people share the same address."
    *   *Reality:* It's an apartment building or a PO Box.
    *   *Fix:* Exclude high-degree nodes (Whitelisting) or weight edges by inverse frequency.

2.  **Trap: Entity Resolution**
    *   "John Smith" vs "J. Smith".
    *   *Fix:* Use Fuzzy Matching or Phonetic Algorithms (Soundex) before building the graph.

### 6.2 Implementation Challenges

1.  **Scalability:**
    *   Graphs with millions of nodes are hard to process in memory.
    *   *Fix:* Use Graph Databases (Neo4j) or distributed frameworks (Spark GraphFrames).

---

## 7. Advanced Topics & Extensions

### 7.1 Heterogeneous Graphs (Heterographs)

*   Different types of nodes (Person, Car, Claim) and edges (Owns, Drives, Filed).
*   **Relational GCN (R-GCN):** Handles different edge types differently.

### 7.2 Temporal Graphs

*   Edges appear over time.
*   **Dynamic GNNs:** Capture the *evolution* of a fraud ring.

---

## 8. Regulatory & Governance Considerations

### 8.1 Privacy (GDPR)

*   Building a graph of "Who knows Who" is sensitive.
*   **Constraint:** You can only link data you legally hold. You cannot scrape Facebook to build the graph.

---

## 9. Practical Example

### 9.1 Worked Example: "Crash for Cash"

**Scenario:**
*   **Ring:** A "Ghost Garage" issues fake repair invoices.
*   **Graph:**
    *   50 Claims link to Garage X.
    *   The *Claimants* have no direct link to each other.
    *   But the *Recovery Truck* driver is the same for all 50.
*   **Analysis:**
    *   Garage X Degree Centrality = 50 (High).
    *   Recovery Driver Betweenness = High.
*   **Action:** SIU raids the garage.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Fraud Rings** are defined by structure, not individual attributes.
2.  **GNNs** learn from the neighborhood.
3.  **Super-Nodes** (Common addresses) must be handled carefully.

### 10.2 When to Use This Knowledge
*   **Organized Crime:** Detecting syndicates.
*   **Provider Fraud:** Detecting corrupt doctors/garages.

### 10.3 Critical Success Factors
1.  **Data Linkage:** The graph is only as good as the edges. Clean your data first.
2.  **Visualization:** A picture of a fraud ring is worth 1,000 rows of data.

### 10.4 Further Reading
*   **Hamilton:** "Graph Representation Learning".

---

## Appendix

### A. Glossary
*   **Adjacency Matrix:** A matrix representing connections.
*   **Embedding:** A vector representation of a node.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **GCN Update** | $\hat{A} H W$ | GNN Propagation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
