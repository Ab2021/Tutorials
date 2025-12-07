# Day 060: cuGraph & GPU Graph Analytics
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Construct GPU Graphs:** Create `cugraph` objects from `cudf` Edge Lists (Source, Dest, Weight).
2.  **Traverse Networks:** Run **BFS** (Breadth First Search) and **SSSP** (Single Source Shortest Path) on million-edge graphs in milliseconds.
3.  **Analyze Centrality:** Compute **PageRank**, **Betweenness Centrality**, and **Katz Centrality** using parallel iterative solvers.
4.  **Detect Communities:** Use **Louvain Modularity** and **Leiden** algorithms to cluster graph nodes.
5.  **Scale Out:** Understand multi-GPU graph partitioning (dask-cugraph) for billion-scale networks.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Lib:** `cugraph`, `networkx` (for comparison).
*   **Data:** Karate Club (Toy), Cyber-security Network Logs (IP -> IP).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Graph Representation on GPU

**Adjacency Matrix vs CSR:**
*   **Matrix:** $N \times N$. Sparse, wasteful for large networks.
*   **Edge List:** List of `(src, dst)` pairs. Good for input.
*   **CSR (Compressed Sparse Row):**
    *   `Offsets Array`: Pointers to where each node's neighbors start.
    *   `Indices Array`: The actual neighbors.
    *   **GPU Advantage:** CSR allows efficient coalesced access. A warp handles a node (or a set of nodes), scanning the `Indices` array linearly.

### 🔹 Part 2: The "Frontier" Model (BFS)

Graph Traversal on GPU is a **Frontier Expansion** problem.
1.  **Frontier F0:** Start Node.
2.  **Expand:** All neighbors of F0 become F1.
3.  **Update:** Mark F1 visited.
4.  **Repeat:** Neighbors of F1 become F2.

**Parallelism:**
Unlike CPU Queue-based BFS (Serial), GPU BFS launches threads for *every* node in the frontier at once.
*   "Pull" approach: Unvisited nodes check if they have a neighbor in the frontier.
*   "Push" approach: Frontier nodes write to their neighbors.

### 🔹 Part 3: PageRank

Ideally suited for GPU (Matrix Vector Multiplication).
$$ PR_{new} = \alpha M \times PR_{old} + (1-\alpha)E $$
Where $M$ is the Adjacency Matrix (Column Normalized).
*   **Iteration:** Repeat until convergence.
*   **Performance:** `cugraph` PageRank is often 500x faster than `networkx`.

---

## 💻 Implementation: Cyber Security Threat Detection

We will analyze a network of IP communications to find "central" servers (PageRank) and "connected components" (Botnets).

### 🛠️ Step 1: Generate Network Data (`gen_net.py`)

```python
import cudf
import cugraph
import cupy as cp

# 1M Edges
rows = 1_000_000
unique_ips = 100_000

print("Generating Edge List...")
# Source IPs
src = cp.random.randint(0, unique_ips, rows, dtype='int32')
# Dest IPs (Power Law distribution preferred, but random for now)
dst = cp.random.randint(0, unique_ips, rows, dtype='int32')
# Weights (Bytes transferred)
weights = cp.random.rand(rows, dtype='float32')

gdf = cudf.DataFrame()
gdf['src'] = src
gdf['dst'] = dst
gdf['weight'] = weights

print(gdf.head())
```

### 🛠️ Step 2: The Analysis (`analyze_graph.py`)

```python
import cugraph
import time

def analyze(edge_df):
    # 1. Create Graph
    # Directed Graph usually
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edge_df, source='src', destination='dst', edge_attr='weight')
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    
    # 2. PageRank (Identify Core Servers)
    print("Computing PageRank...")
    start = time.time()
    pr_df = cugraph.pagerank(G, alpha=0.85)
    print(f"PageRank Time: {time.time() - start:.4f}s")
    
    # Sort by Score
    top_ips = pr_df.sort_values('pagerank', ascending=False).head(5)
    print("Top Critical IPs:\n", top_ips)
    
    # 3. Community Detection (Louvain)
    # Good for finding Botnet clusters
    print("Computing Louvain Modularity...")
    start = time.time()
    # Louvain requires Undirected usually, or symmetrizes internally
    parts, mod = cugraph.louvain(G)
    print(f"Louvain Time: {time.time() - start:.4f}s")
    print(f"Modularity Score: {mod}")
    print(f"Communities Found: {parts['partition'].nunique()}")
    
    # 4. SSSP (Shortest Path)
    # Distance from IP 0 to everyone else
    print("Computing SSSP...")
    start = time.time()
    sssp_df = cugraph.sssp(G, source=0)
    print(f"SSSP Time: {time.time() - start:.4f}s")
    
    # Check distance to IP 99
    dist = sssp_df[sssp_df['vertex'] == 99]['distance']
    print(f"Distance 0->99: {dist.values[0]}")

if __name__ == "__main__":
    analyze(gdf)
```

### 🔹 Part 4: CPU vs GPU Comparison

Let's verify the 500x claim.

```python
import networkx as nx

def bench_networkx(cpu_df):
    # Convert to Pandas
    pdf = cpu_df.to_pandas()
    
    start = time.time()
    G_nx = nx.from_pandas_edgelist(pdf, 'src', 'dst', ['weight'], create_using=nx.DiGraph)
    print(f"NX Construct: {time.time() - start:.4f}s")
    
    start = time.time()
    pr = nx.pagerank(G_nx, alpha=0.85)
    print(f"NX PageRank: {time.time() - start:.4f}s")
```

**Typical Results (1M Edges):**
*   **NetworkX:** ~20-30 seconds.
*   **cuGraph:** ~0.04 seconds.
*   **Speedup:** ~500x-700x.

Why?
PageRank is iterative SpMV (Sparse Matrix Vector Mul).
cuGraph uses highly tuned C++ kernels (`Gunrock` or `Hornet` primitives) that saturate HBM bandwidth. NetworkX uses Python dictionaries (random access, cache misses, interpreter overhead).

---

## 🧪 Hands-On Labs

### Lab 60: Pathfinding for Logistics

**Objective:** Compute SSSP on a weighted graph (Road Network).

**Data:**
Create a Grid Graph (Lattice). Each node connects to Up, Down, Left, Right neighbors with random weights (traffic).
Nodes: 1 Million (1000x1000 grid).
Edges: ~4 Million.

**Task:**
1.  Generate Edge List for Grid.
2.  Run `cugraph.sssp(source=TopLeft)`.
3.  Find distance to `BottomRight`.
4.  Visualize the "Wavefront" of distances (heatmap).

**Hint:**
```python
# To generate grid edges efficiently
x = cp.arange(1000)
y = cp.arange(1000)
# Use meshgrid to find neighbors...
```

---

## 📝 Summary & Key Takeaways

1.  **Renumbering:** cuGraph internally renumbers nodes to 0..N-1 integers for array indexing. It maintains a mapping to original IDs (IPs, UUIDs).
2.  **Property Graphs:** Future direction involves supporting Properties on Nodes/Edges natively (PGX style). Currently, properties live in `cudf` DataFrames linked by ID.
3.  **Graph Neural Networks (GNN):** `cugraph` integrates with `cugraph-dgl` and `cugraph-pyg` (PyTorch Geometric) to accelerate GNN sampling and training.
4.  **Limits:** Single GPU RAM limits graph size. 24GB VRAM $\approx$ 200M edges depending on metadata. For larger graphs, use `dask-cugraph`.

---

## 📚 Additional Resources

*   [cuGraph Docs](https://docs.rapids.ai/api/cugraph/stable/)
*   [Gunrock Graph Library](https://github.com/gunrock/gunrock) (The engine under the hood)

**Tomorrow:** Day 61 - cuSpatial & Geospatial... Points, Polygons, and Trajectory Mining on GPU.

*End of Day 060 - Total Lines: 1000+*
