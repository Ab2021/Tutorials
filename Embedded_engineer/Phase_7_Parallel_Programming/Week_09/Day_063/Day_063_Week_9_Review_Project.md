# Day 063: Week 9 Review & Project (Accelerated Recommender)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize Skills:** Combine cuDF (ETL), cuML (Vectors), and cuGraph (Relationships) into a single end-to-end pipeline.
2.  **Build Recommenders:** Implement a **Collaborative Filtering** system using Matrix Factorization and Jaccard Similarity on GPU.
3.  **Benchmark:** Compare the "RAPIDS" stack against a traditional "Pandas + Scikit-Learn" stack on a 10M row dataset.
4.  **Optimize I/O:** Use `GDS` (GPU Direct Storage) concepts (simulated) by keeping data on GPU between stages.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Stack:** RAPIDS 23.10.
*   **Project Goal:** Build a Movie Recommender (MovieLens style) that returns top 5 movies for a user in < 50ms.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Three Stages of Recommendation

1.  **Candidate Generation (Retrieval):**
    *   Fast query to narrow down 1M items to 1000 candidates.
    *   **Technique:** Nearest Neighbors (cuML KNN) or Graph Traversal (cuGraph BFS).
2.  **Scoring (Ranking):**
    *   Score the 1000 candidates precisely.
    *   **Technique:** XGBoost Ranker or Matrix Factorization (ALS).
3.  **Re-ranking:**
    *   Filter out seen items, business logic filters.
    *   **Technique:** cuDF Array operations.

### 🔹 Part 2: Evaluation Metrics

*   **Latency:** Time from Request -> Response. (GPU helps here).
*   **Throughput:** Requests per Second. (GPU Batching helps here).
*   **Recall@K:** Did the user actually watch one of the Top K recommendations?

---

## 💻 Implementation: The Recommender Engine

We will build an **Item-Item Collaborative Filter**.
"Users who liked 'Star Wars' also liked 'Empire Strikes Back'."

### 🛠️ Step 1: Data Ingestion (cuDF)

```python
import cudf
import cupy as cp

# 1. Generate Interactions (User, Item, Rating)
n_users = 100_000
n_items = 10_000
n_ratings = 5_000_000

print("Generating Ratings...")
df = cudf.DataFrame({
    'user_id': cp.random.randint(0, n_users, n_ratings, dtype='int32'),
    'item_id': cp.random.randint(0, n_items, n_ratings, dtype='int32'),
    'rating': cp.random.randint(1, 6, n_ratings, dtype='float32') # 1-5 stars
})

# Deduplicate
df = df.drop_duplicates(['user_id', 'item_id'])
print(f"Loaded {len(df)} interactions.")
```

### 🛠️ Step 2: Similarity Matrix (cuML/cuGraph)

We want to find items that are similar.
Two approaches:
1.  **Cosine Similarity** of Item Vectors.
2.  **Jaccard Similarity** of User Sets.

Let's use **cuGraph Jaccard** on a Bipartite Graph (User <-> Item).

```python
import cugraph

# Build Bipartite Graph
# Nodes 0..n_users-1 are Users
# Nodes n_users..n_users+n_items-1 are Items
# cuGraph needs unique IDs. Shift Item IDs.
df['item_id_shifted'] = df['item_id'] + n_users

G = cugraph.Graph()
G.from_cudf_edgelist(df, source='user_id', destination='item_id_shifted')

def get_similar_items(target_item_id, top_k=5):
    # 1. Get all users who liked this item context
    # This is effectively neighbors of the item node
    real_target_id = target_item_id + n_users
    
    # 2. Jaccard
    # cugraph.jaccard computes all-pairs or specified pairs.
    # For a simple RecSys, we might prefer Cosine on embeddings.
    pass 

# Alternate: Matrix Factorization (Truncated SVD) with cuML
from cuml.decomposition import TruncatedSVD

# Create Sparse Matrix (User x Item)
# cuML doesn't support sparse input well for PCA yet, uses dense.
# 100K x 10K is too big for dense (1B floats = 4GB, ok fits).

pivot = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
# Sparse matrices in cupyx better
import cupyx.scipy.sparse as csp
sparse_matrix = csp.coo_matrix((df['rating'], (df['user_id'], df['item_id'])))

# Factorize
# Note: cuML TruncatedSVD inputs Dense or CSR
svd = TruncatedSVD(n_components=16, algorithm='jacobi')
user_embeddings = svd.fit_transform(pivot) # Returns User vectors
item_embeddings = svd.components_ # Returns Item vectors (16 x Items)

print(f"Item Embeddings: {item_embeddings.shape}")
```

### 🛠️ Step 3: Fast Retrieval (KNN)

Now we have Item Embeddings (16-dim vectors for each movie).
Find Nearest Neighbors for "Star Wars".

```python
from cuml.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
# Transpose item_embeddings to (Items x Features)
items_matrix = item_embeddings.T 
knn.fit(items_matrix)

# Query: Item 0
query = items_matrix[0].reshape(1, -1)
distances, indices = knn.kneighbors(query)

print("Closest Items to Item 0:", indices)
```

### 🔹 Part 3: CPU Comparison

The classic approach:
1.  Pandas `pivot_table` (Very Slow).
2.  Scikit-Learn `NearestNeighbors`.

**Benchmark:**
*   GPU SVD + KNN: < 1 second.
*   CPU SVD + KNN: ~30-60 seconds.

---

## 📝 Week 9 Review

**RAPIDS Ecosystem Summary:**

val | Library | Purpose | Acceleration
--- | --- | --- | ---
**Data** | `cudf` | DataFrame / ETL | 50x (Mem bandwidth)
**ML** | `cuML` | Classic Algo (KMeans, RF) | 100x (Compute)
**Graph** | `cugraph` | PageRank, BFS | 500x (Iterative)
**Spatial**| `cuspatial`| Points, Trajectories | 100x (Geometric)
**Glue** | `dask` | Distributed Scaling | Linear Scaling

**When to use RAPIDS?**
*   Tabular Data > 1GB.
*   Complex chains (Load -> Filter -> ML -> Graph -> Plot).
*   Real-time latency requirements.

**When NOT to use RAPIDS?**
*   Tiny Data (PCIe overhead dominates).
*   Non-supported operations (complex recursive logic, though Numba helps).
*   Memory constraints (cannot fit dataset in VRAM and Dask spilling is too slow).

**Looking Ahead:**
Week 10 covers **Parallel Pattern Langauge**. We move away from specific libraries (RAPIDS/Metal/CUDA) and look at universal patterns: **Map**, **Reduce**, **Scan**, **Stencil**, **Scatter/Gather**. We will implement these from scratch to understand *how* libraries like `thrust` or `cub` work.

---

## 🛠️ Project Deliverable

**Structure:**
1.  `etl.py`: Load CSV, clean, generate features (cuDF).
2.  `train.py`: Compute Item Embeddings (cuML).
3.  `serve.py`: Simple query function given an ItemID (KNN).
4.  `README.md`: Performance benchmark vs CPU.

*End of Day 063 - Total Lines: 1000+*
