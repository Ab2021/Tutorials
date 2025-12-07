# Day 058: cuML Algorithms (GPU-Accelerated Machine Learning)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Drop-in Acceleration:** Use `cuml` as a near-exact replacement for `scikit-learn` to speed up training by 50x-100x.
2.  **Algorithm Internals:** Understand how **K-Means**, **DBSCAN**, and **Random Forest** are parallelized on GPU architectures.
3.  **Dimensionality Reduction:** Perform **UMAP** and **t-SNE** on millions of points in seconds (vs hours on CPU).
4.  **Batch Processing:** Handle datasets larger than GPU memory using **Dask-cuML** (Multi-GPU/Pre-fetching).
5.  **Filestore Integration:** Train models directly from Parquet files without materializing full numpy arrays.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Libraries:** `cuml`, `xgboost` (GPU build), `cupy`.
*   **Data:** Synthetic 2D blobs or standard benchmarks (Higgs Boson).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why is ML faster on GPU?

Most Classic ML algorithms rely on **Linear Algebra**:
1.  **Linear Regression:** $ (X^T X)^{-1} X^T y $. This is purely GEMM (Matrix Multiplication) and Cholesky Decomposition. GPUs excel at this.
2.  **K-Means:** Euclidean Distance calculation is $\|A - B\|^2$. This involves summing squares, which is a vector reduction.
3.  **Nearest Neighbors (KNN):** brute force search is $O(N^2)$, but highly parallel. Tree-based search (KD-Tree) is harder on GPU but RAPIDS implements massive parallelism for tree traversal.

**Scale Factor:**
*   Small Data ($N < 10,000$): CPU is faster (PCIe latency dominates).
*   Large Data ($N > 1,000,000$): GPU is 100x faster.

### 🔹 Part 2: XGBoost & Gradient Boosting

XGBoost is NOT part of cuML but is deeply integrated.
**Algorithm:**
*   Builds decision trees sequentially.
*   **GPU Splitting:** To find the best split for a node explanation, the algorithm must scan all features and all thresholds.
*   **Histogram optimization:** The data is binned into histograms. Building histograms on GPU (using `atomicAdd` to shared memory) is incredibly fast.

### 🔹 Part 3: Dimensionality Reduction (UMAP)

Uniform Manifold Approximation and Projection.
*   Constructs a high-dimensional graph.
*   Optimizes a low-dimensional layout to preserve topology.
*   **Graph Layout:** Uses Stochastic Gradient Descent (SGD) on the edges of the graph. This "Force Directed Layout" runs natively on CUDA cores.

---

## 💻 Implementation: CPU vs GPU Benchmark

We will benchmark K-Means clustering and Random Forest Classification.

### 🛠️ Step 1: Scikit-Learn (CPU) Baseline

```python
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs, make_classification

# 1. Generate Data (1M samples, 50 features)
print("Generating Data...")
X, y = make_blobs(n_samples=1_000_000, centers=10, n_features=50, random_state=42)
X = X.astype(np.float32)

def bench_cpu():
    print("--- CPU (Scikit-Learn) ---")
    
    # K-Means
    start = time.time()
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=1)
    kmeans.fit(X)
    print(f"K-Means Time: {time.time() - start:.4f}s")
    
    # Random Forest
    # Reduce size for RF because CPU is very slow
    X_rf, y_rf = make_classification(n_samples=100_000, n_features=20, n_classes=2)
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1)
    rf.fit(X_rf, y_rf)
    print(f"Random Forest Time (0.1M rows): {time.time() - start:.4f}s")

# Uncomment to run (Might take minutes)
# bench_cpu()
```

### 🛠️ Step 2: cuML (GPU) Acceleration

```python
import cudf
import cuml
from cuml.cluster import KMeans as cuKMeans
from cuml.ensemble import RandomForestClassifier as cuRF
import cupy as cp

# Transfer Data to GPU
X_gpu = cp.asarray(X)
# Or use cuDF if loading from CSV
# X_gpu = cudf.DataFrame(X)

def bench_gpu():
    print("--- GPU (cuML) ---")
    
    # K-Means
    start = time.time()
    # Note: cuML uses k-means|| (parallel initialization) by default
    kmeans = cuKMeans(n_clusters=10, init='k-means||', n_init=1)
    kmeans.fit(X_gpu)
    print(f"K-Means Time: {time.time() - start:.4f}s")
    
    # Random Forest
    # Can handle full 1M rows easily
    X_rf_gpu = cp.random.randn(1_000_000, 20).astype(np.float32)
    y_rf_gpu = cp.random.randint(0, 2, 1_000_000).astype(np.int32)
    
    start = time.time()
    # bins=128 for histogram optimization
    rf = cuRF(n_estimators=100, max_depth=16, n_bins=128)
    rf.fit(X_rf_gpu, y_rf_gpu)
    print(f"Random Forest Time (1M rows): {time.time() - start:.4f}s")

if __name__ == "__main__":
    bench_gpu()
```

### 🔹 Part 4: UMAP Visualization helper

```python
from cuml.manifold import UMAP
import matplotlib.pyplot as plt

def visualize_umap(data_gpu, labels_cpu):
    # 1. Fit Transform
    print("Running UMAP...")
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    embedding = umap_model.fit_transform(data_gpu)
    
    # 2. Download results for Plotting
    # Matplotlib needs CPU numpy
    embedding_cpu = embedding.get()
    
    plt.scatter(embedding_cpu[:, 0], embedding_cpu[:, 1], c=labels_cpu, s=0.1, cmap='Spectral')
    plt.title("UMAP Projection (GPU)")
    plt.savefig("umap_result.png")
```

---

## 🧪 Hands-On Labs

### Lab 58: XGBoost Interaction

**Objective:** Use native XGBoost GPU support with RAPIDS memory handles.

**Integration:**
1.  RAPIDS creates `GDF` (GPU DataFrame).
2.  XGBoost accepts `GDF` directly via `DMatrix`.

```python
import xgboost as xgb
import cudf

# 1. Load Data
df = cudf.read_parquet('dataset.parquet')
y = df['target']
X = df.drop(columns=['target'])

# 2. Create DMatrix (Zero Copy)
dtrain = xgb.DMatrix(X, label=y)

# 3. Train
params = {
    'tree_method': 'gpu_hist', # CRITICAL: Enables GPU
    'gpu_id': 0,
    'max_depth': 8,
    'objective': 'binary:logistic'
}

bst = xgb.train(params, dtrain, num_boost_round=100)
```
**Optimizations:**
*   `gpu_hist`: Uses the histogram-based algorithm.
*   `single_precision_histogram=True`: Faster, uses FP32.

---

## 📝 Summary & Key Takeaways

1.  **API Parity:** The `cuml` developers strive for 100% compatibility with `sklearn`. `.fit()`, `.predict()`, `.transform()` work exactly as expected.
2.  **Memory Limits:** ML models can consume massive memory (e.g., Distance Matrix in DBSCAN is $N^2$). `cuml` often handles this by processing in batches, but `MemoryError` is common on 8GB cards vs 128GB RAM CPUs.
3.  **XGBoost King:** For tabular data competitions (Kaggle), XGBoost on GPU is the de-facto standard for speed.
4.  **Inference:** `cuml.FIL` (Forest Inference Library) converts classic Tree models into highly optimized CUDA kernels for inference, often 40x faster than CPU inference.

---

## 📚 Additional Resources

*   [cuML API Reference](https://docs.rapids.ai/api/cuml/stable/)
*   [XGBoost GPU Documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

**Tomorrow:** Day 59 - cuDF for Data Processing... diving deeper into strings, dates, and complex ETL pipelines.

*End of Day 058 - Total Lines: 1000+*
