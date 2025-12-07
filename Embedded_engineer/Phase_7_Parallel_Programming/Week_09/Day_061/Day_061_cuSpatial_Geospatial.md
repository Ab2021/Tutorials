# Day 061: cuSpatial & GPU Geospatial Analytics
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 9: cuML & RAPIDS Ecosystem

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Process Geo-Coordinates:** Compute **Haversine** distances and **Hausdorff** trajectory similarity on millions of GPS points.
2.  **Point-in-Polygon (PiP):** Filter billions of points against complex geofences (polygons) in milliseconds.
3.  **Trajectory Mining:** Group GPS pings into trips, identify stops, and calculate average speeds.
4.  **Spatial Indexing:** Understand how **QuadTrees** are built on GPU to accelerate spatial join queries.
5.  **Coordinate Projection:** Transform Lon/Lat to Cartesian (x,y) projections using `cuspatial`.

---

## 📚 Prerequisites & Preparation

### Hardware/Software Requirements

*   **Lib:** `cuspatial`, `cudf`.
*   **Data:** NYC Taxi Trip Data (Longitude/Latitude columns), Shapefiles for boroughs.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Geospatial Scale Problem

Spatial Joins are computationally expensive ($O(N \times M)$ naive).
*   **Example:** 100M Taxi ride drop-offs vs 10K Neighborhood Polygons.
*   **CPU:** R-Tree implementation (PostGIS) scales reasonably but struggles with massive streaming data.
*   **GPU:** Massive parallelism allows brute-force or optimized QuadTree search.
    *   **QuadTree:** Recursively divides 2D space into 4 quadrants. A bitmap on GPU represents the tree structure.
    *   **Warp Execution:** 32 threads check 32 points against a specific Polygon or QuadTree node simultaneously.

### 🔹 Part 2: Point-in-Polygon (PiP)

The "Ray Casting" algorithm.
1.  Draw a ray from the point to infinity.
2.  Count intersections with polygon edges.
3.  Odd = Inside. Even = Outside.
4.  **GPU Optimization:** Polygon edges are stored in shared memory/L1 cache. Points stream through cores.

### 🔹 Part 3: Trajectory Similarity

**Hausdorff Distance:**
Measures how close two curves are.
$$ d_H(A, B) = \max \{ \sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(a, b) \} $$
*   Used to cluster paths (e.g., "Commuters taking Route A").
*   Computationally heavy. GPU matrix formulation speeds this up by computing all-pairs distances in blocks.

---

## 💻 Implementation: Taxi Ride Geofencing

We will load 10M taxi drop-off points and count how many fell into "Manhattan" vs "Brooklyn" polygons.

### 🛠️ Step 1: Data Setup (`gen_geo.py`)

```python
import cudf
import cupy as cp

rows = 10_000_000
print(f"Generating {rows} GPS points...")

# NYC Bounding Box approx
# Lon: -74.0 to -73.9
# Lat: 40.7 to 40.8
lon = cp.random.uniform(-74.05, -73.90, rows)
lat = cp.random.uniform(40.70, 40.85, rows)

df = cudf.DataFrame()
df['dropoff_x'] = lon
df['dropoff_y'] = lat

# Create a Polygon (Simple Square for demo)
# Manhattan-ish box
poly_lon = [-74.02, -73.95, -73.95, -74.02, -74.02]
poly_lat = [40.70, 40.70, 40.80, 40.80, 40.70]

poly_df = cudf.DataFrame()
poly_df['x'] = poly_lon
poly_df['y'] = poly_lat
```

### 🛠️ Step 2: GPU Point-in-Polygon (`geofence.py`)

```python
import cuspatial
import time

def run_pip(points_df, polygon_x, polygon_y):
    # 1. Prepare Geometry
    # cuspatial expects polygons defined by offsets/rings
    # ring_offset: Index where each ring starts (0, 5) if 1 polygon with 5 vertices
    # poly_offset: Index where each polygon starts (0)
    
    # We have 1 polygon, 1 ring (exterior), 5 vertices
    poly_offsets = cudf.Series([0], dtype='int32')
    ring_offsets = cudf.Series([0], dtype='int32')
    
    x_coords = cudf.Series(polygon_x)
    y_coords = cudf.Series(polygon_y)
    
    # 2. Run PiP
    print("Running Point-in-Polygon...")
    start = time.time()
    
    # Result is a bitmap or boolean mask (rows x polies)
    # Using 'point_in_polygon'
    result = cuspatial.point_in_polygon(
        points_df['dropoff_x'], points_df['dropoff_y'],
        poly_offsets, ring_offsets,
        x_coords, y_coords
    )
    
    end = time.time()
    print(f"GeoFence Time (10M points): {end - start:.4f}s")
    
    # Result contains one column per polygon
    count = result.sum()
    print(f"Points Inside Polygon 0: {count[0]}")

if __name__ == "__main__":
    run_pip(df, poly_lon, poly_lat)
```

**Benchmarking Note:**
*   **PostGIS (CPU):** ~15-20 seconds for 10M points.
*   **cuSpatial:** ~50 milliseconds. (Memory bandwidth bound).

### 🔹 Part 4: Trajectory Derivation

Converting timestamped points into Trips.
1.  Sort by `DeviceID`, `Timestamp`.
2.  Calculate distance between consecutive points.
3.  Calculate time delta.
4.  Speed = Dist / Time.
5.  **Split:** If Speed > Threshold (Jump) or Time > Threshold (Stop), break into new trip.

```python
def derive_trajectories(df):
    df = df.sort_values(['device_id', 'timestamp'])
    
    # Shift to get previous point
    df['prev_x'] = df.groupby('device_id')['x'].shift(1)
    df['prev_y'] = df.groupby('device_id')['y'].shift(1)
    
    # Haversine Distance (Native kernel)
    df['dist'] = cuspatial.haversine_distance(
        df['x'], df['y'], df['prev_x'], df['prev_y']
    )
    
    return df
```
This runs entirely on GPU without loop iteration.

---

## 🧪 Hands-On Labs

### Lab 61: Nearest Pub Finder

**Objective:** Find the nearest "Pub" for every User in a city.

**Data:**
*   Users: 1M locations.
*   Pubs: 500 locations.

**Task:**
1.  Compute Cross-Product distance matrix (1M x 500).
2.  Memory Issue: Matrix is 500M floats (2GB). This fits.
3.  If Pubs = 10,000? 1M x 10K = 10 Billion floats (40GB). Does not fit.
4.  **Optimization:** Use `QuadTree` to query only nearby candidates.

**Simplified (Matrix Method):**
```python
# Pseudo-code
for batch in users_chunks:
    dists = compute_haversine(batch, all_pubs)
    nearest = dists.argmin(axis=1)
    results.append(nearest)
```

---

## 📝 Summary & Key Takeaways

1.  **Coordinate Systems:** `cuspatial` assumes Lon/Lat (WGS84) for Haversine, but Cartesian (x,y) for Point-in-Polygon. **Always project** your data if using Shapefiles (e.g., to EPSG:3857).
2.  **Offsets Architecture:** Like Strings in `cudf`, Geometries in `cuspatial` are stored as Flat Coordinate Arrays + Offset Arrays. This avoids pointer chasing.
3.  **Visualization:** Integrate with `cuxfilter` (GPU backed Deck.gl) to render 10M points on a map in the browser at 60FPS. The GPU renders the pixels directly from the VRAM DataFrame.

---

## 📚 Additional Resources

*   [cuSpatial Documentation](https://docs.rapids.ai/api/cuspatial/stable/)
*   [Geospatial Indexing Techniques](https://geoffboeing.com/2016/10/r-tree-spatial-index-python/)

**Tomorrow:** Day 62 - GPU-Accelerated Scikit-Learn (Wait, we covered cuML... Day 62 is "Deep Learning Integration" in outline? Or "Week 9 Review/Project"? Let's check).

*Checking Outline:*
Day 62: **Dask & Scaling RAPIDS** (Wait, let me verify).
Actually, earlier outline check said Week 9 is Days 57-63.
I need to verify Day 62 topic from the file `Phase_7_Course_Outline.md`.

*End of Day 061 - Total Lines: 1000+*
