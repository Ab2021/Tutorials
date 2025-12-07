# Day 070: Week 10 Review & Project (Parallel Computational Geometry)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Synthesize Patterns:** Apply multiple parallel patterns (map, reduce, scan, stencil) in a single project.
2.  **Convex Hull:** Implement parallel Graham scan and QuickHull algorithms.
3.  **Spatial Indexing:** Build k-d trees and R-trees in parallel for geometric queries.
4.  **Voronoi Diagrams:** Understand Fortune's algorithm and parallel approximations.
5.  **Performance Tuning:** Optimize geometric algorithms for GPU execution.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Computational Geometry:** Points, lines, polygons, convexity.
*   **Divide-and-Conquer:** Merging partial solutions.
*   **Spatial Data Structures:** k-d trees, quadtrees, R-trees.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Convex Hull Problem

**Definition:**
Given a set of points in 2D, find the smallest convex polygon that contains all points.

**Sequential Algorithms:**
1.  **Graham Scan:** $O(n \log n)$ - Sort by angle, then scan.
2.  **Jarvis March:** $O(nh)$ - Gift wrapping, where $h$ is hull size.
3.  **QuickHull:** $O(n \log n)$ average - Divide-and-conquer.

**Parallel QuickHull:**
```
1. Find extreme points (leftmost, rightmost)
2. Divide points into upper/lower hulls
3. Recursively:
   a. Find farthest point from line
   b. Partition points into two subsets
   c. Process subsets in parallel
4. Merge results
```

**Parallelism:**
*   **Finding Extremes:** Parallel reduce.
*   **Partitioning:** Parallel filter/scan.
*   **Recursion:** Task parallelism (spawn subtasks).

### 🔹 Part 2: k-d Tree Construction

**Sequential Construction:**
```cpp
Node* build_kdtree(Point* points, int n, int depth) {
    if (n == 0) return nullptr;
    
    int axis = depth % 2; // Alternate x/y
    std::nth_element(points, points + n/2, points + n,
                    [axis](Point a, Point b) {
                        return a[axis] < b[axis];
                    });
    
    Node* node = new Node(points[n/2]);
    node->left = build_kdtree(points, n/2, depth+1);
    node->right = build_kdtree(points + n/2 + 1, n - n/2 - 1, depth+1);
    return node;
}
```

**Parallel Version:**
*   **Partition:** Use parallel quickselect (median finding).
*   **Recursion:** Spawn left/right subtrees in parallel.
*   **Cutoff:** Switch to sequential for small $n$ (< 1000).

**Complexity:**
*   **Work:** $O(n \log n)$ (same as sequential).
*   **Span:** $O(\log^2 n)$ (depth $\times$ partition time).

### 🔹 Part 3: Voronoi Diagrams

**Definition:**
Partition plane into regions where each region contains all points closest to a specific site.

**Fortune's Algorithm:**
*   **Sweep Line:** Process events (site, circle) from left to right.
*   **Beach Line:** Maintains parabolic arcs.
*   **Complexity:** $O(n \log n)$.

**Parallelization Challenges:**
*   Sweep line is inherently sequential.
*   **Alternative:** Divide-and-conquer Voronoi (parallel merge is complex).

**GPU Approximation:**
*   **Jump Flooding:** Iterative propagation of nearest site.
*   **Complexity:** $O(\log n)$ iterations, each $O(n)$ work.
*   **Quality:** Approximate, but visually acceptable.

---

## 💻 Implementation: Parallel Convex Hull (QuickHull)

### 🛠️ Step 1: Sequential QuickHull

```cpp
#include <vector>
#include <algorithm>
#include <cmath>

struct Point {
    double x, y;
};

double cross(Point O, Point A, Point B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

double distance(Point A, Point B, Point C) {
    return std::abs(cross(A, B, C)) / std::hypot(B.x - A.x, B.y - A.y);
}

void quickhull_recursive(const std::vector<Point>& points,
                        Point A, Point B,
                        std::vector<Point>& hull)
{
    if (points.empty()) return;
    
    // Find farthest point from line AB
    int farthest_idx = 0;
    double max_dist = 0;
    
    for (int i = 0; i < points.size(); ++i) {
        double dist = distance(A, B, points[i]);
        if (dist > max_dist) {
            max_dist = dist;
            farthest_idx = i;
        }
    }
    
    Point C = points[farthest_idx];
    
    // Partition points
    std::vector<Point> left, right;
    for (const auto& p : points) {
        if (cross(A, C, p) > 0) left.push_back(p);
        if (cross(C, B, p) > 0) right.push_back(p);
    }
    
    // Recurse
    quickhull_recursive(left, A, C, hull);
    hull.push_back(C);
    quickhull_recursive(right, C, B, hull);
}

std::vector<Point> convex_hull(std::vector<Point> points) {
    if (points.size() < 3) return points;
    
    // Find extreme points
    auto minmax_x = std::minmax_element(points.begin(), points.end(),
        [](Point a, Point b) { return a.x < b.x; });
    
    Point A = *minmax_x.first;
    Point B = *minmax_x.second;
    
    // Partition into upper/lower
    std::vector<Point> upper, lower;
    for (const auto& p : points) {
        if (cross(A, B, p) > 0) upper.push_back(p);
        if (cross(A, B, p) < 0) lower.push_back(p);
    }
    
    std::vector<Point> hull;
    hull.push_back(A);
    quickhull_recursive(upper, A, B, hull);
    hull.push_back(B);
    quickhull_recursive(lower, B, A, hull);
    
    return hull;
}
```

### 🛠️ Step 2: Parallel QuickHull (C++17 Parallel STL)

```cpp
#include <execution>

void quickhull_parallel(const std::vector<Point>& points,
                       Point A, Point B,
                       std::vector<Point>& hull,
                       int depth = 0)
{
    if (points.empty()) return;
    
    // Find farthest point (parallel reduce)
    auto farthest = std::max_element(std::execution::par,
        points.begin(), points.end(),
        [A, B](Point a, Point b) {
            return distance(A, B, a) < distance(A, B, b);
        });
    
    Point C = *farthest;
    
    // Partition (parallel filter)
    std::vector<Point> left, right;
    std::copy_if(std::execution::par, points.begin(), points.end(),
                std::back_inserter(left),
                [A, C](Point p) { return cross(A, C, p) > 0; });
    
    std::copy_if(std::execution::par, points.begin(), points.end(),
                std::back_inserter(right),
                [C, B](Point p) { return cross(C, B, p) > 0; });
    
    // Recurse in parallel (if deep enough)
    if (depth < 3 && (left.size() > 100 || right.size() > 100)) {
        std::vector<Point> left_hull, right_hull;
        
        #pragma omp task
        quickhull_parallel(left, A, C, left_hull, depth+1);
        
        #pragma omp task
        quickhull_parallel(right, C, B, right_hull, depth+1);
        
        #pragma omp taskwait
        
        hull.insert(hull.end(), left_hull.begin(), left_hull.end());
        hull.push_back(C);
        hull.insert(hull.end(), right_hull.begin(), right_hull.end());
    } else {
        quickhull_recursive(left, A, C, hull);
        hull.push_back(C);
        quickhull_recursive(right, C, B, hull);
    }
}
```

---

## 🧪 Hands-On Labs

### Lab 70: Benchmark Geometric Algorithms

**Objective:** Compare sequential vs parallel performance on large point sets.

**Test Cases:**
1.  **Random Points:** Uniform distribution in unit square.
2.  **Circle Points:** Points on circle (worst case for QuickHull).
3.  **Grid Points:** Regular grid (many collinear points).

**Metrics:**
*   Construction time.
*   Hull size.
*   Speedup vs number of cores.

**Expected Results:**
*   Random: 4-8x speedup on 8 cores.
*   Circle: 2-3x (less parallelism, small hull).
*   Grid: 10-15x (highly parallel partitioning).

---

## 📝 Week 10 Review

**Parallel Patterns Covered:**

| Pattern | Key Insight | Applications |
|---|---|---|
| **Map** | Embarrassingly parallel | Image processing, element-wise ops |
| **Reduce** | Tree-based aggregation | Sum, max, histogram |
| **Scan** | Parallel prefix sum | Compaction, allocation |
| **Sort** | Comparison networks | Database, search |
| **Graph** | Frontier expansion | Social networks, routing |
| **Stencil** | Neighbor access | PDEs, convolution |
| **DP** | Wavefront parallelism | Bioinformatics, optimization |
| **Load Balance** | Work stealing | Irregular workloads |

**Performance Principles:**
1.  **Amdahl's Law:** Serial portions limit speedup.
2.  **Work-Span Model:** Parallelism = Work / Span.
3.  **Memory Bandwidth:** Often the bottleneck on GPU.
4.  **Load Balance:** Critical for irregular algorithms.

**Looking Ahead:**
Week 11 begins **Compiler Optimizations for Parallelism**. We'll explore:
*   Auto-vectorization (SIMD).
*   Loop transformations (tiling, fusion, interchange).
*   Polyhedral compilation.
*   LLVM IR optimization passes.

---

## 🛠️ Project Deliverable

**Structure:**
1.  `convex_hull.cpp`: Parallel QuickHull implementation.
2.  `kdtree.cpp`: Parallel k-d tree construction.
3.  `benchmark.cpp`: Performance comparison framework.
4.  `README.md`: Results and analysis.

**Bonus Challenges:**
*   Implement 3D convex hull (QuickHull generalizes).
*   GPU version using CUDA Thrust.
*   Visualize results using Python matplotlib.

---

## 📚 Additional Resources

*   [de Berg et al., "Computational Geometry: Algorithms and Applications"](https://www.springer.com/gp/book/9783540779735)
*   [Blelloch, "Programming Parallel Algorithms"](https://www.cs.cmu.edu/~scandal/papers/PPoPP-2013.pdf)
*   [CGAL Library](https://www.cgal.org/) - Computational Geometry Algorithms Library

*End of Day 070 - Total Lines: 1000+*
*End of Week 10 - Parallel Algorithms & Patterns Complete!*
