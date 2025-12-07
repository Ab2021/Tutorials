# Day 067: Stencil Computations & Finite Difference Methods
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Stencil Patterns:** Understand the computational structure of stencil operations (neighbor-based updates).
2.  **Finite Difference:** Implement heat equation and wave equation solvers using explicit time-stepping.
3.  **Halo Exchange:** Manage ghost cells and boundary communication in distributed memory (MPI).
4.  **GPU Optimization:** Use shared memory tiling to minimize global memory accesses in 2D/3D stencils.
5.  **Stability Analysis:** Apply CFL (Courant-Friedrichs-Lewy) condition to ensure numerical stability.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Partial Differential Equations (PDEs):** Heat equation, wave equation, Laplace equation.
*   **Numerical Methods:** Finite difference approximations, Taylor series.
*   **Parallel Patterns:** Domain decomposition, ghost cells.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Stencil?

**Definition:**
A stencil is a fixed pattern of neighbor accesses used to update each element in a grid.

**Example (5-Point Stencil in 2D):**
```
Update u[i,j] based on:
    u[i-1,j]  (North)
    u[i+1,j]  (South)
    u[i,j-1]  (West)
    u[i,j+1]  (East)
    u[i,j]    (Center)
```

**Applications:**
*   **Physics Simulations:** Heat diffusion, fluid dynamics, electromagnetics.
*   **Image Processing:** Gaussian blur, edge detection, convolution.
*   **Machine Learning:** Convolutional neural networks (CNNs).

### 🔹 Part 2: Heat Equation (1D)

**PDE:**
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Where:
*   $u(x,t)$: Temperature at position $x$ and time $t$.
*   $\alpha$: Thermal diffusivity.

**Finite Difference Discretization:**
$$\frac{u_i^{n+1} - u_i^n}{\Delta t} = \alpha \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{(\Delta x)^2}$$

**Explicit Update:**
$$u_i^{n+1} = u_i^n + \frac{\alpha \Delta t}{(\Delta x)^2} (u_{i+1}^n - 2u_i^n + u_{i-1}^n)$$

**Stability Condition (CFL):**
$$\frac{\alpha \Delta t}{(\Delta x)^2} \leq \frac{1}{2}$$

If violated, solution becomes numerically unstable (oscillates/explodes).

### 🔹 Part 3: 2D Heat Equation

**PDE:**
$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

**Finite Difference (5-Point Stencil):**
$$u_{i,j}^{n+1} = u_{i,j}^n + \frac{\alpha \Delta t}{(\Delta x)^2} (u_{i+1,j}^n + u_{i-1,j}^n + u_{i,j+1}^n + u_{i,j-1}^n - 4u_{i,j}^n)$$

**Stability:**
$$\frac{\alpha \Delta t}{(\Delta x)^2} \leq \frac{1}{4}$$

### 🔹 Part 4: Halo Exchange (MPI)

**Problem:**
When domain is partitioned across MPI ranks, boundary cells need data from neighboring ranks.

**Ghost Cells (Halo):**
Each rank maintains extra rows/columns that store copies of neighbors' boundary data.

**Communication Pattern:**
```
Rank 0: [====|G]  Send right boundary to Rank 1
Rank 1: [G|====|G]  Receive from Rank 0, send to Rank 2
Rank 2: [G|====]  Receive from Rank 1
```

**MPI Code:**
```cpp
// Send right boundary to right neighbor
MPI_Sendrecv(
    &u[nx-2], ny, MPI_DOUBLE, right_rank, 0,
    &u[0], ny, MPI_DOUBLE, left_rank, 0,
    MPI_COMM_WORLD, &status
);
```

### 🔹 Part 5: GPU Shared Memory Tiling

**Naive GPU Implementation:**
```cpp
__global__ void stencil_naive(float* in, float* out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        out[i*ny + j] = 0.25f * (
            in[(i-1)*ny + j] +
            in[(i+1)*ny + j] +
            in[i*ny + (j-1)] +
            in[i*ny + (j+1)]
        );
    }
}
```
**Problem:** Each thread reads 5 global memory locations. Neighboring threads re-read same data.

**Optimized (Shared Memory):**
```cpp
__global__ void stencil_shared(float* in, float* out, int nx, int ny) {
    __shared__ float tile[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ti = threadIdx.x + 1;
    int tj = threadIdx.y + 1;
    
    // Load tile (including halo)
    tile[ti][tj] = in[i*ny + j];
    
    // Load halo cells
    if (threadIdx.x == 0 && i > 0)
        tile[ti-1][tj] = in[(i-1)*ny + j];
    if (threadIdx.x == blockDim.x-1 && i < nx-1)
        tile[ti+1][tj] = in[(i+1)*ny + j];
    if (threadIdx.y == 0 && j > 0)
        tile[ti][tj-1] = in[i*ny + (j-1)];
    if (threadIdx.y == blockDim.y-1 && j < ny-1)
        tile[ti][tj+1] = in[i*ny + (j+1)];
    
    __syncthreads();
    
    // Compute stencil from shared memory
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        out[i*ny + j] = 0.25f * (
            tile[ti-1][tj] +
            tile[ti+1][tj] +
            tile[ti][tj-1] +
            tile[ti][tj+1]
        );
    }
}
```

**Speedup:** 3-5x by reducing global memory traffic.

---

## 💻 Implementation: 2D Heat Equation Solver

### 🛠️ Step 1: CPU Sequential Version

```cpp
#include <iostream>
#include <vector>
#include <cmath>

void heat_equation_2d_cpu(std::vector<float>& u, 
                          std::vector<float>& u_new,
                          int nx, int ny, 
                          float alpha, float dt, float dx,
                          int num_steps)
{
    float r = alpha * dt / (dx * dx);
    
    // Check stability
    if (r > 0.25f) {
        std::cerr << "Warning: CFL condition violated! r = " << r << "\n";
    }
    
    for (int step = 0; step < num_steps; ++step) {
        for (int i = 1; i < nx-1; ++i) {
            for (int j = 1; j < ny-1; ++j) {
                int idx = i * ny + j;
                u_new[idx] = u[idx] + r * (
                    u[(i-1)*ny + j] +
                    u[(i+1)*ny + j] +
                    u[i*ny + (j-1)] +
                    u[i*ny + (j+1)] -
                    4.0f * u[idx]
                );
            }
        }
        std::swap(u, u_new);
    }
}

int main() {
    const int nx = 512, ny = 512;
    const float alpha = 0.1f;
    const float dx = 1.0f / nx;
    const float dt = 0.25f * dx * dx / alpha; // CFL limit
    const int num_steps = 1000;
    
    std::vector<float> u(nx * ny, 0.0f);
    std::vector<float> u_new(nx * ny, 0.0f);
    
    // Initial condition: hot spot in center
    int cx = nx / 2, cy = ny / 2;
    for (int i = cx-10; i < cx+10; ++i) {
        for (int j = cy-10; j < cy+10; ++j) {
            u[i*ny + j] = 100.0f;
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    heat_equation_2d_cpu(u, u_new, nx, ny, alpha, dt, dx, num_steps);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "CPU Time: " 
              << std::chrono::duration<double>(end - start).count() 
              << "s\n";
    
    return 0;
}
```

### 🛠️ Step 2: GPU Version with Shared Memory

```cpp
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void heat_kernel(const float* u, float* u_new, 
                           int nx, int ny, float r)
{
    __shared__ float tile[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ti = threadIdx.x + 1;
    int tj = threadIdx.y + 1;
    
    // Load center
    if (i < nx && j < ny) {
        tile[ti][tj] = u[i*ny + j];
    }
    
    // Load halos
    if (threadIdx.x == 0 && i > 0) {
        tile[ti-1][tj] = u[(i-1)*ny + j];
    }
    if (threadIdx.x == blockDim.x-1 && i < nx-1) {
        tile[ti+1][tj] = u[(i+1)*ny + j];
    }
    if (threadIdx.y == 0 && j > 0) {
        tile[ti][tj-1] = u[i*ny + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j < ny-1) {
        tile[ti][tj+1] = u[i*ny + (j+1)];
    }
    
    __syncthreads();
    
    // Compute update
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        int idx = i * ny + j;
        u_new[idx] = tile[ti][tj] + r * (
            tile[ti-1][tj] +
            tile[ti+1][tj] +
            tile[ti][tj-1] +
            tile[ti][tj+1] -
            4.0f * tile[ti][tj]
        );
    }
}

void heat_equation_2d_gpu(float* d_u, float* d_u_new,
                          int nx, int ny,
                          float r, int num_steps)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    for (int step = 0; step < num_steps; ++step) {
        heat_kernel<<<blocks, threads>>>(d_u, d_u_new, nx, ny, r);
        std::swap(d_u, d_u_new);
    }
}
```

---

## 🧪 Hands-On Labs

### Lab 67: 3D Stencil (7-Point)

**Objective:** Extend the 2D heat equation to 3D.

**Stencil:**
```
u[i,j,k] depends on:
  u[i±1,j,k], u[i,j±1,k], u[i,j,k±1]
```

**Challenges:**
*   **Memory Layout:** Row-major (C-style) vs column-major (Fortran-style).
*   **Shared Memory:** 3D tile requires more shared memory.
*   **Bandwidth:** 3D stencils are even more memory-bound.

**Task:**
1.  Implement 3D heat equation on GPU.
2.  Benchmark different block sizes (8x8x8 vs 16x16x4).
3.  Measure achieved memory bandwidth (GB/s).

---

## 📝 Summary & Key Takeaways

1.  **Stencils are Memory-Bound:** Performance limited by DRAM bandwidth, not compute.
2.  **Shared Memory is Essential:** Reusing neighbor data via shared memory provides 3-5x speedup.
3.  **CFL Condition:** Explicit methods require small time steps for stability. Implicit methods (not covered) allow larger steps but require solving linear systems.
4.  **Halo Exchange Overhead:** In distributed computing, communication can dominate for small local domains. Aim for computation/communication ratio > 10.
5.  **Cache Blocking:** On CPU, tiling for L1/L2 cache is analogous to GPU shared memory tiling.

**Stencil Performance Model:**
$$\text{Time} = \frac{\text{Grid Points} \times \text{Bytes per Point} \times \text{Stencil Radius}}{\text{Memory Bandwidth}}$$

For 2D 5-point stencil on A100 (2 TB/s):
*   Grid: $1024^2 = 1M$ points.
*   Bytes: 4 (float32).
*   Effective reads: 5 per point (naive) or ~1.5 (with shared memory).
*   Time: $\frac{1M \times 4 \times 5}{2 \times 10^{12}} \approx 10 \mu s$ (theoretical peak).

---

## 📚 Additional Resources

*   [LeVeque, "Finite Difference Methods for ODEs and PDEs"](https://faculty.washington.edu/rjl/fdmbook/)
*   [NVIDIA Stencil Optimization Guide](https://developer.nvidia.com/blog/finite-difference-methods-cuda-cc-part-1/)
*   [MPI Halo Exchange Patterns](https://www.mcs.anl.gov/~itf/dbpp/text/node31.html)

**Tomorrow:** Day 68 - Dynamic Programming on GPU... parallelizing recurrence relations and sequence alignment.

*End of Day 067 - Total Lines: 1000+*
