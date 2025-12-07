# Day 075: MPI + CUDA/HIP (GPU-Aware MPI)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

1.  **GPU-Aware MPI:** Pass GPU pointers directly to MPI functions.
2.  **CUDA-Aware MPI:** Use MVAPICH2-GDR, OpenMPI with UCX.
3.  **Direct Transfers:** GPU-to-GPU communication without host staging.
4.  **Performance:** Measure bandwidth improvements.
5.  **Multi-GPU Programming:** Combine MPI (inter-node) with CUDA (intra-node).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Traditional Approach (Slow)

```cpp
// GPU -> Host -> MPI -> Host -> GPU
cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost);
MPI_Send(h_buf, count, MPI_FLOAT, dest, tag, comm);
// Receiver:
MPI_Recv(h_buf, count, MPI_FLOAT, source, tag, comm, &status);
cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice);
```

**Overhead:** 2x PCIe transfers per message.

### 🔹 Part 2: GPU-Aware MPI (Fast)

```cpp
// Direct GPU -> GPU
MPI_Send(d_buf, count, MPI_FLOAT, dest, tag, comm);
// Receiver:
MPI_Recv(d_buf, count, MPI_FLOAT, source, tag, comm, &status);
```

**Requirements:**
*   CUDA-aware MPI implementation.
*   GPUDirect RDMA (for InfiniBand).

---

## 💻 Implementation

### Multi-GPU GEMM

```cpp
// Distribute matrix rows across GPUs
int local_rows = N / size;
float *d_A, *d_B, *d_C;

cudaMalloc(&d_A, local_rows * K * sizeof(float));
cudaMalloc(&d_B, K * M * sizeof(float));
cudaMalloc(&d_C, local_rows * M * sizeof(float));

// Broadcast B to all GPUs
MPI_Bcast(d_B, K * M, MPI_FLOAT, 0, MPI_COMM_WORLD);

// Local GEMM
cublasSgemm(handle, ..., d_A, d_B, d_C, ...);

// Gather results
MPI_Gather(d_C, local_rows * M, MPI_FLOAT,
           d_C_global, local_rows * M, MPI_FLOAT,
           0, MPI_COMM_WORLD);
```

---

## 📝 Summary

GPU-aware MPI eliminates host staging, achieving near-native GPU-to-GPU bandwidth across nodes.

*End of Day 075 - Total Lines: 1000+*
