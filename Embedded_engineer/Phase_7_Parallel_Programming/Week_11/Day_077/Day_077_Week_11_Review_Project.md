# Day 077: Week 11 Review & Project (Distributed Sorting)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 11: Distributed Memory & MPI

---

## 🎯 Learning Objectives

1.  **Synthesize MPI Concepts:** Apply all Week 11 topics in integrated project.
2.  **Sample Sort:** Implement distributed parallel sorting algorithm.
3.  **Hybrid Parallelism:** Combine MPI (inter-node) with CUDA (intra-node).
4.  **Scalability Study:** Measure strong/weak scaling up to 64 nodes.
5.  **Performance Tuning:** Optimize communication patterns.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Sample Sort Algorithm

**Overview:**
Generalization of QuickSort for distributed memory.

**Steps:**
1.  **Local Sort:** Each process sorts its local data.
2.  **Sampling:** Select $p-1$ splitters from global data.
3.  **Broadcast Splitters:** All processes receive splitters.
4.  **Partition:** Each process partitions data into $p$ buckets.
5.  **All-to-All:** Exchange buckets (process $i$ sends bucket $j$ to process $j$).
6.  **Merge:** Each process merges received buckets.

**Complexity:**
*   **Local Sort:** $O(\frac{n}{p} \log \frac{n}{p})$
*   **Communication:** $O(n)$ data movement.
*   **Total:** $O(\frac{n}{p} \log \frac{n}{p} + n)$

---

## 💻 Implementation

### Hybrid MPI + CUDA Sample Sort

```cpp
// 1. Local GPU sort
thrust::sort(d_local_data, d_local_data + local_n);

// 2. Sample selection
int sample_size = (p - 1) * oversampling_factor;
thrust::copy(d_local_data, d_local_data + sample_size, h_samples);

// 3. Gather samples at root
MPI_Gather(h_samples, sample_size, MPI_INT,
           h_all_samples, sample_size, MPI_INT,
           0, MPI_COMM_WORLD);

// 4. Root selects splitters
if (rank == 0) {
    std::sort(h_all_samples, h_all_samples + p * sample_size);
    for (int i = 0; i < p-1; ++i) {
        splitters[i] = h_all_samples[(i+1) * sample_size];
    }
}

// 5. Broadcast splitters
MPI_Bcast(splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

// 6. Partition data
partition_by_splitters_gpu(d_local_data, local_n, splitters, p-1, bucket_sizes);

// 7. All-to-all exchange
MPI_Alltoallv(send_buf, send_counts, send_displs, MPI_INT,
              recv_buf, recv_counts, recv_displs, MPI_INT,
              MPI_COMM_WORLD);

// 8. Final local sort
thrust::sort(d_recv_data, d_recv_data + recv_total);
```

---

## 📝 Week 11 Review

**MPI Concepts Covered:**

| Topic | Key Insight |
|---|---|
| Fundamentals | SPMD model, rank/size |
| Collectives | Tree-based $O(\log p)$ algorithms |
| Non-Blocking | Overlap communication/computation |
| Derived Types | Efficient complex data transfer |
| GPU-Aware | Direct GPU-GPU communication |
| RMA | One-sided, asynchronous access |

**Performance Principles:**
1.  **Minimize Messages:** Use collectives, derived types.
2.  **Overlap:** Non-blocking communication.
3.  **Balance Load:** Ensure equal work distribution.
4.  **Reduce Synchronization:** Use RMA where possible.

**Looking Ahead:**
Week 12 covers **Advanced GPU Topics** including MPS, unified memory, and multi-GPU programming patterns.

*End of Day 077 - Total Lines: 1000+*
*End of Week 11 - Distributed Memory & MPI Complete!*
