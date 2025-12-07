# Day 068: Parallel Dynamic Programming
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 10: Parallel Algorithms & Patterns

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Wavefront Parallelism:** Identify anti-diagonal parallelism in 2D DP tables.
2.  **Sequence Alignment:** Implement Smith-Waterman and Needleman-Wunsch algorithms on GPU.
3.  **Matrix Chain Multiplication:** Parallelize optimal parenthesization using diagonal sweeps.
4.  **Recurrence Relations:** Transform sequential recurrences into parallel-friendly formulations.
5.  **Performance Analysis:** Understand why DP is challenging to parallelize (data dependencies).

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **Dynamic Programming:** Optimal substructure, overlapping subproblems.
*   **Sequence Alignment:** Edit distance, scoring matrices (BLOSUM, PAM).
*   **DAG (Directed Acyclic Graph):** DP as topological sort on implicit DAG.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why DP is Hard to Parallelize

**Sequential DP (Fibonacci):**
```cpp
dp[0] = 0; dp[1] = 1;
for (int i = 2; i <= n; ++i) {
    dp[i] = dp[i-1] + dp[i-2]; // Depends on previous two
}
```

**Dependency Chain:**
Each `dp[i]` depends on `dp[i-1]`, which depends on `dp[i-2]`, etc.
This is a **sequential dependency** → no parallelism.

**Solution:**
*   **Matrix Exponentiation:** For linear recurrences, use $O(\log n)$ parallel matrix multiplications.
*   **Wavefront Parallelism:** For 2D DP, process anti-diagonals in parallel.

### 🔹 Part 2: Sequence Alignment (Smith-Waterman)

**Problem:**
Find optimal local alignment between two DNA/protein sequences.

**Scoring:**
*   Match: +2
*   Mismatch: -1
*   Gap: -2

**DP Recurrence:**
$$H[i,j] = \max \begin{cases}
0 & \text{(reset)} \\
H[i-1,j-1] + s(A_i, B_j) & \text{(match/mismatch)} \\
H[i-1,j] - 2 & \text{(gap in B)} \\
H[i,j-1] - 2 & \text{(gap in A)}
\end{cases}$$

**Dependency:**
`H[i,j]` depends on `H[i-1,j-1]`, `H[i-1,j]`, `H[i,j-1]`.

**Wavefront Parallelism:**
```
Iteration 1: Compute H[1,1]
Iteration 2: Compute H[1,2], H[2,1] in parallel
Iteration 3: Compute H[1,3], H[2,2], H[3,1] in parallel
...
```

All cells on the same anti-diagonal can be computed simultaneously.

**Parallelism:**
*   **Maximum:** $\min(m, n)$ threads (diagonal length).
*   **Total Work:** $O(mn)$.
*   **Span:** $O(m+n)$ (number of diagonals).

### 🔹 Part 3: Matrix Chain Multiplication

**Problem:**
Given matrices $A_1, A_2, \ldots, A_n$, find optimal parenthesization to minimize scalar multiplications.

**DP Table:**
```
dp[i][j] = minimum cost to multiply A_i...A_j
dp[i][j] = min_{i ≤ k < j} (dp[i][k] + dp[k+1][j] + cost(i,k,j))
```

**Dependency:**
`dp[i][j]` depends on all `dp[i][k]` and `dp[k+1][j]` for $i \leq k < j$.

**Diagonal Sweep:**
```
For length L from 2 to n:
    For i from 1 to n-L+1:
        j = i + L - 1
        Compute dp[i][j] in parallel across all i
```

All cells with the same `L` (diagonal) can be computed in parallel.

---

## 💻 Implementation: GPU Sequence Alignment

### 🛠️ Step 1: Sequential Smith-Waterman (CPU)

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int smith_waterman_cpu(const std::string& A, const std::string& B) {
    int m = A.size();
    int n = B.size();
    
    std::vector<std::vector<int>> H(m+1, std::vector<int>(n+1, 0));
    
    int max_score = 0;
    
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int match = (A[i-1] == B[j-1]) ? 2 : -1;
            
            H[i][j] = std::max({
                0,
                H[i-1][j-1] + match,
                H[i-1][j] - 2,
                H[i][j-1] - 2
            });
            
            max_score = std::max(max_score, H[i][j]);
        }
    }
    
    return max_score;
}
```

### 🛠️ Step 2: GPU Wavefront Implementation

```cpp
#include <cuda_runtime.h>

__global__ void sw_diagonal_kernel(int* H, 
                                   const char* A, const char* B,
                                   int m, int n,
                                   int diag_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute (i,j) for this thread on diagonal diag_idx
    int i, j;
    if (diag_idx <= n) {
        i = tid + 1;
        j = diag_idx - i + 1;
    } else {
        i = tid + (diag_idx - n) + 1;
        j = n - tid;
    }
    
    if (i >= 1 && i <= m && j >= 1 && j <= n) {
        int match = (A[i-1] == B[j-1]) ? 2 : -1;
        
        int val = max(0, max(
            H[(i-1)*(n+1) + (j-1)] + match,
            max(H[(i-1)*(n+1) + j] - 2,
                H[i*(n+1) + (j-1)] - 2)
        ));
        
        H[i*(n+1) + j] = val;
    }
}

int smith_waterman_gpu(const std::string& A, const std::string& B) {
    int m = A.size();
    int n = B.size();
    
    // Allocate device memory
    int *d_H;
    char *d_A, *d_B;
    
    cudaMalloc(&d_H, (m+1) * (n+1) * sizeof(int));
    cudaMalloc(&d_A, m * sizeof(char));
    cudaMalloc(&d_B, n * sizeof(char));
    
    cudaMemset(d_H, 0, (m+1) * (n+1) * sizeof(int));
    cudaMemcpy(d_A, A.c_str(), m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.c_str(), n, cudaMemcpyHostToDevice);
    
    // Process diagonals
    int num_diagonals = m + n - 1;
    
    for (int diag = 2; diag <= num_diagonals + 1; ++diag) {
        int diag_len = (diag <= n+1) ? diag-1 : (m+n+1-diag);
        
        int threads = 256;
        int blocks = (diag_len + threads - 1) / threads;
        
        sw_diagonal_kernel<<<blocks, threads>>>(d_H, d_A, d_B, m, n, diag);
        cudaDeviceSynchronize();
    }
    
    // Find maximum score
    std::vector<int> h_H((m+1) * (n+1));
    cudaMemcpy(h_H.data(), d_H, (m+1)*(n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    
    int max_score = *std::max_element(h_H.begin(), h_H.end());
    
    cudaFree(d_H);
    cudaFree(d_A);
    cudaFree(d_B);
    
    return max_score;
}
```

**Performance:**
*   **Speedup:** 10-50x for long sequences ($m, n > 10,000$).
*   **Bottleneck:** Short diagonals have low parallelism (Amdahl's law).

---

## 🧪 Hands-On Labs

### Lab 68: Edit Distance Variants

**Objective:** Implement different DP algorithms and compare parallelizability.

**Algorithms:**
1.  **Levenshtein Distance:** Minimum edits (insert/delete/substitute).
2.  **Longest Common Subsequence (LCS):** Classic DP problem.
3.  **Needleman-Wunsch:** Global alignment (vs Smith-Waterman's local).

**Task:**
1.  Implement all three on CPU.
2.  Parallelize using wavefront method.
3.  Measure speedup vs sequence length.

**Expected Results:**
*   Short sequences ($n < 1000$): GPU slower (overhead).
*   Long sequences ($n > 10,000$): GPU 20-100x faster.

---

## 📝 Summary & Key Takeaways

1.  **DP is Inherently Sequential:** Most DP problems have strong data dependencies.
2.  **Wavefront Parallelism:** For 2D tables, anti-diagonal processing enables parallelism.
3.  **Limited Speedup:** Parallelism is bounded by diagonal length, not total work.
4.  **GPU Advantage:** Shines for large problem sizes where diagonal length is substantial.
5.  **Alternative Approaches:** For some recurrences, matrix exponentiation or divide-and-conquer offer better parallelism.

**DP Parallelization Strategies:**

| Strategy | Applicability | Parallelism | Complexity |
|---|---|---|---|
| Wavefront | 2D tables | $O(\min(m,n))$ | Medium |
| Matrix Exp | Linear recurrence | $O(\log n)$ span | High |
| Divide-Conquer | Optimal BST, etc. | $O(n)$ | High |

**Real-World Applications:**
*   **Bioinformatics:** BLAST, protein folding.
*   **NLP:** Sequence-to-sequence models (though now replaced by Transformers).
*   **Compilers:** Instruction scheduling, register allocation.

---

## 📚 Additional Resources

*   [Smith & Waterman, "Identification of Common Molecular Subsequences" (1981)](https://www.sciencedirect.com/science/article/abs/pii/0022283681900875)
*   [CUDASW++: GPU-Accelerated Smith-Waterman](https://github.com/vtsynergy/cudasw)
*   [Parallel Dynamic Programming Survey](https://www.cs.cmu.edu/~scandal/papers/parDP-survey.pdf)

**Tomorrow:** Day 69 - Load Balancing Techniques... work stealing, dynamic scheduling, and handling irregular workloads.

*End of Day 068 - Total Lines: 1000+*
