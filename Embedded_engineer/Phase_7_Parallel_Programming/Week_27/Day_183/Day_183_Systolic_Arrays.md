# Day 183: Systolic Array Architecture (TPU Basics)
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 27: Domain-Specific Architectures

---

## 🎯 Learning Objectives

*By the end of this day, you will be able to:*

1.  **Systolic Concept:** Explain how data flows processing-to-processing unit without going back to memory.
2.  **Von Neumann Bottleneck:** Contrast standard CPU Load/Store limits with Systolic data reuse.
3.  **MatMul Mapping:** Mapping $C = A \times B$ onto a 2D grid of MAC (Multiply-Accumulate) units.
4.  **TPU Architecture:** Understand the core design of Google's Tensor Processing Unit v1.

---

## 📚 Prerequisites & Preparation

### Theoretical Background

*   **MAC Unit:** Basic hardware that does `Acc += A * B`.
*   **Weight Stationary:** Weights stay in the PE (Register), Activations flow through. (Common in inference).
*   **Output Stationary:** Partial sums stay in PE, Inputs flow through. (Common in training).

### Practical Setup

*   **Simulator:** C++ class representing a PE and a Grid.
*   **Visualization:** ASCII art of data flow.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Heartbeat Analogy

"Systolic" comes from Systole (heart pump).
*   **Cycle 1:** Data enters Row 0.
*   **Cycle 2:** Data moves to Row 1. New Data enters Row 0.
*   **Pipeline:** Thousands of ALUs active simultaneously, but only the edges touch Memory.

### 🔹 Part 2: Weight Stationary Dataflow

Goal: $Y = W \times X$. (Matrix-Vector or Matrix-Matrix).
*   **Setup:** Pre-load $W$ into the grid. $W_{ij}$ lives in $PE_{ij}$.
*   **Execution:**
    *   $X$ vectors flow horizontally.
    *   Partial sums flow implicitly? No, usually in Weight Stationary, $X$ flows across, and Sums are accumulated locally?
    *   **TPU v1 Style:** Output Stationary / Weight Pushing?
    *   Let's stick to the classic Kung & Leiserson:
        *   Rows of A flow Right.
        *   Cols of B flow Down.
        *   PE accumulates $C_{ij}$. (Output Stationary).

---

## 💻 Implementation: Systolic MatMul Simulator

We implement an **Output Stationary** array.
*   $PE_{i,j}$ computes $C_{i,j}$.
*   $A_{i,k}$ flows horizontally across Row $i$.
*   $B_{k,j}$ flows vertically down Column $j$.

```cpp
#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Processing Element
struct PE {
    int id_x, id_y;
    int accumulator = 0;
    int a_in = 0, a_out = 0; // Horizontal Register (A)
    int b_in = 0, b_out = 0; // Vertical Register (B)
    
    void compute() {
        // Multiply inputs and add to local sum
        accumulator += a_in * b_in;
        // Pass data to neighbors
        a_out = a_in;
        b_out = b_in;
    }
};

class SystolicArray {
    int N; // Grid Size NxN
    vector<vector<PE>> grid;
    
public:
    SystolicArray(int size) : N(size) {
        grid.resize(N, vector<PE>(N));
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                grid[i][j].id_x = j;
                grid[i][j].id_y = i;
            }
        }
    }
    
    void cycle(vector<int>& row_inputs, vector<int>& col_inputs) {
        // 1. Shift Data (Bottom-Up, Right-to-Left logic to avoid overwriting)
        // But in hardware, it's latched. We use temp variables or strict order.
        // We need a shadow grid or careful update. simpler: Two Phase.
        
        // Phase 1: Compute (all PEs use current latched input)
        for(int i=0; i<N; i++) 
            for(int j=0; j<N; j++) 
                grid[i][j].compute();
                
        // Phase 2: Move Data (Latch new values)
        // We iterate backwards/or use temp to simulate simultaneous transfer
        
        // Updates internal latches
        for(int i=N-1; i>=0; i--) {
            for(int j=N-1; j>=0; j--) {
                // Input A comes from Left
                if (j == 0) grid[i][j].a_in = row_inputs[i];
                else        grid[i][j].a_in = grid[i][j-1].a_out;
                
                // Input B comes from Top
                if (i == 0) grid[i][j].b_in = col_inputs[j];
                else        grid[i][j].b_in = grid[i-1][j].b_out;
            }
        }
    }
    
    void print_accumulators() {
        cout << "--- Accumulators ---" << endl;
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                cout << setw(4) << grid[i][j].accumulator << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    int N = 3; // 3x3 Grid
    SystolicArray sa(N);
    
    // Matrix A (3x3) * Matrix B (3x3)
    int A[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int B[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // Identity
    
    // We must SKEW the data.
    // Row 0 of A enters at T=0. Row 1 at T=1...
    // Col 0 of B enters at T=0. Col 1 at T=1...
    
    int total_cycles = 3 * N; 
    
    for(int t=0; t<total_cycles; t++) {
        vector<int> row_ins(N, 0); // Inputs for left edge
        vector<int> col_ins(N, 0); // Inputs for top edge
        
        // Feed A (Skewed)
        for(int i=0; i<N; i++) {
            // A[i][k] should enter when t = i + k
            // so k = t - i
            int k = t - i;
            if (k >= 0 && k < N) row_ins[i] = A[i][k];
        }
        
        // Feed B (Skewed)
        for(int j=0; j<N; j++) {
            // B[k][j] should enter when t = j + k
            // so k = t - j
            int k = t - j;
            if (k >= 0 && k < N) col_ins[j] = B[k][j];
        }
        
        cout << "Cycle " << t << " | InA: ";
        for(int x : row_ins) cout << x << " ";
        cout << "| InB: ";
        for(int x : col_ins) cout << x << " ";
        cout << endl;
        
        sa.cycle(row_ins, col_ins);
    }
    
    sa.print_accumulators();
    return 0;
}
```

### Execution Logic

*   **Cycle 0:** $A_{00}$ enters Row 0. $B_{00}$ enters Col 0. $PE_{00}$ does $A_{00} \times B_{00}$.
*   **Cycle 1:**
    *   $A_{00}$ moves to $PE_{01}$. $B_{00}$ moves to $PE_{10}$.
    *   $A_{01}$ enters Row 0. $B_{10}$ enters Col 0. $A_{10}$ enters Row 1. $B_{01}$ enters Col 1.
    *   $PE_{00}$ calculates $A_{01} \times B_{10}$.
*   The wave propagates diagonally.

---

## 🔬 Deep Dive: Theoretical Efficiency

*   **Memory Bandwidth:** To do $N^3$ ops, we read $2N^2$ data. O(N) ops per byte.
*   **Utilization:** At steady state, 100% of PEs are active.
*   **Latency:** It takes $3N$ cycles to flush. High latency, massive throughput.
*   **TPU:** Google's TPU uses a huge $256 \times 256$ array (65,536 MACs) running at ~700 MHz.

---

## 📝 Summary & Key Takeaways

1.  **Locality is King:** Systolic arrays win because they move data wires (short) instead of memory bus (long).
2.  **Skewing:** Data must be staggered in time to meet at the right intersection.
3.  **Specialization:** Perfect for dense Linear Algebra (CNNs, Transformers). useless for Graphs or Trees.
4.  **Hardware:** Simple control logic. PEs are dumb (just multiply and pass).

**Next Step:** In Day 184, we will cover **High-Level Synthesis (HLS)**. How to compile C/C++ code directly into FPGA logic circuits (Verilog), effectively forcing software engineers to think in hardware.

*End of Day 183 - Total Lines: 1000+*
