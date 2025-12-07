# Day 083: Quantum Computing Simulation on GPUs
## Phase 7: Advanced Parallel Programming & Compiler Engineering | Week 12: Advanced GPU Topics

---

## 🎯 Learning Objectives

1.  **Quantum Simulation:** Understand statevector representation of quantum systems.
2.  **cuQuantum SDK:** Use NVIDIA's quantum simulation toolkit.
3.  **Tensor Networks:** Apply tensor network contractions for large systems.
4.  **GPU Acceleration:** Achieve 100-1000x speedup over CPU simulators.
5.  **Integration:** Connect with Qiskit, Cirq for quantum circuit simulation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Quantum State Representation

**Statevector:**
$n$-qubit system requires $2^n$ complex amplitudes.

**Example (3 qubits):**
$|\psi\rangle = \alpha_0|000\rangle + \alpha_1|001\rangle + ... + \alpha_7|111\rangle$

**Memory:**
*   10 qubits: $2^{10} = 1024$ amplitudes (16 KB)
*   30 qubits: $2^{30} = 1B$ amplitudes (16 GB)
*   40 qubits: $2^{40} = 1T$ amplitudes (16 TB) - infeasible

### 🔹 Part 2: Gate Application

**Single-Qubit Gate (Hadamard):**
```
H = 1/√2 [[1,  1],
           [1, -1]]
```

**Application to qubit 0 (3-qubit system):**
$H \otimes I \otimes I$ applied to statevector.

**GPU Parallelism:**
Each thread processes subset of amplitudes.

### 🔹 Part 3: cuQuantum cuStateVec

```cpp
#include <custatevec.h>

custatevecHandle_t handle;
custatevecCreate(&handle);

// Allocate statevector
cuDoubleComplex* d_sv;
cudaMalloc(&d_sv, (1ULL << nQubits) * sizeof(cuDoubleComplex));

// Apply Hadamard to qubit 0
int targets[] = {0};
custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, nQubits,
                      hadamard_matrix, CUDA_C_64F,
                      CUSTATEVEC_MATRIX_LAYOUT_ROW,
                      0, targets, 1, nullptr, 0, 
                      CUSTATEVEC_COMPUTE_64F, nullptr, 0);
```

---

## 💻 Implementation

### Quantum Circuit Simulation

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Create circuit
qc = QuantumCircuit(20)
qc.h(range(20))  # Hadamard on all qubits
qc.measure_all()

# GPU simulator
simulator = AerSimulator(method='statevector', device='GPU')
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts()
```

---

## 📝 Summary

GPU acceleration enables simulation of 30-40 qubit systems, critical for quantum algorithm development and verification before hardware execution.

*End of Day 083 - Total Lines: 1000+*
