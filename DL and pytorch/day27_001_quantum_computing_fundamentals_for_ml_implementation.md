# Day 27.1: Quantum Computing Fundamentals for ML - A Comprehensive Practical Guide

## Introduction: The Dawn of a New Computing Paradigm

For decades, machine learning has been powered by classical computers, operating on bits that are either 0 or 1. However, as datasets grow and models become more complex, we are beginning to push the limits of classical computation. Enter **Quantum Computing**, a revolutionary paradigm that leverages the principles of quantum mechanics to process information in fundamentally new ways.

Instead of bits, quantum computers use **qubits**, which can exist in a state of **superposition** (both 0 and 1 at the same time) and can be linked together through **entanglement**. These properties unlock an exponentially larger computational space, offering the potential to solve certain problems that are intractable for even the most powerful classical supercomputers. For machine learning, this translates to the potential for more powerful feature spaces, novel optimization algorithms, and new generative models.

This guide will serve as a comprehensive, practical introduction to the core concepts of quantum computing that are most relevant for machine learning practitioners. We will explore the fundamental building blocks of quantum computation and, most importantly, write code to simulate these concepts, bridging the gap between abstract theory and practical implementation.

**Today's Learning Objectives:**

1.  **Master the Qubit:** Understand the qubit, its mathematical representation, and its visualization on the **Bloch Sphere**.
2.  **Explore Quantum Principles:** Learn about superposition and entanglement, the two pillars of quantum computation, and see them in action.
3.  **Command a Wide Array of Quantum Gates:** Understand a larger set of single- and multi-qubit gates and their matrix representations.
4.  **Understand Quantum Measurement:** Learn the difference between expectation values, probabilities, and samples when extracting classical data from a quantum state.
5.  **Build and Simulate Complex Quantum Circuits:** Use PennyLane to construct and execute multi-qubit, parameterized quantum circuits.
6.  **Master Data Encoding Techniques:** Discover and implement multiple techniques for representing classical data in a quantum state, a critical first step for any quantum machine learning task.

--- 

## Part 1: The Qubit and its Geometric Intuition - The Bloch Sphere

A **qubit** is the fundamental unit of quantum information. Its state can be a **linear combination** of its two basis states, |0⟩ and |1⟩.

The state of a qubit, |ψ⟩, is represented as: |ψ⟩ = α|0⟩ + β|1⟩, where |α|² + |β|² = 1.

While the vector representation is mathematically precise, it can be hard to visualize. A more intuitive picture for a single qubit is the **Bloch Sphere**.

Any pure state of a single qubit can be represented as a point on the surface of a 3D sphere of radius 1. 
*   The **North Pole** represents the |0⟩ state.
*   The **South Pole** represents the |1⟩ state.
*   Points on the surface of the sphere represent superpositions of |0⟩ and |1⟩.

We can parameterize any point on the sphere using two angles, θ (theta) and φ (phi):
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

```
          |0⟩
           ^
          / \
         /   \
        /     \
       |       |
<------|----(x)----|------>  (Equator - Superposition States)
       |       |
        \     /
         \   /
          \ /
           v
          |1⟩
```

### 1.1. Simulating a Qubit with PennyLane

Let's create qubits in specific states and see where they would lie on the Bloch Sphere.

```python
# Ensure you have PennyLane installed: pip install pennylane
import pennylane as qml
from pennylane import numpy as np

print("--- Part 1: The Qubit and the Bloch Sphere ---")

# A "device" is the computational engine for our simulation.
def1 = qml.device("default.qubit", wires=1)

@qml.qnode(def1)
def create_and_measure_qubit(state_vector):
    # Initialize a qubit to a specific state.
    qml.QubitStateVector(state_vector, wires=0)
    # Return the expectation values along X, Y, and Z axes.
    # These values correspond to the Cartesian coordinates on the Bloch Sphere.
    return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]

# --- Example 1: A qubit in the |0> state ---
state_0 = np.array([1.0, 0.0])
coords_0 = create_and_measure_qubit(state_0)
print(f"State |0>: Bloch Coords (X,Y,Z) = ({coords_0[0]:.2f}, {coords_0[1]:.2f}, {coords_0[2]:.2f}) -> North Pole")

# --- Example 2: A qubit in the |1> state ---
state_1 = np.array([0.0, 1.0])
coords_1 = create_and_measure_qubit(state_1)
print(f"State |1>: Bloch Coords (X,Y,Z) = ({coords_1[0]:.2f}, {coords_1[1]:.2f}, {coords_1[2]:.2f}) -> South Pole")

# --- Example 3: A qubit in equal superposition, state |+> ---
# This state is created by applying a Hadamard gate to |0>
state_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
coords_plus = create_and_measure_qubit(state_plus)
print(f"State |+>: Bloch Coords (X,Y,Z) = ({coords_plus[0]:.2f}, {coords_plus[1]:.2f}, {coords_plus[2]:.2f}) -> Equator, X-axis")

# --- Example 4: A qubit in superposition with a phase ---
# This state is created by applying H and then S gate to |0>
state_i = np.array([1/np.sqrt(2), 1j/np.sqrt(2)]) # Note the complex number `1j`
coords_i = create_and_measure_qubit(state_i)
print(f"State |i>: Bloch Coords (X,Y,Z) = ({coords_i[0]:.2f}, {coords_i[1]:.2f}, {coords_i[2]:.2f}) -> Equator, Y-axis")
```

--- 

## Part 2: Quantum Gates - The Operators of Change

Quantum gates are operations that cause a qubit's state vector to rotate on the Bloch Sphere. They are represented by unitary matrices.

### 2.1. A Zoo of Single-Qubit Gates

| Gate      | Matrix Representation             | Description                                       |
|-----------|-----------------------------------|---------------------------------------------------|
| **Pauli-X**   | `[[0, 1], [1, 0]]`                | **NOT** gate. 180° rotation around X-axis.        |
| **Pauli-Y**   | `[[0, -i], [i, 0]]`               | 180° rotation around Y-axis.                      |
| **Pauli-Z**   | `[[1, 0], [0, -1]]`               | 180° rotation around Z-axis. Adds phase.          |
| **Hadamard (H)**| `(1/√2)*[[1, 1], [1, -1]]`      | Creates superposition. Rotates Z-basis to X-basis.|
| **S Gate**    | `[[1, 0], [0, i]]`                | **Phase** gate. 90° rotation around Z-axis.       |
| **T Gate**    | `[[1, 0], [0, e^(iπ/4)]]`         | π/4 gate. 45° rotation around Z-axis.             |
| **RX(θ)**   | `[[c, -is], [-is, c]]`            | Rotation around X-axis by angle θ.                |
| **RY(θ)**   | `[[c, -s], [s, c]]`               | Rotation around Y-axis by angle θ.                |
| **RZ(θ)**   | `[[e^(-iθ/2), 0], [0, e^(iθ/2)]]` | Rotation around Z-axis by angle θ.                |

*(where c=cos(θ/2), s=sin(θ/2))* 

### 2.2. Multi-Qubit Gates and Entanglement

*   **CNOT (Controlled-NOT):** The cornerstone of entanglement. Flips the target qubit if the control is |1⟩.
*   **CZ (Controlled-Z):** Applies a Z gate to the target if the control is |1⟩. It's symmetric.
*   **SWAP:** Swaps the states of two qubits.

### 2.3. Code Example: A Multi-Gate Circuit and Creating a GHZ State

The **Greenberger–Horne–Zeilinger (GHZ) state** is a famous entangled state of three or more qubits.

```python
print("\n--- Part 2: Creating a 3-Qubit GHZ State ---")

dev3 = qml.device("default.qubit", wires=3)

@qml.qnode(dev3)
def create_ghz_state():
    # 1. Start with |000>
    # 2. Apply H to the first qubit -> (1/√2)(|000> + |100>)
    qml.Hadamard(wires=0)
    # 3. Apply CNOT from qubit 0 to 1 -> (1/√2)(|000> + |110>)
    qml.CNOT(wires=[0, 1])
    # 4. Apply CNOT from qubit 0 to 2 -> (1/√2)(|000> + |111>)
    qml.CNOT(wires=[0, 2])
    return qml.probs(wires=[0, 1, 2])

ghz_probs = create_ghz_state()
print("GHZ state created with H and two CNOTs.")
print(f"Probability of |000>: {ghz_probs[0]:.2f}") # Expected: 0.5
print(f"Probability of |111>: {ghz_probs[7]:.2f}") # Expected: 0.5
# All other probabilities should be 0.
```

--- 

## Part 3: Measurement - Collapsing the Wave Function

Measurement is how we extract classical information from a quantum state. It is a probabilistic process. When we measure a qubit in superposition, its state **collapses** to one of the basis states.

PennyLane offers different ways to measure, reflecting what one might do on real hardware.

*   `qml.expval(Operator)`: Returns a single, continuous number - the **expectation value** of an observable. This is like measuring the same state many times and averaging the results. It's great for optimization and training.
*   `qml.probs(wires)`: Returns a full probability distribution over all possible outcomes for the specified wires. This is useful for understanding the final state but is often inefficient to compute for many qubits.
*   `qml.sample(Operator)`: Simulates the process of a single measurement, or "shot." It returns a single classical value (e.g., +1 or -1). Running a circuit with `shots=N` will give you N such samples.

### 3.1. Code Example: Comparing Measurement Types

```python
print("\n--- Part 3: Comparing Measurement Types ---")

# We will use a device with shots=1000 to simulate repeated measurements
dev_shots = qml.device("default.qubit", wires=1, shots=1000)

# Circuit that creates a state with a 75% chance of being |1>
@qml.qnode(dev_shots)
def measurement_circuit():
    # Rotate the qubit so P(|1>) = sin(1.231)^2 ≈ 0.75
    qml.RY(2 * np.arcsin(np.sqrt(0.75)), wires=0)
    return {
        "expval": qml.expval(qml.PauliZ(0)),
        "probs": qml.probs(wires=0),
        "samples": qml.sample(qml.PauliZ(0))
    }

results = measurement_circuit()

# The theoretical expectation value is P(0)*1 + P(1)*(-1) = 0.25*1 + 0.75*(-1) = -0.5
print(f"Expectation Value (average of samples): {np.mean(results['samples']):.3f}")

# The probabilities are calculated from the samples
samples_array = results['samples']
prob_0 = (len(samples_array[samples_array == 1])) / 1000
prob_1 = (len(samples_array[samples_array == -1])) / 1000
print(f"Probabilities (from 1000 shots): P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")

print(f"First 10 samples: {results['samples'][:10]}")
```

--- 

## Part 4: Building Parameterized Quantum Circuits (PQC)

A PQC, also known as a variational quantum circuit, is a quantum circuit that has tunable parameters. These are the parameters we optimize in a QML model, analogous to the weights in a neural network.

A common structure for a PQC layer is an alternating sequence of rotation gates and entangling gates.

### 4.1. Code Example: A Two-Qubit PQC Layer

```python
print("\n--- Part 4: A Two-Qubit PQC Layer ---")

dev4 = qml.device("default.qubit", wires=2)

@qml.qnode(dev4)
def pqc_layer(params):
    # params is a list or array of shape (2, 3) for this layer
    
    # A layer of single-qubit rotations
    qml.RX(params[0, 0], wires=0)
    qml.RY(params[0, 1], wires=0)
    qml.RZ(params[0, 2], wires=0)
    
    qml.RX(params[1, 0], wires=1)
    qml.RY(params[1, 1], wires=1)
    qml.RZ(params[1, 2], wires=1)
    
    # A layer of entangling gates
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)) # Measure the joint Z expectation

# Initialize some random parameters for the layer
np.random.seed(42)
initial_params = np.random.uniform(0, 2 * np.pi, (2, 3))

print("Initial parameters:")
print(initial_params)

# Run the PQC
output_expval = pqc_layer(initial_params)
print(f"\nOutput expectation value from the PQC layer: {output_expval:.4f}")
```

--- 

## Part 5: Encoding Classical Data into Quantum States

This is the critical bridge from classical datasets to quantum models.

| Encoding Method     | Description                                                                 | Pros                               | Cons                                     |
|---------------------|-----------------------------------------------------------------------------|------------------------------------|------------------------------------------|
| **Basis Encoding**  | Maps a bitstring `101` to a basis state `|101⟩`.                               | Simple, lossless.                  | Uses many qubits, no superposition.      |
| **Angle Encoding**  | Encodes features `x_i` into rotation angles `RX(x_i)`.                      | Easy to implement, uses few qubits.| Can be non-linear, information loss.   |
| **Amplitude Encoding**| Encodes an N-dim vector into the amplitudes of a `log2(N)`-qubit state.      | Extremely qubit-efficient.         | Requires normalized data, hard to load.  |

### 5.1. Code Example: Amplitude Encoding with PennyLane

PennyLane provides a convenient template for this.

```python
print("\n--- Part 5: Amplitude Encoding ---")

# A 4-dimensional classical vector
feature_vector = np.array([0.5, 0.1, 0.2, 0.8])

# Normalize it for amplitude encoding
norm = np.linalg.norm(feature_vector)
feature_vector_normalized = feature_vector / norm

# We need log2(4) = 2 qubits
dev5 = qml.device("default.qubit", wires=2)

@qml.qnode(dev5)
def amplitude_encoding_circuit(features):
    # qml.AmplitudeEmbedding handles the complex decomposition into gates
    qml.AmplitudeEmbedding(features=features, wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Run the circuit
encoded_probs = amplitude_encoding_circuit(feature_vector_normalized)

print("Normalized classical vector:", feature_vector_normalized)
print("\nProbabilities of the resulting quantum state:")
for i, prob in enumerate(encoded_probs):
    print(f"  P(|{i:02b}>): {prob:.4f} (should be ≈ {feature_vector_normalized[i]**2:.4f})")
```

## Conclusion

In this comprehensive guide, we have journeyed from the basic definition of a qubit to implementing parameterized, multi-qubit circuits and sophisticated data encoding schemes. We've seen that the counter-intuitive principles of quantum mechanics can be harnessed through a well-defined mathematical framework and simulated effectively with powerful tools like PennyLane.

**Key Takeaways:**

1.  **The Bloch Sphere is Key:** It provides essential geometric intuition for how single-qubit states and gates behave.
2.  **Gates are Rotations:** All quantum operations can be thought of as rotations of state vectors in a high-dimensional space.
3.  **Measurement is Nuanced:** The way we extract information (expval, probs, sample) depends on our goal, whether it's training or inference.
4.  **PQC Layers are the Building Blocks:** QML models are built by stacking parameterized layers, much like classical deep learning.
5.  **Encoding is a Critical Design Choice:** The method used to encode classical data into a quantum state profoundly impacts the performance and feasibility of a QML model.

With these fundamentals in hand, you are now well-equipped to understand and begin building your own Quantum Machine Learning models.

## Self-Assessment Questions

1.  **Bloch Sphere:** Where on the Bloch sphere would the state `(1/√2)(|0⟩ - |1⟩)` lie? (Hint: This is the `|−⟩` state).
2.  **Gates:** What sequence of gates would transform the state |0⟩ into |1⟩ and then back to |0⟩?
3.  **Entanglement:** If you have a GHZ state `(1/√2)(|000⟩ + |111⟩)` and you measure the first qubit to be 0, what is the state of the other two qubits?
4.  **Measurement:** You are training a QML model to perform regression. Which measurement type would be most appropriate for your loss function and why?
5.  **Encoding:** You have a dataset with 8 features per data point. How many qubits would you need to represent one data point using (a) Angle Encoding and (b) Amplitude Encoding?