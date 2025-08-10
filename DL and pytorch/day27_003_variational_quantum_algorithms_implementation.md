# Day 27.3: Variational Quantum Algorithms - A Practical Deep Dive

## Introduction: A Framework for Near-Term Quantum Advantage

**Variational Quantum Algorithms (VQAs)** represent our most promising path toward achieving a "quantum advantage" on near-term, noisy intermediate-scale quantum (NISQ) devices. The Quantum Neural Network we built in the previous guide is a prime example of a VQA. These algorithms are hybrid quantum-classical methods that use a classical optimizer to train a parameterized quantum circuit.

The core principle is to frame a problem in terms of a **cost function** (or Hamiltonian) whose minimum value corresponds to the solution. The VQA then uses the quantum computer to prepare a trial state (an "ansatz") and estimate the value of the cost function for that state. The classical optimizer uses this estimate to suggest a new set of parameters for the trial state, iterating until the minimum is found.

This guide will take a deeper dive into the VQA framework, focusing on two of its most famous applications: the **Variational Quantum Eigensolver (VQE)** for chemistry and the **Quantum Approximate Optimization Algorithm (QAOA)** for combinatorial optimization.

**Today's Learning Objectives:**

1.  **Understand the VQA Framework:** Formalize the roles of the ansatz, the Hamiltonian, and the classical optimizer.
2.  **Implement VQE:** Use VQE to find the ground state energy of a molecule (H₂), the canonical problem in quantum chemistry.
3.  **Define a Problem Hamiltonian:** Learn how to represent a problem's cost function as a quantum mechanical Hamiltonian operator.
4.  **Implement QAOA:** Use QAOA to find the solution to a classic combinatorial optimization problem (Max-Cut).
5.  **Compare Ansatz Designs:** Understand the difference between a chemically-inspired ansatz (for VQE) and a problem-inspired ansatz (for QAOA).

---

## Part 1: The VQA Framework - A Closer Look

A VQA has three main components:

1.  **Parameterized Quantum Circuit (The Ansatz):**
    *   This is a quantum circuit `U(θ)` with tunable parameters `θ`.
    *   Its job is to prepare a quantum state `|ψ(θ)⟩ = U(θ)|0⟩`.
    *   The **expressivity** of the ansatz determines the range of quantum states it can prepare. A good ansatz should be able to represent the true solution state, but it should also be trainable (not too many parameters, and gradients should be well-behaved).

2.  **Problem Hamiltonian (The Cost Function):**
    *   The **Hamiltonian**, `H`, is an operator that describes the total energy of a quantum system. Its eigenvalues are the possible energy levels of the system.
    *   In a VQA, we map our problem's cost function to a Hamiltonian.
    *   The goal is to find the parameters `θ` that minimize the expectation value `⟨ψ(θ)|H|ψ(θ)⟩`. By the variational principle, this expectation value is always greater than or equal to the true lowest energy (the ground state energy).

3.  **Classical Optimizer:**
    *   This is a standard classical algorithm (e.g., Gradient Descent, Adam, SPSA) that runs on a classical computer.
    *   It receives the estimated cost from the quantum device and proposes the next set of parameters `θ` to try.

---

## Part 2: VQE - Solving for Molecular Energies

The Variational Quantum Eigensolver (VQE) is a VQA designed to find the lowest eigenvalue (ground state energy) of a given Hamiltonian. Its primary application is in quantum chemistry, where it can be used to calculate the properties of molecules.

We will find the ground state energy of the Hydrogen molecule (H₂).

### 2.1. Code: Setting up the H₂ Molecule in PennyLane

PennyLane's `qml.chem` module makes this incredibly easy. It can compute the molecular Hamiltonian for us.

```python
import pennylane as qml
from pennylane import numpy as np

print("--- Part 2: VQE for the Hydrogen Molecule ---")

# --- 1. Define the Molecule ---
# Define the symbols and coordinates of the atoms
symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614]) # In Bohr

# --- 2. Get the Hamiltonian ---
# `qml.chem.molecular_hamiltonian` does all the hard chemistry for us!
# It returns the Hamiltonian operator and the number of qubits required.
H, n_qubits = qml.chem.molecular_hamiltonian(symbols, coordinates)

print(f"Number of qubits required: {n_qubits}")
print("Hamiltonian:")
print(H)

# --- 3. Define the Ansatz ---
# For chemistry, a chemically-inspired ansatz like Unitary Coupled Cluster
# Singles and Doubles (UCCSD) is often used. PennyLane provides a template.
# The Hartree-Fock state is a good starting point.
hf_state = qml.chem.hf_state(electrons=2, qubits=n_qubits)

# The AllSinglesDoubles template creates the UCCSD circuit.
def uccsd_ansatz(weights):
    qml.AllSinglesDoubles(weights, wires=range(n_qubits), hf=hf_state, reps=1)

# --- 4. Create the VQE Cost Function (QNode) ---
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def vqe_cost_function(weights):
    uccsd_ansatz(weights)
    # `qml.expval(H)` calculates the expectation value of our complex Hamiltonian
    return qml.expval(H)

# --- 5. Optimization Loop ---
# Get the number of parameters needed for the ansatz
_, params_shape = qml.AllSinglesDoubles.shape(s=1, d=1, n_wires=n_qubits, n_reps=1)
weights = np.random.uniform(0, 2 * np.pi, params_shape)

optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
energies = []

for i in range(50):
    weights, energy = optimizer.step_and_cost(vqe_cost_function, weights)
    energies.append(energy)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1:2d}: Energy = {energy:.8f} Ha")

print(f"\nFinal Ground State Energy: {energy:.8f} Ha")
# The true value is approximately -1.136 Ha
```

---

## Part 3: QAOA - Solving Combinatorial Optimization

The Quantum Approximate Optimization Algorithm (QAOA) is a VQA designed to find approximate solutions to combinatorial optimization problems.

We will tackle the **Max-Cut** problem. Given a graph, the goal is to partition the nodes into two sets such that the number of edges connecting nodes in different sets is maximized.

### 3.1. Code: Setting up the Max-Cut Problem

1.  **Map to a Cost Hamiltonian:** We assign each node a qubit. The state |0⟩ means the node is in set A, and |1⟩ means it's in set B. The cost Hamiltonian is constructed such that its expectation value is maximized when connected nodes are in different states.
    For an edge (i, j), the term `(1 - ZᵢZⱼ)/2` contributes to the cost, where `Zᵢ` is the Pauli-Z operator on qubit `i`.

2.  **Define the QAOA Ansatz:** The QAOA ansatz has a specific structure. It alternates between applying the **cost Hamiltonian** `H_C` and a **mixer Hamiltonian** `H_M`.
    `|ψ(γ, β)⟩ = e^(-iβ₁H_M) e^(-iγ₁H_C) ... e^(-iβₚH_M) e^(-iγₚH_C) |+⟩^n`
    *   `H_C` is the cost Hamiltonian we defined.
    *   `H_M` is typically the sum of Pauli-X operators on all qubits: `Σ Xᵢ`.
    *   `γ` and `β` are the trainable parameters.

### 3.2. Code: Implementing QAOA for Max-Cut

```python
import networkx as nx

print("\n--- Part 3: QAOA for Max-Cut ---")

# --- 1. Define the Graph ---
n_wires = 4
graph = nx.Graph()
graph.add_edges_from([(0, 1), (0, 3), (1, 2), (2, 3)])
nx.draw(graph, with_labels=True)

# --- 2. Define the Cost Hamiltonian ---
cost_h, _ = qml.qaoa.maxcut(graph)
print("\nCost Hamiltonian:")
print(cost_h)

# --- 3. Define the QAOA Circuit ---
# The QAOA layer in PennyLane simplifies this
def qaoa_layer(gamma, beta):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, qml.qaoa.x_mixer(range(n_wires)))

# The full circuit
@qml.qnode(dev)
def qaoa_cost_function(params):
    # params is a (2, p) array, where p is the number of layers
    # First, prepare the initial state |+>...|+>
    for i in range(n_wires):
        qml.Hadamard(wires=i)
    # Apply the QAOA layers
    qml.layer(qaoa_layer, params.shape[1], params[0], params[1])
    # We want to MAXIMIZE the cost, so we return its negative
    return qml.expval(cost_h)

# --- 4. Optimization Loop ---
p = 2 # Number of QAOA layers
params = np.random.uniform(0, np.pi, (2, p))

optimizer = qml.AdamOptimizer(stepsize=0.1)

for i in range(100):
    params, cost = optimizer.step_and_cost(qaoa_cost_function, params)
    if (i + 1) % 20 == 0:
        print(f"Step {i+1:3d}: Cost = {-cost:.4f}")

print(f"\nFinal Max-Cut Cost: {-cost:.4f}")

# --- 5. Find the Solution ---
# To find the actual partition, we need to sample the final state
@qml.qnode(dev)
def qaoa_solution(params):
    for i in range(n_wires):
        qml.Hadamard(wires=i)
    qml.layer(qaoa_layer, params.shape[1], params[0], params[1])
    return qml.probs(wires=range(n_wires))

probs = qaoa_solution(params)
# The solutions are the bitstrings with the highest probability
solutions = np.argsort(probs)[::-1]
print(f"The most likely solution bitstring is: {solutions[0]:04b}")
# For this graph, the solutions are 0101 and 1010
```

## Conclusion

Variational Quantum Algorithms are a powerful and flexible framework for leveraging near-term quantum hardware to solve meaningful problems. We have seen two of the most prominent examples, VQE and QAOA, in action.

**Key Takeaways:**

1.  **VQAs are General:** The framework of an ansatz, a cost Hamiltonian, and a classical optimizer can be adapted to a wide range of problems.
2.  **The Hamiltonian is the Problem:** The core of applying a VQA is to successfully map your problem's objective function onto a quantum mechanical Hamiltonian.
3.  **Ansatz Design is Crucial:** The choice of the parameterized circuit is critical. A good ansatz should be expressive enough to contain the solution but also trainable.
4.  **VQE is for Chemistry:** VQE's primary use case is finding molecular ground state energies, a problem central to drug discovery and materials science.
5.  **QAOA is for Optimization:** QAOA provides a quantum-native approach to tackling hard combinatorial optimization problems found in logistics, finance, and scheduling.

Understanding the VQA paradigm is essential for anyone looking to apply quantum computing to real-world challenges in the NISQ era.

## Self-Assessment Questions

1.  **Variational Principle:** What does the variational principle guarantee about the expectation value of a Hamiltonian for any trial state?
2.  **VQE:** What is the physical meaning of the value that VQE minimizes?
3.  **Ansatz:** What is the difference between the UCCSD ansatz used in VQE and the alternating operator ansatz used in QAOA?
4.  **QAOA:** In the Max-Cut problem, how do we represent the two partitions of the graph using qubit states?
5.  **Hybrid Nature:** Why are VQAs considered "hybrid" algorithms? What would be the challenge of making them "fully quantum"?
