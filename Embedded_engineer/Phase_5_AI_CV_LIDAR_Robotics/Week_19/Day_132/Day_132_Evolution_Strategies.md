# Day 132: Evolution Strategies (Genetic Algorithms)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 19: Soft Robotics & Bio-Inspired Control

---

> **📝 Content Creator Instructions:**
> Nature didn't use gradients. Nature used death.
> - **Focus:** Evolutionary Algorithms (EA), Genetic Algorithms (GA), Covariance Matrix Adaptation (CMA-ES), and Evolving morphology/control without derivatives.
> - **Code:** A Python script `evolve_walker.py` that evolves the leg lengths and motor phases of a 2D "Walker" robot to run as fast as possible.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** Gradient Descent (Hill Climbing, efficient but local) vs Evolutionary Algorithms (Population based, global, derivative-free).
2.  **Implement** the Loop: Selection $\to$ Crossover $\to$ Mutation $\to$ Evaluation.
3.  **Apply** CMA-ES (Covariance Matrix Adaptation) for continuous parameter tuning.
4.  **Visualize** the spread of the population over generations.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- CPU (Parallel evaluations help).

### Software Environment
```bash
pip install numpy matplotlib pygad
```

### Prior Knowledge
- Optimization.
- Kinematics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Survival of the Fittest

If you can't differentiate the Reward Function (e.g., Reward = "Did the robot fall over?"), you can't use Backprop.
You use Evolution.
1.  **Genotype:** The DNA (List of numbers: `[leg_len, phase_offset, k_p]`).
2.  **Phenotype:** The Robot instance.
3.  **Fitness:** How far it walked.

### 🔹 Part 2: The Operators

*   **Selection:** Pick the top 10% (Elitism) or Roulette Wheel.
*   **Crossover:** Combine Parent A (`[1, 1, 1]`) and Parent B (`[2, 2, 2]`) to make Child (`[1, 2, 1]`).
*   **Mutation:** Add random noise (`[1.1, 2.05, 0.9]`). Critical to escape local optima.

### 🔹 Part 3: Evolution Strategies (ES) vs GA

*   **GA:** Discrete/Strings usually. Heavily relies on Crossover.
*   **ES (e.g., CMA-ES):** Continuous numbers. Relies on Mutation distribution (Gaussian). Adapts the *shape* of the search cloud (Covariance Matrix) to follow the valley.

---

## 💻 Implementation: Evolving a Walker

We will define a simple parameterized walker and evolve its gait.

### 🛠️ Project Structure
```text
day132_evolution/
├── src/
│   ├── evolve_walker.py
└── output/
    ├── evolution_progress.png
```

### 👨‍💻 Evolution Script (`src/evolve_walker.py`)

Using `pygad` (Python Genetic Algorithm Library).

```python
import pygad
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Physics/Evaluation Function
def fitness_func(ga_instance, solution, solution_idx):
    # Genotype: [LegLength_L, LegLength_R, PhaseOffset, Frequency]
    # Constraints handled by bounds in GA setup
    l1 = solution[0]
    l2 = solution[1]
    phase = solution[2]
    freq = solution[3]
    
    # Simulation (Simplified Analytical Proxy)
    # Assume Speed is proportional to Stride Length * Frequency
    # But Stride Length depends on Leg Lengths
    # Stability penalty: If legs are too asymmetric, it falls (Reward = 0)
    
    # Stride ~ (l1 + l2) * sin(Amplitude)
    # Stability ~ 1 / |l1 - l2|
    
    stability = 1.0 - abs(l1 - l2) # Ideally l1 == l2
    if stability < 0.5:
        return 0.0 # Fell over
        
    speed = (l1 + l2) * freq
    
    # Energy Cost ~ Frequency^3 (Motor power)
    efficiency = 1.0 / (freq**2 + 0.1)
    
    # Fitness = Speed * Efficiency (We want fast efficient walker)
    fitness = speed * efficiency
    
    return fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness = {ga_instance.best_solution()[1]}")

def main():
    # 2. Setup GA
    
    # Genes: [L1 (0.5-2.0), L2 (0.5-2.0), Phase (0-2pi), Freq (0.5-5.0)]
    gene_space = [
        {'low': 0.5, 'high': 2.0},
        {'low': 0.5, 'high': 2.0},
        {'low': 0.0, 'high': 6.28},
        {'low': 0.5, 'high': 5.0}
    ]
    
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=4,
        gene_space=gene_space,
        mutation_percent_genes=25,
        mutation_type="random",
        on_generation=on_generation
    )
    
    # 3. Run
    ga_instance.run()
    
    # 4. Result
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution : {solution_fitness}")
    
    # Interpret
    print(f"Left Leg: {solution[0]:.2f}")
    print(f"Right Leg: {solution[1]:.2f} (Symmetry evolved?)")
    print(f"Freq: {solution[3]:.2f} (Optimal trade-off?)")
    
    # Plot
    ga_instance.plot_fitness(save_dir="output/evolution_progress.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Crippled Robot"

### 1. Lab Objectives
- **Run:** The Optimization.
- **Observe:** It finds L1 $\approx$ L2 and decent Frequency.
- **Scenario:** Break Leg 1. Force `L1 = 0.5` (Fixed, cannot change constraint, or simply penalize heavily if L1 > 0.5 inside fitness).
- **Rerun:**
    *   Does it give up?
    *   Or does it find a new "Limping" gait (High Frequency, small steps)?
- **Result:** Evolution adapts to the new constraints without implicit reprogramming.

---

## 🚀 Project: "Morphology Evolution"

**Goal:** Evolve the *Shape* of the robot.
1.  **Genotype:** $N$ links, Joint Types (Hinge/Slider), Muscle Placements.
2.  **Simulator:** PyBullet / MuJoCo.
3.  **Task:** Move forward.
4.  **Result:** "Karl Sims" creatures. Some evolve to be snakes, some to be frogs, some just roll.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Premature Convergence"
*   **Symptom:** Population clusters on a suboptimal peak (Local Optima).
*   **Cause:** Mutation rate too low or Selection pressure too high (Killer elites).
*   **Fix:** Increase Mutation. Diversity is key.

#### 2. "Noisy Fitness"
*   **Symptom:** Best solution performs well once, then fails.
*   **Cause:** Environment randomness.
*   **Fix:** Evaluate each candidate 5 times and take average.

---

## ⚡ Optimization: CMA-ES

For robotics, plain GA is slow.
*   **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) assumes the fitness landscape is smooth-ish.
*   It learns the *direction* of improvement.
*   Standard for tuning RL policies (ES-RL).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** When to use GA over RL?
    *   **A:** When the Reward is very sparse (e.g., Design Optimization) or non-differentiable. Or when optimizing hyperparameters.
2.  **Q:** What is "Elitism"?
    *   **A:** Copying the absolute best parent directly to the next generation without mutation. Guarantees fitness never drops.
3.  **Q:** Computational Cost?
    *   **A:** High. 100 population x 50 generations = 5000 Simulations. Parallelization is mandatory (Run 100 sims on 100 cores).

### Challenge Task
> **Task:** Hardware Tuning.
> 1. Connect GA to a real PID controller.
> 2. Genes: $K_p, K_i, K_d$.
> 3. Fitness: $1 / (SettlingTime + Overshoot)$.
> 4. Run real experiments (Auto-tune).

---

## 📚 Further Reading
- **Karl Sims (1994):** "Evolving Virtual Creatures".
- **Nature:** "Designing robots with evolutionary algorithms".

---

**Day 132 Complete**
