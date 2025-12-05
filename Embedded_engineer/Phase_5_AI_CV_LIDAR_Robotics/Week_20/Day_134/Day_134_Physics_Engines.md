# Day 134: Physics Engines (Isaac Gym / PhysX)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 20: Sim-to-Real & Hardware Acceleration

---

> **📝 Content Creator Instructions:**
> Simulation is the new Datacenter.
> - **Focus:** How Physics Engines work (Rigid Body Dynamics, Contact Solvers, GJK), Massively Parallel Simulation (Isaac Gym), and why simulators are faster than real time.
> - **Code:** A Python script `parallel_sim.py` that benchmarks a scalar simulation (For-Loop) vs a Vectorized simulation (NumPy/Tensor) to demonstrate the "Isaac Gym" concept of simulating 4096 robots at once.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between High-Fidelity Sims (Gazebo/PyBullet) and High-Throughput Sims (Isaac Gym/Brax).
2.  **Explain** the Simulation Loop: Forward Dynamics $\to$ Collision Detection $\to$ Constraint Solver $\to$ Integration.
3.  **Implement** a vectorized stepping function to simulate 10,000 agents instantly.
4.  **Visualize** the computational advantage of GPU-based physics.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- GPU (NVIDIA) recommended for real Isaac Gym. CPU is fine for today's proxy code.

### Software Environment
```bash
pip install numpy matplotlib timeit
```

### Prior Knowledge
- Newton's Laws ($F=ma$).
- Integration (Euler vs Runge-Kutta).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Anatomy of a Simulator

1.  **Broad Phase:** "Who is near who?" (Bounding Box check).
2.  **Narrow Phase:** "Are they touching?" (GJK Algorithm).
3.  **Solver:** "Push them apart." (LCP - Linear Complementarity Problem).
4.  **Integration:** "Update positions." ($x_{t+1} = x_t + v_t \Delta t$).

### 🔹 Part 2: The Bottleneck

*   **CPU Sim (Gazebo):** CPU sends data to GPU for rendering, reads back for sensors. PCIe bus is the bottleneck.
*   **GPU Sim (Isaac Gym):** Physics AND Rendering happen on GPU. No memory transfer.
    *   Result: Run 10,000 robots at 10,000 Hz.

### 🔹 Part 3: Vectorization

Instead of looping through objects:
```python
for robot in robots:
    robot.step()
```
We do Algebra:
```python
All_Robots_State += All_Robots_Vel * dt
```
This maps perfectly to GPU Cores (CUDA).

---

## 💻 Implementation: The 10,000 Robot Swarm

We compare CPU-style coding vs GPU-style coding (proxy using NumPy).

### 🛠️ Project Structure
```text
day134_physics/
├── src/
│   ├── parallel_sim.py
└── output/
    ├── benchmark.png
```

### 👨‍💻 Parallel Sim Benchmark (`src/parallel_sim.py`)

```python
import numpy as np
import time
import matplotlib.pyplot as plt

class ScalarRobot:
    def __init__(self):
        self.pos = np.array([0.0, 0.0])
        self.vel = np.random.randn(2)
        
    def step(self, dt):
        # Physics
        self.pos += self.vel * dt
        # Boundary Check (Bounce)
        if abs(self.pos[0]) > 10: self.vel[0] *= -1
        if abs(self.pos[1]) > 10: self.vel[1] *= -1

def run_scalar(num_robots, steps=100):
    robots = [ScalarRobot() for _ in range(num_robots)]
    dt = 0.01
    
    start = time.time()
    for _ in range(steps):
        for r in robots:
            r.step(dt)
    end = time.time()
    return end - start

def run_vectorized(num_robots, steps=100):
    # State: [N, 2]
    pos = np.zeros((num_robots, 2))
    vel = np.random.randn(num_robots, 2)
    dt = 0.01
    
    start = time.time()
    for _ in range(steps):
        # Physics (All at once)
        pos += vel * dt
        
        # Boundary Check (Boolean Indexing)
        mask_x = np.abs(pos[:, 0]) > 10
        vel[mask_x, 0] *= -1
        
        mask_y = np.abs(pos[:, 1]) > 10
        vel[mask_y, 1] *= -1
        
    end = time.time()
    return end - start

def main():
    counts = [10, 100, 1000, 10000, 100000]
    t_scalar = []
    t_vector = []
    
    print(f"{'Count':<10} | {'Scalar (s)':<10} | {'Vector (s)':<10} | {'Speedup':<10}")
    print("-" * 46)
    
    for c in counts:
        # Scalar
        # Note: 100k scalar is too slow for demo, cap it
        if c <= 10000:
            ts = run_scalar(c)
        else:
            ts = float('nan')
            
        # Vector
        tv = run_vectorized(c)
        
        t_scalar.append(ts)
        t_vector.append(tv)
        
        ratio = f"{ts/tv:.1f}x" if not np.isnan(ts) else "N/A"
        print(f"{c:<10} | {ts:<10.4f} | {tv:<10.4f} | {ratio}")
        
    # Plot
    plt.figure()
    plt.plot(counts[:4], t_scalar[:4], 'r-o', label='Scalar (Loop)')
    plt.plot(counts, t_vector, 'b-o', label='Vectorized (Batch)')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of Robots')
    plt.ylabel('Time (s)')
    plt.title('Physics CPU vs Vectorized Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig("output/benchmark.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Collision"

### 1. Lab Objectives
- **Run:** The Benchmark.
- **Observe:**
    *   10 Robots: Scalar is fine.
    *   10,000 Robots: Vectorized is 100x faster.
- **Modify:** Add Gravity ($v_y -= g \cdot dt$).
- **Challenge:** Implement "Collision with Ground" ($y < 0$) in Vectorized form.
    *   `mask = pos[:,1] < 0`
    *   `pos[mask, 1] = 0`
    *   `vel[mask, 1] *= -0.5` (Damping).

---

## 🚀 Project: "Ant Farm"

**Goal:** Simulate 1000 ants collecting food.
1.  **State:** $(X, Y, HasFood)$.
2.  **Logic:**
    *   If HasFood: Move to Nest $(0,0)$.
    *   Else: Move Randomly.
    *   If near food and !HasFood: `HasFood = True`.
3.  **Implementation:** Pure NumPy/Jax.
4.  **Result:** Emergent trails without loops.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Exploding Simulation"
*   **Cause:** Integration timestep ($dt$) too large. $x_{new} = x + v \cdot dt$. If $v$ is huge, $x$ teleports.
*   **Fix:** Reduce $dt$ or use Semi-Implicit Euler.

#### 2. "Tunneling"
*   **Cause:** Fast object passes entirely through a thin wall in one timestep. No collision detected.
*   **Fix:** Continuous Collision Detection (CCD) - expensive. Or thicker walls.

---

## ⚡ Optimization: Jax / Warp

NumPy is fast, but Jax/Warp is faster.
*   **Jax:** JIT compiles Python to XLA (runs on GPU/TPU).
*   **Warp (NVIDIA):** Write CUDA kernels in Python syntax.
*   **Isaac Gym:** Uses a C++ backend exposed to Python tensors.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Reset" in RL training?
    *   **A:** When a robot falls, we must reset it. In Isaac Gym, we reset *only that robot* in the tensor, while others continue. No need to reload the whole scene.
2.  **Q:** Why not just use Unity/Unreal?
    *   **A:** They are Game Engines. Prioritize "Looking Good" over "Physics Accuracy". Also, hard to run 10k instances headless.
3.  **Q:** Rigid Body vs Soft Body in sim?
    *   **A:** Rigid body is solving 6 equations per body. Soft body (FEM) is solving thousands of points. Soft is much slower.

### Challenge Task
> **Task:** Simple Constraint.
> 1. Link two particles with a distance constraint (Simple pendulum).
> 2. `dist = norm(p1 - p2)`.
> 3. `correction = (dist - L) * 0.5`.
> 4. Move particles towards each other to fix constraint (PBD - Position Based Dynamics).

---

## 📚 Further Reading
- **NVIDIA Isaac Gym:** Paper & Docs.
- **Google Brax:** Differentiable physics engine in Jax.

---

**Day 134 Complete**
