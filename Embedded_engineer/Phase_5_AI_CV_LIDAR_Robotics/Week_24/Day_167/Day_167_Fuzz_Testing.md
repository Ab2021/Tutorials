# Day 167: Fuzz Testing (Chaos Engineering)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 24: Cybersecurity & Robustness

---

> **📝 Content Creator Instructions:**
> Break it before the road does.
> - **Focus:** Fuzzing (Random/Mutational), Sanitizers (AddressSanitizer), Chaos Engineering (Injecting latency/faults), and Robust Error Handling.
> - **Code:** A Python script `fuzz_planner.py` that tests a Path Planner. The Fuzzer generates thousands of random obstacle maps (Corner cases: Max obstacles, NaN coordinates, Negative radius). We catch "Unhandled Exceptions" and "Timeouts".

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** Fuzz Testing and why it finds bugs that Unit Tests miss.
2.  **Implement** a Mutation Fuzzer for a robotics API.
3.  **Detect** Crashes, Memory Leaks, and Latency Spikes during fuzzing.
4.  **Apply** Chaos Engineering principles (Randomly kill sensors) to the stack.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy
# AFL (American Fuzzy Lop) is C++, we will simulate logic in Python.
```

### Prior Knowledge
- Unit Testing (PyTest).
- Exception Handling.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Infinite Monkey Theorem

Unit tests check what you *expect*. Fuzzing checks what you *don't expect*.
*   **Generation Fuzzing:** Generate valid inputs from scratch (Constraint based).
*   **Mutation Fuzzing:** Take a valid input (e.g., a Rosbag) and flip bits / shuffle data.

### 🔹 Part 2: Sanitizers

C++ crashes nicely (Segfault). Python just raises Exceptions.
*   **ASan (AddressSanitizer):** Detects buffer overflows/use-after-free in C++.
*   **TSan (ThreadSanitizer):** Detects race conditions.
*   **Robotics context:** If `Plan()` takes 200ms usually, but 5000ms on a specific input, that's a "DoS Bug".

### 🔹 Part 3: Chaos Engineering

"The Simian Army".
*   **Chaos Monkey:** Randomly kills processes (e.g., kill the Lidar Driver).
*   **Latency Monkey:** Adds 500ms delay to Can Bus.
*   **Goal:** Ensure the System degrades gracefully (Safe Stop) rather than undefined behavior (Full throttle).

---

## 💻 Implementation: Crashing the Planner

We write a `SimplePlanner` with some hidden bugs (Division by Zero, Infinite Loop). The Fuzzer will find them.

### 🛠️ Project Structure
```text
day167_fuzzing/
├── src/
│   ├── fuzz_planner.py
└── output/
    ├── crash_log.txt
```

### 👨‍💻 The Fuzzer (`src/fuzz_planner.py`)

```python
import numpy as np
import time
import random

class SimplePlanner:
    def plan(self, start, goal, obstacles):
        """
        start: (x, y)
        goal: (x, y)
        obstacles: list of (x, y, radius)
        """
        # HIDDEN BUG 1: Division by Zero if start == goal
        dist_total = np.linalg.norm(np.array(goal) - np.array(start))
        if dist_total < 0.001: 
            # Correct handling would be return []
            # But let's check deep logic
            pass
            
        step_size = 1.0
        steps = int(dist_total / step_size)
        
        # HIDDEN BUG 2: Infinite Loop if steps is HUGE (Integer Overflow-ish or just heavy)
        if steps > 10000:
            # Simulation of a "Hang"
            time.sleep(0.5) 
            
        current = np.array(start, dtype=float)
        path = [current.copy()]
        
        direction = (np.array(goal) - np.array(start)) / dist_total
        
        for i in range(steps):
            current += direction * step_size
            
            # HIDDEN BUG 3: List Index Error on empty obstacles? No, safe loop.
            # HIDDEN BUG 4: What if obstacle radius is Negative?
            
            for obs in obstacles:
                ox, oy, r = obs
                # Bug: If r is NaN, this math explodes
                d = np.linalg.norm(current - np.array([ox, oy]))
                if d < r:
                    # Collision
                    return None # Fail
                    
            path.append(current.copy())
            
        return path

def fuzzer_engine(iterations=1000):
    planner = SimplePlanner()
    crashes = []
    hangs = []
    
    print(f"Starting Fuzzing (N={iterations})...")
    
    for i in range(iterations):
        # 1. Generate Random Inputs
        try:
            # Case A: Random Floats
            sx, sy = random.uniform(-100, 100), random.uniform(-100, 100)
            gx, gy = random.uniform(-100, 100), random.uniform(-100, 100)
            
            # Case B: Edge Cases (0, NaN, Inf)
            if random.random() < 0.1:
                gx, gy = sx, sy # Trigger Bug 1?
            
            obs_list = []
            num_obs = random.randint(0, 50)
            for _ in range(num_obs):
                ox = random.uniform(-100, 100)
                oy = random.uniform(-100, 100)
                r = random.uniform(0.1, 5.0)
                
                # Case C: Nasty Obstacles
                if random.random() < 0.05:
                    r = -5.0 # Negative Radius
                    
                obs_list.append((ox, oy, r))
                
            # 2. Execute with Monitoring
            start_time = time.time()
            
            # --- RUN ---
            planner.plan((sx, sy), (gx, gy), obs_list)
            # -----------
            
            duration = time.time() - start_time
            
            # 3. Check Constraints
            if duration > 0.1: # 100ms budget
                print(f"   [HANG] Iter {i}: Duration {duration:.4f}s")
                hangs.append(i)
                
        except Exception as e:
            print(f"   [CRASH] Iter {i}: {e}")
            crashes.append({'iter': i, 'error': str(e), 'input': '...'})
            
    print(f"\nSummary: {len(crashes)} Crashes, {len(hangs)} Hangs.")
    return crashes

def main():
    # Let's verify we catch bugs.
    # We didn't explicitly implement NaN injection in this simple script, 
    # but let's see if 'Negative Radius' or 'Start==Goal' causes issues.
    
    # Actually, in the code above:
    # Bug 1: dist_total approx 0. direction = (0,0)/0 -> RuntimeWarning (NaN) usually.
    # In Python, diff/0 raises ZeroDivisionError, but numpy returns nan/inf and warns.
    
    crashes = fuzzer_engine(100)
    
    # Let's MANUALLY inject a crash case to prove the concept 
    # if the random fuzzer misses it in 100 tries.
    print("\n--- Manual Injection Test ---")
    planner = SimplePlanner()
    try:
        # If we pass NaN coordinates
        print("Injecting NaN...")
        planner.plan((0,0), (np.nan, 0), [])
    except Exception as e:
        print(f"Caught Expected Crash: {e}")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Segfault"

### 1. Lab Objectives
- **Run:** Sim.
- **Observe:** Numpy handles NaNs gracefully (propagates them).
- **Task:** Make it crash hard.
- **Modify:** Use `math.sqrt` instead of `np.linalg.norm`. `math.sqrt(-1)` throws ValueError.
- **Fuzz:** Inject negative numbers into loops.
- **Result:** Fuzzing discovers valid inputs that cause unexpected Exceptions.

---

## 🚀 Project: "Chaos Node"

**Goal:** A ROS 2 Node that kills other nodes.
1.  **Node:** `chaos_monkey_node`.
2.  **Logic:** Every 10 seconds, pick a random active node from `ros2 node list` and `lifecycle_pause` it.
3.  **Observation:** Does the Planning Stack detect the failure and stop? Or does it keep planning with old data?

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Unreproducible Crash"
*   **Cause:** Fuzzer generated 1GB of random data. Crash happened. You didn't save the seed.
*   **Fix:** **ALWAYS** log the Random Seed (`random.seed(42)`). Save the exact input that caused the crash (Minimization).

#### 2. "Memory Leaks"
*   **Cause:** Fuzzer runs forever. RAM usage climbs.
*   **Fix:** Monitor `psutil.Process(pid).memory_info().rss`. If > 1GB, flag leak.

---

## ⚡ Optimization: Coverage-Guided Fuzzing (AFL)

Random guessing is inefficient.
*   **Instrumentation:** Compile code with counters on every branch (if/else).
*   **Logic:** If Input A hit a *new* branch, mutate A more. If B hit same old path, discard B.
*   **Result:** Fuzzer explores deep code paths automatically.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Generation vs Mutation?
    *   **A:** Generation follows grammar (SQL, JSON). Mutation breaks existing valid inputs (Bit flipping).
2.  **Q:** What is a "Heisenbug"?
    *   **A:** A bug that disappears when you try to study it (e.g., Race condition that vanishes when debugger slows down execution).
3.  **Q:** Why fuzz the Path Planner?
    *   **A:** Because map data in the real world is messy. A single NaN coordinate shouldn't crash the car at 60mph.

### Challenge Task
> **Task:** Protocol Fuzzing (V2X).
> 1. Take a valid ASN.1 encoded BSM.
> 2. Fuzz the bitfields.
> 3. Send to `VehicleReceiver`. If Receiver throws Exception, vulnerability found.

---

## 📚 Further Reading
- **Google OSS-Fuzz:** Large scale fuzzing for open source.
- **Netflix TechBlog:** Chaos Engineering.

---

**Day 167 Complete**
