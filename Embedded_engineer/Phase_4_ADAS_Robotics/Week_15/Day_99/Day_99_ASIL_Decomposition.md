# Day 99: ASIL Decomposition
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 99 Focus:**
> You can write the best code in the world, but if the hardware fails, the car crashes. **ISO 26262** is the bible of automotive safety. Today, we learn about **ASIL (Automotive Safety Integrity Level)** and how to cheat the system (legally) using **Decomposition**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** the 4 ASIL levels (A, B, C, D) and QM (Quality Managed).
2.  **Calculate** ASIL based on Severity (S), Exposure (E), and Controllability (C).
3.  **Apply** ASIL Decomposition rules (e.g., ASIL D = ASIL B(D) + ASIL B(D)).
4.  **Design** a redundant architecture (Two MCUs checking each other).
5.  **Analyze** the cost vs. safety trade-off.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **System Architecture:** Sensors, ECUs, Actuators.
-   **Probability:** Failure Rates (FIT - Failures In Time).

### Hardware Requirements
-   **None:** Pure theory/design day.

### Software Stack
-   **Python:** For simple probability calculations.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is ASIL?

ASIL measures **Risk**.
$$ \text{Risk} = \text{Severity (S)} \times \text{Exposure (E)} \times \text{Controllability (C)} $$

-   **Severity (S0-S3):** How bad is the crash? (S3 = Fatal).
-   **Exposure (E0-E4):** How often does this happen? (E4 = High probability, e.g., Highway driving).
-   **Controllability (C0-C3):** Can the driver save it? (C3 = Hard to control).

**Table:**
-   S3 + E4 + C3 = **ASIL D** (Highest Risk - e.g., Steering Lock at high speed).
-   S1 + E2 + C1 = **QM** (Quality Managed - e.g., Radio fails).

### 🔹 Part 2: ASIL Requirements

-   **ASIL A:** Basic testing.
-   **ASIL B:** Moderate testing.
-   **ASIL C:** Strict testing.
-   **ASIL D:** Extreme redundancy, Lockstep CPUs, ECC RAM.

### 🔹 Part 3: ASIL Decomposition

Building an ASIL D system is expensive (Special CPUs).
**Decomposition** allows you to split the requirement into two independent, lower-ASIL systems.
-   **Rule:** $ASIL(X) + ASIL(Y) \ge ASIL(\text{Goal})$
-   **Examples for ASIL D:**
    -   $B(D) + B(D)$ (Most common).
    -   $C(D) + A(D)$.
    -   $D(D) + QM(D)$.
-   **Requirement:** The two systems must be **Independent** (No common cause failure).

---

## 💻 Implementation: Decomposition Calculator

**Scenario:**
-   **Function:** Lane Keep Assist (LKA).
-   **Hazard:** Unintended Steering (ASIL D).
-   **Architecture:**
    -   Main Path (AI Model).
    -   Monitor Path (Rule-based Check).
-   **Task:** Determine the required ASIL for each path.

### 🛠️ Setup
Create `week15_day99` and `asil_calc.py`.

```bash
mkdir -p ~/ros2_ws/src/week15_day99
cd ~/ros2_ws/src/week15_day99
touch asil_calc.py
```

### 👨‍💻 Code: ASIL Algebra

```python
import pandas as pd

# --- ASIL Mapping ---
# 0=QM, 1=A, 2=B, 3=C, 4=D
ASIL_MAP = {'QM': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
REV_ASIL_MAP = {0: 'QM', 1: 'A', 2: 'B', 3: 'C', 4: 'D'}

def get_asil(S, E, C):
    # Simplified lookup table (ISO 26262 Part 3)
    # S: 1-3, E: 1-4, C: 1-3
    
    # Logic: Sum of indices roughly correlates, but let's use explicit rules
    # This is a simplified version of the standard table
    score = S + E + C
    
    if S == 3 and E == 4 and C == 3: return 'D'
    if S == 3 and E == 4 and C == 2: return 'D'
    if S == 3 and E == 3 and C == 3: return 'D'
    
    if score >= 9: return 'C'
    if score >= 7: return 'B'
    if score >= 5: return 'A'
    return 'QM'

def decompose(target_asil):
    target_val = ASIL_MAP[target_asil]
    options = []
    
    # Iterate all pairs (i, j)
    for i in range(5):
        for j in range(5):
            if i + j >= target_val:
                # Check Independence Requirement
                # Usually we want balanced decomposition or Main + Monitor
                asil1 = REV_ASIL_MAP[i]
                asil2 = REV_ASIL_MAP[j]
                options.append(f"{asil1}({target_asil}) + {asil2}({target_asil})")
                
    return options

def main():
    print("--- ASIL Determination ---")
    # Scenario: Steering Lock at High Speed
    S = 3 # Life-threatening
    E = 4 # High probability of being on highway
    C = 3 # Driver cannot control
    
    asil_goal = get_asil(S, E, C)
    print(f"Scenario: S{S} E{E} C{C} -> ASIL {asil_goal}")
    
    print(f"\n--- Decomposition Options for ASIL {asil_goal} ---")
    options = decompose(asil_goal)
    for opt in options:
        print(opt)
        
    print("\n--- Architecture Selection ---")
    print("Option 1: Single Path (ASIL D)")
    print("   - Cost: $$$ (Lockstep CPU, Redundant RAM)")
    print("   - Complexity: Low")
    
    print("Option 2: Main + Monitor (ASIL B + ASIL B)")
    print("   - Cost: $$ (Standard Auto-grade CPUs)")
    print("   - Complexity: High (Sync, Independence proof)")
    print("   - Example: AI (QM) + Safety Check (ASIL B) -> Wait, QM+B=B. Not enough!")
    print("   - Correction: AI (ASIL B) + Monitor (ASIL B) = ASIL D.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Monitor Design

### Lab Objectives
1.  Run the script.
2.  **Observation:** ASIL D can be decomposed into B+B, C+A, or D+QM.
3.  **Design Task:**
    -   **Main Path:** A Deep Neural Network (DNN) predicts steering angle. DNNs are hard to certify as ASIL B (Black box). Let's assume it's **QM**.
    -   **Monitor Path:** A simple check: `if abs(steer_angle) > 5 deg: Block`. This is easy to certify as **ASIL D**.
    -   **Result:** QM(D) + D(D) = ASIL D.
    -   **Trade-off:** The monitor is simple but "dumb". It might block valid maneuvers (False Positive).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Common Cause Failure (CCF)
**Symptom:** Both Main and Monitor fail at the same time.
**Cause:** They share the same Power Supply or the same Clock.
**Solution:** **Freedom from Interference (FFI)**. Use separate power regulators, separate clocks, and ideally separate chips.

#### 2. Dependent Failures
**Symptom:** Decomposition rejected by auditor.
**Cause:** Software libraries shared between paths.
**Solution:** Use different compilers or different teams for Main vs Monitor.

---

## ⚡ Optimization & Best Practices

### 1. E-Gas Monitoring Concept
Standard architecture for Engine Control (3-Level):
-   **Level 1:** Functional Level (Torque calculation).
-   **Level 2:** Function Monitoring (Is Torque > Limit?).
-   **Level 3:** Controller Monitoring (RAM/ROM checks, Watchdog).

### 2. Safety Element out of Context (SEooC)
Buying a chip (e.g., Nvidia Orin) that is "ASIL B Ready".
-   It means the chip *can* support ASIL B if you use it correctly (enable ECC, run self-tests).
-   It does *not* mean your system is automatically ASIL B.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between ASIL and SIL (Safety Integrity Level)?
    *   **A:** ASIL is for Automotive (ISO 26262). SIL is for Industrial (IEC 61508). They are roughly comparable (ASIL D ~ SIL 3).
2.  **Q:** Can I decompose ASIL A?
    *   **A:** Yes (QM + A), but usually not worth the effort. ASIL A is easy to meet with a single path.
3.  **Q:** What is "Controllability C0"?
    *   **A:** Controllable in general (e.g., nuisance only). Results in QM.

### Challenge Task
**Task:** Brake System HARA.
1.  Define Hazard: "Loss of Braking".
2.  S = 3 (Fatal). E = 4 (Always driving). C = 3 (Cannot stop). -> ASIL D.
3.  Define Hazard: "Unintended Braking" (Phantom Brake).
4.  S = 2 (Rear-end collision). E = 4. C = 2 (Driver can accelerate override). -> ASIL B.
5.  Why is Loss of Braking worse?

---

## 📚 Further Reading & References
-   [ISO 26262 Wikipedia](https://en.wikipedia.org/wiki/ISO_26262)
-   [ASIL Decomposition Guide](https://www.synopsys.com/automotive/what-is-asil.html)

---

**Day 99 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
