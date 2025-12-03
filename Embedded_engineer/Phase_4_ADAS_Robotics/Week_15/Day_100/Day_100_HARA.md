# Day 100: Hazard Analysis and Risk Assessment (HARA)
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 100 Focus:**
> Before we write code, we must ask: "What could go wrong?" **HARA** is the systematic process of identifying hazards, assessing their risk (ASIL), and defining **Safety Goals**. It is the foundation of the entire safety case.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Identify** potential malfunctions (e.g., "Lane Keep Assist steers into oncoming traffic").
2.  **Evaluate** Severity (S), Exposure (E), and Controllability (C) for each scenario.
3.  **Determine** the ASIL level for each hazardous event.
4.  **Formulate** Safety Goals (e.g., "LKA shall not apply torque > 3Nm").
5.  **Document** a HARA report for a Lane Keeping System.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 99:** ASIL Levels.
-   **System Functions:** What does the system do?

### Hardware Requirements
-   **None:** Documentation day.

### Software Stack
-   **Markdown/Excel:** For HARA tables.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The HARA Process

1.  **Item Definition:** Define the system boundary (e.g., LKA System: Camera, ECU, EPS Motor).
2.  **Function Definition:** What is it supposed to do? (Keep car in lane).
3.  **Malfunction Analysis:** What if it does the opposite? (Steers out of lane, Stops steering, Oscillates).
4.  **Situation Analysis:** When does this happen? (Highway, City, Rain).
5.  **Risk Assessment:** Calculate S, E, C -> ASIL.
6.  **Safety Goal:** The top-level requirement to prevent the hazard.

### 🔹 Part 2: S, E, C Parameters (ISO 26262-3)

-   **Severity (S):**
    -   S0: No injuries.
    -   S1: Light/moderate injuries.
    -   S2: Severe injuries (survival probable).
    -   S3: Life-threatening/Fatal.
-   **Exposure (E):**
    -   E1: Very low probability (once a year).
    -   E2: Low probability (1% of drive time).
    -   E3: Medium probability (10% of drive time).
    -   E4: High probability (Highway driving).
-   **Controllability (C):**
    -   C0: Controllable in general.
    -   C1: Simply controllable (99% of drivers).
    -   C2: Normally controllable (90% of drivers).
    -   C3: Difficult to control (Uncontrollable).

### 🔹 Part 3: Safety Goals

A Safety Goal is NOT "The system must be safe".
It must be specific:
-   **Bad:** "LKA should work correctly."
-   **Good:** "LKA shall not generate lateral acceleration > 3 m/s²."
-   **Good:** "LKA shall disable itself within 200ms of a fault."

---

## 💻 Implementation: HARA Tool

**Scenario:**
-   **System:** Lane Keeping Assist (LKA).
-   **Task:** Perform HARA for 3 malfunctions.

### 🛠️ Setup
Create `week15_day100` and `hara_lka.md`.

```bash
mkdir -p ~/ros2_ws/src/week15_day100
cd ~/ros2_ws/src/week15_day100
touch hara_lka.md
```

### 👨‍💻 Content: HARA Report (`hara_lka.md`)

```markdown
# HARA Report: Lane Keeping Assist (LKA)

## 1. Item Definition
The LKA system assists the driver in keeping the vehicle within the lane markings. It uses a forward-facing camera and the Electric Power Steering (EPS).

## 2. Function Definition
-   **F1:** Detect lane markings.
-   **F2:** Calculate deviation.
-   **F3:** Apply steering torque to center the vehicle.

## 3. Hazard Analysis

### Hazard 1: Unintended Steering (Hardover)
-   **Malfunction:** System applies max torque to the left suddenly.
-   **Situation:** Highway driving (100 km/h).
-   **Consequence:** Vehicle veers into oncoming traffic or guardrail.
-   **Assessment:**
    -   **Severity (S):** S3 (Head-on collision is fatal).
    -   **Exposure (E):** E4 (Highway driving is common).
    -   **Controllability (C):** C3 (Sudden jerk is hard to catch).
-   **ASIL:** **ASIL D** (S3 + E4 + C3).
-   **Safety Goal (SG1):** LKA shall not apply steering torque > 3 Nm (Driver override limit).

### Hazard 2: Loss of Steering (Function Dead)
-   **Malfunction:** System stops steering while in a curve.
-   **Situation:** Highway curve (Radius 500m).
-   **Consequence:** Vehicle drifts out of lane.
-   **Assessment:**
    -   **Severity (S):** S2 (Run off road).
    -   **Exposure (E):** E3 (Curves are common).
    -   **Controllability (C):** C1 (Driver hands are on wheel, easy to correct).
-   **ASIL:** **QM** or **ASIL A**. (Driver is expected to be in control).
-   **Safety Goal (SG2):** LKA shall warn the driver if function is unavailable.

### Hazard 3: Oscillatory Steering (Ping Pong)
-   **Malfunction:** System steers left-right-left rapidly.
-   **Situation:** Highway straight.
-   **Consequence:** Vehicle instability, driver panic.
-   **Assessment:**
    -   **Severity (S):** S1 (Minor control loss).
    -   **Exposure (E):** E4.
    -   **Controllability (C):** C2 (Annoying, driver fights wheel).
-   **ASIL:** **ASIL B**.
-   **Safety Goal (SG3):** LKA shall detect oscillation and disable within 1 second.

## 4. Safety Goals Summary

| ID | Description | ASIL | Fault Tolerant Time Interval (FTTI) |
|----|-------------|------|-------------------------------------|
| SG1 | Limit Torque < 3Nm | D | 200 ms |
| SG2 | Warn on Failure | A | 1000 ms |
| SG3 | No Oscillation | B | 1000 ms |

```

### 👨‍💻 Code: Python Risk Matrix Calculator

```python
def calculate_asil(S, E, C):
    # ISO 26262-3 Table 4
    # S (0-3), E (0-4), C (0-3)
    
    if S == 0 or E == 0 or C == 0: return "QM"
    
    score = S + E + C
    
    # Specific Rules for D
    if S==3 and E==4 and C==3: return "D"
    if S==3 and E==4 and C==2: return "D" # Or C? Table says D usually.
    
    # Simplified Logic
    if score >= 10: return "D"
    if score >= 9: return "C"
    if score >= 8: return "B"
    if score >= 7: return "A"
    return "QM"

def main():
    scenarios = [
        {"name": "Hardover", "S": 3, "E": 4, "C": 3},
        {"name": "Loss of Function", "S": 2, "E": 3, "C": 1},
        {"name": "Oscillation", "S": 1, "E": 4, "C": 2},
        {"name": "False Brake", "S": 2, "E": 4, "C": 2} # Rear end collision
    ]
    
    print(f"{'Scenario':<20} | S | E | C | ASIL")
    print("-" * 40)
    
    for s in scenarios:
        asil = calculate_asil(s["S"], s["E"], s["C"])
        print(f"{s['name']:<20} | {s['S']} | {s['E']} | {s['C']} | {asil}")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Driver's Role

### Lab Objectives
1.  Review the HARA.
2.  **Critical Thinking:** Why is "Loss of Function" only QM/ASIL A?
    -   **Reason:** LKA is a Level 2 system. The driver is *responsible* for steering. If LKA quits, the driver is already holding the wheel (supposedly).
    -   **Contrast:** If this were a Level 4 Robotaxi, "Loss of Steering" would be **ASIL D** because there is no driver to take over.
3.  **Experiment:** Change C to 3 (Driver asleep).
    -   **Result:** Loss of Function becomes ASIL D. This is why Driver Monitoring Systems (DMS) are crucial.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Over-Classification
**Symptom:** Everything is ASIL D.
**Cause:** Assuming the worst case for everything.
**Solution:** Be realistic about Controllability. Drivers are good at reacting to simple failures (like a light turning off).

#### 2. Under-Classification
**Symptom:** Fatal hazards marked as QM.
**Cause:** Underestimating Severity (e.g., "Airbag deployment at 100km/h is just S1").
**Solution:** Consult accident data. Inadvertent airbag deployment causes loss of control -> S3.

---

## ⚡ Optimization & Best Practices

### 1. FTTI (Fault Tolerant Time Interval)
The Safety Goal must define *how fast* the system must react.
-   **Hardover:** If the wheel turns full left, the car leaves the lane in 300ms.
-   **Requirement:** The system must detect the fault and open the relay within **200ms**.
-   If your code takes 500ms to detect it, you fail.

### 2. Safe State
What happens when a fault is detected?
-   **Fail-Silent:** Turn off. (LKA, Cruise Control).
-   **Fail-Operational:** Keep working (degraded). (Steering in L4, Braking).

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the output of HARA?
    *   **A:** Safety Goals and their ASIL ratings.
2.  **Q:** Does Exposure (E) refer to the failure rate?
    *   **A:** No! It refers to the *operational situation* (e.g., how often are you on a highway?). Failure rate is handled later in hardware metrics.
3.  **Q:** Can I lower the ASIL by adding a warning light?
    *   **A:** Yes, if it improves Controllability (C). If the driver is warned *before* the hazard becomes critical, C improves.

### Challenge Task
**Task:** ACC HARA.
1.  Malfunction: "Unintended Acceleration".
2.  Situation: City traffic (Stop & Go).
3.  Assess S, E, C.
4.  Define Safety Goal. (Hint: "ACC shall not accelerate > X m/s² without driver input").

---

## 📚 Further Reading & References
-   [ISO 26262 Part 3: Concept Phase](https://www.iso.org/standard/43464.html)
-   [NHTSA Functional Safety Guide](https://www.nhtsa.gov/sites/nhtsa.gov/files/documents/12-06-2016-functional-safety-assessment-of-automated-lane-centering-system.pdf)

---

**Day 100 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
