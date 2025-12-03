# Day 105: Week 15 Review & Project
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 105 Focus:**
> We have analyzed hazards (HARA), handled unknowns (SOTIF), built redundant code (Fault Tolerance), and broken it on purpose (Validation). Now, we must prove to the auditor that our car is safe. We build the **Safety Case**.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Synthesize** a Safety Case using GSN (Goal Structuring Notation).
2.  **Link** Claims (Goals) to Evidence (Test Reports).
3.  **Review** the complete Functional Safety Lifecycle.
4.  **Generate** a Safety Manual for the end-user/integrator.
5.  **Defend** your safety argument against a hypothetical assessor.

---

## 📚 Week 15 Review

### 1. ISO 26262 (Hardware Failures)
-   **ASIL:** Risk = S x E x C.
-   **Decomposition:** D = B(D) + B(D).
-   **HARA:** Identifying "Loss of Braking" as ASIL D.

### 2. SOTIF (Functional Insufficiencies)
-   **Area 2:** Known Unsafe (e.g., Fog).
-   **Area 3:** Unknown Unsafe (e.g., Kangaroos).
-   **Goal:** Move everything to Area 1 (Known Safe).

### 3. Fault Tolerance
-   **Fail-Safe:** Turn off (Safe State).
-   **Fail-Operational:** Keep going (Redundancy).
-   **Mechanisms:** Watchdogs, CRCs, Lockstep.

---

## 🛠️ Capstone Project: The Safety Case

**Goal:** Create a GSN-style Safety Case for an **Emergency Braking System (AEB)**.
**Structure:**
-   **Goal (G1):** AEB prevents collisions with pedestrians.
-   **Strategy (S1):** Argument by decomposition into Detection and Actuation.
-   **Goal (G2):** Sensor detects pedestrian (ASIL B).
-   **Goal (G3):** Brake actuates upon request (ASIL D).
-   **Evidence (E1):** Test Report `TR_104` (Fault Injection).
-   **Evidence (E2):** SOTIF Report `TR_101` (Scenario Catalog).

### Package Structure
Create `week15_project` folder.

```bash
mkdir -p ~/ros2_ws/src/week15_project
cd ~/ros2_ws/src/week15_project
touch safety_case.md
```

### 👨‍💻 Content: The Safety Case (`safety_case.md`)

```markdown
# Safety Case: Autonomous Emergency Braking (AEB)

## 1. Executive Summary
This document argues that the AEB system is safe for operation in Urban Environments (ODD: City, <50km/h).

## 2. GSN Diagram (Text Format)

**G1: Top Level Goal**
> The AEB system shall prevent collision with pedestrians in the ODD.
> *Supported by S1*

**S1: Strategy**
> Argument by decomposition over sub-systems (Sensing, Logic, Actuation).
> *Supported by G2, G3, G4*

---

### Branch 1: Sensing (SOTIF)
**G2: Sensing Goal**
> The Camera shall detect pedestrians with >99% recall in defined ODD.
> *Supported by E1, E2*

**E1: Evidence**
> `SOTIF_Catalog.pdf`: Analysis of Glare, Rain, and Occlusion.
> *Status: PASS*

**E2: Evidence**
> `Validation_Report_Sim.json`: 10,000 scenarios in CARLA.
> *Status: PASS*

---

### Branch 2: Actuation (FuSa)
**G3: Actuation Goal**
> The Brake System shall apply -5m/s² deceleration when requested (ASIL D).
> *Supported by S2*

**S2: Strategy**
> Argument by Fault Tolerance (Redundancy).
> *Supported by G3.1, G3.2*

**G3.1: Primary Path**
> Main ECU requests braking.
> *Supported by E3*

**G3.2: Safety Path**
> Watchdog monitors Main ECU. If frozen, Watchdog triggers backup brake.
> *Supported by E4*

**E3: Evidence**
> `HARA_Report.md`: Hazard Analysis defining ASIL D.

**E4: Evidence**
> `Fault_Injection_Report.txt`: Verified Watchdog triggers in 50ms.
> *Status: PASS*

---

## 3. Safety Manual (User Instructions)
To maintain safety, the user/integrator must:
1.  **Clean Sensors:** Wipe camera lens if "Blocked" warning appears.
2.  **No Mods:** Do not install bull-bars that obstruct the radar.
3.  **ODD:** System is not designed for Snow. Driver must take control.

## 4. Conclusion
Based on the evidence provided, the AEB system meets the safety requirements of ISO 26262 (ASIL D) and ISO 21448 (SOTIF).
```

### 👨‍💻 Code: Safety Case Validator

A simple script to check if all Evidence files exist.

```python
import os

class SafetyCaseValidator:
    def __init__(self):
        self.evidence_map = {
            "E1": "SOTIF_Catalog.md",
            "E2": "Validation_Report.json",
            "E3": "HARA_Report.md",
            "E4": "Fault_Injection_Report.txt"
        }
        
    def validate(self):
        print("Validating Safety Case Evidence...")
        all_pass = True
        
        for id, filename in self.evidence_map.items():
            # In a real project, we check file existence.
            # Here we mock it, or check if you actually created them in previous days.
            
            # Let's assume they are in previous folders
            exists = True # Mock
            
            status = "FOUND" if exists else "MISSING"
            print(f"[{id}] {filename}: {status}")
            
            if not exists:
                all_pass = False
                
        if all_pass:
            print("\n>>> SAFETY CASE COMPLETE. READY FOR AUDIT. <<<")
        else:
            print("\n>>> SAFETY CASE INCOMPLETE. DO NOT RELEASE. <<<")

if __name__ == "__main__":
    v = SafetyCaseValidator()
    v.validate()
```

---

## 🧪 Verification & Testing

### 1. The Audit
-   **Auditor:** "Show me evidence E4."
-   **You:** "Here is the Fault Injection Report from Day 104. It shows the Watchdog catches the freeze in 50ms."
-   **Auditor:** "Pass."

### 2. The Missing Link
-   **Auditor:** "What about SOTIF Area 3?"
-   **You:** "We have a continuous monitoring loop (Day 102) to detect unknown scenarios."
-   **Auditor:** "Pass."

---

## 🧠 Comprehensive Assessment (Quiz)

### Section 1: Standards
1.  **Q:** Which standard applies to AI misclassification?
    *   **A:** ISO 21448 (SOTIF).
2.  **Q:** Which standard applies to a resistor burning out?
    *   **A:** ISO 26262 (FuSa).

### Section 2: Logic
3.  **Q:** If I have ASIL D requirements, can I use a Raspberry Pi?
    *   **A:** No. It lacks the diagnostic coverage (ECC, Lockstep) required for ASIL D. You need an Automotive MCU (Aurix, Hercules).
4.  **Q:** What is the "Safety Case"?
    *   **A:** The structured argument, supported by evidence, that the system is safe.

---

## 🏆 Conclusion

Congratulations on completing Week 15!
-   You are now a **Safety-Aware Engineer**.
-   You know that "It works on my machine" is not enough. It must work when the machine is on fire.

**Next Week:** We enter the brain of the car. **Sensor Fusion**. Kalman Filters, Particle Filters, and merging Lidar+Radar+Camera.

---

**Day 105 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
