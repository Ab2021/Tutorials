# Day 101: SOTIF (Safety of the Intended Functionality)
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 101 Focus:**
> ISO 26262 handles things *breaking* (e.g., bit flips, short circuits). **SOTIF (ISO 21448)** handles things *working as designed, but still failing*. If your AI thinks a white truck is the sky (Tesla Autopilot crash), that's a SOTIF issue. The hardware didn't fail; the *function* was insufficient.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Distinguish** between Functional Safety (FuSa) and SOTIF.
2.  **Analyze** the 4 Areas of SOTIF (Known/Unknown, Safe/Unsafe).
3.  **Identify** Triggering Conditions (e.g., Glare, Fog, Adversarial Patches).
4.  **Design** a Validation Plan to move scenarios from "Unknown Unsafe" to "Known Safe".
5.  **Catalog** SOTIF scenarios for a Traffic Sign Recognition system.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 100:** HARA.
-   **Machine Learning:** Bias, Overfitting.

### Hardware Requirements
-   **None:** Theory day.

### Software Stack
-   **Markdown:** For documentation.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The SOTIF Problem

-   **FuSa (ISO 26262):** "The system failed because a resistor burned out."
-   **SOTIF (ISO 21448):** "The system failed because it was foggy and the camera couldn't see the lane."
-   **Root Causes:**
    1.  **Sensor Limitations:** Lidar absorbs on black cars. Camera blinded by sun.
    2.  **Algorithm Limitations:** AI misclassification (False Positives/Negatives).
    3.  **User Misuse:** Driver sleeping in a Level 2 car.

### 🔹 Part 2: The 4 Areas (The SOTIF Matrix)

| | Safe | Unsafe |
|---|---|---|
| **Known** | **Area 1:** Normal Operation (Sunny day). | **Area 2:** Known Limitations (Heavy Rain). *Mitigation: Disengage.* |
| **Unknown** | **Area 4:** Rare safe scenarios. | **Area 3:** The Danger Zone. (Edge cases we haven't thought of). |

**Goal of SOTIF:** Minimize Area 2 (by improving performance) and Eliminate Area 3 (by testing and discovery).

### 🔹 Part 3: Triggering Conditions

What triggers the failure?
-   **Environmental:** Rain, Snow, Fog, Glare.
-   **Infrastructure:** Faded lines, hidden signs.
-   **Actors:** Pedestrian wearing a chicken suit (AI confusion).

---

## 💻 Implementation: SOTIF Scenario Catalog

**Scenario:**
-   **System:** Traffic Sign Recognition (TSR).
-   **Goal:** Identify SOTIF hazards and mitigations.

### 🛠️ Setup
Create `week15_day101` and `sotif_catalog.md`.

```bash
mkdir -p ~/ros2_ws/src/week15_day101
cd ~/ros2_ws/src/week15_day101
touch sotif_catalog.md
```

### 👨‍💻 Content: SOTIF Catalog (`sotif_catalog.md`)

```markdown
# SOTIF Catalog: Traffic Sign Recognition (TSR)

## 1. System Description
TSR uses a front camera to detect speed limits and stop signs. It displays them to the driver and feeds the ACC system.

## 2. Functional Insufficiencies

### A. Sensor Limitations (Camera)
1.  **Sun Glare:**
    -   **Trigger:** Driving East at sunrise.
    -   **Effect:** Camera saturated. Sign not detected (False Negative).
    -   **Risk:** ACC maintains 100km/h in a 50km/h zone.
    -   **Mitigation:** HDR Camera, Map Fusion (Check HD Map for speed limit).
2.  **Dirt/Mud:**
    -   **Trigger:** Off-road driving.
    -   **Effect:** Lens obscured.
    -   **Mitigation:** Blockage detection algorithm -> Warn Driver.

### B. Algorithm Limitations (AI)
3.  **Adversarial Patch:**
    -   **Trigger:** Sticker on a Stop Sign.
    -   **Effect:** Stop Sign classified as "Speed Limit 45".
    -   **Risk:** Car does not stop at intersection.
    -   **Mitigation:** Temporal consistency check (Must see sign for 5 frames), Map Fusion.
4.  **Similar Objects:**
    -   **Trigger:** Billboard with a picture of a Stop Sign.
    -   **Effect:** False Positive braking (Phantom Brake).
    -   **Mitigation:** Check object size/location (Billboards are high up).

## 3. Validation Plan (Reducing Area 3)

To discover "Unknown Unsafe" scenarios:
1.  **Simulation:** Run 1,000,000 miles in CARLA with randomized weather/lighting.
2.  **Shadow Mode:** Deploy code to fleet. Compare AI output vs Human driver behavior.
    -   *If AI says "Stop" but Human accelerates -> Potential False Positive.*
3.  **Fuzzing:** Inject noise into the image to find decision boundaries.

## 4. Acceptance Criteria
-   **False Positive Rate:** < 1 per 10,000 km.
-   **False Negative Rate:** < 1 per 1,000 km (Map backup available).
```

### 👨‍💻 Code: SOTIF Risk Scorer

```python
class SOTIFScenario:
    def __init__(self, name, probability, severity, controllability):
        self.name = name
        self.prob = probability # 0.0 to 1.0 (Exposure)
        self.sev = severity # 0 to 3
        self.ctrl = controllability # 0 to 3 (Driver reaction)
        
    def is_acceptable(self):
        # SOTIF Acceptance Logic
        # If Severity is High (3), Probability must be Extremely Low
        
        risk_score = self.prob * self.sev * self.ctrl
        
        if self.sev == 3 and self.prob > 1e-6:
            return False, "Fatal Risk too frequent"
            
        if self.sev >= 2 and self.ctrl == 3:
            return False, "Uncontrollable Severe Risk"
            
        return True, "Acceptable"

def main():
    scenarios = [
        SOTIFScenario("Sun Glare (Missed Sign)", 1e-2, 2, 1), # Common, Severe, but Driver sees it
        SOTIFScenario("Phantom Brake (Billboard)", 1e-4, 2, 2), # Rare, Severe, Driver startled
        SOTIFScenario("Adversarial Stop Sign", 1e-8, 3, 3), # Very Rare, Fatal, Hard to control
        SOTIFScenario("Snow Covered Sign", 1e-1, 1, 0) # Very Common, Minor, Easy
    ]
    
    print(f"{'Scenario':<30} | {'Status':<10} | Reason")
    print("-" * 60)
    
    for s in scenarios:
        ok, reason = s.is_acceptable()
        status = "OK" if ok else "UNSAFE"
        print(f"{s.name:<30} | {status:<10} | {reason}")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Phantom Brake

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   "Sun Glare" is OK because Controllability is good (Driver sees the sign even if car doesn't).
    -   "Adversarial Stop Sign" is OK because Probability is tiny ($10^{-8}$).
3.  **Critical Thinking:**
    -   What if "Adversarial Stop Sign" becomes a TikTok trend? Probability jumps to $10^{-3}$.
    -   **Result:** Status becomes **UNSAFE**.
    -   **Action:** Manufacturer must issue an OTA update to fix the AI model immediately.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Confusing FuSa and SOTIF
**Symptom:** Listing "Camera broken wire" in SOTIF.
**Cause:** That's a hardware fault.
**Solution:** Move it to ISO 26262 HARA. SOTIF is for "Camera working, but blinded".

#### 2. Infinite Area 3
**Symptom:** "We can never test everything."
**Cause:** The real world is infinite.
**Solution:** Use statistical arguments. "We have tested 1 billion miles and haven't seen a failure. We are 99% confident the failure rate is < $10^{-9}$."

---

## ⚡ Optimization & Best Practices

### 1. ODD (Operational Design Domain)
The best way to solve SOTIF is to limit the ODD.
-   **Problem:** Lane keeping fails in snow.
-   **Solution:** Define ODD = "No Snow".
-   **Implementation:** If wipers are on fast, disable LKA. Move the scenario from "Unsafe" to "Safe (Disabled)".

### 2. Sensor Fusion
Single sensors have SOTIF holes. Fusion fills them.
-   Camera blinded by Sun? Radar works.
-   Radar confused by manhole cover? Camera sees it's flat.
-   **Diversity** is the key to SOTIF.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is Area 2 in SOTIF?
    *   **A:** Known Unsafe. We know the system fails here (e.g., Heavy Rain). We must mitigate it (e.g., Disable system).
2.  **Q:** What is Area 3?
    *   **A:** Unknown Unsafe. The "Unknown Unknowns". We find these via testing and simulation.
3.  **Q:** Does SOTIF apply to L4 Robotaxis?
    *   **A:** Yes, even more so! There is no driver to "Control" the failure, so Controllability is always C3. The system must handle everything.

### Challenge Task
**Task:** Lidar SOTIF.
1.  Identify a SOTIF hazard for LiDAR.
2.  Hint: "Exhaust fumes in cold weather".
3.  Effect: LiDAR sees a "Ghost Obstacle" (Cloud of smoke).
4.  Mitigation: Check persistence (does it disappear?) or Camera confirmation.

---

## 📚 Further Reading & References
-   [ISO 21448 Standard](https://www.iso.org/standard/70939.html)
-   [UL 4600 (Safety for Autonomous Products)](https://ul.org/UL4600)

---

**Day 101 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
