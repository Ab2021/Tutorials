# Day 180: Ergonomic Assistance
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> Lift with your knees? No, lift with the Robot.
> - **Focus:** Industrial Exoskeletons, Biometrics, Fatigue Monitoring, and Adaptive Force Support.
> - **Code:** `ergo_estimator.py`. Uses Human Pose (Day 177) to estimate Lumbar Spinal Load (L5/S1 torque) and triggers a robot assistance to carry the weight.
> - **Theory:** RULA (Rapid Upper Limb Assessment) and REBA metrics.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Calculate** Static Biomechanical Load metrics (RULA).
2.  **Implement** a Fatigue Monitoring system using Computer Vision or Wearables.
3.  **Design** an Assistance Control Loop (Impedance reduction).
4.  **Integrate** an Exoskeleton control signal based on EMG (Simulated).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Camera.
- (Optional) Myo Armband or EMG sensor simulation.

### Software Environment
```bash
pip install mediapipe numpy
```

### Prior Knowledge
- Physics (Torque = Force x Distance).
- HRI (Week 26).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Cost of Work

Musculoskeletal Disorders (MSDs) are the #1 cost in manual labor.
*   **Risk Factors:**
    1.  Force (Heavy loads).
    2.  Posture (Awkward bending).
    3.  Repetition.
*   **RULA Score:** 1 (Low Risk) to 7 (High Risk). Based on arm, wrist, and neck angles.

### 🔹 Part 2: Adaptive Assistance

The robot shouldn't just "do it all". It should "help".
*   **Gravity Compensation:** Robot holds the weight of the tool. User just guides it.
*   **Variable Stiffness:**
    *   *Free Space:* Robot is soft (Low Impedance).
    *   *Load Lifting:* Robot becomes stiff (High Impedance) to assume the burden.

### 🔹 Part 3: Sensing Fatigue

How do we know the human is tired?
1.  **Kinematics:** Movements become jerky (Jerk increases). Trajectories deviate.
2.  **Physiological:** Heart Rate (HRV), Pupil Dilation, EMG (Muscle activity).

---

## 💻 Implementation: The Virtual Ergonomist

We analyze the user's lifting posture in real-time. If they bend their back too much, the robot shouts "Let me help!" (Simulated assistance).

### 🛠️ Project Structure
```text
day180_ergo/
├── src/
│   ├── ergo_estimator.py
│   └── rula_calculator.py
└── README.md
```

### 👨‍💻 RULA Calculator (`src/rula_calculator.py`)

```python
import numpy as np

def calculate_neck_score(angle_deg):
    # Neck 0-10 deg: 1
    # 10-20 deg: 2
    # >20 deg: 3
    if 0 <= angle_deg <= 10: return 1
    elif 10 < angle_deg <= 20: return 2
    else: return 3

def calculate_trunk_score(angle_deg):
    if 0 == angle_deg: return 1
    elif 0 < angle_deg <= 20: return 2
    elif 20 < angle_deg <= 60: return 3
    else: return 4

def get_risk_level(rula_score):
    if rula_score <= 2: return "Acceptable"
    elif rula_score <= 4: return "Investigate Further"
    elif rula_score <= 7: return "Implement Changes Soon"
    else: return "CRITICAL - Stop Work"

# In a real app, this is a complex lookup table (Grandjean Table)
```

### 👨‍💻 Ergo Estimator (`src/ergo_estimator.py`)

```python
import cv2
import mediapipe as mp
import numpy as np
import time

class ErgoMonitor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # State
        self.last_load_state = 0.0 # 0.0 = No Load, 1.0 = Heavy Load
        
    def get_angle(self, a, b, c):
        """ Calculate angle at b given points a, b, c """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
        
    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        h, w, c = frame.shape
        viz_frame = frame.copy()
        
        assistance_level = 0.0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Joints (Hip, Shoulder, Ear)
            hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h]
            shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
            ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].x * w,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR].y * h]
            
            # 1. Calculate Trunk Angle (Vertical vs Hip-Shoulder)
            # Vertical reference
            vertical_point = [hip[0], hip[1] - 100]
            trunk_angle = self.get_angle(vertical_point, hip, shoulder)
            
            # 2. Risk Assess
            risk_color = (0, 255, 0)
            if trunk_angle > 20: 
                risk_color = (0, 255, 255)
            if trunk_angle > 45:
                risk_color = (0, 0, 255)
                # Activate Robot Help
                assistance_level = 1.0
            
            # Visualize
            cv2.line(viz_frame, (int(hip[0]), int(hip[1])), (int(shoulder[0]), int(shoulder[1])), risk_color, 4)
            cv2.putText(viz_frame, f"Bend: {int(trunk_angle)} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, risk_color, 2)
            
            if assistance_level > 0.5:
                 cv2.putText(viz_frame, "ACTIVE ASSIST!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return viz_frame, assistance_level

def main():
    cap = cv2.VideoCapture(0)
    monitor = ErgoMonitor()
    
    # Mock Robot Connection
    robot_torque = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed, assist_req = monitor.process_frame(frame)
        
        # Simple dynamics
        target_torque = assist_req * 50.0 # Provide 50Nm support
        robot_torque = 0.9*robot_torque + 0.1*target_torque # Smooth in
        
        cv2.putText(processed, f"Robot Torque: {robot_torque:.1f} Nm", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow('Ergonomic Monitor', processed)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Tired Worker"

### 1. Lab Objectives
- **Simulate:** Perform a repetitive lifting task (Squats with a box) in front of the camera.
- **Count:** How many reps until "Form Breakdown" (Trunk angle increases)?
- **Trigger:** Set the `max_trunk_angle` thresh to 30 deg.
- **Log:** Record the timestamp when assistance kicks in.

---

## 🚀 Project: "Smart Glove"

**Goal:** Vibration feedback for wrist safety.
1.  **Hardware:** IMU (MPU6050) on wrist.
2.  **Monitor:** Wrist deviation (Flexion/Extension).
3.  **Feedback:** If deviation > 45 deg for > 5 seconds, Vibrate Motor.
4.  **Connect:** Send data to Robot to adjust "Handover Angle" (Day 176) to a more neutral position.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Noisy Landmarks"
*   **Cause:** Baggy clothes or poor lighting.
*   **Fix:** Use `Mediapipe.HOLISTIC` model for better body tracking context. Use simple Moving Average (EMA) on joint angles.

#### 2. "False Positive Assist"
*   **Cause:** User just bending to tie shoe.
*   **Fix:** Context awareness. Only assist if `Object_In_Hand` is detected (Object Detection).

---

## ⚡ Optimization: Energy Regeneration

Active Exoskeletons consume power.
*   **Quasi-Passive:** Use springs/clutches.
*   **Regen:** When lowering a heavy load, drive the motor as a generator to charge the battery (like regenerative braking in EVs).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is the "Neutral Posture"?
    *   **A:** The position where joints are least stressed (usually midrange). E.g., Wrist straight, back straight.
2.  **Q:** Why not provide 100% assistance always?
    *   **A:** Muscle atrophy. If the robot does everything, the human gets weaker. We want "Assist-as-Needed".

### Challenge Task
> **Task:** L5/S1 Load estimator.
> 1. Estimate Upper Body Mass (based on height).
> 2. Estimate Load Mass (e.g., 10kg).
> 3. Calculate Moment Arm from L5/S1 to Hand.
> 4. Torque = Mass * g * Distance.
> 5. If Torque > limit, Warn.

---

## 📚 Further Reading
- **NIOSH Lifting Equation:** Gold standard for industrial lifting limits.
- **Exoskeleton Report:** Industry trends.

---

**Day 180 Complete**
