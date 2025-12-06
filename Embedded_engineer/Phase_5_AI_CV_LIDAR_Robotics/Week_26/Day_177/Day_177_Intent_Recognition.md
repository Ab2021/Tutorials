# Day 177: Intent Recognition
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 26: Human-Robot Collaboration

---

> **📝 Content Creator Instructions:**
> Don't just watch the human; understand them.
> - **Focus:** Human Motion Prediction, Gaze Tracking, Gesture Recognition, and "Proactive HRI".
> - **Code:** `intent_predictor.py`. A Recursive Least Squares (RLS) estimator or simple LSTM model that predicts where the human hand will be in $t+1.0s$.
> - **Theory:** Social Force Model vs Deep Learning approaches.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between Explicit Intent (Voice Command) and Implicit Intent (Reaching for a tool).
2.  **Implement** a Motion Prediction algorithm using a Constant Velocity Kalman Filter.
3.  **Integrate** Gaze direction to infer the "Focus of Attention".
4.  **Design** a robot behavior that reacts *before* the human finishes their action.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- RGB Camera (for Gaze/Pose).
- Python libraries for pose estimation.

### Software Environment
```bash
pip install mediapipe numpy
```

### Prior Knowledge
- Kalman Filters (Day 11-13).
- Computer Vision (Week 1).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Why" of Intent

If a robot waits for a command, it's a tool. If it guesses what you need, it's a partner.
*   **Trajectory Prediction:** "Human hand is moving toward the wrench." $\to$ Robot should release the wrench.
*   **Gaze Tracking:** "Human is looking at the screen." $\to$ Robot should display info there.

### 🔹 Part 2: Models of Human Motion

1.  **Physics Based:**
    *   **Constant Velocity (CV):** Good for 0.5s horizon.
    *   **Social Force Model:** Humans are particles repelled by obstacles and attracted to goals.
2.  **Learning Based:**
    *   **Human-3.6M:** Dataset of human poses.
    *   **LSTM/Transformer:** Input sequence of $(x,y,z)$ joints $\to$ Output future sequence.

### 🔹 Part 3: The Gaze Cone

Gaze is a ray originating between eyes.
*   **Focus of Attention (FOA):** Intersection of Gaze Ray with World Objects.
*   **Joint Attention:** Both Robot and Human looking at the same thing.

---

## 💻 Implementation: The Hand-over Predictor

We use MediaPipe Pose (Google) to track the wrist. We use a Kalman Filter to predict if it crosses a "Handover Zone".

### 🛠️ Project Structure
```text
day177_intent/
├── src/
│   ├── intent_predictor.py
│   └── hand_tracker.py (MediaPipe Wrapper)
└── params/
    └── zones.yaml
```

### 👨‍💻 Intent Predictor (`src/intent_predictor.py`)

```python
import cv2
import mediapipe as mp
import numpy as np
import time

# --- KALMAN FILTER COMPONENT ---
class HandKalmanFilter:
    def __init__(self, dt=0.033):
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Transition Matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])
        
        # Measurement Matrix (We measure x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.01 # Process Noise
        self.R = np.eye(2) * 0.5  # Measurement Noise (Pixel jitter)
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
        
    def update(self, z):
        # z: measurement [x, y]
        y = z - self.H @ self.x # Residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# --- MAIN APP ---

class IntentApp:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.kf = HandKalmanFilter()
        
        self.handover_zone_x = 400 # Pixel coord
        self.is_human_reaching = False
        
    def process_frame(self, frame):
        # 1. Detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        h, w, c = frame.shape
        prediction_vis = (0,0)
        
        if results.pose_landmarks:
            # Get Right Wrist
            wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            px, py = wrist.x * w, wrist.y * h
            
            # 2. Update Filter
            z = np.array([[px], [py]])
            self.kf.update(z)
            
            # 3. Predict Future (1.0 sec ahead -> 30 steps)
            future_x = self.kf.x.copy()
            for _ in range(30):
                future_x = self.kf.F @ future_x
                
            pred_px, pred_py = int(future_x[0]), int(future_x[1])
            prediction_vis = (pred_px, pred_py)
            
            # 4. Intent Logic
            # If current is far, but prediction is close -> Reaching
            if px < self.handover_zone_x and pred_px > self.handover_zone_x:
                self.is_human_reaching = True
                cv2.putText(frame, "INTENT: REACHING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.is_human_reaching = False
                
            # Draw
            cv2.circle(frame, (int(px), int(py)), 5, (0,0,255), -1)
            cv2.line(frame, (int(px), int(py)), (pred_px, pred_py), (255, 0, 0), 2)
            cv2.circle(frame, (pred_px, pred_py), 5, (255, 0, 0), -1)
            
        # Draw Zone
        cv2.line(frame, (self.handover_zone_x, 0), (self.handover_zone_x, h), (0, 255, 255), 2)
        
        return frame

def main():
    cap = cv2.VideoCapture(0) # Webcam
    app = IntentApp()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        out = app.process_frame(frame)
        cv2.imshow('Intent Recognition', out)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Gaze Switch"

### 1. Lab Objectives
- **Run:** MediaPipe Face Mesh.
- **Calculate:** Iris position relative to Eye Center.
- **Logic:**
    *   Look Left $\to$ Robot moves Left.
    *   Look Right $\to$ Robot moves Right.
- **Hysteresis:** Must hold gaze for 0.5s to trigger (filtering saccades).
- **Result:** Hands-free robot control.

---

## 🚀 Project: "Anticipatory Door Opening"

**Goal:** Open door before human touches handle.
1.  **Detect:** Human body.
2.  **Predict:** Trajectory intersecting with Door Plane.
3.  **Trigger:** If Time-to-Intersection < 2.0s, Actuate Door.
4.  **Abort:** If Human turns away, Stop Door.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Prediction Overshoot"
*   **Cause:** Constant Velocity model assumes... constant velocity. If human stops reaching, CV model keeps shooting the prediction forward.
*   **Fix:** Use a Constant Acceleration model or dampen the velocity in the update step (friction).

#### 2. "Jittery Pose"
*   **Cause:** Single view occlusion.
*   **Fix:** Low pass filter raw landmarks before Kalman Filter. Or use `min_tracking_confidence=0.8`.

---

## ⚡ Optimization: Probabilistic Intent

Instead of Binary (Reaching / Not Reaching), output $P(\text{Reaching})$.
*   **Bayesian Filter:**
    *   Likelihood $P(Z | \text{Reaching})$: Observation if reaching (moving toward).
    *   Prior $P(\text{Reaching})$: Was reaching last frame?
*   **Threshold:** Trigger action only if $P > 0.9$.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is "Implicit Intent" harder than "Explicit"?
    *   **A:** Explicit involves a designated signal (button/word). Implicit must be inferred from context and noisy motion, which is ambiguous.
2.  **Q:** What is the limit of Constant Velocity prediction?
    *   **A:** Human motion is jerky and starts/stops. CV works for short horizons (<0.5s). For long horizons (>1s), you need intent goals (e.g., "Human is going to the coffee machine").

### Challenge Task
> **Task:** "Give me that".
> 1. Detect "Open Hand" gesture (Palm facing up).
> 2. Combined with Gaze at Robot.
> 3. Trigger "Place Object in Hand" routine.

---

## 📚 Further Reading
- **Human 3.6M:** Dataset.
- **Social LSTM:** Alahi et al., CVPR 2016.

---

**Day 177 Complete**
