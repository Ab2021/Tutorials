# Day 44: Gesture Recognition (Pose Estimation)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 7: Human-Robot Interaction (HRI)

---

> **📝 Content Creator Instructions:**
> Pointing at an object is faster than saying "Pick up the red mug on the left."
> - **Focus:** MediaPipe (Google), Skeletal Tracking, and Vector Geometry.
> - **Code:** Controlling a simulated robot using Hand Gestures (Open/Close Fist, Pointing).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** MediaPipe Hands/Pose for real-time skeletal tracking (CPU optimized).
2.  **Calculate** geometric features (Joint Angles) to classify gestures reliably.
3.  **Map** hand coordinates to Robot Arm End-Effector commands (Teleoperation).
4.  **Implement** a "Stop" gesture detection for safety.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Webcam.

### Software Environment
```bash
pip install mediapipe opencv-python numpy
```

### Prior Knowledge
- Forward Kinematics (FK) / Inverse Kinematics (IK).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Top-Down vs Bottom-Up

1.  **Top-Down (Detect-then-Track):** Run YOLO to find Person. Run Pose on the box. Accurate but slow if many people. (AlphaPose).
2.  **Bottom-Up:** Find all joints (Keypoints) in image. Connect them into skeletons (PAF - Part Affinity Fields). Fast for crowds. (OpenPose).
3.  **MediaPipe:** BlazePose (Top-Down). High speed, single-person focus. Perfect for HRI.

### 🔹 Part 2: Vector Geometry for Classification

Neural Networks (LSTM) can classify gestures, but simple Math is faster/robust for basic ones.
*   **"Stop" (Open Palm):** Fingers extended. $\text{Dist}(\text{Tip}, \text{Wrist}) > \text{Dist}(\text{MCP}, \text{Wrist})$.
*   **"Fist" (Closed):** Tips close to Palm.
*   **"Pointing":** Index extended, others closed. Vector from Wrist $\to$ IndexTip defines direction.

---

## 💻 Implementation: Hand Teleop

Control a Robot's Velocity $(v, \omega)$ with your hand.
*   **Distance (Thumb-Index):** Controls Linear Velocity $v$.
*   **Tilt (Wrist Rotation):** Controls Angular Velocity $\omega$.

### 🛠️ Project Structure
```text
day44_gesture/
├── src/
│   ├── hand_tracker.py
│   ├── gesture_mapper.py
└── run_teleop.py
```

### 👨‍💻 Hand Tracker (`src/hand_tracker.py`)

```python
import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        lm_list = []
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                
                h, w, c = img.shape
                for id, lm in enumerate(hand_lms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    
        return lm_list, img
```

### 👨‍💻 Gesture Logic (`src/gesture_mapper.py`)

Landmarks: 4=ThumbTip, 8=IndexTip.

```python
import math

class GestureController:
    def __init__(self):
        pass
        
    def calculate_cmd(self, lm_list):
        if not lm_list:
            return 0.0, 0.0
            
        # 1. Linear Velocity (Pinch Distance)
        x1, y1 = lm_list[4][1], lm_list[4][2] # Thumb
        x2, y2 = lm_list[8][1], lm_list[8][2] # Index
        
        dist = math.hypot(x2 - x1, y2 - y1)
        
        # Calibration: 50 pixels = 0 m/s, 200 pixels = 1.0 m/s
        v_cmd = np.interp(dist, [50, 200], [0, 1.0])
        
        # 2. Angular Velocity (Hand Position X center)
        # Use Wrist (0) position relative to image center
        x_wrist = lm_list[0][1]
        width = 640 # Assume standard cam
        
        # If hand is on left side -> Turn Left. Right -> Turn Right.
        # Deadband in center (280-360)
        err = x_wrist - (width / 2)
        if abs(err) < 40:
            w_cmd = 0.0
        else:
            w_cmd = -np.interp(err, [-320, 320], [-1.0, 1.0])
            
        return v_cmd, w_cmd
```

### 👨‍💻 Main Loop (`run_teleop.py`)

```python
import cv2
from src.hand_tracker import HandDetector
from src.gesture_mapper import GestureController

cap = cv2.VideoCapture(0)
detector = HandDetector()
mapper = GestureController()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) # Mirror view for natural interaction
    
    lm_list, img = detector.find_hands(img)
    
    v, w = mapper.calculate_cmd(lm_list)
    
    # Overlay Info
    cv2.rectangle(img, (20, 20), (200, 100), (0, 0, 0), -1)
    cv2.putText(img, f"V: {v:.2f} m/s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"W: {w:.2f} rad/s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Teleop", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
```

---

## 🔬 Lab Exercise: The "Stop" Safety

### 1. Lab Objectives
- Implement a safety override.
- **Gesture:** Open Palm facing camera (High Five).
- **Logic:**
    - fingers 8, 12, 16, 20 are extended (Tip y < PIP y).
    - If detected, override `v, w` to `0, 0`.
- **Test:** Drive robot towards wall, use hand to stop it.

---

## 🚀 Project: "Follow Me Mode"

**Goal:** Robot follows the operator (Skeleton).
1.  **Detection:** MediaPipe Pose determines Person Center $(C_x, C_y)$ and Size (Area).
2.  **Tracking:**
    - `PID(Area_Target - Area_Current)` $\to$ Linear Velocity (Keep distance).
    - `PID(Center_Image - C_x)` $\to$ Angular Velocity (Keep centered).
3.  **HRI:** Raise Right Hand to engage "Follow". Raise Left Hand to "Stay".

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Jittery" Landmarks
*   **Symptom:** Hand coordinates shake even when hand is still. Causes robot control noise.
*   **Fix:** **One Euro Filter** (Low-pass filter with adaptive cutoff). Smooths jitter but keeps latency low during fast movement.

#### 2. "False Positive"
*   **Symptom:** Detects a "face" as a hand, or a chair leg as a finger.
*   **Fix:** Check `detection_confidence`. Also enforce temporal consistency (Hand shouldn't teleport across screen).

---

## ⚡ Optimization: 3D Gesture

Use Depth Camera (RealSense).
*   MediaPipe gives 2D image coords $(u, v)$.
*   Depth Map gives $Z$ at $(u, v)$.
*   Result: True 3D pointing vector $(X, Y, Z)$.
*   Benefit: Can distinguish "Pointing at floor" vs "Pointing at wall".

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why mirror the image?
    *   **A:** Human cognitive mapping. When I move right, I expect the "mirror me" to move right. Un-mirrored feels inverted and confusing for teleop.
2.  **Q:** Limitations of RGB Pose?
    *   **A:** Occulusion. If I put my hand behind my back, tracking is lost. Depth/Lidar fusing helps.
3.  **Q:** Latency required for HRI?
    *   **A:** < 100ms. If robot lags, human over-corrects (Oscillation).

### Challenge Task
> **Task:** Gesture Recognition with SVM.
> 1. Record landmarks for "Fist", "Palm", "Peace".
> 2. Normalize features (relative to wrist).
> 3. Train `sklearn.svm.SVC`.
> 4. Run inference. More robust than if/else rules.

---

## 📚 Further Reading
- **MediaPipe Hands:** Google AI Blog.
- **HRI Metrics:** "Godspeed Questionnaire".

---

**Day 44 Complete**
