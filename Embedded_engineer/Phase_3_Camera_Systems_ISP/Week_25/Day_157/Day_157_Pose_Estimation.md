# Day 157: Human Pose Estimation
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Understand** Pose Estimation: Detecting Keypoints (Joints) of the human body.
2.  **Compare** Top-Down (Detector -> Pose) vs Bottom-Up (Heatmaps -> Grouping) approaches.
3.  **Deploy** a Pose model (e.g., MoveNet or PoseNet) on Edge.
4.  **Calculate** Joint Angles (e.g., Elbow Angle) from keypoints.
5.  **Build** a "Squat Counter" application.

---

## 📚 Prerequisites & Preparation
*   **Software:** TensorFlow Lite or PyTorch.
*   **Model:** MoveNet (Thunder/Lightning) from TensorFlow Hub.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Keypoints
*   Standard COCO format has 17 keypoints:
    *   Nose, Eyes, Ears.
    *   Shoulders, Elbows, Wrists.
    *   Hips, Knees, Ankles.
*   **Output:** $(x, y, confidence)$ for each point.

### 🔹 Part 2: Top-Down vs Bottom-Up
*   **Top-Down (e.g., AlphaPose):**
    1.  Run Object Detector (YOLO) to find Person Bounding Boxes.
    2.  Run Pose Estimator on each box.
    3.  **Pros:** Accurate. **Cons:** Slow if many people.
*   **Bottom-Up (e.g., OpenPose):**
    1.  Predict Heatmaps for all joints in the image.
    2.  Predict Part Affinity Fields (PAF) to connect joints to people.
    3.  **Pros:** Constant time regardless of crowd size. **Cons:** Less accurate for small people.

---

## 💻 Implementation Examples

### Example 1: Running MoveNet (Python)

```python
import tensorflow as tf
import numpy as np
import cv2

# 1. Load Model
interpreter = tf.lite.Interpreter(model_path="movenet_singlepose_thunder.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Preprocess
img = cv2.imread("person.jpg")
input_img = cv2.resize(img, (256, 256))
input_img = np.expand_dims(input_img, axis=0).astype(np.int32) # Thunder expects int32

# 3. Inference
interpreter.set_tensor(input_details[0]['index'], input_img)
interpreter.invoke()
keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

# 4. Draw
# Output shape: (1, 1, 17, 3) -> [y, x, score]
kpts = keypoints_with_scores[0][0]
h, w, _ = img.shape

for kpt in kpts:
    y, x, score = kpt
    if score > 0.3:
        cv2.circle(img, (int(x*w), int(y*h)), 5, (0, 255, 0), -1)

cv2.imshow("Pose", img)
cv2.waitKey(0)
```

### Example 2: Calculating Angles

Geometry 101.

```python
import math

def calculate_angle(a, b, c):
    """
    Calculate angle at point b given points a, b, c.
    a, b, c are (x, y) tuples.
    """
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

# Example: Elbow Angle
# Shoulder (5), Elbow (7), Wrist (9)
shoulder = (kpts[5][1], kpts[5][0])
elbow = (kpts[7][1], kpts[7][0])
wrist = (kpts[9][1], kpts[9][0])

angle = calculate_angle(shoulder, elbow, wrist)
print(f"Elbow Angle: {angle:.2f}")
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: The "Squat Counter"

**Objective:** Count reps.

**Steps:**
1.  Track the Hip, Knee, and Ankle.
2.  Calculate the Knee Angle.
3.  **State Machine:**
    *   State UP: Angle > 160.
    *   State DOWN: Angle < 90.
4.  **Logic:** If State transitions UP -> DOWN -> UP, increment counter.

### Lab 2: Posture Corrector

**Objective:** Sit up straight!

**Steps:**
1.  Track Ear and Shoulder.
2.  If `Ear.x` is far ahead of `Shoulder.x` (Forward Head Posture), trigger alert.
3.  **Calibration:** Capture "Good Posture" first to set the baseline.

### Lab 3: Fall Detection

**Objective:** Safety.

**Steps:**
1.  Track the Head Y-coordinate.
2.  If Head Y drops rapidly (Velocity > Threshold) AND Head Y is near the floor (Bottom of frame), trigger alarm.
3.  **Robustness:** Check aspect ratio of bounding box. Standing = Tall. Fallen = Wide.

---

## 🐛 Debugging Pose Estimation

### Debug 1: "Jittery" Joints

**Symptom:** Keypoints shake even when person is still.

**Cause:**
*   Model noise.
*   **Fix:** Apply a **One Euro Filter** (Low-pass filter optimized for human motion) to the $(x, y)$ coordinates.

### Debug 2: Left/Right Swap

**Symptom:** Left hand detected as Right hand.

**Cause:**
*   Occlusion or Back view.
*   **Fix:** Use a model trained with temporal consistency (VideoPose3D) or enforce geometric constraints (Left arm cannot be on right side of body unless crossed).

---

## ⚡ Performance Optimization

### Optimization 1: Tracking

*   Don't run the heavy detector every frame.
*   Run Detector on Frame 1. Get Box.
*   Run Pose on Frame 1.
*   Frame 2: Use the Keypoints from Frame 1 to estimate the new Bounding Box (ROI). Run Pose on ROI.
*   If Confidence drops, re-run Detector.

### Optimization 2: Skeleton Smoothing

*   Use Kalman Filter to predict the next position of joints.
*   Reduces latency perception and smooths jitter.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **What is a "Heatmap" in Pose Estimation?** (A probability map where the highest value represents the likely location of a joint).
2.  **Why is MoveNet faster than OpenPose?** (MoveNet uses a lightweight architecture (MobileNet-like) and focuses on single-person inference).
3.  **What are "Part Affinity Fields"?** (Vectors that encode the direction from one joint to another, used to group joints into skeletons).

### Practical Challenges

1.  **Virtual Gym Trainer:** Build an app that overlays a "Perfect Form" skeleton over the user's video.
2.  **Gesture Control:** Use Wrist and Elbow position to control volume (Up/Down).

---

## 📚 Further Reading & Resources

### Documentation
*   **TensorFlow Lite Pose Estimation.**
*   **"Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields".**

---

## 🎓 Summary

Today we covered:
- ✅ **Keypoints:** The skeleton.
- ✅ **MoveNet:** Fast and accurate.
- ✅ **Geometry:** Calculating angles.
- ✅ **Logic:** Counting reps.
- ✅ **Smoothing:** One Euro Filter.

**Next:** Day 158 - Dataset Generation & Annotation.

---

**Day 157 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


