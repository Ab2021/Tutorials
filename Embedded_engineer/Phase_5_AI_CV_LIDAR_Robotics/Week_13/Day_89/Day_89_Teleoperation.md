# Day 89: Teleoperation & Imitation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 13: Humanoid Robotics

---

> **📝 Content Creator Instructions:**
> Be the robot.
> - **Focus:** VR Teleoperation (Quest/Vive) via ROS, Motion Retargeting (inverse kinematics for different limb lengths), and Latency Compensation.
> - **Code:** A `retargeting_node` that takes Human Joint Angles (from Mocap/MediaPipe) and maps them to the Robot URDF structure.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** a VR headset (OpenXR) to publish `Pose` messages to ROS 2.
2.  **Solve** the "Correspondence Problem": Mapping Human Arm (Length $L_H$) to Robot Arm (Length $L_R$).
3.  **Implement** "Predictive Display" to help the operator handle 200ms delay.
4.  **Record** a high-quality "Dataset" for Imitation Learning (Day 78/100).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- VR Headset OR Webcam (for MediaPipe tracking).

### Software Environment
```bash
pip install mediapipe numpy scipy
# For VR: ros-humble-vr-ros2-bridge (hypothetical or custom)
```

### Prior Knowledge
- Inverse Kinematics (Day 51).
- TF Trees.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Teleop Pipeline

1.  **Input:** Human Motion (VR controllers, Mocap suit, or RGB Camera).
2.  **Mapping (Retargeting):** Convert Human Frame $\to$ Robot Frame.
3.  **Control:** Send Joint Targets or Cartesian Targets to robot.
4.  **Feedback:** Video Feed (FPV) sent back to Human.

### 🔹 Part 2: Motion Retargeting

Direct joint mapping is bad (Robot elbow breaks).
*   **Method 1: Task Space (IK).** Track the Hand Position relative to Shoulder.
    *   $P_{robot\_hand} = \text{Scale} \times P_{human\_hand}$.
    *   Solver calculates joint angles.
    *   **Pros:** Preserves task intent (Reaching).
    *   **Cons:** Elbow might be naturally down for human, but up for robot.
*   **Method 2: Joint Space (Angles).** Map Shoulder/Elbow angles directly.
    *   **Pros:** Preserves style.
    *   **Cons:** End-effector position mismatch if limb lengths differ.
*   **Hybrid:** Task Space for Hand, Joint Space suggestion for Elbow to resolves redundancy (Nullspace).

### 🔹 Part 3: Latency & Transparency

*   **Round Trip Time:** Camera $\to$ Encode $\to$ Network $\to$ VR $\to$ Human $\to$ Network $\to$ Robot. (Typ. 100-300ms).
*   **Predictive Display:** Show a "Ghost Robot" in VR that moves instantly (simulated) while the Real Robot (video feed) lags behind.
*   **Transparency:** Haptic feedback when the robot hits a wall.

---

## 💻 Implementation: Webcam to Robot (Retargeting)

We will use MediaPipe Pose (Webcam) to control a simulated robot arm.

### 🛠️ Project Structure
```text
day89_teleop/
├── src/
│   ├── mediapipe_tracker.py
│   └── retargeter.py
└── launch/
    └── teleop_demo.launch.py
```

### 👨‍💻 Tracker (`src/mediapipe_tracker.py`)

Extracts normalized landmarks.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import mediapipe as mp
import cv2

class HumanTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')
        self.pub_wrist = self.create_publisher(Point, '/human/wrist', 10)
        self.mp_pose = mp.solutions.pose.Pose()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            # MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_pose.process(rgb)
            
            if res.pose_landmarks:
                # Extract Right Wrist (Index 16) relative to Right Shoulder (12)
                lms = res.pose_landmarks.landmark
                shoulder = lms[12]
                wrist = lms[16]
                
                # Simple relative vector (Normalized 0-1)
                dx = wrist.x - shoulder.x
                dy = wrist.y - shoulder.y
                dz = wrist.z - shoulder.z # Inferred depth
                
                msg = Point(x=dx, y=dy, z=dz)
                self.pub_wrist.publish(msg)
                
            rclpy.spin_once(self, timeout_sec=0.01)

def main():
    rclpy.init()
    HumanTracker().run()
```

### 👨‍💻 Retargeter (`src/retargeter.py`)

Maps Relative Vector $\to$ Robot IK Target.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
# Import IK Solver from Day 51 (or standard KDL/MoveIt)

class Retargeter(Node):
    def __init__(self):
        super().__init__('retargeter')
        self.sub_human = self.create_subscription(Point, '/human/wrist', self.cb, 10)
        self.pub_robot = self.create_publisher(PoseStamped, '/servo_server/target_pose', 10)
        
        self.scale_factor = 0.8 # Human arm ~0.6m, Robot arm ~0.8m?
        # Tuning: How much motion maps to robot workspace
        
    def cb(self, msg):
        target = PoseStamped()
        target.header.frame_id = "base_link"
        target.header.stamp = self.get_clock().now().to_msg()
        
        # Coordinate Flip: MediaPipe Y is Down. Robot Z is Up.
        # Human X (Left/Right) -> Robot Y
        # Human Y (Up/Down) -> Robot Z
        
        # Shoulder at 0,0,0.5
        target.pose.position.x = 0.5 - (msg.z * self.scale_factor) # Forward depth
        target.pose.position.y = -msg.x * self.scale_factor
        target.pose.position.z = 0.5 - (msg.y * self.scale_factor)
        
        # Orientation: Fixed forward grasp
        target.pose.orientation.w = 1.0
        
        self.pub_robot.publish(target)

def main():
    rclpy.init()
    node = Retargeter()
    rclpy.spin(node)
```

---

## 🔬 Lab Exercise: "The Mirror Game"

### 1. Lab Objectives
- Run Tracker + Retargeter + Simulated Robot (MoveIt Servo).
- **Action:** Move your hand in circles.
- **Observation:** Robot mimics motion.
- **Issue:** Jitter. MediaPipe input is noisy.
- **Fix:** Add a `OneEuroFilter` (Low pass filter) to the coordinates before publishing.
- **Issue:** Singularity. Stretching arm too far causes robot to jerk.
- **Fix:** Clamp the target magnitude to $0.95 \times MaxReach$.

---

## 🚀 Project: "Shadow Boxing"

**Goal:** Dual Arm Teleop.
1.  **Setup:** Track Left and Right Wrists.
2.  **Safety:** Self-Collision Avoidance (MoveIt handles this).
3.  **Task:** Shadow boxing. Robot punches the air.
4.  **Critical:** Turn off tracking if confidence is low, or robot will flail when you turn sideways.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Robot moves opposite way"
*   **Cause:** Coordinate frame mismatch (`camera_frame` vs `base_link`). Usually Y/Z flip.
*   **Fix:** Visualize TF. Point your hand Up. See where the TF moves in Rviz.

#### 2. "Lag is intolerable"
*   **Cause:** `MoveIt Servo` publishing too slow? Or Webcam low FPS.
*   **Fix:** Increase Servo rate to 100Hz. Ensure Webcam is 30/60FPS.

---

## ⚡ Optimization: Shared Autonomy

Don't control everything.
*   **Human:** "Go roughly there" (2D input).
*   **Robot:** "I handle orientations and obstacle avoidance" (6D solution).
*   **Example:** You point at a cup. Robot automatically aligns gripper for a side-grasp, ensuring collision-free path.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why is Task Space mapping preferred for manipulation?
    *   **A:** Because the goal is usually "Interacting with valid objects in 3D space". Preserving the joint angles isn't helpful if the hand misses the cup.
2.  **Q:** What is the "Correspondence Problem"?
    *   **A:** Mapping degrees of freedom between dissimilar bodies (e.g., Human with flexible spine vs Rigit Robot Torso).
3.  **Q:** How does VR reduce cognitive load?
    *   **A:** Stereoscopic 3D vision provides Depth Perception. Natural head tracking aligns camera with gaze.

### Challenge Task
> **Task:** Mocap Recording.
> 1. Record `rosbag` of `joint_states` while teleoperating.
> 2. Replay it on the robot.
> 3. Does it look exactly the same? (Yes, Deterministic).
> 4. Use this data for Day 78 imitation learning.

---

## 📚 Further Reading
- **ALOHA Paper:** Low-cost teleop hardware.
- **OpenXR Standard:** The future of VR API.

---

**Day 89 Complete**
