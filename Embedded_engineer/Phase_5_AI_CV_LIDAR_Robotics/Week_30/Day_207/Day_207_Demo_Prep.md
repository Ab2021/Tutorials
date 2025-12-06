# Day 207: Demo Preparation
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 30: Capstone Project Part 2

---

> **📝 Content Creator Instructions:**
> The Demo God demands a sacrifice.
> - **Focus:** Preparing the system for a live (or recorded) demonstration. Creating a "Golden Path" script that works 100% of the time.
> - **Code:** `demo_run.sh` and `agribot.rviz`.
> - **Concept:** Minimizing Risk.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Script** a foolproof demo sequence using hardcoded waypoints (backup plan).
2.  **Configure** a beautiful RViz layout (Camera overlay, Robot Model, TF).
3.  **Record** a high-quality screen capture (OBS/Kazma) for the portfolio.
4.  **Prepare** talking points explaining *why* the robot did what it did.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-rqt* flatpak # Install OBS via flatpak if needed
```

### Prior Knowledge
- RViz Configuration.

---

## 📖 Theoretical Verification

### 🔹 The Golden Path

In a demo, you don't show "Exploration". You show "Success".
*   **Don't:** Let the robot wander randomly looking for fruit.
*   **Do:** Place the fruit exactly where the robot expects it.
*   **Why?** Because a demo is about *capability*, not *robustness* (Testing is for robustness).

### 🔹 Visualization is Key

An audience doesn't understand "Text Logs". They understand:
*   **Point Clouds:** Show what the robot sees.
*   **Markers:** Draw lines showing the planned path.
*   **Overlays:** Project "DETECTED: 99%" on the camera feed.

---

## 💻 Implementation: The Demo Launcher

### 🛠️ Project Structure
```text
agribot_demo/
├── launch/
│   └── demo.launch.py
├── rviz/
│   └── presentation.rviz
└── scripts/
    └── auto_screen_recorder.sh
```

### 👨‍💻 RViz Configuration (`rviz/presentation.rviz`)

*   **Global Options:** Fixed Frame = `map`.
*   **Displays:**
    1.  `RobotModel`: Alpha = 1.0.
    2.  `Map`: Topic = `/map`. Color Scheme = `costmap`.
    3.  `Camera`: Topic = `/camera/color/image_raw`. Overlay Alpha = 0.5.
    4.  `path`: Topic = `/plan`. Color = Green.
    5.  `MarkerArray`: Topic = `/perception/fruits_3d`. Shape = Spheres (Red).

### 👨‍💻 Demo Script (`launch/demo.launch.py`)

A simplified launch that ensures Rviz pops up immediately.

```python
# Standard launch with 'rviz_config' argument pointing to presentation.rviz
# Logic to auto-unpause Gazebo
```

### 👨‍💻 Cheat Mode (`scripts/golden_path.py`)

If perception fails during the demo, use this.

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class DemoDirector(Node):
    def __init__(self):
        super().__init__('demo_director')
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
    def send_fake_goal(self):
        # We KNOW the fruit is at (1.0, 0.0)
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = 0.9 # Stop slightly before
        self.pub.publish(msg)

# This is the "Secret Button" the operator presses if the robot stalls.
```

---

## 🔬 Lab Exercise: "The Dress Rehearsal"

### 1. Lab Objectives
- **Clean:** Delete all old logs and temp files.
- **Run:** Execute the full demo sequence 3 times in a row.
- **Timing:** Measure the duration. Ideally < 2 minutes. (Attention spans are short).
- **Crash:** If it crashes on run #3, fix the memory leak or race condition. Start over.

---

## 🚀 Project Steps

1.  **Overlay:** Write a small PyGame or OpenCV node that subscribes to the camera and draws "AgriBot Internal State: SEARCHING" in big cool font. Relay this to a new topic `/camera/overlay`. Show *this* in RViz.
2.  **Voice:** Use `espeak` ("I have detected a strawberry") to add audio cues.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Laggy Video"
*   **Cause:** Recording 4K screen while running Sim.
*   **Fix:** Record inside a separate machine (HDMI capture) or reduce Sim resolution.

#### 2. "Murphy's Law"
*   **Scenario:** Internet goes out during Live Demo.
*   **Fix:** **Offline Everything.** Docker containers should not try to `git pull` or `apt install` at runtime.

---

## ⚡ Optimization: Narrative

Don't just say "It picked the fruit."
Say: "The robot fused Lidar and Vision to navigate the unstructured terrain, utilized a Deep Convolutional Network to identify the ripeness, and executed a 6-DoF inverse kinematic plan to harvest without damage." **Sell it.**

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is a "Golden Path"?
    *   **A:** The specific sequence of inputs that results in a successful output with the highest probability.
2.  **Q:** Why use Rviz Overlays?
    *   **A:** To communicate internal state (Fear/Confidence) to the human observer.

### Challenge Task
> **Task:** "Blooper Reel".
> 1. Compile a video of all the times the robot crashed, dropped fruit, or drove into a wall.
> 2. Show this *after* the successful demo. It shows honesty and how much work went into the success.

---

## 📚 Further Reading
- **Steve Jobs:** iPhone Launch Keynote (The ultimate "Golden Path" demo).
- **ROS 2 Visualization:** Rviz Plugins tutorials.

---

**Day 207 Complete**
