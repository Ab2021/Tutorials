# Day 108: Event Cameras (Neuromorphic Vision)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> Pixels don't wait for a clock.
> - **Focus:** The biology-inspired "Event" (Change in polarity), Latency (microseconds), and Dynamic Range (>120dB).
> - **Code:** A node that subscribes to `dvs_msgs/EventArray`, accumulates events into a "Frame" for standard CV visualization, and implements a basic "Activity Filter" to ignore noise.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Contrast** Frame-based cameras (Synchronous, Redundant) vs Event-based cameras (Asynchronous, Sparse).
2.  **Process** the `(x, y, t, p)` data stream using Python.
3.  **Explain** why Event cameras cannot be "blinded" by the sun (High Dynamic Range).
4.  **Reconstruct** visual frames from events for debugging (Accumulation).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Prophesee / IniVation / DVXplorer Camera. (Or Simulator/Dataset).

### Software Environment
```bash
sudo apt install ros-humble-dvs-msgs ros-humble-event-camera-msgs
pip install numpy opencv-python
```

### Prior Knowledge
- Logarithmic Scale.
- Asynchronous Systems.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Biological Inspiration

Standard cameras are like taking a photo 30 times a second.
*   **Wasteful:** If nothing moves, you still process 2 million pixels.
*   **Slow:** Latency is at least 33ms (at 30fps).
*   **Blind:** Bright sun washes out the sensor; shadows are black.

Event Cameras (Silicon Retina):
*   **Independent Pixels:** Each pixel monitors its own log-intensity.
*   **Trigger:** If intensity changes by threshold $\theta$, emit an event.
*   **Output:** `(x, y, timestamp, polarity (+1/-1))`.
*   **Speed:** Microsecond resolution.

### 🔹 Part 2: The Data Stream

Instead of `Image`, we get `EventArray`.
```cpp
struct Event {
  uint16 x;
  uint16 y;
  time timestamp;
  bool polarity; // ON (Brighter) or OFF (Darker)
};
```
Bandwidth is proportional to *Scene Activity*.
*   Stationary Camera looking at wall: 0 Bytes/sec.
*   Fast moving drone: High Bandwidth.

### 🔹 Part 3: Algorithms

You can't use `cv2.findContours` on a list of events directly.
*   **Accumulation:** Collect events for 30ms $\to$ Create Image $\to$ Run OpenCV. (Defeats the purpose of low latency, but good for visualization).
*   **Event-by-Event:** Update a Kalman Filter or Spiking Neural Network (SNN) with *each* event.

---

## 💻 Implementation: Event Accumulator

We will write a "Renderer" that turns events into a displayable image.

### 🛠️ Project Structure
```text
day108_events/
├── src/
│   ├── event_renderer.py
└── data/
    └── sample_events.bag (User provides)
```

### 👨‍💻 Renderer Node (`src/event_renderer.py`)

```python
import rclpy
from rclpy.node import Node
from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class EventRenderer(Node):
    def __init__(self):
        super().__init__('event_renderer')
        self.sub = self.create_subscription(EventArray, '/dvs/events', self.cb, 100)
        self.pub = self.create_publisher(Image, '/dvs/reconstruction', 10)
        self.bridge = CvBridge()
        
        # Sensor Resolution (e.g., DAVIS346 is 346x260)
        self.width = 346
        self.height = 260
        
        # Background canvas (Gray)
        self.canvas = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        
        # Decay (Fading effect)
        self.timer = self.create_wall_timer(0.033, self.decay_canvas)

    def cb(self, msg):
        # Unpack events
        # Note: In real C++, this is faster. In Python, looping 100k events is slow.
        # Vectorized approach recommended.
        
        # Simple Loop (for demonstration)
        for e in msg.events:
            if e.x >= self.width or e.y >= self.height: continue
            
            color = (0, 0, 255) if e.polarity else (255, 0, 0) # Red=ON, Blue=OFF
            self.canvas[e.y, e.x] = color

    def decay_canvas(self):
        # Fade back to gray (127)
        # This creates a "Motion Trail" effect
        # current > 127 -> subtract
        # current < 127 -> add
        
        # Vectorized Decay
        # (Simplified: Just Reset for now to see instant activity)
        # self.canvas[:] = 127 
        
        # Publish
        img_msg = self.bridge.cv2_to_imgmsg(self.canvas, encoding="bgr8")
        self.pub.publish(img_msg)
        
        # Reset canvas for next batch accumulation (Alternative strategy)
        self.canvas.fill(127)

def main():
    rclpy.init()
    rclpy.spin(EventRenderer())
```

---

## 🔬 Lab Exercise: "The Spinning Fan"

### 1. Lab Objectives
- **Scenario:** Point camera at a fast spinning fan.
- **Comparison:**
    *   **Standard Camera:** Motion blur. The blades look like a disk.
    *   **Event Camera:** Sharp edges. You can see the individual blades regardless of speed (until pixel saturation).
    *   **Viz:** Run `rqt_image_view` on the reconstruction topic.

---

## 🚀 Project: "High-Speed Dodge"

**Goal:** Detect an incoming ball thrown at the robot.
1.  **Stream:** Events.
2.  **Filter:** Isolate the "Circle" of events moving across the sensor.
3.  **Latency:** Standard camera detects ball at $t=33ms$. Event camera detects the "Leading Edge" of the ball at $t=1ms$.
4.  **Response:** Actuate Servo to dodge.
5.  **Challenge:** Writing the tracker *without* forming an image. (Compute centroid of events in last 1ms).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Too Much Noise"
*   **Cause:** Thermal noise. Hot pixels fire events randomly.
*   **Fix:** "Background Activity Filter" (BAF). Ignore an event at $(x,y)$ if no neighbors fired in the last $dt$.

#### 2. "Bandwidth Saturation"
*   **Cause:** Moving the camera violently creates events on *every* pixel. USB 3.0 chokes.
*   **Fix:** Adjust bias settings (Threshold). Make it less sensitive.

---

## ⚡ Optimization: Graph-Based SNN

How to process this without heavy GPUs?
*   **Spiking Neural Networks:** Mapped to hardware like Intel Loihi.
*   Process events natively.
*   Extremely low power (milliwatts).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if the Event Camera is perfectly still looking at a static scene?
    *   **A:** It outputs NOTHING. Zero data.
2.  **Q:** Dynamic Range?
    *   **A:** ~140 dB. Can see inside a tunnel and outside at the same time.
3.  **Q:** Polarity?
    *   **A:** +1 means pixel got brighter. -1 means pixel got darker.

### Challenge Task
> **Task:** Blink Detection.
> 1. Look at a human face.
> 2. Eyes blink efficiently.
> 3. Detect the specific high-frequency burst of events at the eye location.
> 4. Much faster than standard face mesh.

---

## 📚 Further Reading
- **RPG (Robotics and Perception Group) Configs:** Standard tools for Event Cameras.
- **Metavision SDK:** Prophesee's software suite.

---

**Day 108 Complete**
