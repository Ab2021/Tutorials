# Day 199: Perception Stack Development
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> Eyes on the prize (the strawberry).
> - **Focus:** Building the Perception Node. Detecting fruits in 2D (YOLO) and locating them in 3D (Depth / PCL).
> - **Code:** `fruit_detector_node.py`. ROS 2 Node subscribing to RGB-D. Publishing `PoseArray`.
> - **Concept:** 2D-to-3D Projection (Pinhole Camera Model).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Train** (or deploy) a custom YOLOv8 model for strawberry detection.
2.  **Synchronize** RGB and Depth image streams (`message_filters`).
3.  **Project** a 2D bounding box center $(u, v)$ to a 3D point $(X, Y, Z)$ using camera intrinsics.
4.  **Transform** the point from `camera_link` to `base_link`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Realsense D435 (Simulated or Real).

### Software Environment
```bash
pip install ultralytics
sudo apt install ros-humble-message-filters
```

### Prior Knowledge
- Object Detection (Day 60).
- TF2 (Day 11).

---

## 📖 Theoretical Pipeline

1.  **Input:** RGB ($W \times H \times 3$) and Depth ($W \times H$).
2.  **Detection:** YOLO finds Bounding Box $(x_{min}, y_{min}, x_{max}, y_{max})$.
3.  **Centroid:** Center $(u, v) = ((x_{min}+x_{max})/2, (y_{min}+y_{max})/2)$.
4.  **Depth Lookup:** $d = \text{DepthImage}[v, u]$.
5.  **Back-Projection:**
    $$ Z = d $$
    $$ X = (u - c_x) \cdot Z / f_x $$
    $$ Y = (v - c_y) \cdot Z / f_y $$
6.  **Output:** `geometry_msgs/PoseStamped`.

---

## 💻 Implementation: Fruit Detector Node

### 🛠️ Project Structure
```text
agribot_perception/
├── agribot_perception/
│   ├── __init__.py
│   └── fruit_detector.py
├── config/
│   └── camera_info.yaml
└── launch/
    └── perception.launch.py
```

### 👨‍💻 Detector Node (`agribot_perception/fruit_detector.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
from ultralytics import YOLO
import tf2_ros
import tf2_geometry_msgs

class FruitDetector(Node):
    def __init__(self):
        super().__init__('fruit_detector')
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt') # Pretrained, finetune for fruit
        
        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Subscribers (Sync RGB + Depth)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_rect_raw')
        
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)
        
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.intrinsics_calibrated = False
        
        # Publisher
        self.pose_pub = self.create_publisher(PoseArray, '/perception/fruits_3d', 10)
        
    def info_callback(self, msg):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.intrinsics_calibrated = True
        self.cam_frame_id = msg.header.frame_id
        
    def image_callback(self, rgb_msg, depth_msg):
        if not self.intrinsics_calibrated: return
        
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough") # mm or m? Usually mm
        
        # Inference
        results = self.model(cv_image, verbose=False)
        
        pose_array = PoseArray()
        pose_array.header = rgb_msg.header # Keep Frame and Stamp
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] != 'apple': continue # Filter
                
                # BBox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                
                # Depth Check
                if 0 <= u < cv_depth.shape[1] and 0 <= v < cv_depth.shape[0]:
                    d_raw = cv_depth[v, u]
                    if d_raw == 0: continue # Invalid depth
                    
                    z_m = d_raw / 1000.0 # Convert mm to m
                    
                    # Back Project
                    x_m = (u - self.cx) * z_m / self.fx
                    y_m = (v - self.cy) * z_m / self.fy
                    
                    # Create Pose
                    pose = Pose()
                    pose.position.x = x_m
                    pose.position.y = y_m
                    pose.position.z = z_m
                    pose.orientation.w = 1.0
                    
                    pose_array.poses.append(pose)
                    
                    # Draw for Debug
                    cv2.circle(cv_image, (u, v), 5, (0, 255, 0), -1)
                    
        self.pose_pub.publish(pose_array)
        
        # Debug View (Optional)
        cv2.imshow("Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = FruitDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 🔬 Lab Exercise: "TF Transform"

### 1. Lab Objectives
- **Run:** Launch the Detector Node.
- **Visualize:** `rviz2`. Add "PoseArray". Topic: `/perception/fruits_3d`.
- **Issue:** The poses are in `camera_link` (Z is forward). We need them in `base_link` (X is forward) for the arm.
- **Fix:** Modify code to transform points using `tf_buffer.transform()`.
- **Verify:** When the robot turns, the fruit markers stay fixed in the world frame (Rviz "Fixed Frame: map").

---

## 🚀 Project Steps

1.  **Dataset:** If simulating "red spheres", YOLO won't detect them as apples unless re-trained or if the model generalizes well (COCO has apples).
2.  **Color Thresholding:** Backup plan. Use HSV thresholding for Red pixels -> Centroid -> Depth. Simpler, faster, but less robust to lighting.
3.  **Filtering:** Temporal accumulation. Don't publish a fruit unless seen for 5 consecutive frames.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Zero Depth"
*   **Cause:** Stereovision/Lidar has a min range. If fruit is < 20cm, depth is 0 (Blind spot).
*   **Fix:** Ensure the simulation places fruits > 30cm away, or use a customized ToF sensor with close range.

#### 2. "Sync Failed"
*   **Cause:** `message_filters` needs identical timestamps. If Gazebo publishes RGB at 30Hz and Depth at 15Hz, exact match fails.
*   **Fix:** Use `ApproximateTimeSynchronizer` instead of `TimeSynchronizer`.

---

## ⚡ Optimization: ROI (Region of Interest)

Don't scan the sky.
*   **Crop:** Only run YOLO on the bottom half of the image (ground/bushes).
*   **Benefit:** 2x Speedup.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why use `PoseArray` instead of `PointCloud2`?
    *   **A:** Bandwidth. Sending 5 coordinates (Fruits) is bytes. Sending 1M points is MegaBytes. We only care about the fruit locations.
2.  **Q:** What happens if $f_x$ (focal length) is wrong?
    *   **A:** Scale ambiguity. Variable $Z$ will be wrong, so the arm will stop short or punch the fruit.

### Challenge Task
> **Task:** "Ripe vs Unripe".
> 1. Classification logic.
> 2. If Class "Apple" detected $\to$ Calculate average Hue of ROI.
> 3. If Hue < 20 (Red) $\to$ Pick. If Hue > 40 (Green) $\to$ Ignore.

---

## 📚 Further Reading
- **OpenCV:** `solvePnP` (For 6D pose estimation if size is known).
- **YOLOv8:** Training on Custom Data.

---

**Day 199 Complete**
