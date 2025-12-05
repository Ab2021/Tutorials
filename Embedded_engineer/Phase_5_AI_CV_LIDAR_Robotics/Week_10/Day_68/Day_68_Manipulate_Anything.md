# Day 68: Manipulate-Anything (VLM Grasping)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 10: Robot Manipulation & Grasping

---

> **📝 Content Creator Instructions:**
> "Robot, pick up the thing that looks like a banana but is blue."
> - **Focus:** Open-Vocabulary Manipulation, Vision-Language Models (OWL-ViT, CLIP), and Zero-Shot Grasping.
> - **Code:** A node that combines OWL-ViT for detection and our Day 66 Grasp Sampler for actuation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Integrate** Multi-Modal Foundation Models (CLIP/OWL-ViT) into a ROS 2 pipeline.
2.  **Mapping** Natural Language Prompts $\to$ 2D Bounding Boxes $\to$ 3D Frustums $\to$ Grasp Targets.
3.  **Implement** a "Prompt-to-Grasp" node.
4.  **Solve** ambiguity: "Pick the *left* cup" using spatial reasoning.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- RGB-D Camera (Must be calibrated!).
- GPU with 8GB+ VRAM (Running Transformers).

### Software Environment
```bash
pip install transformers opencv-python ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Prior Knowledge
- Day 29 (Vision-Language Models).
- Day 66 (Grasp Sampling).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The "Open Vocabulary" Problem

Traditional detectors (YOLO) are trained on 80 classes (COCO).
*   **Problem:** If you ask for "Screwdriver", YOLO says "Background".
*   **Solution:** Open-Vocabulary Detectors (OWL-ViT, GroundingDINO).
    *   Input: Image + Text "A photo of a screwdriver".
    *   Output: Bounding Boxes matching the text embedding.

### 🔹 Part 2: 2D-to-3D Lifting

We have a 2D Box $(u, v, w, h)$ from the VLM.
How do we grasp it?
1.  **Crop:** Extract the Point Cloud region corresponding to the 2D Box (Frustum Culling).
2.  **Filter:** Remove background points (Depth Thresholding).
3.  **Grasp:** Feed *only* the object points to the Grasp Sampler (Day 66).

### 🔹 Part 3: Spatial Reasoning (Relation Networks)

"Pick the cup *on the plate*."
*   VLM detects "Cup" and "Plate".
*   Planner logic: Find Cup Box $B_c$ and Plate Box $B_p$.
*   Condition: $Center(B_c) \in B_p$.
*   Modern LLMs (GPT-4V) can do this reasoning natively, but for real-time control, we often use geometric heuristics post-detection.

---

## 💻 Implementation: CLIP-Grasp Node

We will use `OWL-ViT` (Open-Vocabulary Object Detection) from Hugging Face.

### 🛠️ Project Structure
```text
day68_vlm_grasp/
├── src/
│   ├── vlm_detector.py
│   └── prompt_grasp.py
└── launch/
    └── interact.launch.py
```

### 👨‍💻 VLM Detector (`src/vlm_detector.py`)

Takes Text Prompt $\to$ Publishes ROI (Region of Interest).

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image as PILImage

class VLMDetector(Node):
    def __init__(self):
        super().__init__('vlm_detector')
        self.sub_img = self.create_subscription(Image, '/camera/color/image_raw', self.img_cb, 10)
        self.sub_prompt = self.create_subscription(String, '/command/prompt', self.prompt_cb, 10)
        self.pub_roi = self.create_publisher(RegionOfInterest, '/target/roi', 10)
        
        self.bridge = CvBridge()
        self.prompt = "a robot" # Default
        
        # Load Model
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.eval()
        
    def prompt_cb(self, msg):
        self.prompt = msg.data
        self.get_logger().info(f"Looking for: {self.prompt}")

    def img_cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        pil_img = PILImage.fromarray(cv_img)
        texts = [[f"a photo of {self.prompt}"]]
        
        inputs = self.processor(text=texts, images=pil_img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process
        target_sizes = torch.Tensor([pil_img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )[0]
        
        # Pick highest confidence
        if len(results["scores"]) > 0:
            idx = results["scores"].argmax()
            box = results["boxes"][idx].tolist() # [xmin, ymin, xmax, ymax]
            
            roi = RegionOfInterest()
            roi.x_offset = int(box[0])
            roi.y_offset = int(box[1])
            roi.width = int(box[2] - box[0])
            roi.height = int(box[3] - box[1])
            self.pub_roi.publish(roi)
```

### 👨‍💻 Integrated Logic (`src/prompt_grasp.py`)

Subscribes to `ROI` and `PointCloud`. Mask the Cloud. Sample Grasps.

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, RegionOfInterest
from geometry_msgs.msg import PoseStamped
import numpy as np
# Assume we import the Day 66 Grasp Sampler
# from grasp_net import NeuralGrasp

class PromptGraspController(Node):
    def __init__(self):
        super().__init__('prompt_grasp')
        self.target_roi = None
        self.sub_roi = self.create_subscription(RegionOfInterest, '/target/roi', self.roi_cb, 10)
        self.sub_pcl = self.create_subscription(PointCloud2, '/camera/depth/points', self.pcl_cb, 10)
        self.pub_grasp = self.create_publisher(PoseStamped, '/mtc/target_grasp', 10)
        
    def roi_cb(self, msg):
        self.target_roi = msg
        
    def pcl_cb(self, msg):
        if self.target_roi is None: return
        
        # 1. Project PointCloud to Image Plane to filter by ROI
        # (Simplified: In reality, we need Camera Intrinsics K)
        # Assume we have a function:
        # points_inside = filter_by_roi(cloud, self.target_roi, K)
        
        # 2. Sample Grasps on filtered points
        # grasps = neural_grasp_model(points_inside)
        
        # 3. Pick Best Grasp closest to ROI Center depth
        # best_grasp = select_best(grasps)
        
        # 4. Publish
        # ...
        pass
```

---

## 🔬 Lab Exercise: "Simon Says"

### 1. Lab Objectives
- Run the system.
- User types: `ros2 topic pub /command/prompt std_msgs/msg/String "data: 'red apple'"`
- **Expectation:** Rviz displays a Bounding Box around the apple. Gripper moves to hover over it.
- **Challenge:** Put a Red Apple and a Green Apple.
- **Input:** "green apple".
- **Result:** Ensure the robot differentiates based on the text embedding.

---

## 🚀 Project: "Grocery Bagger"

**Goal:** Sort items into bags.
1.  **Scene:** Table with Cereal, Soap, Orange, Banana.
2.  **Instruction:** "Put potential food items in the left bag."
3.  **VLM Logic:**
    *   Query: "Is Cereal food?" (LLM) -> Yes. "Is Soap food?" -> No.
    *   Action: Pick Cereal -> Left Bag. Pick Soap -> Right Bag.
4.  **Implementation:** Simple Python script chaining OpenAI API (Decision) + OWL-ViT (Detection) + MTC (Action).

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Hallucinations"
*   **Symptom:** Robot grasps empty table when asked for "Unicorn".
*   **Cause:** OWL-ViT forces a prediction even if low confidence.
*   **Fix:** **Confidence Thresholding**. If `score < 0.2`, reply "I cannot find a Unicorn".

#### 2. "Depth Misalignment"
*   **Symptom:** ROI is correct on image, but 3D point cloud cropping is offset.
*   **Cause:** RGB-to-Depth extrinsic calibration error.
*   **Fix:** Check `depth_registered` topic (Hardware Sync). Re-calibrate if needed.

---

## ⚡ Optimization: FastAM (Fast Anything Model)

Running ViT-Base takes ~200ms.
*   **Optimization:** Use `TensorRT` to compile OWL-ViT.
*   **Quantization:** FP16.
*   **NanoOWL:** Specifically optimized for Jetson Orin. Runs at 30 FPS.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Zero-Shot vs Few-Shot?
    *   **A:** Zero-Shot: Model has never seen the object class during training (Found via Language). Few-Shot: We give 5 example images of the object.
2.  **Q:** Why not just use YOLO?
    *   **A:** YOLO is faster but "Closed Vocabulary". You must retrain it for every new object type.
3.  **Q:** What is "Affordance"?
    *   **A:** Not just "Where is the object?", but "Where can I grab it?". (Handle of the mug, not the hot body).

### Challenge Task
> **Task:** Affordance Detection.
> 1. Prompt: "Pick up the mug by the handle".
> 2. VLM must detect "mug handle" specifically, not just "mug".
> 3. Pass the "handle" ROI to the grasp sampler.

---

## 📚 Further Reading
- **OWL-ViT Paper:** "Simple Open-Vocabulary Object Detection".
- **CLIPort:** "What and Where Pathways".

---

**Day 68 Complete**
