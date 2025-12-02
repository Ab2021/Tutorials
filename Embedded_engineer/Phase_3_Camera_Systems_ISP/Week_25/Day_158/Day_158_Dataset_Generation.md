# Day 158: Dataset Generation & Annotation
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Understand** the importance of Data: "Garbage In, Garbage Out".
2.  **Collect** a custom dataset using your camera (Images/Video).
3.  **Annotate** data using tools like CVAT (Computer Vision Annotation Tool) or LabelImg.
4.  **Augment** data to increase diversity (Rotation, Flip, Color Jitter).
5.  **Format** the dataset for YOLO (YOLO format) or COCO (JSON).

---

## 📚 Prerequisites & Preparation
*   **Software:** CVAT (Web-based) or LabelImg (Python), Albumentations library.
*   **Hardware:** Camera.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Data Lifecycle
1.  **Collection:** Gathering raw images. Diversity is key (Lighting, Backgrounds, Angles).
2.  **Cleaning:** Removing blurry, duplicate, or irrelevant images.
3.  **Annotation:** Drawing boxes/polygons/keypoints. The most expensive part.
4.  **Augmentation:** Artificially expanding the dataset.
5.  **Splitting:** Train (70%), Validation (20%), Test (10%). Never leak Test data into Train!

### 🔹 Part 2: Annotation Formats
*   **Pascal VOC (XML):** Old standard. One XML per image.
*   **COCO (JSON):** Industry standard. One huge JSON file for the whole dataset. Supports Boxes, Segments, Keypoints.
*   **YOLO (TXT):** Simple. One TXT per image. Line: `class_id x_center y_center width height` (Normalized 0-1).

### 🔹 Part 3: Active Learning
*   Instead of labeling *everything*, train a model on a small set.
*   Run inference on the unlabeled set.
*   Only label the images where the model is **Unsure** (Low confidence).
*   Retrain. Repeat. Saves 80% of labeling time.

---

## 💻 Implementation Examples

### Example 1: Data Augmentation (Albumentations)

```python
import albumentations as A
import cv2
import matplotlib.pyplot as plt

# 1. Define Pipeline
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 2. Load Image and Label
image = cv2.imread("dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[0.5, 0.5, 0.4, 0.4]] # YOLO format: x_c, y_c, w, h
class_labels = ['dog']

# 3. Augment
augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
aug_img = augmented['image']
aug_bboxes = augmented['bboxes']

# 4. Visualize
cv2.rectangle(aug_img, ... ) # Draw box
plt.imshow(aug_img)
plt.show()
```

### Example 2: Converting COCO to YOLO

```python
import json

with open('instances_train2017.json') as f:
    data = json.load(f)

for img in data['images']:
    img_id = img['id']
    file_name = img['file_name']
    w, h = img['width'], img['height']
    
    # Find annotations for this image
    anns = [a for a in data['annotations'] if a['image_id'] == img_id]
    
    with open(f"labels/{file_name.replace('.jpg', '.txt')}", 'w') as out:
        for a in anns:
            bbox = a['bbox'] # x_top_left, y_top_left, w, h
            
            # Convert to YOLO (x_center, y_center, w, h) normalized
            x_c = (bbox[0] + bbox[2]/2) / w
            y_c = (bbox[1] + bbox[3]/2) / h
            w_n = bbox[2] / w
            h_n = bbox[3] / h
            
            out.write(f"{a['category_id']} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Capture a Custom Dataset

**Objective:** "My Coffee Mug".

**Steps:**
1.  Record a video of your coffee mug on your desk. Move it around. Change lighting.
2.  Extract frames: `ffmpeg -i video.mp4 -r 1 frames/%04d.jpg` (1 fps).
3.  Delete blurry frames. Aim for 50 good images.

### Lab 2: Annotate with LabelImg

**Objective:** Create Ground Truth.

**Steps:**
1.  `pip install labelImg`.
2.  Run `labelImg`.
3.  Open Dir -> `frames/`. Change Save Format to **YOLO**.
4.  Draw box around mug. Label "mug".
5.  Save. Next Image. (Hotkeys: W for box, D for next).
6.  **Result:** 50 `.txt` files.

### Lab 3: Train YOLOv8 on Custom Data

**Objective:** Fine-tuning.

**Steps:**
1.  Create `data.yaml`:
    ```yaml
    train: ../train/images
    val: ../valid/images
    nc: 1
    names: ['mug']
    ```
2.  Run Training:
    ```bash
    yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
    ```
3.  **Result:** `best.pt`. Run inference on your webcam. Does it detect the mug?

---

## 🐛 Debugging Dataset Issues

### Debug 1: "NaN Loss" during Training

**Symptom:** Loss goes to Infinity or NaN.

**Cause:**
*   Corrupt labels.
*   Coordinates > 1.0 (Normalization error).
*   Negative coordinates.
*   **Fix:** Write a script to validate every `.txt` file. Ensure $0 \le x, y, w, h \le 1$.

### Debug 2: Overfitting

**Symptom:** Training Loss decreases, Validation Loss increases.

**Cause:**
*   Dataset too small.
*   Not enough diversity.
*   **Fix:** Use heavy Augmentation (Mosaic, MixUp). Collect more data.

---

## ⚡ Performance Optimization

### Optimization 1: Synthetic Data

*   Use Blender/Unity to generate thousands of images.
*   Randomize lighting, background, and texture.
*   **Benefit:** Perfect labels (Segmentation/Depth) for free.
*   **Sim2Real:** You might need some real images to bridge the "Reality Gap".

### Optimization 2: Auto-Labeling

*   Train a model on 10% of data.
*   Use it to predict labels for the other 90%.
*   Manually correct the mistakes (much faster than drawing from scratch).

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why do we split data into Train/Val/Test?** (Train: Learn. Val: Tune Hyperparameters. Test: Final Evaluation. Prevents cheating).
2.  **What is "Class Imbalance"?** (1000 images of Dogs, 10 images of Cats. Model will ignore Cats).
3.  **What is "Mosaic Augmentation"?** (Stitching 4 images together into one. Helps model detect small objects and context).

### Practical Challenges

1.  **Validate Labels:** Write a script that reads images and YOLO labels, draws the boxes, and saves `debug_images/`. Visually check if boxes match objects.
2.  **Merge Datasets:** Combine your "Mug" dataset with a friend's "Pen" dataset into a single "Office" dataset. Remap Class IDs!

---

## 📚 Further Reading & Resources

### Documentation
*   **Albumentations Documentation.**
*   **CVAT (Computer Vision Annotation Tool).**

---

## 🎓 Summary

Today we covered:
- ✅ **Collection:** Quality over Quantity.
- ✅ **Annotation:** The hard work.
- ✅ **Formats:** YOLO vs COCO.
- ✅ **Augmentation:** Free data.
- ✅ **Training:** Fine-tuning YOLO.

**Next:** Day 159 - Model Quantization & Pruning.

---

**Day 158 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


