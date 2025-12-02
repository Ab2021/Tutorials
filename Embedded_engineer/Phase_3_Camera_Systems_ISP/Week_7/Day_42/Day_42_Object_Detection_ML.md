# Day 42: Object Detection with ML (Haar Cascades & HOG)
## Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI

---

## 🎯 Learning Objectives
1.  **Understand** the difference between Classical ML (Haar/HOG) and Deep Learning (CNNs).
2.  **Implement** Face Detection using Haar Cascades.
3.  **Implement** Pedestrian Detection using HOG (Histogram of Oriented Gradients) + SVM.
4.  **Train** a custom Haar Cascade classifier (conceptually).
5.  **Optimize** detection speed using Image Pyramids and Sliding Windows.
6.  **Debug** False Positives and False Negatives.

---

## 📚 Prerequisites & Preparation
*   **Hardware:** Camera input.
*   **Software:** OpenCV (contains pre-trained models).
*   **Knowledge:** Integral Images, Gradients.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Haar Cascades (Viola-Jones)
*   **Features:** Simple rectangular features (black/white regions) similar to Haar wavelets.
*   **Integral Image:** Allows calculating the sum of pixels in any rectangle in O(1) time.
*   **AdaBoost:** Selects the best features from thousands of candidates.
*   **Cascade:** A series of "Weak Classifiers".
    *   Stage 1: Checks simple features. If fail, reject immediately (Fast).
    *   Stage 2: Checks harder features.
    *   ...
    *   Stage N: Final confirmation.
*   **Pros:** Extremely fast on CPU.
*   **Cons:** Not robust to rotation/occlusion.

### 🔹 Part 2: HOG (Histogram of Oriented Gradients)
*   **Concept:** Objects (like pedestrians) have a characteristic distribution of edge directions.
*   **Process:**
    1.  Calculate Gradients (Magnitude & Angle).
    2.  Divide image into cells (e.g., 8x8).
    3.  Compute Histogram of Orientations for each cell.
    4.  Normalize blocks (contrast invariance).
    5.  Feed vector to a Linear SVM (Support Vector Machine).
*   **Pros:** Good for upright people.
*   **Cons:** Slower than Haar.

---

## 💻 Implementation Examples

### Example 1: Face Detection (Haar Cascade)

```cpp
/**
 * @brief Face Detection using Haar
 */
void detect_faces(cv::Mat& img) {
    cv::CascadeClassifier face_cascade;
    // Load pre-trained model (XML file included with OpenCV)
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading cascade" << std::endl;
        return;
    }
    
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray); // Improve contrast
    
    std::vector<cv::Rect> faces;
    // ScaleFactor=1.1, MinNeighbors=3
    face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    
    for (const auto& face : faces) {
        cv::rectangle(img, face, cv::Scalar(255, 0, 0), 2);
    }
}
```

### Example 2: Pedestrian Detection (HOG + SVM)

```cpp
/**
 * @brief Pedestrian Detection using HOG
 */
void detect_people(cv::Mat& img) {
    cv::HOGDescriptor hog;
    // Load the default People Detector (trained on INRIA dataset)
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
    std::vector<cv::Rect> found, found_filtered;
    // winStride=(8,8), padding=(32,32), scale=1.05
    hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
    
    // Draw
    for (const auto& r : found) {
        // HOG often returns slightly large boxes, shrink them a bit
        cv::Rect r_shrink = r;
        r_shrink.x += r.width * 0.1;
        r_shrink.width *= 0.8;
        r_shrink.y += r.height * 0.07;
        r_shrink.height *= 0.8;
        
        cv::rectangle(img, r_shrink, cv::Scalar(0, 255, 0), 2);
    }
}
```

### Example 3: Training a Custom Cascade (Concept)

To detect a custom object (e.g., a Logo):
1.  **Collect Positives:** 1000 images containing the logo.
2.  **Collect Negatives:** 2000 images NOT containing the logo.
3.  **Create Samples:** Use `opencv_createsamples`.
4.  **Train:** Use `opencv_traincascade`.
    *   Output: `my_logo_cascade.xml`.
5.  **Use:** Load this XML in Example 1.

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Tuning Parameters

**Objective:** Understand `detectMultiScale` arguments.

**Steps:**
1.  Run Face Detection on a group photo.
2.  **ScaleFactor:** Change from 1.1 to 1.5.
    *   Result: Faster, but misses small faces (pyramid steps too big).
3.  **MinNeighbors:** Change from 3 to 6.
    *   Result: Fewer False Positives, but might miss real faces.
4.  **MinSize:** Set to (100, 100).
    *   Result: Ignores faces smaller than 100px.

### Lab 2: Eye Detection (Nested)

**Objective:** Detect eyes *inside* the face region.

**Steps:**
1.  Detect Face.
2.  Extract Face ROI.
3.  Run `haarcascade_eye.xml` ONLY on the Face ROI.
4.  **Benefit:** Much faster and fewer false positives than searching the whole image for eyes.

### Lab 3: HOG Speed Test

**Objective:** Measure HOG performance.

**Steps:**
1.  Run HOG on 640x480 video. Measure FPS.
2.  Run HOG on 1280x720 video. Measure FPS.
3.  **Observation:** HOG is computationally expensive.
4.  **Optimization:** Resize input to 640x480 before detection.

---

## 🐛 Debugging Techniques

### Debug 1: Too Many False Positives

**Symptom:** Detecting faces in trees/clouds.

**Cause:**
*   `MinNeighbors` too low.
*   Training data didn't include enough "Hard Negatives" (backgrounds that look like faces).
*   **Fix:** Increase `MinNeighbors`.

### Debug 2: Missed Detections (False Negatives)

**Symptom:** Not detecting a clear face.

**Cause:**
*   Face is rotated (Haar is not rotation invariant).
*   Lighting is too dark/harsh.
*   **Fix:** Use Histogram Equalization (`equalizeHist`) before detection. Or switch to CNN-based detectors (Day 43).

---

## ⚡ Performance Optimization

### Optimization 1: LBP Cascades

*   Local Binary Patterns (LBP) are faster to compute than Haar features (integer vs float).
*   Use `lbpcascade_frontalface.xml`.
*   Slightly less accurate, but 2-3x faster.

### Optimization 2: Skip Frames

*   Don't detect on every frame.
*   Frame 1: Detect (Expensive). Get ROIs.
*   Frame 2-5: Track ROIs using Optical Flow (Cheap).
*   Frame 6: Re-detect.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why is the "Integral Image" crucial for Haar Cascades?**
2.  **What is the role of "AdaBoost" in training?**
3.  **Why does HOG work well for pedestrians but poorly for cats?** (Hint: Structure vs Deformability).
4.  **What is a "Sliding Window"?**

### Practical Challenges

1.  **Implement a "Smile Detector":** Use `haarcascade_smile.xml` within the face ROI. Trigger a "Selfie" when a smile is detected.
2.  **Create a "Drowsiness Detector":** Detect eyes. If eyes are closed (not detected) for > 2 seconds, sound an alarm.

---

## 📚 Further Reading & Resources

### Papers
*   **"Rapid Object Detection using a Boosted Cascade of Simple Features"** - Viola & Jones.
*   **"Histograms of Oriented Gradients for Human Detection"** - Dalal & Triggs.

---

## 🎓 Summary

Today we covered:
- ✅ **Classical ML:** Feature engineering + Classifier.
- ✅ **Haar Cascades:** Fast, rigid object detection (Faces).
- ✅ **HOG:** Gradient-based detection (Pedestrians).
- ✅ **Tuning:** Balancing Speed vs Accuracy.
- ✅ **Limitations:** Why we need Deep Learning.

**Next:** Day 43 - Deep Learning for Vision (CNNs & YOLO).

---

**Day 42 Complete** | Phase 3: Camera Systems & ISP | Week 7: Machine Vision & Edge AI
