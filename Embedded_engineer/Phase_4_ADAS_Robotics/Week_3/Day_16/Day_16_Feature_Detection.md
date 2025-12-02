# Day 16: Feature Detection & Matching
## Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning

---

> **📝 Day 16 Focus:**
> To track movement or recognize objects, a computer needs to find "interesting" points in an image that are recognizable across different views. These are **Features**. Today, we master the art of detecting corners and blobs, describing them mathematically, and matching them to estimate motion.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** what makes a "good" feature (Invariance to scale, rotation, illumination).
2.  **Compare** classic detectors: Harris Corner, SIFT, SURF, and ORB.
3.  **Implement** ORB (Oriented FAST and Rotated BRIEF) for real-time feature extraction.
4.  **Match** features between frames using Brute Force and FLANN matchers.
5.  **Filter** bad matches using Lowe's Ratio Test and RANSAC.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Image Gradients:** Sobel operators, derivatives.
-   **Histograms:** Used in descriptors.
-   **Python:** OpenCV.

### Hardware Requirements
-   **Development Machine:** Ubuntu 22.04 LTS (or Windows/Mac with Python).

### Software Stack
-   **Python Libraries:** `opencv-python`, `numpy`, `matplotlib`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: What is a Feature?

A feature is a point of interest.
-   **Flat Region:** Bad feature. Moving the window in any direction results in no change.
-   **Edge:** Okay feature. Moving along the edge results in no change (Aperture Problem).
-   **Corner:** Good feature! Moving in *any* direction results in a large change.

#### 1.1 Harris Corner Detector
Mathematically, we look at the Sum of Squared Differences (SSD) of pixel intensities ($I$) when shifting a window ($u, v$).
$$ E(u, v) = \sum_{x,y} w(x,y) [ I(x+u, y+v) - I(x,y) ]^2 $$

Approximated by the **Structure Tensor** ($M$):
$$ M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} $$

Eigenvalues $\lambda_1, \lambda_2$ of $M$ tell us the shape:
-   $\lambda_1 \approx 0, \lambda_2 \approx 0$: Flat.
-   $\lambda_1 \gg 0, \lambda_2 \approx 0$: Edge.
-   $\lambda_1 \gg 0, \lambda_2 \gg 0$: Corner.

---

### 🔹 Part 2: Modern Detectors & Descriptors

Harris finds corners, but it's not scale-invariant (zooming in makes a corner look like an edge).

#### 2.1 SIFT (Scale-Invariant Feature Transform)
-   **Detection:** Difference of Gaussians (DoG) in Scale Space. Finds blobs at different sizes.
-   **Description:** 128-dimensional vector based on gradient histograms.
-   **Pros:** Very robust, scale/rotation invariant.
-   **Cons:** Slow, patented (until recently).

#### 2.2 ORB (Oriented FAST and Rotated BRIEF)
The open-source, real-time alternative.
-   **Detector (FAST):** Compares a pixel to a circle of 16 neighbors. If $N$ contiguous pixels are brighter/darker, it's a corner. Very fast.
-   **Descriptor (BRIEF):** Binary string. Compares intensity of random pixel pairs in the patch ($p_1 < p_2 \to 1$, else $0$).
-   **Rotation:** ORB adds orientation to FAST (using intensity centroid) and steers BRIEF to match.

---

### 🔹 Part 3: Feature Matching

Once we have descriptors from Image A and Image B, we match them.

#### 3.1 Distance Metrics
-   **L2 Norm (Euclidean):** For float descriptors (SIFT, SURF).
-   **Hamming Distance:** For binary descriptors (ORB, BRIEF). XOR operation (count differing bits). *Extremely fast.*

#### 3.2 Matchers
1.  **Brute Force:** Compare every feature in A with every feature in B. $O(N^2)$.
2.  **FLANN (Fast Library for Approximate Nearest Neighbors):** Uses KD-Trees or LSH (Locality Sensitive Hashing) to speed up search.

#### 3.3 Outlier Rejection
1.  **Lowe's Ratio Test:**
    -   Find the 2 nearest neighbors ($m, n$).
    -   If $dist(m) < 0.75 \times dist(n)$, keep it.
    -   *Logic:* A good match should be much closer than the second-best match (ambiguity check).
2.  **RANSAC (Random Sample Consensus):**
    -   Used when fitting a model (e.g., Homography) to matches.
    -   Randomly pick 4 matches -> Compute Model -> Count Inliers.
    -   Repeat and keep best model.

---

## 💻 Implementation: Feature Matching Pipeline

We will build a script `feature_matcher.py` that:
1.  Reads two images (or video frames).
2.  Detects ORB features.
3.  Matches them using Brute Force (Hamming).
4.  Filters using Ratio Test.
5.  Visualizes the matches.

### 🛠️ Setup
Create `week3_day16` and `feature_matcher.py`.

```bash
mkdir -p ~/ros2_ws/src/week3_day16
cd ~/ros2_ws/src/week3_day16
touch feature_matcher.py
```

### 👨‍💻 Code: ORB Matcher

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FeatureMatcher:
    def __init__(self):
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Initialize Matcher (Hamming for Binary Descriptors)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def process_images(self, img1_path, img2_path):
        # Read images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print("Error reading images")
            return

        # 1. Detect and Compute
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        print(f"Found {len(kp1)} features in Image 1")
        print(f"Found {len(kp2)} features in Image 2")

        # 2. Match (k=2 for Ratio Test)
        matches = self.bf.knnMatch(des1, des2, k=2)

        # 3. Filter (Lowe's Ratio Test)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        print(f"Kept {len(good_matches)} good matches")

        # 4. Visualize
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches)
        plt.title(f"ORB Matches (Ratio Test Passed: {len(good_matches)})")
        plt.axis('off')
        plt.show()
        
        return kp1, kp2, good_matches

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        kp_prev, des_prev = self.orb.detectAndCompute(prev_gray, None)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect
            kp_curr, des_curr = self.orb.detectAndCompute(gray, None)
            
            if des_curr is not None and len(des_curr) > 0:
                # Match
                matches = self.bf.knnMatch(des_prev, des_curr, k=2)
                
                good = []
                try:
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good.append(m)
                except ValueError:
                    pass
                
                # Draw
                img_out = cv2.drawMatches(prev_gray, kp_prev, gray, kp_curr, good, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                cv2.imshow('Video Matching', img_out)
                
                # Update previous
                prev_gray = gray
                kp_prev = kp_curr
                des_prev = des_curr
            
            if cv2.waitKey(1) == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    matcher = FeatureMatcher()
    
    # Mode 1: Image Pair (Download sample images first)
    # matcher.process_images('data/left.jpg', 'data/right.jpg')
    
    # Mode 2: Live Video
    matcher.process_video(0)
```

---

## 🔬 Lab Exercise: Homography Estimation

### Lab Objectives
1.  Use the `good_matches` to compute the **Homography Matrix** ($H$) between two images.
    -   $H$ maps points from Image 1 to Image 2.
2.  Use `cv2.findHomography` with `cv2.RANSAC`.
3.  **Task:** Take a photo of a book cover. Take another photo of the book on a desk (rotated/scaled). Use Homography to draw a box around the book in the second image.

### Code Snippet
```python
if len(good_matches) > 10:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Project corners
    h, w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Not enough matches"
**Cause:**
-   Images have little texture (blank walls).
-   Motion blur.
-   Extreme rotation (> 45 degrees) or scale change.
**Solution:** Use SIFT (slower but better) or increase `nfeatures`.

#### 2. False Matches (Outliers)
**Symptom:** Lines crossing each other wildly.
**Cause:** Repetitive patterns (e.g., a fence or windows).
**Solution:** Stricter Ratio Test (0.7 instead of 0.75) or RANSAC.

#### 3. Slow Performance
**Cause:** Too many features (5000+).
**Solution:** Limit `nfeatures` to 500-1000. Use ORB instead of SIFT.

---

## ⚡ Optimization & Best Practices

### 1. Grid-Based Detection
To ensure features are spread across the image (not clustered in one high-contrast area), split the image into a grid (e.g., 4x4) and detect features in each cell independently.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Pre-process images with CLAHE to enhance local contrast. This helps find features in shadows.

### 3. GPU Acceleration
OpenCV has CUDA implementations: `cv2.cuda.ORB_create()`. This is essential for 30fps+ processing on Jetson.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** Why is a corner better than an edge for tracking?
    *   **A:** An edge suffers from the Aperture Problem (cannot determine motion along the edge). A corner has gradients in two directions, locking the position.
2.  **Q:** What is the difference between a Detector and a Descriptor?
    *   **A:** Detector finds *where* the feature is $(u, v)$. Descriptor describes *what* it looks like (vector).
3.  **Q:** Why use Hamming distance for ORB?
    *   **A:** ORB descriptors are binary strings. Hamming distance (XOR) is a CPU instruction, making it orders of magnitude faster than Euclidean distance.

### Challenge Task
**Task:** Panorama Stitching.
1.  Take 2 overlapping photos.
2.  Find matches.
3.  Compute Homography.
4.  Warp Image 1 using `cv2.warpPerspective` to align with Image 2.
5.  Blend them.

---

## 📚 Further Reading & References
-   [OpenCV Feature Matching Tutorial](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
-   [ORB Paper (Rublee et al.)](http://www.willowgarage.com/sites/default/files/orb_final.pdf)

---

**Day 16 Complete** | Phase 4: ADAS & Robotics Systems | Week 3: Computer Vision & Deep Learning
