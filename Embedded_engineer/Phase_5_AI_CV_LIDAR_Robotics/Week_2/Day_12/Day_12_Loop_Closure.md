# Day 12: Loop Closure & Optimization
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 2: Advanced SLAM & State Estimation

---

> **📝 Content Creator Instructions:**
> A robot that assumes it is "lost" when it returns to the start is useless. Loop Closure is the "Aha! I've been here before" moment.
> - **Focus:** Appearance-based Place Recognition (Bag of Words), Geometric Verification, and Global Relaxation.
> - **Code:** Implementation of DBoW vocabulary creation and query + Pose Graph update.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Construct** a Visual Vocabulary (Bag of Words) from image features.
2.  **Explain** the concept of Perceptual Aliasing (confusing two similar hallways) and how to mitigate it.
3.  **Implement** a Loop Closure Detector using `DBoW3` (or Python equivalent).
4.  **Perform** Geometric Verification using RANSAC PnP to confirm a loop candidate.
5.  **Integrate** loop constraints into the Factor Graph to snap the map consistent.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Camera stream for training vocabulary.

### Software Environment
```bash
pip install opencv-python numpy
# For DBoW3, usually C++ is required, but we will mock it or use `scikit-image`.
```

### Prior Knowledge
- ORB Features.
- Factor Graphs (Day 11).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Bag of Words (BoW)

How do we search "Have I seen this image before?" in a database of 100,000 Keyframes?
We cannot compare every image to every other image ($O(N^2)$).

**The BoW Approach:**
1.  **Vocabulary Tree:** Cluster ORB descriptors from training data (k-means) into a Hierarchical Tree. The leaves are "Words" (Visual Words).
2.  **Forward Index:** Each Image is converted to a Histogram of Words (Bag of Words vector).
    *   Image A = {Word_45: 2, Word_99: 1, ...}
3.  **Inverted Index:** For each Word, store which images strictly contain it.
    *   Word_45 -> {Image_1, Image_10, Image_205}
4.  **Query:** To find similar images, look up words in the Inverted Index and score by Term Frequency-Inverse Document Frequency (TF-IDF).

**TF-IDF:**
*   **TF:** How often word appears in current image.
*   **IDF:** Punish words that appear everywhere (e.g., "Sky" features, "Asphalt" texture) effectively acting as "Stop Words".

### 🔹 Part 2: Geometric Verification

BoW is purely appearance-based. It makes mistakes (Perceptual Aliasing).
*   *Example:* All white corridors look the same.
*   *Verification:* If Image A and Image B are the same place, their feature points must satisfy geometric constraints (Fundamental Matrix or Sim3).
*   **RANSAC:** We try to find a transformation between Current Frame and Loop Candidate. If `inliers > Threshold`, we accept the loop.

### 🔹 Part 3: Pose Graph Relaxation

Once a loop is verified:
1.  Compute relative pose $T_{loop}$ between Current Frame ($X_{curr}$) and Old Frame ($X_{old}$).
2.  Add a `BetweenFactor` to the graph: $Factor(X_{curr}, X_{old}, T_{loop})$.
3.  **Optimize:** The errors distribute along the whole trajectory, bending the "drifted" path into a consistent loop.

---

## 💻 Implementation: Visual Place Recognition

We will build a simple BoW-like system using KMeans.

### 🛠️ Project Structure
```text
day12_loop_closure/
├── data/
│   ├── train_images/
│   └── test_loop.mp4
├── src/
│   ├── vocab_builder.py
│   ├── loop_detector.py
└── train_vocab.py
```

### 👨‍💻 Code Implementation (`src/vocab_builder.py`)

```python
import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans

class VisualVocabulary:
    def __init__(self, k=1000):
        self.k = k
        self.kmeans = None
        self.orb = cv2.ORB_create()
        
    def train(self, image_paths):
        descriptors = []
        print(f"Extracting features from {len(image_paths)} images...")
        
        for path in image_paths:
            img = cv2.imread(path, 0)
            kp, des = self.orb.detectAndCompute(img, None)
            if des is not None:
                descriptors.append(des)
                
        # Stack all descriptors
        X = np.vstack(descriptors)
        print(f"Clustering {X.shape[0]} descriptors into {self.k} words...")
        
        # Train KMeans (Visual Words)
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=42).fit(X)
        print("Vocabulary Trained.")
        
    def get_bow_vector(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        if des is None: return np.zeros(self.k)
        
        # Predict words for each descriptor
        words = self.kmeans.predict(des.astype(float))
        
        # Create Histogram
        hist, _ = np.histogram(words, bins=range(self.k+1))
        
        # Normalize (TF)
        hist = hist.astype(float) / np.sum(hist)
        return hist
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.kmeans, f)
```

### 👨‍💻 Code Implementation (`src/loop_detector.py`)

```python
class LoopDetector:
    def __init__(self, vocab_path):
        with open(vocab_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        self.orb = cv2.ORB_create()
        self.k = self.kmeans.n_clusters
        self.database = [] # List of (ImageID, BowVector)
        
    def add_image(self, img_id, img):
        bow_vec = self._compute_bow(img)
        self.database.append((img_id, bow_vec))
        
    def query(self, img):
        query_vec = self._compute_bow(img)
        scores = []
        
        for img_id, db_vec in self.database:
            # L1 Score (Simple similarity)
            # In production, use Dot Product (Cosine Similarity) or Chi-Square
            score = np.dot(query_vec, db_vec)
            
            # Avoid self-matching (exclude recent 50 frames)
            if abs(img_id - current_id) > 50: 
                 scores.append((img_id, score))
                 
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:5] # Top 5 candidates

    def _compute_bow(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        if des is None: return np.zeros(self.k)
        words = self.kmeans.predict(des.astype(float))
        hist, _ = np.histogram(words, bins=range(self.k+1))
        return hist / np.sum(hist)
```

---

## 🔬 Lab Exercise: Loop Closure in Action

### 1. Lab Objectives
- Train a vocabulary on the "Kitchen" dataset.
- Run the video: robot enters Kitchen -> Hallway -> Living Room -> Kitchen.
- **Goal:** Detect when the robot re-enters the Kitchen.

### 2. Step-by-Step Guide
1.  **Extract Frames:** `ffmpeg -i robot_video.mp4 data/images/%04d.jpg`
2.  **Train:** Run `train_vocab.py` on the first 500 images.
3.  **Run Detector:**
    ```python
    detector = LoopDetector("vocab.pkl")
    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        candidates = detector.query(img)
        detector.add_image(i, img)
        
        if candidates and candidates[0][1] > 0.7:
            print(f"LOOP DETECTED! Frame {i} looks like Frame {candidates[0][0]}")
            # Optional: Show side-by-side
    ```

---

## 🚀 Project: "The Time Machine"

**Goal:** Integrate the Loop Detector with the GTSAM Graph from Day 11.
**Application:**
1.  Run Odometry simulation.
2.  At step 100 (Kitchen), save image.
3.  Robot drives around.
4.  At step 500 (Kitchen again), Loop Detector triggers.
5.  **Action:** Compute Relative Pose ($T_{500, 100}$) using PnP.
6.  **Action:** Add `BetweenFactor(500, 100, T)` to GTSAM.
7.  **Action:** `optimizer.optimize()`.
8.  **Visualize:** The map snaps shut.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Perceptual Aliasing (False Positives)
*   **Symptom:** Loop closure detected between two identical brick walls in different cities.
*   **Result:** Map collapses catastrophically.
*   **Fix:** **Geometric Verification** is mandatory. If RANSAC cannot find sufficient inliers (meaning the 3D structure matches), reject the loop even if BoW score is high.

#### 2. Temporal Consistency
*   **Symptom:** Loop detector flickers (Found, Not Found, Found).
*   **Fix:** Require **Temporal Consistency**. Only accept a loop if $N$ consecutive frames match $N$ consecutive old frames. (e.g., Frame 500 matches 100, 501 matches 101, 502 matches 102).

---

## ⚡ Optimization: Binary Descriptors

ORB uses Binary Descriptors (BRIEF).
*   Use **Hamming Distance** / **Hierarchical Clustering** for the Vocabulary Tree.
*   Much faster than K-Means (Euclidean) on float descriptors (SIFT/SURF).
*   Tool: `DBoW2` / `DBoW3` library (C++).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why not just compare raw pixels for Loop Closure?
    *   **A:** Viewpoint changes, lighting changes, and dynamic objects make pixel difference metrics ($L_2$) useless. We need invariant features.
2.  **Q:** What is the "Vocabulary Tree" structure?
    *   **A:** A k-ary tree (usually k=10, depth=6). It allows finding the "word ID" for a descriptor in logarithmic time $O(L \cdot k)$ instead of linear time.
3.  **Q:** What is False Positive vs False Negative in Loop Closure?
    *   **A:** False Positive (Bad): Closing a loop that isn't there (Map Broken). False Negative (Bad-ish): Missing a loop (Map drifts, but stays consistent locally). **False Positives are much worse.**

### Challenge Task
> **Task:** Implement Geometric Verification.
> 1. When `query()` returns a candidate.
> 2. Use `cv2.findHomography(curr_pts, old_pts, cv2.RANSAC)`.
> 3. If `mask.sum()` (inliers) < 15, reject candidate.

---

## 📚 Further Reading
- **FAB-MAP:** Cummins and Newman (IJRR 2008).
- **DBoW2:** Galvez-Lopez and Tardos (TRO 2012).
- **NetVLAD:** Arandjelovic et al. (CVPR 2016) - Deep Learning approach to Place Recognition (covered in Phase 5 later).

---

**Day 12 Complete**
