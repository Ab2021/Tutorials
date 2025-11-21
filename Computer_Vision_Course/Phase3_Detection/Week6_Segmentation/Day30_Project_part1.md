# Day 30 Deep Dive: Kalman Filter & Re-ID

## 1. Kalman Filter Explained
A recursive algorithm to estimate the state of a dynamic system from noisy measurements.
*   **State:** $[x, y, a, h, v_x, v_y, v_a, v_h]$. (Position + Velocity).
*   **Predict Step:**
    *   Use current state and velocity to predict next position.
    *   $x_{t+1} = x_t + v_x \cdot \Delta t$.
    *   Uncertainty (Covariance) increases.
*   **Update Step:**
    *   Observe actual measurement (Detection).
    *   Correct the prediction based on the difference (Residual).
    *   Uncertainty decreases.
*   **Result:** Smooths out jittery detections and handles missing frames (coasting).

## 2. The Re-ID Model (Appearance Descriptor)
A small CNN trained to distinguish identities.
*   **Input:** Crop of the person ($128 \times 64$).
*   **Output:** 128-dim vector.
*   **Training:** Triplet Loss (Anchor, Positive, Negative).
*   **Goal:** Same person $\to$ High Cosine Similarity. Different person $\to$ Low Similarity.

## 3. Handling Occlusion
What happens when Person A walks behind Person B?
1.  **Detection Lost:** Detector fails to see Person A.
2.  **Kalman Predict:** Tracker predicts Person A continues moving (coasting).
3.  **Age:** Track age increases.
4.  **Re-appearance:** When Person A emerges, the detection is matched to the predicted track using Appearance (DeepSORT) even if they are far from the last seen position.
5.  **Max Age:** If Person A stays hidden too long (>30 frames), the track is deleted.

## 4. Evaluation Metrics (MOT Challenge)
*   **MOTA (Multiple Object Tracking Accuracy):** Combines FP, FN, and ID Switches.
    $$ MOTA = 1 - \frac{\sum (FN + FP + IDSW)}{\sum GT} $$
*   **IDSW (ID Switch):** When the tracker wrongly changes the ID of a person (e.g., ID 1 becomes ID 2).
*   **IDF1:** F1 score for identification.

## Summary
Tracking is about data association. Kalman Filter handles short-term motion, while Re-ID handles long-term occlusion and appearance matching.
