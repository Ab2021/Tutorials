# Day 30 Interview Questions: Tracking

## Q1: What is the difference between Single Object Tracking (SOT) and Multiple Object Tracking (MOT)?
**Answer:**
*   **SOT:** Tracks a single object defined in the first frame (e.g., "Track this specific car"). The object is usually always visible.
*   **MOT:** Tracks all objects of a certain class (e.g., "Track all pedestrians"). Objects enter and leave the scene. Requires detection and data association.

## Q2: Why do we need a Kalman Filter? Why not just match the nearest box?
**Answer:**
*   **Motion Smoothness:** Detections are noisy (jitter). Kalman filter smooths the trajectory.
*   **Velocity:** It estimates velocity, allowing us to predict where the object will be in the next frame.
*   **Occlusion:** If detection is missed for a few frames, Kalman filter keeps the track alive by predicting its position (coasting), allowing re-association when it reappears.

## Q3: What is an "ID Switch"?
**Answer:**
*   A tracking error where the system assigns a new ID to an object that was already being tracked, or swaps IDs between two objects.
*   Example: Person A is ID 1. They cross paths with Person B. After crossing, Person A is labeled ID 2 (or ID of Person B).
*   Low IDSW is crucial for counting applications.

## Q4: Explain the "Hungarian Algorithm" in the context of tracking.
**Answer:**
*   We have $N$ existing tracks and $M$ new detections.
*   We compute a Cost Matrix ($N \times M$) based on IoU distance (or appearance distance).
*   Hungarian Algorithm finds the optimal assignment that minimizes the total cost.
*   It tells us: "Detection 3 belongs to Track 5".

## Q5: How does DeepSORT improve over SORT?
**Answer:**
*   **SORT:** Relies only on IoU (spatial overlap). Fails if objects move fast or are occluded (no overlap).
*   **DeepSORT:** Adds **Appearance Matching**. It extracts a feature vector (Re-ID) from the image crop. Even if the object jumps far away (zero IoU), it can be matched if it looks the same.

## Q6: What is "Tracklet"?
**Answer:**
A fragment of a track.
*   Ideally, one object = one long track.
*   In practice, occlusion breaks the track into multiple short tracklets (ID 1 $\to$ ID 5 $\to$ ID 9).
*   Offline tracking methods try to stitch these tracklets together globally.

## Q7: Why is "Max Age" parameter important?
**Answer:**
*   It determines how long a track is kept alive without new detections.
*   **Too Low:** Tracks die instantly upon occlusion. High fragmentation.
*   **Too High:** Old tracks ("Ghosts") linger around and might wrongly match new objects.

## Q8: Can we use Optical Flow for tracking?
**Answer:**
**Yes.**
*   Optical flow computes pixel-level motion between frames.
*   It is often used in SOT or to refine bounding box positions in MOT.
*   However, it is computationally expensive and drifts over time.

## Q9: What is "Data Association"?
**Answer:**
The problem of determining which measurement (detection) corresponds to which state (track).
*   It is the core problem of MOT.
*   Solved using Gating (filtering impossible matches) and Assignment (Hungarian/Greedy).

## Q10: Implement a simple IoU matching logic.
**Answer:**
```python
def match_detections_to_tracks(detections, tracks, iou_threshold=0.3):
    # detections: list of boxes
    # tracks: list of boxes
    matches = []
    
    for i, det in enumerate(detections):
        best_iou = 0
        best_track_idx = -1
        
        for j, trk in enumerate(tracks):
            iou_val = iou(det, trk)
            if iou_val > best_iou:
                best_iou = iou_val
                best_track_idx = j
                
        if best_iou > iou_threshold:
            matches.append((i, best_track_idx))
            
    return matches
```
