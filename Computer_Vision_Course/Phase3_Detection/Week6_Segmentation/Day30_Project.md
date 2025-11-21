# Day 30: Phase 3 Project - Pedestrian Detection & Tracking

## 1. Project Overview
**Goal:** Build a system to Detect and Track pedestrians in a video stream.
**Dataset:** MOT17 (Multiple Object Tracking) or CityPersons.
**Components:**
1.  **Detector:** Faster R-CNN or YOLOv5 (Pretrained on COCO/Person).
2.  **Tracker:** DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric).

## 2. Object Tracking Basics
**Detection vs Tracking:**
*   **Detection:** Finds objects in individual frames. No identity.
*   **Tracking:** Links detections across frames. Assigns unique ID (ID #1, ID #2).

**Tracking-by-Detection Paradigm:**
1.  Run Detector on Frame $t$.
2.  Associate detections with existing Tracks from Frame $t-1$.
3.  Create new Tracks for unmatched detections.
4.  Delete Tracks that disappear.

## 3. SORT (Simple Online and Realtime Tracking)
**Core Components:**
1.  **Kalman Filter:** Predicts the future position of a track in the next frame based on velocity.
2.  **Hungarian Algorithm:** Matches predicted track positions with actual detections based on IoU.

## 4. DeepSORT
**Improvement:** Adds Appearance Information.
*   **Problem with SORT:** Fails if objects are occluded or move fast (IoU overlap is zero).
*   **Solution:** Use a CNN (Re-ID model) to extract an appearance vector for each detection.
*   **Matching Cost:** Combination of Mahalanobis distance (Motion) and Cosine distance (Appearance).

```python
# Pseudo-code for DeepSORT Loop
from deep_sort import DeepSort
from detector import YOLODetector

tracker = DeepSort(max_age=30, n_init=3)
detector = YOLODetector(weights='yolov5s.pt')

while video.is_open():
    frame = video.read()
    
    # 1. Detect
    boxes, confs = detector.detect(frame)
    
    # 2. Extract Features (Re-ID)
    # Done internally by DeepSort or external CNN
    
    # 3. Update Tracker
    tracks = tracker.update_tracks(boxes, confs, frame=frame)
    
    # 4. Draw
    for track in tracks:
        if not track.is_confirmed(): continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        cv2.rectangle(frame, bbox, color=(0, 255, 0))
        cv2.putText(frame, f"ID: {track_id}", bbox[0:2])
```

## Summary
This project combines Object Detection (Phase 3) with Motion Estimation (Kalman Filter) and Feature Matching (Phase 2) to solve a real-world video analysis task.
