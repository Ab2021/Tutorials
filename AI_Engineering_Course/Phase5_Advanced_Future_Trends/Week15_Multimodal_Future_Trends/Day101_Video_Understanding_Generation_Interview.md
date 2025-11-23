# Day 101: Video Understanding & Generation
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle the massive memory requirements of Video?

**Answer:**
*   **Keyframe Extraction:** Only process I-frames (keyframes), ignore P/B-frames (deltas).
*   **Token Dropping:** Discard background tokens (sky, wall) that don't change.
*   **Streaming:** Process the video in chunks (Sliding Window).

#### Q2: What is "Zero-Shot Video Classification"?

**Answer:**
Using a model like X-CLIP.
*   Average the embeddings of all frames.
*   Compare against text embeddings ("Playing Tennis", "Cooking").
*   No video-specific training required.

#### Q3: Explain "Flicker" in AI Video.

**Answer:**
Temporal instability.
*   **Cause:** Independent generation of frames.
*   **Fix:** Temporal Attention or Optical Flow constraints (forcing pixels to move smoothly).

#### Q4: How does Sora simulate physics?

**Answer:**
It doesn't have a physics engine. It learns physics as a statistical pattern.
*   "Objects that fall usually stop when they hit the ground."
*   It's an emergent property of scaling.

### Production Challenges

#### Challenge 1: Content Moderation

**Scenario:** User generates deepfake violence.
**Root Cause:** Generative capability.
**Solution:**
*   **C2PA:** Cryptographic watermarking to label AI content.
*   **Classifier:** Run a safety classifier on the generated frames *before* showing them to the user.

#### Challenge 2: Inference Cost

**Scenario:** Generating 1s of video takes 1 minute on an A100.
**Root Cause:** Diffusion steps.
**Solution:**
*   **Distillation:** Consistency Models (LCM) reduce steps from 50 to 4.
*   **Caching:** Reuse background generation.

#### Challenge 3: Long Video Coherence

**Scenario:** In a 1-minute video, the character changes clothes halfway.
**Root Cause:** Forgetting the initial condition.
**Solution:**
*   **Reference Attention:** Keep the first frame's embedding in the context window at all times.

### System Design Scenario: Smart CCTV Search

**Requirement:** "Find all red cars that passed between 2 PM and 4 PM."
**Design:**
1.  **Edge:** Run YOLO (Object Detection) on the camera. Detect "Car".
2.  **Filter:** Crop the car.
3.  **Cloud:** Send crop to CLIP.
4.  **Index:** Store `(Timestamp, CLIP_Embedding)` in Vector DB.
5.  **Query:** Search Vector DB for "Red Car".

### Summary Checklist for Production
*   [ ] **Privacy:** Blur faces in CCTV footage before processing.
*   [ ] **Storage:** Don't store raw video. Store embeddings + metadata.
