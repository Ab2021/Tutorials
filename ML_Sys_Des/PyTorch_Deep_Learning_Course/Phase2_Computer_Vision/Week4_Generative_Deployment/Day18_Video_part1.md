# Day 18: Video Understanding - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: Optical Flow, SlowFast, and Tubelets

## 1. Optical Flow

The pattern of apparent motion of objects.
Vector field $(u, v)$ for each pixel.
*   **Hand-crafted**: Lucas-Kanade.
*   **Deep Learning**: FlowNet, RAFT.
*   **Usage**: Explicitly feeds motion info to the network. Crucial for "Two-Stream" networks (Stream 1: RGB, Stream 2: Flow).

## 2. SlowFast Networks (FAIR)

Inspired by biological vision (Parvocellular vs Magnocellular pathways).
Two parallel pathways:
1.  **Slow Pathway**:
    *   Low Frame Rate (Sample every 16th frame).
    *   High Channel Capacity.
    *   Focus: **Spatial Semantics** (Colors, Objects).
2.  **Fast Pathway**:
    *   High Frame Rate (Sample every 2nd frame).
    *   Low Channel Capacity ($\beta=1/8$).
    *   Focus: **Temporal Motion** (Movement).
3.  **Lateral Connections**: Fuse Fast features into Slow pathway.

State-of-the-art for Action Recognition.

## 3. Tubelet Embedding (ViViT)

In Image ViT, we embed 2D patches ($16 \times 16$).
In Video ViT, we embed 3D **Tubelets** ($2 \times 16 \times 16$).
*   Captures local spatiotemporal info immediately.
*   Reduces token count compared to 2D patches per frame.

## 4. Temporal Shift Module (TSM)

How to get 3D performance with 2D cost?
**Shift**: Move 1/8th of channels from current frame to next frame, and 1/8th to previous frame.
*   Zero parameters. Zero FLOPs.
*   Allows 2D Conv to see information from neighbors.
*   Can turn any ResNet into a Video ResNet.

## 5. Action Localization

Classification: "This video contains Cricket".
Localization: "Cricket happens from 00:05 to 00:10".
**Temporal Action Localization (TAL)**:
*   Similar to Object Detection (1D instead of 2D).
*   Predict Start/End times instead of Bounding Boxes.
