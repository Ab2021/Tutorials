# Day 101: Video Understanding & Generation
## Core Concepts & Theory

### The Final Frontier

Video is the heaviest modality.
*   **Data:** 1 minute of 4K video = 10GB uncompressed.
*   **Temporal Dimension:** It's not just a stack of images; it's *motion* and *causality*.

### 1. Video Understanding (Video-LLMs)

How to fit a video into an LLM?
*   **Sampling:** Extract 1 frame every second.
*   **Encoding:** Run each frame through a Vision Encoder (CLIP).
*   **Temporal Adapter:** A layer (like Q-Former) that aggregates frame embeddings into a "Video Embedding".
*   **Models:** Video-LLaMA, Video-ChatGPT.

### 2. Video Generation (Sora, Runway)

*   **Diffusion Transformers (DiT):** Combining Diffusion (for detail) with Transformers (for scaling).
*   **Spacetime Patches:** Treating video as a 3D volume of cubes (x, y, t).
*   **Physics Simulation:** Emergent understanding of gravity, collision, and object permanence.

### 3. Applications

*   **Search:** "Find the clip where the car turns left."
*   **Summarization:** "Summarize this 1-hour meeting."
*   **Creation:** "Generate a B-roll of a futuristic city."

### Summary

Video AI is where Text AI was in 2020. Compute intensive, expensive, but rapidly improving.
