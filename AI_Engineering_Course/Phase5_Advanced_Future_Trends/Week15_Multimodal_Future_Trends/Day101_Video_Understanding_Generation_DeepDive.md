# Day 101: Video Understanding & Generation
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Video QA System

We will build a system that answers questions about a video file.

```python
import cv2
from transformers import CLIPProcessor, CLIPModel

class VideoQA:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract_frames(self, video_path, interval=1):
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Take 1 frame every 'interval' seconds
            if cap.get(cv2.CAP_PROP_POS_FRAMES) % int(fps * interval) == 0:
                frames.append(frame)
        cap.release()
        return frames

    def search(self, video_path, query):
        frames = self.extract_frames(video_path)
        inputs = self.processor(text=[query], images=frames, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        # Find frame with highest similarity
        probs = outputs.logits_per_image.softmax(dim=0)
        best_frame_idx = probs.argmax().item()
        return f"Best match at {best_frame_idx} seconds."

# Usage
qa = VideoQA()
print(qa.search("meeting.mp4", "When did John enter the room?"))
```

### Diffusion Transformers (DiT) Architecture

Sora uses DiT.
1.  **VAE:** Compress video into latent space.
2.  **Patchify:** Turn latent video into tokens.
3.  **Transformer:** Standard Transformer backbone processes tokens.
4.  **Diffusion:** Predict noise to denoise the tokens.

### Temporal Consistency

The hardest part of generation.
*   **Problem:** The face changes shape between frame 1 and frame 10.
*   **Solution:** 3D Attention. The model attends to the same spatial location across all time steps.

### Summary

*   **Compute:** Video training requires 100x more compute than Text.
*   **Data:** High-quality video data is scarce (YouTube is noisy).
