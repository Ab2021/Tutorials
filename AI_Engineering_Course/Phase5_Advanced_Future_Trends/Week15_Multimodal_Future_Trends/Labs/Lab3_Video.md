# Lab 3: Video Captioning

## Objective
Understand video content.
Extract frames -> VLM (Vision Language Model).

## 1. The Pipeline (`video_cap.py`)

```python
# Mock VLM
def describe_image(frame_id):
    return f"A person walking in frame {frame_id}"

# 1. Extract Frames (Mock)
frames = [1, 10, 20] # Frame indices

# 2. Caption
captions = []
for f in frames:
    cap = describe_image(f)
    captions.append(cap)

# 3. Summarize
summary = " ".join(captions)
print(f"Video Summary: {summary}")
```

## 2. Challenge
Use `opencv` to actually read a video file and `Salesforce/blip-image-captioning-base` to caption frames.

## 3. Submission
Submit the summary of a 10-second video clip.
