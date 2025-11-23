# Lab 5: Image Gen Control (ControlNet)

## Objective
Text-to-Image is random. **ControlNet** gives structure.
Edge detection -> Image Gen.

## 1. The Logic (`control.py`)

```python
# Mock ControlNet Pipeline
def detect_edges(image):
    print("Detecting Canny edges...")
    return "edge_map"

def generate(prompt, control_image):
    print(f"Generating '{prompt}' using control image...")
    return "generated_image"

# Usage
input_image = "sketch.png"
edges = detect_edges(input_image)
output = generate("A realistic photo of a cat", edges)
print("Done.")
```

## 2. Challenge
Use `diffusers` library with `ControlNetModel` to actually run this.

## 3. Submission
Submit the generated image based on a Canny edge map of a building.
