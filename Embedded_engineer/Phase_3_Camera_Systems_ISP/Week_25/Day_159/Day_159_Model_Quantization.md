# Day 159: Model Quantization & Pruning
## Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems

---

## 🎯 Learning Objectives
1.  **Understand** Model Compression techniques: Quantization, Pruning, Distillation.
2.  **Perform** Post-Training Quantization (PTQ) to convert FP32 models to INT8.
3.  **Implement** Quantization Aware Training (QAT) for minimal accuracy loss.
4.  **Prune** a network (Structured vs Unstructured) to remove redundant weights.
5.  **Evaluate** the trade-off: Accuracy vs Model Size vs Latency.

---

## 📚 Prerequisites & Preparation
*   **Software:** PyTorch, TensorFlow Lite, NVIDIA TensorRT.
*   **Model:** The YOLOv8n model trained in Day 158.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Quantization (FP32 -> INT8)
*   **Weights:** 32-bit floats take 4 bytes. 8-bit integers take 1 byte. **4x Size Reduction.**
*   **Compute:** Integer math is faster and uses less energy than Floating point math.
*   **Mapping:** $r = S(q - Z)$.
    *   $r$: Real value (float).
    *   $q$: Quantized value (int8).
    *   $S$: Scale factor.
    *   $Z$: Zero point.
*   **Calibration:** We need to run a few images through the model to determine the range ($min, max$) of activations to calculate $S$ and $Z$.

### 🔹 Part 2: Pruning
*   **Unstructured Pruning:** Set individual weights to 0. Result is a Sparse Matrix. Hard to accelerate on standard hardware (needs specialized Sparse Accelerators).
*   **Structured Pruning:** Remove entire filters (channels) or layers. Result is a smaller Dense Matrix. Easy to accelerate.
*   **Iterative Pruning:** Prune 10% -> Retrain -> Prune 10% -> Retrain...

### 🔹 Part 3: Knowledge Distillation
*   **Teacher:** Large model (ResNet-101). High accuracy.
*   **Student:** Small model (MobileNet). Fast.
*   **Loss:** Student learns to mimic the Teacher's logits (Soft targets) in addition to the Ground Truth (Hard targets).

---

## 💻 Implementation Examples

### Example 1: PyTorch Post-Training Quantization (Static)

```python
import torch
import torchvision

# 1. Load Model
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 2. Fuse Layers (Conv + BN + ReLU)
# Fusion improves accuracy and speed
model.fuse_model() 

# 3. Prepare
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# 4. Calibrate (Run inference on sample data)
# (Assuming data_loader is defined)
# for img, _ in data_loader:
#     model(img)

# 5. Convert
torch.quantization.convert(model, inplace=True)

print("Model converted to INT8")
```

### Example 2: TensorRT INT8 Calibration (Python)

```python
import tensorrt as trt

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file="calib.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.cache_file = cache_file
        self.d_input = cuda.mem_alloc(input_size)
        self.batches = load_batches() # Generator

    def get_batch(self, names):
        try:
            batch = next(self.batches)
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        # Read from file
        return open(self.cache_file, "rb").read() if os.path.exists(self.cache_file) else None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Build Engine with INT8
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Calibrator(stream)
```

### Example 3: Pruning (PyTorch)

```python
import torch.nn.utils.prune as prune

# Prune 30% of connections in a specific layer
module = model.conv1
prune.l1_unstructured(module, name="weight", amount=0.3)

# Make it permanent
prune.remove(module, 'weight')
```

---

## 🔬 Hands-On Lab Exercises

### Lab 1: Size Comparison

**Objective:** Measure the shrink.

**Steps:**
1.  Save `model_fp32.pth`. Check size (e.g., 45MB).
2.  Quantize to INT8. Save `model_int8.pth`. Check size (e.g., 12MB).
3.  **Result:** ~4x reduction.

### Lab 2: Accuracy Drop

**Objective:** Is it worth it?

**Steps:**
1.  Evaluate FP32 model on Validation Set. (e.g., mAP = 0.75).
2.  Evaluate INT8 model on Validation Set. (e.g., mAP = 0.74).
3.  **Conclusion:** 1% drop for 4x speedup is usually acceptable.

### Lab 3: Latency Benchmarking

**Objective:** Speed test.

**Steps:**
1.  Run FP32 model 100 times. Avg time: 50ms.
2.  Run INT8 model 100 times. Avg time: 15ms.
3.  **Speedup:** 3.3x.

---

## 🐛 Debugging Quantization

### Debug 1: Massive Accuracy Drop

**Symptom:** mAP drops from 0.75 to 0.10.

**Cause:**
*   **Calibration Data Mismatch:** Did you calibrate on "Cats" but test on "Cars"?
*   **Preprocessing Mismatch:** Did you normalize 0-1 during calibration but 0-255 during inference?
*   **Sensitive Layers:** First and Last layers are sensitive. Keep them in FP16/FP32.
*   **Fix:** Use Partial Quantization (Mixed Precision).

### Debug 2: "Zero" Output

**Symptom:** Model predicts all zeros.

**Cause:**
*   ReLU6 vs ReLU. Some quantization schemes require bounded activations (ReLU6).
*   **Fix:** Replace ReLU with ReLU6 before training/quantizing.

---

## ⚡ Performance Optimization

### Optimization 1: QAT (Quantization Aware Training)

*   Simulate quantization noise *during* training.
*   The model learns to be robust to the rounding errors.
*   Result: INT8 accuracy matches FP32 accuracy (sometimes even beats it!).

### Optimization 2: Channel-Last Memory Format

*   CPUs/GPUs prefer `NHWC` (Channel-Last) over `NCHW` (Channel-First) for INT8 dot products.
*   Convert memory format before inference.

---

## 📝 Assessment Questions

### Conceptual Questions

1.  **Why does Pruning require Retraining?** (Removing weights damages the network. Retraining allows remaining weights to compensate).
2.  **What is the difference between "Dynamic" and "Static" Quantization?** (Dynamic calculates scale factors at runtime. Static calculates them offline during calibration. Static is faster).
3.  **What is "Sparsity"?** (The percentage of zero-valued elements in a matrix).

### Practical Challenges

1.  **Deploy to Mobile:** Convert your quantized model to TFLite and run it on an Android phone using the TFLite Benchmark App.
2.  **Visual Pruning:** Visualize the filters of a CNN. Prune the ones that look like "Noise" or are all near zero.

---

## 📚 Further Reading & Resources

### Documentation
*   **PyTorch Quantization Guide.**
*   **"Deep Compression" Paper by Han et al.**

---

## 🎓 Summary

Today we covered:
- ✅ **Quantization:** 32-bit -> 8-bit.
- ✅ **Calibration:** Finding the range.
- ✅ **Pruning:** Cutting the fat.
- ✅ **Distillation:** Teacher-Student.
- ✅ **Trade-offs:** Speed vs Accuracy.

**Next:** Day 160 - Week 25 Review & Project (Smart Camera).

---

**Day 159 Complete** | Phase 3: Camera Systems & ISP | Week 25: Machine Learning for Camera Systems


