# Days 22-28: Week 4 - TensorRT & Inference Optimization Labs
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## Day 22: TensorRT Basics

```python
# Export model to ONNX first
import torch
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True).cuda().eval()
dummy = torch.randn(1, 3, 224, 224, device='cuda')
torch.onnx.export(model, dummy, "resnet18.onnx")
print("Exported to ONNX")
```

---

## Day 23: ONNX to TensorRT

```bash
# Convert using trtexec
trtexec --onnx=resnet18.onnx --saveEngine=resnet18.trt --fp16
```

---

## Day 24: INT8 Quantization

```python
# Collect calibration data
def calibrate(model, dataloader, num_batches=100):
    model.eval()
    for i, (images, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        _ = model(images.cuda())
```

---

## Day 25: Dynamic Shapes

```python
# TensorRT optimization profile
# min=(1,3,224,224), opt=(8,3,224,224), max=(32,3,224,224)
```

---

## Day 26-27: Triton Inference Server

```bash
# Model repository structure
models/
└── resnet18/
    ├── config.pbtxt
    └── 1/
        └── model.plan

# config.pbtxt
name: "resnet18"
platform: "tensorrt_plan"
max_batch_size: 32
```

---

## Day 28: Week 4 Project

```python
# End-to-end optimization pipeline
# 1. Train PyTorch model
# 2. Export to ONNX
# 3. Convert to TensorRT
# 4. Deploy to Triton
# 5. Benchmark with perf_analyzer
```

---

## 📝 Week 4 Summary
| Day | Topic | Tool |
|-----|-------|------|
| 22 | TensorRT Intro | trtexec |
| 23 | ONNX Conversion | onnx, trtexec |
| 24 | INT8 Quantization | Calibration |
| 25 | Dynamic Shapes | Optimization Profiles |
| 26 | Triton Server | tritonserver |
| 27 | Triton Performance | perf_analyzer |
| 28 | Project | Full pipeline |
