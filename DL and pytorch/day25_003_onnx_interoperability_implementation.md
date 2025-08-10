# Day 25.3: ONNX & Interoperability - A Practical Guide

## Introduction: A Common Language for AI Models

PyTorch is a fantastic framework, but it's not the only one. The AI ecosystem is filled with different deep learning frameworks (TensorFlow, MXNet, Caffe2), specialized inference engines (TensorRT, OpenVINO), and hardware accelerators (CPUs, GPUs, TPUs, mobile NPUs). How can you ensure that a model you train in PyTorch can be run efficiently in all these different environments?

The answer is the **Open Neural Network Exchange (ONNX)**. ONNX is an open-source, intermediate representation format for machine learning models. It acts as a universal translator. You can train your model in PyTorch, export it to the ONNX format, and then deploy that single ONNX file to a wide variety of target platforms and inference engines.

This guide provides a practical introduction to exporting PyTorch models to ONNX and running them with the ONNX Runtime.

**Today's Learning Objectives:**

1.  **Understand the Need for a Standard Model Format:** Grasp why interoperability is a major challenge in production ML.
2.  **Learn what ONNX is:** Understand its role as a common intermediate representation for both model architecture and weights.
3.  **Export a PyTorch Model to ONNX:** Use the built-in `torch.onnx.export` function to convert a model.
4.  **Run an ONNX Model with ONNX Runtime:** Use the `onnxruntime` library to load an ONNX file and perform inference in a Python environment.
5.  **Visualize an ONNX Graph:** See how tools like Netron can be used to inspect the architecture of a saved ONNX model.

---

## Part 1: Why ONNX?

Without a standard format, deploying models is a nightmare. You would need to:
*   Re-implement your model in a different language (e.g., C++) for your production server.
*   Use a framework-specific tool to convert the model for a mobile device.
*   Use another specific tool to optimize it for an Intel CPU.

This is brittle and time-consuming. ONNX solves this by providing a single format that a growing number of platforms and tools understand.

**The ONNX Ecosystem:**
*   **Export:** Frameworks like PyTorch, TensorFlow, and Keras can export their models to ONNX.
*   **Optimize:** Tools can take an ONNX model and apply graph optimizations, quantization, and hardware-specific tuning.
*   **Deploy:** Runtimes like ONNX Runtime, TensorRT, and OpenVINO can execute the optimized ONNX model with high performance on various hardware targets.

![ONNX Ecosystem](https://i.imgur.com/3h4Y5fG.png)

---

## Part 2: Exporting a PyTorch Model to ONNX

PyTorch has excellent, built-in support for ONNX export via the `torch.onnx.export()` function. The export process works by **tracing** the model, similar to `torch.jit.trace`. You provide a dummy input, and PyTorch executes the model, recording the graph of operations into the ONNX format.

### 2.1. A Simple Export Example

Let's export a pre-trained `torchvision` model.

```python
import torch
import torchvision.models as models

print("--- Part 2: Exporting to ONNX ---")

# --- 1. Load a Pre-trained Model ---
# We use MobileNetV2 as it's a common, efficient model.
model = models.mobilenet_v2(weights='DEFAULT')
model.eval() # Crucial: set to evaluation mode

# --- 2. Create a Dummy Input Tensor ---
# The dummy input must have the correct shape and type for the model.
# For MobileNetV2, it's a batch of 3-channel 224x224 images.
batch_size = 1
dummy_input = torch.randn(batch_size, 3, 224, 224)

# --- 3. Define the Export Path ---
onnx_model_path = "mobilenet_v2.onnx"

# --- 4. Export the Model ---
print(f"Exporting model to {onnx_model_path}...")
torch.onnx.export(
    model,                     # The model to export
    dummy_input,               # A sample input to trace the graph
    onnx_model_path,           # Where to save the model
    export_params=True,        # Store the trained weights in the model file
    opset_version=11,          # The ONNX version to use
    do_constant_folding=True,  # Execute constant folding for optimization
    input_names=['input'],     # The model's input names
    output_names=['output'],   # The model's output names
    dynamic_axes={'input' : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}}
)

print("Model successfully exported to ONNX format.")
```

**Key Parameters for `torch.onnx.export`:**
*   `model`: The `nn.Module` you are exporting.
*   `dummy_input`: A tensor with the correct shape and type. The values don't matter, but the shape does.
*   `f`: The path to save the file.
*   `input_names` / `output_names`: Names to assign to the input and output nodes in the graph.
*   `dynamic_axes`: This is a very important parameter. It specifies which dimensions of your input/output can have a variable size. In the example above, we mark the batch dimension (`axis 0`) as dynamic, so the exported model can handle batches of any size (e.g., 1, 4, 8, etc.).

---

## Part 3: Running Inference with ONNX Runtime

Now that we have our `.onnx` file, we can no longer use PyTorch to run it. We need a dedicated ONNX inference engine. The official, cross-platform engine is **ONNX Runtime**.

**Installation:** `pip install onnx onnxruntime`

### 3.1. Performing Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from torchvision import transforms

print("\n--- Part 3: Running Inference with ONNX Runtime ---")

# --- 1. Create an ONNX Runtime Inference Session ---
session = ort.InferenceSession(onnx_model_path)

# --- 2. Prepare an Input Image ---
# (Using the same process as for a standard PyTorch model)
url = 'https://www.si.edu/sites/default/files/blog/files/2018/11/persian-cat-551554_1920.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image).unsqueeze(0) # Add batch dimension

# --- 3. Run Inference ---
# The input to the session must be a dictionary where keys are the `input_names`
# we defined during export. The values must be NumPy arrays.
input_name = session.get_inputs()[0].name
ort_inputs = {input_name: input_tensor.cpu().numpy()}

# `session.run` returns a list of outputs (as NumPy arrays)
ort_outputs = session.run(None, ort_inputs)

# --- 4. Process the Output ---
# The output is the logits for the 1000 ImageNet classes
output_logits = ort_outputs[0]
predicted_idx = output_logits.argmax()

print(f"Inference complete.")
print(f"Predicted class index: {predicted_idx}")
# In a real app, you would map this index back to a class name.
```

---

## Part 4: Visualizing the ONNX Graph

An `.onnx` file is a standardized graph definition. We can use tools to visualize this graph, which is excellent for debugging and understanding the model's architecture.

A popular, easy-to-use tool is **Netron**.

1.  **Install Netron:** It's available as a desktop app or can be run from a browser.
2.  **Open the `.onnx` file:** Simply open the `mobilenet_v2.onnx` file we created.
3.  **Explore:** Netron will display a clean, interactive diagram of the entire model. You can click on each layer to inspect its properties, weights, and input/output shapes.

![Netron Visualization](https://i.imgur.com/5h6Y7fG.png)

## Conclusion

ONNX is the glue that connects the diverse world of AI frameworks, hardware, and deployment platforms. By providing a common language for representing models, it enables **interoperability**, allowing you to train a model in your favorite framework (like PyTorch) and confidently deploy it to a completely different environment (like a C++ server or an Android app).

**Key Takeaways:**

1.  **ONNX for Portability:** Exporting to ONNX is the standard way to make your PyTorch models portable and ready for a wide range of production environments.
2.  **Export via Tracing:** The `torch.onnx.export` function works by tracing your model with a dummy input. Be sure to define `dynamic_axes` to create a flexible model.
3.  **Use ONNX Runtime for Inference:** The `onnxruntime` library is the official, high-performance engine for executing `.onnx` models in Python and many other languages.
4.  **Decoupling Training and Deployment:** ONNX allows you to completely separate your training stack from your inference stack. Your production server does not need PyTorch installed at all; it only needs `onnxruntime`.

Learning to export and use ONNX models is an essential skill for any machine learning engineer looking to put their models into production.

## Self-Assessment Questions

1.  **Interoperability:** In one sentence, what is the main purpose of the ONNX format?
2.  **Export Process:** The `torch.onnx.export` function uses a method similar to which `torch.jit` function: `trace` or `script`?
3.  **`dynamic_axes`:** What is the purpose of the `dynamic_axes` argument during export?
4.  **ONNX Runtime:** What is the data type of the input you must provide to an `onnxruntime` inference session?
5.  **Use Case:** You have trained a model in PyTorch, but your company's mobile development team needs to run it inside an iOS app written in Swift. What would be your recommended workflow?

