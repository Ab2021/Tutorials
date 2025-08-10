# Day 25.2: TorchServe & TorchScript - A Practical Guide

## Introduction: From a `.pth` File to a Production API

Saving a model's `state_dict` is the first step in deployment, but it's not the end of the story. A `.pth` file is still a Python-specific object. To serve a model in a robust, high-performance production environment, we need to move beyond simple Python scripts.

This guide introduces two crucial tools from the PyTorch ecosystem that are designed to bridge the gap from research to production:

1.  **TorchScript:** A way to create a serialized, statically-typed, and graph-based representation of your PyTorch model. A TorchScript model can be run in environments where there is no Python dependency, such as a C++ application.

2.  **TorchServe:** An official, high-performance tool from PyTorch for serving models as a production-ready REST API. It's designed to be easy to use, scalable, and flexible.

**Today's Learning Objectives:**

1.  **Understand the Need for TorchScript:** Grasp why a standard "eager mode" PyTorch model is not ideal for all deployment scenarios.
2.  **Learn the Two Ways to Create a TorchScript Model:** Understand and implement **tracing** and **scripting**.
3.  **Explore the Basics of TorchServe:** Learn about the key components of TorchServe: model archives (`.mar`), model handlers, and the inference API.
4.  **Package a Model for TorchServe:** Walk through the steps of creating a model archive file.
5.  **Start TorchServe and Make Predictions:** Learn the commands to start the server and request predictions from your deployed model.

---

## Part 1: TorchScript - Making PyTorch Models Portable

Standard PyTorch models run in **eager mode**. This is dynamic and flexible, making it great for research and development. However, this Python dependency can be a problem for production environments that might be written in C++, Java, or run on mobile devices.

**TorchScript** is an intermediate representation of a PyTorch model that can be run independently of Python. It converts the dynamic model into a static graph representation that can be optimized and executed in a high-performance C++ runtime.

### 1.1. Method 1: Tracing (`torch.jit.trace`)

*   **How it works:** You provide a sample input tensor to your model. PyTorch executes the model once, **traces** the operations that are performed on the tensor as it flows through the model, and records these operations as a static graph.
*   **Pros:** Very simple to use.
*   **Cons:** It **cannot** capture any control flow that depends on the data (e.g., `if` statements or `for` loops). It only records the path taken by the single trace input.

```python
import torch
import torch.nn as nn

print("--- Part 1: TorchScript ---")

# --- A simple model ---
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        # This model has no control flow, so it's safe for tracing.
        return self.linear(x)

model = MyModel()
model.eval()

# --- Create the traced model ---
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

print("--- 1.1 Tracing ---")
print("Original model successfully traced.")

# You can inspect the graph code
# print(traced_model.code)

# Save the traced model
traced_model.save("traced_model.pt")
```

### 1.2. Method 2: Scripting (`torch.jit.script`)

*   **How it works:** This method directly analyzes your Python source code and compiles it into the TorchScript graph representation. It understands a subset of the Python language.
*   **Pros:** It **can** correctly capture data-dependent control flow, making it more robust than tracing.
*   **Cons:** It can be more complex and may require you to refactor your code slightly to be compatible with the TorchScript compiler.

```python
# --- A model with control flow ---
class ModelWithControlFlow(nn.Module):
    def __init__(self):
        super(ModelWithControlFlow, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        # Tracing would fail to capture this if statement correctly.
        if x.mean() > 0:
            x = torch.relu(x)
        return self.linear2(x)

model_cf = ModelWithControlFlow()
model_cf.eval()

# --- Create the scripted model ---
scripted_model = torch.jit.script(model_cf)

print("\n--- 1.2 Scripting ---")
print("Model with control flow successfully scripted.")

# Inspect the code
# print(scripted_model.code)

# Save the scripted model
scripted_model.save("scripted_model.pt")

# --- Loading a TorchScript model ---
loaded_script = torch.jit.load("scripted_model.pt")
print("\nScripted model loaded successfully.")
```

**Best Practice:** Use `torch.jit.script` unless you have a very simple model with no control flow, in which case `torch.jit.trace` is fine.

---

## Part 2: TorchServe - Production-Ready Model Serving

TorchServe is a tool designed specifically to serve PyTorch models in production. It provides a high-performance, multi-threaded server with a REST API for inference, explanation, and management.

**The Workflow:**
1.  **Write a Handler:** Create a Python script that defines how to preprocess the incoming data, how to run inference, and how to post-process the model's output.
2.  **Archive the Model:** Use the `torch-model-archiver` command-line tool to package your TorchScript model (`.pt`) and your handler script into a single model archive (`.mar`) file.
3.  **Start TorchServe:** Run the `torchserve` command, pointing it to your model store directory.
4.  **Make Predictions:** Send requests to the server's REST API endpoint using tools like `curl` or the Python `requests` library.

### 2.1. Step 1: The Handler Script

Let's create a simple handler for an MNIST classifier.

```python
# Save this code as 'mnist_handler.py'

from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import torch
from PIL import Image
import io
import base64

class MNISTHandler(BaseHandler):
    """Custom handler for MNIST image classification."""
    
    # Define the image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def preprocess(self, data):
        """Converts the incoming request data into a tensor."""
        images = []
        for row in data:
            # The input data is a dictionary. We look for the 'body' key.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # If the image is a string, decode it from base64
                image = base64.b64decode(image)
            
            # Convert bytes to a PIL Image
            image = Image.open(io.BytesIO(image))
            image = self.transform(image)
            images.append(image)
            
        return torch.stack(images).to(self.device)

    def postprocess(self, inference_output):
        """Converts the model's output tensor into a human-readable format."""
        # Get the class with the highest probability
        pred = torch.argmax(inference_output, dim=1)
        return [{"digit": p.item()} for p in pred]

```

### 2.2. Step 2: Archiving the Model

First, we need a trained model. Let's create and save a simple scripted MNIST model.

```python
# --- Create and save a dummy MNIST model ---
class MNISTClassifier(nn.Module): 
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x.view(-1, 28*28))

mnist_model = MNISTClassifier()
# In a real scenario, this model would be trained.
scripted_mnist_model = torch.jit.script(mnist_model)
scripted_mnist_model.save("mnist_model.pt")
```

Now, from your **terminal**, you would run the `torch-model-archiver` command.

```bash
# This command is run in your terminal, not in Python.

# Create a directory to store the models
# mkdir model_store

# torch-model-archiver \
#   --model-name mnist \
#   --version 1.0 \
#   --model-file mnist_model.pt \
#   --serialized-file mnist_model.pt \
#   --handler mnist_handler.py \
#   --export-path model_store
```

This creates a file named `mnist.mar` inside the `model_store` directory.

### 2.3. Step 3 & 4: Serving and Inference

**To start the server**, you run this command in your terminal:

```bash
# torchserve --start --ncs --model-store model_store --models mnist=mnist.mar
```

This starts the server and loads the `mnist.mar` model under the endpoint name `mnist`.

**To make a prediction**, you can use `curl` from another terminal:

```bash
# curl http://127.0.0.1:8080/predictions/mnist -T path/to/your/test_image.png
```

The server will return a JSON response like: `{"digit": 7}`.

## Conclusion

Moving a model from a research environment to a production environment requires a different set of tools. TorchScript provides the means to create portable, high-performance versions of your models, free from Python dependencies. TorchServe provides a robust, scalable, and industry-standard solution for serving those models over a network.

**Key Takeaways:**

1.  **TorchScript for Portability:** Use `torch.jit.script` to convert your dynamic PyTorch model into a static graph representation (`.pt` file) that can be run in non-Python environments.
2.  **TorchServe for Production:** It is the official, recommended tool for deploying PyTorch models as a REST API.
3.  **The `.mar` file is Key:** The model archiver packages your model weights, architecture, and custom handling code into a single, deployable artifact.
4.  **Handlers are the Glue:** A custom handler script defines the specific pre-processing and post-processing logic required to translate raw API request data into tensors for your model and to translate your model's output back into a user-friendly format.

By mastering this workflow, you can take your trained models and turn them into real, scalable applications.

## Self-Assessment Questions

1.  **TorchScript:** What is the main advantage of a TorchScript model over a standard eager-mode PyTorch model?
2.  **Tracing vs. Scripting:** If your model contains an `if` statement that changes its behavior based on the input data, which method (`trace` or `script`) should you use to convert it to TorchScript?
3.  **TorchServe:** What is the file extension for a TorchServe model archive?
4.  **Handler:** What are the three main methods you might implement in a custom TorchServe handler, and what does each one do?
5.  **API Call:** If you start TorchServe with the command `torchserve --models my_classifier=classifier.mar`, what would be the URL for the prediction endpoint?
