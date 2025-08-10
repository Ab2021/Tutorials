# Day 25.1: Model Serialization & Serving - A Practical Guide

## Introduction: From Training to Production

We have spent a great deal of time learning how to build and train powerful deep learning models. But a trained model is only useful if you can put it to work. The process of taking a trained model and making it available for others to use is called **deployment**. The very first step in any deployment workflow is **serialization**: saving your model to a file.

Once a model is saved, it can be loaded into a separate application—a web server, a mobile app, an embedded device—to perform **inference** (i.e., make predictions on new, live data). This process is known as **model serving**.

This guide provides a practical, hands-on walkthrough of the standard methods for saving and loading PyTorch models, which is the foundational skill for model deployment.

**Today's Learning Objectives:**

1.  **Understand the Difference Between Saving an Entire Model vs. Just the `state_dict`:** Learn the pros and cons of each approach and why saving the `state_dict` is the recommended best practice.
2.  **Implement Model Saving and Loading:** Write the code to save a model's learned parameters to a file and then load them back into a new model instance for inference.
3.  **Learn to Save a Training Checkpoint:** See how to save not just the model, but also the optimizer state and other information needed to resume training later.
4.  **Understand the Importance of `model.eval()`:** Revisit this crucial method and understand why it's essential to switch a model to evaluation mode before performing inference.

---

## Part 1: Saving and Loading Models in PyTorch

PyTorch provides a simple and powerful way to save models using `torch.save()` and load them with `torch.load()`. The key question is *what* you save.

### 1.1. The `state_dict`

A PyTorch model (`nn.Module`) has a property called `.state_dict()`. This is a simple Python dictionary that maps each layer to its learnable parameters (weights and biases).

*   It only contains the **parameters**, not the model's architecture.
*   It is small, portable, and flexible.
*   **This is the recommended method for saving and loading models.**

Let's inspect the `state_dict` of a simple model.

```python
import torch
import torch.nn as nn

print("--- Part 1: The state_dict ---")

# --- Define a simple model ---
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.layer2(self.layer1(x))

model = SimpleModel()

# --- Print the state_dict ---
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")
```

### 1.2. Method 1 (Recommended): Saving and Loading the `state_dict`

**Saving:**

```python
# --- Save the state_dict ---
# Define a path
PATH = "simple_model_state_dict.pth"

# Save the dictionary
torch.save(model.state_dict(), PATH)

print(f"\nModel state_dict saved to {PATH}")
```

**Loading:**
To load the `state_dict`, you must first **create an instance of the model class**. The `state_dict` is then loaded into this new instance.

```python
# --- Load the state_dict ---
# 1. Create a new instance of the model
loaded_model = SimpleModel()

# 2. Load the saved state_dict into the model
loaded_model.load_state_dict(torch.load(PATH))

# 3. Set the model to evaluation mode
# This is a crucial step!
loaded_model.eval()

print("\nModel loaded successfully from state_dict.")

# --- Verify that the weights are the same ---
# We can check if the weights of a layer are identical
print(f"Weights are the same: {torch.equal(model.layer1.weight, loaded_model.layer1.weight)}")
```

### 1.3. Method 2 (Not Recommended): Saving the Entire Model

This method saves the entire model object using Python's `pickle` module. 

**The Problem:** The serialized data is bound to the specific classes and the exact directory structure used when the model was saved. This can make the code brittle. Your project might not work when you try to load the model in a different project or after a refactor.

**Saving:**

```python
# --- Save the entire model ---
PATH_FULL = "simple_model_full.pth"
torch.save(model, PATH_FULL)
print(f"\nFull model saved to {PATH_FULL}")
```

**Loading:**

```python
# --- Load the entire model ---
# You don't need to instantiate the class first
loaded_full_model = torch.load(PATH_FULL)
loaded_full_model.eval()

print("\nFull model loaded successfully.")
```

---

## Part 2: The Importance of `model.eval()`

Why is calling `model.eval()` so important before inference?

This method sets the model to **evaluation mode**. This has a specific effect on certain layers that behave differently during training and testing:

1.  **Dropout Layers (`nn.Dropout`):** During training, dropout randomly zeros out some neurons. During evaluation, we want to use the entire network, so `model.eval()` **deactivates** the dropout layers.

2.  **Batch Normalization Layers (`nn.BatchNorm1d`):** During training, batch norm calculates the mean and standard deviation of the current mini-batch. During evaluation, we want consistent predictions for a single input, regardless of the batch it's in. `model.eval()` switches the batch norm layers to use the **running statistics** (the overall mean and std) that were learned during training.

Forgetting to call `model.eval()` is a very common source of bugs that can lead to inconsistent and poor-quality predictions.

```python
print("\n--- Part 2: The Importance of model.eval() ---")

# --- Create a model with Dropout and BatchNorm ---
model_with_special_layers = nn.Sequential(
    nn.Linear(10, 20),
    nn.BatchNorm1d(20),
    nn.Dropout(p=0.5),
    nn.Linear(20, 1)
)

dummy_input = torch.randn(4, 10) # A mini-batch

# --- Get output in training mode ---
model_with_special_layers.train() # Set to training mode
output_train = model_with_special_layers(dummy_input)

# --- Get output in evaluation mode ---
model_with_special_layers.eval() # Set to evaluation mode
output_eval = model_with_special_layers(dummy_input)

print(f"Output in train mode (first sample): {output_train[0].item():.4f}")
print(f"Output in eval mode (first sample):  {output_eval[0].item():.4f}")
print("--> The outputs are different because Dropout and BatchNorm behave differently.")
```

---

## Part 3: Saving a Checkpoint for Resuming Training

Sometimes, you don't just want to save the final model for inference; you want to save the entire state of your training process so you can resume it later. This is called **checkpointing** and is crucial for long training runs.

A checkpoint should contain:
*   The current epoch number.
*   The model's `state_dict`.
*   The optimizer's `state_dict` (this contains its internal state, like momentum values).
*   The current loss.
*   Any other necessary information.

### 3.1. Saving and Loading a Checkpoint

```python
import torch.optim as optim

print("\n--- Part 3: Saving and Loading a Checkpoint ---")

# --- Setup ---
model_ckpt = SimpleModel()
optimizer_ckpt = optim.SGD(model_ckpt.parameters(), lr=0.001, momentum=0.9)

# --- Saving a Checkpoint ---
# Let's assume we are at epoch 10 with a certain loss
epoch = 10
loss = 0.42
CHECKPOINT_PATH = "training_checkpoint.pth"

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model_ckpt.state_dict(),
    'optimizer_state_dict': optimizer_ckpt.state_dict(),
    'loss': loss,
}

torch.save(checkpoint, CHECKPOINT_PATH)
print(f"Checkpoint saved at epoch {epoch}.")

# --- Loading a Checkpoint ---
# Create new instances
model_to_resume = SimpleModel()
optimizer_to_resume = optim.SGD(model_to_resume.parameters(), lr=0.01) # Note the different LR

# Load the checkpoint dictionary
checkpoint = torch.load(CHECKPOINT_PATH)

# Load the states into the new model and optimizer
model_to_resume.load_state_dict(checkpoint['model_state_dict'])
optimizer_to_resume.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_resumed = checkpoint['epoch']
loss_resumed = checkpoint['loss']

# Now you can continue training from where you left off
model_to_resume.train()

print(f"\nCheckpoint loaded. Resuming training from epoch {epoch_resumed}.")
print(f"The optimizer's learning rate has been restored to: {optimizer_to_resume.param_groups[0]['lr']}")
```

## Conclusion

Serialization is the critical bridge between model development and model deployment. By understanding how to correctly save and load your model's state, you can create robust pipelines for inference and ensure that long training runs are not lost due to interruptions.

**Key Takeaways:**

1.  **Save the `state_dict`, Not the Model:** This is the most robust and recommended method. It decouples the learned weights from the code that defines the architecture.
2.  **Always Call `model.eval()`:** Before any inference or evaluation, switch your model to evaluation mode to get correct and deterministic results from layers like Dropout and BatchNorm.
3.  **Checkpoint for Long Runs:** Save the model state, optimizer state, and epoch number periodically so you can resume training seamlessly if it gets interrupted.

With these fundamental skills, you are now ready to explore more advanced deployment techniques like TorchScript and ONNX, which are used to create even more portable and high-performance versions of your models.

## Self-Assessment Questions

1.  **`state_dict`:** What information is contained in a model's `state_dict`? What information is missing?
2.  **Loading:** What is the first thing you must do before you can load a `state_dict` into a model?
3.  **`model.eval()`:** Name the two main types of layers that are affected by switching between `model.train()` and `model.eval()` mode.
4.  **Checkpoints:** Besides the model's `state_dict`, what is another crucial component to save in a training checkpoint if you want to resume training properly?
5.  **Robustness:** Why is saving the entire model object with `torch.save(model, PATH)` considered less robust than saving just the `state_dict`?
