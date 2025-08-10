# Day 2.4: Cloud Computing & Colab Guide - A Practical Introduction

## Introduction: Your Free GPU in the Cloud

Training deep learning models, especially on large datasets, can be computationally expensive and requires a powerful GPU. Not everyone has access to a high-end NVIDIA GPU on their local machine. This is where cloud computing platforms come in, and **Google Colaboratory (Colab)** is one of the most accessible and popular choices.

Colab is essentially a free Jupyter Notebook environment that runs entirely in the cloud. Crucially, it provides **free access to GPUs (and even TPUs)**, making it an invaluable tool for students, researchers, and anyone looking to get started with deep learning without investing in expensive hardware.

This guide will provide a practical, step-by-step walkthrough of using Google Colab for your deep learning projects.

**Today's Learning Objectives:**

1.  **Navigate the Colab UI:** Understand the basic interface of a Colab notebook.
2.  **Enable and Use a GPU:** Learn how to switch to a GPU runtime and verify that PyTorch can use it.
3.  **Manage Files in Colab:**
    *   Upload files directly to the Colab environment.
    *   Mount your Google Drive to persist files and datasets between sessions.
4.  **Install Libraries:** Learn how to install packages in the Colab environment using `pip`.
5.  **Understand Colab's Limitations:** Be aware of session timeouts and storage impermanence.

---

## Part 1: Getting Started with Google Colab

### 1.1. Accessing Colab

All you need is a Google account.

1.  Go to [https://colab.research.google.com/](https://colab.research.google.com/).
2.  You will be greeted with a splash screen. You can click "New notebook" to create a new `.ipynb` file.

Your new notebook is a standard Jupyter Notebook. You can create code cells, text (Markdown) cells, and run them just like you would locally.

### 1.2. Enabling the GPU Runtime

By default, your notebook runs on a CPU. To enable the free GPU:

1.  Click on the **"Runtime"** menu at the top of the screen.
2.  Select **"Change runtime type"**.
3.  In the dropdown menu under **"Hardware accelerator,"** select **GPU**.
4.  Click **"Save"**.

The environment will now restart, and you will be connected to a machine with an NVIDIA GPU.

### 1.3. Verifying the GPU

Let's run the same verification code we used in the environment setup guide. Colab comes with PyTorch pre-installed, so we can use it immediately.

Execute the following code in a Colab cell:

```python
import torch

# Check if CUDA (the platform for GPU operations) is available
is_available = torch.cuda.is_available()

print(f"Is GPU available? {is_available}")

if is_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    # Get the name of the GPU. This can vary (Tesla T4, K80, P100, etc.)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("GPU not available. Please go to Runtime > Change runtime type and select GPU.")
```

If the output shows that a GPU is available and prints its name, you are all set!

---

## Part 2: Managing Files and Data

This is the most important concept to understand about Colab: the environment is **temporary**. When your session ends (either because you close the tab, or it times out due to inactivity), **all files you have saved in the local Colab environment will be deleted.**

Therefore, you need a way to work with persistent storage. The best way to do this is by using your Google Drive.

### 2.1. The Temporary Colab Filesystem

You can see the local filesystem by clicking the folder icon in the left-hand sidebar.

*   **Uploading Files:** You can click the "Upload to session storage" button to upload small files directly. **Remember, these will be deleted when the session ends.** This is fine for temporary scripts or very small datasets.

*   **Using `!wget`:** You can download datasets directly from the web into the Colab environment using shell commands. You can run any shell command by prefixing it with an exclamation mark (`!`).

    ```bash
    # This command will download a small sample dataset into the current directory
    !wget https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/airline_passengers.csv
    
    # You can then list the files to see it
    !ls
    ```

### 2.2. The Permanent Solution: Mounting Google Drive

This is the standard workflow for any serious project. Mounting your Google Drive connects it to the Colab filesystem, allowing you to read and write files directly from/to your Drive.

Execute the following code in a Colab cell:

```python
from google.colab import drive

# This will prompt you for authorization.
# Click the link, sign in to your Google account, copy the authorization code,
# and paste it into the box in Colab.
drive.mount('/content/drive')
```

After you authorize it, your entire Google Drive will appear in the filesystem under the `/content/drive/` directory. The most common location to work from is `/content/drive/MyDrive/`.

**A Typical Workflow:**

1.  Create a folder for your project in your Google Drive (e.g., `My Drive/Colab Notebooks/MyProject`).
2.  Upload your datasets, scripts, and other necessary files to this folder.
3.  In your Colab notebook, mount your drive.
4.  Use standard file I/O operations to access your files using their full path, e.g., `/content/drive/MyDrive/Colab Notebooks/MyProject/my_data.csv`.
5.  Save your trained models, logs, and results back to this same directory to ensure they are not lost.

```python
import os

# Define the path to your project folder on Google Drive
# Make sure you have created this folder in your Google Drive!
project_path = '/content/drive/MyDrive/Colab_DL_Course'
os.makedirs(project_path, exist_ok=True)

# Now you can save files there
model_save_path = os.path.join(project_path, 'my_model.pth')

# --- Dummy Model Saving Example ---
import torch
import torch.nn as nn

# Create a simple model
model = nn.Linear(10, 2)

# Save the model's state dictionary to your Google Drive
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to: {model_save_path}")

# Verify that the file exists
!ls -l '/content/drive/MyDrive/Colab_DL_Course'

# You can now load this model back in a future session
model_reloaded = nn.Linear(10, 2)
model_reloaded.load_state_dict(torch.load(model_save_path))
print("\nModel reloaded successfully from Google Drive.")
```

---

## Part 3: Installing Libraries

Colab comes with most major data science and deep learning libraries pre-installed (PyTorch, TensorFlow, Scikit-learn, Pandas, etc.). However, if you need a specific or newer version of a library, or a more obscure one, you can easily install it using `pip`.

Just like with other shell commands, you prefix the command with `!`.

```bash
# Install the `transformers` library from Hugging Face
!pip install transformers

# You can also upgrade existing libraries
!pip install --upgrade pandas

# You can check the version of an installed package
!pip show torch
```

**Important Note:** Installing a library this way only installs it for the **current session**. If the session restarts, you will need to run the installation cell again. It's common practice to have a setup cell at the top of your Colab notebook that contains all your `pip install` commands.

---

## Part 4: A Complete Training Example in Colab

Let's put it all together by training a simple CNN on the Fashion-MNIST dataset, saving the model to Google Drive.

```python
# This is a single cell to run in Google Colab

# --- 1. Setup and Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from google.colab import drive
import os

# --- 2. Mount Google Drive ---
print("Mounting Google Drive...")
drive.mount('/content/drive')

# --- 3. Define Paths and Parameters ---
project_path = '/content/drive/MyDrive/Colab_FMNIST'
os.makedirs(project_path, exist_ok=True)
model_path = os.path.join(project_path, 'fashion_mnist_cnn.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 128
num_epochs = 5

# --- 4. Load Data ---
print("Loading Fashion-MNIST dataset...")
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 5. Define the CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return x

model = SimpleCNN().to(device)

# --- 6. Training Loop ---
print("Starting training...")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished.")

# --- 7. Save the Model to Google Drive ---
print(f"Saving model to {model_path}")
torch.save(model.state_dict(), model_path)

# --- 8. Evaluation ---
print("Evaluating model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
```

## Conclusion: Your Deep Learning Playground

Google Colab is an essential tool in the modern data scientist's toolkit. It democratizes deep learning by removing the hardware barrier, providing a powerful, pre-configured environment that you can access from any browser.

**Key Best Practices for Colab:**

*   **Activate the GPU:** Always remember to switch to the GPU runtime for deep learning tasks.
*   **Mount Your Drive:** For any project that you want to save, mounting your Google Drive is the first step.
*   **Organize Your Drive:** Keep your Colab projects in well-named folders on your Google Drive.
*   **Save Checkpoints:** For long training runs, save your model checkpoints periodically to your Drive so you don't lose progress if the session times out.
*   **Be Mindful of Timeouts:** Colab notebooks will disconnect after a period of inactivity (typically 90 minutes) and the entire environment is deleted after about 12 hours.

By mastering the workflow of using Colab with Google Drive, you can effectively train and manage deep learning projects of significant scale without needing a powerful local machine.

## Self-Assessment Questions

1.  **Runtime:** What are the three types of hardware accelerators available in Colab?
2.  **Persistence:** What happens to a file you upload to the default Colab filesystem when your session ends? How do you prevent this data loss?
3.  **Shell Commands:** How do you run a shell command like `ls` or `pip` in a Colab cell?
4.  **File Path:** After mounting your Google Drive, what is the standard path prefix for files located in the main "My Drive" folder?
5.  **Session Management:** If you are training a model that will take 24 hours, is Colab a suitable tool? Why or why not?
