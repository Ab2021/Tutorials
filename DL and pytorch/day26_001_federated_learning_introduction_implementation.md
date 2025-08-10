# Day 26.1: Federated Learning Introduction - A Practical Guide

## Introduction: Learning Without Centralizing Data

Traditional machine learning, and everything we have studied so far, operates on a centralized paradigm. You gather all your data from various sources, collect it in a central location (like a cloud server), and train a single, global model on this combined dataset.

However, this approach is becoming increasingly challenging due to:
*   **Data Privacy:** Regulations like GDPR and CCPA make it difficult and risky to collect and store sensitive user data.
*   **Data Gravity & Bandwidth:** Datasets are getting larger (e.g., from IoT devices, medical scanners). Moving this massive amount of data to a central server can be slow and expensive.
*   **Data Sovereignty:** Laws may require that data generated in a certain country never leaves its borders.

**Federated Learning (FL)** is a revolutionary machine learning paradigm that flips this on its head. It enables a model to be trained on data from multiple, decentralized sources **without the data ever leaving the source device**. The model trains locally where the data is, and only the learned updates are sent back to a central server.

This guide provides a high-level, practical introduction to the concepts and workflow of Federated Learning.

**Today's Learning Objectives:**

1.  **Understand the Motivation for Federated Learning:** Grasp the key privacy and logistical challenges of the centralized training paradigm.
2.  **Learn the Federated Learning Workflow:** Understand the cyclical process of broadcasting a model, training locally, and aggregating updates.
3.  **Differentiate Between the Server and the Clients:** Understand the distinct roles of the central server and the decentralized client devices.
4.  **Explore the Federated Averaging (FedAvg) Algorithm:** Learn about the standard algorithm used to aggregate the model updates from multiple clients.
5.  **Implement a Simplified Federated Learning Simulation:** Write a basic Python script to simulate the FedAvg process and see it in action.

---

## Part 1: The Federated Learning Workflow

Federated Learning is an iterative process involving a central **server** and a large number of **clients** (e.g., mobile phones, hospitals, smart cars).

A single training round proceeds as follows:

1.  **Selection:** The server selects a random subset of the available clients to participate in the current training round.

2.  **Broadcast:** The server sends the current version of the global model (its weights) to all the selected clients.

3.  **Local Training:** Each selected client **trains the received model on its own local data**. It performs several epochs of standard training (e.g., using SGD) on its local dataset. This is the key step—the data never leaves the client device.

4.  **Update & Communication:** After local training, each client sends its updated model weights (or the *change* in its weights) back to the central server. **Crucially, it only sends the weights, not its private data.**

5.  **Aggregation:** The server waits to receive updates from many of the selected clients. It then **aggregates** these updates to produce a new, improved global model. The standard aggregation method is **Federated Averaging (FedAvg)**.

6.  **Repeat:** The process repeats, with the new global model being broadcast in the next round.

![Federated Learning Workflow](https://i.imgur.com/1fA4s4p.png)

---

## Part 2: The Federated Averaging (FedAvg) Algorithm

How does the server combine the updates from multiple clients into a single, better model?

**The Idea:** The server simply takes a **weighted average** of the model weights received from all the clients.

`Global_Weights_{t+1} = sum_k ( (n_k / N) * Client_Weights_k )`

*   `k`: The index of a client.
*   `n_k`: The number of data samples on client `k`.
*   `N`: The total number of data samples across all participating clients.

By weighting the average by the number of samples on each client, we give more importance to the updates from clients that have more data, leading to a more stable and effective global model.

---

## Part 3: A Simplified Federated Learning Simulation

Let's implement this process in PyTorch. We will simulate a scenario with a central server and several clients, each with their own private slice of the MNIST dataset.

### 3.1. Setup: Data Partitioning and Model

First, we need to load MNIST and distribute it among our simulated clients.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

print("--- Part 3: Federated Learning Simulation ---")

# --- 1. Parameters and Model ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(-1, 28*28))

num_clients = 10
num_epochs_local = 3
learning_rate = 0.01

# --- 2. Load Data and Distribute it to Clients ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Create a non-IID distribution: sort data by label and partition
# This simulates a more realistic scenario where each client has a biased dataset.
labels = train_dataset.targets
sorted_indices = labels.argsort()
sorted_dataset = Subset(train_dataset, sorted_indices)

# Partition the sorted dataset among the clients
client_data_size = len(sorted_dataset) // num_clients
client_datasets = [Subset(sorted_dataset, np.arange(i*client_data_size, (i+1)*client_data_size))
                   for i in range(num_clients)]

client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]

print(f"Created {num_clients} clients, each with {client_data_size} data samples.")
```

### 3.2. The Simulation Loop

Now we implement the main FL loop: broadcast, local train, and aggregate.

```python
# --- 3. The Server and Global Model ---
global_model = SimpleMLP()

# --- 4. The Federated Learning Simulation Loop ---
num_rounds = 5

for round_idx in range(num_rounds):
    print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
    
    # Store the weights from all clients for this round
    local_weights = []
    local_data_sizes = []
    
    # --- Broadcast and Local Training ---
    # In a real system, this would be a random subset of clients.
    # Here, we use all of them for simplicity.
    for client_id in range(num_clients):
        # Create a local copy of the global model
        local_model = SimpleMLP()
        local_model.load_state_dict(global_model.state_dict())
        
        # Train the local model on the client's data
        optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        local_model.train()
        for _ in range(num_epochs_local):
            for data, target in client_loaders[client_id]:
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Store the updated local weights and data size
        local_weights.append(local_model.state_dict())
        local_data_sizes.append(len(client_loaders[client_id].dataset))
        
        if client_id % 4 == 0:
            print(f"  - Client {client_id} finished local training.")

    # --- Aggregation (Federated Averaging) ---
    print("Server aggregating client weights...")
    total_data_size = sum(local_data_sizes)
    # Get the state dict of the first client to use as a template
    global_state_dict = local_weights[0]
    
    # Zero out the template
    for key in global_state_dict:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # Perform the weighted average
    for i, state_dict in enumerate(local_weights):
        weight = local_data_sizes[i] / total_data_size
        for key in state_dict:
            global_state_dict[key] += state_dict[key] * weight
            
    # Load the new averaged weights into the global model
    global_model.load_state_dict(global_state_dict)
    
    print("Aggregation complete. New global model is ready for the next round.")

print("\nFederated Learning simulation finished.")
```

## Conclusion

Federated Learning represents a fundamental shift in how we approach training machine learning models. By bringing the model to the data, rather than the data to the model, it offers a powerful solution to the growing challenges of data privacy, security, and logistics.

**Key Takeaways:**

1.  **Decentralized Training:** The core principle is to train on decentralized data without ever moving that data to a central server.
2.  **The FL Workflow:** The process is a cycle of the server broadcasting a model, clients training it locally on their private data, and the server aggregating the resulting model updates.
3.  **Federated Averaging (FedAvg):** This is the standard algorithm for aggregation, where the server computes a weighted average of the client models' parameters.
4.  **Privacy is the Goal:** FL is not just a training algorithm; it's a privacy-preserving framework. The raw data never leaves the user's device.

While this simple simulation illustrates the core concepts, real-world federated learning systems are much more complex, dealing with issues like unreliable client connections, secure aggregation, and statistical heterogeneity. However, the fundamental principles remain the same, and FL is becoming an increasingly important tool for building responsible and scalable AI systems.

## Self-Assessment Questions

1.  **Centralized vs. Federated:** What is the key difference between the traditional centralized training paradigm and Federated Learning?
2.  **The FL Loop:** What are the five main steps in a single round of Federated Learning?
3.  **Data Privacy:** What is actually sent from the client device back to the central server?
4.  **Federated Averaging:** In the FedAvg algorithm, why are the client models' weights weighted by the number of samples on each client?
5.  **Non-IID Data:** What does it mean for the data in an FL system to be "Non-IID," and why is this a common scenario?
