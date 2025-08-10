# Day 26.2: Federated Averaging (FedAvg) - A Practical Deep Dive

## Introduction: The Workhorse of Federated Learning

Federated Learning (FL) is built on the idea of training a shared global model by aggregating updates from multiple decentralized clients. The algorithm that made this practical and popular is **Federated Averaging (FedAvg)**.

While the high-level concept of "averaging weights" seems simple, the details of the FedAvg algorithm are crucial for understanding why FL works and how it is implemented. It elegantly combines local client computation with global server aggregation to produce a high-quality model without ever seeing the raw data.

This guide provides a detailed, practical deep dive into the FedAvg algorithm, breaking down its steps and implementing it from scratch to build a strong intuition for its mechanics.

**Today's Learning Objectives:**

1.  **Deconstruct the FedAvg Algorithm:** Understand the three key steps: client selection, local client updates, and server aggregation.
2.  **Analyze the Role of Local Epochs:** Grasp why running multiple local training steps on the client is a key factor in the efficiency of FedAvg.
3.  **Implement the Server-Side Aggregation:** Write the Python code that performs the weighted averaging of client model `state_dict`s.
4.  **Implement the Client-Side Update:** Write the code for a client that receives a model, trains it on its local data, and prepares the update for the server.
5.  **Integrate into a Full Simulation:** Combine the server and client logic into a complete, runnable simulation to see the algorithm in action.

---

## Part 1: The FedAvg Algorithm in Detail

Let's formalize the steps described in the original 2016 paper, "Communication-Efficient Learning of Deep Networks from Decentralized Data."

**Server Executes:**
1.  Initialize a global model with weights `w_0`.
2.  For each communication round `t = 1, 2, ...`:
    a. Determine the number of clients `K` and a fraction `C` of clients to participate in the round. Select a random subset of `m = max(C * K, 1)` clients.
    b. **Broadcast:** Send the current global model weights `w_{t-1}` to all `m` selected clients.
    c. **Wait and Aggregate:** Wait for the clients to send back their updated weights. Once a sufficient number of updates are received:
        i. Calculate the total number of samples `N` across all participating clients.
        ii. For each client `k`, get its updated weights `w_t^k` and the number of local samples `n_k`.
        iii. Update the global model weights by performing a weighted average: `w_t = sum_{k=1 to m} ( (n_k / N) * w_t^k )`.

**Client `k` Executes:**
1.  Receive the current global model weights `w_{t-1}` from the server.
2.  Set its local model weights to `w_{t-1}`.
3.  Partition its local data into batches.
4.  For a specified number of **local epochs** `E`:
    a. For each local batch `b`:
        i. Perform a standard training step (forward pass, loss calculation, backpropagation, optimizer step) on the local model.
5.  Return the updated local model weights `w_t^k` to the server.

**The Key Efficiency Gain:**
The communication between clients and the server is usually the biggest bottleneck. By allowing each client to perform multiple local updates (`E > 1`), FedAvg dramatically reduces the number of communication rounds needed to train the model, making it much more communication-efficient than a simple federated gradient descent approach.

---

## Part 2: A From-Scratch FedAvg Implementation

Let's build a more structured simulation than the one in the previous guide, with clear classes for the `Server` and `Client`.

### 2.1. The Client Logic

A client needs to be able to receive a model, train it on its local data loader, and return the updated weights.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

print("--- Part 2: From-Scratch FedAvg Implementation ---")

# --- We assume a simple MLP model is defined ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x.view(-1, 784))

class Client:
    def __init__(self, client_id, data_loader):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = SimpleMLP()
        self.data_size = len(data_loader.dataset)

    def set_weights(self, global_weights):
        """Load the global model weights into the local model."""
        self.model.load_state_dict(global_weights)

    def train(self, local_epochs, learning_rate):
        """Train the local model on the local data."""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(local_epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Return the updated weights
        return self.model.state_dict()

print("Client class defined.")
```

### 2.2. The Server Logic

The server manages the global model, orchestrates the rounds, and performs the aggregation.

```python
class Server:
    def __init__(self, num_clients, client_datasets):
        self.global_model = SimpleMLP()
        self.num_clients = num_clients
        # Create all the client objects
        self.clients = [Client(i, DataLoader(ds, batch_size=32, shuffle=True)) 
                        for i, ds in enumerate(client_datasets)]

    def broadcast_weights(self, clients_to_train):
        """Send the global model weights to a list of clients."""
        global_weights = self.global_model.state_dict()
        for client in clients_to_train:
            client.set_weights(deepcopy(global_weights))

    def aggregate_weights(self, local_updates):
        """
        Performs Federated Averaging.
        Args:
            local_updates: A list of tuples (data_size, state_dict).
        """
        total_data_size = sum([update[0] for update in local_updates])
        aggregated_weights = deepcopy(local_updates[0][1]) # Use first as template

        # Zero out the template
        for key in aggregated_weights:
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        # Perform weighted average
        for data_size, state_dict in local_updates:
            weight = data_size / total_data_size
            for key in state_dict:
                aggregated_weights[key] += state_dict[key] * weight
        
        # Load the new weights into the global model
        self.global_model.load_state_dict(aggregated_weights)

    def run_simulation(self, num_rounds, local_epochs, lr):
        for round_idx in range(num_rounds):
            # For simplicity, we train on all clients each round
            clients_to_train = self.clients
            
            # 1. Broadcast
            self.broadcast_weights(clients_to_train)
            
            # 2. Local Training
            local_updates = []
            for client in clients_to_train:
                updated_weights = client.train(local_epochs, lr)
                local_updates.append((client.data_size, updated_weights))
            
            # 3. Aggregation
            self.aggregate_weights(local_updates)
            
            print(f"Round {round_idx + 1}/{num_rounds} complete.")
            # (In a real scenario, you would evaluate the global model here)

print("Server class defined.")
```

### 2.3. Running the Simulation

```python
from torchvision import datasets, transforms
from torch.utils.data import Subset

print("\n--- Running the FedAvg Simulation ---")

# --- 1. Prepare the partitioned MNIST data ---
# (Using the same non-IID partitioning as the previous guide)
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
labels = train_dataset.targets
sorted_indices = labels.argsort()
sorted_dataset = Subset(train_dataset, sorted_indices)
client_data_size = len(sorted_dataset) // 10
client_datasets = [Subset(sorted_dataset, np.arange(i*client_data_size, (i+1)*client_data_size)) for i in range(10)]

# --- 2. Create the Server and run the simulation ---
server = Server(num_clients=10, client_datasets=client_datasets)

server.run_simulation(num_rounds=5, local_epochs=3, lr=0.01)

print("\nFedAvg simulation finished.")
```

## Conclusion

Federated Averaging is a simple yet remarkably effective algorithm that forms the backbone of most federated learning systems. By understanding its mechanics—client-side local training and server-side weighted averaging—you grasp the core technology that enables collaborative machine learning without centralizing data.

**Key Takeaways:**

1.  **Local Epochs are Key:** The communication efficiency of FedAvg comes from performing multiple updates on the local client data (`E > 1`) before communicating with the server.
2.  **Weighted Averaging:** The server aggregates client models by taking a weighted average of their parameters, with the weights being proportional to the amount of data each client has. This gives more influence to clients who have learned from more data.
3.  **Decoupled Logic:** The server and client have distinct roles. The server orchestrates and aggregates, while the client performs the actual training on its private data.
4.  **Foundation for Advanced FL:** While FedAvg is the baseline, more advanced federated algorithms build upon it to handle issues like statistical heterogeneity (FedProx), personalization, and enhanced privacy.

This from-scratch implementation provides a clear and concrete understanding of the moving parts in a federated system and serves as a strong foundation for exploring more advanced topics in the field.

## Self-Assessment Questions

1.  **Communication Efficiency:** What is the main technique used by FedAvg to reduce the number of communication rounds required for training?
2.  **Aggregation:** In the FedAvg aggregation step, why is it important to weight the average by the number of samples on each client?
3.  **Client Update:** What does a client send back to the server after its local training is complete?
4.  **Server Role:** What are the two main responsibilities of the server in the FedAvg algorithm?
5.  **Stateful Clients:** In our simulation, the `Client` class holds its own data loader. In a real-world mobile phone setting, would the client be stateful like this, or would the server need to send the data? Why is this distinction important?
