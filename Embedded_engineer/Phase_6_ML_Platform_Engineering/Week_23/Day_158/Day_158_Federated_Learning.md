# Day 158: The Data Never Moves: Federated Learning
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 23: Security & Governance

---

> **🎯 Focus Area:** You want to train a predictive typing model on user text messages. You cannot upload those messages to the cloud (Privacy Violation). **Federated Learning (FL)** trains the model *on the phone* and only uploads the weight updates.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Understand** the FedAvg (Federated Averaging) algorithm.
2.  **Implement** a Federated Simulation using the **Flower (Flwr)** framework.
3.  **Apply** Differential Privacy (DP) to the weight updates to prevent reconstruction.
4.  **Differentiate** between Cross-Silo (Hospital to Hospital) and Cross-Device (Phone to Cloud) FL.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install flwr torch torchvision`.

---

## 📖 Theoretical Foundation

### 1. FedAvg Algorithm
1.  **Server:** Sends Global Model $W_t$ to selected clients.
2.  **Client $k$:** Trains on local data $D_k$ for $E$ epochs -> $W_{t+1}^k$.
3.  **Client $k$:** Sends $\Delta W = W_{t+1}^k - W_t$ back to Server.
4.  **Server:** Aggregates updates: $W_{t+1} = W_t + \eta \sum \frac{|D_k|}{|D|} \Delta W$.

### 2. Challenges
*   **System Heterogeneity:** Some phones are fast, some slow. Droagglers.
*   **Statistical Heterogeneity:** Data is Non-IID. One user only types "LOL", another only "Bonjour".
*   **Communication:** Upstream bandwidth is limited.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Flower Server

The central coordinator.

#### 📁 `src/fl_server.py`
```python
import flwr as fl

def main():
    # Define aggregation strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # Sample 50% of available clients for training
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_available_clients=2,
    )

    # Start Server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
```

### 👨‍💻 Core Implementation: Flower Client

The worker node (simulating a Phone or Hospital).

#### 📁 `src/fl_client.py`
```python
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# 1. Standard PyTorch Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 5 * 5)
        return self.fc2(F.relu(self.fc1(x)))

# 2. FL Client Wrapper
class CifarClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        # Extract model weights as list of numpy arrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        # Apply global weights
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1) # Local Training
        return self.get_parameters(config=None), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}

# 3. Connect to Server
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient(...))
```

---

## 🔬 Lab Exercise: "The Poisoned Update"

### Task
Simulate a malicious client.
1.  Spin up Server.
2.  Spin up Client A (Honest).
3.  Spin up Client B (Malicious).
    *   In `fit()`, Client B sends random noise * 100 as weights.
4.  **Observation:** Global Model accuracy tanks to 10% (Random Guessing). FedAvg is not robust to outliers.
5.  **Fix:** Change Strategy to `FedMedian` or `Krum`. These ignore updates that are too far from the Euclidean mean.

---

## 📖 Advanced Theory: Secure Aggregation
Even receiving gradient updates leaks info.
**Secure Aggregation** enables the Server to calculate the Sum of Updates $\sum \Delta W$ *without seeing individual* $\Delta W_k$.
*   Technique: Masking.
*   Client A sends $X + R$. Client B sends $Y - R$.
*   Server sums: $(X + R) + (Y - R) = X + Y$.
*   Server knows sum, but doesn't know R.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Data Sovereignty:** FL is critical for GDPR. Data stays in the EU (on the device), only insights leave.
2.  **Bandwidth:** The cost of FL is communication. Techniques like **Gradient Compression** (sending only top 1% gradients) are essential.
3.  **Synchronization:** FedAvg is synchronous (Blocking). FedAsync is non-blocking but harder to converge.

### API Summary
```python
fl.server.start_server()
fl.client.start_numpy_client()
```

---

**Day 158 Complete** ✅

*Next: Day 159 - Compliance - GDPR, HIPAA, and the Right to be Forgotten.*
