# Day 21.4: PyTorch Geometric Implementation - A Case Study

## Introduction: From Theory to a Real-World Model

We have explored the mathematical foundations and the high-level architectures of various Graph Neural Networks. Now, it's time to put it all together and build a complete, end-to-end GNN model for a real-world task using **PyTorch Geometric (PyG)**.

PyG is the premier library for deep learning on graphs in PyTorch. It provides a clean API, highly optimized GNN layers, and a vast collection of benchmark datasets, making it the essential tool for any GNN practitioner.

This guide will provide a detailed, step-by-step case study of using PyG to build a **GraphSAGE** model for the **Cora citation network** node classification task. This will serve as a complete blueprint for your own GNN projects.

**Today's Learning Objectives:**

1.  **Master the PyG Workflow:** Understand the standard pipeline: `Dataset -> DataLoader -> Model -> Training Loop`.
2.  **Use PyG `DataLoader`s:** Learn how PyG provides specialized data loaders for handling graphs.
3.  **Build a Multi-Layer GNN with PyG:** Construct a complete GNN model using pre-built PyG layers like `SAGEConv`.
4.  **Write a Full Training and Evaluation Loop:** Implement the code to train the model, calculate the loss on the correct node masks, and evaluate its performance.
5.  **Perform Inference and Visualize Embeddings:** Use the trained model to make predictions and visualize the learned node embeddings with t-SNE.

---

## Part 1: The Dataset and `NeighborLoader`

We will again use the Cora dataset. However, for a more scalable approach, we won't process the full graph at once. Instead, we will use PyG's `NeighborLoader`. This is a `DataLoader` that implements the **sampling** part of the GraphSAGE algorithm.

*   **How it works:** For each mini-batch, it takes a set of target nodes (e.g., the training nodes). Then, it samples a fixed number of neighbors for these nodes. Then it samples neighbors of those neighbors, and so on, for a specified number of hops (`num_neighbors`). This creates a small, localized computation graph for each batch, making it possible to train on massive graphs that don't fit on the GPU.

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

print("--- Part 1: Data Loading with NeighborLoader ---")

# --- 1. Load the Dataset ---
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

# --- 2. Create the NeighborLoader for Training ---
# This loader will create mini-batches centered around the training nodes.
# `num_neighbors=[10, 5]` means: for each target node, sample 10 of its direct neighbors (1-hop),
# and for each of those neighbors, sample 5 of their neighbors (2-hop).
# The number of elements in the list corresponds to the number of GNN layers.
train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],
    batch_size=16,
    input_nodes=data.train_mask,
)

# We can also create a loader for the full graph for inference
subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1024)

# --- 3. Inspect a Mini-Batch ---
batched_graph = next(iter(train_loader))

print(f"A mini-batch from NeighborLoader is a graph object:")
print(batched_graph)
print(f"\nIt contains {batched_graph.num_nodes} nodes, which is more than the batch size ({batched_graph.batch_size}).")
print("This includes the 16 target nodes and all their sampled neighbors.")
```

---

## Part 2: The GraphSAGE Model

We will now build our GNN model using PyG's `SAGEConv` layers. The model will have two GraphSAGE layers.

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

print("\n--- Part 2: The GraphSAGE Model ---")

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # We use the mean aggregator for this example
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        # The forward pass is a simple sequential application of the layers
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- Instantiate the model ---
model = GraphSAGE(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
)

print("GraphSAGE model created:")
print(model)
```

---

## Part 3: The Full Training and Evaluation Loop

Now we write the code to train our model on the mini-batches from the `NeighborLoader` and evaluate its performance.

```python
print("\n--- Part 3: Training and Evaluation ---")

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# --- Training Function for one epoch ---
def train_one_epoch():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        # Get the output for the nodes in the current mini-batch computation graph
        out = model(batch.x, batch.edge_index)
        # The loss is computed only on the target nodes of the batch, not the neighbors.
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# --- Test Function ---
@torch.no_grad()
def test():
    model.eval()
    # For testing, we get the embeddings for ALL nodes using the subgraph_loader
    # In a real large-graph scenario, you might do this in batches.
    out = model.inference(data.x, subgraph_loader, device)
    pred = out.argmax(dim=1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = (pred[mask] == data.y[mask]).sum() / mask.sum()
        accs.append(acc.item())
    return accs

# --- The Main Loop ---
for epoch in range(1, 51):
    loss = train_one_epoch()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
```

---

## Part 4: Inference and Embedding Visualization

After training, our model is a powerful feature extractor. We can use its final node embeddings for various downstream tasks or visualize them to see the structure it has learned.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("\n--- Part 4: Visualization ---")

# --- 1. Get the final node embeddings ---
model.eval()
with torch.no_grad():
    # The `inference` method of the model gets the final output (logits)
    # To get the embeddings, we can modify the model to return the output of the first layer.
    final_logits = model.inference(data.x, subgraph_loader, device)
    
    # For visualization, let's get the embeddings from the first layer
    h = model.conv1(data.x.to(device), data.edge_index.to(device)).relu()
    node_embeddings = h.cpu().numpy()

# --- 2. Use t-SNE to reduce to 2D ---
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(node_embeddings)

# --- 3. Plot ---
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y.cpu(), cmap="jet", s=15)
plt.title('t-SNE Visualization of Cora Node Embeddings from GraphSAGE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Node Class')
plt.show()
```

**Interpretation:** The t-SNE plot visualizes the high-dimensional node embeddings in 2D. If the GNN has learned effectively, the nodes should form distinct clusters based on their class label (their color). This shows that the model has learned to map nodes with similar neighborhood structures and features to nearby points in the embedding space.

## Conclusion

This case study demonstrates the complete, end-to-end workflow for a practical GNN project using PyTorch Geometric. We have seen how to leverage PyG's powerful and efficient tools to handle data loading, model creation, and training for a node classification task.

**Key Takeaways from the Case Study:**

1.  **Use PyG for Practicality:** Libraries like PyG are essential for real-world GNN applications. They abstract away the complex and inefficient parts of handling graph data.
2.  **`NeighborLoader` for Scalability:** For large graphs, sampling-based data loaders like `NeighborLoader` are the key to training models that would otherwise not fit into memory.
3.  **The Training Loop is Standard:** The core training loop is very similar to other PyTorch models, with the main difference being how the data batches (which are graph objects) are handled.
4.  **Loss on Target Nodes:** When using a sampler like `NeighborLoader`, it's important to remember to compute the loss only on the target nodes for that batch, not on the entire computation graph which includes the sampled neighbors.
5.  **GNNs are Powerful Feature Extractors:** The trained GNN produces rich embeddings for each node, which can then be used for visualization or any downstream task.

This blueprint provides a solid foundation that you can adapt and extend to tackle your own graph-based machine learning problems.

## Self-Assessment Questions

1.  **`NeighborLoader`:** What is the main purpose of using a `NeighborLoader`?
2.  **Batch Object:** What does a single "batch" object yielded by the `NeighborLoader` represent?
3.  **Loss Calculation:** In our training loop, why do we use `out[:batch.batch_size]` when calculating the loss?
4.  **Inference:** What is the purpose of the `model.inference()` method provided by PyG?
5.  **Visualization:** What does a good t-SNE visualization of the final node embeddings look like, and what does it signify?

```