# Lab 5: GraphRAG Construction

## Objective
Text is linear. Knowledge is connected.
Build a simple **Knowledge Graph** from text.

## 1. The Builder (`graph.py`)

```python
import networkx as nx
import matplotlib.pyplot as plt

text = "Elon Musk is the CEO of Tesla. Tesla produces EVs."

# 1. Extract Triples (Mock LLM)
triples = [
    ("Elon Musk", "CEO of", "Tesla"),
    ("Tesla", "produces", "EVs")
]

# 2. Build Graph
G = nx.DiGraph()
for subj, pred, obj in triples:
    G.add_edge(subj, obj, label=pred)

# 3. Visualize
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()
```

## 2. Challenge
Implement a **Graph Traversal** search.
Query: "What does Elon Musk's company produce?"
Path: Elon Musk -> Tesla -> EVs.

## 3. Submission
Submit the image of the generated graph.
