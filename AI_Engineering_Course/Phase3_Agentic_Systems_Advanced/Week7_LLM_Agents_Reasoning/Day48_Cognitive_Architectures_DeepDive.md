# Day 48: Cognitive Architectures (Generative Agents)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Memory Stream (Generative Agents)

The core of a Generative Agent is how it retrieves memories.

```python
import numpy as np

class MemoryStream:
    def __init__(self):
        self.memories = [] # List of {text, embedding, timestamp, importance}
        self.now = 0

    def add_memory(self, text, importance):
        embedding = get_embedding(text)
        self.memories.append({
            "text": text,
            "embedding": embedding,
            "created": self.now,
            "importance": importance,
            "last_accessed": self.now
        })

    def retrieve(self, query, k=3):
        q_embed = get_embedding(query)
        scores = []
        
        for mem in self.memories:
            # 1. Recency (Exponential decay)
            recency = 0.99 ** (self.now - mem["last_accessed"])
            
            # 2. Importance (Integer 1-10)
            importance = mem["importance"] / 10
            
            # 3. Relevance (Cosine Similarity)
            relevance = cosine_sim(q_embed, mem["embedding"])
            
            # Weighted Sum
            score = (1 * recency) + (1 * importance) + (5 * relevance)
            scores.append((score, mem))
            
        # Top K
        scores.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in scores[:k]]

    def tick(self):
        self.now += 1
```

### Reflection Tree

Raw memories ("Ate breakfast", "Saw Bob") are too detailed.
**Reflection** creates abstractions.
1.  **Sample:** Pick 100 recent memories.
2.  **Prompt:** "What high-level insights can you derive from these?"
3.  **Output:** "I usually eat breakfast at 8am." (Insight).
4.  **Store:** Save the Insight back into the Memory Stream.
Now, when asked "When do you eat?", the agent retrieves the Insight, not the 100 raw logs.

### Planning with Recursion

**Prompt:** "Plan your day."
**Output:** "Wake up, Work, Sleep."
**Decomposition:** "Expand 'Work'."
**Output:** "Check email, Write code, Meeting."
This hierarchical planning keeps the agent focused.

### Summary

*   **Retrieval Function:** The "Secret Sauce" of Generative Agents. Balancing Recency vs Relevance is key.
*   **Reflection:** Prevents memory overflow and allows generalization.
