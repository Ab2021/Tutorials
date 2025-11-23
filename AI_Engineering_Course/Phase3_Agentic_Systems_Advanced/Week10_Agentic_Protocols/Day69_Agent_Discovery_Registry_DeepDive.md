# Day 69: Agent Discovery & Registry
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building a Semantic Agent Registry

We will build a simple Registry using a Vector Database (ChromaDB) to allow agents to find each other.

**Dependencies:**
`pip install chromadb sentence-transformers`

**Code (`registry.py`):**

```python
import chromadb
from chromadb.utils import embedding_functions

class AgentRegistry:
    def __init__(self):
        self.client = chromadb.Client()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.create_collection(
            name="agents", 
            embedding_function=self.ef
        )

    def register_agent(self, agent_id, endpoint, capabilities):
        """Register an agent with a description of what it does."""
        # We embed the capabilities string
        self.collection.add(
            documents=[capabilities],
            metadatas=[{"endpoint": endpoint, "id": agent_id}],
            ids=[agent_id]
        )
        print(f"Registered {agent_id}")

    def find_agent(self, intent, n=1):
        """Find the best agent for a given intent."""
        results = self.collection.query(
            query_texts=[intent],
            n_results=n
        )
        
        if not results['ids'][0]:
            return None
            
        best_match = {
            "id": results['ids'][0][0],
            "endpoint": results['metadatas'][0][0]["endpoint"],
            "capabilities": results['documents'][0][0],
            "distance": results['distances'][0][0]
        }
        return best_match

# Usage
registry = AgentRegistry()

# 1. Registration Phase
registry.register_agent(
    "weather_bot", 
    "http://weather.com/api", 
    "I can provide current weather, forecasts, and humidity for any city."
)
registry.register_agent(
    "stock_bot", 
    "http://finance.com/api", 
    "I can check stock prices, market caps, and P/E ratios."
)

# 2. Discovery Phase
intent = "Is it raining in London?"
agent = registry.find_agent(intent)
print(f"Found Agent: {agent['id']} ({agent['endpoint']})")
# Output: Found Agent: weather_bot
```

### The "Well-Known" Protocol

How do I find the Registry itself?
*   **DNS TXT Records:** `_agent_registry.example.com TXT "v=1;url=https://registry.example.com"`
*   **/.well-known/agent.json:** A standardized path on a domain that lists the agents hosted there.

```json
{
  "agents": [
    {
      "id": "support-bot",
      "endpoint": "/api/agent/support",
      "description": "Customer support for this website."
    }
  ]
}
```

### Crawler Agents

Just like Google crawls the web, **Agent Crawlers** will crawl `/.well-known/agent.json` files to build a global index of available agents.

### Summary

*   **Registration:** Agents announce their capabilities.
*   **Indexing:** The Registry embeds these capabilities.
*   **Querying:** Client agents search by intent.
This loop enables **Open-Endedness**. Your agent can solve problems it wasn't programmed for by finding a tool that was.
