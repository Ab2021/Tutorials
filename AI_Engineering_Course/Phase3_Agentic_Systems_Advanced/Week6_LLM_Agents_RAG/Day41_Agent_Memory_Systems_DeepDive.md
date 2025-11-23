# Day 41: Agent Memory Systems
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Memory System Implementation

```python
import openai
import numpy as np
from datetime import datetime
from typing import List, Dict
import json

class AgentMemory:
    def __init__(self, user_id: str, vector_db):
        self.user_id = user_id
        self.vector_db = vector_db
        
        # Memory stores
        self.working_memory = []
        self.short_term_memory = []
        self.semantic_memory = {}
        self.long_term_memory = vector_db
    
    def add_message(self, role: str, content: str):
        """Add message to short-term memory."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term_memory.append(message)
        self._store_long_term(message)
    
    def _store_long_term(self, message: Dict):
        """Store in vector database."""
        text = f"{message['role']}: {message['content']}"
        metadata = {
            "user_id": self.user_id,
            "timestamp": message["timestamp"],
            "role": message["role"]
        }
        self.long_term_memory.add(text, metadata)
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories."""
        results = self.long_term_memory.search(
            query,
            filter={"user_id": self.user_id},
            top_k=top_k
        )
        return results
    
    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Build context from all memory types."""
        context_parts = []
        
        # Semantic memory
        if self.semantic_memory:
            context_parts.append("User Preferences:")
            for key, value in self.semantic_memory.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        # Recent short-term
        if self.short_term_memory:
            context_parts.append("Recent Conversation:")
            for msg in self.short_term_memory[-10:]:
                context_parts.append(f"{msg['role']}: {msg['content']}")
            context_parts.append("")
        
        # Relevant long-term
        relevant = self.recall(query, top_k=3)
        if relevant:
            context_parts.append("Relevant Past:")
            for mem in relevant:
                context_parts.append(f"- {mem['text']}")
        
        return "\n".join(context_parts)
    
    def extract_learnings(self):
        """Extract semantic knowledge."""
        if len(self.short_term_memory) < 5:
            return
        
        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.short_term_memory
        ])
        
        prompt = f"""Extract key facts about the user from this conversation.

{conversation}

Facts (JSON):"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        try:
            learnings = json.loads(response.choices[0].message.content)
            self.semantic_memory.update(learnings)
        except:
            pass
    
    def consolidate(self):
        """Consolidate old memories."""
        if len(self.short_term_memory) > 50:
            old = self.short_term_memory[:30]
            summary = self._summarize(old)
            self.short_term_memory = [
                {"role": "system", "content": f"Summary: {summary}"}
            ] + self.short_term_memory[30:]
    
    def _summarize(self, messages: List[Dict]) -> str:
        """Summarize messages."""
        conv = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"Summarize in 2-3 sentences:\n{conv}\n\nSummary:"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content

class MemoryAgent:
    def __init__(self, user_id: str, vector_db):
        self.memory = AgentMemory(user_id, vector_db)
    
    def chat(self, user_message: str) -> str:
        """Chat with memory."""
        self.memory.add_message("user", user_message)
        context = self.memory.get_context(user_message)
        
        prompt = f"{context}\n\nUser: {user_message}\nAssistant:"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        self.memory.add_message("assistant", answer)
        
        # Extract learnings periodically
        if len(self.memory.short_term_memory) % 10 == 0:
            self.memory.extract_learnings()
        
        return answer
```

### 2. Memory Scoring and Ranking

```python
def score_memory(memory: Dict, query: str, current_time: datetime) -> float:
    """Score memory by relevance, recency, and importance."""
    # Relevance (cosine similarity)
    relevance = memory.get("similarity", 0.0)
    
    # Recency (exponential decay)
    memory_time = datetime.fromisoformat(memory["timestamp"])
    age_days = (current_time - memory_time).days
    recency = np.exp(-age_days / 30)  # Half-life of 30 days
    
    # Importance (access count)
    importance = memory.get("access_count", 0) / 100
    
    # Combined score
    score = 0.5 * relevance + 0.3 * recency + 0.2 * importance
    
    return score
```

### 3. Memory Reflection

```python
def reflect_on_memories(memories: List[str]) -> str:
    """Generate high-level insights from memories."""
    prompt = f"""Analyze these memories and generate 3 high-level insights.

Memories:
{chr(10).join(memories)}

Insights:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content
```
