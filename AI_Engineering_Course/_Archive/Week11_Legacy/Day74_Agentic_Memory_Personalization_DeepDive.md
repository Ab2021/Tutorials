# Day 74: Agentic Memory & Personalization
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Memory Module Implementation (Vector + SQL)

Hybrid memory for facts and history.

```python
import sqlite3
import numpy as np

class AgentMemory:
    def __init__(self):
        # SQL for Structured Facts (User Profile)
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE profile (key TEXT PRIMARY KEY, value TEXT)")
        
        # Vector List for Episodic History
        self.episodes = [] # (embedding, text, timestamp)
        
    def update_profile(self, key, value):
        self.conn.execute("INSERT OR REPLACE INTO profile VALUES (?, ?)", (key, value))
        self.conn.commit()
        
    def get_profile(self):
        cursor = self.conn.execute("SELECT * FROM profile")
        return {k: v for k, v in cursor.fetchall()}
        
    def add_episode(self, text, embedding_model):
        vec = embedding_model.encode(text)
        self.episodes.append({"vec": vec, "text": text})
        
    def recall(self, query, embedding_model, k=3):
        q_vec = embedding_model.encode(query)
        
        # Simple Cosine Search
        scores = []
        for ep in self.episodes:
            score = np.dot(q_vec, ep['vec'])
            scores.append((score, ep['text']))
            
        scores.sort(reverse=True)
        return [text for _, text in scores[:k]]

# Usage
# mem = AgentMemory()
# mem.update_profile("name", "Alice")
# mem.add_episode("User asked about Python", model)
# context = mem.recall("What did we talk about?", model)
```

### 2. User Profile Builder (Auto-Extraction)

Using LLM to extract facts from conversation.

```python
def update_user_profile(llm, current_profile, conversation_history):
    prompt = f"""
    Current Profile: {current_profile}
    
    Recent Conversation:
    {conversation_history}
    
    Extract new facts about the user (Name, Role, Preferences) and update the profile.
    Return JSON.
    """
    
    # response = llm.generate(prompt)
    # Mock response
    return {"name": "Alice", "role": "Data Scientist", "language": "Python"}

# This runs in the background after every session.
```

### 3. MemGPT Context Manager (Concept)

Managing the context window like an OS manages RAM.

```python
class ContextManager:
    def __init__(self, max_tokens=1000):
        self.max_tokens = max_tokens
        self.system_prompt = "You are a helpful assistant."
        self.working_memory = [] # Recent messages
        self.long_term_memory = [] # Summaries
        
    def add_message(self, message):
        self.working_memory.append(message)
        self.check_overflow()
        
    def check_overflow(self):
        current_tokens = self.count_tokens(self.working_memory)
        if current_tokens > self.max_tokens:
            # Evict oldest messages to Long Term
            to_evict = self.working_memory.pop(0)
            self.archive(to_evict)
            
    def archive(self, message):
        # Summarize or Embed
        print(f"Archiving: {message}")
        self.long_term_memory.append(message)
        
    def construct_prompt(self):
        # System + Retrieved Long Term + Working Memory
        return self.system_prompt + "\n" + str(self.working_memory)

    def count_tokens(self, msgs):
        return len(str(msgs)) / 4 # Approx
```

### 4. Reflection Loop (Lessons Learned)

Storing procedural memory.

```python
def reflect_on_failure(llm, task, error):
    prompt = f"""
    Task: {task}
    Error: {error}
    
    What went wrong? What should I do differently next time?
    Write a 'Lesson'.
    """
    lesson = llm.generate(prompt)
    return lesson

# Store lesson in Vector DB
# Next time task is similar to 'task', retrieve lesson.
```
