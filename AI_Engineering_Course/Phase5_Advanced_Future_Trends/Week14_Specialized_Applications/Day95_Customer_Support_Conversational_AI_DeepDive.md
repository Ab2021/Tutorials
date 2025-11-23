# Day 95: Customer Support & Conversational AI
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a RAG Support Bot with History

We will build a bot that answers questions based on a manual, maintaining history.

```python
class SupportBot:
    def __init__(self, client, kb):
        self.client = client
        self.kb = kb # Dict of {topic: content}
        self.history = [{"role": "system", "content": "You are a helpful support agent."}]

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        
        # 1. Intent Classification
        intent = self.classify_intent(user_input)
        
        # 2. Retrieval
        context = ""
        if intent in self.kb:
            context = f"Relevant Policy: {self.kb[intent]}"
            
        # 3. Generation
        prompt = f"""
        Context: {context}
        History: {self.history[-5:]} # Last 5 turns
        
        Answer the user. If you don't know, say "ESCALATE".
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        
        self.history.append({"role": "assistant", "content": response})
        return response

    def classify_intent(self, text):
        # Simple keyword matching (Use an LLM or BERT in production)
        if "refund" in text.lower(): return "refund_policy"
        if "password" in text.lower(): return "password_reset"
        return "general"

# Usage
kb = {"refund_policy": "Refunds are allowed within 30 days."}
bot = SupportBot(client, kb)
print(bot.chat("Can I get my money back?"))
```

### Detecting Frustration (Sentiment)

```python
def check_sentiment(text):
    prompt = f"Rate the anger level of this text from 0 (Calm) to 10 (Furious): '{text}'"
    score = int(client.chat.completions.create(...).content)
    return score

# In loop:
if check_sentiment(user_input) > 7:
    return "Transferring to human..."
```

### Evaluation: RAGAS

How do you know the bot is good?
**RAGAS (Retrieval Augmented Generation Assessment):**
*   **Faithfulness:** Is the answer supported by the context?
*   **Answer Relevance:** Does it answer the user's question?
*   **Context Precision:** Did we retrieve the right document?

### Summary

*   **State:** You must store the `history` object (usually in Redis).
*   **Latency:** Support bots must be fast. Use Streaming.
*   **Fallback:** Always have a path to a human.
