# Day 71: Agent-to-Agent (A2A) Communication Patterns
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Modern FIPA-style Protocol

We will implement a simplified version of FIPA-ACL using JSON.

**Structure:**
```json
{
  "sender": "agent_a",
  "receiver": "agent_b",
  "performative": "REQUEST",
  "content": "buy_stock('AAPL', 10)",
  "ontology": "stock_trading_v1",
  "reply_with": "msg_123",
  "language": "json"
}
```

**Code (`mailbox.py`):**

```python
import queue
import time
import threading
import json

class AgentMailbox:
    def __init__(self, name):
        self.name = name
        self.inbox = queue.Queue()
        self.directory = {} # Mock directory

    def send(self, receiver_name, performative, content, reply_with=None):
        msg = {
            "sender": self.name,
            "receiver": receiver_name,
            "performative": performative,
            "content": content,
            "timestamp": time.time(),
            "reply_with": reply_with
        }
        # In a real system, this would be a network call
        receiver = self.directory.get(receiver_name)
        if receiver:
            receiver.inbox.put(msg)
            print(f"[{self.name} -> {receiver_name}] {performative}: {content}")
        else:
            print(f"Error: Agent {receiver_name} not found.")

    def receive(self):
        if not self.inbox.empty():
            return self.inbox.get()
        return None

    def loop(self):
        while True:
            msg = self.receive()
            if msg:
                self.handle_message(msg)
            time.sleep(0.1)

    def handle_message(self, msg):
        # To be implemented by subclasses
        pass

# Example Usage
class TraderAgent(AgentMailbox):
    def handle_message(self, msg):
        if msg["performative"] == "REQUEST":
            print(f"{self.name}: Received request to {msg['content']}")
            # Logic to execute trade...
            self.send(msg["sender"], "INFORM", "Trade Executed", reply_with=msg["reply_with"])

class UserAgent(AgentMailbox):
    def handle_message(self, msg):
        if msg["performative"] == "INFORM":
            print(f"{self.name}: Task Complete! {msg['content']}")

# Setup
trader = TraderAgent("Trader")
user = UserAgent("User")

# Link them (Mock Network)
trader.directory = {"User": user}
user.directory = {"Trader": trader}

# Start Threads
t1 = threading.Thread(target=trader.loop)
t2 = threading.Thread(target=user.loop)
t1.daemon = True
t2.daemon = True
t1.start()
t2.start()

# Trigger
user.send("Trader", "REQUEST", "Buy 10 AAPL", reply_with="req_1")
time.sleep(1)
```

### The "Contract Net" Protocol (CNP)

A standard pattern for task allocation.
1.  **Manager** sends `CFP` (Call for Proposal) to 3 Workers.
2.  **Workers** send `PROPOSE` (Price/Time) or `REFUSE`.
3.  **Manager** selects best proposal and sends `ACCEPT_PROPOSAL`.
4.  **Worker** performs task and sends `INFORM` (Result).

This allows for dynamic, market-based task distribution.

### Ontology Alignment with LLMs

In the old days, you needed strict XML schemas.
Today, we use LLMs to translate.
*   Agent A sends JSON: `{"price": 100}`.
*   Agent B expects XML: `<cost>100</cost>`.
*   **Translator Layer:** An LLM intercepts the message: "Translate this JSON to the XML schema expected by Agent B."

### Summary

*   **Performatives:** The "Verb" of the message (Request, Inform, Refuse).
*   **Content:** The "Noun".
*   **Mailbox:** The mechanism for async delivery.
By structuring communication this way, we avoid the ambiguity of "just chatting".
