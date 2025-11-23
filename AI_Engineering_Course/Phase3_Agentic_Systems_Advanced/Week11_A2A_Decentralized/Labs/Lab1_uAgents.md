# Lab 1: uAgents Network

## Objective
Create two autonomous agents that communicate over the Fetch.ai network.
Agent A (Alice) sends a message to Agent B (Bob).

## 1. Setup

```bash
poetry add uagents
```

## 2. The Agents (`agents.py`)

```python
from uagents import Agent, Context, Model

class Message(Model):
    text: str

# 1. Define Alice
alice = Agent(name="alice", seed="alice_recovery_phrase")

# 2. Define Bob
bob = Agent(name="bob", seed="bob_recovery_phrase", port=8001, endpoint=["http://127.0.0.1:8001/submit"])

@alice.on_interval(period=2.0)
async def send_message(ctx: Context):
    await ctx.send(bob.address, Message(text="Hello Bob!"))

@bob.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from {sender}: {msg.text}")

# 3. Run (In separate terminals usually, but here we simulate)
# In reality, run `python alice.py` and `python bob.py`
```

## 3. Running the Lab
1.  Create `alice.py` and `bob.py`.
2.  Run Bob first (he needs to listen).
3.  Run Alice.
4.  Watch Bob's logs.

## 4. Submission
Submit the logs showing the message receipt.
