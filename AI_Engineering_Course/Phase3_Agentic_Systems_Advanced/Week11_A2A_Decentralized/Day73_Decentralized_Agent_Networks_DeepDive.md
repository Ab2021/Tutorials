# Day 73: Decentralized Agent Networks (uAgents)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building a uAgent

We will build a simple "Ping-Pong" agent using the `uagents` library.

**Dependencies:**
`pip install uagents`

**Code (`agent.py`):**

```python
from uagents import Agent, Context, Model

# 1. Define Message Schema
class Ping(Model):
    text: str

class Pong(Model):
    text: str

# 2. Initialize Agent
# Seed ensures the address remains the same
alice = Agent(name="alice", seed="alice_recovery_phrase")
bob = Agent(name="bob", seed="bob_recovery_phrase")

print(f"Alice Address: {alice.address}")
print(f"Bob Address: {bob.address}")

# 3. Define Handlers
@alice.on_interval(period=2.0)
async def send_ping(ctx: Context):
    ctx.logger.info("Sending Ping to Bob...")
    await ctx.send(bob.address, Ping(text="Hello Bob"))

@alice.on_message(model=Pong)
async def handle_pong(ctx: Context, sender: str, msg: Pong):
    ctx.logger.info(f"Received Pong from {sender}: {msg.text}")

@bob.on_message(model=Ping)
async def handle_ping(ctx: Context, sender: str, msg: Ping):
    ctx.logger.info(f"Received Ping from {sender}: {msg.text}")
    await ctx.send(sender, Pong(text="Hello Alice"))

# 4. Run
if __name__ == "__main__":
    # In a real scenario, these would run in separate processes
    # For demo, uagents supports running multiple in one script via Bureau
    from uagents import Bureau
    bureau = Bureau()
    bureau.add(alice)
    bureau.add(bob)
    bureau.run()
```

### Integrating with LLMs

An AEA usually wraps an LLM.

```python
from uagents import Agent, Context, Model
from openai import OpenAI

client = OpenAI()
agent = Agent(name="gpt_agent", seed="gpt_seed")

class Query(Model):
    prompt: str

class Response(Model):
    answer: str

@agent.on_message(model=Query)
async def handle_query(ctx: Context, sender: str, msg: Query):
    # Call LLM
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": msg.prompt}]
    )
    answer = completion.choices[0].message.content
    
    # Send back
    await ctx.send(sender, Response(answer=answer))
```

### Registration in Almanac

To be discoverable, the agent must register.

```python
agent = Agent(
    name="service_agent",
    seed="seed",
    port=8000,
    endpoint=["http://127.0.0.1:8000/submit"]
)

# The library handles the registration handshake with the Almanac contract automatically on startup.
```

### Summary

*   **Models:** Pydantic classes define the protocol.
*   **Handlers:** Async functions process messages.
*   **Context:** Provides access to storage, logger, and wallet.
This is a micro-framework for building **Stateful, Addressable** agents.
