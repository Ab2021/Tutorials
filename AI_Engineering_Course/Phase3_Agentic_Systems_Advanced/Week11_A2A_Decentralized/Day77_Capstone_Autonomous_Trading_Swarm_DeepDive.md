# Day 77: Capstone: Building an Autonomous Trading Swarm
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing the Trading Swarm

We will build a simplified version in Python using `asyncio`.

**Code (`swarm.py`):**

```python
import asyncio
import random
import json

# --- Shared Ledger (The "Blockchain") ---
class Ledger:
    def __init__(self):
        self.balances = {"Trader": 1000, "MM": 1000}
        self.assets = {"Trader": 0, "MM": 100} # MM has 100 stocks
        
    def transfer(self, sender, receiver, amount):
        self.balances[sender] -= amount
        self.balances[receiver] += amount
        
    def transfer_asset(self, sender, receiver, amount):
        self.assets[sender] -= amount
        self.assets[receiver] += amount

ledger = Ledger()

# --- Agents ---
class Agent:
    def __init__(self, name):
        self.name = name
        self.inbox = asyncio.Queue()
        
    async def send(self, receiver, msg):
        await receiver.inbox.put((self, msg))
        
    async def run(self):
        while True:
            sender, msg = await self.inbox.get()
            await self.handle(sender, msg)

class MarketMaker(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.price = 100
        
    async def handle(self, sender, msg):
        if msg["type"] == "QUOTE_REQ":
            # Simple logic: Random walk price
            self.price += random.choice([-1, 1])
            quote = {
                "type": "QUOTE_RESP", 
                "bid": self.price - 1, 
                "ask": self.price + 1
            }
            await self.send(sender, quote)
            
        elif msg["type"] == "BUY_ORDER":
            cost = msg["price"]
            print(f"[MM] Selling to {sender.name} at {cost}")
            ledger.transfer(sender.name, self.name, cost)
            ledger.transfer_asset(self.name, sender.name, 1)
            await self.send(sender, {"type": "ORDER_FILLED"})

class Trader(Agent):
    def __init__(self, name, mm):
        super().__init__(name)
        self.mm = mm
        
    async def run(self):
        # Active Loop
        while True:
            await asyncio.sleep(1)
            # 1. Request Quote
            await self.send(self.mm, {"type": "QUOTE_REQ"})
            
            # 2. Wait for Response
            sender, msg = await self.inbox.get()
            if msg["type"] == "QUOTE_RESP":
                price = msg["ask"]
                print(f"[Trader] Price is {price}. Thinking...")
                
                # 3. Decide (Buy if cheap)
                if price < 105:
                    print(f"[Trader] Buying at {price}!")
                    await self.send(self.mm, {"type": "BUY_ORDER", "price": price})
                    
                    # Wait for fill
                    _, fill_msg = await self.inbox.get()
                    if fill_msg["type"] == "ORDER_FILLED":
                        print(f"[Trader] Trade Complete. Balance: {ledger.balances['Trader']}")

# --- Simulation ---
async def main():
    mm = MarketMaker("MM")
    trader = Trader("Trader", mm)
    
    # Run concurrently
    await asyncio.gather(
        mm.run(),
        trader.run()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Adding the Analyst (Signal Service)

```python
class Analyst(Agent):
    async def run(self):
        while True:
            await asyncio.sleep(2)
            signal = random.choice(["BULLISH", "BEARISH"])
            # Broadcast to subscribers (simplified)
            # await subscriber.inbox.put(...)
            print(f"[Analyst] Market Sentiment: {signal}")
```

### The "Flash Crash" Scenario

If we add 10 Traders and they all listen to the same Analyst:
1.  Analyst says "BULLISH".
2.  10 Traders buy simultaneously.
3.  MM runs out of inventory (Liquidity Crisis).
4.  Price spikes to infinity (or MM rejects orders).
This simulates real-world market dynamics.

### Summary

This simulation captures the essence of A2A:
*   **Async:** Agents act at their own pace.
*   **State:** Ledger tracks truth.
*   **Protocol:** Defined message types (`QUOTE_REQ`, `BUY_ORDER`).
*   **Emergence:** Price discovery happens through interaction.
