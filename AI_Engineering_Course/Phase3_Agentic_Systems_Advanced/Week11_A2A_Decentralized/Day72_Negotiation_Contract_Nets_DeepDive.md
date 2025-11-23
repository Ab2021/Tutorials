# Day 72: Negotiation & Contract Nets
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Contract Net Protocol (Python)

We will build a Manager that hires a Worker to process a file.

```python
import random
import time

class Agent:
    def __init__(self, name):
        self.name = name

class Manager(Agent):
    def __init__(self, name, workers):
        super().__init__(name)
        self.workers = workers

    def announce_task(self, task_difficulty):
        print(f"[{self.name}] Announcing task (Difficulty: {task_difficulty})")
        bids = []
        
        # Phase 1 & 2: Broadcast and Collect Bids
        for worker in self.workers:
            bid = worker.receive_cfp(task_difficulty)
            if bid:
                bids.append((worker, bid))
        
        if not bids:
            print("No bids received.")
            return

        # Phase 3: Award
        # Strategy: Lowest cost wins
        bids.sort(key=lambda x: x[1]) 
        winner, winning_price = bids[0]
        
        print(f"[{self.name}] Winner is {winner.name} with bid ${winning_price}")
        
        # Notify Winner and Losers
        winner.receive_accept(task_difficulty)
        for worker, _ in bids[1:]:
            worker.receive_reject()

class Worker(Agent):
    def __init__(self, name, skill_level, hourly_rate):
        super().__init__(name)
        self.skill = skill_level
        self.rate = hourly_rate

    def receive_cfp(self, difficulty):
        # Logic: Can I do this?
        if self.skill < difficulty:
            print(f"[{self.name}] Refusing (Skill too low)")
            return None
        
        # Logic: Calculate Bid
        estimated_hours = difficulty * 1.5 # Simple heuristic
        bid_price = estimated_hours * self.rate
        
        # Add some randomness (market fluctuation)
        bid_price += random.uniform(-5, 5)
        
        print(f"[{self.name}] Proposing ${bid_price:.2f}")
        return bid_price

    def receive_accept(self, task):
        print(f"[{self.name}] Hooray! I got the job. Working...")
        time.sleep(1)
        print(f"[{self.name}] Done.")

    def receive_reject(self):
        print(f"[{self.name}] Darn, I lost.")

# Simulation
w1 = Worker("Junior", skill_level=5, hourly_rate=20)
w2 = Worker("Senior", skill_level=10, hourly_rate=100)
w3 = Worker("Mid", skill_level=7, hourly_rate=50)

manager = Manager("Boss", [w1, w2, w3])

# Scenario 1: Easy Task
manager.announce_task(difficulty=3)
# Junior should win (lowest rate)

print("-" * 20)

# Scenario 2: Hard Task
manager.announce_task(difficulty=8)
# Junior refuses. Mid or Senior wins.
```

### LLM-Based Bargaining

For complex negotiations, we use LLMs.

```python
PROMPT = """
You are a Buyer negotiating the price of a car.
Your goal: Buy for under $10,000.
Current Offer: {offer}
History: {history}

Decide:
1. ACCEPT
2. REJECT (Walk away)
3. COUNTER <price> <reasoning>
"""

def negotiate(buyer_llm, seller_llm):
    price = 15000 # Initial Ask
    history = []
    
    for round in range(5):
        # Buyer Turn
        buyer_resp = buyer_llm.invoke(PROMPT.format(offer=price, history=history))
        history.append(f"Buyer: {buyer_resp}")
        
        if "ACCEPT" in buyer_resp:
            return price
        
        # Extract Counter
        # ... parsing logic ...
        
        # Seller Turn
        # ... similar logic ...
```

### Summary

*   **CNP:** Good for structured, well-defined tasks.
*   **Auctions:** Good for scarce resources.
*   **LLM Bargaining:** Good for nuanced, human-like negotiation (e.g., customer support refunds).
