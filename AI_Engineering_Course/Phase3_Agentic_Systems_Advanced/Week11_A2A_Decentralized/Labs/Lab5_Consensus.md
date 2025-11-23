# Lab 5: Swarm Consensus

## Objective
A swarm must agree on a decision.
Implement **Majority Voting**.

## 1. The Vote (`vote.py`)

```python
import random

agents = [f"Agent_{i}" for i in range(5)]
options = ["Go Left", "Go Right"]

votes = {}
for agent in agents:
    # Random vote
    choice = random.choice(options)
    votes[agent] = choice
    print(f"{agent} voted: {choice}")

# Tally
counts = {}
for v in votes.values():
    counts[v] = counts.get(v, 0) + 1

winner = max(counts, key=counts.get)
print(f"Consensus: {winner} with {counts[winner]} votes")
```

## 2. Challenge
Implement **Weighted Voting** based on agent reputation.

## 3. Submission
Submit the consensus result for a weighted vote.
