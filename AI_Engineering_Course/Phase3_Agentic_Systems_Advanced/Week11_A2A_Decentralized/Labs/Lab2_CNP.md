# Lab 2: Contract Net Protocol (CNP)

## Objective
Implement a **Task Auction**.
Manager broadcasts a task. Workers bid. Manager awards task to lowest bidder.

## 1. The Protocol

1.  **CFP (Call for Proposal):** Manager -> All Workers.
2.  **Propose:** Worker -> Manager ("I can do it for $10").
3.  **Accept:** Manager -> Best Worker ("You win").
4.  **Reject:** Manager -> Others ("You lose").

## 2. Implementation (`cnp.py`)

(Simplified Python simulation without network overhead)

```python
class Worker:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost

    def bid(self, task):
        return self.cost

class Manager:
    def __init__(self, workers):
        self.workers = workers

    def auction(self, task):
        print(f"Manager: Auctioning task '{task}'")
        bids = []
        for w in self.workers:
            bid = w.bid(task)
            print(f"{w.name} bids ${bid}")
            bids.append((w, bid))
        
        # Select Winner
        winner, price = min(bids, key=lambda x: x[1])
        print(f"Winner: {winner.name} with ${price}")

# Run
w1 = Worker("Worker1", 10)
w2 = Worker("Worker2", 5)
w3 = Worker("Worker3", 20)

mgr = Manager([w1, w2, w3])
mgr.auction("Clean the floor")
```

## 3. Challenge
*   **Capabilities:** Add a `capabilities` list to workers. Only workers with the right capability (e.g., "cleaning") should bid.

## 4. Submission
Submit the auction log.
