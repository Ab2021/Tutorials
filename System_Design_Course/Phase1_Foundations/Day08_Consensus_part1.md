# Day 8 Deep Dive: Split Brain

## 1. What is Split Brain?
*   **Scenario:** Network partition divides the cluster into two halves (e.g., 5 nodes split into 2 and 3).
*   **Risk:** Both sides might elect a leader. Both accept writes. Data corruption.
*   **Solution:** **Quorum (Majority).**
    *   To be a leader, you need votes from $> N/2$ nodes.
    *   If $N=5$, you need 3 votes.
    *   The partition with 2 nodes cannot elect a leader. The partition with 3 nodes can.

## 2. Fencing Tokens
*   **Scenario:** Old leader (GC pause) wakes up and thinks it's still leader.
*   **Risk:** It sends a write to storage.
*   **Solution:** Fencing Token (Monotonically increasing ID).
    *   Storage rejects writes with older tokens.

## 3. Leader Election Code (Conceptual Python)
```python
import time
import random

class Node:
    def __init__(self, id, peers):
        self.id = id
        self.peers = peers
        self.state = "FOLLOWER"
        self.term = 0
        self.votes = 0

    def start_election(self):
        self.state = "CANDIDATE"
        self.term += 1
        self.votes = 1 # Vote for self
        print(f"Node {self.id} starting election for term {self.term}")
        
        for peer in self.peers:
            self.request_vote(peer)

    def request_vote(self, peer):
        # Simulate network call
        if random.random() > 0.5:
            self.votes += 1
            
    def check_result(self):
        if self.votes > len(self.peers) / 2:
            self.state = "LEADER"
            print(f"Node {self.id} is LEADER!")
```
