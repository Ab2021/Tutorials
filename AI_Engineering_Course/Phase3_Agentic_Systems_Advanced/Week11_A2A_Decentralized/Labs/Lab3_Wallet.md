# Lab 3: Agent Wallet

## Objective
Agents need to pay for services.
Implement a **Mock Crypto Wallet**.

## 1. The Wallet (`wallet.py`)

```python
import hashlib
import json

class Wallet:
    def __init__(self, address, balance=100):
        self.address = address
        self.balance = balance
        
    def sign(self, message):
        # Mock signature
        return hashlib.sha256(f"{self.address}:{message}".encode()).hexdigest()
        
    def transfer(self, to_wallet, amount):
        if self.balance >= amount:
            self.balance -= amount
            to_wallet.balance += amount
            return True
        return False

alice = Wallet("0xAlice")
bob = Wallet("0xBob")

print(f"Alice: {alice.balance}, Bob: {bob.balance}")
alice.transfer(bob, 10)
print(f"Alice: {alice.balance}, Bob: {bob.balance}")
```

## 2. Analysis
In a real decentralized network (like Fetch.ai), this would use actual blockchain transactions.

## 3. Submission
Submit the final balances after 3 transactions.
