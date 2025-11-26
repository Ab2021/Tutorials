# Lab: Day 50 - Event Sourcing

## Goal
Build a Bank Account using Event Sourcing.

## Step 1: The Code (`bank.py`)

```python
from dataclasses import dataclass
from typing import List

# 1. Events
@dataclass
class Event:
    pass

@dataclass
class Deposited(Event):
    amount: int

@dataclass
class Withdrew(Event):
    amount: int

# 2. Aggregate (The Domain Object)
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.changes: List[Event] = []

    # Command: Deposit
    def deposit(self, amount):
        event = Deposited(amount)
        self.apply(event)
        self.changes.append(event)

    # Command: Withdraw
    def withdraw(self, amount):
        if self.balance < amount:
            raise ValueError("Insufficient funds")
        event = Withdrew(amount)
        self.apply(event)
        self.changes.append(event)

    # Apply Event (Rebuild State)
    def apply(self, event: Event):
        if isinstance(event, Deposited):
            self.balance += event.amount
        elif isinstance(event, Withdrew):
            self.balance -= event.amount

# 3. Event Store (Simulation)
event_store = []

# Scenario
account = BankAccount()
account.deposit(100)
account.withdraw(30)
account.deposit(50)

print(f"Current Balance: {account.balance}") # 120

# Save to Store
event_store.extend(account.changes)

# 4. Replay (Time Travel)
print("\n--- Replaying ---")
replayed_account = BankAccount()
for event in event_store:
    replayed_account.apply(event)
    print(f"Applied {event}, Balance: {replayed_account.balance}")
```

## Step 2: Run It
`python bank.py`

## Challenge: Projection
Create a **Read Model**.
1.  Create a `TransactionHistory` class.
2.  Listen to events.
3.  Build a list of strings: `["+100", "-30", "+50"]`.
4.  Print the history.
