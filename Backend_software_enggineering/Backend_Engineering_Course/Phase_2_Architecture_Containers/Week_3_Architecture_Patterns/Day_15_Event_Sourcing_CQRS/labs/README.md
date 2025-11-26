# Lab: Day 15 - Event Sourcing from Scratch

## Goal
Build a tiny Event Sourcing engine for a Bank Account. You will implement the "Append Event" and "Replay" logic.

## Directory Structure
```
day15/
├── bank.py
└── README.md
```

## Step 1: The Code (`bank.py`)

```python
import json
from datetime import datetime

# 1. The Event Store (In-Memory)
event_store = []

# 2. Events
class Event:
    def __init__(self, aggregate_id, type, data):
        self.aggregate_id = aggregate_id
        self.type = type
        self.data = data
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return self.__dict__

# 3. The Aggregate (Bank Account)
class BankAccount:
    def __init__(self, id):
        self.id = id
        self.balance = 0
        self.is_active = False

    # Apply Event: Updates state based on event
    def apply(self, event):
        if event.type == "AccountOpened":
            self.is_active = True
            self.balance = event.data['opening_balance']
        elif event.type == "Deposited":
            self.balance += event.data['amount']
        elif event.type == "Withdrawn":
            self.balance -= event.data['amount']

    # Rehydrate: Rebuild state from history
    def load_from_history(self, events):
        for event in events:
            if event.aggregate_id == self.id:
                self.apply(event)

# 4. Commands (Business Logic)
def open_account(account_id, opening_balance):
    # Validation
    if opening_balance < 0:
        raise ValueError("Cannot open with negative balance")
    
    # Create Event
    event = Event(account_id, "AccountOpened", {"opening_balance": opening_balance})
    event_store.append(event)
    print(f"📝 Event: AccountOpened")

def deposit(account_id, amount):
    # Load current state to validate
    account = BankAccount(account_id)
    account.load_from_history(event_store)
    
    if not account.is_active:
        raise ValueError("Account not active")
    
    event = Event(account_id, "Deposited", {"amount": amount})
    event_store.append(event)
    print(f"📝 Event: Deposited ${amount}")

def withdraw(account_id, amount):
    account = BankAccount(account_id)
    account.load_from_history(event_store)
    
    if account.balance < amount:
        raise ValueError("Insufficient funds")
    
    event = Event(account_id, "Withdrawn", {"amount": amount})
    event_store.append(event)
    print(f"📝 Event: Withdrawn ${amount}")

# 5. Query (Projection)
def get_balance(account_id):
    account = BankAccount(account_id)
    account.load_from_history(event_store)
    return account.balance

# --- Simulation ---
if __name__ == "__main__":
    print("--- Banking Sim ---")
    
    # User flows
    open_account("acc_1", 100)
    deposit("acc_1", 50)
    withdraw("acc_1", 30)
    
    # Check State
    print(f"\n💰 Current Balance: ${get_balance('acc_1')}") # Should be 120
    
    # Audit Trail
    print("\n📜 Audit Log:")
    for e in event_store:
        print(f" - {e.timestamp}: {e.type} {e.data}")
        
    # Time Travel (What was balance before withdrawal?)
    print("\n⏳ Time Travel (First 2 events):")
    past_account = BankAccount("acc_1")
    past_account.load_from_history(event_store[:2])
    print(f"   Balance was: ${past_account.balance}")
```

## Step 2: Run It
`python bank.py`

## Challenge
Implement a **Snapshot**.
*   Modify `BankAccount` to save its state to a `snapshots` dict after every 5 events.
*   Modify `load_from_history` to look for a snapshot first, then replay only subsequent events.
