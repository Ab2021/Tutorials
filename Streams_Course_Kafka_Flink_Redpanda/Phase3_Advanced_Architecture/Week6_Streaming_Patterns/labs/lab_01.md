# Lab 01: Event Sourcing (Replay)

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Replay events to rebuild state.

## Problem Statement
1.  Create a list of events: `[("Deposit", 100), ("Withdraw", 50), ("Deposit", 20)]`.
2.  Replay them to calculate the final balance.
3.  Print the balance after each event.

## Starter Code
```python
events = [("Deposit", 100), ("Withdraw", 50)]
# Loop and update balance
```

## Hints
<details>
<summary>Hint 1</summary>
This is a simple fold/reduce operation.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
def run():
    events = [("Deposit", 100), ("Withdraw", 50), ("Deposit", 20)]
    balance = 0
    
    for event, amount in events:
        if event == "Deposit":
            balance += amount
        elif event == "Withdraw":
            balance -= amount
        print(f"Event: {event} {amount}, Balance: {balance}")

if __name__ == '__main__':
    run()
```
</details>
