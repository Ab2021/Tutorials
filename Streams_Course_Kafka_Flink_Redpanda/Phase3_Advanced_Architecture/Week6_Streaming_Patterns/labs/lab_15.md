# Lab 15: Command Sourcing

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement Command Sourcing.

## Problem Statement
Stream of `Commands` (e.g., "Transfer").
Process function validates command and emits `Event` ("Transferred") or `Failure` ("InsufficientFunds").

## Starter Code
```python
class CommandHandler(KeyedProcessFunction):
    def process_element(self, cmd, ctx, out):
        # check balance
        # emit event
```

## Hints
<details>
<summary>Hint 1</summary>
This is the core of the Event Sourcing Write Model.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class Bank(KeyedProcessFunction):
    def process_element(self, cmd, ctx, out):
        current_balance = self.balance_state.value() or 0
        if cmd['amount'] <= current_balance:
            self.balance_state.update(current_balance - cmd['amount'])
            out.collect(f"Transferred {cmd['amount']}")
        else:
            ctx.output(failure_tag, "Insufficient Funds")
```
</details>
