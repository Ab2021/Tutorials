# Lab 09: Broadcast State

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement the Broadcast Pattern.
-   Dynamic Rules.

## Problem Statement
Stream 1: `Transactions (Item, Price)`.
Stream 2: `Thresholds (Item, MaxPrice)`.
Requirement: Alert if Price > MaxPrice. MaxPrice is updated dynamically via Stream 2.

## Starter Code
```python
rule_state_desc = MapStateDescriptor("rules", Types.STRING(), Types.INT())
broadcast_stream = rules.broadcast(rule_state_desc)

tx.connect(broadcast_stream).process(MyBroadcastFunction())
```

## Hints
<details>
<summary>Hint 1</summary>
`process_broadcast_element` updates the state. `process_element` reads it (read-only).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class AlertFunction(BroadcastProcessFunction):
    def process_broadcast_element(self, value, ctx):
        # Update Rule
        item, threshold = value
        ctx.get_broadcast_state(rule_state_desc).put(item, threshold)

    def process_element(self, value, ctx, out):
        # Check Rule
        item, price = value
        threshold = ctx.get_broadcast_state(rule_state_desc).get(item)
        if threshold and price > threshold:
            out.collect(f"Alert: {item} price {price} > {threshold}")
```
</details>
