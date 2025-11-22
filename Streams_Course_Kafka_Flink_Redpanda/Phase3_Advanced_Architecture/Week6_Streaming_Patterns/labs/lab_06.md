# Lab 06: Broadcast Join

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Broadcast Join.

## Problem Statement
Stream A: `Transactions`.
Stream B: `CurrencyRates` (Broadcast).
Join to convert Transaction Amount to USD.

## Starter Code
```python
# See Week 4 Lab 09 (Broadcast State)
```

## Hints
<details>
<summary>Hint 1</summary>
Store rates in Broadcast MapState.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class CurrencyConverter(BroadcastProcessFunction):
    def process_broadcast_element(self, rate, ctx):
        ctx.get_broadcast_state(desc).put(rate['currency'], rate['val'])

    def process_element(self, tx, ctx, out):
        rate = ctx.get_broadcast_state(desc).get(tx['currency'])
        if rate:
            out.collect(tx['amount'] * rate)
```
</details>
