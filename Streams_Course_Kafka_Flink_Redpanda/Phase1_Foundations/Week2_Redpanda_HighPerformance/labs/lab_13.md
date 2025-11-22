# Lab 13: Partition Rebalancing

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Trigger the partition balancer.
-   Understand `on-demand` vs `continuous` balancing.

## Problem Statement
1.  Create a topic with 30 partitions.
2.  Observe they are spread evenly.
3.  Kill one node. Wait.
4.  Bring the node back.
5.  Partitions might not move back immediately. Trigger a rebalance.

## Starter Code
```bash
rpk cluster partitions balancer-status
```

## Hints
<details>
<summary>Hint 1</summary>
Redpanda's balancer is usually continuous. You can tweak `partition_autobalancing_mode`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# Check status
rpk cluster partitions balancer-status

# Force rebalance (if needed)
# Usually Redpanda does this automatically, but you can inspect the movement:
rpk cluster partitions movement-cancel --all # To stop it
```
</details>
