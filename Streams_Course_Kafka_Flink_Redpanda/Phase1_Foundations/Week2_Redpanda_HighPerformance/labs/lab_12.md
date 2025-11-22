# Lab 12: Maintenance Mode

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Perform a rolling upgrade simulation.
-   Drain a node.

## Problem Statement
1.  Enable maintenance mode on Node 1.
2.  Verify that leaderships are transferred to other nodes.
3.  Disable maintenance mode.

## Starter Code
```bash
rpk cluster maintenance enable <node-id>
```

## Hints
<details>
<summary>Hint 1</summary>
Use `rpk cluster status` to find the Node ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# Get ID
rpk cluster status

# Enable
rpk cluster maintenance enable 1

# Check status (should say "draining" or "finished")
rpk cluster maintenance status

# Disable
rpk cluster maintenance disable 1
```
</details>
