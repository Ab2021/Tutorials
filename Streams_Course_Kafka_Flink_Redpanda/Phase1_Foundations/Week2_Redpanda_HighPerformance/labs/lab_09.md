# Lab 09: Tuning Redpanda

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `rpk redpanda tune`.
-   Understand OS-level optimizations.

## Problem Statement
*Note: This requires running on Linux (not Docker Desktop on Mac/Windows).*
Run the autotuner to optimize for network and disk.
1.  Run `rpk redpanda tune help`.
2.  Enable `aio_events` and `disk_irq` tuners.

## Starter Code
```bash
rpk redpanda tune all
```

## Hints
<details>
<summary>Hint 1</summary>
This modifies system files (`/etc/sysctl.conf`). Run with sudo.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
sudo rpk redpanda tune disk_irq aio_events
```
Output will show which parameters were changed (e.g., IRQ affinity).
</details>
