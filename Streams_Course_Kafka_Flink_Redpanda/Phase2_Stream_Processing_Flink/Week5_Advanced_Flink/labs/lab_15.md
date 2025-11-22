# Lab 15: Reactive Mode

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Configure Reactive Mode.
-   Scale TaskManagers.

## Problem Statement
1.  Configure `scheduler-mode: reactive`.
2.  Deploy cluster.
3.  Scale TaskManagers from 1 to 2.
4.  Verify job rescales automatically.

## Starter Code
```yaml
jobmanager.scheduler: reactive
```

## Hints
<details>
<summary>Hint 1</summary>
Reactive mode requires Application Mode.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Config
```yaml
jobmanager.scheduler: reactive
```

### Scaling
```bash
kubectl scale deployment/flink-taskmanager --replicas=2
```
Watch the logs. The job should restart with parallelism 2.
</details>
