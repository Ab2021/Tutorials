# Lab 14: Redpanda Security (SASL/SCRAM)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Enable SASL authentication.
-   Create users and ACLs.

## Problem Statement
1.  Enable `enable_sasl: true` in `redpanda.yaml`.
2.  Create a superuser `admin`.
3.  Try to access without auth (should fail).
4.  Access with auth.

## Starter Code
```yaml
redpanda:
  enable_sasl: true
  superusers: ["admin"]
```

## Hints
<details>
<summary>Hint 1</summary>
You need to pass `--user` and `--password` to `rpk` commands.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Create User
```bash
rpk acl user create admin -p secret
```

### Access
```bash
rpk cluster info --user admin --password secret --sasl-mechanism SCRAM-SHA-256
```
</details>
