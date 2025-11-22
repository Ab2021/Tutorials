# Lab 08: Admin API

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use the Redpanda Admin API (port 9644).
-   Manage users and config.

## Problem Statement
The Admin API allows operational control.
1.  Query the cluster health via the API.
2.  Create a user `admin` with password `secret` via the API.

## Starter Code
```bash
curl http://localhost:9644/v1/status
```

## Hints
<details>
<summary>Hint 1</summary>
User management endpoint is `/v1/security/users`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Check Status
```bash
curl http://localhost:9644/v1/cluster_view
```

### Create User
```bash
curl -X POST http://localhost:9644/v1/security/users   -H "Content-Type: application/json"   -d '{"username": "admin", "password": "secret", "algorithm": "SCRAM-SHA-256"}'
```
</details>
