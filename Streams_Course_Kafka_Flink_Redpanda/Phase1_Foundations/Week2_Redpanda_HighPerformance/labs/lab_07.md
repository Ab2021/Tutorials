# Lab 07: Schema Registry in Redpanda

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use the built-in Schema Registry.
-   Register a schema via `curl`.

## Problem Statement
Redpanda exposes the Registry at port 8081.
1.  Create a JSON schema file `user.avsc`.
2.  Register it using `curl`.
3.  List subjects.

## Starter Code
```json
{
  "schema": "{"type": "string"}"
}
```

## Hints
<details>
<summary>Hint 1</summary>
The API endpoint is `POST /subjects/{subject}/versions`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Schema File
```json
{
  "schema": "{"type":"record","name":"User","fields":[{"name":"name","type":"string"}]}"
}
```

### Step 2: Register
```bash
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json"   --data @user.avsc   http://localhost:8081/subjects/user-value/versions
```

### Step 3: Verify
```bash
curl http://localhost:8081/subjects
```
</details>
