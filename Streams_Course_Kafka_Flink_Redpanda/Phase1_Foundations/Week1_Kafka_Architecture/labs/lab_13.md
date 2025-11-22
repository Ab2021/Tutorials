# Lab 13: ACLs & Security

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Understand Kafka ACLs (Access Control Lists).
-   Restrict access to a topic.

## Problem Statement
*Note: This requires a Kafka cluster configured with an Authorizer (e.g., `SimpleAclAuthorizer`). For this lab, we will assume the environment is set up or we will use the CLI to simulate the commands.*

1.  Create a user `alice`.
2.  Deny `alice` from reading topic `secret`.
3.  Verify that `alice` cannot consume.

## Starter Code
```bash
kafka-acls --bootstrap-server localhost:9092 --add --allow-principal User:bob --operation Read --topic secret
```

## Hints
<details>
<summary>Hint 1</summary>
By default, if no ACLs exist, access might be allowed (depending on `allow.everyone.if.no.acl.found`).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Command to Add ACL
```bash
# Allow Bob to read/write
kafka-acls --bootstrap-server localhost:9092   --add   --allow-principal User:bob   --operation Read   --operation Write   --topic secret

# Deny Alice (if implicit allow is on, or just don't add her)
kafka-acls --bootstrap-server localhost:9092   --add   --deny-principal User:alice   --operation All   --topic secret
```
</details>
