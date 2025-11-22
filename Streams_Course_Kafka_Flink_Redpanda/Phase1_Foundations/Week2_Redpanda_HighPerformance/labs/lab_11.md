# Lab 11: Shadow Indexing Fetch

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Verify that data can be read from S3 transparently.

## Problem Statement
(Requires Lab 05 setup).
1.  Set retention on `archived-topic` to 1 minute (local).
2.  Produce data. Wait 2 minutes.
3.  Local data should be deleted, but S3 data remains.
4.  Consume from offset 0. Redpanda should fetch from S3.

## Starter Code
```bash
rpk topic alter-config archived-topic --set retention.ms=60000
```

## Hints
<details>
<summary>Hint 1</summary>
Watch the Redpanda logs for "Downloading segment from remote".
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Commands
```bash
# Set short local retention
rpk topic alter-config archived-topic --set retention.local.target.bytes=100

# Produce
for i in {1..1000}; do echo "msg-$i" | rpk topic produce archived-topic; done

# Wait... then Consume
rpk topic consume archived-topic --offset oldest
```
If you see "msg-1", it worked (fetched from S3).
</details>
