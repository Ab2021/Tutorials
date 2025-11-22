# Lab 06: Log Compaction

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand how log compaction works.
-   Configure a topic for compaction.
-   Verify that old values for the same key are deleted.

## Problem Statement
1.  Create a topic `user-states` with `cleanup.policy=compact` and `segment.ms=100` (to trigger compaction quickly).
2.  Produce: `Key=A, Val=1`, then `Key=A, Val=2`, then `Key=A, Val=3`.
3.  Consume from the beginning.
4.  **Expected Result**: Eventually, you should only see `Key=A, Val=3`. (Note: You might see Val=2 momentarily before the cleaner thread runs).

## Starter Code
```python
# Topic Config
config = {
    'cleanup.policy': 'compact',
    'segment.ms': '100',
    'min.cleanable.dirty.ratio': '0.01' # Aggressive cleaning
}
```

## Hints
<details>
<summary>Hint 1</summary>
Log compaction is not instant. The "active segment" is never compacted. You need to roll the segment (hence low `segment.ms`) to make it eligible for compaction.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Compacted Topic
```python
from confluent_kafka.admin import AdminClient, NewTopic

admin = AdminClient({'bootstrap.servers': 'localhost:9092'})
topic = NewTopic("user-states", num_partitions=1, replication_factor=1,
                 config={
                     'cleanup.policy': 'compact',
                     'segment.ms': '100',
                     'min.cleanable.dirty.ratio': '0.01'
                 })
admin.create_topics([topic])
```

### Step 2: Produce Updates
```python
p = Producer({'bootstrap.servers': 'localhost:9092'})
key = b"user1"
p.produce("user-states", key=key, value=b"login")
p.produce("user-states", key=key, value=b"update_profile")
p.produce("user-states", key=key, value=b"logout") # This should be the final state
p.flush()
```

### Step 3: Consume
Wait a few seconds for the cleaner to run.
```python
c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'state-reader',
    'auto.offset.reset': 'earliest'
})
c.subscribe(["user-states"])

# You should eventually see only "logout" for user1, 
# though you might see all 3 if you consume too fast before compaction.
```
</details>
