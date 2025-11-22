# Lab 09: Consumer Offsets

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand how offsets are committed.
-   Manually commit offsets.
-   Seek to a specific offset (replay).

## Problem Statement
1.  Produce 10 messages to a topic.
2.  Consume them and print them.
3.  Use `consumer.seek()` to rewind to offset 5.
4.  Consume again. You should see messages 5-9 again.

## Starter Code
```python
from confluent_kafka import TopicPartition

# ... inside consumer loop ...
# Rewind logic
tp = TopicPartition(topic, partition, 5)
consumer.assign([tp])
consumer.seek(tp)
```

## Hints
<details>
<summary>Hint 1</summary>
You cannot `seek()` on a subscribed consumer directly if you are using dynamic partition assignment (`subscribe()`). You usually need to do it inside the `on_assign` callback or use `assign()` (manual assignment) instead of `subscribe()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from confluent_kafka import Consumer, TopicPartition

def run_seek():
    c = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'rewind-group',
        'auto.offset.reset': 'earliest'
    })
    
    # Manual assignment is easier for seek demos
    tp = TopicPartition('user-clicks', 0)
    c.assign([tp])
    
    # Read first 10
    for _ in range(10):
        msg = c.poll(1.0)
        if msg: print(f"Read offset {msg.offset()}")
        
    print("Rewinding to offset 5...")
    tp.offset = 5
    c.seek(tp)
    
    # Read again
    while True:
        msg = c.poll(1.0)
        if msg is None: break
        print(f"Re-read offset {msg.offset()}")
        
    c.close()

if __name__ == "__main__":
    run_seek()
```
</details>
