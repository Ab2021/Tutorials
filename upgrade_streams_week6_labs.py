import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase3_Advanced_Architecture\Week6_Streaming_Patterns\labs"

labs_content = {
    "lab_01.md": """# Lab 01: Event Sourcing (Replay)

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Replay events to rebuild state.

## Problem Statement
1.  Create a list of events: `[("Deposit", 100), ("Withdraw", 50), ("Deposit", 20)]`.
2.  Replay them to calculate the final balance.
3.  Print the balance after each event.

## Starter Code
```python
events = [("Deposit", 100), ("Withdraw", 50)]
# Loop and update balance
```

## Hints
<details>
<summary>Hint 1</summary>
This is a simple fold/reduce operation.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
def run():
    events = [("Deposit", 100), ("Withdraw", 50), ("Deposit", 20)]
    balance = 0
    
    for event, amount in events:
        if event == "Deposit":
            balance += amount
        elif event == "Withdraw":
            balance -= amount
        print(f"Event: {event} {amount}, Balance: {balance}")

if __name__ == '__main__':
    run()
```
</details>
""",
    "lab_02.md": """# Lab 02: CQRS Projection

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Separate Write and Read models.

## Problem Statement
Stream: `OrderCreated(id, user, amount)`.
Projection: Maintain a `UserSpent` table (User -> Total Amount).
Implement the projection using a Flink Map/Reduce job.

## Starter Code
```python
ds.key_by(lambda x: x['user']).reduce(...)
```

## Hints
<details>
<summary>Hint 1</summary>
The "Read Model" here is the state of the Reduce function.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
ds.key_by(lambda x: x['user']) \
  .reduce(lambda a, b: {'user': a['user'], 'amount': a['amount'] + b['amount']}) \
  .print()
```
</details>
""",
    "lab_03.md": """# Lab 03: Kappa Backfill

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate a Kappa Architecture backfill.

## Problem Statement
1.  Write a job that reads from a file (simulating Kafka history).
2.  Process all records.
3.  Switch to reading from a socket (simulating real-time).
*Note: In Flink, you can chain sources or use `HybridSource`.*

## Starter Code
```python
# HybridSource is complex in PyFlink.
# Simulate by reading file first, then socket.
```

## Hints
<details>
<summary>Hint 1</summary>
For this lab, just write a job that reads a file. The concept is that the *same code* runs on the file as on the stream.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Same logic for both
def process(ds):
    ds.map(lambda x: x.upper()).print()

# Backfill Job
env = StreamExecutionEnvironment.get_execution_environment()
ds = env.read_text_file("history.txt")
process(ds)
env.execute("Backfill")

# Realtime Job
# ds = env.socket_text_stream(...)
# process(ds)
```
</details>
""",
    "lab_04.md": """# Lab 04: Debezium JSON Parsing

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Parse Debezium JSON format.

## Problem Statement
Input: Debezium JSON string `{"before": null, "after": {"id": 1, "val": "A"}, "op": "c"}`.
Task: Extract the "after" state. Filter out deletes (`op="d"`).

## Starter Code
```python
import json
# ds.map(json.loads).filter(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Check `op` field. `c`=create, `u`=update, `d`=delete, `r`=read (snapshot).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import json

def parse_cdc(record):
    data = json.loads(record)
    if data['op'] != 'd':
        return data['after']
    return None

ds.map(parse_cdc).filter(lambda x: x is not None).print()
```
</details>
""",
    "lab_05.md": """# Lab 05: Stream Enrichment (Async)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate Async Enrichment.

## Problem Statement
Stream: `UserIDs`.
Enrichment: Call `fake_api(user_id)` which sleeps 0.1s and returns Name.
Use `AsyncDataStream` (or simulate with ThreadPool in Map if Async not available).

## Starter Code
```python
# PyFlink Async support requires specific setup.
# We will simulate the latency impact.
```

## Hints
<details>
<summary>Hint 1</summary>
Compare throughput of blocking `map` vs `async`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Conceptual Async Function
class AsyncEnricher(AsyncFunction):
    def async_invoke(self, input, result_future):
        # Call external API in thread
        val = call_api(input)
        result_future.complete([val])
```
</details>
""",
    "lab_06.md": """# Lab 06: Broadcast Join

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement Broadcast Join.

## Problem Statement
Stream A: `Transactions`.
Stream B: `CurrencyRates` (Broadcast).
Join to convert Transaction Amount to USD.

## Starter Code
```python
# See Week 4 Lab 09 (Broadcast State)
```

## Hints
<details>
<summary>Hint 1</summary>
Store rates in Broadcast MapState.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class CurrencyConverter(BroadcastProcessFunction):
    def process_broadcast_element(self, rate, ctx):
        ctx.get_broadcast_state(desc).put(rate['currency'], rate['val'])

    def process_element(self, tx, ctx, out):
        rate = ctx.get_broadcast_state(desc).get(tx['currency'])
        if rate:
            out.collect(tx['amount'] * rate)
```
</details>
""",
    "lab_07.md": """# Lab 07: Temporal Table Join (SQL)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use SQL Temporal Join.

## Problem Statement
Join `Orders` with `Rates` using `FOR SYSTEM_TIME AS OF`.
(Requires setting up versioned table).

## Starter Code
```sql
SELECT * FROM Orders o
JOIN Rates FOR SYSTEM_TIME AS OF o.ts r
ON o.curr = r.curr
```

## Hints
<details>
<summary>Hint 1</summary>
Rates table must have a primary key and watermark.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
t_env.execute_sql(\"\"\"
    SELECT o.id, o.amount * r.rate
    FROM Orders o
    JOIN Rates FOR SYSTEM_TIME AS OF o.ts r
    ON o.currency = r.currency
\"\"\").print()
```
</details>
""",
    "lab_08.md": """# Lab 08: Idempotent Sink

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Understand Idempotency.

## Problem Statement
Write a Sink that writes to a Python Dictionary (simulating a KV store).
Ensure that writing `(Key, Val)` twice results in the same state.

## Starter Code
```python
store = {}
def sink(k, v):
    store[k] = v
```

## Hints
<details>
<summary>Hint 1</summary>
A dictionary assignment is naturally idempotent. `list.append` is NOT.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class IdempotentSink(SinkFunction):
    def __init__(self):
        self.store = {}

    def invoke(self, value, context):
        # Idempotent: Overwrite
        self.store[value[0]] = value[1]
        print(self.store)
```
</details>
""",
    "lab_09.md": """# Lab 09: Transactional Sink (2PC)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement `TwoPhaseCommitSinkFunction`.

## Problem Statement
(Reuse Week 4 Lab 15 logic).
Implement a sink that writes to a temporary file and renames it on commit.

## Starter Code
```python
# See Week 4 Lab 15
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure `pre_commit` flushes data.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Refer to Week 4 Lab 15 for the 2PC implementation.
# Key takeaway: The file is not visible until the checkpoint completes.
```
</details>
""",
    "lab_10.md": """# Lab 10: Dead Letter Queue (DLQ)

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Handle poison pills.

## Problem Statement
Process a stream of integers. If input is not an integer, send to DLQ (Side Output).
Do not fail the job.

## Starter Code
```python
try:
    val = int(s)
except:
    # side output
```

## Hints
<details>
<summary>Hint 1</summary>
Use `OutputTag`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
dlq_tag = OutputTag("dlq", Types.STRING())

class SafeMap(RichProcessFunction):
    def process_element(self, value, ctx, out):
        try:
            out.collect(int(value))
        except ValueError:
            ctx.output(dlq_tag, value)

main = ds.process(SafeMap())
main.get_side_output(dlq_tag).print()
```
</details>
""",
    "lab_11.md": """# Lab 11: Throttling

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Implement a Throttler.

## Problem Statement
Limit the stream to 1 element per second.
(Useful for Kappa backfill to protect DB).

## Starter Code
```python
time.sleep(1)
```

## Hints
<details>
<summary>Hint 1</summary>
`time.sleep` in a MapFunction blocks the thread. In Flink, this effectively throttles the source (backpressure).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import time

ds.map(lambda x: (time.sleep(1), x)[1]).print()
```
</details>
""",
    "lab_12.md": """# Lab 12: Deduplication

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Deduplicate a stream using State.

## Problem Statement
Filter out duplicate events (based on ID) seen within the last 10 minutes.

## Starter Code
```python
state_desc = ValueStateDescriptor("seen", Types.BOOLEAN())
ttl = StateTtlConfig...
```

## Hints
<details>
<summary>Hint 1</summary>
Use Keyed State with TTL.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class Dedupe(RichFlatMapFunction):
    def open(self, ctx):
        desc = ValueStateDescriptor("seen", Types.BOOLEAN())
        desc.enable_time_to_live(ttl_config) # 10 mins
        self.seen = ctx.get_state(desc)

    def flat_map(self, value, out):
        if self.seen.value() is None:
            self.seen.update(True)
            out.collect(value)
```
</details>
""",
    "lab_13.md": """# Lab 13: Event Time Join

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Join two streams on Event Time.

## Problem Statement
Join `Clicks` and `Views` on `ad_id` where Click is within 10 mins of View.

## Starter Code
```python
ds1.join(ds2).where(...).equal_to(...).window(...)
```

## Hints
<details>
<summary>Hint 1</summary>
Use `IntervalJoin` (KeyedStream.intervalJoin) for relative time.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Java/Scala API has intervalJoin. 
# In PyFlink, use SQL or Window Join.
t_env.execute_sql(\"\"\"
    SELECT * FROM Clicks c, Views v
    WHERE c.ad_id = v.ad_id
    AND c.ts BETWEEN v.ts AND v.ts + INTERVAL '10' MINUTE
\"\"\")
```
</details>
""",
    "lab_14.md": """# Lab 14: Speculative Execution (Concept)

## Difficulty
🔴 Hard

## Estimated Time
30 mins

## Learning Objectives
-   Understand Speculative Execution.

## Problem Statement
*Conceptual Lab*.
Flink does not support Speculative Execution (running duplicate tasks) because of State.
Explain why.

## Starter Code
```text
Write a short paragraph.
```

## Hints
<details>
<summary>Hint 1</summary>
If you run two copies of a stateful task, which one updates the state? Which one writes to the sink?
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**Answer**: Flink tasks are stateful. If you run two copies, they would both try to update the state (concurrency issues) or produce duplicate side effects (writing to sink twice). Speculative execution works for stateless batch (MapReduce/Spark) but not for stateful streaming.
</details>
""",
    "lab_15.md": """# Lab 15: Command Sourcing

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement Command Sourcing.

## Problem Statement
Stream of `Commands` (e.g., "Transfer").
Process function validates command and emits `Event` ("Transferred") or `Failure` ("InsufficientFunds").

## Starter Code
```python
class CommandHandler(KeyedProcessFunction):
    def process_element(self, cmd, ctx, out):
        # check balance
        # emit event
```

## Hints
<details>
<summary>Hint 1</summary>
This is the core of the Event Sourcing Write Model.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class Bank(KeyedProcessFunction):
    def process_element(self, cmd, ctx, out):
        current_balance = self.balance_state.value() or 0
        if cmd['amount'] <= current_balance:
            self.balance_state.update(current_balance - cmd['amount'])
            out.collect(f"Transferred {cmd['amount']}")
        else:
            ctx.output(failure_tag, "Insufficient Funds")
```
</details>
"""
}

print("🚀 Upgrading Week 6 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 6 Labs Upgrade Complete!")
