import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase2_Stream_Processing_Flink\Week4_Stateful_Processing\labs"

labs_content = {
    "lab_01.md": """# Lab 01: ValueState

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use `ValueState`.
-   Understand state lifecycle.

## Problem Statement
Implement a `RichFlatMapFunction` that keeps a running sum of integers per key.
Input: `(key, value)`.
Output: `(key, current_sum)`.

## Starter Code
```python
class SumFunction(RichFlatMapFunction):
    def open(self, ctx):
        state_desc = ValueStateDescriptor("sum", Types.INT())
        self.sum_state = ctx.get_state(state_desc)

    def flat_map(self, value, out):
        # current = self.sum_state.value()
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
`value()` returns `None` if state is empty. Handle that case.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RichFlatMapFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common import Types

class RunningSum(RichFlatMapFunction):
    def open(self, runtime_context):
        descriptor = ValueStateDescriptor("sum", Types.INT())
        self.sum_state = runtime_context.get_state(descriptor)

    def flat_map(self, value, out):
        current_sum = self.sum_state.value()
        if current_sum is None:
            current_sum = 0
        
        new_sum = current_sum + value[1]
        self.sum_state.update(new_sum)
        out.collect((value[0], new_sum))

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    ds = env.from_collection([("A", 1), ("A", 2), ("B", 5)])
    ds.key_by(lambda x: x[0]).flat_map(RunningSum()).print()
    env.execute()

if __name__ == '__main__':
    run()
```
</details>
""",
    "lab_02.md": """# Lab 02: ListState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `ListState`.
-   Buffer elements.

## Problem Statement
Buffer elements for a key until you have 5 elements, then emit the average and clear the buffer.

## Starter Code
```python
class BufferAverage(RichFlatMapFunction):
    def open(self, ctx):
        self.list_state = ctx.get_list_state(...)
```

## Hints
<details>
<summary>Hint 1</summary>
`list_state.get()` returns an iterable. `list_state.clear()` empties it.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class BufferAverage(RichFlatMapFunction):
    def open(self, ctx):
        self.buffer = ctx.get_list_state(ListStateDescriptor("buffer", Types.INT()))

    def flat_map(self, value, out):
        self.buffer.add(value[1])
        
        # Check size (inefficient for large lists, but okay for 5)
        elements = list(self.buffer.get())
        if len(elements) >= 5:
            avg = sum(elements) / len(elements)
            out.collect((value[0], avg))
            self.buffer.clear()
```
</details>
""",
    "lab_03.md": """# Lab 03: MapState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `MapState`.
-   Manage a dictionary per key.

## Problem Statement
Input: `(User, URL)`.
Keep a count of visits *per URL* for each User.
Output: `(User, URL, NewCount)`.

## Starter Code
```python
class UrlCounter(RichFlatMapFunction):
    def open(self, ctx):
        self.map_state = ctx.get_map_state(MapStateDescriptor("counts", Types.STRING(), Types.INT()))
```

## Hints
<details>
<summary>Hint 1</summary>
`map_state.get(key)` returns None if not found.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class UrlCounter(RichFlatMapFunction):
    def open(self, ctx):
        self.counts = ctx.get_map_state(MapStateDescriptor("counts", Types.STRING(), Types.INT()))

    def flat_map(self, value, out):
        user, url = value
        current = self.counts.get(url)
        if current is None:
            current = 0
        
        new_count = current + 1
        self.counts.put(url, new_count)
        out.collect((user, url, new_count))
```
</details>
""",
    "lab_04.md": """# Lab 04: ReducingState

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use `ReducingState`.
-   Optimize aggregation.

## Problem Statement
Keep the "Max" value seen so far for each key. Use `ReducingState` instead of `ValueState` (it's more efficient because it merges on write).

## Starter Code
```python
class MaxReducer(ReduceFunction):
    def reduce(self, value1, value2):
        return max(value1, value2)

# In open():
# ctx.get_reducing_state(ReducingStateDescriptor("max", MaxReducer(), Types.INT()))
```

## Hints
<details>
<summary>Hint 1</summary>
`ReducingState.add(value)` automatically merges the new value with the existing one using your reducer.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class MaxFunction(RichFlatMapFunction):
    def open(self, ctx):
        self.max_state = ctx.get_reducing_state(
            ReducingStateDescriptor("max", lambda a, b: max(a, b), Types.INT())
        )

    def flat_map(self, value, out):
        self.max_state.add(value[1])
        out.collect((value[0], self.max_state.get()))
```
</details>
""",
    "lab_05.md": """# Lab 05: State TTL

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Configure State Time-To-Live.
-   Prevent state leaks.

## Problem Statement
Modify Lab 01 (Running Sum) so that the state expires after **1 minute** of inactivity (no updates for that key).

## Starter Code
```python
ttl_config = StateTtlConfig.new_builder(Time.minutes(1)) \
    .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
    .build()

descriptor.enable_time_to_live(ttl_config)
```

## Hints
<details>
<summary>Hint 1</summary>
TTL is configured on the `StateDescriptor`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.state import StateTtlConfig

    def open(self, ctx):
        ttl_config = StateTtlConfig.new_builder(Time.minutes(1)) \
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite) \
            .cleanup_full_snapshot() \
            .build()

        descriptor = ValueStateDescriptor("sum", Types.INT())
        descriptor.enable_time_to_live(ttl_config)
        
        self.sum_state = ctx.get_state(descriptor)
```
</details>
""",
    "lab_06.md": """# Lab 06: Checkpointing Config

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Enable Checkpointing.
-   Configure interval and storage.

## Problem Statement
Configure the environment to:
1.  Checkpoint every 10 seconds.
2.  Use `ExactlyOnce` mode.
3.  Store checkpoints in `file:///tmp/flink-checkpoints`.

## Starter Code
```python
env.enable_checkpointing(10000)
# env.get_checkpoint_config()...
```

## Hints
<details>
<summary>Hint 1</summary>
Use `CheckpointingMode.EXACTLY_ONCE`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import CheckpointingMode

env.enable_checkpointing(10000)
config = env.get_checkpoint_config()
config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
config.set_checkpoint_storage("file:///tmp/flink-checkpoints")
```
</details>
""",
    "lab_07.md": """# Lab 07: RocksDB Backend

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Switch State Backend to RocksDB.
-   Understand dependencies.

## Problem Statement
Configure the job to use `EmbeddedRocksDBStateBackend`.
*Note: You need the `flink-statebackend-rocksdb` JAR.*

## Starter Code
```python
env.set_state_backend(EmbeddedRocksDBStateBackend())
```

## Hints
<details>
<summary>Hint 1</summary>
In PyFlink, you might need to add the JAR via `env.add_jars()`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.state import EmbeddedRocksDBStateBackend

# Ensure JAR is loaded
env.add_jars("file:///path/to/flink-statebackend-rocksdb.jar")

env.set_state_backend(EmbeddedRocksDBStateBackend())
```
</details>
""",
    "lab_08.md": """# Lab 08: Savepoints

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Trigger a Savepoint.
-   Stop and Resume a job.

## Problem Statement
1.  Run the `RunningSum` job (Lab 01).
2.  Trigger a savepoint via CLI: `flink savepoint <job_id> /tmp/savepoints`.
3.  Cancel the job.
4.  Resume from savepoint: `flink run -s /tmp/savepoints/savepoint-xxx ...`.

## Starter Code
```bash
# CLI commands
```

## Hints
<details>
<summary>Hint 1</summary>
Use `flink list` to find the Job ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# 1. List jobs
flink list

# 2. Savepoint
flink savepoint <job_id> /tmp/savepoints

# 3. Cancel
flink cancel <job_id>

# 4. Resume
flink run -s /tmp/savepoints/savepoint-<id> -py job.py
```
</details>
""",
    "lab_09.md": """# Lab 09: Broadcast State

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement the Broadcast Pattern.
-   Dynamic Rules.

## Problem Statement
Stream 1: `Transactions (Item, Price)`.
Stream 2: `Thresholds (Item, MaxPrice)`.
Requirement: Alert if Price > MaxPrice. MaxPrice is updated dynamically via Stream 2.

## Starter Code
```python
rule_state_desc = MapStateDescriptor("rules", Types.STRING(), Types.INT())
broadcast_stream = rules.broadcast(rule_state_desc)

tx.connect(broadcast_stream).process(MyBroadcastFunction())
```

## Hints
<details>
<summary>Hint 1</summary>
`process_broadcast_element` updates the state. `process_element` reads it (read-only).
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class AlertFunction(BroadcastProcessFunction):
    def process_broadcast_element(self, value, ctx):
        # Update Rule
        item, threshold = value
        ctx.get_broadcast_state(rule_state_desc).put(item, threshold)

    def process_element(self, value, ctx, out):
        # Check Rule
        item, price = value
        threshold = ctx.get_broadcast_state(rule_state_desc).get(item)
        if threshold and price > threshold:
            out.collect(f"Alert: {item} price {price} > {threshold}")
```
</details>
""",
    "lab_10.md": """# Lab 10: Operator State (List)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Implement `CheckpointedFunction`.
-   Use Operator ListState.

## Problem Statement
Implement a Sink that buffers 10 elements and then prints them. It must persist the buffer in Operator State so that no data is lost on crash.

## Starter Code
```python
class BufferingSink(SinkFunction, CheckpointedFunction):
    def snapshot_state(self, context):
        self.checkpointed_state.clear()
        for element in self.buffer:
            self.checkpointed_state.add(element)

    def initialize_state(self, context):
        self.checkpointed_state = context.get_operator_state_store().get_list_state(...)
```

## Hints
<details>
<summary>Hint 1</summary>
`snapshot_state` is called on checkpoint. `initialize_state` is called on start/restore.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
class BufferingSink(SinkFunction, CheckpointedFunction):
    def __init__(self):
        self.buffer = []
        self.checkpointed_state = None

    def invoke(self, value, context):
        self.buffer.append(value)
        if len(self.buffer) >= 10:
            print(self.buffer)
            self.buffer.clear()

    def snapshot_state(self, context):
        self.checkpointed_state.clear()
        for element in self.buffer:
            self.checkpointed_state.add(element)

    def initialize_state(self, context):
        descriptor = ListStateDescriptor("buffered-elements", Types.INT())
        self.checkpointed_state = context.get_operator_state_store().get_list_state(descriptor)
        
        if context.is_restored():
            for element in self.checkpointed_state.get():
                self.buffer.append(element)
```
</details>
""",
    "lab_11.md": """# Lab 11: Schema Evolution (Avro)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Simulate schema evolution.

## Problem Statement
1.  Define an Avro schema `User(name)`.
2.  Run a job using this state. Take a savepoint.
3.  Update schema to `User(name, age=0)`.
4.  Resume job. Verify it works.

## Starter Code
```json
// user_v1.avsc
{"type":"record", "name":"User", "fields":[{"name":"name", "type":"string"}]}
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure you use `AvroSerializer`. Flink handles the mapping if the new schema has a default value for the new field.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

*Conceptual Steps*:
1.  Compile `User` class from V1 schema.
2.  Run job, populate state.
3.  Stop with Savepoint.
4.  Compile `User` class from V2 schema (with default for `age`).
5.  Update job code to use new class.
6.  Restore. Flink detects the schema change and adapts.
</details>
""",
    "lab_12.md": """# Lab 12: Queryable State

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Understand Queryable State (Deprecated but useful concept).
-   Alternative: Expose state via Side Output to a DB.

## Problem Statement
Since Queryable State is deprecated, implement the modern equivalent:
Write a `RichFlatMapFunction` that updates state AND sends the update to a "Query" stream (Side Output) that writes to Redis.

## Starter Code
```python
# Side Output
ctx.output(query_tag, (key, new_value))
```

## Hints
<details>
<summary>Hint 1</summary>
This is the "CQRS" pattern in streaming.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    def flat_map(self, value, out):
        # Update State
        self.state.update(value)
        
        # Emit to Main Stream
        out.collect(value)
        
        # Emit to Query Stream (Side Output)
        # In reality, you might write directly to Redis here in async mode, 
        # or use a Sink on the side output.
        pass 
```
</details>
""",
    "lab_13.md": """# Lab 13: UID Assignment

## Difficulty
🟡 Medium

## Estimated Time
30 mins

## Learning Objectives
-   Understand `uid()`.
-   Prevent state loss during topology changes.

## Problem Statement
1.  Write a job with `ds.map(...).keyBy(...).sum(...)`.
2.  Assign UIDs: `ds.map(...).uid("my-map")...`.
3.  Change the chain (insert a filter).
4.  Verify that state can still be restored because UIDs match.

## Starter Code
```python
ds.map(MyMap()).uid("mapper-1")
```

## Hints
<details>
<summary>Hint 1</summary>
If you don't assign UIDs, Flink generates them based on the graph structure. Changing the graph changes the IDs, making state unrecoverable.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
    ds.map(lambda x: x).uid("source-map") \
      .key_by(...) \
      .map(StatefulMap()).uid("stateful-op") \
      .print()
```
Always assign UIDs to stateful operators in production!
</details>
""",
    "lab_14.md": """# Lab 14: State Processor API (Read)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Read a Savepoint offline.

## Problem Statement
Write a batch job using the State Processor API to read a Savepoint from Lab 01 and print the total sum of all keys.

## Starter Code
```java
// Java only (Python support is limited/experimental)
ExecutionEnvironment bEnv = ExecutionEnvironment.getExecutionEnvironment();
ExistingSavepoint savepoint = Savepoint.load(bEnv, "file:///tmp/savepoint", new MemoryStateBackend());
```

## Hints
<details>
<summary>Hint 1</summary>
You need to define a `KeyedStateReaderFunction`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
DataSet<Integer> sums = savepoint.readKeyedState(
    "stateful-op", 
    new ReaderFunction());

sums.sum(0).print();
```
</details>
""",
    "lab_15.md": """# Lab 15: Two-Phase Commit (Sink)

## Difficulty
🔴 Hard

## Estimated Time
90 mins

## Learning Objectives
-   Understand `TwoPhaseCommitSinkFunction`.
-   Implement Exactly-Once to external systems.

## Problem Statement
Implement a dummy `TwoPhaseCommitSinkFunction` that simulates writing to a transactional file system.
-   `beginTransaction`: Create temp file.
-   `preCommit`: Flush to temp file.
-   `commit`: Rename temp to final.
-   `abort`: Delete temp.

## Starter Code
```python
class File2PCSink(TwoPhaseCommitSinkFunction):
    # Implement methods
    pass
```

## Hints
<details>
<summary>Hint 1</summary>
This is complex. Focus on the lifecycle logs.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
# Conceptual Python implementation (requires Java wrapper usually)
class My2PC(TwoPhaseCommitSinkFunction):
    def begin_transaction(self):
        return create_temp_file()

    def invoke(self, transaction, value, context):
        transaction.write(value)

    def pre_commit(self, transaction):
        transaction.flush()

    def commit(self, transaction):
        transaction.move_to_final()

    def abort(self, transaction):
        transaction.delete()
```
</details>
"""
}

print("🚀 Upgrading Week 4 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 4 Labs Upgrade Complete!")
