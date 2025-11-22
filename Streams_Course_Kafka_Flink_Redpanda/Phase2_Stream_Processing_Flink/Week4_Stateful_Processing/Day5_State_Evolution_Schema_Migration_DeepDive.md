# Day 5: State Evolution - Deep Dive

## Deep Dive & Internals

### Serializer Snapshots
When a checkpoint is taken, Flink saves the **Serializer Configuration** (schema).
-   On restore, Flink checks: "Is the registered serializer compatible with the saved one?"
-   **Compatible**: Proceed.
-   **Compatible after Reconfiguration**: Proceed (maybe slower).
-   **Incompatible**: Fail.

### Avro Schema Evolution
If you use Avro:
-   Store the writer schema in the checkpoint.
-   If the new reader schema is compatible (according to Avro rules), Flink handles the migration on-the-fly during restore.

### Advanced Reasoning
**Blue/Green Deployment with State**
1.  Take Savepoint of Job A (Blue).
2.  Use State Processor API to convert Savepoint A -> Savepoint B.
3.  Start Job B (Green) from Savepoint B.
4.  Switch traffic.

### Performance Implications
-   **Migration Cost**: On-the-fly migration (Avro) adds CPU overhead during the first read of each key.
-   **State Processor API**: Is a batch job. Can take time for TB-sized states.
