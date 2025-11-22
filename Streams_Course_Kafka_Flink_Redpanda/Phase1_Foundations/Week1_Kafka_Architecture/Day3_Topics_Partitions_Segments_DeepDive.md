# Day 3: Topics, Partitions, Segments - Deep Dive

## Deep Dive & Internals

### Segment File Internals
A segment consists of:
1.  `.log`: The actual messages.
2.  `.index`: Maps **Offset -> Physical Position** (Byte offset in .log file).
3.  `.timeindex`: Maps **Timestamp -> Offset**.

### Indexing Strategy
Kafka indexes are **sparse**. It doesn't index every message.
-   It might index every 4KB of data.
-   To find offset 100:
    1.  Look in `.index` for the largest offset <= 100 (say, 90).
    2.  Jump to the physical position of 90 in `.log`.
    3.  Scan forward linearly until you find 100.
-   **Trade-off**: Saves RAM (index fits in memory) at the cost of a tiny linear scan.

### Log Compaction Details
-   **Cleaner Thread**: Background thread that compacts logs.
-   **Head vs Tail**:
    -   **Tail**: Cleaned portion. Contains only unique keys.
    -   **Head**: Active portion. Contains duplicates.
-   **Tombstones**: A message with `Value=null` is a delete marker. It removes the key from the log eventually.

### Advanced Reasoning
**Why not one file per partition?**
-   **File Size**: A single file is hard to manage (delete old data). Segments allow deleting old data by simply deleting the oldest file (`rm segment-000.log`).
-   **OS Limits**: Filesystems handle smaller files better than one multi-TB file.

### Performance Implications
-   **Open File Descriptors**: Each segment uses FDs. Too many partitions * too many segments = `Too many open files` error.
