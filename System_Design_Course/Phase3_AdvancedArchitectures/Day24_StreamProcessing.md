# Day 24: Stream Processing

## 1. Batch vs Stream
*   **Batch:** Finite data. High latency. (e.g., Daily Report).
*   **Stream:** Infinite data. Low latency. (e.g., Fraud Detection, Real-time Analytics).

## 2. Processing Models
*   **One-at-a-time:** Process each event as it arrives. (Storm).
*   **Micro-batching:** Group events into small batches (e.g., 1s). (Spark Streaming).
*   **Windowing:** Group events by time.

## 3. Window Types
*   **Tumbling Window:** Fixed size, non-overlapping. (e.g., "Count per minute"). `[00:00-00:01], [00:01-00:02]`.
*   **Sliding Window:** Fixed size, overlapping. (e.g., "Last 1 minute, updated every 10s").
*   **Session Window:** Dynamic size. Based on user activity. (e.g., "User active until 30s timeout").

## 4. Time Semantics
*   **Event Time:** When the event actually happened (Timestamp in JSON).
*   **Processing Time:** When the system received the event.
*   **Watermark:** A heuristic to handle late data. "I have seen all data up to 12:00:00".

## 5. Tools
*   **Apache Flink:** True streaming. State management. Exactly-once.
*   **Apache Storm:** Low latency. At-least-once.
*   **Kafka Streams:** Library (not cluster). Easy to deploy.
