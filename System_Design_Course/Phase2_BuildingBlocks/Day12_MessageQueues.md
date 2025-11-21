# Day 12: Message Queues

## 1. Why Message Queues?
*   **Decoupling:** Producer doesn't need to know who Consumer is.
*   **Asynchronous:** Fire and forget. Return response to user immediately.
*   **Throttling (Load Leveling):** If Producer sends 1000 req/s but Consumer handles 100 req/s, Queue buffers the spike.
*   **Reliability:** If Consumer dies, message stays in Queue.

## 2. Models
### Point-to-Point (Queue)
*   **One-to-One.**
*   Message is consumed by exactly one consumer.
*   **Example:** Task Queue (Celery), RabbitMQ.

### Publish-Subscribe (Topic)
*   **One-to-Many.**
*   Producer publishes to a Topic. All Subscribers get a copy.
*   **Example:** Notification System (Email Service + SMS Service both listen to "UserCreated"), Kafka.

## 3. Push vs Pull
*   **Push (RabbitMQ):** Broker pushes messages to Consumer.
    *   **Pros:** Low latency.
    *   **Cons:** Can overwhelm consumer.
*   **Pull (Kafka):** Consumer pulls messages from Broker.
    *   **Pros:** Consumer controls rate (Backpressure).
    *   **Cons:** Polling loop overhead.

## 4. Kafka vs RabbitMQ
| Feature | RabbitMQ | Kafka |
| :--- | :--- | :--- |
| **Model** | Smart Broker, Dumb Consumer | Dumb Broker, Smart Consumer |
| **Persistence** | Memory (mostly) | Disk (Log) |
| **Throughput** | 20k - 50k msg/sec | 1 Million+ msg/sec |
| **Retention** | Delete after ack | Keep for X days |
| **Use Case** | Complex routing, Task Queue | Event Streaming, Logs, Analytics |
