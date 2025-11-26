# Day 47: Message Brokers & RabbitMQ

## 1. The Smart Broker

RabbitMQ is a "Smart Broker, Dumb Consumer". It handles complex routing logic.

### 1.1 Core Concepts
*   **Producer**: Sends message to an **Exchange**.
*   **Exchange**: The router. Decides where the message goes.
*   **Queue**: The buffer. Stores messages until consumed.
*   **Binding**: The rule linking Exchange to Queue.
*   **Consumer**: Reads from Queue.

---

## 2. Exchange Types

### 2.1 Direct Exchange
*   **Logic**: Exact match.
*   **Routing Key**: `error`.
*   **Scenario**: Send log to `error_queue`.

### 2.2 Fanout Exchange
*   **Logic**: Broadcast.
*   **Routing Key**: Ignored.
*   **Scenario**: "New User Registered". Send to `EmailQueue`, `AnalyticsQueue`, `WelcomeQueue`.

### 2.3 Topic Exchange
*   **Logic**: Pattern match.
*   **Routing Key**: `order.usa.electronics`.
*   **Binding**: `order.*.electronics`.
*   **Scenario**: Complex routing based on geography and category.

---

## 3. Reliability

### 3.1 Acknowledgements (Ack)
*   RabbitMQ holds the message until Consumer says `ack`.
*   If Consumer dies (TCP close) without `ack`, RabbitMQ re-queues the message.

### 3.2 Durability
*   **Durable Queue**: Survives RabbitMQ restart.
*   **Persistent Message**: Written to disk.

---

## 4. Summary

Today we routed the traffic.
*   **RabbitMQ**: Flexible routing.
*   **Exchanges**: Direct, Fanout, Topic.
*   **Ack**: Guarantee delivery.

**Tomorrow (Day 48)**: We meet the "Dumb Broker, Smart Consumer". **Apache Kafka**.
