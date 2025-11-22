# Day 2: IoT Telemetry Processing

## Core Concepts & Theory

### The Use Case
Process sensor data from 1 Million connected cars.
-   **Input**: GPS, Speed, EngineTemp, FuelLevel (1 msg/sec per car).
-   **Volume**: 1M msg/sec.
-   **Goal**: Real-time dashboard, Geofencing, Predictive Maintenance.

### Architecture
1.  **Edge**: MQTT Broker (e.g., HiveMQ/VerneMQ).
2.  **Bridge**: MQTT -> Kafka Connect -> Kafka.
3.  **Processing**: Flink (Windowed Aggregation, Geofencing).
4.  **Storage**:
    -   **Hot**: TimescaleDB / Druid (Dashboards).
    -   **Cold**: S3 (Data Lake).

### Key Patterns
-   **Downsampling**: Convert 1Hz data to 1-minute averages for long-term storage.
-   **Sessionization**: Detect "Trips" (Ignition ON to Ignition OFF).

### Architectural Reasoning
**MQTT vs Kafka**
-   **MQTT**: Lightweight, good for unreliable networks (IoT devices). Push model.
-   **Kafka**: Heavy, high throughput. Pull model.
-   **Pattern**: Use MQTT for Device-to-Cloud. Bridge to Kafka for Cloud-Internal processing.
