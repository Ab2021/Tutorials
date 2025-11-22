# Day 3: Clickstream Analytics

## Core Concepts & Theory

### The Use Case
Track user behavior on a website (Clicks, Views, AddToCart).
-   **Goal**: Real-time personalization, A/B test monitoring, Funnel analysis.
-   **Volume**: Extremely high (billions of events/day).

### Architecture
1.  **Collection**: JavaScript SDK -> Beacon API -> Nginx -> Kafka.
2.  **Processing**: Flink (Sessionization, Enrichment).
3.  **Serving**:
    -   **Real-time**: Redis (User Profile).
    -   **Analytics**: ClickHouse / Druid / Pinot.

### Key Patterns
-   **Sessionization**: Group events by UserID with a "Session Timeout" (e.g., 30 mins inactivity).
-   **Enrichment**: Join IP address with GeoIP database. Join UserAgent with Device database.

### Architectural Reasoning
**Why ClickHouse/Druid?**
Traditional Data Warehouses (Snowflake/BigQuery) are great for batch but slow/expensive for real-time ingestion. Real-time OLAP engines (ClickHouse, Druid, Pinot) can ingest from Kafka instantly and serve sub-second aggregations.
