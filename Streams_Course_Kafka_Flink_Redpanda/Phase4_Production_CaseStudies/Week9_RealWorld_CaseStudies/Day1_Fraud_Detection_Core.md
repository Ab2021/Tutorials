# Day 1: Real-Time Fraud Detection

## Core Concepts & Theory

### The Use Case
Detect fraudulent credit card transactions in real-time (< 200ms).
-   **Input**: Stream of transactions (CardID, Amount, Merchant, Location).
-   **Logic**: Complex patterns (e.g., "Card used in London and NYC within 5 mins").
-   **Output**: Block transaction or alert analyst.

### Architecture
1.  **Ingestion**: Payment Gateway -> Kafka (`transactions` topic).
2.  **Enrichment**: Flink joins with `CustomerProfile` (Redis/State).
3.  **Pattern Matching**: Flink CEP (Complex Event Processing).
4.  **Action**: Flink writes to `alerts` topic -> Consumer blocks card.

### Key Patterns
-   **Dynamic Rules**: Rules are stored in a database and broadcasted to Flink. No code redeployment needed for new rules.
-   **Feature Engineering**: Calculating rolling aggregates (e.g., "Avg spend in last 24h") in real-time.

### Architectural Reasoning
**Why Flink CEP?**
Standard SQL is good for aggregation ("Sum of sales"). CEP is good for **sequences** ("Event A followed by Event B within 10 mins"). Fraud is almost always a sequence pattern.
