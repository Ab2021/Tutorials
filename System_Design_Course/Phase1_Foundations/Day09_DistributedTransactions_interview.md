# Day 9 Interview Prep: Distributed Transactions

## Q1: 2PC vs Sagas?
**Answer:**
*   **2PC:** Strong consistency, blocking, hard to scale. Use for money transfer.
*   **Sagas:** Eventual consistency, non-blocking, scalable. Use for order processing.

## Q2: What is a Compensating Transaction?
**Answer:**
*   An action that undoes the effect of a previous committed transaction in a Saga.
*   Example: If "Charge Payment" fails, the compensating action for "Reserve Inventory" is "Release Inventory".

## Q3: What happens if the Compensating Transaction fails?
**Answer:**
*   **Retry:** Ideally, compensating actions should be idempotent and retried until success.
*   **Manual Intervention:** If it fails permanently, alert a human operator.

## Q4: Choreography vs Orchestration?
**Answer:**
*   **Choreography:** Decentralized. Services listen to events. Hard to track complex flows. Good for simple flows.
*   **Orchestration:** Centralized. One service directs others. Easier to manage state and error handling. Good for complex flows.
