# Day 9: Distributed Transactions

## 1. The Problem
*   **Monolith:** `BEGIN TX; UPDATE Orders; UPDATE Inventory; COMMIT;` (Easy, ACID).
*   **Microservices:** Order Service (DB1) and Inventory Service (DB2).
*   **Challenge:** If Order succeeds but Inventory fails, we have data inconsistency.

## 2. Two-Phase Commit (2PC)
*   **Coordinator:** The boss.
*   **Phase 1 (Prepare):** Coordinator asks all nodes: "Can you commit?" Nodes lock resources and say "Yes".
*   **Phase 2 (Commit):** If all said "Yes", Coordinator says "Commit". If any said "No", Coordinator says "Rollback".
*   **Pros:** Strong Consistency.
*   **Cons:** Blocking (if Coordinator dies, everyone waits). Slow.

## 3. Sagas (Long-Running Transactions)
*   **Concept:** Break transaction into a sequence of local transactions.
*   **Compensation:** If step $N$ fails, execute compensating transactions for $N-1, N-2...$ to undo changes.
*   **Example:**
    1.  `OrderService`: Create Order. (Success).
    2.  `InventoryService`: Reserve Item. (Fail - Out of Stock).
    3.  `OrderService`: Cancel Order (Compensating Action).

## 4. Saga Patterns
*   **Choreography:** Services talk to each other via Events. (OrderCreated -> InventoryService).
*   **Orchestration:** A central Orchestrator tells services what to do.
