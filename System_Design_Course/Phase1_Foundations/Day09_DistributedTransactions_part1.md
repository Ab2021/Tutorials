# Day 9 Deep Dive: Sagas vs 2PC

## 1. 2PC (Two-Phase Commit)
*   **Type:** ACID (Atomic).
*   **Latency:** High (Round trips + Locks).
*   **Availability:** Low (Locks resources).
*   **Use Case:** Banking (where consistency is paramount).

## 2. Sagas
*   **Type:** BASE (Eventual Consistency).
*   **Latency:** Low (Local transactions).
*   **Availability:** High (No long locks).
*   **Use Case:** E-commerce, Travel Booking (Book Flight -> Book Hotel).

## 3. TCC (Try-Confirm-Cancel)
A variation of 2PC for business logic.
*   **Try:** Reserve resource (e.g., "Pending" state).
*   **Confirm:** Finalize (Change to "Confirmed").
*   **Cancel:** Release reservation.

## 4. Code: Saga Orchestrator (Python Concept)
```python
class OrderSaga:
    def execute(self, order):
        try:
            # Step 1
            order_id = OrderService.create(order)
            
            # Step 2
            try:
                InventoryService.reserve(order_id)
            except Exception:
                OrderService.cancel(order_id) # Compensate Step 1
                raise
            
            # Step 3
            try:
                PaymentService.charge(order_id)
            except Exception:
                InventoryService.release(order_id) # Compensate Step 2
                OrderService.cancel(order_id) # Compensate Step 1
                raise
                
        except Exception as e:
            print("Saga Failed:", e)
```
