# Lab: Day 46 - EDA Simulation

## Goal
Understand Producer-Consumer pattern using Python Threads.

## Step 1: The Code (`eda_sim.py`)

```python
import threading
import queue
import time
import random

# The Broker
event_queue = queue.Queue()

def producer():
    """Generates Orders"""
    for i in range(5):
        order_id = f"ORD-{random.randint(1000, 9999)}"
        print(f"📦 [Producer] Order Placed: {order_id}")
        event_queue.put(order_id)
        time.sleep(1) # User places order every 1s

def consumer(name, delay):
    """Processes Orders"""
    while True:
        try:
            # Wait for event (timeout to exit loop if empty)
            order_id = event_queue.get(timeout=5)
        except queue.Empty:
            break
        
        print(f"⚙️  [{name}] Processing {order_id}...")
        time.sleep(delay) # Simulate work
        print(f"✅ [{name}] Finished {order_id}")
        event_queue.task_done()

# Run
t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer, args=("EmailService", 2)) # Slow consumer
t3 = threading.Thread(target=consumer, args=("InventoryService", 0.5)) # Fast consumer

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

print("All orders processed.")
```

## Step 2: Run It
`python eda_sim.py`

*   **Observe**:
    *   The Producer finishes quickly (doesn't wait for consumers).
    *   `InventoryService` finishes fast.
    *   `EmailService` takes its time.
    *   This is **Decoupling**.

## Challenge
Modify the code to handle **Failures**.
1.  Make `EmailService` fail randomly (raise Exception).
2.  Catch the exception and put the message back in the queue (Retry).
3.  Add a "Retry Count" to the message to prevent infinite loops.
