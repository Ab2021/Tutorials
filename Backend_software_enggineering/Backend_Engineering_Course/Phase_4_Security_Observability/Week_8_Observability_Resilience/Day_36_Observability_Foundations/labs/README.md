# Lab: Day 36 - Structured Logging

## Goal
Implement JSON logging with Correlation IDs.

## Prerequisites
- `pip install python-json-logger`

## Step 1: The Code (`logger_demo.py`)

```python
import logging
import uuid
from pythonjsonlogger import jsonlogger

# 1. Configure Logger
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s %(request_id)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# 2. Context (Simulate Middleware)
class ContextFilter(logging.Filter):
    def filter(self, record):
        # In a real app, get this from Flask/FastAPI context
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return True

logger.addFilter(ContextFilter())

# 3. Simulate Request
def process_payment(user_id, amount, request_id):
    # Pass request_id via "extra" or ContextVars
    extra = {'request_id': request_id, 'user_id': user_id, 'amount': amount}
    
    logger.info("Starting payment processing", extra=extra)
    
    try:
        if amount < 0:
            raise ValueError("Negative amount")
        logger.info("Payment successful", extra=extra)
    except Exception as e:
        logger.error("Payment failed", extra={**extra, 'error': str(e)})

# Run
req_id = str(uuid.uuid4())
process_payment(user_id=123, amount=50, request_id=req_id)

print("\n--- Error Case ---")
req_id_2 = str(uuid.uuid4())
process_payment(user_id=123, amount=-10, request_id=req_id_2)
```

## Step 2: Run It
`python logger_demo.py`

*   **Output**:
    ```json
    {"asctime": "...", "levelname": "INFO", "message": "Starting payment processing", "request_id": "...", "user_id": 123, "amount": 50}
    ```

## Challenge
Integrate this with FastAPI.
1.  Create a Middleware that generates `X-Request-ID`.
2.  Store it in `contextvars`.
3.  Configure the logger to read from `contextvars` automatically so you don't have to pass `extra={...}` every time.
