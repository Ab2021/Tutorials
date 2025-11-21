# Day 3: API Design

## 1. REST (Representational State Transfer)
*   **Resource-Oriented:** `/users`, `/posts`.
*   **Verbs:** GET (Read), POST (Create), PUT (Replace), PATCH (Update), DELETE (Remove).
*   **Stateless:** Server stores no client context.
*   **HATEOAS:** Hypermedia as the Engine of Application State (Links in response).

## 2. GraphQL
*   **Query Language:** Client asks for exactly what it needs.
*   **Pros:** No over-fetching or under-fetching. Single endpoint (`/graphql`).
*   **Cons:** Complex caching (POST requests), N+1 query problem.

## 3. RPC (Remote Procedure Call)
*   **Action-Oriented:** `createUser()`, `sendEmail()`.
*   **Modern:** gRPC.
*   **Use Case:** Internal services where performance matters.

## 4. Idempotency
*   **Definition:** Making the same request multiple times has the same effect as making it once.
*   **Crucial for:** Payments, Retries.
*   **Implementation:** Client sends `Idempotency-Key` (UUID). Server checks if key exists in DB/Redis. If yes, return cached response.

## 5. Code: Idempotency Middleware (Python/Flask)
```python
from flask import Flask, request, jsonify
import redis

app = Flask(__name__)
r = redis.Redis()

@app.route('/pay', methods=['POST'])
def pay():
    key = request.headers.get('Idempotency-Key')
    if not key:
        return jsonify({"error": "Missing Key"}), 400
        
    # Check cache
    if r.exists(key):
        return jsonify({"status": "Already Processed", "tx_id": r.get(key).decode()})
        
    # Process Payment
    tx_id = "tx_12345" # Call Stripe/PayPal
    
    # Save to cache (expire in 24h)
    r.setex(key, 86400, tx_id)
    
    return jsonify({"status": "Success", "tx_id": tx_id})
```
