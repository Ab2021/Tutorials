# Lab: Day 12 - Database per Service

## Goal
Implement strict data isolation. You will build two services, each with its own SQLite database, and stitch the data together using API Composition.

## Directory Structure
```
day12/
├── product_service/
│   ├── app.py
│   └── products.db (Auto-created)
├── inventory_service/
│   ├── app.py
│   └── inventory.db (Auto-created)
└── requirements.txt
```

## Step 1: Inventory Service (`inventory_service/app.py`)
Owns the stock data.

```python
from fastapi import FastAPI
import sqlite3

app = FastAPI()

def init_db():
    conn = sqlite3.connect('inventory.db')
    conn.execute('CREATE TABLE IF NOT EXISTS stock (product_id INT PRIMARY KEY, quantity INT)')
    conn.execute('INSERT OR IGNORE INTO stock (product_id, quantity) VALUES (1, 100), (2, 50)')
    conn.commit()
    conn.close()

init_db()

@app.get("/inventory/{product_id}")
def get_stock(product_id: int):
    conn = sqlite3.connect('inventory.db')
    cur = conn.execute('SELECT quantity FROM stock WHERE product_id = ?', (product_id,))
    row = cur.fetchone()
    conn.close()
    return {"product_id": product_id, "stock": row[0] if row else 0}
```

## Step 2: Product Service (`product_service/app.py`)
Owns product details. Calls Inventory Service for stock.

```python
from fastapi import FastAPI
import sqlite3
import httpx

app = FastAPI()

def init_db():
    conn = sqlite3.connect('products.db')
    conn.execute('CREATE TABLE IF NOT EXISTS products (id INT PRIMARY KEY, name TEXT)')
    conn.execute('INSERT OR IGNORE INTO products (id, name) VALUES (1, "Laptop"), (2, "Mouse")')
    conn.commit()
    conn.close()

init_db()

INVENTORY_URL = "http://localhost:8002"

@app.get("/products/{product_id}")
async def get_product_details(product_id: int):
    # 1. Get local data
    conn = sqlite3.connect('products.db')
    cur = conn.execute('SELECT name FROM products WHERE id = ?', (product_id,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return {"error": "Product not found"}
    
    name = row[0]
    
    # 2. Call Inventory Service (API Composition)
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{INVENTORY_URL}/inventory/{product_id}")
        stock_data = resp.json()
        
    # 3. Merge
    return {
        "id": product_id,
        "name": name,
        "stock": stock_data.get("stock", 0),
        "source": "Aggregated from ProductDB + InventoryDB"
    }
```

## Step 3: Run It

1.  **Inventory Service**:
    `uvicorn inventory_service.app:app --port 8002`

2.  **Product Service**:
    `uvicorn product_service.app:app --port 8001`

3.  **Test**:
    `curl http://localhost:8001/products/1`
    
    *Expected Output*:
    ```json
    {
      "id": 1,
      "name": "Laptop",
      "stock": 100,
      "source": "Aggregated from ProductDB + InventoryDB"
    }
    ```

## Reflection
*   Try deleting `products.db`. Does Inventory Service still work? (Yes, fault isolation).
*   Try killing Inventory Service. What does Product Service return? (It crashes unless you add error handling/circuit breaker).
