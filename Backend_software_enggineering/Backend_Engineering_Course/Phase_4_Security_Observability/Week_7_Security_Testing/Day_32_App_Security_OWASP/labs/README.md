# Lab: Day 32 - SQL Injection (Attack & Defense)

## Goal
See a real SQL Injection in action and fix it.

## Directory Structure
```
day32/
├── vulnerable.py
├── secure.py
└── README.md
```

## Step 1: The Vulnerable App (`vulnerable.py`)

```python
import sqlite3

def login(username, password):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INT, name TEXT, password TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'admin', 'secret123')")
    
    # VULNERABLE: String Concatenation
    query = f"SELECT * FROM users WHERE name = '{username}' AND password = '{password}'"
    print(f"Executing: {query}")
    
    cursor.execute(query)
    user = cursor.fetchone()
    conn.close()
    
    if user:
        print("✅ Login Success!")
    else:
        print("❌ Login Failed.")

# Normal Login
print("--- Normal Login ---")
login("admin", "secret123")

# Attack
print("\n--- Attack ---")
# The password is: ' OR '1'='1
login("admin", "' OR '1'='1")
```

## Step 2: Run It
`python vulnerable.py`

*   **Output**:
    ```
    Executing: SELECT * FROM users WHERE name = 'admin' AND password = '' OR '1'='1'
    ✅ Login Success!
    ```
    *   The attacker logged in without the password!

## Step 3: The Secure App (`secure.py`)

```python
import sqlite3

def login_secure(username, password):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INT, name TEXT, password TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'admin', 'secret123')")
    
    # SECURE: Parameterized Query
    query = "SELECT * FROM users WHERE name = ? AND password = ?"
    # print(f"Executing: {query} with params {(username, password)}")
    
    cursor.execute(query, (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        print("✅ Login Success!")
    else:
        print("❌ Login Failed.")

print("--- Secure Attack ---")
login_secure("admin", "' OR '1'='1")
```

## Step 4: Run Secure
`python secure.py`

*   **Output**: `❌ Login Failed.`
    *   The DB treats `' OR '1'='1` as the literal password string, not SQL commands.

## Challenge
Implement a **Stored XSS** simulation.
1.  Create a simple HTML file that reads a "comment" from a JSON file and displays it using `innerHTML`.
2.  Put `<img src=x onerror=alert(1)>` in the JSON.
3.  Open HTML in browser -> See Alert.
4.  Fix it by using `innerText` or a sanitization library.
