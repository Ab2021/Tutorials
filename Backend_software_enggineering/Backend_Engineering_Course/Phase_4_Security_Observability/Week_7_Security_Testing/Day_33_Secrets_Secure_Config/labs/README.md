# Lab: Day 33 - Managing Secrets

## Goal
Learn how to handle configuration securely using Environment Variables.

## Prerequisites
- `pip install python-dotenv`

## Step 1: The `.env` file
Create a file named `.env`:
```ini
DB_HOST=localhost
DB_USER=admin
# NEVER commit this file to Git!
DB_PASSWORD=super-secret-password-123
```

## Step 2: The App (`app.py`)

```python
import os
from dotenv import load_dotenv

# 1. Load env vars from .env file
load_dotenv()

def connect_db():
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    
    if not password:
        raise ValueError("DB_PASSWORD is missing!")
        
    print(f"Connecting to {host} as {user} with password: {password[:2]}***")

if __name__ == "__main__":
    connect_db()
```

## Step 3: Run It
`python app.py`

## Step 4: Git Ignore
Create a `.gitignore` file:
```text
.env
__pycache__/
```
*   This ensures you don't accidentally commit the secrets.

## Challenge
Simulate a **Secret Manager**.
Write a script `vault_sim.py` that:
1.  Stores secrets in a JSON file `vault.json` (encrypted with a master password).
2.  Accepts a master password to unlock and read a secret.
    *   *Hint*: Use `cryptography.fernet` for encryption.
