# Lab: Day 1 - Environment Setup & "Hello Backend"

## Goal
The goal of this lab is to set up a professional backend development environment. You will configure Docker, create a project structure, and write a simple script to connect to a database running in a container.

## Prerequisites
- Docker Desktop (or Docker Engine) installed and running.
- Python 3.11+ installed (or use `pyenv`).
- Git installed.
- VS Code (or your preferred editor).

## Steps

### 1. Project Setup
Create a new directory for your course work.
```bash
mkdir backend-course
cd backend-course
mkdir -p day01/src
```

Initialize Git:
```bash
git init
echo "__pycache__/" > .gitignore
echo ".venv/" >> .gitignore
echo ".env" >> .gitignore
```

### 2. Docker Compose
Create `day01/docker-compose.yml` to define our infrastructure (Postgres).

```yaml
version: '3.8'
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: backend_course
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d backend_course"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### 3. Start the Database
Run the database container in the background:
```bash
cd day01
docker-compose up -d
```

Verify it's running:
```bash
docker ps
```
You should see a container named `day01-db-1` (or similar) with status `Up`.

### 4. Python Connection Script
We will write a script to verify we can talk to the database.

**Setup Python Environment:**
```bash
# Inside day01/ directory
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install psycopg2-binary
```

**Create Script:**
Create `day01/src/check_db.py`:

```python
import psycopg2
import time
import sys

def connect():
    print("🔌 Connecting to database...")
    try:
        conn = psycopg2.connect(
            dbname="backend_course",
            user="user",
            password="password",
            host="localhost",
            port="5432"
        )
        print("✅ Successfully connected to Postgres!")
        
        # Create a cursor to execute SQL commands
        cur = conn.cursor()
        
        # Execute a simple query
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"ℹ️  Database Version: {db_version[0]}")
        
        # Create a dummy table to prove write access
        cur.execute("CREATE TABLE IF NOT EXISTS access_log (id SERIAL PRIMARY KEY, access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP);")
        cur.execute("INSERT INTO access_log DEFAULT VALUES;")
        conn.commit()
        print("✅ Successfully wrote to the database (created table & inserted row).")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Simple retry logic in case DB is still starting up
    max_retries = 5
    for i in range(max_retries):
        try:
            connect()
            break
        except Exception:
            if i < max_retries - 1:
                print(f"⚠️ Connection failed, retrying in 2 seconds... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("❌ Could not connect after multiple attempts.")
```

### 5. Run the Lab
Execute the script:
```bash
python src/check_db.py
```

**Expected Output:**
```text
🔌 Connecting to database...
✅ Successfully connected to Postgres!
ℹ️  Database Version: PostgreSQL 15.x ...
✅ Successfully wrote to the database (created table & inserted row).
```

### 6. Cleanup
To stop the database and save resources:
```bash
docker-compose down
```
To stop and **delete** the data volume (reset DB):
```bash
docker-compose down -v
```
