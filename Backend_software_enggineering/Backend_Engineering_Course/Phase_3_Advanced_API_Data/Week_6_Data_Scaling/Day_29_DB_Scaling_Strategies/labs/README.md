# Lab: Day 29 - Master-Slave Replication

## Goal
Set up a real Postgres Replication cluster using Docker.

## Directory Structure
```
day29/
├── docker-compose.yml
├── setup_master.sh
└── README.md
```

## Step 1: Docker Compose (`docker-compose.yml`)

```yaml
version: '3'
services:
  master:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    command: |
      postgres -c wal_level=replica -c hot_standby=on -c max_wal_senders=10 -c max_replication_slots=10 -c hot_standby_feedback=on
    volumes:
      - ./master_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  slave:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    depends_on:
      - master
    command: |
      bash -c "
      rm -rf /var/lib/postgresql/data/*
      until pg_isready -h master -p 5432 -U user; do sleep 1; done
      PGPASSWORD=password pg_basebackup -h master -D /var/lib/postgresql/data -U user -v -P --wal-method=stream
      echo \"primary_conninfo = 'host=master port=5432 user=user password=password'\" >> /var/lib/postgresql/data/postgresql.conf
      touch /var/lib/postgresql/data/standby.signal
      postgres
      "
    ports:
      - "5433:5432"
```

## Step 2: Run It
```bash
docker-compose up
```
*   **Master** starts on port 5432.
*   **Slave** starts on port 5433, pulls data from Master, and starts in Read-Only mode.

## Step 3: Verify

1.  **Write to Master**:
    ```bash
    PGPASSWORD=password psql -h localhost -p 5432 -U user -d mydb -c "CREATE TABLE foo (id int); INSERT INTO foo VALUES (1);"
    ```

2.  **Read from Slave**:
    ```bash
    PGPASSWORD=password psql -h localhost -p 5433 -U user -d mydb -c "SELECT * FROM foo;"
    ```
    *   You should see `id: 1`.

3.  **Try Write to Slave**:
    ```bash
    PGPASSWORD=password psql -h localhost -p 5433 -U user -d mydb -c "INSERT INTO foo VALUES (2);"
    ```
    *   *Error*: `cannot execute INSERT in a read-only transaction`.

## Challenge
Simulate **Replication Lag**.
Use `tc` (Traffic Control) in the slave container to add 500ms network delay.
Observe how long it takes for data to appear.
