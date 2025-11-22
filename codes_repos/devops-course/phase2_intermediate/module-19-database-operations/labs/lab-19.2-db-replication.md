# Lab 19.2: Database Replication (Master-Slave)

## 🎯 Objective

Scale Reads and ensure High Availability. You will set up **Streaming Replication** between a Primary (Master) and a Standby (Slave) PostgreSQL instance.

## 📋 Prerequisites

-   Docker & Docker Compose.

## 📚 Background

### Concepts
-   **Primary**: Handles Writes (INSERT/UPDATE) and Reads.
-   **Standby**: Handles Reads only. Replicates data from Primary.
-   **Async Replication**: Primary confirms write immediately. Slave catches up (ms delay). Fast, small risk of data loss.
-   **Sync Replication**: Primary waits for Slave to confirm. Slow, zero data loss.

---

## 🔨 Hands-On Implementation

### Part 1: Docker Compose 🐳

We need a custom network and two containers.

1.  **Create `docker-compose.yml`:**
    ```yaml
    version: '3'
    services:
      primary:
        image: postgres:14
        environment:
          POSTGRES_PASSWORD: password
        command: |
          postgres 
          -c wal_level=replica 
          -c hot_standby=on 
          -c max_wal_senders=10 
          -c max_replication_slots=10 
          -c hot_standby_feedback=on
        volumes:
          - ./primary_data:/var/lib/postgresql/data
        ports:
          - "5432:5432"

      standby:
        image: postgres:14
        environment:
          POSTGRES_PASSWORD: password
        # We need to run a base backup script before starting postgres
        entrypoint: /bin/bash -c "chmod 0700 /var/lib/postgresql/data && rm -rf /var/lib/postgresql/data/* && PGPASSWORD=password pg_basebackup -h primary -D /var/lib/postgresql/data -U postgres -v -P -X stream -R && docker-entrypoint.sh postgres"
        depends_on:
          - primary
        ports:
          - "5433:5432"
    ```
    *Note:* The `standby` entrypoint is a hack for this lab. It clears its data directory and clones the primary using `pg_basebackup`.

### Part 2: Verify Replication 🔄

1.  **Start:**
    ```bash
    docker-compose up -d
    ```
    Wait 10-20 seconds.

2.  **Write to Primary:**
    ```bash
    docker exec -it <PRIMARY_CONTAINER_ID> psql -U postgres -c "CREATE TABLE repl_test (id int); INSERT INTO repl_test VALUES (1);"
    ```

3.  **Read from Standby:**
    ```bash
    docker exec -it <STANDBY_CONTAINER_ID> psql -U postgres -c "SELECT * FROM repl_test;"
    ```
    *Result:* You see `1`. It replicated!

### Part 3: Read-Only Check 🚫

1.  **Try Write to Standby:**
    ```bash
    docker exec -it <STANDBY_CONTAINER_ID> psql -U postgres -c "INSERT INTO repl_test VALUES (2);"
    ```
    *Result:* `ERROR: cannot execute INSERT in a read-only transaction`.
    **Success!** It is a true Read Replica.

---

## 🎯 Challenges

### Challenge 1: Promotion (Difficulty: ⭐⭐⭐)

**Task:**
Simulate Primary Failure.
1.  Stop the Primary container.
2.  Promote the Standby to be the new Primary.
    Command: `pg_ctl promote -D /var/lib/postgresql/data`.
3.  Now try writing to the (former) Standby. It should succeed.

### Challenge 2: Connection Pooling (Difficulty: ⭐⭐⭐)

**Task:**
Research **PgBouncer**.
Add a PgBouncer container to the compose file.
Configure it to route "Write" queries to Primary and "Read" queries to Standby (requires advanced config or app-side logic).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
You need to run the promote command inside the standby container.
`docker exec -it <STANDBY_ID> pg_ctl promote -D /var/lib/postgresql/data -U postgres`
</details>

---

## 🔑 Key Takeaways

1.  **Read Scaling**: If your app is 90% reads (like a blog), adding Read Replicas is the easiest way to scale.
2.  **Failover**: Automated failover (e.g., using Patroni) is complex. Manual failover is safer for beginners.
3.  **Lag**: Replication lag can happen. If you write to Master and immediately read from Slave, the data might not be there yet.

---

## ⏭️ Next Steps

We have handled Data. Now let's look at Architecture Patterns.

Proceed to **Module 20: Cloud Architecture Patterns**.
