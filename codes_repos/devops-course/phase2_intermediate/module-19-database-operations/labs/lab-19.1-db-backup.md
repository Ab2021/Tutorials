# Lab 19.1: Database Backup & Restore (PostgreSQL)

## 🎯 Objective

Don't lose data. The most important task of a DevOps engineer is ensuring data durability. You will run PostgreSQL in Docker, populate it with data, perform a logical backup (`pg_dump`), drop the database, and restore it.

## 📋 Prerequisites

-   Docker installed.

## 📚 Background

### Backup Types
1.  **Logical (`pg_dump`)**: Exports SQL commands (`CREATE TABLE`, `INSERT`). Slow, but portable across versions.
2.  **Physical (WAL Archiving)**: Copies the actual files on disk. Fast, but version-specific.

---

## 🔨 Hands-On Implementation

### Part 1: Start Database 🐘

1.  **Run Postgres:**
    ```bash
    docker run -d \
      --name postgres-db \
      -e POSTGRES_PASSWORD=password123 \
      -v pgdata:/var/lib/postgresql/data \
      postgres:14
    ```

2.  **Connect & Add Data:**
    ```bash
    docker exec -it postgres-db psql -U postgres
    ```
    ```sql
    CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(50));
    INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie');
    SELECT * FROM users;
    \q
    ```

### Part 2: The Backup (Dump) 💾

1.  **Run pg_dump:**
    We run the command *inside* the container (or use a client container) and pipe the output to a file on the *host*.
    ```bash
    docker exec postgres-db pg_dump -U postgres postgres > backup.sql
    ```

2.  **Verify:**
    ```bash
    cat backup.sql
    ```
    *Result:* You should see SQL statements.

### Part 3: The Disaster 💥

1.  **Drop the Table:**
    ```bash
    docker exec -it postgres-db psql -U postgres -c "DROP TABLE users;"
    ```

2.  **Verify Data Loss:**
    ```bash
    docker exec -it postgres-db psql -U postgres -c "SELECT * FROM users;"
    ```
    *Result:* `ERROR: relation "users" does not exist`.

### Part 4: The Restore 🚑

1.  **Restore:**
    Pipe the file back into `psql`.
    ```bash
    cat backup.sql | docker exec -i postgres-db psql -U postgres postgres
    ```

2.  **Verify Recovery:**
    ```bash
    docker exec -it postgres-db psql -U postgres -c "SELECT * FROM users;"
    ```
    *Result:* Alice, Bob, and Charlie are back!

---

## 🎯 Challenges

### Challenge 1: Automated Backups (Difficulty: ⭐⭐)

**Task:**
Write a bash script `backup.sh` that:
1.  Creates a backup with a timestamp (e.g., `backup-2023-10-27.sql`).
2.  Deletes backups older than 7 days.
3.  Add it to `crontab` to run every night at 3 AM.

### Challenge 2: Point-in-Time Recovery (PITR) (Difficulty: ⭐⭐⭐⭐)

**Task:**
Research **WAL Archiving**.
Configure Postgres to archive WAL files to a local directory.
This allows you to restore to *exactly* 14:05:00, not just the last nightly backup.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
#!/bin/bash
DATE=$(date +%Y-%m-%d)
docker exec postgres-db pg_dump -U postgres postgres > /backups/backup-$DATE.sql
find /backups -name "*.sql" -mtime +7 -delete
```
</details>

---

## 🔑 Key Takeaways

1.  **Test Your Backups**: A backup is useless if you can't restore it. Test restoration regularly (Schrodinger's Backup).
2.  **Offsite Storage**: Storing `backup.sql` on the same server as the DB is bad. Upload it to S3.
3.  **Consistency**: `pg_dump` ensures a consistent snapshot of the DB at the time it started.

---

## ⏭️ Next Steps

Backups are for disasters. Replication is for uptime.

Proceed to **Lab 19.2: Database Replication**.
