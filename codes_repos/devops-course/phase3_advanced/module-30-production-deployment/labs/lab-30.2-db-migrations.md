# Lab 30.2: Zero Downtime Database Migrations

## 🎯 Objective

Change the engine while flying. Renaming a database column usually requires downtime. The **Expand and Contract** pattern allows you to make schema changes without stopping the application.

## 📋 Prerequisites

-   PostgreSQL (Docker).
-   Python (to simulate the app).

## 📚 Background

### The Problem
If you rename `address` to `billing_address`:
1.  Old App writes to `address`.
2.  Migration runs (Rename).
3.  Old App crashes (Column `address` missing).
4.  New App deploys.

### The Solution (4 Steps)
1.  **Expand**: Add new column (nullable).
2.  **Double Write**: App writes to *both* columns.
3.  **Backfill**: Copy old data to new column.
4.  **Contract**: Remove old column.

---

## 🔨 Hands-On Implementation

### Part 1: Setup 🐘

1.  **Start DB:**
    ```bash
    docker run -d --name mydb -e POSTGRES_PASSWORD=pass postgres
    ```

2.  **Create Table:**
    `docker exec -it mydb psql -U postgres -c "CREATE TABLE users (id SERIAL, name VARCHAR(50));"`

3.  **App v1 (Writes to Name):**
    ```python
    # app_v1.py
    def create_user(name):
        print(f"INSERT INTO users (name) VALUES ('{name}')")
    ```

### Part 2: Step 1 - Expand (Add Column) ➕

1.  **Migration:**
    ```sql
    ALTER TABLE users ADD COLUMN full_name VARCHAR(50);
    ```
    *Status:* App v1 still works (it ignores `full_name`).

### Part 3: Step 2 - Double Write (App v1.1) ✍️✍️

1.  **App v1.1:**
    ```python
    # app_v1_1.py
    def create_user(name):
        # Write to BOTH
        print(f"INSERT INTO users (name, full_name) VALUES ('{name}', '{name}')")
    ```
    *Deploy this version.* Now new data goes to both.

### Part 4: Step 3 - Backfill 🚜

1.  **Migration:**
    ```sql
    UPDATE users SET full_name = name WHERE full_name IS NULL;
    ```
    *Status:* Now all data is in `full_name`.

### Part 5: Step 4 - Contract (Remove Old) ➖

1.  **App v2 (Read from New):**
    ```python
    # app_v2.py
    def create_user(name):
        print(f"INSERT INTO users (full_name) VALUES ('{name}')")
    ```
    *Deploy this version.*

2.  **Migration (Final):**
    ```sql
    ALTER TABLE users DROP COLUMN name;
    ```
    *Status:* Done. No downtime.

---

## 🎯 Challenges

### Challenge 1: Not Null Constraint (Difficulty: ⭐⭐⭐)

**Task:**
In Step 1, `full_name` must be nullable.
In Step 4, after backfill, you want to make it `NOT NULL`.
Add the SQL command to add the constraint safely.

### Challenge 2: Large Table Backfill (Difficulty: ⭐⭐⭐⭐)

**Task:**
If the table has 1 Billion rows, `UPDATE users ...` will lock the table for hours.
Research how to backfill in batches (e.g., 1000 rows at a time).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
`ALTER TABLE users ALTER COLUMN full_name SET NOT NULL;` (Run this only after backfill is 100% complete).
</details>

---

## 🔑 Key Takeaways

1.  **Backwards Compatibility**: Every database change must be compatible with the *previous* version of the code.
2.  **Decoupling**: Never deploy code and run migrations in the same atomic step.
3.  **Patience**: A rename that used to take 5 minutes now takes 4 deployments and 1 week. But it guarantees 100% uptime.

---

## ⏭️ Next Steps

**CONGRATULATIONS!** 🎓
You have completed the **DevOps Master Course**.

You have built a platform from scratch, automated it, secured it, and learned how to operate it at scale.

**What now?**
1.  Build the Capstone Projects.
2.  Contribute to Open Source.
3.  Get Certified (CKA, AWS Pro).
4.  **Go Build Great Things!** 🚀
