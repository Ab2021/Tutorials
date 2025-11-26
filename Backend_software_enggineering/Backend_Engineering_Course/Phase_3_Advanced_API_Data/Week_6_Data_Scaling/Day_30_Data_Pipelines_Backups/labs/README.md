# Lab: Day 30 - Database Migrations with Alembic

## Goal
Manage database schema changes as code.

## Prerequisites
- `pip install alembic sqlalchemy`
- Docker (Postgres).

## Step 1: Initialize
```bash
alembic init alembic
```
*   Creates `alembic.ini` and `alembic/` folder.

## Step 2: Configure (`alembic.ini`)
Edit `sqlalchemy.url` to point to your local Postgres:
`sqlalchemy.url = postgresql://user:password@localhost/mydb`

## Step 3: Create First Migration
```bash
alembic revision -m "create users table"
```
Edit the generated file in `alembic/versions/`:

```python
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False),
    )

def downgrade():
    op.drop_table('users')
```

## Step 4: Apply
```bash
alembic upgrade head
```
*   Check DB: Table `users` exists.

## Step 5: Modify Schema
```bash
alembic revision -m "add email column"
```
Edit file:
```python
def upgrade():
    op.add_column('users', sa.Column('email', sa.String(100)))

def downgrade():
    op.drop_column('users', 'email')
```

## Step 6: Apply & Rollback
1.  **Apply**: `alembic upgrade head`. (Column added).
2.  **Rollback**: `alembic downgrade -1`. (Column removed).

## Challenge
Implement a **Data Migration**.
Create a migration that not only adds a column `full_name` but also populates it by combining `first_name` and `last_name` (assuming those existed).
*   *Hint*: Use `op.execute("UPDATE users SET ...")`.
