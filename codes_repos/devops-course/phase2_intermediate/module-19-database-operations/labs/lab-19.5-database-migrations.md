# Lab 19.5: Database Migrations

## Objective
Manage database schema migrations safely.

## Learning Objectives
- Use Flyway for migrations
- Implement versioned migrations
- Handle rollbacks
- Test migrations

---

## Flyway Setup

```bash
# Install Flyway
wget https://repo1.maven.org/maven2/org/flywaydb/flyway-commandline/9.0.0/flyway-commandline-9.0.0-linux-x64.tar.gz
tar -xzf flyway-commandline-9.0.0-linux-x64.tar.gz

# Configure
cat > flyway.conf << 'EOF'
flyway.url=jdbc:postgresql://localhost:5432/mydb
flyway.user=admin
flyway.password=secret
EOF
```

## Create Migrations

```sql
-- V1__Create_users_table.sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50) NOT NULL,
  email VARCHAR(100) NOT NULL
);

-- V2__Add_created_at.sql
ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NOW();
```

## Run Migrations

```bash
# Migrate
flyway migrate

# Check status
flyway info

# Validate
flyway validate
```

## Rollback Strategy

```sql
-- V3__Add_status.sql
ALTER TABLE users ADD COLUMN status VARCHAR(20);

-- U3__Rollback_status.sql (undo)
ALTER TABLE users DROP COLUMN status;
```

## Success Criteria
✅ Flyway configured  
✅ Migrations applied  
✅ Schema versioned  
✅ Rollback tested  

**Time:** 40 min
