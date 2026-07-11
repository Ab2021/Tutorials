# Database Schema

We use **PostgreSQL 16** for persistent storage and **Redis 7** for ephemeral session state.

Given our 8GB RAM constraint, PostgreSQL is configured conservatively (`shared_buffers=128MB`, `max_connections=20`).

## PostgreSQL Tables (SQLAlchemy)

No complex ORM mappings are needed. The schema is intentionally flattened for simplicity.

### 1. Users

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Interviews

```sql
CREATE TABLE interviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    job_title VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- pending, active, completed
    config_json JSONB, -- stores difficulty, interview type, etc.
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    score FLOAT
);
```

### 3. Transcripts

```sql
CREATE TABLE transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID REFERENCES interviews(id),
    speaker VARCHAR(50) NOT NULL, -- 'candidate' or 'interviewer'
    content TEXT NOT NULL,
    timestamp_s FLOAT NOT NULL
);
```

### 4. Video Metrics (Client-Side Telemetry)

```sql
CREATE TABLE video_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID REFERENCES interviews(id),
    timestamp_s FLOAT NOT NULL,
    engagement FLOAT,
    emotions_json JSONB,
    eye_contact BOOLEAN
);
```

### 5. Reports

```sql
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interview_id UUID REFERENCES interviews(id),
    score FLOAT,
    feedback_json JSONB, -- Stores strengths, weaknesses, tips
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Redis (Session State)

Redis is used strictly for storing the state of active interviews. This prevents us from hammering PostgreSQL during high-frequency WebSocket events.

```text
# Key: session:{interview_id}
# Value (Hash):
{
  "status": "active",
  "current_question": 2,
  "history": "[ {role: user, content: ...}, ... ]"
}
```

## Backups

Because we don't have S3/MinIO configured on this small VPS, we run a simple `cron` job on the Hostinger VPS to run `pg_dump` every week, saving the `.sql` file locally to the `100GB SSD`.
