# Day 5: Database Fundamentals - Choosing the Right Data Store

## Table of Contents
1. [Database Classification](#1-database-classification)
2. [Relational Databases (SQL)](#2-relational-databases-sql)
3. [NoSQL Databases](#3-nosql-databases)
4. [Vector Databases](#4-vector-databases)
5. [ACID vs BASE](#5-acid-vs-base)
6. [CAP Theorem Applied](#6-cap-theorem-applied)
7. [Data Modeling Fundamentals](#7-data-modeling-fundamentals)
8. [Choosing the Right Database](#8-choosing-the-right-database)
9. [Production Considerations](#9-production-considerations)
10. [Summary](#10-summary)

---

## 1. Database Classification

### 1.1 The Database Landscape (2025)

```mermaid
graph TB
    DB[Databases]
    DB --> Relational[Relational SQL]
    DB --> NoSQL[NoSQL]
    DB --> Search[Search Engines]
    DB --> Vector[Vector Databases]
    DB --> TimeSeries[Time-Series]
    DB --> Graph[Graph Databases]
    
    Relational --> Postgres[Postgres]
    Relational --> MySQL[MySQL]
    
    NoSQL --> Doc[Document - MongoDB]
    NoSQL --> KV[Key-Value - Redis]
    NoSQL --> Wide[Wide-Column - Cassandra]
    
    Vector --> Pinecone[Pinecone]
    Vector --> Qdrant[Qdrant]
    
    Search --> Elastic[Elasticsearch]
    TimeSeries --> Influx[InfluxDB]
    Graph --> Neo4j[Neo4j]
```

### 1.2 Decision Tree

```mermaid
graph TD
    Start[Choose Database] --> Need{What's Primary Need?}
    
    Need -->|Transactions| ACID{Need ACID?}
    ACID -->|Yes| Postgres[Postgres/MySQL]
    ACID -->|No| Cassandra[Cassandra]
    
    Need -->|Caching| InMemory{Need Persistence?}
    InMemory -->|Yes| Redis[Redis + AOF]
    InMemory -->|No| Memcached[Memcached]
    
    Need -->|Search| Vector{Semantic Search?}
    Vector -->|Yes| Pinecone[Pinecone/Qdrant]
    Vector -->|No| Elastic[Elasticsearch]
    
    Need -->|Analytics| Stream{Real-time?}
    Stream -->|Yes| ClickHouse[ClickHouse]
    Stream -->|No| Snowflake[Snowflake/BigQuery]
```

---

## 2. Relational Databases (SQL)

### 2.1 What Makes a Database "Relational"?

**Core Concepts**:
1. **Tables**: Structured data in rows/columns
2. **Schemas**: Defined structure enforced
3. **Relationships**: Foreign keys link tables
4. **SQL**: Declarative query language
5. **ACID**: Strong consistency guarantees

### 2.2 Example Schema

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    published_at TIMESTAMP
);

CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    post_id INTEGER REFERENCES posts(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    text TEXT NOT NULL
);
```

**Relationships**:
- User **has many** Posts (`user_id → users.id`)
- Post **has many** Comments (`post_id → posts.id`)

### 2.3 SQL Query Basics

#### SELECT
```sql
-- Get all users
SELECT * FROM users;

-- Get specific columns
SELECT id, email FROM users;

-- Filter rows
SELECT * FROM users WHERE created_at > '2024-01-01';

-- Order results
SELECT * FROM users ORDER BY created_at DESC;

-- Limit results
SELECT * FROM users LIMIT 10 OFFSET 20;
```

#### JOINs
```sql
-- Get posts with author info
SELECT 
    posts.title,
    users.email AS author_email
FROM posts
JOIN users ON posts.user_id = users.id;

-- LEFT JOIN (include posts without authors)
SELECT 
   posts.title,
    users.email
FROM posts
LEFT JOIN users ON posts.user_id = users.id;
```

#### Aggregation
```sql
-- Count posts per user
SELECT 
    users.email,
    COUNT(posts.id) AS post_count
FROM users
LEFT JOIN posts ON users.id = posts.user_id
GROUP BY users.id, users.email;
```

### 2.4 Pros & Cons

**Strengths**:
- ✅ **ACID Transactions**: Data integrity (bank transfers)
- ✅ **Complex Queries**: JOINs across multiple tables
- ✅ **Mature Tooling**: ORMs, migration tools, GUIs
- ✅ **Data Integrity**: Foreign key constraints

**Weaknesses**:
- ❌ **Rigid Schema**: Hard to evolve (add column = migrations)
- ❌ **Vertical Scaling**: Limited to single machine power
- ❌ **Performance**: Joins expensive on large datasets

**Use Cases**:
- 🎯 Transactional systems (e-commerce, banking)
- 🎯 Complex reporting
- 🎯 Data with clear relationships

---

## 3. NoSQL Databases

### 3.1 Document Stores (MongoDB, Couchbase)

**Data Model**: Store JSON-like documents.

```javascript
// User document
{
  "_id": "user_123",
  "email": "alice@example.com",
  "profile": {
    "name": "Alice",
    "age": 30
  },
  "posts": [
    {"title": "First Post", "content": "..."},
    {"title": "Second Post", "content": "..."}
  ]
}
```

**Query (MongoDB)**:
```javascript
// Find user
db.users.findOne({email: "alice@example.com"})

// Find users older than 25
db.users.find({"profile.age": {$gt: 25}})

// Update
db.users.updateOne(
  {_id: "user_123"},
  {$set: {"profile.age": 31}}
)
```

**Pros**:
- ✅ **Flexible Schema**: Add fields without migrations
- ✅ **Denormalization**: Embed related data (fast reads)
- ✅ **Horizontal Scaling**: Sharding built-in

**Cons**:
- ❌ **No Joins**: Must manually fetch related docs
- ❌ **Eventual Consistency**: By default
- ❌ **Data Duplication**: Denormalization = redundancy

**Use Cases**:
- 🎯 Content management (blogs, catalogs)
- 🎯 Rapidly evolving schemas
- 🎯 High write throughput

### 3.2 Key-Value Stores (Redis, DynamoDB)

**Data Model**: Simple hash map.

```
SET user:123:name "Alice"
SET user:123:email "alice@example.com"
GET user:123:name  → "Alice"
```

**Redis Data Structures**:
```redis
# String
SET key "value"

# Hash (object)
HSET user:123 name "Alice" email "alice@example.com"
HGET user:123 name  → "Alice"

# List
LPUSH queue:jobs "job1" "job2"
RPOP queue:jobs  → "job1"

# Set (unique values)
SADD tags:post:456 "python" "redis"
SMEMBERS tags:post:456  → ["python", "redis"]

# Sorted Set (leaderboard)
ZADD leaderboard 100 "player1" 200 "player2"
ZRANGE leaderboard 0 -1  → ["player1", "player2"]
```

**Pros**:
- ✅ **Blazing Fast**: In-memory (microsecond latency)
- ✅ **Simple**: Get/Set operations
- ✅ **TTL**: Auto-expire keys

**Cons**:
- ❌ **No Complex Queries**: Can't filter/join
- ❌ **Memory Cost**: RAM is expensive
- ❌ **Size Limits**: Redis max value ~512 MB

**Use Cases**:
- 🎯 Caching
- 🎯 Session storage
- 🎯 Rate limiting
- 🎯 Leaderboards

### 3.3 Wide-Column Stores (Cassandra, HBase)

**Data Model**: Rows with dynamic columns.

```
Table: user_activity
Row Key: user_123

Columns:
  login:2024-01-01 → "192.168.1.1"
  login:2024-01-02 → "192.168.1.2"
  action:2024-01-01 → "clicked_button"
```

**CQL (Cassandra Query Language)**:
```sql
CREATE TABLE user_activity (
    user_id UUID,
    timestamp TIMESTAMP,
    action TEXT,
    PRIMARY KEY (user_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

INSERT INTO user_activity (user_id, timestamp, action)
VALUES (uuid(), '2024-01-01 12:00:00', 'login');

SELECT * FROM user_activity WHERE user_id = uuid() AND timestamp > '2024-01-01';
```

**Pros**:
- ✅ **Massive Scale**: Petabytes of data
- ✅ **High Availability**: Multi-datacenter replication
- ✅ **Write Optimized**: LSM trees for fast writes

**Cons**:
- ❌ **Eventual Consistency**: Tunable but default is AP
- ❌ **Limited Queries**: Must design schema for access patterns
- ❌ **Operational Complexity**: Difficult to operate

**Use Cases**:
- 🎯 Time-series data (IoT sensors)
- 🎯 Audit logs
- 🎯 Massive write throughput

---

## 4. Vector Databases

### 4.1 The AI Era Database

**Purpose**: Store and search high-dimensional vectors (embeddings).

**Use Case**: "Find products similar to this image."

```python
# Text to vector (embedding)
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Python programming tutorial"
)
vector = response.data[0].embedding  # [-0.02, 0.45, ..., 0.12] (1536 dims)
```

### 4.2 Vector Search (Similarity)

**Problem**: Traditional SQL can't do "find similar":
```sql
❌ SELECT * FROM products WHERE embedding = [0.1, 0.2, ...] -- No exact match
```

**Solution**: Vector databases use distance metrics.

**Distance Metrics**:
1. **Cosine Similarity**: Angle between vectors (ignores magnitude)
2. **Euclidean Distance**: Straight-line distance
3. **Dot Product**: Raw similarity

```python
# Qdrant example
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

client = QdrantClient("localhost", port=6333)

# Store vectors
client.upsert(
    collection_name="products",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, ...],  # Product embedding
            payload={"name": "Laptop", "price": 999}
        )
    ]
)

# Search
results = client.search(
    collection_name="products",
    query_vector=[0.12, 0.19, 0.31, ...],  # Query embedding
    limit=5
)
```

### 4.3 Approximate Nearest Neighbor (ANN)

**Exact Search**: Check every vector (slow for millions of vectors).

**Approximate Search (HNSW)**:
```
Build a graph where similar vectors are connected.
Search by jumping along edges (logarithmic time).
```

**Trade-off**: 95% accuracy, 100x faster.

### 4.4 Hybrid Search

**Problem**: Keyword + semantic search together.

```python
# Combine results
keyword_results = elasticsearch.search("laptop")
semantic_results = qdrant.search(embed("laptop"))

# Reciprocal Rank Fusion (RRF)
combined = merge_results(keyword_results, semantic_results)
```

**Use Cases**:
- 🎯 Semantic search (Google-like)
- 🎯 Recommendation engines
- 🎯 Image search
- 🎯 RAG for LLMs

---

## 5. ACID vs BASE

### 5.1 ACID (SQL Databases)

**Atomicity**: All-or-nothing transactions.
```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT; -- Both updates or neither
```

**Consistency**: Data follows constraints.
```sql
CREATE TABLE users (
    age INT CHECK (age >= 0)
);
INSERT INTO users (age) VALUES (-5);  -- Error!
```

**Isolation**: Concurrent transactions don't interfere.
```sql
-- Transaction 1: Read balance
SELECT balance FROM accounts WHERE id = 1;  → 100

-- Transaction 2 (concurrent): Updates balance
UPDATE accounts SET balance = 200 WHERE id = 1;

-- Transaction 1: Read again (Repeatable Read isolation)
SELECT balance FROM accounts WHERE id = 1;  → Still 100 (not 200)
```

**Durability**: Committed data survives crashes.
```sql
COMMIT;
-- Power outage happens here
-- Data still persists (WAL log)
```

### 5.2 BASE (NoSQL Databases)

**Basically Available**: System responds (maybe with stale data).

**Soft state**: Data may change without input (replicas syncing).

**Eventually consistent**: Given time, all replicas converge.

**Example (Cassandra)**:
```
Write to Replica A: user.name = "Alice"
→ Returns success immediately

Read from Replica B (100ms later): user.name = "Bob" (stale!)
Read from Replica B (500ms later): user.name = "Alice" (synced!)
```

---

## 6. CAP Theorem Applied

### 6.1 Real-World Examples

**CP System (MongoDB with majority writes)**:
```javascript
db.users.insertOne(
    {email: "alice@example.com"},
    {writeConcern: {w: "majority"}}
)
// Blocks until majority of replicas acknowledge
// If network partition, rejects writes to minority side
```

**AP System (Cassandra)**:
```sql
INSERT INTO users (id, email) VALUES (1, 'alice@example.com');
-- Returns immediately
-- If network partition, both sides accept writes
-- Conflict resolution: Last Write Wins (LWW) or timestamp-based
```

### 6.2 Tunable Consistency (Cassandra)

```sql
-- Strong consistency (CP-like)
SELECT * FROM users WHERE id = 1 USING CONSISTENCY  QUORUM;

-- Weak (AP)
SELECT * FROM users WHERE id = 1 USING CONSISTENCY ONE;
```

**Tuning Formula**:
```
R + W > N (Strong consistency)
Where:
  R = Read replicas
  W = Write replicas
  N = Total replicas

Example: N=3, W=2, R=2
→ 2 + 2 > 3 (Always read latest write)
```

---

## 7. Data Modeling Fundamentals

### 7.1 Normalization (SQL)

**1st Normal Form (1NF)**: Atomic values, no arrays.
```sql
BAD:
users (id, emails) → (1, "alice@example.com,bob@example.com")

GOOD:
users (id, email)
  (1, "alice@example.com")
  (1, "bob@example.com")
```

**2nd Normal Form (2NF)**: No partial dependencies.

**3rd  Normal Form (3NF)**: No transitive dependencies.
```sql
BAD:
orders (id, customer_id, customer_name, customer_city)
-- customer_name depends on customer_id (transitive)

GOOD:
orders (id, customer_id)
customers (id, name, city)
```

### 7.2 Denormalization (NoSQL)

**Trade-off**: Faster reads, redundant data.

```javascript
// Normalized (SQL approach)
users: {id: 1, name: "Alice"}
posts: {id: 1, user_id: 1, title: "Post"}
// Need JOIN to get author name

// Denormalized (NoSQL approach)
posts: {
  id: 1,
  title: "Post",
  author: {id: 1, name: "Alice"}  // Embedded
}
// No JOIN needed
```

**Downside**: Update author name → must update all posts.

### 7.3 Schema Design Patterns

#### Pattern 1: Embedded Documents
```javascript
// Good for 1-to-few relationships
{
  "_id": "user_123",
  "name": "Alice",
  "addresses": [
    {"street": "123 Main", "city": "NYC"},
    {"street": "456 Oak", "city": "LA"}
  ]
}
```

#### Pattern 2: References
```javascript
// Good for 1-to-many (many is large)
posts: {
  "_id": "post_123",
  "title": "My Post",
  "comment_ids": ["c1", "c2", "c3"]  // Reference
}

comments: {
  "_id": "c1",
  "text": "Great post!"
}
```

---

## 8. Choosing the Right Database

### 8.1 Decision Matrix

| Requirement | Database Choice |
|:------------|:----------------|
| ACID transactions | Postgres, MySQL |
| Flexible schema | MongoDB, Couchbase |
| Caching / Session | Redis, Memcached |
| Semantic search | Pinecone, Qdrant, Weaviate |
| Full-text search | Elasticsearch, Typesense |
| Time-series | InfluxDB, TimescaleDB |
| Analytics (OLAP) | ClickHouse, BigQuery |
| Graph relationships | Neo4j, ArangoDB |
| High write throughput | Cassandra, ScyllaDB |

### 8.2 Polyglot Persistence

**Use multiple databases**:

```mermaid
graph LR
    App[App] --> Postgres[(Postgres: Users, Orders)]
    App --> Redis[(Redis: Cache, Sessions)]
    App --> Elastic[(Elasticsearch: Product Search)]
    App --> Qdrant[(Qdrant: Semantic Search)]
    App --> S3[(S3: Images, Videos)]
```

**Example: E-commerce**
- **Postgres**: User accounts, orders (ACID)
- **Redis**: Shopping carts, rate limiting
- **Elasticsearch**: Product catalog search
- **Qdrant**: "Find similar products"
- **S3**: Product images

---

## 9. Production Considerations

### 9.1 Backups

**Postgres**:
```bash
# Full backup
pg_dump dbname > backup.sql

# Restore
psql dbname < backup.sql

# Point-in-time recovery
# Enable WAL archiving in postgresql.conf
archive_mode = on
archive_command = 'cp %p /backups/%f'
```

**MongoDB**:
```bash
mongodump --db mydb --out /backups/
mongorestore /backups/mydb
```

### 9.2 Replication

**Primary-Replica (Postgres)**:
```
Primary (Writes) →  Replica 1 (Reads)
                 →  Replica 2 (Reads)
```

**Benefits**:
- Read scaling
- High availability (promote replica if primary fails)

**Lag**: Replica might be 100ms-1s behind primary.

### 9.3 Connection Pooling

**Problem**: Opening DB connections is slow (SSL handshake, auth).

**Solution**: Connection pool (PgBouncer, HikariCP).

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=20,         # Keep 20 open
    max_overflow=10,      # Allow 10 more if needed
    pool_pre_ping=True    # Check if connection alive
)
```

### 9.4 Indexing

**Without index**:
```sql
SELECT * FROM users WHERE email = 'alice@example.com';
-- Scans all 1 million rows (slow!)
```

**With index**:
```sql
CREATE INDEX idx_users_email ON users(email);
SELECT * FROM users WHERE email = 'alice@example.com';
-- Uses B-tree index (fast!)
```

**Cost**: Indexes slow down writes (must update index on INSERT/UPDATE).

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **SQL for ACID** - Banking, e-commerce orders
2. ✅ **NoSQL for flexibility** - CMS, rapidly changing schemas
3. ✅ **Redis for speed** - Caching, sessions, rate limiting
4. ✅ **Vector DBs for AI** - Semantic search, RAG
5. ✅ **Polyglot persistence** - Use multiple DBs
6. ✅ **CAP theorem limits** - Can't have all three (C, A, P)

### 10.2 The Polyglot Recommendation

**Typical modern stack**:
- **Postgres**: Primary transactional data
- **Redis**: Caching layer
- **Elasticsearch**: Full-text search
- **Qdrant**: Semantic search (AI)
- **S3**: Blob storage

### 10.3 Tomorrow (Day 6): SQL Deep Dive

We'll master SQL:
- **Joins**: INNER, LEFT, RIGHT, FULL
- **Subqueries**: Correlated vs non-correlated
- **Window Functions**: ROW_NUMBER, RANK
- **Transactions**: SERIALIZABLE isolation
- **Query Optimization**: EXPLAIN ANALYZE

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | Database fundamentals mastered ✅
