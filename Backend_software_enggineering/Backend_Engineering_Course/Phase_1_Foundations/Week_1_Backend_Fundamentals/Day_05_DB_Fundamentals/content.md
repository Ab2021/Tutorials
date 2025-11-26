# Day 5: Database Fundamentals Overview

## 1. The Data Layer Landscape

The most critical decision in backend engineering is **where to store the data**. In 2025, we have an embarrassment of riches.

### 1.1 The Main Categories
1.  **Relational (RDBMS)**: Postgres, MySQL, SQLite.
    *   *Data Model*: Rows and Columns in Tables. Strict Schema.
    *   *Strength*: ACID transactions, complex joins, data integrity.
    *   *Use Case*: Core business data (Users, Orders, Billing).
2.  **NoSQL - Document**: MongoDB, DynamoDB.
    *   *Data Model*: JSON-like documents. Flexible Schema.
    *   *Strength*: Fast iteration, hierarchical data, horizontal scaling.
    *   *Use Case*: Content management, catalogs, user profiles.
3.  **NoSQL - Key-Value**: Redis, Memcached.
    *   *Data Model*: Map<Key, Value>.
    *   *Strength*: Microsecond latency (RAM-based).
    *   *Use Case*: Caching, session storage, rate limiting.
4.  **Vector Databases**: Pinecone, Qdrant, Weaviate, pgvector.
    *   *Data Model*: High-dimensional float arrays (embeddings).
    *   *Strength*: Semantic search ("Find images that look like this").
    *   *Use Case*: RAG, Recommendation systems.
5.  **Time-Series**: InfluxDB, TimescaleDB.
    *   *Data Model*: Timestamp + Value.
    *   *Strength*: High write throughput, time-based aggregation.
    *   *Use Case*: IoT metrics, server monitoring, financial ticks.

---

## 2. ACID Transactions

If you are handling money, you need ACID.

*   **Atomicity**: "All or nothing". If you transfer money from A to B, two updates happen (A-10, B+10). If the power fails halfway, the transaction rolls back. Money is never lost.
*   **Consistency**: The database moves from one valid state to another. Constraints (Foreign Keys, Unique) are enforced.
*   **Isolation**: Concurrent transactions don't interfere. (See Isolation Levels in Week 2).
*   **Durability**: Once committed, it stays committed, even if the server catches fire (Write-Ahead Log).

---

## 3. The CAP Theorem (Revisited)

We touched on this in Day 1, but let's map it to databases.
*   **CP (Consistency + Partition Tolerance)**:
    *   *Examples*: HBase, MongoDB (default), Redis Cluster.
    *   *Behavior*: If a node goes down, the system might reject writes to prevent data divergence.
*   **AP (Availability + Partition Tolerance)**:
    *   *Examples*: Cassandra, DynamoDB, CouchDB.
    *   *Behavior*: If a node goes down, other nodes accept writes. You might read stale data for a moment (Eventual Consistency).
*   **CA (Consistency + Availability)**:
    *   *Examples*: Traditional RDBMS (Postgres/MySQL) *on a single machine*.
    *   *Note*: CA doesn't exist in distributed systems because Partitions (P) are inevitable.

---

## 4. Choosing the Right Database (Decision Matrix)

| Requirement | Recommended DB Type | Example |
| :--- | :--- | :--- |
| **Financial Transactions** | Relational (ACID) | Postgres |
| **Product Catalog (Unstructured)** | Document | MongoDB |
| **High Speed Caching** | Key-Value | Redis |
| **Semantic Search / AI** | Vector | Qdrant / pgvector |
| **Server Metrics / IoT** | Time-Series | TimescaleDB |
| **Social Graph (Friends of Friends)** | Graph | Neo4j |

### 4.1 The "Postgres for Everything" Stack
In 2025, Postgres has become a "multi-model" database.
*   Need JSON? Use `JSONB` column (better than Mongo in many cases).
*   Need Vectors? Use `pgvector` extension.
*   Need Time-series? Use `TimescaleDB` extension.
*   *Advice*: Start with Postgres. Only spin up specialized DBs when you hit scale limits or need specific features Postgres lacks.

---

## 5. Summary

Today we surveyed the vast landscape of data persistence.
*   **ACID** is the gold standard for reliability.
*   **NoSQL** offers scale and flexibility but sacrifices strict consistency.
*   **Vector DBs** are the new kid on the block, essential for AI.

**Week 1 Wrap-Up**:
We have covered:
1.  Role of the Backend Engineer.
2.  Languages (Python/Go/Node).
3.  HTTP & Networking.
4.  REST API Design.
5.  Database Fundamentals.

**Next Week (Week 2)**: We will go deep into **Databases**. We will write complex SQL, learn about Indexes (B-Trees), normalize schemas, and then switch gears to NoSQL and Vector DBs.
