# Day 8: Distributed Consensus

## 1. The Problem
In a distributed system, how do multiple nodes agree on a single value (e.g., "Who is the leader?", "What is the next log entry?")?
*   **Challenge:** Nodes fail, messages get lost/delayed.

## 2. Paxos
*   **Status:** The "Holy Grail" of consensus.
*   **Roles:** Proposers, Acceptors, Learners.
*   **Mechanism:** Prepare Phase -> Promise Phase -> Accept Phase -> Accepted Phase.
*   **Pros:** Mathematically proven.
*   **Cons:** Extremely hard to understand and implement correctly.

## 3. Raft
*   **Goal:** Understandability. (Designed to be easier than Paxos).
*   **Roles:** Leader, Follower, Candidate.
*   **Mechanism:**
    1.  **Leader Election:** Nodes start as Followers. If no heartbeat, become Candidate. Request votes. Majority wins.
    2.  **Log Replication:** Leader accepts client command, appends to log, sends `AppendEntries` to Followers. Once majority replicate, Leader commits.
*   **Used by:** Etcd, Consul, CockroachDB.

## 4. Zab (Zookeeper Atomic Broadcast)
*   **Used by:** Apache Zookeeper (Kafka depends on this).
*   **Focus:** Total ordering of updates.
*   **Mechanism:** Similar to Raft (Leader-based).
