# Day 8 Interview Prep: Consensus

## Q1: Paxos vs Raft?
**Answer:**
*   **Paxos:** The original consensus algorithm. Flexible but complex. Hard to implement.
*   **Raft:** Designed for understandability. Decomposes problem into Leader Election, Log Replication, and Safety. Equivalent fault tolerance to Paxos.

## Q2: What is Split Brain and how to prevent it?
**Answer:**
*   **Split Brain:** When a cluster splits into two sub-clusters that both think they are active.
*   **Prevention:** Quorum (Majority Vote). You need $N/2 + 1$ nodes to make a decision.

## Q3: Why does Zookeeper need an odd number of nodes?
**Answer:**
*   To survive $F$ failures, you need $2F+1$ nodes.
*   Example: 3 nodes can survive 1 failure. 4 nodes can also only survive 1 failure (need 3 for majority).
*   Adding the 4th node adds cost/complexity without adding fault tolerance.

## Q4: What is a Lease?
**Answer:**
*   A contract that gives a node exclusive rights to a resource for a limited time (TTL).
*   Used in Leader Election to avoid constant voting. "I am leader for 10 seconds".
