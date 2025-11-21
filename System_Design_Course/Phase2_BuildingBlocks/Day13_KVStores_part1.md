# Day 13 Deep Dive: DynamoDB Internals

## 1. The Dynamo Paper (2007)
The foundation of modern NoSQL.
*   **Goal:** "Always Writable" (High Availability).
*   **Techniques:**
    *   **Partitioning:** Consistent Hashing.
    *   **High Availability:** Vector Clocks (for conflict resolution).
    *   **Temporary Failures:** Hinted Handoff.
    *   **Permanent Failures:** Merkle Trees (Anti-entropy).
    *   **Membership:** Gossip Protocol.

## 2. Hinted Handoff (Sloppy Quorum)
*   **Scenario:** Node A is the coordinator for Key K. Node A is down.
*   **Action:** Write is sent to Node B (neighbor) with a "hint": "This belongs to A".
*   **Recovery:** When A comes back, B sends the data to A.
*   **Benefit:** Write succeeds even if owner is down.

## 3. Merkle Trees (Anti-Entropy)
*   **Problem:** How to sync data between replicas efficiently?
*   **Solution:** Merkle Tree (Hash Tree).
    *   Leaves are hashes of data blocks.
    *   Parents are hashes of children.
*   **Compare:** Compare Root Hash. If same, data is synced. If different, go down the tree to find the specific block that differs.
*   **Benefit:** Minimize data transfer during sync.

## 4. Case Study: Amazon Shopping Cart
*   **Requirement:** Never lose an "Add to Cart" event.
*   **Conflict:** User adds Item A on Phone. Adds Item B on Laptop. Network partition prevents sync.
*   **Resolution:**
    *   Both writes accepted (AP system).
    *   Cart state: `[Item A]` and `[Item B]`.
    *   **Read Repair:** When user views cart, system detects conflict (Vector Clocks).
    *   **Client Logic:** Merge items -> `[Item A, Item B]`.
