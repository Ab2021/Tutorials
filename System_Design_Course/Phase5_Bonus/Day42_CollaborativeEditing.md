# Day 42: Collaborative Editing (Google Docs)

## 1. Requirements
*   **Functional:** Multiple users edit the same document simultaneously. See each other's changes in real-time.
*   **Non-Functional:** Low latency (< 100ms), Eventual Consistency, Conflict Resolution.
*   **Scale:** 100 concurrent editors per document.

## 2. The Challenge
*   **Scenario:** User A types "Hello" at position 0. User B types "World" at position 0.
*   **Naive Approach:** Last Write Wins. User A's "Hello" disappears.
*   **Problem:** Data loss. Unacceptable.

## 3. Operational Transformation (OT)
*   **Concept:** Transform operations based on concurrent operations.
*   **Example:**
    *   Initial: `""`
    *   User A: `Insert("H", 0)` -> `"H"`
    *   User B: `Insert("W", 0)` -> `"W"`
    *   **Server receives A's op first:** State = `"H"`.
    *   **Server receives B's op:** Transform `Insert("W", 0)` based on A's op.
        *   New position = `0 + 1 = 1` (because A inserted 1 char before position 0).
        *   Result: `Insert("W", 1)` -> `"HW"`.
*   **Complexity:** Hard to implement correctly. Google Wave tried and failed.

## 4. CRDT (Conflict-Free Replicated Data Types)
*   **Concept:** Data structures that automatically merge without conflicts.
*   **Example:** Each character has a unique ID (timestamp + user).
*   **Merge:** Sort by ID. Always converges to the same state.
*   **Pros:** Simpler than OT. Works offline.
*   **Cons:** Metadata overhead (each char has ID).
*   **Used by:** Figma, Apple Notes.
