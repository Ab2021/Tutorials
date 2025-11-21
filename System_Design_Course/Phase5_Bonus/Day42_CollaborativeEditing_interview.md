# Day 42 Interview Prep: Collaborative Editing

## Q1: OT vs CRDT?
**Answer:**
*   **OT:** Central server transforms operations. Complex. Used by Google Docs.
*   **CRDT:** Peer-to-peer. Automatic merge. Simpler. Used by Figma, Notion.
*   **Trade-off:** OT is more efficient (less metadata). CRDT is easier to implement and works offline.

## Q2: How to handle large documents?
**Answer:**
*   **Chunking:** Split document into blocks (paragraphs).
*   **Lazy Loading:** Only load visible blocks.
*   **Pagination:** Server-side pagination for very large docs.

## Q3: What if two users delete the same text?
**Answer:**
*   **OT:** Transform delete operations. If both delete same range, second delete becomes no-op.
*   **CRDT:** Each character has tombstone marker. Both deletes mark it as deleted. Idempotent.

## Q4: How to implement "Undo"?
**Answer:**
*   **Local Undo:** Client maintains operation history.
*   **Inverse Operation:** For each op, generate inverse (Insert -> Delete).
*   **Challenge:** If other users made changes, inverse might not apply cleanly. Need to transform the inverse operation too.
