# Day 39 Deep Dive: Rsync & Deduplication

## 1. Rsync Algorithm (Rolling Hash)
*   **Goal:** Sync only changed bytes, even if offsets shift.
*   **Mechanism:**
    *   Calculate Rolling Hash for every window.
    *   Compare hashes.
    *   Send only non-matching chunks.

## 2. Global Deduplication
*   **Scenario:** 1000 users upload "Avengers.mp4".
*   **Hash:** `SHA256(Avengers.mp4)`.
*   **Check:** Does this hash exist in S3?
    *   **Yes:** Don't upload. Just add a reference in Metadata DB (`UserB -> AvengersHash`).
    *   **No:** Upload.
*   **Benefit:** Massive storage saving (Client-side dedupe saves bandwidth too).

## 3. Metadata Database
*   **Structure:** Filesystem is a Tree.
*   **SQL:** `ParentID` column. Recursive queries are slow.
*   **NoSQL (DynamoDB):** Flatten the tree.
    *   `PK: FolderID`, `SK: ItemName`.
    *   Fast listing of folder contents.

## 4. Notification Service
*   **Long Polling:** Client polls server: "Any changes since Cursor X?".
*   **Server:** "Yes, File Y changed".
*   **Client:** Downloads File Y metadata, then downloads changed blocks.
