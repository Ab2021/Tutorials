# Day 39: Design Google Drive

## 1. Requirements
*   **Functional:** Upload/Download files, Sync across devices, Versioning, Sharing.
*   **Non-Functional:** Reliability (Never lose data), Consistency (Sync).
*   **Scale:** 1 Billion users. 50GB per user.

## 2. Architecture
*   **Block Server:** Splits files into blocks (4MB). Hashes them. Uploads to S3.
*   **Metadata DB:** Stores file hierarchy (`/folder/file.txt`) and block list (`file.txt -> [hash1, hash2]`).
*   **Synchronization Service:** Notifies clients of changes.
*   **Cold Storage:** Move old files to Glacier.

## 3. Sync Protocol (Differential Sync)
*   **Scenario:** User changes 1 line in a 10MB file.
*   **Naive:** Upload 10MB. (Slow).
*   **Smart:**
    *   Client splits file into blocks.
    *   Calculates hashes.
    *   Sends only the *changed* block.
    *   Server updates the metadata to point to new block.

## 4. Consistency
*   **Client A** writes `ver 1`.
*   **Client B** writes `ver 1`.
*   **Conflict:** Server detects version mismatch.
*   **Resolution:** Last Write Wins OR Create "Conflicted Copy".
