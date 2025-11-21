# Day 39 Interview Prep: Design Google Drive

## Q1: How to handle large file uploads?
**Answer:**
*   **Multipart Upload:** Split into chunks. Upload in parallel.
*   **Resumable:** If network fails, retry only failed chunks.
*   **Presigned URL:** Upload directly to S3 (bypass API server).

## Q2: How to implement "Offline Mode"?
**Answer:**
*   **Local DB:** SQLite on client.
*   **Queue:** Store local changes in a queue.
*   **Sync:** When online, replay queue to server.
*   **Conflict:** If server changed meanwhile, prompt user to resolve.

## Q3: Block Level vs File Level Deduplication?
**Answer:**
*   **File Level:** Hash entire file. Simple. Misses small changes.
*   **Block Level:** Hash 4MB chunks. Complex. Catches partial duplicates. (Better for Drive).

## Q4: Security (Encryption)?
**Answer:**
*   **At Rest:** AES-256 in S3.
*   **In Transit:** TLS.
*   **Client-Side Encryption:** Encrypt before uploading. Server cannot read files (Zero Knowledge).
