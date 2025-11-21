# Day 14 Interview Prep: Blob Storage

## Q1: How do you design a system like Dropbox?
**Answer:**
*   **Client:** Chunk files (4MB). Hash chunks. Sync only changed chunks.
*   **Metadata DB:** Stores file hierarchy (`/folder/file.txt`) and maps to chunks.
*   **Block Server:** Stores the actual chunks (CAS).
*   **Notification:** Long polling to notify clients of changes.

## Q2: What is Erasure Coding?
**Answer:**
*   A method to protect data without 3x replication.
*   **Math:** Divide data into $N$ chunks. Create $K$ parity chunks.
*   **Recovery:** Any $N$ chunks (out of $N+K$) can reconstruct the data.
*   **Trade-off:** High CPU usage (math) but Low Storage overhead. Used for cold storage.

## Q3: How to handle large file uploads?
**Answer:**
*   **Presigned URLs:** Server generates a temporary URL (S3 Presigned URL). Client uploads directly to S3.
*   **Multipart Upload:** Split file. Upload parts in parallel. Retry failed parts.

## Q4: Block Storage vs File Storage vs Object Storage?
**Answer:**
*   **Block (SSD/EBS):** Raw blocks. Fast. Bootable. Expensive.
*   **File (NAS/NFS):** Hierarchical folders. Shared.
*   **Object (S3):** Flat namespace. HTTP API. Metadata. Scalable.
