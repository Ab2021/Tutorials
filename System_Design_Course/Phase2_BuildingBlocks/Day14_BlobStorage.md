# Day 14: Blob Storage

## 1. What is Blob Storage?
*   **Blob:** Binary Large Object (Images, Videos, Backups).
*   **Characteristics:**
    *   **Immutable:** Write once, read many (WORM).
    *   **Unstructured:** No schema.
    *   **Massive Scale:** Petabytes/Exabytes.
    *   **Cheaper:** Than Block Storage (SSD) or DB.

## 2. Architecture (S3 Style)
*   **Bucket:** Logical container (Namespace).
*   **Object:** The file itself.
*   **Key:** Unique identifier (e.g., `/photos/2023/dog.jpg`).
*   **Metadata:** Key-Value pairs (Content-Type, Owner).

## 3. Consistency Model
*   **Strong Consistency:** (New S3). Read-after-write is guaranteed.
*   **Eventual Consistency:** (Old S3). Might get 404 immediately after upload.

## 4. Storage Classes
*   **Standard:** Hot data. Frequent access. Expensive.
*   **Infrequent Access (IA):** Cool data. Cheaper storage, expensive retrieval.
*   **Glacier/Archive:** Cold data. Very cheap. Retrieval takes hours.

## 5. Multipart Upload
*   **Problem:** Uploading a 5GB file fails at 99%.
*   **Solution:** Break file into 5MB chunks. Upload in parallel. Merge at the end.
