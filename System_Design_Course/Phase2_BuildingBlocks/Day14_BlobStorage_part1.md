# Day 14 Deep Dive: Facebook Photo Storage

## 1. The Problem
*   **Scale:** Billions of photos uploaded per day.
*   **Small Files:** Most photos are small (KB).
*   **Metadata Overhead:** Storing billions of files on a standard filesystem (ext4) kills the inode table. Disk seeks for metadata are slow.

## 2. Haystack (Hot Storage)
*   **Idea:** Pack many small files into one giant file (100GB).
*   **Structure:**
    *   **Physical File:** A huge log of images.
    *   **In-Memory Index:** Maps `PhotoID` -> `(FileID, Offset, Size)`.
*   **Read:**
    1.  Look up `PhotoID` in memory. Get Offset.
    2.  One Disk Seek to `Offset`. Read `Size` bytes.
    3.  **Result:** 1 Disk Seek per photo (Optimal).

## 3. f4 (Warm/Cold Storage)
*   **Observation:** Photos are accessed frequently when new (Hot), then rarely (Warm/Cold).
*   **Haystack Issue:** Compaction is expensive.
*   **f4 Design:**
    *   **BLOBs:** Binary Large OBjects.
    *   **Immutable:** Once a volume is full, it is sealed. No more writes.
    *   **Erasure Coding:** Instead of replicating 3x (300% overhead), use Reed-Solomon (1.4x overhead).
    *   **Benefit:** Massive storage savings for old data.

## 4. Content Addressable Storage (CAS)
*   **Key:** `hash(content)`.
*   **Benefit:** Automatic deduplication. If two users upload the same meme, we only store it once.
