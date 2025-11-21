# Day 31 Deep Dive: Optimization

## 1. Adaptive Bitrate Streaming (ABR)
*   **Problem:** User's bandwidth fluctuates.
*   **Solution:**
    *   Encode video in multiple bitrates (Low, Medium, High).
    *   Client detects bandwidth.
    *   If fast, download High quality chunks.
    *   If slow, switch to Low quality chunks *mid-stream*.
*   **Protocols:** HLS (Apple), DASH (Google).

## 2. Storage Optimization
*   **Deduplication:** If two users upload the same video (MD5 hash match), store only once.
*   **Erasure Coding:** Use Reed-Solomon (1.5x overhead) instead of Replication (3x overhead) for cold videos.
*   **Thumbnail Storage:** Bigtable (Google) or Haystack (Facebook). Pack small images into large files to reduce disk seeks.

## 3. Safety Optimization
*   **Pre-signed URLs:**
    *   Client uploads directly to S3.
    *   Server generates a URL with a signature valid for 1 hour.
    *   Reduces load on API servers.
*   **Content Moderation:**
    *   Async job.
    *   AI Model (Classifier) checks for Nudity/Copyright.
    *   Human review if confidence is low.

## 4. Caching Strategy
*   **Long Tail:** 80% of views come from 20% of videos.
*   **Hot Videos:** Cache in CDN and Edge.
*   **Cold Videos:** Fetch from Origin (S3).
