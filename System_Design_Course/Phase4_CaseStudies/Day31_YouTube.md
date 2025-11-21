# Day 31: Design YouTube

## 1. Requirements
*   **Functional:** Upload video, Watch video, Search.
*   **Non-Functional:** High availability, Low latency (No buffering), High throughput (Uploads).
*   **Scale:** 1 Billion users. 500 hours video uploaded per minute.

## 2. High Level Design
*   **Client:** Mobile/Web.
*   **API Gateway:** Auth, Rate Limit.
*   **Web Server:** Handles metadata (Title, Description).
*   **Video Upload Service:** Handles raw file upload.
*   **Transcoding Service:** Converts raw video to MP4/HLS (480p, 720p, 1080p).
*   **CDN:** Caches popular videos.
*   **Storage:**
    *   **Blob Storage (S3):** Raw and Transcoded videos.
    *   **SQL/NoSQL:** Metadata (User, Video Info).

## 3. Video Streaming Flow
1.  User clicks play.
2.  Client requests `video.m3u8` (HLS Manifest).
3.  CDN serves the manifest.
4.  Client downloads chunks (`chunk_0.ts`, `chunk_1.ts`) from CDN.
5.  Player buffers and plays.

## 4. Transcoding (DAG)
*   Raw video is huge. Must compress and resize.
*   **Steps:**
    1.  Validation (Is it a video?).
    2.  Audio Extraction.
    3.  Thumbnail Generation.
    4.  Transcoding (FFmpeg) to multiple resolutions.
    5.  Watermarking.
*   **Parallelism:** Split video into chunks. Transcode chunks in parallel (MapReduce).
