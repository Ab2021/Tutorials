# Lab 06: Tiered Storage Configuration

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Enable Redpanda Tiered Storage
- Configure S3 backend
- Understand local vs remote data

## Problem Statement
Configure Redpanda to use S3 for tiered storage. Create a topic with a short local retention (1 hour) but infinite cloud retention. Verify that old data is archived to S3.

## Starter Code
```yaml
# redpanda.yaml
cloud_storage_enabled: true
cloud_storage_region: us-east-1
cloud_storage_bucket: my-redpanda-bucket
cloud_storage_access_key: YOUR_KEY
cloud_storage_secret_key: YOUR_SECRET
```

## Hints
<details>
<summary>Hint 1</summary>
Use MinIO for local S3-compatible storage during testing.
</details>

<details>
<summary>Hint 2</summary>
Set topic config `redpanda.remote.write=true` to enable archiving.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

**docker-compose.yml (with MinIO):**
```yaml
version: '3'
services:
  minio:
    image: minio/minio
    ports:
      - \"9000:9000\"
      - \"9001:9001\"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address \":9001\"
  
  redpanda:
    image: redpandadata/redpanda:latest
    command:
      - redpanda start
      - --smp 1
      - --overprovisioned
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
      - --set redpanda.cloud_storage_enabled=true
      - --set redpanda.cloud_storage_region=us-east-1
      - --set redpanda.cloud_storage_bucket=redpanda-bucket
      - --set redpanda.cloud_storage_access_key=minioadmin
      - --set redpanda.cloud_storage_secret_key=minioadmin
      - --set redpanda.cloud_storage_api_endpoint=minio
      - --set redpanda.cloud_storage_api_endpoint_port=9000
    ports:
      - \"19092:19092\"
    depends_on:
      - minio
```

**Create Bucket:**
```bash
# Access MinIO console at http://localhost:9001
# Create bucket: redpanda-bucket
```

**Create Topic with Tiered Storage:**
```bash
rpk topic create tiered-topic \
  --config redpanda.remote.write=true \
  --config redpanda.remote.read=true \
  --config retention.local.target.bytes=1048576  # 1MB local
```

**Verify:**
```bash
# Produce data
rpk topic produce tiered-topic

# Check S3 (MinIO)
# Data should appear in s3://redpanda-bucket/

# Read old data (fetched from S3)
rpk topic consume tiered-topic --offset 0
```
</details>
