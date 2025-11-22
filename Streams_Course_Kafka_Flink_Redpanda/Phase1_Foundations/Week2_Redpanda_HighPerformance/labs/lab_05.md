# Lab 05: Tiered Storage Config

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Configure S3 (or MinIO) for Tiered Storage.
-   Enable Tiered Storage on a topic.

## Problem Statement
1.  Start MinIO (S3 compatible) in Docker.
2.  Configure Redpanda to use MinIO as the bucket.
3.  Create a topic `archived-topic` with `redpanda.remote.write=true`.

## Starter Code
```yaml
# redpanda.yaml config snippet
cloud_storage_enabled: true
cloud_storage_access_key: minioadmin
cloud_storage_secret_key: minioadmin
cloud_storage_region: us-east-1
cloud_storage_bucket: redpanda-bucket
cloud_storage_api_endpoint: http://minio:9000
```

## Hints
<details>
<summary>Hint 1</summary>
You need to restart Redpanda after changing `redpanda.yaml` or pass these as `--set` flags in `rpk redpanda start`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Topic
```bash
rpk topic create archived-topic -c redpanda.remote.write=true -c redpanda.remote.read=true
```

### Step 2: Verify
Produce data. Wait for segment roll. Check MinIO browser (localhost:9001) to see if files appear in the bucket.
</details>
