import os

base_path = r"G:\My Drive\Codes & Repos\Streams_Course_Kafka_Flink_Redpanda\Phase1_Foundations\Week2_Redpanda_HighPerformance\labs"

labs_content = {
    "lab_01.md": """# Lab 01: Redpanda Docker Setup

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Run Redpanda using Docker.
-   Understand the single-binary architecture.
-   Access the `rpk` CLI inside the container.

## Problem Statement
Start a single-node Redpanda cluster. Expose port 9092. Use `rpk` inside the container to check the cluster status.

## Starter Code
```bash
docker run -d --name redpanda -p 9092:9092 ...
```

## Hints
<details>
<summary>Hint 1</summary>
Use the official image `docker.redpanda.com/redpandadata/redpanda:latest`. You need to set `redpanda start --overprovisioned --smp 1 --memory 1G` for local dev.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Command
```bash
docker run -d --name redpanda --rm \
    -p 9092:9092 \
    -p 9644:9644 \
    docker.redpanda.com/redpandadata/redpanda:latest \
    redpanda start \
    --overprovisioned \
    --smp 1  \
    --memory 1G \
    --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092 \
    --advertise-kafka-addr internal://redpanda:9092,external://localhost:9092 \
    --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082 \
    --advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082 \
    --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081 \
    --rpc-addr redpanda:33145 \
    --advertise-rpc-addr redpanda:33145
```

### Verification
```bash
docker exec -it redpanda rpk cluster info
```
</details>
""",
    "lab_02.md": """# Lab 02: rpk CLI Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Master the `rpk` (Redpanda Keeper) CLI.
-   Create topics, produce, and consume without writing code.

## Problem Statement
1.  Create a topic `chat-room` with 5 partitions.
2.  Produce 3 messages ("Hello", "World", "Redpanda") using `rpk`.
3.  Consume them using `rpk` with offset `oldest`.

## Starter Code
```bash
rpk topic create ...
rpk topic produce ...
```

## Hints
<details>
<summary>Hint 1</summary>
`rpk topic produce` reads from stdin. You can pipe data into it.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Commands
```bash
# Create Topic
rpk topic create chat-room -p 5 -r 1

# Produce
echo "Hello" | rpk topic produce chat-room
echo "World" | rpk topic produce chat-room
echo "Redpanda" | rpk topic produce chat-room

# Consume
rpk topic consume chat-room --offset oldest
```
</details>
""",
    "lab_03.md": """# Lab 03: Redpanda vs Kafka Benchmarking

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Measure throughput using `rpk`'s built-in benchmark tool.
-   Compare performance (if you have a Kafka cluster running).

## Problem Statement
Run a producer benchmark against your Redpanda node.
-   Message Size: 1KB
-   Count: 100,000 messages
-   Compression: LZ4

## Starter Code
```bash
rpk topic produce --help # Look for benchmark flags? 
# Actually, rpk has a specific plugin or you use kafka-producer-perf-test
```
*Correction*: `rpk` has a simplified interface, but often we use the standard `kafka-producer-perf-test` tool which comes with Redpanda too.

## Hints
<details>
<summary>Hint 1</summary>
Try `rpk redpanda tune` (if on Linux) to see tuning options. For benchmarking, use `kafka-producer-perf-test.sh` inside the container.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
docker exec -it redpanda /bin/bash

# Inside container
kafka-producer-perf-test.sh \
  --topic bench-test \
  --num-records 100000 \
  --record-size 1024 \
  --throughput -1 \
  --producer-props bootstrap.servers=localhost:9092 compression.type=lz4
```
</details>
""",
    "lab_04.md": """# Lab 04: WASM Data Transforms

## Difficulty
🔴 Hard

## Estimated Time
90 mins

## Learning Objectives
-   Understand Data Transforms in Redpanda.
-   Deploy a WASM function to mask data.

## Problem Statement
*Note: This feature is in technical preview/beta in some versions. Ensure you have a compatible version.*
Write a Go/Rust transform that reads from `input-topic`, replaces any text "SECRET" with "****", and writes to `output-topic`.

## Starter Code
```go
// main.go (Go example)
package main

import (
    "github.com/redpanda-data/redpanda/src/transform-sdk/go/transform"
)

func main() {
    transform.OnRecordWritten(doTransform)
}

func doTransform(e transform.WriteEvent) ([]transform.Record, error) {
    // Logic here
}
```

## Hints
<details>
<summary>Hint 1</summary>
Use `rpk transform init` to generate a project template.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Init Project
```bash
rpk transform init --language=go my-transform
cd my-transform
```

### Step 2: Code
```go
package main

import (
    "bytes"
    "github.com/redpanda-data/redpanda/src/transform-sdk/go/transform"
)

func main() {
    transform.OnRecordWritten(doTransform)
}

func doTransform(e transform.WriteEvent) ([]transform.Record, error) {
    val := e.Record().Value()
    newVal := bytes.ReplaceAll(val, []byte("SECRET"), []byte("****"))
    
    return []transform.Record{
        {
            Key:   e.Record().Key(),
            Value: newVal,
        },
    }, nil
}
```

### Step 3: Deploy
```bash
rpk transform build
rpk transform deploy --input-topic=input --output-topic=output
```
</details>
""",
    "lab_05.md": """# Lab 05: Tiered Storage Config

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
""",
    "lab_06.md": """# Lab 06: Redpanda Console

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Run Redpanda Console (Kowl).
-   Inspect messages via UI.

## Problem Statement
Add `redpanda-console` to your Docker Compose. Connect it to your Redpanda broker. Open the UI and view the `chat-room` topic messages.

## Starter Code
```yaml
  console:
    image: docker.redpanda.com/redpandadata/console:latest
    environment:
      KAFKA_BROKERS: redpanda:9092
    ports:
      - "8080:8080"
```

## Hints
<details>
<summary>Hint 1</summary>
Ensure the console container is on the same network as the broker.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Docker Compose
```yaml
version: '3.7'
services:
  redpanda:
    # ... existing config ...
  console:
    image: docker.redpanda.com/redpandadata/console:latest
    restart: on-failure
    entrypoint: /bin/sh
    command: -c "echo \"$$CONSOLE_CONFIG_FILE\" > /tmp/config.yaml; /app/console"
    environment:
      CONFIG_FILEPATH: /tmp/config.yaml
      CONSOLE_CONFIG_FILE: |
        kafka:
          brokers: ["redpanda:9092"]
    ports:
      - "8080:8080"
    depends_on:
      - redpanda
```
Run `docker-compose up -d`. Open `http://localhost:8080`.
</details>
""",
    "lab_07.md": """# Lab 07: Schema Registry in Redpanda

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use the built-in Schema Registry.
-   Register a schema via `curl`.

## Problem Statement
Redpanda exposes the Registry at port 8081.
1.  Create a JSON schema file `user.avsc`.
2.  Register it using `curl`.
3.  List subjects.

## Starter Code
```json
{
  "schema": "{\"type\": \"string\"}"
}
```

## Hints
<details>
<summary>Hint 1</summary>
The API endpoint is `POST /subjects/{subject}/versions`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Schema File
```json
{
  "schema": "{\"type\":\"record\",\"name\":\"User\",\"fields\":[{\"name\":\"name\",\"type\":\"string\"}]}"
}
```

### Step 2: Register
```bash
curl -X POST -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data @user.avsc \
  http://localhost:8081/subjects/user-value/versions
```

### Step 3: Verify
```bash
curl http://localhost:8081/subjects
```
</details>
""",
    "lab_08.md": """# Lab 08: Admin API

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Use the Redpanda Admin API (port 9644).
-   Manage users and config.

## Problem Statement
The Admin API allows operational control.
1.  Query the cluster health via the API.
2.  Create a user `admin` with password `secret` via the API.

## Starter Code
```bash
curl http://localhost:9644/v1/status
```

## Hints
<details>
<summary>Hint 1</summary>
User management endpoint is `/v1/security/users`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Check Status
```bash
curl http://localhost:9644/v1/cluster_view
```

### Create User
```bash
curl -X POST http://localhost:9644/v1/security/users \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secret", "algorithm": "SCRAM-SHA-256"}'
```
</details>
""",
    "lab_09.md": """# Lab 09: Tuning Redpanda

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `rpk redpanda tune`.
-   Understand OS-level optimizations.

## Problem Statement
*Note: This requires running on Linux (not Docker Desktop on Mac/Windows).*
Run the autotuner to optimize for network and disk.
1.  Run `rpk redpanda tune help`.
2.  Enable `aio_events` and `disk_irq` tuners.

## Starter Code
```bash
rpk redpanda tune all
```

## Hints
<details>
<summary>Hint 1</summary>
This modifies system files (`/etc/sysctl.conf`). Run with sudo.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
sudo rpk redpanda tune disk_irq aio_events
```
Output will show which parameters were changed (e.g., IRQ affinity).
</details>
""",
    "lab_10.md": """# Lab 10: Redpanda Connect (Benthos)

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Use Redpanda Connect (formerly Benthos) for simple ETL.
-   Ingest data from an HTTP API into a topic.

## Problem Statement
Create a Benthos config `pipeline.yaml` that:
1.  Polls `https://randomuser.me/api/` every 1 second.
2.  Writes the JSON to topic `random-users`.
Run it using `rpk connect run`.

## Starter Code
```yaml
input:
  http_client:
    url: https://randomuser.me/api/
    verb: GET

output:
  kafka_franz:
    topic: random-users
    seed_brokers: ["localhost:9092"]
```

## Hints
<details>
<summary>Hint 1</summary>
Redpanda Connect is bundled in `rpk`. Use `rpk connect run pipeline.yaml`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### pipeline.yaml
```yaml
input:
  generate:
    interval: 1s
    mapping: |
      root = http_get("https://randomuser.me/api/")

output:
  kafka_franz:
    seed_brokers: ["localhost:9092"]
    topic: random-users
```

### Run
```bash
rpk connect run pipeline.yaml
```
</details>
""",
    "lab_11.md": """# Lab 11: Shadow Indexing Fetch

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Verify that data can be read from S3 transparently.

## Problem Statement
(Requires Lab 05 setup).
1.  Set retention on `archived-topic` to 1 minute (local).
2.  Produce data. Wait 2 minutes.
3.  Local data should be deleted, but S3 data remains.
4.  Consume from offset 0. Redpanda should fetch from S3.

## Starter Code
```bash
rpk topic alter-config archived-topic --set retention.ms=60000
```

## Hints
<details>
<summary>Hint 1</summary>
Watch the Redpanda logs for "Downloading segment from remote".
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Commands
```bash
# Set short local retention
rpk topic alter-config archived-topic --set retention.local.target.bytes=100

# Produce
for i in {1..1000}; do echo "msg-$i" | rpk topic produce archived-topic; done

# Wait... then Consume
rpk topic consume archived-topic --offset oldest
```
If you see "msg-1", it worked (fetched from S3).
</details>
""",
    "lab_12.md": """# Lab 12: Maintenance Mode

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Perform a rolling upgrade simulation.
-   Drain a node.

## Problem Statement
1.  Enable maintenance mode on Node 1.
2.  Verify that leaderships are transferred to other nodes.
3.  Disable maintenance mode.

## Starter Code
```bash
rpk cluster maintenance enable <node-id>
```

## Hints
<details>
<summary>Hint 1</summary>
Use `rpk cluster status` to find the Node ID.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# Get ID
rpk cluster status

# Enable
rpk cluster maintenance enable 1

# Check status (should say "draining" or "finished")
rpk cluster maintenance status

# Disable
rpk cluster maintenance disable 1
```
</details>
""",
    "lab_13.md": """# Lab 13: Partition Rebalancing

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Trigger the partition balancer.
-   Understand `on-demand` vs `continuous` balancing.

## Problem Statement
1.  Create a topic with 30 partitions.
2.  Observe they are spread evenly.
3.  Kill one node. Wait.
4.  Bring the node back.
5.  Partitions might not move back immediately. Trigger a rebalance.

## Starter Code
```bash
rpk cluster partitions balancer-status
```

## Hints
<details>
<summary>Hint 1</summary>
Redpanda's balancer is usually continuous. You can tweak `partition_autobalancing_mode`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```bash
# Check status
rpk cluster partitions balancer-status

# Force rebalance (if needed)
# Usually Redpanda does this automatically, but you can inspect the movement:
rpk cluster partitions movement-cancel --all # To stop it
```
</details>
""",
    "lab_14.md": """# Lab 14: Redpanda Security (SASL/SCRAM)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Enable SASL authentication.
-   Create users and ACLs.

## Problem Statement
1.  Enable `enable_sasl: true` in `redpanda.yaml`.
2.  Create a superuser `admin`.
3.  Try to access without auth (should fail).
4.  Access with auth.

## Starter Code
```yaml
redpanda:
  enable_sasl: true
  superusers: ["admin"]
```

## Hints
<details>
<summary>Hint 1</summary>
You need to pass `--user` and `--password` to `rpk` commands.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Create User
```bash
rpk acl user create admin -p secret
```

### Access
```bash
rpk cluster info --user admin --password secret --sasl-mechanism SCRAM-SHA-256
```
</details>
""",
    "lab_15.md": """# Lab 15: HTTP Proxy (PandaProxy)

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Use the REST API to produce/consume.

## Problem Statement
Redpanda has a built-in HTTP proxy on port 8082.
1.  Create a topic `http-test`.
2.  POST a message to it using `curl`.
3.  GET messages from it using `curl`.

## Starter Code
```bash
curl -X POST http://localhost:8082/topics/http-test ...
```

## Hints
<details>
<summary>Hint 1</summary>
Content-Type must be `application/vnd.kafka.json.v2+json`.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Produce
```bash
curl -X POST "http://localhost:8082/topics/http-test" \
  -H "Content-Type: application/vnd.kafka.json.v2+json" \
  -d '{"records":[{"value":"hello http"}]}'
```

### Consume
First, create a consumer:
```bash
curl -X POST "http://localhost:8082/consumers/my-group" \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -d '{"name": "my-consumer", "format": "json", "auto.offset.reset": "earliest"}'
```

Subscribe:
```bash
curl -X POST "http://localhost:8082/consumers/my-group/instances/my-consumer/subscription" \
  -H "Content-Type: application/vnd.kafka.v2+json" \
  -d '{"topics":["http-test"]}'
```

Fetch:
```bash
curl "http://localhost:8082/consumers/my-group/instances/my-consumer/records" \
  -H "Accept: application/vnd.kafka.json.v2+json"
```
</details>
"""
}

print("🚀 Upgrading Week 2 Labs with Comprehensive Solutions...")

for filename, content in labs_content.items():
    full_path = os.path.join(base_path, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Upgraded {filename}")

print("✅ Week 2 Labs Upgrade Complete!")
