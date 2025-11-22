# Lab 10: Kafka Connect Basics

## Difficulty
🟢 Easy

## Estimated Time
45 mins

## Learning Objectives
-   Understand the role of Kafka Connect.
-   Configure a standalone connector (FileStreamSource).

## Problem Statement
Use the `connect-standalone` mode (available in the Confluent container) to read lines from a text file and publish them to a Kafka topic.
1.  Create a file `data.txt` with some lines.
2.  Configure `source.properties`.
3.  Run the connector.

## Starter Code
```properties
# source.properties
name=local-file-source
connector.class=FileStreamSource
tasks.max=1
file=/tmp/data.txt
topic=connect-test
```

## Hints
<details>
<summary>Hint 1</summary>
You need to mount the file into the container or create it inside the container.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Create Data File
```bash
docker exec -it kafka1 bash -c "echo 'hello world' > /tmp/data.txt"
```

### Step 2: Create Config File
Create `connect-file-source.properties` inside the container:
```properties
name=local-file-source
connector.class=org.apache.kafka.connect.file.FileStreamSourceConnector
tasks.max=1
file=/tmp/data.txt
topic=connect-test
```

### Step 3: Run Connect (Standalone)
```bash
# This command assumes you are inside the container and have the config
connect-standalone /etc/kafka/connect-standalone.properties connect-file-source.properties
```

### Step 4: Verify
Consume from `connect-test`:
```bash
kafka-console-consumer --bootstrap-server localhost:9092 --topic connect-test --from-beginning
```
</details>
