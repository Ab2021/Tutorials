# Lab 9.4: ELK Stack Basics (Logging)

## 🎯 Objective

Centralize your logs. Instead of SSHing into 10 servers to `tail -f /var/log/syslog`, you will send all logs to Elasticsearch and view them in Kibana.

## 📋 Prerequisites

-   Docker & Docker Compose (Allocated at least 4GB RAM to Docker).

## 📚 Background

### The Stack
1.  **Elasticsearch**: The Database. Stores logs as JSON documents. Search engine.
2.  **Logstash** (or Filebeat): The Shipper. Collects logs, parses them, sends to ES.
3.  **Kibana**: The UI. Visualizes logs.

*Note: This stack is heavy. We will use a simplified setup.*

---

## 🔨 Hands-On Implementation

### Part 1: The Compose File 🐳

1.  **Create `elk/docker-compose.yml`:**
    ```yaml
    version: '3'
    services:
      elasticsearch:
        image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
        environment:
          - discovery.type=single-node
          - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
        ports:
          - "9200:9200"

      kibana:
        image: docker.elastic.co/kibana/kibana:7.17.0
        ports:
          - "5601:5601"
        depends_on:
          - elasticsearch

      # Simulating an App generating logs
      log-generator:
        image: alpine
        command: sh -c "while true; do echo 'Hello ELK $(date)'; sleep 5; done"
    ```

### Part 2: Launch 🚀

1.  **Start:**
    ```bash
    docker-compose up -d
    ```
    *Warning:* Elasticsearch takes 30-60 seconds to start. Be patient.

2.  **Verify ES:**
    `curl localhost:9200` -> Should return JSON with "You Know, for Search".

3.  **Verify Kibana:**
    Open `http://localhost:5601`.

### Part 3: Ingesting Logs (Manual) 📥

We don't have Logstash set up yet, so let's push a log manually to understand the API.

1.  **Send Log:**
    ```bash
    curl -X POST "localhost:9200/app-logs/_doc" -H 'Content-Type: application/json' -d'
    {
      "timestamp": "2023-10-27T10:00:00",
      "level": "INFO",
      "message": "User logged in",
      "user_id": 123
    }
    '
    ```

### Part 4: Viewing in Kibana 👁️

1.  **Create Index Pattern:**
    -   Go to Kibana -> **Stack Management** -> **Index Patterns**.
    -   Create Index Pattern: `app-logs*`.
    -   Timestamp field: `timestamp` (or I don't want to use time filter).

2.  **Discover:**
    -   Go to **Discover** (Compass icon).
    -   Select `app-logs*`.
    -   You should see your "User logged in" message.

---

## 🎯 Challenges

### Challenge 1: Filebeat (Difficulty: ⭐⭐⭐⭐)

**Task:**
Add **Filebeat** to the docker-compose.
Configure it to read Docker container logs (`/var/lib/docker/containers/*/*.log`) and send them to Elasticsearch.
*Result:* You will see the "Hello ELK" logs from the `log-generator` service in Kibana.

### Challenge 2: Search Query (Difficulty: ⭐⭐)

**Task:**
In Kibana Discover, use KQL (Kibana Query Language) to find logs where `user_id` equals `123`.
Query: `user_id : 123`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
Add Filebeat service mounting docker socket/logs.
Config:
```yaml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```
</details>

---

## 🔑 Key Takeaways

1.  **Centralization**: Debugging distributed systems is impossible without centralized logging.
2.  **Structured Logging**: JSON logs (`{"user": "alice"}`) are better than text logs (`User alice logged in`) because you can filter by field.
3.  **Cost**: ELK is resource-hungry. For smaller setups, consider **Loki** (PLG Stack).

---

## ⏭️ Next Steps

We have covered the basics of Phase 1. Let's wrap up Module 9 with a Capstone.

Proceed to **Lab 9.5: Monitoring Capstone Project**.
