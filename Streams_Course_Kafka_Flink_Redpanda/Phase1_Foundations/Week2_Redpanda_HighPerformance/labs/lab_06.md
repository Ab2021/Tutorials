# Lab 06: Redpanda Console

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
    command: -c "echo "$$CONSOLE_CONFIG_FILE" > /tmp/config.yaml; /app/console"
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
