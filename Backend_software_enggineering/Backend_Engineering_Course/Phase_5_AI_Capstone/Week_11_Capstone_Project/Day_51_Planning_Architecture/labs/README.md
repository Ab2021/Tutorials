# Lab: Day 51 - Project Setup

## Goal
Initialize the infrastructure for "DocuMind".

## Step 1: Folder Structure
```text
documind/
├── auth_service/
├── doc_service/
├── collab_service/
├── ai_service/
└── docker-compose.yml
```

## Step 2: Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: documind
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"

volumes:
  pgdata:
```

## Step 3: Run It
`docker-compose up -d`

## Step 4: Verify
1.  Check logs: `docker-compose logs -f`.
2.  Connect to Postgres: `psql -h localhost -U user -d documind`.
3.  Check Qdrant: `curl localhost:6333`.

## Challenge
Create a `Makefile` to automate common tasks:
*   `make up`: Start containers.
*   `make down`: Stop containers.
*   `make logs`: Tail logs.
