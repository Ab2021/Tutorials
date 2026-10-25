"""Idempotent ADT event producer for the Module 26 lab."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import ValidationError

from schemas import AdtEvent


def ensure_topic(bootstrap: str, topic: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    futures = admin.create_topics([NewTopic(topic, num_partitions=3, replication_factor=1)])
    for name, future in futures.items():
        try:
            future.result()
            print(f"created topic {name}")
        except Exception as exc:  # topic may already exist
            print(f"topic {name}: {exc}")


def delivery(err, msg) -> None:
    if err is not None:
        print(f"delivery failed: {err}", file=sys.stderr)
    else:
        print(f"delivered {msg.key().decode()} -> {msg.topic()}[{msg.partition()}]@{msg.offset()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="events.jsonl")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="adt.events")
    args = parser.parse_args()

    ensure_topic(args.bootstrap, args.topic)
    producer = Producer({
        "bootstrap.servers": args.bootstrap,
        "enable.idempotence": True,
        "acks": "all",
        "retries": 5,
        "max.in.flight.requests.per.connection": 5,
    })

    sent = 0
    for line_no, line in enumerate(Path(args.file).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = AdtEvent.model_validate_json(line)
        except ValidationError as exc:
            print(f"line {line_no}: invalid event: {exc}", file=sys.stderr)
            continue
        producer.produce(
            args.topic,
            key=event.patient_id.encode(),
            value=event.model_dump_json().encode(),
            headers=[("schema_version", event.schema_version.encode())],
            callback=delivery,
        )
        producer.poll(0)
        sent += 1

    producer.flush(10)
    print(f"submitted {sent} events")


if __name__ == "__main__":
    main()
