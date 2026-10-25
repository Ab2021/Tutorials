"""At-least-once consumer with deduplication and a dead-letter topic."""
from __future__ import annotations
import argparse
import json
import signal
import sqlite3
import hashlib
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, Producer
from pydantic import ValidationError

from schemas import AdtEvent

running = True


def stop(*_args) -> None:
    global running
    running = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="adt.events")
    parser.add_argument("--group", default="biomed-adt-consumer")
    parser.add_argument("--dlq", default="adt.events.dlq")
    parser.add_argument("--db", default="consumer.db")
    parser.add_argument("--max-messages", type=int, default=0)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    db = sqlite3.connect(args.db)
    db.execute("CREATE TABLE IF NOT EXISTS processed (event_id TEXT PRIMARY KEY, processed_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    db.commit()

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "enable.partition.eof": False,
    })
    dlq = Producer({"bootstrap.servers": args.bootstrap, "enable.idempotence": True, "acks": "all"})
    consumer.subscribe([args.topic])

    handled = 0
    while running and (args.max_messages == 0 or handled < args.max_messages):
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            print(f"consumer error: {msg.error()}")
            continue

        raw = msg.value().decode("utf-8")
        try:
            event = AdtEvent.model_validate_json(raw)
        except ValidationError as exc:
            dlq.produce(args.dlq, key=msg.key(), value=raw.encode(), headers=[("error", str(exc).encode())])
            dlq.flush(5)
            consumer.commit(message=msg, asynchronous=False)
            print(f"dead-lettered {msg.key()}: {exc}")
            continue

        try:
            db.execute("INSERT INTO processed(event_id) VALUES (?)", (event.event_id,))
            db.commit()
        except sqlite3.IntegrityError:
            consumer.commit(message=msg, asynchronous=False)
            print(f"duplicate {event.event_id}")
            continue

        # Replace with clinical processing. No PHI in logs.
        patient_hash = hashlib.sha256(event.patient_id.encode()).hexdigest()[:12]
        print(f"processed {event.event_type} event for patient hash {patient_hash}")
        consumer.commit(message=msg, asynchronous=False)
        handled += 1

    consumer.close()
    dlq.flush(5)
    db.close()


if __name__ == "__main__":
    main()
