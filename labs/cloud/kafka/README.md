# Kafka/Redpanda Lab

## Run

```bash
cd labs/cloud
make up
make produce
make consume
```

## Design

- Keyed by `patient_id` so a patient's events stay ordered within a partition.
- Idempotent producer (`enable.idempotence`, `acks=all`).
- Consumer commits after processing; `event_id` deduplicates replays.
- Invalid events go to `adt.events.dlq`.
- No PHI in logs; the sample IDs are synthetic.
