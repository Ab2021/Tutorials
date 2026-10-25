.PHONY: up down produce consume test

up:
	docker compose up -d

down:
	docker compose down -v

produce:
	python kafka/producer.py --file kafka/events.jsonl

consume:
	python kafka/consumer.py --max-messages 10

test:
	python -m pytest -q
