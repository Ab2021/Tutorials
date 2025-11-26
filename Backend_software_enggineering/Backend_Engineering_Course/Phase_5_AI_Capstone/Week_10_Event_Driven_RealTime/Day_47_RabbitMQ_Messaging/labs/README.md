# Lab: Day 47 - RabbitMQ Basics

## Goal
Send and receive messages using RabbitMQ.

## Prerequisites
- Docker (RabbitMQ).
- `pip install pika`

## Step 1: Start RabbitMQ
```bash
docker run -d --hostname my-rabbit --name some-rabbit -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```
*   **UI**: `http://localhost:15672` (guest/guest).

## Step 2: Producer (`send.py`)

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare Queue (Idempotent)
channel.queue_declare(queue='hello')

channel.basic_publish(exchange='',
                      routing_key='hello',
                      body='Hello World!')
print(" [x] Sent 'Hello World!'")
connection.close()
```

## Step 3: Consumer (`receive.py`)

```python
import pika, sys, os

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        print(f" [x] Received {body}")

    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
```

## Step 4: Run It
1.  Terminal 1: `python receive.py`
2.  Terminal 2: `python send.py`

## Challenge: Routing
Implement a **Log System**.
1.  Producer emits logs with severity: `info`, `warning`, `error`.
2.  Consumer 1: Listens to ALL logs (writes to file).
3.  Consumer 2: Listens ONLY to `error` (prints to screen).
*   *Hint*: Use `exchange_type='direct'`.
