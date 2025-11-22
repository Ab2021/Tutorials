# Lab 14: Monitoring with JMX

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Enable JMX in Kafka.
-   Connect using JConsole or VisualVM.
-   Identify key metrics (MessagesInPerSec, BytesOutPerSec).

## Problem Statement
1.  Configure `KAFKA_JMX_OPTS` in Docker Compose.
2.  Expose the JMX port.
3.  Connect via JConsole on your host machine.
4.  Find the `MessagesInPerSec` MBean.

## Starter Code
```yaml
environment:
  KAFKA_JMX_PORT: 9101
  KAFKA_JMX_HOSTNAME: localhost
```

## Hints
<details>
<summary>Hint 1</summary>
You might need to set `-Dcom.sun.management.jmxremote.rmi.port=9101` as well.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Docker Compose Update
```yaml
    environment:
      KAFKA_JMX_PORT: 9101
      KAFKA_JMX_HOSTNAME: localhost
    ports:
      - "9101:9101"
```

### Connection
1.  Open `jconsole` (part of JDK).
2.  Connect to `localhost:9101`.
3.  Navigate to `MBeans` -> `kafka.server` -> `BrokerTopicMetrics` -> `MessagesInPerSec`.
</details>
