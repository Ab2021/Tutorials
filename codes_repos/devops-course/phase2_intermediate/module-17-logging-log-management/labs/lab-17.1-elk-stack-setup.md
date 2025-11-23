# Lab 17.1: ELK Stack Setup

## Objective
Set up Elasticsearch, Logstash, and Kibana for centralized logging.

## Learning Objectives
- Install ELK stack
- Configure Logstash pipelines
- Create Kibana dashboards
- Query logs with Elasticsearch

---

## Docker Compose Setup

```yaml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
  
  logstash:
    image: logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
  
  kibana:
    image: kibana:8.11.0
    ports:
      - "5601:5601"
```

## Logstash Pipeline

```ruby
# logstash.conf
input {
  tcp {
    port => 5000
    codec => json
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
  date {
    match => [ "timestamp", "dd/MMM/yyyy:HH:mm:ss Z" ]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{+YYYY.MM.dd}"
  }
}
```

## Send Logs

```python
import logging
import logstash

logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.INFO)
logger.addHandler(logstash.TCPLogstashHandler('localhost', 5000))

logger.info('Test log message', extra={'user': 'john', 'action': 'login'})
```

## Kibana Queries

```
# Search logs
status:500

# Time range
@timestamp:[now-1h TO now]

# Aggregation
status:* | stats count() by status
```

## Success Criteria
✅ ELK stack running  
✅ Logs ingested via Logstash  
✅ Kibana dashboard created  
✅ Elasticsearch queries working  

**Time:** 50 min
