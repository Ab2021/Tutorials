# Logging & Log Management

## 🎯 Learning Objectives

By the end of this module, you will have a comprehensive understanding of Log Management, including:
- **Centralized Logging**: Aggregating logs from all servers into a single searchable interface.
- **ELK Stack**: Setting up Elasticsearch, Logstash, and Kibana.
- **Fluentd**: Using the CNCF standard for unified log collection.
- **Loki**: Implementing lightweight, label-based logging with Grafana.
- **Parsing**: Transforming unstructured text logs into structured JSON data.

---

## 📖 Theoretical Concepts

### 1. The Need for Centralized Logging

In a distributed system with 100 containers, you cannot SSH into each one to check `/var/log/syslog`.
- **Aggregation**: Collect logs from all sources (App, OS, DB, LB).
- **Indexing**: Make them searchable ("Show me all errors from Service A").
- **Retention**: Keep logs for 30 days (or 7 years for compliance).

### 2. ELK Stack (Elastic Stack)

The industry veteran.
- **Elasticsearch**: A distributed search engine. Stores the logs.
- **Logstash**: A server-side data processing pipeline. Ingests, transforms, and sends data to ES.
- **Kibana**: The UI. Visualize data and manage the stack.
- **Beats**: Lightweight shippers (Filebeat) installed on edge nodes.

### 3. Fluentd & Fluent Bit

The Cloud Native choice.
- **Unified Logging Layer**: Decouples data sources from data backends.
- **Plugins**: 500+ plugins to talk to S3, Kafka, Elasticsearch, Splunk, etc.
- **Fluent Bit**: A super-lightweight version written in C, often used as a DaemonSet in K8s.

### 4. Loki (PLG Stack)

Designed by Grafana Labs.
- **Index-free**: Unlike ES, Loki doesn't index the text of the logs. It only indexes metadata (labels).
- **Cost**: Much cheaper storage (S3) and less RAM usage.
- **LogQL**: Query language similar to PromQL.

---

## 🔧 Practical Examples

### Fluentd Config (`fluent.conf`)

```xml
<source>
  @type forward
  port 24224
</source>

<filter app.**>
  @type parser
  key_name log
  format json
</filter>

<match **>
  @type elasticsearch
  host elasticsearch
  port 9200
  logstash_format true
</match>
```

### Logstash Pipeline (`pipeline.conf`)

```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
  }
}

output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "%{[@metadata][beat]}-%{[@metadata][version]}-%{+YYYY.MM.dd}"
  }
}
```

### LogQL Query (Loki)

```logql
{app="nginx"} |= "error" | json | status >= 500
```

---

## 🎯 Hands-on Labs

- [Lab 17.1: Fluentd Basics](./labs/lab-17.1-fluentd-basics.md)
- [Lab 17.2: Loki & Grafana (PLG Stack)](./labs/lab-17.2-loki-grafana.md)
- [Lab 17.3: Logstash Pipelines](./labs/lab-17.3-logstash-pipelines.md)
- [Lab 17.4: Kibana Dashboards](./labs/lab-17.4-kibana-dashboards.md)
- [Lab 17.5: Fluentd Setup](./labs/lab-17.5-fluentd-setup.md)
- [Lab 17.6: Log Parsing](./labs/lab-17.6-log-parsing.md)
- [Lab 17.7: Log Retention](./labs/lab-17.7-log-retention.md)
- [Lab 17.8: Log Analysis](./labs/lab-17.8-log-analysis.md)
- [Lab 17.9: Centralized Logging](./labs/lab-17.9-centralized-logging.md)
- [Lab 17.10: Logging Best Practices](./labs/lab-17.10-logging-best-practices.md)

---

## 📚 Additional Resources

### Official Documentation
- [Elastic Stack Docs](https://www.elastic.co/guide/index.html)
- [Fluentd Docs](https://docs.fluentd.org/)
- [Grafana Loki Docs](https://grafana.com/docs/loki/latest/)

### Tools
- [Grok Debugger](https://grokdebug.herokuapp.com/) - Test your Grok patterns.

---

## 🔑 Key Takeaways

1.  **Log Structurally**: Use JSON. Parsing text with Regex is brittle and slow.
2.  **Separate Concerns**: Your app should write to `stdout`. The infrastructure (Docker/K8s) should handle routing the logs.
3.  **Context**: Always include Trace IDs in logs to correlate with Traces.
4.  **Security**: Sanitize logs. Never log PII (Personally Identifiable Information) or Secrets.

---

## ⏭️ Next Steps

1.  Complete the labs to build a centralized logging pipeline.
2.  Proceed to **[Module 18: Security & Compliance](../module-18-security-compliance/README.md)** to learn how to secure your entire stack.
