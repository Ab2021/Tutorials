# Lab 17.3: Logstash Pipelines

## Objective
Create advanced Logstash pipelines for log processing.

## Learning Objectives
- Parse different log formats
- Use Grok patterns
- Enrich logs with filters
- Handle multiple inputs/outputs

---

## Multi-Input Pipeline

```ruby
input {
  file {
    path => "/var/log/nginx/access.log"
    type => "nginx"
  }
  file {
    path => "/var/log/app/*.log"
    type => "application"
  }
  beats {
    port => 5044
  }
}

filter {
  if [type] == "nginx" {
    grok {
      match => { "message" => "%{COMBINEDAPACHELOG}" }
    }
  }
  
  if [type] == "application" {
    json {
      source => "message"
    }
  }
  
  mutate {
    add_field => { "environment" => "production" }
  }
}

output {
  if [type] == "nginx" {
    elasticsearch {
      hosts => ["localhost:9200"]
      index => "nginx-%{+YYYY.MM.dd}"
    }
  } else {
    elasticsearch {
      hosts => ["localhost:9200"]
      index => "app-%{+YYYY.MM.dd}"
    }
  }
}
```

## Custom Grok Patterns

```
# patterns/custom
MYAPP %{TIMESTAMP_ISO8601:timestamp} \[%{LOGLEVEL:level}\] %{GREEDYDATA:message}
```

```ruby
filter {
  grok {
    patterns_dir => ["/etc/logstash/patterns"]
    match => { "message" => "%{MYAPP}" }
  }
}
```

## Success Criteria
✅ Multiple inputs configured  
✅ Grok patterns parsing logs  
✅ Filters enriching data  
✅ Conditional outputs working  

**Time:** 45 min
