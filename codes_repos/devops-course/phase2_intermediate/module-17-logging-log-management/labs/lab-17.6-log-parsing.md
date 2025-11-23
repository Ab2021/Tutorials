# Lab 17.6: Log Parsing

## Objective
Parse and structure unstructured log data.

## Learning Objectives
- Use Grok for pattern matching
- Parse JSON logs
- Extract fields from logs
- Handle parsing failures

---

## Grok Patterns

```ruby
filter {
  grok {
    match => {
      "message" => "%{IP:client_ip} - - \[%{HTTPDATE:timestamp}\] \"%{WORD:method} %{URIPATHPARAM:request} HTTP/%{NUMBER:http_version}\" %{NUMBER:status} %{NUMBER:bytes}"
    }
  }
}
```

## JSON Parsing

```ruby
filter {
  json {
    source => "message"
    target => "parsed"
  }
}
```

## Multiline Logs

```ruby
input {
  file {
    path => "/var/log/app.log"
    codec => multiline {
      pattern => "^%{TIMESTAMP_ISO8601}"
      negate => true
      what => "previous"
    }
  }
}
```

## Error Handling

```ruby
filter {
  grok {
    match => { "message" => "%{COMBINEDAPACHELOG}" }
    tag_on_failure => ["_grokparsefailure"]
  }
}

output {
  if "_grokparsefailure" in [tags] {
    file {
      path => "/var/log/logstash/failures.log"
    }
  }
}
```

## Success Criteria
✅ Grok patterns parsing logs  
✅ JSON logs parsed correctly  
✅ Multiline logs handled  
✅ Parse failures logged  

**Time:** 40 min
