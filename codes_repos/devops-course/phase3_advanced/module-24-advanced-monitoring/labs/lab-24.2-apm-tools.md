# Lab 24.2: APM Tools

## Objective
Implement Application Performance Monitoring with New Relic or Datadog.

## Learning Objectives
- Install APM agent
- Monitor application performance
- Analyze traces
- Set up alerts

---

## New Relic Setup

```python
# Install
pip install newrelic

# Configure
newrelic-admin generate-config LICENSE_KEY newrelic.ini

# Run app
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python app.py
```

## Custom Instrumentation

```python
import newrelic.agent

@newrelic.agent.function_trace()
def slow_function():
    # Your code
    pass

@newrelic.agent.background_task()
def background_job():
    # Background task
    pass
```

## Success Criteria
✅ APM agent installed  
✅ Traces visible  
✅ Performance metrics collected  
✅ Alerts configured  

**Time:** 40 min
