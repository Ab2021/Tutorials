# Lab 24.6: Synthetic Monitoring

## Objective
Implement synthetic monitoring for proactive issue detection.

## Learning Objectives
- Create synthetic checks
- Monitor uptime
- Test user journeys
- Alert on failures

---

## Blackbox Exporter

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_status_codes: [200]
      method: GET
  
  tcp_connect:
    prober: tcp
    timeout: 5s
```

## Prometheus Config

```yaml
scrape_configs:
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - https://example.com
        - https://api.example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

## Selenium Tests

```python
from selenium import webdriver

def test_login_flow():
    driver = webdriver.Chrome()
    driver.get("https://example.com/login")
    
    driver.find_element_by_id("username").send_keys("test@example.com")
    driver.find_element_by_id("password").send_keys("password")
    driver.find_element_by_id("submit").click()
    
    assert "Dashboard" in driver.title
    driver.quit()
```

## Success Criteria
✅ Synthetic checks running  
✅ Uptime monitored  
✅ User journeys tested  
✅ Alerts configured  

**Time:** 40 min
