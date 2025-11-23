# Lab 30.7: Smoke Testing

## Objective
Create comprehensive smoke tests for production deployments.

## Learning Objectives
- Design smoke test suites
- Test critical paths
- Automate execution
- Integrate with CI/CD

---

## Smoke Test Suite

```python
import requests
import pytest

BASE_URL = "https://api.example.com"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_authentication():
    response = requests.post(f"{BASE_URL}/auth/login", json={
        "username": "test@example.com",
        "password": "password"
    })
    assert response.status_code == 200
    assert "token" in response.json()

def test_critical_endpoint():
    response = requests.get(f"{BASE_URL}/api/products")
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_database_connectivity():
    response = requests.get(f"{BASE_URL}/api/status/db")
    assert response.json()["database"] == "connected"
```

## CI/CD Integration

```yaml
# .github/workflows/deploy.yaml
- name: Deploy to production
  run: kubectl apply -f k8s/

- name: Wait for rollout
  run: kubectl rollout status deployment/myapp

- name: Run smoke tests
  run: |
    pytest tests/smoke/ --verbose
    if [ $? -ne 0 ]; then
      kubectl rollout undo deployment/myapp
      exit 1
    fi
```

## Success Criteria
✅ Smoke tests cover critical paths  
✅ Tests run automatically  
✅ Failures trigger rollback  

**Time:** 40 min
