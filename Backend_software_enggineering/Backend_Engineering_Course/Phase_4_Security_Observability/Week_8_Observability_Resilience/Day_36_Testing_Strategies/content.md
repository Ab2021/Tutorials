# Day 36: Testing Strategies - Ensuring Quality

## Table of Contents
1. [Testing Fundamentals](#1-testing-fundamentals)
2. [Unit Testing](#2-unit-testing)
3. [Integration Testing](#3-integration-testing)
4. [End-to-End Testing](#4-end-to-end-testing)
5. [Test Coverage](#5-test-coverage)
6. [Mocking & Fixtures](#6-mocking--fixtures)
7. [TDD (Test-Driven Development)](#7-tdd-test-driven-development)
8. [CI/CD Integration](#8-cicd-integration)
9. [Performance Testing](#9-performance-testing)
10. [Summary](#10-summary)

---

## 1. Testing Fundamentals

### 1.1 Testing Pyramid

```
       /\
      /E2E\       (Few, slow, expensive)
     /------\
    /  Int   \    (Some, medium speed)
   /----------\
  /    Unit    \  (Many, fast, cheap)
 /--------------\
```

**Unit Tests**: 70% (test individual functions)  
**Integration Tests**: 20% (test components together)  
**E2E Tests**: 10% (test full user workflows)

### 1.2 Why Test?

✅ Catch bugs early  
✅ Refactor with confidence  
✅ Document behavior  
✅ Improve design

---

## 2. Unit Testing

### 2.1 pytest Basics

```python
# test_calculator.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```

**Run tests**:
```bash
pytest test_calculator.py
```

### 2.2 Testing API Endpoints

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"id": user_id, "name": "Alice"}

# Test
client = TestClient(app)

def test_get_user():
    response = client.get("/users/123")
    assert response.status_code == 200
    assert response.json() == {"id": 123, "name": "Alice"}

def test_get_user_not_found():
    response = client.get("/users/999")
    assert response.status_code == 404
```

### 2.3 Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300)
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

---

## 3. Integration Testing

### 3.1 Database Testing

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use in-memory SQLite for tests
@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()

def test_create_user(db_session):
    user = User(name="Alice", email="alice@example.com")
    db_session.add(user)
    db_session.commit()
    
    retrieved = db_session.query(User).filter(User.email == "alice@example.com").first()
    assert retrieved.name == "Alice"
```

### 3.2 API Integration Tests

```python
def test_create_and_get_user():
    # Create user
    create_response = client.post("/users", json={
        "name": "Alice",
        "email": "alice@example.com"
    })
    assert create_response.status_code == 201
    user_id = create_response.json()["id"]
    
    # Get user
    get_response = client.get(f"/users/{user_id}")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Alice"
```

### 3.3 External Service Mocking

```python
from unittest.mock import patch

@patch('requests.get')
def test_fetch_external_data(mock_get):
    # Mock external API response
    mock_get.return_value.json.return_value = {"data": "mocked"}
    mock_get.return_value.status_code = 200
    
    response = fetch_from_external_api()
    assert response == {"data": "mocked"}
```

---

## 4. End-to-End Testing

### 4.1 Playwright (Python)

```python
from playwright.sync_api import sync_playwright

def test_login_flow():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Navigate to login page
        page.goto("http://localhost:3000/login")
        
        # Fill form
        page.fill("#email", "alice@example.com")
        page.fill("#password", "password123")
        page.click("#login-button")
        
        # Assert redirected to dashboard
        page.wait_for_url("http://localhost:3000/dashboard")
        assert page.title() == "Dashboard"
        
        browser.close()
```

### 4.2 Selenium

```python
from selenium import webdriver
from selenium.webdriver.common.by import By

def test_search():
    driver = webdriver.Chrome()
    
    driver.get("http://localhost:3000")
    
    search_box = driver.find_element(By.ID, "search")
    search_box.send_keys("test query")
    search_box.submit()
    
    results = driver.find_elements(By.CLASS_NAME, "search-result")
    assert len(results) > 0
    
    driver.quit()
```

---

## 5. Test Coverage

### 5.1 pytest-cov

```bash
# Install
pip install pytest-cov

# Run with coverage
pytest --cov=myapp --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 5.2 Coverage Goals

- **Critical code**: 100% coverage
- **Business logic**: 90%+ coverage
- **Overall**: 80%+ coverage

### 5.3 Example Coverage

```python
# myapp/calculator.py
def divide(a, b):
    if b == 0:  # Edge case
        raise ValueError("Cannot divide by zero")
    return a / b

# tests/test_calculator.py
def test_divide():
    assert divide(10, 2) == 5

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

# Coverage: 100% (both normal case and edge case tested)
```

---

## 6. Mocking & Fixtures

### 6.1 pytest Fixtures

```python
import pytest

@pytest.fixture
def sample_user():
    return {"id": 1, "name": "Alice", "email": "alice@example.com"}

def test_user_name(sample_user):
    assert sample_user["name"] == "Alice"

def test_user_email(sample_user):
    assert sample_user["email"] == "alice@example.com"
```

### 6.2 Fixture Scope

```python
@pytest.fixture(scope="session")
def db_connection():
    """Created once per test session"""
    conn = create_connection()
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def api_client():
    """Created once per test module"""
    return TestClient(app)

@pytest.fixture(scope="function")
def clean_db():
    """Created for each test function"""
    reset_database()
```

### 6.3 Mocking with unittest.mock

```python
from unittest.mock import Mock, patch

# Mock object
mock_db = Mock()
mock_db.query.return_value = [{"id": 1, "name": "Alice"}]

def test_get_users():
    users = get_users(mock_db)
    assert len(users) == 1
    assert users[0]["name"] == "Alice"

# Patch function
@patch('myapp.send_email')
def test_user_signup(mock_send_email):
    signup_user("alice@example.com")
    
    # Verify email was sent
    mock_send_email.assert_called_once_with("alice@example.com", subject="Welcome")
```

---

## 7. TDD (Test-Driven Development)

### 7.1 TDD Cycle

```
1. RED:   Write failing test
2. GREEN: Write minimal code to pass
3. REFACTOR: Improve code
```

### 7.2 Example TDD Flow

**Step 1: Write test (RED)**:
```python
def test_get_user_by_email():
    user = get_user_by_email("alice@example.com")
    assert user["name"] == "Alice"

# Run test → FAILS (function doesn't exist)
```

**Step 2: Minimal implementation (GREEN)**:
```python
def get_user_by_email(email):
    # Hardcoded to pass test
    return {"name": "Alice"}

# Run test → PASSES
```

**Step 3: Refactor (improve)**:
```python
def get_user_by_email(email):
    # Proper implementation
    return db.query(User).filter(User.email == email).first()

# Run test → still PASSES
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=myapp --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### 8.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

---

## 9. Performance Testing

### 9.1 Locust

```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def get_users(self):
        self.client.get("/users")
    
    @task(3)  # 3x more frequent
    def get_user_by_id(self):
        self.client.get("/users/123")

# Run: locust -f locustfile.py --host=http://localhost:8000
```

### 9.2 pytest-benchmark

```python
import pytest

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_fibonacci_performance(benchmark):
    result = benchmark(fibonacci, 10)
    assert result == 55

# Output shows execution time statistics
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Testing Pyramid** - Many unit, some integration, few E2E
2. ✅ **pytest** - Python testing framework
3. ✅ **Fixtures** - Reusable test setup
4. ✅ **Mocking** - Isolate components
5. ✅ **Coverage** - Aim for 80%+
6. ✅ **TDD** - Write tests first
7. ✅ **CI/CD** - Automate testing

### 10.2 Testing Checklist

- [ ] Unit tests for business logic
- [ ] Integration tests for API endpoints
- [ ] E2E tests for critical user flows
- [ ] 80%+ code coverage
- [ ] Tests run in CI/CD
- [ ] Pre-commit hooks configured
- [ ] Performance tests for critical paths
- [ ] Mock external services

### 10.3 Tomorrow (Day 37): Observability & Tracing

- **Distributed tracing**: OpenTelemetry, Jaeger
- **Spans & traces**: Request flow visualization
- **Context propagation**: Trace IDs across services
- **Instrumentation**: Auto vs manual
- **Production patterns**: Sampling, performance impact

See you tomorrow! 🚀

---

**File Statistics**: ~950 lines | Testing Strategies mastered ✅
