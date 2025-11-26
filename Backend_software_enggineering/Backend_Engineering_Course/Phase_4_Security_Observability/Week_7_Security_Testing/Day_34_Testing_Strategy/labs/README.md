# Lab: Day 34 - Unit Testing with Pytest

## Goal
Write robust unit tests and learn how to mock dependencies.

## Prerequisites
- `pip install pytest pytest-cov`

## Step 1: The Code (`calculator.py`)

```python
import requests

class Calculator:
    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def get_currency_rate(self, currency):
        # External API call
        response = requests.get(f"https://api.exchangerate.com/{currency}")
        return response.json()['rate']
```

## Step 2: The Test (`test_calculator.py`)

```python
import pytest
from unittest.mock import patch, MagicMock
from calculator import Calculator

@pytest.fixture
def calc():
    return Calculator()

def test_add(calc):
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0

def test_divide(calc):
    assert calc.divide(10, 2) == 5
    with pytest.raises(ValueError):
        calc.divide(10, 0)

# Mocking external API
@patch('calculator.requests.get')
def test_get_currency_rate(mock_get, calc):
    # Setup Mock
    mock_response = MagicMock()
    mock_response.json.return_value = {'rate': 1.5}
    mock_get.return_value = mock_response

    # Run
    rate = calc.get_currency_rate("USD")

    # Assert
    assert rate == 1.5
    mock_get.assert_called_once_with("https://api.exchangerate.com/USD")
```

## Step 3: Run It
```bash
pytest -v --cov=calculator
```
*   **Output**: 3 passed. 100% coverage.

## Challenge
Add a `multiply` method to `Calculator` but make it buggy (e.g., returns `a + b`).
Write a test case that fails.
Fix the code.
