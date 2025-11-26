# Lab: Day 35 - E2E Testing with Playwright

## Goal
Write an End-to-End test that drives a real browser.

## Prerequisites
- `pip install pytest-playwright`
- `playwright install`

## Step 1: The Test (`test_e2e.py`)

```python
import re
from playwright.sync_api import Page, expect

def test_homepage_has_title(page: Page):
    # 1. Go to URL
    page.goto("https://playwright.dev/")

    # 2. Assert Title
    expect(page).to_have_title(re.compile("Playwright"))

def test_get_started_link(page: Page):
    page.goto("https://playwright.dev/")

    # 3. Click Link
    page.get_by_role("link", name="Get started").click()

    # 4. Assert URL
    expect(page).to_have_url(re.compile(".*intro"))
    
    # 5. Assert Heading
    expect(page.get_by_role("heading", name="Installation")).to_be_visible()
```

## Step 2: Run It
```bash
pytest test_e2e.py
```
*   It runs "Headless" (invisible) by default.

## Step 3: Run Headed (See the browser)
```bash
pytest test_e2e.py --headed --slowmo 1000
```
*   You will see the browser open and click links.

## Challenge
Write a test for `http://google.com`.
1.  Type "Python" into the search box.
2.  Press Enter.
3.  Verify that the results page contains "Welcome to Python.org".
