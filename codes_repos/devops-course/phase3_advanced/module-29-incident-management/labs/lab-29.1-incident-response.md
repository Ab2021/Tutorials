# Lab 29.1: Incident Response Simulation

## 🎯 Objective

Don't panic. When the pager goes off at 3 AM, you need a process. You will simulate the **Incident Lifecycle**: Trigger, Acknowledge, Triage, Resolve.

## 📋 Prerequisites

-   Python installed.
-   (Optional) PagerDuty/Opsgenie Free Account.

## 📚 Background

### The Lifecycle
1.  **Trigger**: Monitoring system detects failure. Pages the On-Call Engineer.
2.  **Ack (Acknowledge)**: "I am looking at it." Stops the escalation policy.
3.  **Triage**: "Is this critical? Do I need help?"
4.  **Resolve**: "Fixed."

---

## 🔨 Hands-On Implementation

### Part 1: The Simulator Script 🐍

We will write a script to simulate an Alerting System.

1.  **Create `pager.py`:**
    ```python
    import time
    import sys

    status = "OK"
    on_call = "Alice"
    escalation = ["Alice", "Bob", "Charlie"]

    def trigger_incident(details):
        global status
        status = "TRIGGERED"
        print(f"🚨 ALERT: {details}")
        page_engineer(0)

    def page_engineer(level):
        person = escalation[level]
        print(f"📟 Paging {person}...")
        response = input(f"Are you {person}? (ack/ignore): ")
        
        if response == "ack":
            print(f"✅ Incident Acknowledged by {person}.")
            solve_incident()
        else:
            print(f"❌ {person} did not answer. Escalating...")
            if level + 1 < len(escalation):
                page_engineer(level + 1)
            else:
                print("💀 Major Outage! No one answered.")

    def solve_incident():
        print("🛠️  Investigating... (Check Logs, Metrics)")
        time.sleep(2)
        print("💡 Found root cause: Database CPU 100%.")
        action = input("Action (restart/ignore): ")
        if action == "restart":
            print("✅ Service Restarted. Incident Resolved.")
        else:
            print("⚠️  Incident still active.")

    if __name__ == "__main__":
        trigger_incident("Database Connection Failed")
    ```

### Part 2: Run the Simulation 🏃‍♂️

1.  **Scenario A (Happy Path):**
    Run `python pager.py`.
    Type `ack`.
    Type `restart`.
    *Result:* Incident resolved by Alice.

2.  **Scenario B (Escalation):**
    Run `python pager.py`.
    Type `ignore` (Alice is sleeping).
    Type `ack` (Bob answers).
    *Result:* Escalation worked.

### Part 3: Runbook 📖

An alert without a Runbook is useless.

1.  **Create `RUNBOOK.md`:**
    ```markdown
    # Runbook: Database Connection Failed

    ## Severity: High
    ## Symptoms: 500 Errors on Frontend.

    ## Steps:
    1. Check RDS CPU Utilization in CloudWatch.
    2. If CPU > 90%, check for long-running queries:
       `SELECT * FROM pg_stat_activity WHERE state = 'active';`
    3. Kill bad queries:
       `SELECT pg_terminate_backend(pid);`
    4. If unresponsive, Reboot.
    ```

---

## 🎯 Challenges

### Challenge 1: Webhook Integration (Difficulty: ⭐⭐⭐)

**Task:**
Modify the script to send a real message to a Slack channel using a Webhook URL.
`requests.post("https://hooks.slack.com/...", json={"text": "🚨 Alert!"})`

### Challenge 2: On-Call Schedule (Difficulty: ⭐⭐)

**Task:**
Modify the script to pick the on-call person based on the current time.
(e.g., Alice: 9-5, Bob: 5-9).

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 2:**
```python
hour = datetime.datetime.now().hour
if 9 <= hour < 17:
    person = "Alice"
else:
    person = "Bob"
```
</details>

---

## 🔑 Key Takeaways

1.  **MTTA (Mean Time To Acknowledge)**: How fast you pick up the phone.
2.  **MTTR (Mean Time To Resolve)**: How fast you fix it.
3.  **Burnout**: If the pager goes off every night, people will quit. Fix the root cause.

---

## ⏭️ Next Steps

The fire is out. How do we prevent it next time?

Proceed to **Lab 29.2: Blameless Post-Mortems**.
