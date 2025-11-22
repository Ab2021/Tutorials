# Lab 30.1: Feature Flags (Decoupling Deploy from Release)

## 🎯 Objective

Deploy on Friday. Release on Monday. **Feature Flags** allow you to push code to production but keep it hidden from users. You can then toggle it on for specific users (Canary) or everyone instantly.

## 📋 Prerequisites

-   Python installed.
-   (Optional) Unleash/LaunchDarkly account.

## 📚 Background

### Concepts
-   **Deploy**: Moving code to the server. (Technical).
-   **Release**: Making features visible to users. (Business).
-   **Toggle**: A simple `if` statement wrapping the new code.

---

## 🔨 Hands-On Implementation

### Part 1: The Monolith (Hardcoded) 🗿

1.  **Create `app.py`:**
    ```python
    def checkout():
        # Old Checkout
        print("Processing payment via Stripe...")

    if __name__ == "__main__":
        checkout()
    ```

### Part 2: The Flag 🚩

1.  **Update `app.py`:**
    ```python
    import os
    import json

    # Load flags from a file (Simulating a Flag Server)
    FLAGS = json.load(open("flags.json"))

    def is_enabled(feature_name, user_id):
        flag = FLAGS.get(feature_name, {})
        if not flag.get("enabled"):
            return False
        # Percentage Rollout
        if user_id % 100 < flag.get("percentage", 0):
            return True
        return False

    def checkout(user_id):
        if is_enabled("new_checkout", user_id):
            print(f"User {user_id}: 🚀 Using NEW PayPal Checkout!")
        else:
            print(f"User {user_id}: 🐢 Using OLD Stripe Checkout.")

    if __name__ == "__main__":
        # Simulate 10 users
        for i in range(10):
            checkout(i)
    ```

2.  **Create `flags.json`:**
    ```json
    {
      "new_checkout": {
        "enabled": true,
        "percentage": 30
      }
    }
    ```

### Part 3: Test Rollout 🧪

1.  **Run:**
    `python app.py`.
    *Result:* Users 0, 1, 2 get New Checkout. Users 3-9 get Old.

2.  **Update Flag:**
    Change `"percentage": 100` in `flags.json`.
    Run again.
    *Result:* Everyone gets New Checkout.
    *Note:* We changed the behavior *without* changing the code!

---

## 🎯 Challenges

### Challenge 1: User Targeting (Difficulty: ⭐⭐)

**Task:**
Add a `whitelist` to the flag.
"If user_id is 99 (Admin), always show new feature."
Update `is_enabled` logic.

### Challenge 2: Kill Switch (Difficulty: ⭐)

**Task:**
Simulate a bug in New Checkout.
Set `"enabled": false` in `flags.json`.
Verify that 100% of traffic goes back to Old Checkout immediately.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```python
if user_id in flag.get("whitelist", []):
    return True
```
</details>

---

## 🔑 Key Takeaways

1.  **Trunk Based Development**: Feature Flags enable you to merge incomplete code to `main` without breaking production (just keep the flag off).
2.  **A/B Testing**: Flags allow you to measure which version performs better.
3.  **Technical Debt**: Flags are debt. Once the feature is 100% released, **delete the flag code**.

---

## ⏭️ Next Steps

Code is easy to change. Data is hard.

Proceed to **Lab 30.2: Zero Downtime Database Migrations**.
