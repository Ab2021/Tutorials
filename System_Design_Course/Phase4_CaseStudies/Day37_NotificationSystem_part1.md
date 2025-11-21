# Day 37 Deep Dive: Priority & Templates

## 1. Priority Queues
*   **Problem:** Marketing emails (Bulk) shouldn't block OTPs (Critical).
*   **Solution:** Separate Queues.
    *   **Critical Queue:** OTP, Password Reset. High number of workers.
    *   **Normal Queue:** Social updates.
    *   **Bulk Queue:** Marketing. Low priority.
*   **Worker Logic:** `while (true) { check(Critical) || check(Normal) || check(Bulk) }`.

## 2. Template Engine
*   **Requirement:** "Hello {name}, your order {id} is ready."
*   **Storage:** Store templates in DB/S3.
*   **Rendering:** Worker fetches template, substitutes variables (Mustache/Jinja), and generates HTML.
*   **Optimization:** Cache compiled templates in memory.

## 3. Tracking & Analytics
*   **Open Tracking:** Embed a 1x1 transparent pixel image in email. `<img src="server.com/track?id=123">`.
*   **Click Tracking:** Rewrite links. `<a href="server.com/click?dest=google.com">`.
*   **Data Pipeline:** Tracking Server -> Kafka -> Hadoop (Analytics).

## 4. Bulk Sending
*   **Problem:** Sending 1M emails.
*   **Batching:** Don't call SendGrid 1M times. Call `send_batch(1000 emails)`.
*   **Flow Control:** Don't overwhelm the 3rd party provider. Respect their Rate Limits.
