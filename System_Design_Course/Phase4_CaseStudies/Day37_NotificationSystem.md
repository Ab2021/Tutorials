# Day 37: Design Notification System

## 1. Requirements
*   **Functional:** Send Email, SMS, Push Notification.
*   **Non-Functional:** High throughput, Reliability (Don't lose alerts), Rate Limiting (Don't spam users).
*   **Scale:** 10M notifications/minute.

## 2. Architecture
*   **Notification Service:** Entry point. Validates request.
*   **User Preferences DB:** "User A wants Email, not SMS".
*   **Message Queue:** Buffers requests (Kafka/RabbitMQ).
*   **Workers:**
    *   **Email Worker:** Calls SendGrid/SES.
    *   **SMS Worker:** Calls Twilio.
    *   **Push Worker:** Calls APNS (iOS) / FCM (Android).
*   **Logs:** Store status (Sent, Failed, Clicked).

## 3. Deduplication & Rate Limiting
*   **Deduplication:** If Service A sends "Alert" 10 times in 1s, send only 1.
    *   Use Redis Key: `alert:user_id:type`. TTL 5 mins.
*   **Rate Limiting:** "Max 3 SMS per day".
    *   Check Redis Counter before pushing to Queue.

## 4. Retry Mechanism
*   **Soft Failure:** Timeout. Retry immediately.
*   **Hard Failure:** Invalid Email. Don't retry.
*   **Exponential Backoff:** Wait 1s, 2s, 4s, 8s.
*   **Dead Letter Queue (DLQ):** After 5 retries, move to DLQ for manual inspection.
