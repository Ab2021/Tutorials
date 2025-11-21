# Day 41: Design Payment System

## 1. Requirements
*   **Functional:** Move money from User A to User B (or Merchant). Support Refunds.
*   **Non-Functional:**
    *   **Zero Data Loss:** Money cannot disappear.
    *   **Exactly-Once:** Cannot charge user twice.
    *   **Consistency:** ACID is mandatory.

## 2. Architecture
*   **Payment Service:** Orchestrates the flow.
*   **PSP (Payment Service Provider):** Stripe, PayPal, Visa. (External).
*   **Ledger Service:** Double-entry accounting system.
*   **Wallet Service:** Manages user balance.

## 3. The Flow
1.  User clicks "Pay".
2.  **Payment Service** creates a `Payment` record (Status: `PENDING`).
3.  Calls **PSP** (Stripe) with a unique `Idempotency Key`.
4.  **PSP** responds: `SUCCESS`.
5.  **Payment Service** updates DB: `SUCCESS`.
6.  **Ledger Service** records the transaction.

## 4. Double-Entry Ledger
*   Every transaction has two entries: Debit and Credit.
*   `Sum(Debits) == Sum(Credits)`.
*   **Example:** User pays $10 to Merchant.
    *   `User Wallet`: Debit $10.
    *   `Merchant Wallet`: Credit $10.
*   **Benefit:** Easy to audit. If sum is not zero, money is missing.
