# Day 41 Deep Dive: Idempotency & Reconciliation

## 1. Idempotency (The Holy Grail)
*   **Problem:** Network timeout. Did the payment go through?
    *   If I retry, I might double charge.
    *   If I don't, user might get free product.
*   **Solution:** Idempotency Key (UUID).
    *   Client generates `Key = hash(User, Amount, Time)`.
    *   Server checks DB: "Have I seen this Key?"
        *   **Yes:** Return previous response (Cached).
        *   **No:** Process payment. Save Key.
*   **DB Constraint:** `UNIQUE(idempotency_key)`.

## 2. Reconciliation (The Safety Net)
*   **Problem:** Bugs happen. Distributed systems drift.
    *   Our DB says `FAILED`. Stripe says `SUCCESS`.
*   **Solution:** Nightly Reconciliation Job.
    1.  Download "Settlement Report" from Stripe (CSV).
    2.  Compare with internal DB.
    3.  **Match:** Good.
    4.  **Mismatch:** Flag for manual review (or auto-refund).

## 3. Handling Distributed Transactions
*   **Scenario:** Update Wallet (Internal) AND Call Stripe (External).
*   **Pattern:** Two-Phase Commit is impossible with external API.
*   **Pattern:** **State Machine**.
    *   `CREATED` -> `CONTACTING_PSP` -> `PSP_SUCCESS` -> `WALLET_UPDATED`.
    *   If crash at `PSP_SUCCESS`, a background worker sees the stuck state and retries the `WALLET_UPDATED` step.
