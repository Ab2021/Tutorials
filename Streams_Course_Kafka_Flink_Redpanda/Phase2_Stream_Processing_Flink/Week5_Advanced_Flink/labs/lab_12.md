# Lab 12: TestHarness (ProcessFunction)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use `KeyedOneInputStreamOperatorTestHarness`.
-   Test time-dependent logic.

## Problem Statement
*Java Lab*. Test a ProcessFunction that sets a timer for 1 minute.
1.  Push element.
2.  Advance time by 1 minute.
3.  Verify output.

## Starter Code
```java
harness.processElement("A", 1000);
harness.setProcessingTime(1000 + 60000);
```

## Hints
<details>
<summary>Hint 1</summary>
Harness allows full control over time.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
// Conceptual
OneInputStreamOperatorTestHarness<String, String, String> harness = 
    new KeyedOneInputStreamOperatorTestHarness<>(operator, ...);

harness.open();
harness.processElement("key", 100);
harness.setProcessingTime(60100); // Trigger timer
// Assert output contains alert
```
</details>
