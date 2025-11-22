# Lab 11: MiniCluster Test

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
-   Write a JUnit test using MiniCluster.

## Problem Statement
*Java Lab*. Write a JUnit test that starts a `MiniClusterWithClientResource`, submits a job, and verifies output.

## Starter Code
```java
@ClassRule
public static MiniClusterWithClientResource flinkCluster =
    new MiniClusterWithClientResource(...);
```

## Hints
<details>
<summary>Hint 1</summary>
Use `Sink.collect()` (test utility) to capture output.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```java
// Conceptual
@Test
public void testJob() throws Exception {
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    // ... build job ...
    env.execute();
    // Verify results
}
```
</details>
