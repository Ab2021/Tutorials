# Lab 13: CoProcessFunction (Connect)

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Connect two streams.
-   Share state between streams.

## Problem Statement
Stream A: `ControlStream` (Switch: ON/OFF).
Stream B: `DataStream` (Words).
Requirement: Only print words from Stream B if the Switch (Stream A) is ON.

## Starter Code
```python
connected = control_stream.connect(data_stream)
connected.process(MyCoProcessFunction())
```

## Hints
<details>
<summary>Hint 1</summary>
You need a `ValueState` to store the current switch status.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream.functions import CoProcessFunction

class Switch(CoProcessFunction):
    def process_element1(self, value, ctx, out):
        # Control Stream
        self.enabled = (value == "ON")

    def process_element2(self, value, ctx, out):
        # Data Stream
        if getattr(self, 'enabled', False):
            out.collect(value)

# Note: In real Flink, you'd use Flink State (ValueState) to persist 'enabled' across restarts.
# For this simple lab, a python attribute works if no failure.
```
</details>
