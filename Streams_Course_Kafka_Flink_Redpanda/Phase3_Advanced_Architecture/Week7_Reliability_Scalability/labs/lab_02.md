# Lab 02: Backpressure Detection in Flink

## Difficulty
🟡 Medium

## Estimated Time
45 mins

## Learning Objectives
- Detect backpressure in Flink jobs
- Use Flink Web UI metrics
- Identify bottleneck operators

## Problem Statement
Create a Flink job with an intentionally slow sink. Access the Flink Web UI and identify which operator is causing backpressure using the backpressure monitoring feature.

## Starter Code
```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import SinkFunction
import time

class SlowSink(SinkFunction):
    def invoke(self, value, context):
        time.sleep(0.5)  # Slow sink
        print(f"Sink: {value}")

env = StreamExecutionEnvironment.get_execution_environment()
# TODO: Create pipeline with slow sink
```

## Hints
<details>
<summary>Hint 1</summary>
Access Flink Web UI at `http://localhost:8081` and navigate to the Backpressure tab.
</details>

<details>
<summary>Hint 2</summary>
The task with "HIGH" backpressure is usually the bottleneck.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import SinkFunction
from pyflink.common import Types
import time

class SlowSink(SinkFunction):
    def invoke(self, value, context):
        # Simulate slow external system
        time.sleep(0.5)
        print(f"Processed: {value}")

def run():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    
    # Fast source
    ds = env.from_collection(range(1000), type_info=Types.INT())
    
    # Fast map
    ds = ds.map(lambda x: x * 2)
    
    # Slow sink - this will cause backpressure
    ds.add_sink(SlowSink())
    
    env.execute("Backpressure Detection")

if __name__ == '__main__':
    run()
```

**Verification Steps:**
1. Run the job
2. Open `http://localhost:8081`
3. Click on the running job
4. Navigate to "Backpressure" tab
5. Observe "HIGH" backpressure on the Sink operator
6. The Map operator will show backpressure propagating upstream

**Expected Behavior:**
- Sink shows HIGH backpressure (it's the bottleneck)
- Map shows backpressure (waiting for sink)
- Source slows down automatically
</details>
