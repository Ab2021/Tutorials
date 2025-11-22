# Lab 04: WASM Data Transforms

## Difficulty
🔴 Hard

## Estimated Time
90 mins

## Learning Objectives
-   Understand Data Transforms in Redpanda.
-   Deploy a WASM function to mask data.

## Problem Statement
*Note: This feature is in technical preview/beta in some versions. Ensure you have a compatible version.*
Write a Go/Rust transform that reads from `input-topic`, replaces any text "SECRET" with "****", and writes to `output-topic`.

## Starter Code
```go
// main.go (Go example)
package main

import (
    "github.com/redpanda-data/redpanda/src/transform-sdk/go/transform"
)

func main() {
    transform.OnRecordWritten(doTransform)
}

func doTransform(e transform.WriteEvent) ([]transform.Record, error) {
    // Logic here
}
```

## Hints
<details>
<summary>Hint 1</summary>
Use `rpk transform init` to generate a project template.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Step 1: Init Project
```bash
rpk transform init --language=go my-transform
cd my-transform
```

### Step 2: Code
```go
package main

import (
    "bytes"
    "github.com/redpanda-data/redpanda/src/transform-sdk/go/transform"
)

func main() {
    transform.OnRecordWritten(doTransform)
}

func doTransform(e transform.WriteEvent) ([]transform.Record, error) {
    val := e.Record().Value()
    newVal := bytes.ReplaceAll(val, []byte("SECRET"), []byte("****"))
    
    return []transform.Record{
        {
            Key:   e.Record().Key(),
            Value: newVal,
        },
    }, nil
}
```

### Step 3: Deploy
```bash
rpk transform build
rpk transform deploy --input-topic=input --output-topic=output
```
</details>
