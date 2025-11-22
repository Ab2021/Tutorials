# Lab 11: Kafka Streams DSL (Faust)

## Difficulty
🟡 Medium

## Estimated Time
60 mins

## Learning Objectives
-   Understand stream processing concepts (map, filter, group_by).
-   Use a Python streaming library (Faust) to mimic Kafka Streams.

## Problem Statement
Implement a "Word Count" application using Faust.
1.  Read from `sentences` topic.
2.  Split lines into words.
3.  Count occurrences of each word.
4.  Print the counts to stdout (or a topic).

## Starter Code
```python
import faust

app = faust.App('word-count', broker='kafka://localhost:9092')
topic = app.topic('sentences', value_type=str)

@app.agent(topic)
async def count_words(sentences):
    async for sentence in sentences:
        # Logic here...
        pass
```

## Hints
<details>
<summary>Hint 1</summary>
Faust tables (`app.Table`) are used for stateful aggregations like counting.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

```python
import faust

app = faust.App('word-count', broker='kafka://localhost:9092')
topic = app.topic('sentences', value_type=str)

# Table to store counts. Default value is 0.
word_counts = app.Table('word_counts', default=int)

@app.agent(topic)
async def count_words(sentences):
    async for sentence in sentences:
        for word in sentence.split():
            word_counts[word] += 1
            print(f"Word: {word}, Count: {word_counts[word]}")

if __name__ == '__main__':
    app.main()
```
Run with: `python worker.py worker -l info`
</details>
