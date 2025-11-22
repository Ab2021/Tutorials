# Lab 02: rpk CLI Basics

## Difficulty
🟢 Easy

## Estimated Time
30 mins

## Learning Objectives
-   Master the `rpk` (Redpanda Keeper) CLI.
-   Create topics, produce, and consume without writing code.

## Problem Statement
1.  Create a topic `chat-room` with 5 partitions.
2.  Produce 3 messages ("Hello", "World", "Redpanda") using `rpk`.
3.  Consume them using `rpk` with offset `oldest`.

## Starter Code
```bash
rpk topic create ...
rpk topic produce ...
```

## Hints
<details>
<summary>Hint 1</summary>
`rpk topic produce` reads from stdin. You can pipe data into it.
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

### Commands
```bash
# Create Topic
rpk topic create chat-room -p 5 -r 1

# Produce
echo "Hello" | rpk topic produce chat-room
echo "World" | rpk topic produce chat-room
echo "Redpanda" | rpk topic produce chat-room

# Consume
rpk topic consume chat-room --offset oldest
```
</details>
