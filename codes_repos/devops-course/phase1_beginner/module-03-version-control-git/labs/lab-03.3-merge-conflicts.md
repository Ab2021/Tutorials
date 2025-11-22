# Lab 3.3: Handling Merge Conflicts

## 🎯 Objective

Face your fears! Learn what a merge conflict is, why it happens, and how to resolve it calmly. You will intentionally create a conflict and fix it.

## 📋 Prerequisites

-   Completed Lab 3.2.

## 📚 Background

### What is a Conflict?

Git is smart. If Alice changes line 10 and Bob changes line 20, Git merges them automatically.
**Conflict** happens when Alice changes line 10 and Bob *also* changes line 10. Git doesn't know which one is "correct," so it pauses and asks you.

---

## 🔨 Hands-On Implementation

### Part 1: Creating the Conflict ⚔️

1.  **Setup:**
    Create a new repo or use an existing one.
    Create `story.txt`:
    ```text
    Once upon a time, there was a cat.
    The cat liked to sleep.
    The End.
    ```
    Commit it to `main`.

2.  **Branch A (Dog Lover):**
    ```bash
    git checkout -b feature/dog
    ```
    Edit `story.txt` (Line 1):
    `Once upon a time, there was a DOG.`
    Commit: `git commit -am "Change cat to dog"`

3.  **Branch B (Bird Lover):**
    Go back to main first!
    ```bash
    git checkout main
    git checkout -b feature/bird
    ```
    Edit `story.txt` (Line 1):
    `Once upon a time, there was a BIRD.`
    Commit: `git commit -am "Change cat to bird"`

### Part 2: The Collision 💥

1.  **Merge Branch A:**
    ```bash
    git checkout main
    git merge feature/dog
    ```
    *Result:* Success. `story.txt` now has "DOG".

2.  **Merge Branch B:**
    ```bash
    git merge feature/bird
    ```
    *Result:*
    `CONFLICT (content): Merge conflict in story.txt`
    `Automatic merge failed; fix conflicts and then commit the result.`

### Part 3: Resolution 🛠️

1.  **Check Status:**
    ```bash
    git status
    ```
    *Output:* `both modified: story.txt`

2.  **Open the File:**
    Open `story.txt` in VS Code. You will see:
    ```text
    <<<<<<< HEAD
    Once upon a time, there was a DOG.
    =======
    Once upon a time, there was a BIRD.
    >>>>>>> feature/bird
    The cat liked to sleep.
    The End.
    ```

3.  **Decide:**
    You are the author. You decide the story. Let's make it a "Dragon".
    Delete the markers (`<<<<`, `====`, `>>>>`) and the old lines.
    New content:
    `Once upon a time, there was a DRAGON.`

4.  **Finalize:**
    ```bash
    git add story.txt
    git commit -m "Resolve conflict: Changed to Dragon"
    ```
    *Result:* Conflict resolved.

---

## 🎯 Challenges

### Challenge 1: Aborting (Difficulty: ⭐⭐)

**Scenario:** You started a merge, hit a conflict, and panicked. You want to go back to how it was before you typed `git merge`.
**Task:**
Find the command to abort a merge in progress.
*Hint: `git merge --help`*

### Challenge 2: "Theirs" vs "Ours" (Difficulty: ⭐⭐⭐)

**Scenario:** You know Branch B is 100% correct and you want to discard Branch A's changes entirely for that file.
**Task:**
Resolve a conflict using `git checkout --theirs filename`.

---

## 💡 Solution

<details>
<summary>Click to reveal Solutions</summary>

**Challenge 1:**
```bash
git merge --abort
```

**Challenge 2:**
```bash
git checkout --theirs story.txt
git add story.txt
git commit -m "Kept changes from feature/bird"
```
</details>

---

## 🔑 Key Takeaways

1.  **Don't Panic**: A conflict just means Git needs human help.
2.  **Read the Markers**: `HEAD` is where you are (Main). The other hash/name is what you are merging in.
3.  **Communicate**: In a real team, talk to the person who wrote the other code before deleting it!

---

## ⏭️ Next Steps

We've mastered local Git. Now let's share code with the world.

Proceed to **Lab 3.4: Remote Repositories (GitHub)**.
