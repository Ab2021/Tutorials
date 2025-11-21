# Day 42 Deep Dive: Google Docs Architecture

## 1. Architecture
*   **Client:** Browser. Sends operations via WebSocket.
*   **Collaboration Server:** Receives ops. Transforms them. Broadcasts to other clients.
*   **Storage:** Periodic snapshots to DB (Bigtable). Replay ops from snapshot to reconstruct current state.

## 2. OT Algorithm (Simplified)
```python
def transform(op1, op2):
    """Transform op1 against op2."""
    if op1.type == "insert" and op2.type == "insert":
        if op1.pos < op2.pos:
            return op1  # No change
        else:
            return Insert(op1.char, op1.pos + len(op2.char))
    # ... handle delete, etc.
```

## 3. Presence & Cursors
*   **Cursor Position:** Each user's cursor is broadcast.
*   **Highlighting:** Show colored cursor for each user.
*   **Challenge:** Cursor positions must also be transformed when text changes.

## 4. Offline Editing
*   **Local Queue:** Store ops locally.
*   **Reconnect:** Send queued ops to server.
*   **Conflict:** Server transforms and merges.
*   **CRDT Advantage:** Works better for offline (no central server needed for transformation).

## 5. Case Study: Figma's Multiplayer
*   Uses CRDT for design elements.
*   **Optimization:** Only sync visible viewport (don't send off-screen changes).
*   **Compression:** Delta encoding for vector paths.
