# Lab: Day 53 - Real-Time Sync

## Goal
Sync edits across clients.

## Prerequisites
- `pip install fastapi uvicorn websockets redis asyncio`

## Step 1: The Code (`collab.py`)

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from redis import asyncio as aioredis
import asyncio
import json

app = FastAPI()
redis = aioredis.from_url("redis://localhost")

class ConnectionManager:
    def __init__(self):
        self.active_connections = {} # {doc_id: [ws1, ws2]}

    async def connect(self, websocket: WebSocket, doc_id: str):
        await websocket.accept()
        if doc_id not in self.active_connections:
            self.active_connections[doc_id] = []
        self.active_connections[doc_id].append(websocket)

    def disconnect(self, websocket: WebSocket, doc_id: str):
        self.active_connections[doc_id].remove(websocket)

    async def broadcast_local(self, message: str, doc_id: str):
        # Send to all local connections
        if doc_id in self.active_connections:
            for connection in self.active_connections[doc_id]:
                await connection.send_text(message)

manager = ConnectionManager()

# Background Task: Listen to Redis
async def redis_listener():
    pubsub = redis.pubsub()
    await pubsub.subscribe("doc_updates")
    async for message in pubsub.listen():
        if message["type"] == "message":
            data = json.loads(message["data"])
            doc_id = data["doc_id"]
            msg = data["message"]
            await manager.broadcast_local(msg, doc_id)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(redis_listener())

@app.websocket("/ws/{doc_id}")
async def websocket_endpoint(websocket: WebSocket, doc_id: str):
    await manager.connect(websocket, doc_id)
    try:
        while True:
            data = await websocket.receive_text()
            # 1. Publish to Redis (so other servers get it)
            await redis.publish("doc_updates", json.dumps({"doc_id": doc_id, "message": data}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, doc_id)
```

## Step 2: Run It
`uvicorn collab:app --reload --port 8001`

## Step 3: Test
1.  Open WebSocket client (Day 49 lab or Postman).
2.  Connect `ws://localhost:8001/ws/1`.
3.  Send "Hello".
4.  Open another client. Connect `ws://localhost:8001/ws/1`.
5.  Verify it receives "Hello".

## Challenge
Run **Two Instances** of the server on ports 8001 and 8002.
Connect Client A to 8001.
Connect Client B to 8002.
Verify they still sync (thanks to Redis).
