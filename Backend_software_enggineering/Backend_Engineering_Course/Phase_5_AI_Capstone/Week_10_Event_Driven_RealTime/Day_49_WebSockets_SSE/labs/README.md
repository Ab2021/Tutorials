# Lab: Day 49 - WebSocket Chat

## Goal
Build a real-time chat server.

## Prerequisites
- `pip install fastapi uvicorn websockets`

## Step 1: The Server (`server.py`)

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client #{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
```

## Step 2: The Client (`index.html`)

```html
<!DOCTYPE html>
<html>
    <body>
        <h1>WebSocket Chat</h1>
        <input type="text" id="messageText" autocomplete="off"/>
        <button onclick="sendMessage()">Send</button>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now()
            var ws = new WebSocket("ws://localhost:8000/ws/" + client_id);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage() {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
            }
        </script>
    </body>
</html>
```

## Step 3: Run It
1.  `uvicorn server:app --reload`
2.  Open `index.html` in 2 different browser tabs.
3.  Chat!

## Challenge
Implement **Private Messaging**.
1.  Store connections in a Dict: `active_connections = {client_id: websocket}`.
2.  Send message format: `to_id:message`.
3.  Parse and route to specific socket.
