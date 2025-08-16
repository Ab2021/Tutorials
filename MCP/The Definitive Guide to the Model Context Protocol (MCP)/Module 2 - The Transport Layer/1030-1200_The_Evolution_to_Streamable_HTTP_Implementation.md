# The Definitive Guide to the Model Context Protocol (MCP)

## Module 2: The Transport Layer: How MCP Communicates

### Lesson 2.2: The Evolution to Streamable HTTP (10:30 - 12:00)

### **Implementation & Examples**

---

### **1. Streamable HTTP in Practice: A Python (Flask) Implementation**

This section provides a complete, runnable example demonstrating the Streamable HTTP model. We will build a single Flask server that exposes one endpoint, `/mcp`. This endpoint will intelligently decide whether to respond with a simple JSON object or to upgrade the connection to a `text/event-stream` based on the request.

**The Scenario:**

1.  **Stateless Server:** The server will maintain a simple in-memory dictionary to store session data, keyed by a session token provided by the client. This simulates a stateless architecture where session context could be retrieved from a database or cache.
2.  **Simple Request:** The client will first send a `session/start` request. The server will respond with a new session token. This will be a simple request-response interaction.
3.  **Streaming Request:** The client will then use the session token to make a `tools/call` request to a long-running `job/run` tool. The server will respond to this by upgrading the connection to a stream, sending the initial result, and then pushing progress notifications over the open connection.

#### **The MCP Server (`mcp_streamable_server.py`)**

This script uses the Flask web framework to create the server. It manages sessions and handles the logic for upgrading connections to streams.

```python
# mcp_streamable_server.py

from flask import Flask, request, Response, jsonify
import json
import time
import uuid
import logging

# --- Server Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# In-memory storage to simulate a session database (e.g., Redis, a DB).
# This allows the server itself to be stateless.
SESSIONS = {}

# --- Helper Functions ---

def get_session(token):
    """Retrieves session data using the token."""
    return SESSIONS.get(token)

def create_session():
    """Creates a new session and returns the token."""
    token = str(uuid.uuid4())
    SESSIONS[token] = {"history": []}
    logging.info(f"Created new session: {token}")
    return token

# --- The Unified MCP Endpoint ---

@app.route('/mcp', methods=['POST'])
def handle_mcp_message():
    """The single endpoint for all MCP communication."""
    json_rpc_request = request.json
    method = json_rpc_request.get("method")
    params = json_rpc_request.get("params", {})
    request_id = json_rpc_request.get("id")

    # Check for session token in headers for all methods except session/start
    session_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if method != "session/start" and not get_session(session_token):
        return jsonify({
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32001, "message": "Invalid or expired session token"}
        }), 401

    # --- Method Routing ---

    if method == "session/start":
        # Simple Request-Response: Create a session and return the token.
        token = create_session()
        return jsonify({
            "jsonrpc": "2.0", "id": request_id,
            "result": {"session_token": token}
        })

    elif method == "tools/call" and params.get("name") == "job/run":
        # Upgrade to Stream: Handle a long-running job.
        
        def event_stream():
            # 1. Send the initial response for the request.
            initial_response = {
                "jsonrpc": "2.0", "id": request_id,
                "result": {"status": "Job started. Awaiting progress..."}
            }
            yield f"data: {json.dumps(initial_response)}\n\n"
            logging.info(f"Upgraded connection to stream for session {session_token}")

            # 2. Simulate the long-running job and send progress notifications.
            for i in range(5):
                time.sleep(1.5)
                progress = (i + 1) * 20
                notification = {
                    "jsonrpc": "2.0",
                    "method": "job/progress",
                    "params": {"job_id": params.get("arguments",{}).get("job_id"), "progress": progress}
                }
                yield f"data: {json.dumps(notification)}\n\n"
                logging.info(f"Sent progress update {progress}% for session {session_token}")
            
            # 3. Send a final notification that the job is done.
            final_notification = {
                "jsonrpc": "2.0",
                "method": "job/completed",
                "params": {"job_id": params.get("arguments",{}).get("job_id"), "final_status": "Success"}
            }
            yield f"data: {json.dumps(final_notification)}\n\n"
            logging.info(f"Sent final notification for session {session_token}")

        # Return the Response object with the special mimetype to start the stream.
        return Response(event_stream(), mimetype='text/event-stream')

    else:
        # Handle other simple request-response methods here.
        return jsonify({
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": "Method not found"}
        }), 404

if __name__ == '__main__':
    # To run: pip install Flask
    # Then: python mcp_streamable_server.py
    app.run(port=5002, threaded=True)

```

#### **The Python Client (`mcp_streamable_client.py`)**

This client script will use the popular `requests` library to interact with our Flask server. It will demonstrate how to handle both a simple response and a streaming response.

```python
# mcp_streamable_client.py

import requests
import json

class MCPStreamableClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_token = None
        self.request_id_counter = 0

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
        return headers

    def start_session(self):
        """Demonstrates a simple request-response call."""
        print("CLIENT: Attempting to start a new session...")
        self.request_id_counter += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id_counter,
            "method": "session/start"
        }
        response = requests.post(f"{self.base_url}/mcp", json=payload, headers=self._get_headers())
        response.raise_for_status()
        
        data = response.json()
        self.session_token = data.get("result", {}).get("session_token")
        print(f"CLIENT: Session started successfully. Token: {self.session_token}\n")

    def run_long_job(self):
        """Demonstrates a call that gets upgraded to a stream."""
        if not self.session_token:
            print("CLIENT: No session token. Please start a session first.")
            return

        print("CLIENT: Requesting to run a long job...")
        self.request_id_counter += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id_counter,
            "method": "tools/call",
            "params": {"name": "job/run", "arguments": {"job_id": "job-123"}}
        }
        
        # The key is to set stream=True in the requests call
        with requests.post(f"{self.base_url}/mcp", json=payload, headers=self._get_headers(), stream=True) as resp:
            resp.raise_for_status()
            print(f"CLIENT: Server responded with status {resp.status_code}. Connection open.")
            print("CLIENT: Content-Type: ", resp.headers.get('Content-Type'))
            print("--- Start of Stream ---")
            
            # iter_lines() will process the stream as data arrives
            for line in resp.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        json_data = line_str[len('data: '):]
                        message = json.loads(json_data)
                        print(f"CLIENT: Received Message -> {message}")
            
            print("--- End of Stream ---")

# --- Simulation ---

if __name__ == "__main__":
    # Make sure the server is running first!
    client = MCPStreamableClient("http://localhost:5002")
    client.start_session()
    client.run_long_job()
    print("\nCLIENT: Simulation finished.")

```

**How to Run This Example:**

1.  Install dependencies: `pip install Flask requests`
2.  In one terminal, run the server: `python mcp_streamable_server.py`
3.  In a second terminal, run the client: `python mcp_streamable_client.py`

**Expected Output:**

*(Server Terminal)*
```
INFO:werkzeug:127.0.0.1 - - [..] "POST /mcp HTTP/1.1" 200 -
INFO:root:Created new session: ...
INFO:werkzeug:127.0.0.1 - - [..] "POST /mcp HTTP/1.1" 200 -
INFO:root:Upgraded connection to stream for session ...
INFO:root:Sent progress update 20% for session ...
INFO:root:Sent progress update 40% for session ...
INFO:root:Sent progress update 60% for session ...
INFO:root:Sent progress update 80% for session ...
INFO:root:Sent progress update 100% for session ...
INFO.root:Sent final notification for session ...
```

*(Client Terminal)*
```
CLIENT: Attempting to start a new session...
CLIENT: Session started successfully. Token: ...

CLIENT: Requesting to run a long job...
CLIENT: Server responded with status 200. Connection open.
CLIENT: Content-Type:  text/event-stream
--- Start of Stream ---
CLIENT: Received Message -> {"jsonrpc": "2.0", "id": 2, "result": {"status": "Job started. Awaiting progress..."}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/progress", "params": {"job_id": "job-123", "progress": 20}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/progress", "params": {"job_id": "job-123", "progress": 40}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/progress", "params": {"job_id": "job-123", "progress": 60}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/progress", "params": {"job_id": "job-123", "progress": 80}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/progress", "params": {"job_id": "job-123", "progress": 100}}
CLIENT: Received Message -> {"jsonrpc": "2.0", "method": "job/completed", "params": {"job_id": "job-123", "final_status": "Success"}}
--- End of Stream ---

CLIENT: Simulation finished.
```

This implementation clearly shows the power and flexibility of the Streamable HTTP model. The server is clean, with a single entry point. The client can handle both simple and complex interactions, and the use of a session token in the header makes the server-side logic stateless and highly scalable.
