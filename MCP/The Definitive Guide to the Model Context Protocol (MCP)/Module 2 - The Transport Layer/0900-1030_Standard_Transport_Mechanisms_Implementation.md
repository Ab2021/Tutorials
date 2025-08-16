# The Definitive Guide to the Model Context Protocol (MCP)

## Module 2: The Transport Layer: How MCP Communicates

### Lesson 2.1: Standard Transport Mechanisms (09:00 - 10:30)

### **Implementation & Examples**

---

### **1. `stdio` Transport in Practice: A Python Implementation**

This section provides a complete, runnable example of an MCP client and server communicating over standard I/O (`stdio`). The client will launch the server as a subprocess and they will exchange JSON-RPC messages to perform a simple task.

**The Scenario:** The client will send a request to a `math/add` tool. The server will perform the addition and return the result.

#### **The MCP Server (`mcp_stdio_server.py`)**

This script will be the MCP server. When run, it will listen for JSON-RPC messages on its `stdin`, process them, and write responses to its `stdout`.

```python
# mcp_stdio_server.py

import sys
import json
import logging

# Configure logging to stderr to keep stdout clean for JSON-RPC responses.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def handle_math_add(request_id, params):
    """Handler for the 'math/add' tool."""
    logging.info(f"Handling 'math/add' with params: {params}")
    if not isinstance(params, list) or len(params) != 2:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "Invalid Params", "data": "Expected an array of two numbers."}
        }
    
    result = params[0] + params[1]
    
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"sum": result}
    }

def main():
    logging.info("MCP Stdio Server Started. Listening on stdin...")
    
    # The main server loop
    for line in sys.stdin:
        try:
            request = json.loads(line)
            logging.info(f"Received request: {request}")
            
            method = request.get("method")
            request_id = request.get("id")

            if method == "tools/call":
                tool_name = request.get("params", {}).get("name")
                if tool_name == "math/add":
                    arguments = request.get("params", {}).get("arguments", [])
                    response = handle_math_add(request_id, arguments)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"}
                    }
            else:
                 response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}
                }

            # Write the JSON response to stdout, followed by a newline.
            # This is the message framing.
            sys.stdout.write(json.dumps(response) + '\n')
            sys.stdout.flush() # Ensure the message is sent immediately

        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
            sys.stdout.write(json.dumps(error_response) + '\n')
            sys.stdout.flush()
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": "Internal error", "data": str(e)}
            }
            sys.stdout.write(json.dumps(error_response) + '\n')
            sys.stdout.flush()

if __name__ == "__main__":
    main()

```

#### **The MCP Client (`mcp_stdio_client.py`)**

This script will act as the Host/Client. It will use Python's `subprocess` module to launch the server, write to its `stdin`, and read from its `stdout`.

```python
# mcp_stdio_client.py

import subprocess
import json
import sys
import threading

class MCPStdioClient:
    def __init__(self, server_script_path):
        self.server_script_path = server_script_path
        self.process = None
        self.request_id_counter = 0

    def start(self):
        """Starts the MCP server as a subprocess."""
        print(f"CLIENT: Starting server: {self.server_script_path}")
        self.process = subprocess.Popen(
            [sys.executable, self.server_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Work with text streams (automatic encoding/decoding)
            bufsize=1   # Line-buffered
        )
        
        # It's good practice to monitor stderr in a separate thread
        threading.Thread(target=self._log_stderr, daemon=True).start()
        print("CLIENT: Server started successfully.")

    def _log_stderr(self):
        """Reads from the server's stderr and logs it."""
        for line in self.process.stderr:
            print(f"SERVER_STDERR: {line.strip()}")

    def stop(self):
        """Stops the server process."""
        if self.process:
            print("CLIENT: Stopping server...")
            self.process.terminate() # or .kill()
            self.process.wait()
            print("CLIENT: Server stopped.")

    def send_request(self, method, params):
        """Sends a JSON-RPC request to the server and gets the response."""
        self.request_id_counter += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id_counter,
            "method": method,
            "params": params
        }
        
        request_str = json.dumps(request)
        print(f"CLIENT: Sending -> {request_str}")
        
        # Write the request to the server's stdin
        self.process.stdin.write(request_str + '\n')
        self.process.stdin.flush()
        
        # Read the response from the server's stdout
        response_str = self.process.stdout.readline()
        print(f"CLIENT: Received <- {response_str.strip()}")
        
        return json.loads(response_str)

# --- Simulation ---

if __name__ == "__main__":
    client = MCPStdioClient("mcp_stdio_server.py")
    try:
        client.start()
        
        # Call the math/add tool
        add_params = {
            "name": "math/add",
            "arguments": [15, 27]
        }
        response = client.send_request("tools/call", add_params)
        
        print(f"\nCLIENT: The result of the addition is: {response.get('result', {}).get('sum')}")

    finally:
        client.stop()

```

**How to Run This Example:**

1.  Save the two scripts above as `mcp_stdio_server.py` and `mcp_stdio_client.py` in the same directory.
2.  Open your terminal and run the client: `python mcp_stdio_client.py`

**Expected Output:**

```
CLIENT: Starting server: mcp_stdio_server.py
CLIENT: Server started successfully.
CLIENT: Sending -> {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "math/add", "arguments": [15, 27]}}
SERVER_STDERR: MCP Stdio Server Started. Listening on stdin...
SERVER_STDERR: Received request: {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "math/add", "arguments": [15, 27]}}
SERVER_STDERR: Handling 'math/add' with params: [15, 27]
CLIENT: Received <- {"jsonrpc": "2.0", "id": 1, "result": {"sum": 42}}

CLIENT: The result of the addition is: 42
CLIENT: Stopping server...
CLIENT: Server stopped.
```

**Key Implementation Points:**

*   **`subprocess.Popen`:** This is the core of the client. It launches the server and, crucially, sets `stdin`, `stdout`, and `stderr` to `subprocess.PIPE` to gain programmatic access to them.
*   **`text=True`:** This simplifies communication by handling the encoding and decoding of strings automatically.
*   **`bufsize=1`:** This sets the streams to be line-buffered, which is perfect for our newline-delimited message framing.
*   **`sys.stdout.flush()`:** This is important on the server side to ensure that the output buffer is written to the pipe immediately and not held back by the OS.
*   **`stderr` Monitoring:** The client uses a separate thread to monitor the server's `stderr`. This is a robust pattern that prevents the client from deadlocking if the server writes a lot of error data, and it provides excellent visibility for debugging.

---

### **2. SSE Transport in Practice: A Conceptual Python Server**

Implementing a full SSE client is browser-specific, but we can easily show how a Python server using the **Flask** web framework would implement the server-side of the original HTTP + SSE transport model.

**Scenario:** A server that exposes a `/message` endpoint for requests and an `/sse` endpoint for streaming notifications.

```python
# mcp_sse_server.py

from flask import Flask, request, Response, jsonify
import json
import time

app = Flask(__name__)

# 1. The /message endpoint for standard requests
@app.route('/message', methods=['POST'])
def handle_message():
    json_rpc_request = request.json
    # ... process the request as in the stdio server ...
    response = {
        "jsonrpc": "2.0",
        "id": json_rpc_request.get("id"),
        "result": {"status": "request_processed"}
    }
    return jsonify(response)

# 2. The /sse endpoint for streaming notifications
@app.route('/sse')
def sse_stream():
    def event_stream():
        count = 0
        while True:
            # In a real app, this would be triggered by an actual event
            count += 1
            notification = {
                "jsonrpc": "2.0",
                "method": "server/heartbeat",
                "params": {"count": count}
            }
            
            # Format the message in the text/event-stream format
            # data: { ...json... }
            # 
            sse_formatted_message = f"data: {json.dumps(notification)}\n\n"
            yield sse_formatted_message
            time.sleep(5) # Wait 5 seconds before sending the next one

    # The Response object has the special mimetype for SSE
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    # To run: pip install Flask
    # Then: python mcp_sse_server.py
    app.run(port=5001, threaded=True)

```

**How to Test This:**

1.  Run the server: `python mcp_sse_server.py`
2.  Open a browser and navigate to `http://localhost:5001/sse`. You will see a new heartbeat notification appear every 5 seconds.
3.  In a separate terminal, you could use `curl` to interact with the `/message` endpoint:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"any/method"}' http://localhost:5001/message
    ```

This example clearly separates the two communication patterns, illustrating the original design of MCP's HTTP transport and setting the stage for understanding why a more unified, streamable approach was needed.
