# The Definitive Guide to the Model Context Protocol (MCP)

## Module 1: Foundations of MCP

### Lesson 1.3: The Language of MCP: JSON-RPC 2.0 (13:00 - 14:30)

### **Implementation & Examples**

---

### **1. Working with JSON-RPC in Python**

This implementation guide will walk through the practical steps of creating, serializing, deserializing, and handling JSON-RPC 2.0 messages using Python. We will not use an external library for the core message structures to ensure a clear understanding of the underlying format. We will simulate the interaction between an MCP Client and an MCP Server.

**Scenario:** Our client will ask a server to perform two actions:
1.  List available tools (a request with no parameters).
2.  Call a specific tool to concatenate two strings (a request with parameters).

We will also show how the server can send a notification.

---

### **2. The Client-Side Implementation: Crafting and Sending Requests**

Our simulated MCP client will be responsible for creating valid Request objects and printing them as JSON strings.

```python
import json
import uuid

class MCPClientSimulator:
    """A class to simulate the creation of MCP client-side messages."""

    def _create_request_object(self, method: str, params: dict | list | None = None) -> dict:
        """A helper method to construct a standard JSON-RPC Request."""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params if params is not None else {},
            "id": str(uuid.uuid4())  # Always use a unique ID for each request
        }

    def serialize_request(self, request_obj: dict) -> str:
        """Converts the request dictionary to a JSON string."""
        # In a real stdio transport, we would add a newline
        return json.dumps(request_obj)

    def create_list_tools_request(self) -> dict:
        """Creates a request to list all available tools."""
        print("CLIENT: Creating a 'tools/list' request...")
        return self._create_request_object("tools/list")

    def create_call_tool_request(self, tool_name: str, arguments: dict) -> dict:
        """Creates a request to call a specific tool."""
        print(f"CLIENT: Creating a 'tools/call' request for tool '{tool_name}'...")
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        return self._create_request_object("tools/call", params)

# --- Simulation ---

if __name__ == "__main__":
    client = MCPClientSimulator()

    # 1. Create and serialize a request to list tools
    list_tools_req = client.create_list_tools_request()
    serialized_list_tools = client.serialize_request(list_tools_req)
    print(f"CLIENT: Serialized Request -> {serialized_list_tools}\n")
    # In a real application, this string would be sent to the server's stdin.

    # 2. Create and serialize a request to call the "string/concat" tool
    concat_tool_req = client.create_call_tool_request(
        tool_name="string/concat",
        arguments={"str1": "Hello, ", "str2": "MCP!"}
    )
    serialized_concat_tool = client.serialize_request(concat_tool_req)
    print(f"CLIENT: Serialized Request -> {serialized_concat_tool}\n")
    # This string would also be sent to the server.

```

**Expected Output of the Client Simulation:**

```
CLIENT: Creating a 'tools/list' request...
CLIENT: Serialized Request -> {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "...some-uuid..."}

CLIENT: Creating a 'tools/call' request for tool 'string/concat'...
CLIENT: Serialized Request -> {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "string/concat", "arguments": {"str1": "Hello, ", "str2": "MCP!"}}, "id": "...another-uuid..."}
```

**Key Implementation Points:**

*   **Unique IDs:** We use the `uuid` library to generate a unique `id` for every request. This is crucial for the client to correctly map responses back to their original requests, especially in an asynchronous environment.
*   **Serialization:** `json.dumps()` is the standard Python way to serialize a dictionary into a JSON string. This is the exact data that would travel over the wire (or through the `stdio` pipe).
*   **Structured `params`:** For the `tools/call` method, the `params` object is itself structured, containing the `name` of the tool and its `arguments`. This is a common pattern in MCP.

---

### **3. The Server-Side Implementation: Parsing and Handling Messages**

Our simulated MCP server will receive these JSON strings, parse them, and route them to the correct handler function.

```python
import json

class MCPServerSimulator:
    """A class to simulate the handling of MCP server-side messages."""

    def __init__(self):
        # In a real server, these would be defined more formally with schemas.
        self._tool_handlers = {
            "string/concat": self._handle_string_concat
        }
        self._available_tools = [
            {
                "name": "string/concat",
                "description": "Concatenates two strings.",
                "input_schema": { "type": "object", "properties": { "str1": {"type": "string"}, "str2": {"type": "string"} } }
            }
        ]

    def _create_success_response(self, request_id: str, result: dict) -> dict:
        """Constructs a standard JSON-RPC Success Response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def _create_error_response(self, request_id: str, code: int, message: str, data: any = None) -> dict:
        """Constructs a standard JSON-RPC Error Response."""
        error_obj = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_obj
        }

    def _create_notification(self, method: str, params: dict) -> dict:
        """Constructs a JSON-RPC Notification."""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

    def handle_message(self, message_str: str):
        """The main entry point for processing an incoming message."""
        print(f"SERVER: Received Message -> {message_str}")
        try:
            message_obj = json.loads(message_str)
        except json.JSONDecodeError:
            response = self._create_error_response(None, -32700, "Parse error")
            print(f"SERVER: Sending Response -> {json.dumps(response)}\n")
            return

        # Check if it's a notification (no id)
        if "id" not in message_obj:
            print("SERVER: Received a notification. No response will be sent.")
            return

        request_id = message_obj["id"]
        method = message_obj.get("method")

        if method == "tools/list":
            result = {"tools": self._available_tools}
            response = self._create_success_response(request_id, result)
        elif method == "tools/call":
            tool_name = message_obj.get("params", {}).get("name")
            if tool_name in self._tool_handlers:
                arguments = message_obj.get("params", {}).get("arguments", {})
                response = self._tool_handlers[tool_name](request_id, arguments)
            else:
                response = self._create_error_response(request_id, -32601, "Method not found", f"Tool '{tool_name}' is not defined.")
        else:
            response = self._create_error_response(request_id, -32601, "Method not found")
        
        print(f"SERVER: Sending Response -> {json.dumps(response)}\n")

    def _handle_string_concat(self, request_id: str, arguments: dict) -> dict:
        """The actual implementation of the 'string/concat' tool."""
        str1 = arguments.get("str1")
        str2 = arguments.get("str2")

        if str1 is None or str2 is None:
            return self._create_error_response(request_id, -32602, "Invalid Params", "Both 'str1' and 'str2' are required.")

        concatenated_string = str1 + str2
        result = {"concatenated": concatenated_string}
        return self._create_success_response(request_id, result)

# --- Simulation ---

if __name__ == "__main__":
    server = MCPServerSimulator()

    # Simulate receiving the two requests from the client
    # (using the serialized strings from the client example)
    list_tools_req_str = '{"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "a1b2c3d4"}'
    concat_tool_req_str = '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "string/concat", "arguments": {"str1": "Hello, ", "str2": "MCP!"}}, "id": "e5f6g7h8"}'
    invalid_tool_req_str = '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "nonexistent/tool"}, "id": "i9j0k1l2"}'

    server.handle_message(list_tools_req_str)
    server.handle_message(concat_tool_req_str)
    server.handle_message(invalid_tool_req_str)

    # Simulate the server sending a notification
    notification = server._create_notification("server/log", {"level": "info", "message": "Server is healthy"})
    print(f"SERVER: Sending Notification -> {json.dumps(notification)}")

```

**Expected Output of the Server Simulation:**

```
SERVER: Received Message -> {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": "a1b2c3d4"}
SERVER: Sending Response -> {"jsonrpc": "2.0", "id": "a1b2c3d4", "result": {"tools": [...tool definition...]}}

SERVER: Received Message -> {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "string/concat", ...}}
SERVER: Sending Response -> {"jsonrpc": "2.0", "id": "e5f6g7h8", "result": {"concatenated": "Hello, MCP!"}}

SERVER: Received Message -> {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "nonexistent/tool"}, "id": "i9j0k1l2"}
SERVER: Sending Response -> {"jsonrpc": "2.0", "id": "i9j0k1l2", "error": {"code": -32601, "message": "Method not found", "data": "Tool 'nonexistent/tool' is not defined."}}

SERVER: Sending Notification -> {"jsonrpc": "2.0", "method": "server/log", "params": {"level": "info", "message": "Server is healthy"}}
```

**Key Implementation Points:**

*   **Deserialization:** `json.loads()` is used to parse the incoming JSON string into a Python dictionary.
*   **Robust Parsing:** The first step in the handler is a `try...except` block to catch `json.JSONDecodeError`. This is how a server correctly implements the `-32700 Parse error`.
*   **Routing:** The `handle_message` function acts as a router, inspecting the `method` field to decide which internal function should handle the request.
*   **Error Handling:** The server code explicitly checks for missing parameters (`str1`, `str2`) and returns a well-formed `-32602 Invalid Params` error. It also handles the case of a non-existent tool with a `-32601 Method not found` error.
*   **Notifications:** The server can proactively create and send messages without an `id`. The client-side logic would need a listener to handle these asynchronous, one-way messages.

This Python simulation provides a clear, hands-on demonstration of how the JSON-RPC 2.0 protocol works in practice and serves as a solid foundation for understanding the more complex interactions in the modules to come.
