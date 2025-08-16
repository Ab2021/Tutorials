
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 3: Building an MCP Server: The Three Primitives

### Lesson 3.1: Understanding the Server Primitives (09:00 - 10:30)

### **Implementation & Examples**

---

### **1. Defining Primitives: From Concept to Code**

This section translates the conceptual understanding of Prompts, Resources, and Tools into concrete data structures and schema definitions. We will use Python dictionaries to represent the definitions of these primitives, as they map directly to the JSON structures that an MCP server would expose.

This is the information a server would return when a client calls the `prompts/list`, `resources/list`, or `tools/list` methods.

---

### **2. Implementing a Prompt Definition**

Let's define the `git/commit-message` prompt we discussed in the previous lesson. The definition includes its name, a human-readable description, and, crucially, the arguments it accepts.

**The Scenario:** A prompt that takes the type of commit (e.g., `feat`, `fix`, `docs`) as an argument and uses the staged git diff as context.

```python
# Conceptual definition of a Prompt in a server's registry.

def get_git_commit_message_prompt_definition():
    """
    Returns the schema for the git/commit-message prompt.
    This is the data the server would send to the client.
    """
    return {
        "name": "git/commit-message",
        "description": "Generates a conventional commit message based on the currently staged changes.",
        
        # The 'arguments' array defines the parameters the prompt handler expects.
        "arguments": [
            {
                "name": "commit_type",
                "description": "The type of the commit (e.g., 'feat', 'fix', 'docs').",
                "schema": { "type": "string" },
                "required": True
            },
            {
                "name": "context_diff",
                "description": "The git diff to be used as context for the commit message.",
                
                # This argument is defined by the resource it should be populated from.
                "source": {
                    "type": "resource",
                    "uri": "git:/diff/staged" # The server tells the client where to get this data
                },
                "required": True
            }
        ]
    }

# --- Example Usage ---
if __name__ == "__main__":
    prompt_def = get_git_commit_message_prompt_definition()
    print("--- Prompt Definition ---")
    import json
    print(json.dumps(prompt_def, indent=2))

    # This definition tells the client everything it needs to know to use the prompt.
    # A client could use this to dynamically build a UI with a text input for 'commit_type'.
    # It also knows it MUST first fetch the 'git:/diff/staged' resource before calling the prompt.

```

**Key Implementation Points:**

*   **`name` and `description`:** These are essential for both machines and humans. The `name` is the unique identifier, and the `description` allows a client to build a helpful UI.
*   **`arguments`:** This is an array of argument definitions.
*   **`schema`:** Each argument has a simple JSON schema to define its expected type.
*   **`source`:** This is a powerful concept. The prompt argument `context_diff` specifies that its value should be populated from an MCP **Resource**. The server is explicitly telling the client: "To use this prompt, you must first call `resources/get` with the URI `git:/diff/staged` and pass the result as the `context_diff` argument." This makes prompts composable with resources.

---

### **3. Implementing a Resource Definition**

Now let's define the `git:/diff/staged` resource itself. The definition is simpler than a prompt, primarily consisting of its URI, name, and MIME type.

```python
# Conceptual definition of a Resource in a server's registry.

def get_staged_diff_resource_definition():
    """
    Returns the schema for the git:/diff/staged resource.
    """
    return {
        "uri": "git:/diff/staged",
        "name": "Staged Git Diff",
        "description": "The output of `git diff --staged`, showing all changes that are ready to be committed.",
        
        # The MIME type is critical for the client to know how to interpret the content.
        "mimetype": "text/plain"
    }

# A second example: a resource that returns structured data.
def get_user_profile_resource_definition():
    """
    Returns the schema for a hypothetical user profile resource.
    """
    return {
        "uri": "user:/profile/current",
        "name": "Current User Profile",
        "description": "A JSON object containing details for the currently logged-in user.",
        "mimetype": "application/json"
    }

# --- Example Usage ---
if __name__ == "__main__":
    diff_res_def = get_staged_diff_resource_definition()
    user_res_def = get_user_profile_resource_definition()
    
    print("--- Resource Definitions ---")
    import json
    print(json.dumps(diff_res_def, indent=2))
    print(json.dumps(user_res_def, indent=2))

    # When the client calls `resources/get` with `git:/diff/staged`,
    # the server would execute `git diff --staged` and return the raw text output.
    
    # When the client calls `resources/get` with `user:/profile/current`,
    # the server would return a JSON string, e.g., '{"name": "Alice", "email": "alice@example.com"}'.

```

**Key Implementation Points:**

*   **`uri`:** The unique identifier. The scheme (`git:`, `user:`) provides a namespace.
*   **`mimetype`:** This is essential. `text/plain` tells the client to treat the content as simple text. `application/json` tells it to parse the content as JSON. A client could use this to render different views (e.g., a plain text view vs. a formatted JSON tree view).
*   **Separation of Definition from Data:** This script defines the *metadata* about the resource. The actual *data* of the resource is only fetched when a client calls the `resources/get` method with the corresponding URI.

---

### **4. Implementing a Tool Definition**

Finally, let's define the `github/create_issue` tool. The most important part of this definition is the `inputSchema`, which provides the detailed contract for how to use the tool.

```python
# Conceptual definition of a Tool in a server's registry.

import json

def get_create_issue_tool_definition():
    """
    Returns the schema for the github/create_issue tool.
    This JSON Schema is the instruction manual for the LLM.
    """
    return {
        "name": "github/create_issue",
        "description": "Creates a new issue in a specified GitHub repository.",
        
        # The inputSchema is a formal JSON Schema object.
        "inputSchema": {
            "type": "object",
            "properties": {
                "repository": {
                    "type": "string",
                    "description": "The repository to create the issue in, formatted as 'owner/repo'."
                },
                "title": {
                    "type": "string",
                    "description": "The title of the new issue."
                },
                "body": {
                    "type": "string",
                    "description": "Optional. The markdown content for the body of the issue."
                },
                "labels": {
                    "type": "array",
                    "description": "Optional. A list of labels to add to the issue.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            # The 'required' array tells the LLM which parameters are mandatory.
            "required": ["repository", "title"]
        },
        
        # An optional outputSchema can be defined to describe the tool's return value.
        "outputSchema": {
            "type": "object",
            "properties": {
                "issue_url": {
                    "type": "string",
                    "description": "The URL of the newly created issue."
                },
                "issue_number": {
                    "type": "integer",
                    "description": "The number of the newly created issue."
                }
            }
        }
    }

# --- Example Usage ---
if __name__ == "__main__":
    tool_def = get_create_issue_tool_definition()
    print("--- Tool Definition ---")
    print(json.dumps(tool_def, indent=2))

    # The LLM receives this schema along with the user's query.
    # Based on the properties and the required fields, it can construct
    # a valid `arguments` object for the `tools/call` method.
    # For example, for the query "Create a bug report in my-org/my-repo about the login",
    # the LLM would generate:
    # {
    #   "repository": "my-org/my-repo",
    #   "title": "Bug report about the login"
    # }

```

**Key Implementation Points:**

*   **`inputSchema`:** This is the most critical part. It's a formal [JSON Schema](https://json-schema.org/) that provides a machine-readable contract for the tool's inputs. The richness of the descriptions within the schema is vital for helping the LLM understand the purpose of each parameter.
*   **`required`:** This array is essential for validation and for the LLM to know which arguments it absolutely must provide.
*   **`outputSchema`:** While optional, defining an `outputSchema` is a best practice. It makes the tool's output predictable and verifiable. A client or another tool can rely on the structure of the data returned by this tool.

These Python examples illustrate the concrete data structures that represent the three MCP primitives. A real MCP server would have a registry or a list of these definitions, which it would serialize to JSON and send to a client in response to the `*/list` discovery methods. This metadata is the foundation of the dynamic, interoperable ecosystem that MCP enables.
