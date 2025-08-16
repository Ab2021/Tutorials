
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 3: Building an MCP Server: The Three Primitives

### Lesson 3.1: Understanding the Server Primitives (09:00 - 10:30)

---

### **1. The Building Blocks of Capability**

Welcome to Module 3. Having established the architecture and transport layers, we now arrive at the heart of MCP: **building a server**. An MCP server is, at its core, a collection of capabilities that it exposes to the world. But what *is* a capability? The MCP specification defines three fundamental building blocks, or **primitives**, that a server can implement. These three primitives are the only types of capabilities a server can offer. They are:

1.  **Prompts:** Reusable, parameterized templates for interacting with an LLM.
2.  **Resources:** Read-only contextual data, identified by a unique URI.
3.  **Tools:** Executable functions that an LLM can call to perform actions.

Mastering the purpose and design of these three primitives is the single most important step in becoming a proficient MCP developer. They provide a clear, structured vocabulary for defining what your server can do. This lesson will provide a deep, conceptual understanding of each primitive, setting the stage for their practical implementation in the lessons to come.

---

### **2. The First Primitive: Prompts**

**Definition:** A Prompt is a **reusable, parameterized template for a user-initiated interaction with an LLM**. Think of them as pre-packaged recipes for common tasks that require AI assistance.

**Core Purpose:** Prompts are designed to streamline and standardize common workflows that involve generating text or receiving guidance from an LLM. They encapsulate the instructions that need to be sent to the model, often combining them with contextual data (Resources).

**Key Characteristics:**

*   **User-Initiated:** The invocation of a Prompt is almost always tied directly to a user action. In a chat interface, this is often a "slash command" (e.g., `/commit`, `/explain`, `/translate`). In a GUI, it might be a button or a context menu item (e.g., "Summarize this document").
*   **Parameterized:** Prompts are not just static strings. They can define arguments that need to be provided at runtime. This allows them to be flexible and adaptable to different contexts.
*   **LLM-Focused:** The ultimate output of a Prompt handler is a `GetPromptResult`, which is a structured message designed to be sent to an LLM. The Prompt primitive is the server's way of telling the client, "Here is a task that I think an LLM would be good at. Here are the instructions and the data it needs."

**A Concrete Example: The `/git-commit-message` Prompt**

Let's consider a Git MCP server. One of its most valuable capabilities would be to help a developer write a good commit message. This is a perfect use case for a Prompt.

*   **Name:** `git/commit-message`
*   **Description:** "Generates a conventional commit message based on the currently staged changes."
*   **How it works conceptually:**
    1.  The user, in their IDE (the Host), types `/commit`.
    2.  The IDE's Client sees this and knows it corresponds to the `git/commit-message` prompt.
    3.  The Client calls the `prompts/get` method on the Git MCP Server.
    4.  The Server's handler for this prompt first needs the context: the staged changes. It fetches the `git:/diff/staged` resource.
    5.  The handler then constructs a `GetPromptResult`. This result contains a list of `PromptMessage` objects. It might look something like this:
        *   **System Message:** "You are an expert programmer who writes conventional commit messages. The user wants a commit message for the following changes."
        *   **User Message:** (The content of the `git diff --staged` output is inserted here).
    6.  The Client receives this `GetPromptResult` and sends it to the LLM.
    7.  The LLM generates the commit message, which the Client then displays to the user.

**Why use a Prompt instead of just having the client hard-code the prompt string?**

*   **Encapsulation:** The logic for how to best prompt the LLM for a commit message is encapsulated within the server that specializes in Git. The server's author is the expert on this task.
*   **Reusability:** Any MCP client that connects to this Git server can now offer a `/commit` command without having to implement any of the logic themselves.
*   **Evolvability:** If a better way to prompt the LLM for commit messages is discovered, the server author can update the prompt handler, and every client that uses it will instantly get the benefit of the improvement without any changes to their own code.

---

### **3. The Second Primitive: Resources**

**Definition:** A Resource is **read-only contextual data, identified by a unique URI**. Resources are the nouns of MCP; they represent the information or data that a server can provide to a client.

**Core Purpose:** Resources are the primary mechanism for providing context to the LLM. They are the data that Prompts and Tools operate on. They allow the server to expose relevant information from its environment in a standardized way.

**Key Characteristics:**

*   **Read-Only:** From the client's perspective, resources are immutable. A client can `get` a resource, but it cannot directly `set` or `update` a resource. Any changes to a resource are managed by the server itself, which can then notify the client of the update.
*   **URI-Identified:** Every resource has a unique Uniform Resource Identifier (URI). The URI scheme (the part before the `://`) is defined by the server, creating a namespace for its resources.
*   **MIME Type:** Every resource has a MIME type (e.g., `text/plain`, `application/json`, `text/x-python`) that tells the client how to interpret its content.

**Concrete Examples:**

*   **File System Server:**
    *   `file:///path/to/my/code.py`: The content of a specific file.
    *   `file:///path/to/my/project`: A listing of the files in a directory.
*   **Git Server:**
    *   `git:/diff/staged`: The output of `git diff --staged`.
    *   `git:/log/recent`: The last 10 commit messages.
*   **Database Server:**
    *   `postgres://mydb/users/123`: A JSON representation of the user with ID 123.
    *   `postgres://mydb/products?category=electronics`: A list of all products in the electronics category.
*   **Desktop Server:**
    *   `screen://localhost/display1`: A PNG image of the current content of the main display.

**The `resources/get` and `resources/list` Methods:**

The client interacts with resources primarily through two methods:

*   `resources/list`: This asks the server, "What resources are currently available?" The server might respond with a list of open files, active database connections, etc.
*   `resources/get`: This asks the server for the content of a specific resource, identified by its URI.

**The Power of the Resource Primitive:**

Resources decouple the *need* for information from the *implementation* of how to get it. The client (and by extension, the LLM) doesn't need to know how to run `git diff` or query a PostgreSQL database. It only needs to know how to ask for the `git:/diff/staged` or `postgres://mydb/users/123` resource. All the complexity is hidden behind the server's implementation of the resource handler.

---

### **4. The Third Primitive: Tools**

**Definition:** A Tool is an **executable function that an LLM can call to perform an action**. Tools are the verbs of MCP; they are the capabilities that allow the AI to affect change in the world.

**Core Purpose:** Tools are the bridge from the AI's knowledge and reasoning to concrete, real-world actions. They are the mechanism by which an LLM can go beyond simply answering questions and start *doing* things.

**Key Characteristics:**

*   **Action-Oriented:** Tools perform actions: creating a file, sending an email, deploying a service, updating a database record, etc.
*   **Schema-Defined:** This is the most critical aspect of a Tool. Every tool **MUST** have a well-defined `inputSchema` (and optionally an `outputSchema`). This schema, typically a JSON Schema, is the contract that tells the LLM exactly how to use the tool. It defines the parameters the tool expects, their types, and which ones are required.
*   **Model-Controlled (but User-Approved):** The decision to call a tool is often made by the LLM itself. Based on the user's request and the list of available tools, the LLM will reason that it needs to call a specific tool to fulfill the request. However, for any tool that performs a significant action, the MCP client should always get final approval from the user before executing it.

**A Concrete Example: The `github/create_issue` Tool**

Imagine an MCP server that integrates with GitHub.

*   **Name:** `github/create_issue`
*   **Description:** "Creates a new issue in a GitHub repository."
*   **Input Schema:**
    ```json
    {
      "type": "object",
      "properties": {
        "repository": {"type": "string", "description": "The owner/repo name, e.g., 'my-org/my-project'"},
        "title": {"type": "string", "description": "The title of the issue."},
        "body": {"type": "string", "description": "The markdown content of the issue."}
      },
      "required": ["repository", "title"]
    }
    ```
*   **How it works conceptually:**
    1.  The user says to their AI assistant, "Create an issue in the `my-org/my-project` repo about the login button being broken."
    2.  The Client sends this user query to the LLM, along with the list of available tools, including the schema for `github/create_issue`.
    3.  The LLM analyzes the user's request and the tool schema. It recognizes that it can fulfill the request by calling this tool. It extracts the necessary parameters from the user's natural language.
    4.  The LLM responds to the Client not with an answer, but with a request to call the tool:
        ```json
        {
          "tool_call": {
            "name": "github/create_issue",
            "arguments": {
              "repository": "my-org/my-project",
              "title": "Login button is broken",
              "body": "When a user clicks the login button, nothing happens."
            }
          }
        }
        ```
    5.  The Client receives this tool call request. It presents it to the user for approval: "The AI wants to create a GitHub issue. [Show Details] [Approve] [Deny]".
    6.  Upon user approval, the Client sends the `tools/call` request to the GitHub MCP Server.
    7.  The Server's handler for the tool receives the arguments, makes the actual API call to GitHub, and returns the result (e.g., the URL of the new issue).

**The Schema is Everything:**

Without the `inputSchema`, the LLM would be guessing how to use the tool. The schema is the instruction manual that the AI reads to understand the tool's function and its required inputs. A well-designed schema is the key to creating reliable and predictable tools.

By understanding these three primitives—Prompts for user-led LLM interaction, Resources for providing read-only context, and Tools for enabling AI-led action—you now have the complete conceptual vocabulary needed to design and build powerful, capable, and well-structured MCP servers.
