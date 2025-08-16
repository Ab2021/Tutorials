
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 3: Building an MCP Server: The Three Primitives

### Lesson 3.2: The Control Level Hierarchy (10:30 - 12:00)

---

### **1. A Framework for Interaction: Who is in Control?**

We have learned *what* the three server primitives are: Prompts, Resources, and Tools. Now, we need to understand *how* and *when* they are used. While all three primitives can be invoked by a client at any time, they are designed to fit into a natural hierarchy of control. This hierarchy provides a powerful mental model for designing and understanding the flow of interaction in any MCP application.

The central question is: **Who initiates the action?** Is it the human user, the host application, or the AI model itself? The answer to this question determines which primitive is most appropriate for the task at hand.

The Control Level Hierarchy categorizes the primitives as follows:

1.  **User-Controlled:** Actions initiated directly and explicitly by the user. This is the domain of **Prompts**.
2.  **Application-Controlled:** Context provided automatically by the host application based on the user's environment. This is the domain of **Resources**.
3.  **Model-Controlled:** Actions initiated by the LLM as part of its reasoning process to fulfill a user's request. This is the domain of **Tools**.

Understanding this hierarchy is not just an academic exercise. It is a practical guide to designing clean, predictable, and user-friendly AI systems. When the lines of control are clear, applications are easier to build, debug, and use.

**(ASCII Art Diagram of the Control Level Hierarchy)**

```
+--------------------------------------------------------------------+
|                            User Input                              |
| (e.g., "/commit", "Translate this text", button click)             |
+--------------------------------------------------------------------+
                               | (Explicit Command)
                               v
+--------------------------------------------------------------------+
|                          USER-CONTROLLED                           |
|--------------------------------------------------------------------|
| Primitive:      PROMPTS                                            |
| Purpose:        Pre-packaged, user-facing commands for the LLM.    |
| Example:        A `/commit` slash command that triggers the        |
|                 `git/commit-message` prompt.                       |
+--------------------------------------------------------------------+
                               | (Needs Context)
                               v
+--------------------------------------------------------------------+
|                       APPLICATION-CONTROLLED                       |
|--------------------------------------------------------------------|
| Primitive:      RESOURCES                                          |
| Purpose:        Provide automatic, read-only context to the system.|
| Example:        The application automatically makes the content of |
|                 the open file available as `file:///path/to/file`.   |
+--------------------------------------------------------------------+
                               | (Used by the Model)
                               v
+--------------------------------------------------------------------+
|                          MODEL-CONTROLLED                          |
|--------------------------------------------------------------------|
| Primitive:      TOOLS                                              |
| Purpose:        Action-taking functions for the LLM to call.       |
| Example:        The LLM decides to call `github/create_issue` to   |
|                 fulfill the user's request.                        |
+--------------------------------------------------------------------+
```

---

### **2. The Top Level: User-Controlled Primitives (Prompts)**

**The Principle:** The user is the ultimate authority. When a user wants to explicitly command the AI to perform a specific, pre-defined task, they are operating at the user-controlled level.

**The Primitive:** **Prompts** are the natural fit for this level of control.

*   **Why Prompts?** A prompt represents a complete, user-facing command. It has a name and a description that can be shown in a UI (like a list of slash commands). When the user selects a prompt, they are making a conscious, explicit choice to initiate a specific workflow.

**The Flow of Control:**

1.  **User Action:** The user performs a direct action. This is the trigger.
    *   Typing `/explain` in a chat window.
    *   Clicking a "Summarize Selection" button.
    *   Selecting a "Refactor Code" option from a context menu.
2.  **Client Invokes Prompt:** The Host application maps this user action to a specific Prompt and calls the `prompts/get` method on the appropriate MCP server.
3.  **Server Gathers Context:** The prompt handler on the server may, in turn, need context. It gets this context by accessing **Resources** (the next level down in the hierarchy).
4.  **Client Executes Prompt:** The client receives the `GetPromptResult` from the server and sends it to the LLM.

**Example: The `/translate` command**

*   **User Action:** The user highlights a block of text in their editor and types `/translate to Spanish`.
*   **Control Level:** This is clearly a user-controlled action. The user has given a direct and unambiguous command.
*   **Primitive Choice:** The server should expose a `translator/translate` **Prompt**.
*   **Implementation:**
    *   The prompt would take one argument: `target_language` (a string).
    *   It would also need the text to be translated. This would be provided by the Host application as a contextual **Resource** (e.g., `editor:/selection`).
    *   The prompt's handler would construct the necessary messages for the LLM: a system message like "You are a helpful translation assistant" and a user message containing the text to be translated.

By framing this as a Prompt, we align the implementation with the user's mental model. The user feels in direct control, issuing a command and getting a result.

---

### **3. The Middle Level: Application-Controlled Primitives (Resources)**

**The Principle:** Powerful AI interactions require rich context. The Host application is best positioned to provide this context automatically and seamlessly, without requiring explicit user action for every piece of information.

**The Primitive:** **Resources** are the ideal primitive for this level of control.

*   **Why Resources?** Resources are read-only snapshots of the application's state. The application can decide which resources are relevant at any given moment and make them available to the system. The user doesn't need to say, "Here is the file I'm working on." The application already knows and can expose it as a resource.

**The Flow of Control:**

1.  **Application State Changes:** The user's interaction with the Host application changes its state.
    *   The user opens a new file in an IDE.
    *   The user navigates to a different page in a web app.
    *   The user connects to a new database.
2.  **Server Updates Resources:** The MCP server, which is often monitoring the application's state, recognizes this change. It can then do two things:
    *   Make a new resource available (e.g., a `file:///` URI for the newly opened file).
    *   Send a `resources/updated` notification to the client to let it know that the set of available resources has changed.
3.  **Context for Other Primitives:** This automatically-provided context is now available to be used by user-controlled Prompts and model-controlled Tools.

**Example: The Open File in an IDE**

*   **Application Action:** A developer opens the file `user_service.py` in their IDE.
*   **Control Level:** The application is in control of managing its state. The user isn't issuing an AI command; they are just performing a normal action within the Host application.
*   **Primitive Choice:** The IDE's file system MCP server should expose this as a **Resource**.
*   **Implementation:**
    *   The server now makes the resource `file:///path/to/user_service.py` available.
    *   It might send a notification to the client, effectively saying, "FYI, the user is now looking at this file."
    *   Later, when the user invokes a `/explain-code` **Prompt**, that prompt's handler can access the `file:///path/to/user_service.py` resource to get the content it needs to explain. The user doesn't have to specify the file; the application provides it as context.

This middle layer is the glue that connects the user's explicit commands with the data they operate on. It makes the interaction feel seamless and intelligent.

---

### **4. The Bottom Level: Model-Controlled Primitives (Tools)**

**The Principle:** To fulfill a user's complex or ambiguous request, the LLM needs the autonomy to decide on a course of action and execute it. This often involves breaking down a request into a series of steps and using external capabilities to accomplish those steps.

**The Primitive:** **Tools** are the only primitive designed for this level of control.

*   **Why Tools?** The defining feature of a Tool is its **JSON Schema**. This schema is not for humans; it is the instruction manual for the LLM. It allows the model to reason about the function's purpose, its required inputs, and how to provide them. This is what enables the model to take control of the execution flow.

**The Flow of Control:**

1.  **User Request:** The user gives a high-level, often conversational, request.
    *   "There's a bug with the login page. Create a ticket for it."
    *   "Deploy the latest version of the web app."
    *   "Find the email address for our contact at Acme Corp."
2.  **LLM Reasoning:** The Client sends this request to the LLM, along with the list of available **Tools** and their schemas.
3.  **Model Decides to Act:** The LLM analyzes the request and the available tools. It determines that it cannot answer the request directly and must use one or more tools. It formulates a plan.
4.  **Model Invokes Tool:** The LLM responds to the client with a `tool_call` object, specifying the name of the tool and the arguments it has extracted from the user's query.
5.  **Client (and User) Approval:** The client receives the `tool_call`, validates it, and, for any potentially impactful action, asks the user for final approval.
6.  **Client Executes Tool:** Upon approval, the client sends the `tools/call` request to the MCP server.

**Example: The Vague Bug Report**

*   **User Request:** "Ugh, the login is broken again. Can you open a ticket for it in the main repo?"
*   **Control Level:** The user's intent is clear, but the parameters are ambiguous. This is a perfect scenario for the model to take control.
*   **Primitive Choice:** The server should expose a `github/create_issue` **Tool**.
*   **Implementation:**
    *   The LLM receives the user's query and the schema for the `github/create_issue` tool.
    *   It sees that the tool requires a `repository` and a `title`.
    *   It can infer the `title` ("Login is broken"), but it might not know what "the main repo" is. 
    *   Here, the model might even decide to call *another* tool first, like a `project/get_main_repo` tool, to find the repository name.
    *   Once it has all the parameters, it constructs the `tool_call` and returns it.

This hierarchy—User, Application, Model—provides a clear and powerful framework. It encourages developers to design servers that map cleanly to the expected flow of control, resulting in AI applications that are not only powerful but also intuitive and safe to use.
