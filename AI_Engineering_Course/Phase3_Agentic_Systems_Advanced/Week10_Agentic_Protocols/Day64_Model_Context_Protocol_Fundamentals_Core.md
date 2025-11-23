# Day 64: Model Context Protocol (MCP) Fundamentals
## Core Concepts & Theory

### The Fragmentation Problem

In the pre-MCP world, connecting an LLM to a data source (e.g., Google Drive, Slack, PostgreSQL) required writing a custom "Tool" or "Plugin" for *every* specific agent framework.
*   LangChain has `GoogleDriveTool`.
*   LlamaIndex has `GoogleDriveReader`.
*   AutoGPT has its own.
*   Claude has its own.

This is an **M x N** problem. M frameworks * N data sources = Fragmentation.

### What is MCP?

The **Model Context Protocol (MCP)** is an open standard (open-sourced by Anthropic in 2024) that solves this by standardizing the connection between **AI Models** (Clients) and **Data/Tools** (Servers).
It works like USB-C for AI. You write the "Google Drive MCP Server" *once*, and it works with Claude, LangChain, LlamaIndex, or any other MCP-compliant client.

### Core Architecture

MCP follows a **Client-Host-Server** architecture:

1.  **MCP Host:** The application the user interacts with (e.g., Claude Desktop, an IDE, a custom Agent app). The Host runs the Client.
2.  **MCP Client:** The protocol implementation inside the Host. It maintains 1:1 connections with Servers.
3.  **MCP Server:** A lightweight process that exposes three things:
    *   **Resources:** Passive data (files, logs, database rows) that can be read. Like `GET` requests.
    *   **Prompts:** Pre-defined templates for interacting with the data.
    *   **Tools:** Executable functions (API calls, write operations). Like `POST` requests.

### Key Primitives

#### 1. Resources (Context)
Resources are identified by a URI (e.g., `postgres://db/users`).
*   **Direct Read:** The LLM can say "Read `file:///logs/error.txt`".
*   **Subscription:** The LLM can subscribe to updates on a resource (e.g., "Tell me when the log changes").

#### 2. Tools (Action)
Tools are function definitions (JSON Schema).
*   `execute_sql(query: str)`
*   `send_slack_message(channel: str, text: str)`
The Server defines the tool; the Client (LLM) calls it; the Server executes it and returns the result.

#### 3. Prompts (Guidance)
Servers can define "Slash Commands".
*   A Git MCP Server might define a prompt `git-commit` that automatically gathers the diff, runs `git status`, and asks the LLM to generate a commit message.

### Transport Layer

MCP is transport-agnostic but typically runs over:
*   **Stdio:** The Client spawns the Server as a subprocess and talks via `stdin`/`stdout` (JSON-RPC). This is secure (local only) and fast.
*   **SSE (Server-Sent Events):** For remote servers over HTTP.

### Why this matters for AI Engineers

1.  **Write Once, Run Anywhere:** Build a connector for your internal API once, and every agent in your company can use it.
2.  **Security:** The Host controls the connection. You don't paste API keys into the LLM; the MCP Server holds the keys locally.
3.  **Context Window Management:** MCP allows the server to "sample" content (send only the first 100 lines) to save tokens.

### Summary

MCP moves us from "Custom Integrations" to "Standardized Drivers". Just as your OS doesn't need to know how a specific mouse works (it just speaks "Mouse Protocol"), your Agent shouldn't need to know how to talk to Jira specifically—it should just speak "MCP".
