# MCP Host CLI

A FastAPI-based CLI application that hosts and manages MCP (Model Context Protocol) servers, providing an HTTP API for interacting with tools and resources.

## Features

- Manages multiple MCP server connections
- Provides HTTP API endpoints for:
  - User requests
  - Tool approval workflow
  - Session state management
- Supports both direct LLM API and Ollama local models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/VyacheslavVanin/mcp-host-cli.git
cd mcp-host-cli
```

2. Run:

```bash
uv run main.py
```

## Configuration

### Server Configuration

1. Create/edit `servers_config.json` to configure your MCP servers:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "node",
      "args": ["path/to/server.js"],
      "env": {
        "API_KEY": "your-api-key"
      }
    }
  }
}
```

### Application Configuration

Configuration can be set via environment variables or command line arguments (CLI args take precedence).

#### Environment Variables

- `LLM_API_KEY`: API key for LLM service (if not using Ollama)
- `LLM_PROVIDER`: "ollama" (default) or "openai"
- `LLM_MODEL`: Model name (default: "qwen2.5-coder:latest")
- `PORT`: Server port (default: 8000)
- `OPENAI_BASE_URL`: Base URL for OpenAI-compatible API (default: "<https://openrouter.ai/api/v1>")
- `USE_OLLAMA`: Set to "true" to use local Ollama models

#### Command Line Arguments

```bash
python main.py --model MODEL_NAME --port PORT_NUMBER --provider PROVIDER --openai-base-url URL
```

Where:

- PROVIDER is either "ollama" (default) or "openai"
- URL is the base URL for OpenAI-compatible API (default: "<https://openrouter.ai/api/v1>")

#### Configuration Precedence

1. Command line arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)

#### Examples

```bash
# Using environment variables
export LLM_MODEL="llama3:latest"
export PORT=8080
python main.py

# Using CLI arguments
python main.py --model "llama3:latest" --port 8080

# Using defaults
python main.py
```

## API Endpoints

### POST /user_request

Handle user input and return LLM response or tool approval request.

Request:

```json
{
  "input": "your question or command"
}
```

Response:

```json
{
  "message": "response text",
  "request_id": "uuid-if-approval-needed",
  "requires_approval": true/false,
  "tool": "tool-name-if-applicable"
}
```

### POST /approve

Approve or deny a tool execution request.

Request:

```json
{
  "request_id": "uuid-from-user_request",
  "approve": true/false
}
```

Response:

```json
{
  "message": "execution result or denial message",
  "request_id": "same-request-id",
  "tool": "tool-name"
}
```

### GET /session_state

Get current chat session state including messages and pending requests.

Response:

```json
{
  "messages": [
    {"role": "system/user/assistant", "content": "message text"}
  ],
  "_pending_request_id": "uuid-or-null",
  "_pending_tool_call": {
    "tool": "tool-name",
    "arguments": {}
  }
}
```
