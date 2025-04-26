# LLM Chat Server

FastAPI server for chat interactions with LLMs (Ollama or OpenAI).

## Configuration

Configuration is managed via environment variables and CLI arguments.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_API_KEY` | API key for LLM provider | - |
| `LLM_MODEL` | Model name to use | `qwen2.5-coder:latest` |
| `PORT` | Port to run server on | `8000` |
| `LLM_PROVIDER` | LLM provider (`ollama` or `openai`) | `ollama` |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API | `https://openrouter.ai/api/v1` |
| `OLLAMA_BASE_URL` | Base URL for Ollama-compatible API | `http://localhost:11434` |

Example `.env` file:

```ini
LLM_API_KEY=your-api-key
LLM_MODEL=qwen2.5-coder:latest
PORT=8000
LLM_PROVIDER=ollama
```

### CLI Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | LLM model to use | `--model qwen2.5-coder:latest` |
| `--port` | Port to run server on | `--port 8000` |
| `--provider` | LLM provider (`ollama` or `openai`) | `--provider ollama` |
| `--openai-base-url` | Base URL for OpenAI API | `--openai-base-url https://api.openai.com/v1` |
| `--ollama-base-url` | Base URL for Ollama API | `--ollama-base-url http://localhost:11434` |
| `--servers-config` | Path to servers config file | `--servers-config config/servers.json` |
| `--current-directory` | Working directory | `--current-directory /projects` |
| `--context-window-size` | Context window size | `--context-window-size 2048` |
| `--temperature` | Temperature parameter | `--temperature 0.7` |
| `--stream` | Enable streaming mode | `--stream` |

Example CLI usage:

```bash
python main.py --model qwen2.5-coder:latest --port 8000 --provider ollama
```

## API Endpoints

- `POST /user_request` - Handle user chat requests
- `POST /approve` - Handle tool approval/denial
- `GET /session_state` - Get current session state

## Request/Response Format

### Request (POST /user_request)

```json
{
    "input": "user message"
}
```

### Response

```json
{
    "request_id": "uuid",
    "requires_approval": true,
    "tool": {
        "name": "tool-name",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2"
        }
    },
    "message": "response content",
    "model": "model-name",
    "role": "assistant"
}
```

### Tool Approval (POST /approve)

```json
{
    "request_id": "uuid",
    "approve": true
}
```

## Example Usage

```python
import requests

# Start session
response = requests.post("http://localhost:8000/start_session", json={
    "current_directory": "/projects",
    "llm_provider": "ollama",
    "model": "qwen2.5-coder:latest"
})

# Send message
response = requests.post("http://localhost:8000/user_request", json={
    "input": "Hello, how are you?"
})

# Approve tool call
response = requests.post("http://localhost:8000/approve", json={
    "request_id": "12345",
    "approve": true
})
