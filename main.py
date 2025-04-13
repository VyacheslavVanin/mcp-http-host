import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.configuration import Configuration

from core.server import Server
from core.chat_session import ChatSession
from core.llm_client import LLMClient, OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

config = Configuration()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserRequest(BaseModel):
    input: str


class ApproveRequest(BaseModel):
    request_id: str
    approve: bool


chat_session: ChatSession | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize the chat session on startup."""
    global chat_session
    server_config = config.load_config(config.servers_config_path)
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    model = config.model_name
    if config.use_ollama:
        llm_client = OllamaClient(model)
    else:
        llm_client = LLMClient(config.llm_api_key, model)

    chat_session = ChatSession(servers, llm_client)
    if not await chat_session.init_session():
        raise RuntimeError("Failed to initialize chat session")


@app.post("/user_request")
async def handle_user_request(request: UserRequest) -> dict:
    """Handle a user request through HTTP endpoint."""
    if chat_session is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = await chat_session.user_request(request.input)
    if response is None:
        raise HTTPException(status_code=400, detail="Invalid request")

    return {
        "message": response.message,
        "request_id": response.request_id,
        "requires_approval": response.request_id is not None,
        "tool": response.tool,
    }


@app.post("/approve")
async def handle_approval(request: ApproveRequest) -> dict:
    """Handle tool approval/denial through HTTP endpoint."""
    if chat_session is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    response = await chat_session.approve(request.request_id, request.approve)
    return {
        "message": response.message,
        "request_id": response.request_id,
        "tool": response.tool,
    }


@app.get("/session_state")
async def get_session_state() -> dict:
    """Get current session state including messages and pending requests."""
    if chat_session is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return chat_session.get_session_state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.server_port)
