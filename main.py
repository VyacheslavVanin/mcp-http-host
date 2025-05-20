import logging
import os
import copy
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.configuration import Configuration

from core.server import Server
from core.chat_session import ChatSession
from core.chat_session_manager import ChatSessionManager
from core.llm_client import OpenaiClient, OllamaClient, LLMClientBase

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


class StartSession(BaseModel):
    current_directory: str
    # ollama or openai
    llm_provider: str = "ollama"
    model: str
    provider_base_url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.2
    context_size: int = 2048
    stream: bool = False


class UserRequest(BaseModel):
    session_id: str
    input: str
    context: str | None


class ApproveRequest(BaseModel):
    session_id: str
    request_id: str
    approve: bool


session_manager: ChatSessionManager = ChatSessionManager()
servers: list[Server] = []


def _get_llm_client(request, config) -> LLMClientBase:
    if request.llm_provider is None or request.llm_provider == "ollama":
        llm_client = OllamaClient(copy.deepcopy(config))
        if request.provider_base_url:
            llm_client.config.ollama_base_url = request.provider_base_url
    elif request.llm_provider == "openai":
        llm_client = OpenaiClient(copy.deepcopy(config))
        if request.provider_base_url:
            llm_client.config.openai_base_url = request.provider_base_url
    if request.model:
        llm_client.config.model = request.model
    if request.api_key:
        llm_client.config.api_key = request.api_key
    llm_client.config.temperature = request.temperature
    llm_client.config.context_size = request.context_size
    llm_client.config.stream = request.stream
    return llm_client


@app.on_event("startup")
async def startup_event():
    """Initialize the chat session on startup."""
    global session_manager
    global servers
    server_config = config.load_config(config.servers_config_path)
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]


@app.post("/user_request")
async def handle_user_request(request: UserRequest) -> dict:
    """Handle a user request through HTTP endpoint."""
    chat_session = session_manager.get_session(request.session_id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    validation = await chat_session.validate_request(request.input)
    if validation is not None:
        return validation

    if not chat_session.llm_client.config.stream:
        response = await chat_session.user_request(request.input, request.context)
        if response is None:
            raise HTTPException(status_code=400, detail="Invalid request")

        return response
    else:
        return chat_session.user_request_stream(request.input, request.context)


@app.post("/approve")
async def handle_approval(request: ApproveRequest) -> dict:
    """Handle tool approval/denial through HTTP endpoint."""
    chat_session = session_manager.get_session(request.session_id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return await chat_session.approve(request.request_id, request.approve)


@app.get("/session_state")
async def get_session_state() -> dict:
    """Get current session state including messages and pending requests."""
    chat_session = session_manager.get_session(request.session_id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return chat_session.get_session_state()


@app.post("/start_session")
async def start_session(request: StartSession) -> dict:
    llm_client = _get_llm_client(request, config)

    chat_session, session_id = await session_manager.create_session(
        servers,
        llm_client,
        request.current_directory,
    )

    os.chdir(request.current_directory)

    await chat_session.init_system_message()
    return {
        "session_id": session_id,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.server_port)
