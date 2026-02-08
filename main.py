import logging
import os
import sys
import copy
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.configuration import Configuration

from core.mcpserver import Server
from core.chat_session import ChatSession
from core.chat_session_manager import ChatSessionManager, ChatType
from core.llm_client import (
    OpenaiClient,
    OpenaiClientOfficial,
    LLMClientBase,
)

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


def redirect_std_outputs(stdout_file, stderr_file):
    """Redirect stdout and stderr if file paths are provided."""
    if stdout_file:
        stdout_fd = os.open(stdout_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.close(stdout_fd)

    if stderr_file:
        stderr_fd = os.open(stderr_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.close(stderr_fd)


class StartSession(BaseModel):
    current_directory: str
    model: str
    provider_base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    temperature: float = 0.2
    context_size: int = 2048
    stream: bool = False
    # Either 'chat' or 'agent'
    chat_type: str = "chat"


class UserRequest(BaseModel):
    session_id: str
    input: str
    context: str | None


class ApproveRequest(BaseModel):
    session_id: str
    request_id: str
    approve: bool


session_manager: ChatSessionManager = ChatSessionManager()


def _get_llm_client(request:StartSession, config) -> LLMClientBase:
    if config.verify_ssl:
        llm_client = OpenaiClientOfficial(copy.deepcopy(config))
    else:
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
async def get_session_state(session_id: str) -> dict:
    """Get current session state including messages and pending requests."""
    chat_session = session_manager.get_session(session_id)
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return chat_session.get_session_state()


@app.post("/start_session")
async def start_session(request: StartSession) -> dict:
    llm_client = _get_llm_client(request, config)

    chat_type = ChatType[request.chat_type.upper()]
    chat_session, session_id = await session_manager.create_session(
        config,
        llm_client,
        request.current_directory,
        chat_type,
    )

    os.chdir(request.current_directory)

    await chat_session.init_system_message()
    return {
        "session_id": session_id,
    }


if __name__ == "__main__":
    import uvicorn
    redirect_std_outputs(config.stdout_file, config.stderr_file)
    uvicorn.run(app, host="0.0.0.0", port=config.server_port)
