from core.chat_session import ChatSession
from core.llm_client_base import LLMClientBase
from core.server import Server
import uuid


class ChatSessionManager:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = dict()

    async def create_session(
        self, servers: list[Server], llm_client: LLMClientBase, current_directory: str
    ):
        session_id = str(uuid.uuid4())
        chat_session = ChatSession(current_directory, servers, llm_client)
        if not await chat_session.init_session():
            raise RuntimeError("Failed to initialize chat session")
        self.sessions[session_id] = chat_session
        return chat_session, session_id

    def get_session(self, session_id: str) -> ChatSession:
        return self.sessions.get(session_id)
