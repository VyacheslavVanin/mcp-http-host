import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from core.chat_session import ChatSession
from core.llm_client_base import LLMClientBase, Response
from core.configuration import Configuration
from core.chat_type import ChatType
from fastapi.responses import StreamingResponse


class LLMClientMock(LLMClientBase):
    """Mock implementation of LLMClientBase for testing purposes."""

    def __init__(
        self,
        config: Configuration = None,
        response_content: str = "Test response",
        stream_responses: list = None,
    ):
        super().__init__(config)
        self.response_content = response_content
        self.stream_responses = stream_responses or []
        self.get_response_call_count = 0
        self.get_response_stream_call_count = 0

    def get_response(self, messages: list[dict[str, str]]) -> Response:
        """Mock implementation of get_response."""
        self.get_response_call_count += 1
        return Response(
            role="assistant",
            content=self.response_content,
            model="test-model",
            created_timestamp=1234567890,
            usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        )

    def get_response_stream(self, messages: list[dict[str, str]]):
        """Mock implementation of get_response_stream."""
        self.get_response_stream_call_count += 1
        for content in self.stream_responses:
            yield Response(
                role="assistant",
                content=content,
                model="test-model",
                created_timestamp=1234567890,
                usage={"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3},
            )


@pytest.fixture
def mock_config():
    """Fixture to create a mock configuration."""
    config = Configuration()
    # Override the parse_args method to return default values
    config.parse_args = lambda: type(
        "Args",
        (),
        {
            "model": "test-model",
            "port": 8000,
            "provider": "openai",
            "openai_base_url": "https://test.example.com",
            "current_directory": "./",
            "servers_config": None,
            "context_size": None,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "stream": True,  # Enable streaming for these tests
            "max_rps": 100,
            "verify_ssl": True,
            "timeout": None,
        },
    )()
    return config


@pytest.fixture
def mock_llm_client_streaming():
    """Fixture to create a mock LLM client with streaming capability."""
    return LLMClientMock(stream_responses=["Part 1", "Part 2", "Part 3"])


@pytest.fixture
def chat_session_streaming(mock_config, mock_llm_client_streaming):
    """Fixture to create a ChatSession instance with streaming enabled."""
    return ChatSession(
        current_directory="/tmp/test",
        config=mock_config,
        llm_client=mock_llm_client_streaming,
        system_prompt_template="System prompt for {current_directory}",
        chat_type=ChatType.CHAT,
    )


class TestChatSessionStreaming:
    """Test cases for streaming functionality in ChatSession."""

    def test_user_request_stream_basic(self, chat_session_streaming):
        """Test basic user request streaming."""
        # Mock the toolbox to avoid external dependencies
        chat_session_streaming.toolbox = MagicMock()
        chat_session_streaming.toolbox.get_tools_descriptions.return_value = (
            "Test tools"
        )
        chat_session_streaming.toolbox.initialize = AsyncMock(return_value=True)
        chat_session_streaming.toolbox.cleanup_servers = AsyncMock(return_value=None)

        # Initialize system message first
        asyncio.run(chat_session_streaming.init_system_message())

        # Make a user request with streaming
        response = chat_session_streaming.user_request_stream("Hello, bot!")

        # Check that response is a StreamingResponse
        assert isinstance(response, StreamingResponse)

    def test_user_request_stream_with_context(self, chat_session_streaming):
        """Test user request streaming with additional context."""
        # Mock the toolbox to avoid external dependencies
        chat_session_streaming.toolbox = MagicMock()
        chat_session_streaming.toolbox.get_tools_descriptions.return_value = (
            "Test tools"
        )
        chat_session_streaming.toolbox.initialize = AsyncMock(return_value=True)
        chat_session_streaming.toolbox.cleanup_servers = AsyncMock(return_value=None)

        # Initialize system message first
        asyncio.run(chat_session_streaming.init_system_message())

        # Make a user request with streaming and context
        response = chat_session_streaming.user_request_stream(
            "Hello!", system_context="Current file: test.py"
        )

        # Check that response is a StreamingResponse
        assert isinstance(response, StreamingResponse)

        # Check that system context was added to messages
        context_found = False
        for msg in chat_session_streaming.messages:
            if "additional_context:" in msg["content"] and "test.py" in msg["content"]:
                context_found = True
                break
        assert context_found, "System context was not added to messages"
