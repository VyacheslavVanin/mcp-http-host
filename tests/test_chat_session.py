import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.chat_session import ChatSession, make_response
from core.llm_client_base import LLMClientBase, Response
from core.configuration import Configuration
from core.chat_type import ChatType


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
            "stream": False,
            "max_rps": 100,
            "verify_ssl": True,
            "timeout": None,
        },
    )()
    return config


@pytest.fixture
def mock_llm_client():
    """Fixture to create a mock LLM client."""
    return LLMClientMock()


@pytest.fixture
def chat_session(mock_config, mock_llm_client):
    """Fixture to create a ChatSession instance."""
    return ChatSession(
        current_directory="/tmp/test",
        config=mock_config,
        llm_client=mock_llm_client,
        system_prompt_template="System prompt for {current_directory}",
        chat_type=ChatType.CHAT,
    )


class TestChatSessionHappyPath:
    """Test cases for the happy path scenarios of ChatSession."""

    def test_initialization(self, chat_session, mock_config, mock_llm_client):
        """Test that ChatSession initializes correctly."""
        assert chat_session.current_directory == "/tmp/test"
        assert chat_session.llm_client == mock_llm_client
        assert chat_session.messages == []
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None
        assert chat_session.chat_type == ChatType.CHAT

    def test_init_system_message(self, chat_session):
        """Test initialization of system message."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Call init_system_message
        result = asyncio.run(chat_session.init_session())

        assert result is True

        # Check that pending calls are cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None

        # Check that system message was added
        expected_messages = [
            {
                "role": "system",
                "content": "System prompt for /tmp/test",  # We'll check the content separately
            }
        ]
        assert expected_messages == chat_session.messages

    def test_user_request_basic(self, chat_session):
        """Test basic user request handling."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Mock the LLM client response
        llm_client_mock = LLMClientMock(response_content="Hello, user!")
        # Set up a config with stream=False to match the fixture
        config_mock = MagicMock()
        config_mock.stream = False
        llm_client_mock.config = config_mock
        chat_session.llm_client = llm_client_mock

        # Make a user request
        response = asyncio.run(chat_session.user_request("Hello, bot!"))

        # Check that response is properly formatted
        expected_response = {
            "request_id": None,
            "requires_approval": False,
            "message": "Hello, user!",
            "model": "test-model",
            "created_timestamp": 1234567890,
            "role": "assistant",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "tools": [],
        }
        assert response == expected_response

        # Check that messages were added correctly
        expected_messages = [
            {
                "role": "system",
                "content": chat_session.messages[0]["content"],
            },  # system message
            {"role": "user", "content": "Hello, bot!"},
            {"role": "assistant", "content": "Hello, user!"},
        ]
        assert chat_session.messages == expected_messages

    def test_user_request_with_system_context(self, chat_session):
        """Test user request with additional system context."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Mock the LLM client response
        llm_client_mock = LLMClientMock(response_content="Response with context")
        # Set up a config with stream=False to match the fixture
        config_mock = MagicMock()
        config_mock.stream = False
        llm_client_mock.config = config_mock
        chat_session.llm_client = llm_client_mock

        # Make a user request with system context
        response = asyncio.run(
            chat_session.user_request("Hello!", system_context="Current file: test.py")
        )

        # Check that system context was added
        assert len(chat_session.messages) >= 3  # system + context + user + assistant
        # Find the context message
        context_message_exists = any(
            "additional_context:" in msg["content"] and "test.py" in msg["content"]
            for msg in chat_session.messages
        )
        assert context_message_exists, "System context was not added to messages"

        # Check that response is properly formatted
        expected_response = {
            "request_id": None,
            "requires_approval": False,
            "message": "Response with context",
            "model": "test-model",
            "created_timestamp": 1234567890,
            "role": "assistant",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "tools": [],
        }
        assert response == expected_response

    def test_validate_request_no_pending(self, chat_session):
        """Test validate_request when there are no pending requests."""
        # Initially, there should be no pending requests
        result = asyncio.run(chat_session.validate_request("test input"))
        assert result is None

    def test_validate_request_exit_command(self, chat_session):
        """Test validate_request with exit command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Test exit command
        result = asyncio.run(chat_session.validate_request("exit"))
        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Session was reseted",
            "tools": [],
        }
        assert result == expected_result

    def test_validate_request_quit_command(self, chat_session):
        """Test validate_request with quit command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Test quit command
        result = asyncio.run(chat_session.validate_request("quit"))
        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Session was reseted",
            "tools": [],
        }
        assert result == expected_result

    def test_validate_request_clear_command(self, chat_session):
        """Test validate_request with clear command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Test clear command
        result = asyncio.run(chat_session.validate_request("/clear"))
        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Session was reseted",
            "tools": [],
        }
        assert result == expected_result

    def test_get_session_state(self, chat_session):
        """Test getting session state."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Add a user message
        chat_session._append_user_message("Test message")

        # Get session state
        state = chat_session.get_session_state()

        expected_state = {
            "messages": [
                {
                    "role": "system",
                    "content": "System prompt for /tmp/test",
                },  # system message
                {"role": "user", "content": "Test message"},
            ],
            "_pending_request_id": None,
            "_pending_tool_call": None,
        }
        assert state == expected_state


class TestMakeResponseFunction:
    """Test cases for the make_response utility function."""

    def test_make_response_with_response_object(self):
        """Test make_response with Response object."""
        response = Response(
            role="assistant",
            content="Test content",
            model="test-model",
            created_timestamp=1234567890,
            usage={"total_tokens": 10},
        )

        result = make_response(response, request_id="test-id")

        expected_result = {
            "request_id": "test-id",
            "requires_approval": True,
            "message": "Test content",
            "model": "test-model",
            "created_timestamp": 1234567890,
            "role": "assistant",
            "usage": {"total_tokens": 10},
            "tools": [],
        }
        assert result == expected_result

    def test_make_response_with_string(self):
        """Test make_response with string."""
        result = make_response("Test string", request_id=None)

        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Test string",
            "tools": [],
        }
        assert result == expected_result


class TestChatSessionToolApproval:
    """Test cases for tool approval functionality in ChatSession."""

    def test_approve_valid_request(self, chat_session):
        """Test approving a valid tool request."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None
        chat_session.toolbox.execute_tool = AsyncMock(
            return_value="Tool executed successfully"
        )

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call
        request_id = "test-request-id"
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}
        chat_session._pending_request_id = request_id
        chat_session._pending_tool_call = tool_call

        # Mock the LLM client to return a response after tool execution
        llm_client_mock = LLMClientMock(
            response_content="Processed after tool execution"
        )
        # Set up a config with stream=False to match the fixture
        config_mock = MagicMock()
        config_mock.stream = False
        llm_client_mock.config = config_mock
        chat_session.llm_client = llm_client_mock

        # Approve the request
        result = asyncio.run(chat_session.approve(request_id, True))

        # Check that the result contains the expected content
        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Processed after tool execution",
            "model": "test-model",
            "created_timestamp": 1234567890,
            "role": "assistant",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "tools": [],
        }
        # The result could be a response dict or a streaming response depending on config
        # Since we didn't set stream=True, it should be a regular response dict
        assert result["message"] == "Processed after tool execution"
        assert result["model"] == "test-model"

        # Verify that the pending call was cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None

        # Verify that the tool was executed
        chat_session.toolbox.execute_tool.assert_called_once_with(
            "test_tool", {"param": "value"}
        )

    def test_deny_valid_request(self, chat_session):
        """Test denying a valid tool request."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call
        request_id = "test-request-id"
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}
        chat_session._pending_request_id = request_id
        chat_session._pending_tool_call = tool_call

        # Deny the request
        result = asyncio.run(chat_session.approve(request_id, False))

        # Check that the result indicates denial
        expected_result = {
            "request_id": None,
            "requires_approval": False,
            "message": "Tool execution denied",
            "tools": [],
        }
        assert result == expected_result

        # Verify that the pending call was cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None

        # Verify that system message was added about denial
        last_message = chat_session.messages[-1]
        assert last_message == {
            "role": "system",
            "content": "User denied tool execution",
        }

    def test_approve_invalid_request_id(self, chat_session):
        """Test approving with an invalid request ID."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.initialize.return_value = True
        chat_session.toolbox.cleanup_servers.return_value = None

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call with a different request ID
        valid_request_id = "valid-request-id"
        invalid_request_id = "invalid-request-id"
        tool_call = {"name": "test_tool", "arguments": {"param": "value"}}
        chat_session._pending_request_id = valid_request_id
        chat_session._pending_tool_call = tool_call

        # Try to approve with invalid request ID
        result = asyncio.run(chat_session.approve(invalid_request_id, True))

        # Check that the result indicates invalid request
        expected_result = {
            "request_id": valid_request_id,
            "requires_approval": True,
            "message": "Invalid or expired request ID",
            "tools": [{"name": "test_tool", "arguments": {"param": "value"}}],
        }
        assert result == expected_result

        # Verify that the pending call was NOT cleared
        assert chat_session._pending_request_id == valid_request_id
        assert chat_session._pending_tool_call == tool_call
