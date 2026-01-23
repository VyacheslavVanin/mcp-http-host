import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.chat_session import ChatSession, make_response
from core.llm_client_base import LLMClientBase, Response
from core.configuration import Configuration
from core.chat_type import ChatType


class LLMClientMock(LLMClientBase):
    """Mock implementation of LLMClientBase for testing purposes."""
    
    def __init__(self, config: Configuration = None, response_content: str = "Test response", stream_responses: list = None):
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
            usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
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
                usage={"total_tokens": 5, "prompt_tokens": 2, "completion_tokens": 3}
            )


@pytest.fixture
def mock_config():
    """Fixture to create a mock configuration."""
    config = Configuration()
    # Override the parse_args method to return default values
    config.parse_args = lambda: type('Args', (), {
        'model': 'test-model',
        'port': 8000,
        'provider': 'openai',
        'openai_base_url': 'https://test.example.com',
        'current_directory': './',
        'servers_config': None,
        'context_size': None,
        'temperature': None,
        'top_k': None,
        'top_p': None,
        'stream': False,
        'max_rps': 100,
        'verify_ssl': True,
        'timeout': None
    })()
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
        chat_type=ChatType.CHAT
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
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"

        # Call init_system_message
        result = asyncio.run(chat_session.init_session())

        assert result is True

        # Check that system message was added
        assert len(chat_session.messages) == 1
        assert chat_session.messages[0]["role"] == "system"
        assert "System prompt for /tmp/test" in chat_session.messages[0]["content"]

        # Check that pending calls are cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None
    
    def test_user_request_basic(self, chat_session):
        """Test basic user request handling."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        
        # Initialize system message first
        asyncio.run(chat_session.init_system_message())
        
        # Mock the LLM client response
        chat_session.llm_client = LLMClientMock(response_content="Hello, user!")
        
        # Make a user request
        response = asyncio.run(chat_session.user_request("Hello, bot!"))

        # Check that response is properly formatted
        assert "message" in response
        assert response["message"] == "Hello, user!"
        
        # Check that messages were added correctly
        assert len(chat_session.messages) == 3  # system + user + assistant
        assert chat_session.messages[1]["role"] == "user"
        assert chat_session.messages[1]["content"] == "Hello, bot!"
        assert chat_session.messages[2]["role"] == "assistant"
        assert chat_session.messages[2]["content"] == "Hello, user!"
        
    
    def test_user_request_with_system_context(self, chat_session):
        """Test user request with additional system context."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        
        # Initialize system message first
        asyncio.run(chat_session.init_system_message())
        
        # Mock the LLM client response
        chat_session.llm_client = LLMClientMock(response_content="Response with context")
        
        # Make a user request with system context
        response = asyncio.run(chat_session.user_request("Hello!", system_context="Current file: test.py"))
        
        # Check that system context was added
        assert len(chat_session.messages) >= 3  # system + context + user + assistant
        # Find the context message
        context_found = False
        for msg in chat_session.messages:
            if "additional_context:" in msg["content"] and "test.py" in msg["content"]:
                context_found = True
                break
        assert context_found, "System context was not added to messages"
        
        # Check that response is properly formatted
        assert "message" in response
        assert response["message"] == "Response with context"
    
    def test_validate_request_no_pending(self, chat_session):
        """Test validate_request when there are no pending requests."""
        # Initially, there should be no pending requests
        result = asyncio.run(chat_session.validate_request("test input"))
        assert result is None
    
    def test_validate_request_exit_command(self, chat_session):
        """Test validate_request with exit command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        
        # Initialize system message first
        asyncio.run(chat_session.init_system_message())
        
        # Test exit command
        result = asyncio.run(chat_session.validate_request("exit"))
        assert result is not None
        assert "Session was reseted" in result["message"]
    
    def test_validate_request_quit_command(self, chat_session):
        """Test validate_request with quit command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        
        # Initialize system message first
        asyncio.run(chat_session.init_system_message())
        
        # Test quit command
        result = asyncio.run(chat_session.validate_request("quit"))
        assert result is not None
        assert "Session was reseted" in result["message"]
    
    def test_validate_request_clear_command(self, chat_session):
        """Test validate_request with clear command."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        
        # Initialize system message first
        asyncio.run(chat_session.init_system_message())
        
        # Test clear command
        result = asyncio.run(chat_session.validate_request("/clear"))
        assert result is not None
        assert "Session was reseted" in result["message"]
    
    def test_get_session_state(self, chat_session):
        """Test getting session state."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Add a user message
        chat_session._append_user_message("Test message")

        # Get session state
        state = chat_session.get_session_state()

        assert "messages" in state
        assert "_pending_request_id" in state  # Fixed key name
        assert "_pending_tool_call" in state
        assert len(state["messages"]) == 2  # system + user
        assert state["_pending_request_id"] is None
        assert state["_pending_tool_call"] is None


class TestMakeResponseFunction:
    """Test cases for the make_response utility function."""

    def test_make_response_with_response_object(self):
        """Test make_response with Response object."""
        response = Response(
            role="assistant",
            content="Test content",
            model="test-model",
            created_timestamp=1234567890,
            usage={"total_tokens": 10}
        )

        result = make_response(response, request_id="test-id")

        assert result["request_id"] == "test-id"
        assert result["requires_approval"] is True
        assert result["message"] == "Test content"
        assert result["model"] == "test-model"
        assert result["created_timestamp"] == 1234567890
        assert result["usage"] == {"total_tokens": 10}
        assert result["tools"] == []

    def test_make_response_with_string(self):
        """Test make_response with string."""
        result = make_response("Test string", request_id=None)

        assert result["request_id"] is None
        assert result["requires_approval"] is False
        assert result["message"] == "Test string"
        assert "model" not in result
        assert result["tools"] == []


class TestChatSessionToolApproval:
    """Test cases for tool approval functionality in ChatSession."""

    def test_approve_valid_request(self, chat_session):
        """Test approving a valid tool request."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = AsyncMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"
        chat_session.toolbox.execute_tool = AsyncMock(return_value="Tool executed successfully")
        chat_session.toolbox.cleanup_servers = AsyncMock()

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call
        request_id = "test-request-id"
        tool_call = {
            "name": "test_tool",
            "arguments": {"param": "value"}
        }
        chat_session._pending_request_id = request_id
        chat_session._pending_tool_call = tool_call

        # Mock the LLM client to return a response after tool execution
        chat_session.llm_client = LLMClientMock(response_content="Processed after tool execution")

        # Approve the request
        result = asyncio.run(chat_session.approve(request_id, True))

        # Check that the result contains the expected content
        assert "message" in result
        # The result could be a response dict or a streaming response depending on config
        # Since we didn't set stream=True, it should be a regular response dict

        # Verify that the pending call was cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None

        # Verify that the tool was executed
        chat_session.toolbox.execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    def test_deny_valid_request(self, chat_session):
        """Test denying a valid tool request."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call
        request_id = "test-request-id"
        tool_call = {
            "name": "test_tool",
            "arguments": {"param": "value"}
        }
        chat_session._pending_request_id = request_id
        chat_session._pending_tool_call = tool_call

        # Deny the request
        result = asyncio.run(chat_session.approve(request_id, False))

        # Check that the result indicates denial
        assert "message" in result
        assert "Tool execution denied" in result["message"]

        # Verify that the pending call was cleared
        assert chat_session._pending_request_id is None
        assert chat_session._pending_tool_call is None

        # Verify that system message was added about denial
        last_message = chat_session.messages[-1]
        assert "User denied tool execution" in last_message["content"]

    def test_approve_invalid_request_id(self, chat_session):
        """Test approving with an invalid request ID."""
        # Mock the toolbox to avoid external dependencies
        chat_session.toolbox = MagicMock()
        chat_session.toolbox.get_tools_descriptions.return_value = "Test tools"

        # Initialize system message first
        asyncio.run(chat_session.init_system_message())

        # Set up a pending tool call with a different request ID
        valid_request_id = "valid-request-id"
        invalid_request_id = "invalid-request-id"
        tool_call = {
            "name": "test_tool",
            "arguments": {"param": "value"}
        }
        chat_session._pending_request_id = valid_request_id
        chat_session._pending_tool_call = tool_call

        # Try to approve with invalid request ID
        result = asyncio.run(chat_session.approve(invalid_request_id, True))

        # Check that the result indicates invalid request
        assert "message" in result
        assert "Invalid or expired request ID" in result["message"]

        # Verify that the pending call was NOT cleared
        assert chat_session._pending_request_id == valid_request_id
        assert chat_session._pending_tool_call == tool_call
