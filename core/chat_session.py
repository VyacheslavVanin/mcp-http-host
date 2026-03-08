import asyncio
import json
import logging
import re
import uuid
import os
import sys
from typing import Any, NoReturn
from enum import Enum

from mcptoolbox.mcpserver import Server, ToolBox
from .llm_client_base import LLMClientBase, Response
from .configuration import Configuration

from fastapi.responses import StreamingResponse
from .chat_type import ChatType


class ChatContinuation(Enum):
    PROMPT = 0  # User entered a prompt
    RESET_CHAT = 1  # User requested chat reset
    EXIT = 2  # User requested exit application


class ToolCallValidationError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


def make_response(
    orig_response: Response | str, request_id: str = None, tool_calls=[]
) -> dict:
    tools = []
    for tool_call in tool_calls:
        tool = dict()
        tool["name"] = tool_call["name"]
        arguments = tool_call.get("arguments")
        if arguments:
            tool["arguments"] = dict()
            for k, v in arguments.items():
                tool["arguments"][k] = f"{v}"
        tools.append(tool)

    ret = dict()
    ret["request_id"] = request_id
    ret["requires_approval"] = request_id is not None
    ret["tools"] = tools

    if isinstance(orig_response, Response):
        ret["message"] = orig_response.content
        ret["model"] = orig_response.model
        ret["created_timestamp"] = orig_response.created_timestamp
        ret["role"] = orig_response.role
        ret["usage"] = orig_response.usage
    else:
        ret["message"] = orig_response

    return ret


def to_stream_response(
    orig_response: Response,
    request_id: str = None,
    tool_calls=[],
    end=False,
) -> str:
    ret = make_response(orig_response, request_id, tool_calls)
    ret["done"] = end
    return json.dumps(ret) + "\n"


class PendingToolsManager:
    """Manages pending tool calls in a chat session."""

    def __init__(self):
        self._pending_tools: dict[str, dict] = {}  # Maps request_id to tool_call

    def get_pending_tool_calls(self) -> list[dict]:
        return list(self._pending_tools.values())

    def add_pending_tool_call(self, request_id: str, tool_call: dict):
        self._pending_tools[request_id] = tool_call

    def clear_pending_calls(self):
        self._pending_tools.clear()

    def clear_pending_call(self, request_id: str):
        if request_id in self._pending_tools:
            del self._pending_tools[request_id]

    def get_pending_call(self, request_id: str) -> dict | None:
        return self._pending_tools.get(request_id)

    def has_pending_calls(self) -> bool:
        return len(self._pending_tools) > 0

    @property
    def pending_request_ids(self):
        return list(self._pending_tools.keys())

    # Backward compatibility methods
    @property
    def pending_request_id(self):
        """Returns the first pending request ID for backward compatibility."""
        if self._pending_tools:
            return next(iter(self._pending_tools))
        return None

    @property
    def pending_tool_call(self):
        """Returns the first pending tool call for backward compatibility."""
        if self._pending_tools:
            return self._pending_tools[self.pending_request_id]
        return None


configuration = Configuration()
tool_box = ToolBox(Configuration.load_config(configuration.servers_config_path))


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(
        self,
        current_directory: str,
        config: Configuration,
        llm_client: LLMClientBase,
        system_prompt_template: str,
        chat_type: ChatType,
    ) -> None:
        self.toolbox: ToolBox = tool_box
        self.llm_client: LLMClientBase = llm_client
        self.messages: list[dict[str, str]] = []
        self.pending_tools_manager: PendingToolsManager = PendingToolsManager()
        self.current_directory: str = current_directory
        self.system_prompt_template: str = system_prompt_template
        self.chat_type = chat_type
        self.retries_on_llm_error: int = config.retries_on_llm_error

    def debug_write_messages_to_tmp_file(
        self, file_name="/tmp/llm-requester.messages.log"
    ):
        """Write a debug message to a temporary file for debugging purposes.
        Args:
            file_name: The path to the debug log file. Defaults to '/tmp/llm-requester.messages.log'.
        """
        try:
            with open(file_name, "w") as f:
                f.write(json.dumps(self.messages))
        except Exception as e:
            logging.error(f"Failed to write debug message to {file_name}: {e}")

    def get_pending_tool_calls(self) -> list[dict]:
        return self.pending_tools_manager.get_pending_tool_calls()

    def clear_pending_calls(self):
        self.pending_tools_manager.clear_pending_calls()

    async def init_servers(self) -> bool:
        return await self.toolbox.initialize()

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        await self.toolbox.cleanup_servers()

    def try_get_tool_call(self, text: str) -> str:
        # need to replace to kinda escape this tags to be able use tools
        # on this file)
        pattern = r"BEGIN-USE-TOOL(.*?)END-USE-TOOL".replace("-", "_")
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) != 1:
            return dict()
        ret = json.loads(matches[0])

        # convert relative path to absolute
        arguments = ret.get("arguments")
        if arguments:
            path: str = arguments.get("path")
            if path and not path.startswith("/"):
                arguments["path"] = os.path.join(self.current_directory, path)

        return ret

    def _validate_tool_call(self, tool_call: dict) -> str | None:
        """Validate a tool call against its schema.

        Args:
            tool_call: The tool call to validate

        Returns:
            None if valid, error message string if invalid
        """

        def raise_error(message: str) -> NoReturn:
            logging.warning(message)
            raise ToolCallValidationError(f"Tool call validation failed: {message}")

        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        # Find the tool schema
        tool_schema = None
        for server_tools in self.toolbox.tools.values():
            for tool in server_tools:
                if tool.name == tool_name:
                    tool_schema = tool.inputSchema
                    break
            if tool_schema:
                break

        if not tool_schema:
            raise_error(f"Tool '{tool_name}' not found in available tools")

        # Validate required arguments
        required = tool_schema.get("required", [])
        for req_arg in required:
            if req_arg not in arguments:
                raise_error(
                    f"Missing required argument: '{req_arg}' for  tool '{tool_name}'"
                )

        # Validate argument types and unknown arguments
        properties = tool_schema.get("properties", {})
        for arg_name, arg_value in arguments.items():
            # Check for unknown arguments
            if arg_name not in properties:
                raise_error(f"Unknown argument '{arg_name}' for tool '{tool_name}'")

            # Check argument type
            prop_schema = properties[arg_name]
            expected_type = prop_schema.get("type")
            if expected_type:
                actual_type = type(arg_value).__name__
                type_mapping = {
                    "string": "str",
                    "integer": "int",
                    "number": "float",
                    "boolean": "bool",
                    "array": "list",
                    "object": "dict",
                }
                expected_python_type = type_mapping.get(expected_type)
                if expected_python_type and actual_type != expected_python_type:
                    raise_error(
                        f"Invalid type for argument '{arg_name}' of tool '{tool_name}': expected {expected_type}, got {actual_type}"
                    )

            # Validate enum values
            if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
                raise_error(
                    f"Invalid value for argument '{arg_name}' of tool '{tool_name}': expected one of {prop_schema['enum']}, got '{arg_value}'"
                )

        return None

    def _append_llm_response(self, message):
        self.messages.append({"role": "assistant", "content": message})
        self.debug_write_messages_to_tmp_file()

    def _append_system_message(self, message):
        self.messages.append({"role": "system", "content": message})
        self.debug_write_messages_to_tmp_file()

    def _append_tool_message(self, message):
        self.messages.append({"role": "tool", "content": message})
        self.debug_write_messages_to_tmp_file()

    def _append_user_message(self, message):
        self.messages.append({"role": "user", "content": message})
        self.debug_write_messages_to_tmp_file()

    def add_rulesmd_to_context(self) -> None:
        """
        Reads file rules.md from current directory and adds its content as system message.
        It can be rules.md, RULES.md or Rules.md.
        """
        for name in ["rules.md", "RULES.md", "Rules.md"]:
            rules_file_path = os.path.join(self.current_directory, name)
            if os.path.exists(rules_file_path):
                with open(rules_file_path, "r", encoding="utf-8") as f:
                    rules_content = f.read()
                    self._append_system_message(
                        f"Rules from rules.md:\n{rules_content}\n"
                    )
                    break

    async def init_system_message(self) -> None:
        """Initialize the system message with tool descriptions."""
        tools_description = self.toolbox.get_tools_descriptions()

        system_prompt_content = self.system_prompt_template.format(
            current_directory=self.current_directory,
            tools_description=tools_description,
            begin_tool="BEGIN-USE-TOOL".replace("-", "_"),
            end_tool="END-USE-TOOL".replace("-", "_"),
        )

        self.messages = [
            {
                "role": "system",
                "content": system_prompt_content,
            }
        ]
        if self.chat_type == ChatType.AGENT:
            self.add_rulesmd_to_context()
        self.clear_pending_calls()

    async def init_session(self):
        try:
            if not await self.init_servers():
                raise RuntimeError("Failed to initialize servers")
            await self.init_system_message()
            return True
        except Exception:
            await self.cleanup_servers()
            return False

    async def _process_user_input(
        self, user_input: str
    ) -> tuple[ChatContinuation, str]:
        if user_input in ["quit", "exit"]:
            return (ChatContinuation.EXIT, user_input)
        if user_input in ["/clear"]:
            await self.init_system_message()
            return (ChatContinuation.RESET_CHAT, user_input)
        return (ChatContinuation.PROMPT, user_input)

    async def _llm_request(self, messages, max_retries: int = 3) -> dict:
        """Request LLM response with retry logic on validation failure.

        Args:
            messages: List of conversation messages
            max_retries: Maximum number of retry attempts

        Returns:
            dict: The response or error message
        """
        last_error = None
        for attempt in range(max_retries + 1):
            llm_response = self.llm_client.get_response(messages)
            content = llm_response.content
            try:
                tool_call = self.try_get_tool_call(content)
                if "name" in tool_call and "arguments" in tool_call:
                    # Validate tool call before requesting approval
                    self._validate_tool_call(tool_call)

                    request_id = str(uuid.uuid4())
                    self.pending_tools_manager.add_pending_tool_call(
                        request_id, tool_call
                    )
                    logging.info(
                        f"Request user confirmation for {tool_call['name']} {request_id=}"
                    )
                    self._append_llm_response(content)
                    return make_response(
                        llm_response,
                        request_id=request_id,
                        tool_calls=self.get_pending_tool_calls(),
                    )

                return make_response(llm_response)
            except (json.JSONDecodeError, ToolCallValidationError, AttributeError) as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        f"Validation failed on attempt {attempt + 1}/{max_retries + 1}, retrying ..."
                    )
                else:
                    logging.error(
                        f"All {max_retries + 1} attempts failed. Last error: {e}"
                    )
                    self._append_llm_response(content)
                    self._append_tool_message(f"Tool call failed: {e}")
                    return make_response(
                        f"\nFailed after {max_retries + 1} attempts: {e}\n"
                    )

    def _llm_request_stream(self, messages, max_retries: int = 3):
        """Request LLM response with retry logic on validation failure.

        Args:
            messages: List of conversation messages
            max_retries: Maximum number of retry attempts

        Yields:
            str: Streamed response parts
        """
        last_error = None
        for attempt in range(max_retries + 1):
            llm_response = ""
            for part in self.llm_client.get_response_stream(messages):
                if part and part.content is not None:
                    yield to_stream_response(part)
                    llm_response += part.content
            try:
                tool_call = self.try_get_tool_call(llm_response)
                if "name" in tool_call and "arguments" in tool_call:
                    # Validate tool call before requesting approval
                    self._validate_tool_call(tool_call)

                    request_id = str(uuid.uuid4())
                    self.pending_tools_manager.add_pending_tool_call(
                        request_id, tool_call
                    )
                    self._append_llm_response(llm_response)
                    yield to_stream_response(
                        f"\napprove required {tool_call['name']}\n",
                        request_id=self.pending_tools_manager.pending_request_id,
                        tool_calls=self.get_pending_tool_calls(),
                        end=True,
                    )
                    return

                self._append_llm_response(llm_response)
                yield to_stream_response("", end=True)
                return
            except (json.JSONDecodeError, ToolCallValidationError, AttributeError) as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        f"Validation failed on attempt {attempt + 1}/{max_retries + 1}, retrying ..."
                    )
                else:
                    logging.error(
                        f"All {max_retries + 1} attempts failed. Last error: {e}"
                    )
                    self._append_llm_response(llm_response)
                    self._append_tool_message(f"Tool call failed: {e}")
                    yield to_stream_response(
                        f"\nFailed after {max_retries + 1} attempts: {e}\n",
                        end=True,
                    )
                    return

    async def validate_request(self, user_input: str) -> dict | None:
        """Validate a user request.

        Args:
            user_input: The user's input string

        Returns:
            str: The LLM response with error
            None: If request is ok
        """
        if self.pending_tools_manager.has_pending_calls():
            return make_response(
                "approve required ",
                request_id=self.pending_tools_manager.pending_request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

        continuation, user_input = await self._process_user_input(user_input)
        if continuation in [ChatContinuation.EXIT, ChatContinuation.RESET_CHAT]:
            await self.init_system_message()
            return make_response("Session was reseted")
        return None

    async def user_request(
        self, user_input: str, system_context: str = ""
    ) -> dict | None:
        """Handle a user request, potentially involving tool execution.

        Args:
            user_input: The user's input string
            system_context: The additional system info like: "current open file: main.cpp, tabs: ['header.hpp', 'source.cpp']"

        Returns:
            str: The LLM response or tool approval request
            None: For exit/reset commands
        """

        if system_context:
            self._append_tool_message(f"additional_context:\n{system_context}\n")
        self._append_user_message(user_input)
        return await self._llm_request(
            self.messages, max_retries=self.retries_on_llm_error
        )

    def user_request_stream(
        self, user_input: str, system_context: str = ""
    ) -> dict | None:
        """Handle a user request, potentially involving tool execution.

        Args:
            user_input: The user's input string
            system_context: The additional system info like: "current open file: main.cpp, tabs: ['header.hpp', 'source.cpp']"

        Returns:
            str: The LLM response or tool approval request
            None: For exit/reset commands
        """

        if system_context:
            self._append_tool_message(f"additional_context:\n{system_context}\n")
        self._append_user_message(user_input)

        def stream_generator():
            for r in self._llm_request_stream(
                self.messages, max_retries=self.retries_on_llm_error
            ):
                yield r

        return StreamingResponse(stream_generator())

    async def approve(self, request_id: str, approve: bool) -> dict | StreamingResponse:
        """Handle tool approval/denial.

        Args:
            request_id: The request ID to approve/deny
            approve: True to approve, False to deny

        Returns:
            str: The result of tool execution or denial message
        """
        if (
            not self.pending_tools_manager.has_pending_calls()
            or self.pending_tools_manager.get_pending_call(request_id) is None
        ):
            logging.warning(
                f"Tool approval request not found or expired. {request_id=}"
            )
            return make_response(
                "Invalid or expired request ID",
                request_id=self.pending_tools_manager.pending_request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

        if not approve:
            self.clear_pending_calls()
            self._append_tool_message(f"User denied tool execution")
            return make_response("Tool execution denied")

        try:
            tool_call = self.pending_tools_manager.get_pending_call(request_id)
            if tool_call is None:
                return make_response(
                    f"Tool call not found for request ID: {request_id}",
                    request_id=request_id,
                    tool_calls=self.get_pending_tool_calls(),
                )
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            result = await self.toolbox.execute_tool(tool_name, args)
            if result is None:
                return make_response(
                    f"No server found with tool: ",
                    request_id=request_id,
                    tool_calls=self.get_pending_tool_calls(),
                )

            self._append_tool_message(
                f"User approved {tool_name} tool execution. {tool_name} tool execution result:\n{result}"
            )

            self.clear_pending_calls()
            if self.llm_client.config.stream:

                def stream_generator():
                    for r in self._llm_request_stream(
                        self.messages, max_retries=self.retries_on_llm_error
                    ):
                        yield r

                return StreamingResponse(stream_generator())
            else:
                return await self._llm_request(
                    self.messages, max_retries=self.retries_on_llm_error
                )

        except Exception as e:
            self._append_tool_message(f"Tool execution failed with error {str(e)}")
            return make_response(
                f"Error executing tool: {str(e)}",
                request_id=request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

    def get_session_state(self) -> dict:
        """Get current session state including messages and pending requests.

        Returns:
            dict: Contains messages, pending_request_id and pending_tool_call
        """
        return {
            "messages": self.messages,
            "_pending_request_id": self.pending_tools_manager.pending_request_id,
            "_pending_tool_call": self.pending_tools_manager.pending_tool_call,
        }

    async def start(self) -> None:
        """Main chat session handler."""
        if not await self.init_session():
            return

        try:
            while True:
                try:
                    if not self.pending_tools_manager.has_pending_calls():
                        user_input = input("You: ").strip().lower()
                        response = await self.user_request(user_input)
                        if not response:
                            break
                        print(response.message)
                        continue
                    else:
                        user_input = (
                            input(f"Execute tool '{response.tool['name']}'? (y/n): ")
                            .strip()
                            .lower()
                        )
                        response = await self.approve(
                            response.request_id, user_input == "y"
                        )
                        print(response.message)
                        continue

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.toolbox.cleanup_servers()
