import asyncio
import json
import logging
import re
import uuid
import os
from typing import Any
from enum import Enum

from .mcpserver import Server, ToolBox
from .llm_client_base import LLMClientBase, Response
from core.configuration import Configuration

from fastapi.responses import StreamingResponse
from .chat_type import ChatType


class ChatContinuation(Enum):
    PROMPT = 0  # User entered a prompt
    RESET_CHAT = 1  # User requested chat reset
    EXIT = 2  # User requested exit application


def LLMResponse(
    orig_response: Response | str, request_id: str = None, tool_calls=[]
) -> dict:
    tools = []
    for tool_call in tool_calls:
        tool = dict()
        tool["name"] = tool_call["tool"]
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


def LLMStreamResponse(
    orig_response: Response,
    request_id: str = None,
    tool_calls=[],
    end=False,
) -> str:
    ret = LLMResponse(orig_response, request_id, tool_calls)
    ret["done"] = end
    return json.dumps(ret) + "\n"


tool_box = ToolBox(Configuration())


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
        self._pending_request_id: str | None = None
        self._pending_tool_call: dict | None = None
        self.current_directory: str = current_directory
        self.system_prompt_template: str = system_prompt_template
        self.chat_type = chat_type

    def get_pending_tool_calls(self) -> list[dict]:
        if self._pending_tool_call is not None:
            return [self._pending_tool_call]
        return []

    def clear_pending_calls(self):
        self._pending_request_id = None
        self._pending_tool_call = None

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

    def _append_llm_response(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def _append_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def _append_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

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

    async def _llm_request(self, messages) -> dict:
        llm_response = self.llm_client.get_response(messages)
        content = llm_response.content
        self._append_llm_response(content)
        try:
            tool_call = self.try_get_tool_call(content)
            if "tool" in tool_call and "arguments" in tool_call:
                request_id = str(uuid.uuid4())
                self._pending_request_id = request_id
                self._pending_tool_call = tool_call
                logging.info(
                    f"Request user confirmation for {tool_call['tool']} {request_id=}"
                )
                return LLMResponse(
                    llm_response,
                    request_id=request_id,
                    tool_calls=self.get_pending_tool_calls(),
                )

            self._append_llm_response(content)
            return LLMResponse(llm_response)
        except (json.JSONDecodeError, AttributeError):
            self._append_llm_response(content)
            return LLMResponse(llm_response)

    def _llm_request_stream(self, messages) -> LLMStreamResponse:
        llm_response = ""
        for part in self.llm_client.get_response_stream(messages):
            if part and part.content is not None:
                yield LLMStreamResponse(part)
                llm_response += part.content
        self._append_llm_response(llm_response)
        try:
            tool_call = self.try_get_tool_call(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                request_id = str(uuid.uuid4())
                self._pending_request_id = request_id
                self._pending_tool_call = tool_call
                yield LLMStreamResponse(
                    f"\napprove required {tool_call['tool']}\n",
                    request_id=request_id,
                    tool_calls=self.get_pending_tool_calls(),
                    end=True,
                )
                return

            self._append_llm_response(llm_response)
            yield LLMStreamResponse("", end=True)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error in tool call: {e}")
            self._append_llm_response(llm_response)
            yield LLMStreamResponse("", end=True)

    async def validate_request(self, user_input: str) -> dict | None:
        """Validate a user request.

        Args:
            user_input: The user's input string

        Returns:
            str: The LLM response with error
            None: If request is ok
        """
        if self._pending_request_id is not None:
            return LLMResponse(
                "approve required ",
                request_id=self._pending_request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

        continuation, user_input = await self._process_user_input(user_input)
        if continuation in [ChatContinuation.EXIT, ChatContinuation.RESET_CHAT]:
            await self.init_system_message()
            return LLMResponse("Session was reseted")
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
            self._append_system_message(f"additional_context:\n{system_context}\n")
        self._append_user_message(user_input)
        return await self._llm_request(self.messages)

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
            self._append_system_message(f"additional_context:\n{system_context}\n")
        self._append_user_message(user_input)

        def stream_generator():
            for r in self._llm_request_stream(self.messages):
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
            self._pending_request_id is None
            or self._pending_request_id != request_id
            or self._pending_tool_call is None
        ):
            logging.warning(
                f"Tool approval request not found or expired. {request_id=} {self._pending_request_id=}"
            )
            return LLMResponse(
                "Invalid or expired request ID",
                request_id=self._pending_request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

        if not approve:
            self.clear_pending_calls()
            self._append_system_message(f"User denied tool execution")
            return LLMResponse("Tool execution denied")

        try:
            tool_call = self._pending_tool_call
            tool_name = tool_call["tool"]
            args = tool_call["arguments"]
            result = await self.toolbox.execute_tool(tool_name, args)
            if result is None:
                return LLMResponse(
                    f"No server found with tool: ",
                    request_id=self._pending_request_id,
                    tool_calls=self.get_pending_tool_calls(),
                )

            self._append_system_message(
                f"User approved {tool_name} tool execution. {tool_name} tool execution result:\n{result}"
            )

            self.clear_pending_calls()
            if self.llm_client.config.stream:

                def stream_generator():
                    for r in self._llm_request_stream(self.messages):
                        yield r

                return StreamingResponse(stream_generator())
            else:
                return await self._llm_request(self.messages)

        except Exception as e:
            self._append_system_message(f"Tool execution failed with error {str(e)}")
            return LLMResponse(
                f"Error executing tool: {str(e)}",
                request_id=self._pending_request_id,
                tool_calls=self.get_pending_tool_calls(),
            )

    def get_session_state(self) -> dict:
        """Get current session state including messages and pending requests.

        Returns:
            dict: Contains messages, pending_request_id and pending_tool_call
        """
        return {
            "messages": self.messages,
            "_pending_request_id": self._pending_request_id,
            "_pending_tool_call": self._pending_tool_call,
        }

    async def start(self) -> None:
        """Main chat session handler."""
        if not await self.init_session():
            return

        try:
            while True:
                try:
                    if not self._pending_request_id:
                        user_input = input("You: ").strip().lower()
                        response = await self.user_request(user_input)
                        if not response:
                            break
                        print(response.message)
                        continue
                    else:
                        user_input = (
                            input(f"Execute tool '{response.tool['tool']}'? (y/n): ")
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
