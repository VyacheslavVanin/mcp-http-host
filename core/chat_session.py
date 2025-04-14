import asyncio
import json
import logging
import re
import xml
import xmltodict
from enum import Enum

from .server import Server
from .llm_client import LLMClient


def get_xmldict_in_tags(text, tag):
    pattern = r"(<{0}>(.*?)<\/{0}>)".format(tag)
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) != 1:
        return dict()
    return xmltodict.parse(matches[0][0])["use_tool"]


class ChatContinuation(Enum):
    PROMPT = 0  # User entered a prompt
    RESET_CHAT = 1  # User requested chat reset
    EXIT = 2  # User requested exit application


class LLMResponse:
    def __init__(self, message: str, request_id: str = None, tool_call=None):
        self.message: str = message
        self.request_id: str = request_id
        self.tool: str | None = None
        if tool_call:
            tool_text = f"{tool_call['tool']}("
            for k, v in tool_call["arguments"].items():
                tool_text += f'"{k}": "{v}",'
            tool_text += ")"
            self.tool: str = tool_text


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.messages: list[dict[str, str]] = []
        self._pending_request_id: str | None = None
        self._pending_tool_call: dict | None = None
        self.current_directory: str = "./"

    async def init_servers(self) -> bool:
        """Initialize all servers.

        Returns:
            bool: True if all servers initialized successfully, False otherwise
        """
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return False
            return True
        except Exception as e:
            logging.error(f"Error initializing servers: {e}")
            await self.cleanup_servers()
            return False

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    def try_get_tool_call(self, llm_response: str) -> str:
        return get_xmldict_in_tags(llm_response, "use_tool")

    def _append_llm_response(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def _append_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def _append_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    async def init_system_message(self) -> None:
        """Initialize the system message with tool descriptions."""
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)

        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        sys_content = f"""
You are a helpful assistant with access to these tools:
{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact format below:
<use_tool>
    <tool> tool-name </tool>
    <arguments>
        <argument-name>value</argument-name>
        <another-argument-name>another-value</another-argument-name>
    </arguments>
</use_tool>
After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data
Please use only the tools that are explicitly defined above.
Yor current directory is {self.current_directory}
"""

        self.messages = [
            {
                "role": "system",
                "content": sys_content,
            }
        ]
        self._pending_request_id = None
        self._pending_tool_call = None

    async def init_session(self):
        try:
            if not await self.init_servers():
                return
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
        self._append_user_message(user_input)
        return (ChatContinuation.PROMPT, user_input)

    async def _llm_request(self, messages) -> LLMResponse:
        llm_response = self.llm_client.get_response(messages)
        self._append_llm_response(llm_response)
        try:
            tool_call = self.try_get_tool_call(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                import uuid

                request_id = str(uuid.uuid4())
                self._pending_request_id = request_id
                self._pending_tool_call = tool_call
                return LLMResponse(
                    f"approve required {tool_call['tool']}",
                    request_id=request_id,
                    tool_call=tool_call,
                )

            self._append_llm_response(llm_response)
            return LLMResponse(llm_response)
        except (json.JSONDecodeError, AttributeError, xml.parsers.expat.ExpatError):
            self._append_llm_response(llm_response)
            return LLMResponse(llm_response)

    async def user_request(self, user_input: str) -> LLMResponse | None:
        """Handle a user request, potentially involving tool execution.

        Args:
            user_input: The user's input string

        Returns:
            str: The LLM response or tool approval request
            None: For exit/reset commands
        """
        if self._pending_request_id is not None:
            return LLMResponse(
                "approve required ",
                request_id=self._pending_request_id,
                tool_call=self._pending_tool_call,
            )

        continuation, user_input = await self._process_user_input(user_input)
        if continuation in [ChatContinuation.EXIT, ChatContinuation.RESET_CHAT]:
            await self.init_system_message()
            return LLMResponse("Session was reseted")

        return await self._llm_request(self.messages)

    async def approve(self, request_id: str, approve: bool) -> LLMResponse:
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
            return LLMResponse(
                "Invalid or expired request ID",
                request_id=self._pending_request_id,
                tool_call=self._pending_tool_call,
            )

        if not approve:
            self._pending_request_id = None
            self._pending_tool_call = None
            return LLMResponse("Tool execution denied")

        try:
            tool_call = self._pending_tool_call
            for server in self.servers:
                tools = await server.list_tools()
                if any(tool.name == tool_call["tool"] for tool in tools):
                    result = await server.execute_tool(
                        tool_call["tool"], tool_call["arguments"]
                    )

                    self._append_system_message(f"Tool execution result: {result}")

                    self._pending_request_id = None
                    self._pending_tool_call = None
                    return await self._llm_request(self.messages)

            return LLMResponse(
                f"No server found with tool: {tool_call['tool']}",
                request_id=self._pending_request_id,
                tool_call=self._pending_tool_call,
            )
        except Exception as e:
            return LLMResponse(
                f"Error executing tool: {str(e)}",
                request_id=self._pending_request_id,
                tool_call=self._pending_tool_call,
            )
        finally:
            self._pending_request_id = None
            self._pending_tool_call = None

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
            await self.cleanup_servers()
