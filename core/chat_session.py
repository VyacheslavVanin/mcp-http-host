import asyncio
import json
import logging
import re
import uuid
import xml
import xmltodict
from typing import Any
from enum import Enum

from .server import Server
from .llm_client import LLMClient, Response

from fastapi.responses import StreamingResponse


def extract_tool_call(text, tag):


class ChatContinuation(Enum):
    PROMPT = 0  # User entered a prompt
    RESET_CHAT = 1  # User requested chat reset
    EXIT = 2  # User requested exit application


def LLMResponse(
    orig_response: Response | str, request_id: str = None, tool_call=None
) -> dict:
    tool = None
    if tool_call:
        tool = dict()
        tool["name"] = tool_call['tool']
        arguments = tool_call.get("arguments")
        if arguments:
            tool["arguments"] = dict()
            for k, v in arguments.items():
                tool["arguments"][k] = f"{v}"

    ret = dict()
    ret["request_id"] = request_id
    ret["requires_approval"] = request_id is not None
    ret["tool"] = tool

    if isinstance(orig_response, Response):
        ret["message"] = orig_response.content
        ret["model"] = orig_response.model
        ret["created_timestamp"] = orig_response.created_timestamp
        ret["role"] = orig_response.role
    else:
        ret["message"] = orig_response

    return ret


def LLMStreamResponse(
    orig_response: Response,
    request_id: str = None,
    tool_call=None,
    end=False,
) -> str:
    ret = LLMResponse(orig_response, request_id, tool_call)
    ret["done"] = end
    return json.dumps(ret) + "\n"


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(
        self, current_directory: str, servers: list[Server], llm_client: LLMClient
    ) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.messages: list[dict[str, str]] = []
        self._pending_request_id: str | None = None
        self._pending_tool_call: dict | None = None
        self.current_directory: str = current_directory
        self.tools: dict[str, list[Any]] = dict()

    async def init_servers(self) -> bool:
        """Initialize all servers.

        Returns:
            bool: True if all servers initialized successfully, False otherwise
        """
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                    self.tools[server.name] = await server.list_tools()
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

    def try_get_tool_call(self, text: str) -> str:
        pattern = r"BEGIN_USE_TOOL(.*?)END_USE_TOOL"
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) != 1:
            return dict()
        return json.loads(matches[0])

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
            tools = self.tools[server.name]
            all_tools.extend(tools)

        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        system_prompt_content = f"""
You are a helpful and highly skilled software developer assistant with access to these tools:
{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.
If user wants to create some application then look in current directory for more clues. Then create necessary files or modify existing.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact this format below (json between two tags):
BEGIN_USE_TOOL
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value",
        "another-argument-name": "another-value"
    }}
}}
END_USE_TOOL

# Tool Use Guidelines

1. In <thinking> tags, assess what information you already have and what information you need to proceed with the task.
2. Choose the most appropriate tool based on the task and the tool descriptions provided. Assess if you need additional information to proceed, and which of the available tools would be most effective for gathering this information. For example using the list_files tool is more effective than running a command like `ls` in the terminal. It's critical that you think about each available tool and use the one that best fits the current step in the task.
3. If multiple actions are needed, use one tool at a time per message to accomplish the task iteratively, with each tool use being informed by the result of the previous tool use. Do not assume the outcome of any tool use. Each step must be informed by the previous step's result.
4. After each tool use, the system will respond with the result of that tool use. This result will provide you with the necessary information to continue your task or make further decisions. This response may include:
  - Information about whether the tool succeeded or failed, along with any reasons for failure.
  - Linter errors that may have arisen due to the changes you made, which you'll need to address.
  - New terminal output in reaction to the changes, which you may need to consider or act upon.
  - Any other relevant feedback or information related to the tool use.
6. ALWAYS wait for user confirmation after each tool use before proceeding. Never assume the success of a tool use without explicit confirmation of the result from the system.

It is crucial to proceed step-by-step, waiting for the user's message after each tool use before moving forward with the task. This approach allows you to:
1. Confirm the success of each step before proceeding.
2. Address any issues or errors that arise immediately.
3. Adapt your approach based on new information or unexpected results.
4. Ensure that each action builds correctly on the previous ones.

By waiting for and carefully considering the user's or system response after each tool use, you can react accordingly and make informed decisions about how to proceed with the task. This iterative process helps ensure the overall success and accuracy of your work.

You can only use one tool per message. If you need execute multiple tools ask for execution one by one.

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
                "content": system_prompt_content,
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
                return LLMResponse(
                    llm_response,
                    request_id=request_id,
                    tool_call=tool_call,
                )

            self._append_llm_response(content)
            return llm_response
        except (json.JSONDecodeError, AttributeError, xml.parsers.expat.ExpatError):
            self._append_llm_response(content)
            return llm_response

    def _llm_request_stream(self, messages) -> LLMStreamResponse:
        llm_response = ""
        for part in self.llm_client.get_response_stream(messages):
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
                    tool_call=tool_call,
                    end=True,
                )
                return

            self._append_llm_response(llm_response)
            yield LLMStreamResponse("", end=True)
        except (json.JSONDecodeError, AttributeError, xml.parsers.expat.ExpatError) as e:
            print(f'Error in tool call: {e}')
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
                tool_call=self._pending_tool_call,
            )

        continuation, user_input = await self._process_user_input(user_input)
        if continuation in [ChatContinuation.EXIT, ChatContinuation.RESET_CHAT]:
            await self.init_system_message()
            return LLMResponse("Session was reseted")
        return None

    async def user_request(self, user_input: str) -> dict | None:
        """Handle a user request, potentially involving tool execution.

        Args:
            user_input: The user's input string

        Returns:
            str: The LLM response or tool approval request
            None: For exit/reset commands
        """
        return await self._llm_request(self.messages)

    def user_request_stream(self, user_input: str) -> dict | None:
        """Handle a user request, potentially involving tool execution.

        Args:
            user_input: The user's input string

        Returns:
            str: The LLM response or tool approval request
            None: For exit/reset commands
        """

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
                tools = self.tools[server.name]
                if any(tool.name == tool_call["tool"] for tool in tools):
                    result = await server.execute_tool(
                        tool_call["tool"], tool_call["arguments"]
                    )

                    self._append_system_message(f"Tool execution result: {result}")

                    self._pending_request_id = None
                    self._pending_tool_call = None
                    if self.llm_client.config.stream:

                        def stream_generator():
                            for r in self._llm_request_stream(self.messages):
                                yield r

                        return StreamingResponse(stream_generator())
                    else:
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
