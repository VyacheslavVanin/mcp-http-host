from core.chat_session import ChatSession
from core.llm_client_base import LLMClientBase
from core.server import Server
import uuid
from enum import Enum


class ChatType(Enum):
    CHAT = 0  # Simple chat
    AGENT = 1  # Chat with access to mcp-tools


def _create_system_prompt_template_chat() -> str:
    return """
You are a helpful and highly skilled software developer assistant.
You must help user with their task.
"""


def _create_system_prompt_template_all_tools() -> str:
    return """
You are a helpful and highly skilled software developer assistant with access to these tools:
{tools_description}

With these tools you must help user with their task.
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.
If user wants to create some application then look in current directory for more clues. Then create necessary files or modify existing.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact this format below (json between {begin_tool} and {end_tool}):
{begin_tool}
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value",
        "another-argument-name": "another-value"
    }}
}}
{end_tool}


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

When you use the 'read_file' tool do not reply with file content unless asked explicitly (user can see files himself).
When you use the 'edit_files' or 'write_whole_file' tools do not reply with resulting file to user.
When you need to write or edit files DO NOT print to user contents of file before of after editing.
Prefere to use 'edit_files' over 'write_whole_file' if file already exists.
Use 'write_whle_file' to create file or overwrite whole file if it is small.
If you you need create large file (more than 100 lines) create some skeleton file and then use series of 'edit_files' by about 50 lines.

Yor current directory is {current_directory}
"""


def _get_system_prompt_template(chat_type: ChatType) -> str:
    if chat_type == ChatType.CHAT:
        return _create_system_prompt_template_chat()
    elif chat_type == ChatType.AGENT:
        return _create_system_prompt_template_all_tools()


class ChatSessionManager:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = dict()

    async def create_session(
        self,
        servers: list[Server],
        llm_client: LLMClientBase,
        current_directory: str,
        chat_type: ChatType = ChatType.AGENT,
    ):
        session_id = str(uuid.uuid4())
        chat_session = ChatSession(
            current_directory,
            servers,
            llm_client,
            _get_system_prompt_template(chat_type),
        )
        if not await chat_session.init_session():
            raise RuntimeError("Failed to initialize chat session")
        self.sessions[session_id] = chat_session
        return chat_session, session_id

    def get_session(self, session_id: str) -> ChatSession:
        return self.sessions.get(session_id)
