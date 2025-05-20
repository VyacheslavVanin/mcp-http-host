import os
from typing import Any
import argparse
import json
from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables and CLI args."""
        self.load_env()
        self.api_key = os.getenv("LLM_API_KEY")

        # Parse CLI args first
        args = self.parse_args()

        # Set configuration with CLI args taking precedence over env vars
        self.model = (
            args.model if args.model else os.getenv("LLM_MODEL", "qwen2.5-coder:latest")
        )
        self.port = int(args.port) if args.port else int(os.getenv("PORT", "8000"))
        self.llm_provider = (
            args.provider.lower()
            if args.provider
            else os.getenv("LLM_PROVIDER", "ollama").lower()
        )
        self.openai_base_url = (
            args.openai_base_url
            if args.openai_base_url
            else os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.ollama_base_url = (
            args.ollama_base_url
            if args.ollama_base_url
            else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.current_directory = args.current_directory
        self.servers_config_path = args.servers_config
        self.context_size = int(args.context_size) if args.context_size else None
        self.temperature = float(args.temperature) if args.temperature else None
        self.top_k = float(args.top_k) if args.top_k else None
        self.top_p = float(args.top_p) if args.top_p else None
        self.stream: bool = bool(args.stream)
        self.max_rps: int = int(args.max_rps) if args.max_rps else 100

    @staticmethod
    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="MCP Host CLI")
        parser.add_argument("--model", help="LLM model to use", default=None)
        parser.add_argument(
            "--port", type=int, help="Port to run server on", default=None
        )
        parser.add_argument(
            "--provider",
            help="LLM provider (ollama/openai)",
            choices=["ollama", "openai"],
            default=None,
        )
        parser.add_argument(
            "--openai-base-url", help="Base URL for OpenAI-compatible API", default=None
        )
        parser.add_argument(
            "--ollama-base-url", help="Base URL for Ollama-compatible API", default=None
        )
        parser.add_argument(
            "--servers-config",
            help="Path to servers config file",
            default=None,
        )
        parser.add_argument(
            "--current-directory",
            help="Set current working directory",
            default="./",
        )
        parser.add_argument(
            "--context-size",
            help="Set context window size",
            default=None,
        )
        parser.add_argument(
            "--temperature",
            help="Set temperature",
            default=None,
        )
        parser.add_argument(
            "--top_k",
            help="Set top_k",
            default=None,
        )
        parser.add_argument(
            "--top_p",
            help="Set top_p",
            default=None,
        )
        parser.add_argument(
            "--stream",
            help="Set stream mode",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--max-rps",
            help="Limit requests to llm provider",
            type=int,
            default=100,
        )
        return parser.parse_args()

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @property
    def model_name(self) -> str:
        """Get the configured LLM model name.

        Returns:
            The model name as a string.
        """
        return self.model

    @property
    def server_port(self) -> int:
        """Get the configured server port.

        Returns:
            The port number as integer.
        """
        return self.port

    @property
    def use_ollama(self) -> bool:
        """Check if Ollama should be used as the LLM provider.

        Returns:
            True if Ollama should be used, False for openai.
        """
        return self.llm_provider == "ollama"

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        ret = {"mcpServers": dict()}
        if file_path is not None:
            with open(file_path, "r") as f:
                ret = json.load(f)

        ret["mcpServers"]["basic-read-files-server"] = {
            "command": "uv",
            "args": [
                "--directory",
                f"{os.getcwd()}/basic-mcp-server",
                "run",
                "read_files.py",
            ],
            "disabled": False,
            "autoApprove": ["list_tools", "list_files", "read_file", ""],
        }
        ret["mcpServers"]["basic-write-files-server"] = {
            "command": "uv",
            "args": [
                "--directory",
                f"{os.getcwd()}/basic-mcp-server",
                "run",
                "write_files.py",
            ],
            "disabled": False,
            "autoApprove": ["list_tools", ""],
        }
        ret["mcpServers"]["basic-exec-cli-server"] = {
            "command": "uv",
            "args": [
                "--directory",
                f"{os.getcwd()}/basic-mcp-server",
                "run",
                "exec_cli.py",
            ],
            "disabled": False,
            "autoApprove": ["list_tools", ""],
        }
        return ret

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key
