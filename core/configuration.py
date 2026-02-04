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

        # Parse CLI args first
        args = self.parse_args()

        # Set configuration with CLI args taking precedence over env vars
        self.model = (
            args.model if args.model else os.getenv("LLM_MODEL", "qwen2.5-coder:latest")
        )
        self.port = int(args.port) if args.port else int(os.getenv("PORT", "8000"))

        # Handle API key: from CLI arg, env var, or file
        self.api_key = self._get_api_key(args.api_key_file)

        self.llm_provider = (
            args.provider.lower()
            if args.provider
            else os.getenv("LLM_PROVIDER", "openai").lower()
        )
        self.openai_base_url = (
            args.openai_base_url
            if args.openai_base_url
            else os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        )
        self.current_directory = args.current_directory
        self.servers_config_path = args.servers_config
        self.context_size = int(args.context_size) if args.context_size else None
        self.temperature = float(args.temperature) if args.temperature else None
        self.top_k = float(args.top_k) if args.top_k else None
        self.top_p = float(args.top_p) if args.top_p else None
        self.stream: bool = bool(args.stream)
        self.max_rps: int = int(args.max_rps) if args.max_rps else 100
        self.verify_ssl: bool = bool(args.verify_ssl)
        self.timeout: float = float(args.timeout) if args.timeout else None

    def _get_api_key(self, api_key_file_path: str = None) -> str:
        """Get API key from CLI argument, environment variable, or file path.

        Args:
            api_key_file_path: Optional path to file containing the API key

        Returns:
            The API key as a string
        """
        # Check if API key is provided as a command-line argument (would be the file path)
        if api_key_file_path:
            try:
                with open(api_key_file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"API key file not found: {api_key_file_path}")
            except Exception as e:
                raise ValueError(f"Error reading API key file: {str(e)}")

        # Fall back to environment variable
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            return api_key

        # If no API key is found anywhere, return None (will be checked later)
        return None

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
            help="LLM provider (openai)",
            choices=["openai"],
            default=None,
        )
        parser.add_argument(
            "--openai-base-url", help="Base URL for OpenAI-compatible API", default=None
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
        parser.add_argument(
            "--verify-ssl",
            help="Enable ssl verification on requests (disable only when you sure that it is safe to do so)",
            action=argparse.BooleanOptionalAction,
            type=bool,
            default=True,
        )
        parser.add_argument(
            "--timeout",
            help="Set request timeout in seconds (float). None or 0 - no timeout",
            default=None,
        )
        parser.add_argument(
            "--api-key-file",
            help="Path to file containing the API key",
            default=None,
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
            ValueError: If the API key is not found from any source.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found. Please provide it via:\n"
                           "- Environment variable LLM_API_KEY\n"
                           "- Command-line argument --api-key-file\n"
                           "- Or pass it in the API request body")
        return self.api_key
