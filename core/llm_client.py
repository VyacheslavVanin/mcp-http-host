import logging
from datetime import datetime
import time

import httpx

from core.configuration import Configuration
from core.json_reconstruct import JsonReconstruct


def iso8601_to_unixtimestamp(date_str):
    # return int(datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())
    if '.' in date_str and 'Z' in date_str:
        # Split into main part and fractional seconds + timezone
        date_part, rest = date_str.split('.', 1)
        fractional_part, tz_part = rest.split('Z', 1)
        # Truncate to 6 digits and pad with zeros if necessary
        fractional_truncated = fractional_part[:6].ljust(6, '0')
        # Rebuild the string with UTC offset
        new_date_str = f"{date_part}.{fractional_truncated}+0000"
    else:
        # Handle case without fractional seconds
        new_date_str = date_str.replace('Z', '+0000')
    
    # Parse the datetime
    dt = datetime.strptime(new_date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    # Convert to Unix timestamp (float)
    return dt.timestamp()
    

class Response:
    def __init__(self, role:str, content:str, model:str, created_timestamp:int, end:bool=False):
        self.content: str = content
        self.role:str = role
        self.model:str = model
        self.created_timestamp:int = created_timestamp
        self.done: bool = end

    def to_dict(self):
        return {
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "created_timestamp": self.created_timestamp,
            "done": self.done,
        }

class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(
        self,
        config: Configuration = None,
    ) -> None:
        self.config = config

    def get_response(self, messages: list[dict[str, str]]) -> Response:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        base_url = (
            self.config.openai_base_url
            if self.config
            else "https://openrouter.ai/api/v1"
        )
        url = f"{base_url}/chat/completions"
        model = (
            self.config.model if self.config.model else "llama-3.2-90b-vision-preview"
        )
        api_key = self.config.api_key

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": model,
            "max_tokens": 16384,
            "top_p": 0.9,
            "stream": False,
            "stop": None,
        }
        if self.config.temperature:
            payload["temperature"] = self.config.temperature

        try:
            with httpx.Client() as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=None,
                    stream=False,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                role = choice["message"]["role"]
                model = data["model"]
                created = data["created"]
                return Response(role, content, model, created, end=True)

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return Response(
                "system",
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request.",
                "",
                int(time.time()),
                end=True,
            )


class OllamaClient:
    """Manages communication with a local Ollama server."""

    def __init__(self, config: Configuration = None) -> None:
        """Initialize the Ollama client.

        Args:
            model: The model name to use (default: "llama3")
        """
        self.config = config

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the local Ollama server.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to Ollama fails.
        """
        url = self.config.ollama_base_url + "/api/chat"
        model = self.config.model

        payload = {"model": model, "messages": messages, "stream": False}
        if self.config.temperature:
            payload["options"]["temperature"] = self.config.temperature
        if self.config.context_window_size:
            payload["options"]["num_ctx"] = self.config.context_window_size

        try:
            with httpx.Client() as client:
                response = client.post(url, json=payload, timeout=None)
                response.raise_for_status()
                data = response.json()
                role = data["message"]["role"]
                content = data["message"]["content"]
                model = data["model"]
                created = iso8601_to_unixtimestamp(data["created_at"])
                return Response(role, content, model, created, end=True)
        except httpx.RequestError as e:
            error_message = f"Error getting Ollama response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return Response(
                "system",
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request.",
                "",
                int(time.time()),
                end=True,
            )

    def get_response_stream(self, messages: list[dict[str, str]]) -> Response:
        """Get a response from the local Ollama server.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to Ollama fails.
        """
        url = self.config.ollama_base_url + "/api/chat"
        model = self.config.model

        payload = {"model": model, "messages": messages, "stream": True}
        if self.config.temperature:
            payload["options"]["temperature"] = self.config.temperature
        if self.config.context_window_size:
            payload["options"]["num_ctx"] = self.config.context_window_size

        try:
            with httpx.stream("POST", url, json=payload, timeout=None) as response:
                ret: Response = Response(
                    "assistant", "", model, time.time(), end=True,
                )

                def cb(obj):
                    nonlocal ret
                    ret = Response(
                         obj["message"]["role"],
                         obj["message"]["content"],
                         obj["model"],
                         iso8601_to_unixtimestamp(obj["created_at"]),
                         end=obj["done"],
                    )

                jr = JsonReconstruct()
                for chunk in response.iter_text():
                    jr.process_part(chunk, cb)
                    if ret:
                        yield ret
                        ret = Response(
                            "assistant", "", model, time.time(), end=True,
                        )
                jr.finalize(cb)
                yield ret
        except httpx.RequestError as e:
            error_message = f"Error getting Ollama response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )
