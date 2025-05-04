import logging
from datetime import datetime
import time

import httpx
from openai import OpenAI, AsyncOpenAI

from core.configuration import Configuration
from core.json_reconstruct import JsonReconstruct
from core.llm_client_base import LLMClientBase, Response


def iso8601_to_unixtimestamp(date_str):
    if "." in date_str and "Z" in date_str:
        # Split into main part and fractional seconds + timezone
        date_part, rest = date_str.split(".", 1)
        fractional_part, tz_part = rest.split("Z", 1)
        # Truncate to 6 digits and pad with zeros if necessary
        fractional_truncated = fractional_part[:6].ljust(6, "0")
        # Rebuild the string with UTC offset
        new_date_str = f"{date_part}.{fractional_truncated}+0000"
    else:
        # Handle case without fractional seconds
        new_date_str = date_str.replace("Z", "+0000")

    # Parse the datetime
    dt = datetime.strptime(new_date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    # Convert to Unix timestamp (float)
    return dt.timestamp()


def _rate_limit(rps: int):
    delay: float = 1.0 / rps
    time.sleep(delay)


class OpenaiClient(LLMClientBase):
    """Manages communication with the LLM provider."""

    def __init__(self, config: Configuration = None) -> None:
        super().__init__(config)

    def get_response(self, messages: list[dict[str, str]]) -> Response:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.
        """
        base_url = self.config.openai_base_url
        model = self.config.model
        api_key = self.config.api_key

        _rate_limit(self.config.max_rps)
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=self.config.temperature,
        )

        model = response.model
        created = response.created
        choice = response.choices[0]
        role = choice.message.role
        content = choice.message.content
        return Response(role, content, model, created, end=True)

    def get_response_stream(self, messages: list[dict[str, str]]) -> Response:
        """Get a response from the local Ollama server.

        Args:
            messages: A list of message dictionaries.
        """
        base_url = self.config.openai_base_url
        model = self.config.model
        api_key = self.config.api_key

        _rate_limit(self.config.max_rps)
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=self.config.temperature,
        )
        for chunk in response:
            role = chunk.choices[0].delta.role
            content = chunk.choices[0].delta.content
            created = chunk.created
            end = chunk.choices[0].finish_reason is not None
            yield Response(
                role,
                content,
                model,
                created,
                end=end,
            )


class OllamaClient(LLMClientBase):
    """Manages communication with a local Ollama server."""

    def __init__(self, config: Configuration = None) -> None:
        super().__init__(config)

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

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": dict(),
        }
        if self.config.temperature:
            payload["options"]["temperature"] = self.config.temperature
        if self.config.context_window_size:
            payload["options"]["num_ctx"] = self.config.context_window_size

        _rate_limit(self.config.max_rps)
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
        """
        url = self.config.ollama_base_url + "/api/chat"
        model = self.config.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": dict(),
        }
        if self.config.temperature:
            payload["options"]["temperature"] = self.config.temperature
        if self.config.context_window_size:
            payload["options"]["num_ctx"] = self.config.context_window_size

        _rate_limit(self.config.max_rps)
        try:
            with httpx.stream("POST", url, json=payload, timeout=None) as response:
                ret: Response = Response(
                    "assistant",
                    "",
                    model,
                    time.time(),
                    end=True,
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
                            "assistant",
                            "",
                            model,
                            time.time(),
                            end=True,
                        )
                jr.finalize(cb)
                yield ret
        except httpx.RequestError as e:
            error_message = f"Error getting response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )
