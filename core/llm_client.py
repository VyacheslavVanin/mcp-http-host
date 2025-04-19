import logging

import httpx

from core.configuration import Configuration


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(
        self,
        config: Configuration = None,
    ) -> None:
        self.config = config

    def get_response(self, messages: list[dict[str, str]]) -> str:
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
        model = self.config.model if self.config.model else "llama-3.2-90b-vision-preview"
        api_key = self.config.api_key

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": 16384,
            "top_p": 0.9,
            "stream": False,
            "stop": None,
        }
        if self.config.temperature:
            payload["temperature"] = self.config.temperature

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload, timeout=None)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
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
        url = self.config.ollama_base_url + '/api/chat'
        model = self.config.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if self.config.temperature:
            payload["options"]["temperature"] = self.config.temperature
        if self.config.context_window_size:
            payload["options"]["num_ctx"] = self.config.context_window_size

        try:
            with httpx.Client() as client:
                response = client.post(url, json=payload, timeout=None)
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]

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
