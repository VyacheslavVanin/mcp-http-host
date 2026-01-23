from .configuration import Configuration
from typing import Dict


class Response:
    def __init__(
        self,
        role: str,
        content: str,
        model: str,
        created_timestamp: int,
        end: bool = False,
        usage: Dict[str, int] = None,
    ):
        self.content: str = content
        self.role: str = role
        self.model: str = model
        self.created_timestamp: int = created_timestamp
        self.done: bool = end
        self.usage: Dict[str, int] = usage

    def to_dict(self):
        ret = {
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "created_timestamp": self.created_timestamp,
            "done": self.done,
        }
        if self.total_tokens is not None:
            ret.update({"usage": self.usage})
        return ret


class LLMClientBase:
    def __init__(self, config: Configuration = None) -> None:
        self.config = config

    # messages looks like this:
    # [{"role": "user", "content": "how many letters in word strawberry?"}]
    def get_response(self, messages: list[dict[str, str]]) -> Response:
        raise NotImplementedError("This method should be overridden by subclasses")

    # messages looks like this:
    # [{"role": "user", "content": "how many letters in word strawberry?"}]
    def get_response_stream(self, messages: list[dict[str, str]]) -> Response:
        raise NotImplementedError("This method should be overridden by subclasses")
