from core.configuration import Configuration


class Response:
    def __init__(
        self,
        role: str,
        content: str,
        model: str,
        created_timestamp: int,
        end: bool = False,
    ):
        self.content: str = content
        self.role: str = role
        self.model: str = model
        self.created_timestamp: int = created_timestamp
        self.done: bool = end

    def to_dict(self):
        return {
            "content": self.content,
            "role": self.role,
            "model": self.model,
            "created_timestamp": self.created_timestamp,
            "done": self.done,
        }


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
