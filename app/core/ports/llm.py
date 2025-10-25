from typing import Iterable, List, Dict, Any, Protocol
from app.core.types import Message, ChatChunk

class LLMClient(Protocol):
    def list_models(self) -> List[str]: ...
    def chat_stream(
        self, messages: List[Message], model: str, options: Dict[str, Any]
    ) -> Iterable[ChatChunk]: ...
