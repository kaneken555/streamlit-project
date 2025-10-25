from typing import Protocol, Tuple

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 6) -> Tuple[str, list[str]]: ...
