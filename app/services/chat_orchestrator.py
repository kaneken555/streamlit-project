from typing import List, Dict, Any, Tuple, Iterable
from app.core.types import Message, ChatChunk
from app.core.ports.llm import LLMClient
from app.core.ports.retriever import Retriever
from app.core.prompts import build_system_prompt

class ChatOrchestrator:
    def __init__(self, llm: LLMClient, retriever: Retriever, base_system_prompt: str):
        self.llm = llm
        self.retriever = retriever
        self.base_system_prompt = base_system_prompt

    def run_stream(
        self,
        user_input: str,
        history: List[Message],
        model: str,
        options: Dict[str, Any],
        top_k: int = 4,
    ) -> Tuple[Iterable[ChatChunk], list[str]]:
        # 1) RAG
        context, sources = self.retriever.retrieve(user_input, top_k=top_k)

        # 2) Prompt 合成
        system = build_system_prompt(self.base_system_prompt, context)

        # 3) メッセージ
        messages = [Message(role="system", content=system)]
        messages.extend(m for m in history if m.role in ("user", "assistant"))
        messages.append(Message(role="user", content=user_input))

        # 4) 実行
        stream = self.llm.chat_stream(messages=messages, model=model, options=options)
        return stream, sources
