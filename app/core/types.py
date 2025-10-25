from dataclasses import dataclass
from typing import Dict, Optional

Role = str  # "system" | "user" | "assistant" | "tool"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChatChunk:
    content: str
    done: bool = False
    usage: Optional[Dict[str, int]] = None  # tokens 等（必要に応じて）
