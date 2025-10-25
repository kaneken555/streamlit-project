import os
from dataclasses import dataclass

@dataclass
class Settings:
    provider: str = os.environ.get("PROVIDER_KIND", "ollama")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    default_model: str = os.environ.get("DEFAULT_MODEL", "llama3:8b")
    embed_model: str = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
    chroma_path: str = os.environ.get("CHROMA_PATH", "chroma_db")
    num_ctx: int = int(os.environ.get("NUM_CTX", "8192"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.2"))

settings = Settings()
