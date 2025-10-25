from app.adapters.providers.ollama_client import OllamaClient
from app.adapters.embeddings.sbert_embedder import SbertEmbedder
from app.adapters.rag.chroma_retriever import ChromaRetriever

def build_stack(kind: str, **kwargs):
    if kind == "ollama":
        llm = OllamaClient(base_url=kwargs.get("base_url", "http://localhost:11434"))
        embed = SbertEmbedder(kwargs.get("embed_model", "intfloat/multilingual-e5-small"))
        retriever = ChromaRetriever(path=kwargs.get("chroma_path", "chroma_db"), embedder=embed)
        return llm, retriever
    # 追って openai/claude を追加
    raise ValueError(f"unknown provider kind: {kind}")
