import chromadb
from app.core.ports.retriever import Retriever
from app.core.ports.embeddings import Embedder

class ChromaRetriever(Retriever):
    def __init__(self, path="chroma_db", collection="rag_docs", embedder: Embedder | None = None):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 6):
        try:
            qvec = self.embedder.embed_query(query)
            res = self.col.query(
                query_embeddings=[qvec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]

            lines, sources = [], []
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                src = (m or {}).get("source", f"doc{i}")
                sources.append(str(src))
                lines.append(f"[{i}] 出典: {src}\n{d}")
            context = "\n\n---\n\n".join(lines)
            return context, sorted(set(sources))
        except Exception:
            # コレクションが空などでも落とさない
            return "", []
