from sentence_transformers import SentenceTransformer
from app.core.ports.embeddings import Embedder

class SbertEmbedder(Embedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self.model.encode([f"query: {text}"], normalize_embeddings=True).tolist()[0]

    def embed_texts(self, texts):
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()
