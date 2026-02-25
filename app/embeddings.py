from sentence_transformers import SentenceTransformer
from .settings import settings

_model = None

def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [e.tolist() for e in embs]