from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    embedding_model_name: str = "all-MiniLM-L6-v2"

    summarizer_mode: str = "heuristic" # change to "ollama" to use llm-based summaries (use for code repos)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    rerank_mode: str = "cross_encoder"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"


settings = Settings()