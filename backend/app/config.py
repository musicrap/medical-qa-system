import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application config"""
    APP_NAME: str = "Medical QA System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"

    EMBEDDING_PROVIDER: str = "dashscope"  # openai / local / dashscope
    EMBEDDING_MODEL: str = "text-embedding-v4"
    LOCAL_EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese"

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "medical_knowledge"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    # 混合检索配置
    HYBRID_VECTOR_K: int = 10
    HYBRID_KEYWORD_K: int = 10
    RERANK_TOP_K: int = 3
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    DATA_PATH: str = "./data/train-sft.jsonl"

    CORS_ORIGINS: list = ["http://localhost:5173", "http://127.0.0.1:5173"]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
