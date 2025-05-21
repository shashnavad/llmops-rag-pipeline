import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LLMOps RAG Pipeline"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    MODEL_NAME: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector DB Settings
    VECTOR_DB_PATH: str = "data/vectordb"
    
    # Experiment Tracking
    COMET_API_KEY: str = os.getenv("COMET_API_KEY", "")
    COMET_PROJECT_NAME: str = "llmops-rag-pipeline"
    
    class Config:
        env_file = ".env"

settings = Settings()
