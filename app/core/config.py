import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    PROJECT_NAME: str = "LLMOps RAG Pipeline"
    API_V1_STR: str = "/api/v1"
    
    # LLM Settings
    MODEL_NAME: str = "microsoft/Phi-3-mini-4k-instruct"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector DB Settings
    VECTOR_DB_PATH: str = "data/vectordb"
    
    # Experiment Tracking
    COMET_API_KEY: str = os.getenv("COMET_API_KEY", "")
    COMET_PROJECT_NAME: str = "llmops-rag-pipeline"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
