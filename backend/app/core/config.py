from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    VERSION: str = "1.0.0"
    PROJECT_NAME: str = "RAG Application"
    API_V1_STR: str = "/api/v1"
    OLLAMA_API_URL: str = "http://localhost:11434"
    
    # Model settings
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL: str = "google/vit-base-patch16-224"
    
    # FAISS settings
    TEXT_EMBEDDING_DIM: int = 384
    IMAGE_EMBEDDING_DIM: int = 768
    
    # Directory settings
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    INDEX_FILE: str = os.path.join(DATA_DIR, "enhanced_index.faiss")
    CHUNKS_FILE: str = os.path.join(DATA_DIR, "enhanced_chunks.pkl")
    
    # Chunk settings
    CHUNK_SIZE: int = 10000
    CHUNK_OVERLAP: int = 100
    
    # Batch processing settings
    BATCH_PROCESSING_SIZE_THRESHOLD: int = 30 * 1024 * 1024  # 30MB in bytes
    BATCH_PROCESSING_FILE_COUNT_THRESHOLD: int = 3  # Process as batch if more than 3 files
    BATCH_PROCESSING_MAX_WORKERS: int = 4  # Maximum concurrent file processing workers
    BATCH_PROCESSING_TIMEOUT: int = 3600  # 1 hour timeout for batch processing
    BATCH_TEMP_DIR: str = os.path.join(DATA_DIR, "batch_temp")  # Directory for temporary batch files

    # Rate limiting settings
    RATE_LIMIT_DURATION: int = 60  # seconds
    RATE_LIMIT_MAX_REQUESTS: int = 100  # requests per duration
    
    # Batch cleanup settings
    BATCH_CLEANUP_AGE: int = 24 * 3600  # Clean up batch files older than 24 hours
    BATCH_AUTO_CLEANUP: bool = True  # Automatically clean up temporary batch files
    
    def setup_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.BATCH_TEMP_DIR, exist_ok=True)

    class Config:
        case_sensitive = True

settings = Settings()