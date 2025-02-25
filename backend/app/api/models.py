from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

@dataclass
class DocumentChunk:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    chunk_type: str  # 'text' or 'image'
    page_num: Optional[int] = None

class Query(BaseModel):
    question: str
    context_window: int = 10000
    model: Optional[str] = "deepseek-r1:8b"
    timeout: Optional[int] = 300
    max_messages: Optional[int] = 10
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class SearchFilters(BaseModel):
    """Advanced search filters"""
    date_range: Optional[Dict[str, datetime]] = None
    document_types: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, str]] = None
    content_type: Optional[str] = None
    min_relevance: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

class SortOptions(BaseModel):
    """Sort options for search results"""
    field: str
    order: str = "desc"  # "asc" or "desc"

class EnhancedQuery(BaseModel):
    """Enhanced query model with advanced search options"""
    question: str
    filters: Optional[SearchFilters] = None
    sort: Optional[SortOptions] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    semantic_search: bool = True
    fuzzy_matching: bool = False
    include_facets: bool = True
    context_window: int = 10000
    model: Optional[str] = "deepseek-r1:8b"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class SearchResult(BaseModel):
    """Individual search result"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_type: str
    highlights: Optional[List[Dict[str, str]]] = None
    page_num: Optional[int] = None

class SearchFacets(BaseModel):
    """Faceted search results"""
    document_types: Dict[str, int]
    content_types: Dict[str, int]
    date_ranges: Dict[str, int]
    metadata_facets: Dict[str, Dict[str, int]]

class EnhancedSearchResponse(BaseModel):
    """Enhanced search response"""
    results: List[SearchResult]
    facets: Optional[SearchFacets] = None
    total_results: int
    page: int
    total_pages: int
    query_expansion_used: bool = False
    expanded_terms: Optional[List[str]] = None
    metadata: Dict[str, Any]

# Batch Processing Models
class BatchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchFileMetadata(BaseModel):
    filename: str
    file_type: str
    content_type: str
    size: int
    created_at: datetime = Field(default_factory=datetime.now)
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    chunks_generated: int = 0
    error: Optional[str] = None
    status: BatchStatus = BatchStatus.QUEUED

class BatchProcessingSettings(BaseModel):
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    embedding_model: Optional[str] = None
    max_tokens_per_chunk: Optional[int] = 2000
    language: Optional[str] = "english"
    ocr_enabled: Optional[bool] = True
    image_processing_enabled: Optional[bool] = True

class BatchProcessingRequest(BaseModel):
    files: List[str]
    settings: Optional[BatchProcessingSettings] = None
    callback_url: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)
    notification_email: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BatchProcessingError(BaseModel):
    error_code: str
    message: str
    file: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[Dict[str, Any]] = None

class BatchProcessingStatus(BaseModel):
    batch_id: str
    status: BatchStatus
    total_files: int
    processed_files: int
    failed_files: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    files: Dict[str, BatchFileMetadata] = {}
    errors: List[BatchProcessingError] = []
    settings: Optional[BatchProcessingSettings] = None
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class BatchProcessingResult(BaseModel):
    batch_id: str
    status: BatchStatus
    total_chunks: int
    processing_time: float
    files_processed: List[str]
    files_failed: List[str]
    errors: List[BatchProcessingError]
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True