from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import time
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from uuid import uuid4
import logging
from ..core import settings, logger
from ..services import EnhancedRAGEngine, query_ollama, BatchProcessor
from ..utils.helpers import read_file_content, validate_file_size
from ..prompts import PromptHandler
from .models import Query
import io
import aiofiles
import os
from ..api.models import DocumentChunk, Query, SearchFilters, SortOptions, EnhancedQuery, SearchResult,SearchFacets,EnhancedSearchResponse, BatchStatus, BatchFileMetadata, BatchProcessingSettings, BatchProcessingRequest,BatchProcessingError, BatchProcessingStatus, BatchProcessingResult
from ..core import settings, logger
from .batch_endpoints import router as batch_router
from fastapi import BackgroundTasks  # Add this to your imports

# Initialize FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine and prompt handler
rag_engine = EnhancedRAGEngine()
prompt_handler = PromptHandler()

# Add batch router here
app.include_router(batch_router)

# Rate limiting settings
RATE_LIMIT_DURATION = 60  # seconds
MAX_REQUESTS = 100  # requests per duration
RATE_LIMIT_STORE = defaultdict(list)  # Store for rate limiting

# Request validation
async def validate_file_request(file: UploadFile = File(...)) -> UploadFile:
    """Validate file upload request."""
    allowed_types = {
        'application/pdf', 'text/plain', 
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/svg+xml',
        'text/csv', 'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/xml', 'text/xml'
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # Check file size (50MB limit)
    MAX_SIZE = 50 * 1024 * 1024
    file_size = 0
    chunk_size = 8192
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > MAX_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File too large (max 50MB)"
            )
    
    await file.seek(0)
    return file

async def check_rate_limit(request: Request):
    """Rate limiting middleware."""
    client_ip = request.client.host
    now = time.time()
    
    # Clean old requests
    RATE_LIMIT_STORE[client_ip] = [
        timestamp for timestamp in RATE_LIMIT_STORE[client_ip]
        if timestamp > now - RATE_LIMIT_DURATION
    ]
    
    if len(RATE_LIMIT_STORE[client_ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests"
        )
    
    RATE_LIMIT_STORE[client_ip].append(now)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        error_id = str(uuid4())
        
        # Log the error with exception information
        logger.error(
            f"Unhandled error {error_id}: {str(e)}",
            exc_info=True  # Now supported by our ConsoleLogger
        )
        
        # Return a JSON response with error details
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_id": error_id,
                "error_type": e.__class__.__name__,
                "error_message": str(e)
            }
        )





@app.post("/upload")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_request)
    
):
    """Enhanced file upload endpoint with batch processing for large files."""
    await check_rate_limit(request)
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    
    try:
        # Check file size before reading content
        file_size = 0
        chunk_size = 8192
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > settings.BATCH_PROCESSING_SIZE_THRESHOLD:
                break
        await file.seek(0)
        
        # Create base metadata
        metadata = {
            'filename': file.filename,
            'content_type': file.content_type,
            'timestamp': str(datetime.now()),
            'request_id': request_id,
            'file_size': file_size
        }
        
        logger.info(f"[{request_id}] Processing file: {file.filename} (Size: {file_size} bytes)")
        
        # Determine if batch processing is needed
        use_batch = file_size > settings.BATCH_PROCESSING_SIZE_THRESHOLD
        
        if use_batch:
            logger.info(f"[{request_id}] Using batch processing due to file size")
            
            # Create temporary file for batch processing
            temp_path = os.path.join(
                settings.BATCH_TEMP_DIR, 
                f"upload_{request_id}_{file.filename}"
            )
            
            try:
                # Ensure temp directory exists
                os.makedirs(settings.BATCH_TEMP_DIR, exist_ok=True)
                
                # Save file to temp location
                async with aiofiles.open(temp_path, 'wb') as f:
                    await file.seek(0)
                    while chunk := await file.read(chunk_size):
                        await f.write(chunk)
                
                # Initialize batch processing settings
                batch_settings = BatchProcessingSettings(
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP,
                    max_tokens_per_chunk=2000,
                    language='english',
                    ocr_enabled=True,
                    image_processing_enabled=True
                )
                
                # Create batch processing request
                batch_request = BatchProcessingRequest(
                    files=[temp_path],
                    settings=batch_settings,
                    priority=1,  # High priority for single file
                    metadata={
                        **metadata,
                        'original_filename': file.filename,
                        'temp_path': temp_path
                    }
                )
                
                # Create BatchProcessor instance and submit batch job
                batch_processor = BatchProcessor(rag_engine)  # Pass the rag_engine here
                batch_id = await batch_processor.submit_batch(batch_request)
                
                # Add batch processing to background tasks
                background_tasks.add_task(
                    batch_processor._process_batch,
                    batch_id,
                    batch_request
                )
                
                # Add cleanup task for temporary file
                background_tasks.add_task(
                    batch_processor.cleanup_temp_file,
                    temp_path,
                    batch_id
                )
                
                return {
                    "message": "File submitted for batch processing",
                    "request_id": request_id,
                    "batch_id": batch_id,
                    "metadata": {
                        **metadata,
                        "batch_processed": True,
                        "submission_time": datetime.now().isoformat()
                    },
                    "status": "queued",
                    "batch_status_endpoint": f"/api/v1/batch/{batch_id}/status"
                }
                
            except Exception as e:
                # Clean up temp file if batch submission fails
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up temporary file: {str(cleanup_error)}")
                raise
                
        else:
            # Regular processing for smaller files
            logger.info(f"[{request_id}] Using standard processing")
            
            # Read file content
            content = await read_file_content(file)
            chunks = []
            filename_lower = file.filename.lower()
            
            # Process file based on type
            if filename_lower.endswith('.pdf'):
                chunks = await rag_engine.process_pdf(content, metadata)
            elif filename_lower.endswith('.docx'):
                chunks = await rag_engine.process_docx(content, metadata)
            elif filename_lower.endswith('.csv') or file.content_type == 'text/csv':
                chunks = await rag_engine.process_csv(content, metadata)
            elif filename_lower.endswith(('.xlsx', '.xls')) or file.content_type in [
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                'application/vnd.ms-excel'
            ]:
                chunks = await rag_engine.process_excel(content, metadata)
            elif filename_lower.endswith('.xml') or file.content_type in ['application/xml', 'text/xml']:
                chunks = await rag_engine.process_xml(content, metadata)
            elif file.content_type.startswith('image/'):
                from PIL import Image
                image = Image.open(io.BytesIO(content))
                chunks = [await rag_engine.process_image(image, metadata)]
            else:
                text = content.decode('utf-8')
                chunks = await rag_engine.process_text(text, metadata)
            
            # Add chunks to indices
            await rag_engine.add_chunks(chunks)
            
            # Save the updated chunks and indices
            await rag_engine._save_data()
            
            completion_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "message": "File processed successfully",
                "request_id": request_id,
                "chunks": len(chunks),
                "metadata": {
                    **metadata,
                    "batch_processed": False
                },
                "processing_time": completion_time
            }
        
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "request_id": request_id,
                "processing_mode": "batch" if use_batch else "standard"
            }
        )

   
    
@app.post("/query")
async def query_documents(
    request: Request,
    query: Query
):
    """Enhanced query endpoint with validation and rate limiting."""
    await check_rate_limit(request)
    request_id = str(uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] Processing query: {query.question}")
        logger.info(f"[{request_id}] Using model: {query.model}")
        
        # Validate query
        if not query.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Handle no context case
        if not rag_engine.chunks:
            logger.info(f"[{request_id}] No context available - using direct query")
            response = await query_ollama(query.question, query.dict())
            return {
                "request_id": request_id,
                "response": response,
                "context": None,
                "results": []
            }
        
        # Use enhanced search
        search_results = await rag_engine.search(query.question)
        
        # Use prompt handler to create appropriate prompt
        prompt = prompt_handler.create_prompt(
            context=search_results['formatted_context'],
            query=query.question,
            query_type=search_results['query_intent'],
            metadata={
                'request_id': request_id,
                **search_results
            }
        )
        
        logger.info(f"[{request_id}] Query intent: {search_results['query_intent']}")
        logger.info(f"[{request_id}] Sending query to LLM")
        
        response = await query_ollama(prompt, query.dict())
        logger.info(f"[{request_id}] Received response from LLM")
        
        return {
            "request_id": request_id,
            "response": response,
            "context": search_results['formatted_context'],
            "query_intent": search_results['query_intent'],
            "results": search_results['chunks'],
            "metadata": {
                "chunks_searched": len(rag_engine.chunks),
                "relevant_chunks": len(search_results['chunks']),
                "processing_time": time.time() - start_time
            }
        }
        
    except Exception as e:
        logger.error(f"[{request_id}] Query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "request_id": request_id
            }
        )
    
@app.post("/enhanced-search", response_model=EnhancedSearchResponse)
async def enhanced_search(
    request: Request,
    query: EnhancedQuery
):
    """
    Enhanced search endpoint with advanced features.
    
    Example request:
    ```json
    {
        "question": "What are the key points about machine learning?",
        "filters": {
            "document_types": ["pdf", "docx"],
            "date_range": {
                "start": "2024-01-01T00:00:00",
                "end": "2024-02-14T23:59:59"
            },
            "content_type": "text",
            "min_relevance": 0.5
        },
        "sort": {
            "field": "score",
            "order": "desc"
        },
        "page": 1,
        "page_size": 10,
        "semantic_search": true,
        "fuzzy_matching": true,
        "include_facets": true
    }
    ```
    
    Returns enhanced search results with facets, highlights, and metadata.
    """
    await check_rate_limit(request)
    request_id = str(uuid4())
    start_time = time.time()

    try:
        logger.info(f"[{request_id}] Processing enhanced search: {query.question}")
        logger.info(f"[{request_id}] Using model: {query.model}")
        
        # Validate query
        if not query.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Handle no context case
        if not rag_engine.chunks:
            logger.info(f"[{request_id}] No context available - using direct query")
            response = await query_ollama(query.question, query.dict())
            return EnhancedSearchResponse(
                results=[],
                total_results=0,
                page=1,
                total_pages=0,
                metadata={
                    'no_context': True,
                    'direct_query': True,
                    'processing_time': time.time() - start_time
                }
            )
        
        # Use enhanced search
        search_response = await rag_engine.enhanced_search(query)
        
        # If results found, generate response using LLM
        if search_response.results:
            # Create context from top results
            context = "\n\n".join([
                f"Context {i+1}:\n{result.text}" 
                for i, result in enumerate(search_response.results[:3])
            ])
            
            # Use prompt handler to create appropriate prompt
            prompt = prompt_handler.create_prompt(
                context=context,
                query=query.question,
                query_type='enhanced_search',
                metadata={
                    'request_id': request_id,
                    'expanded_terms': search_response.expanded_terms
                }
            )
            
            logger.info(f"[{request_id}] Sending enhanced query to LLM")
            
            # Get LLM response
            llm_response = await query_ollama(prompt, {
                'model': query.model,
                'temperature': query.temperature,
                'top_p': query.top_p,
                'context_window': query.context_window
            })
            
            # Add LLM response to search metadata
            search_response.metadata['llm_response'] = llm_response
        
        logger.info(
            f"[{request_id}] Enhanced search completed. "
            f"Found {search_response.total_results} results "
            f"in {time.time() - start_time:.2f}s"
        )
        
        return search_response
        
    except Exception as e:
        logger.error(f"[{request_id}] Enhanced search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "request_id": request_id
            }
        )

@app.post("/clear-context")
async def clear_context(request: Request):
    """Clear all stored documents with rate limiting."""
    await check_rate_limit(request)
    try:
        await rag_engine.clear()
        return {"message": "Context cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context-status")
async def get_context_status(request: Request):
    """Get current context status with rate limiting."""
    await check_rate_limit(request)
    try:
        return await rag_engine.get_status()
    except Exception as e:
        logger.error(f"Error getting context status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(request: Request):
    """List available models with rate limiting."""
    await check_rate_limit(request)
    from ..services import list_available_models
    return await list_available_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION
    }

@app.get("/test-logging")
async def test_logging():
    """Test endpoint to verify logging is working."""
    print("=== Starting Logging Test ===")
    
    print("Trying INFO message...")
    logger.info("TEST - This is an INFO message")
    
    print("Trying WARNING message...")
    logger.warning("TEST - This is a WARNING message")
    
    print("Trying ERROR message...")
    logger.error("TEST - This is an ERROR message")
    
    print("=== Logging Test Complete ===")
    
    # Also try with root logger
    root_logger = logging.getLogger()
    root_logger.info("Root logger INFO test")
    
    return {"message": "Logging test completed"}



@app.exception_handler(Exception)
async def enhanced_error_handler(request: Request, exc: Exception):
    error_id = str(uuid4())
    
    # Log the error with full traceback
    logger.error(
        f"Enhanced error {error_id}: {str(exc)}",
        exc_info=True
    )
    
    # Prepare error response
    error_response = {
        "detail": "Internal server error",
        "error_id": error_id,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc)
    }
    
    # Add additional context for specific error types
    if isinstance(exc, HTTPException):
        error_response["detail"] = exc.detail
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    return JSONResponse(
        status_code=500,
        content=error_response
    )