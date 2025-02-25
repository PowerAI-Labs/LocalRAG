from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Request, File, UploadFile
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import uuid4
import os
import aiofiles


from ..core import logger, settings
from ..services import BatchProcessor
from .models import (
    BatchProcessingRequest, 
    BatchProcessingStatus,
    BatchStatus,
    BatchProcessingSettings
)
from .rate_limiter import check_rate_limit
from ..services.batch_processor import BatchProcessor
# Create router for batch processing endpoints
router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

# Dependency to get batch processor instance
from fastapi import Depends
from ..services.batch_processor import BatchProcessor
from ..services.rag_engine import EnhancedRAGEngine

# Global instances
_rag_engine = None
_batch_processor = None

def get_batch_processor() -> BatchProcessor:
    """
    Dependency to get batch processor instance.
    Creates a single shared RAG engine and BatchProcessor.
    """
    global _rag_engine, _batch_processor
    
    try:
        if _batch_processor is None:
            # Initialize RAG engine if not exists
            if _rag_engine is None:
                logger.info("Creating shared RAG engine instance")
                _rag_engine = EnhancedRAGEngine()
                logger.info("RAG engine initialized successfully")
            
            # Create single BatchProcessor instance
            logger.info("Creating shared BatchProcessor instance")
            _batch_processor = BatchProcessor(_rag_engine)
            
        return _batch_processor
        
    except Exception as e:
        logger.error(f"Error initializing batch processor: {str(e)}")
        raise

@router.post("/upload")
async def upload_batch_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """Upload multiple files for batch processing"""
    request_id = str(uuid4())
    temp_paths = []
    
    try:
        # Create temp directory if it doesn't exist
        os.makedirs(settings.BATCH_TEMP_DIR, exist_ok=True)
        
        logger.info(f"[{request_id}] Starting batch upload of {len(files)} files")
        
        # Save each file to temp storage
        for file in files:
            try:
                # Validate file type and size
                allowed_types = {
                    'application/pdf', 'text/plain', 
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/svg+xml',
                    'text/csv', 'application/vnd.ms-excel',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'application/xml', 'text/xml'
                }
                
                if file.content_type not in allowed_types:
                    raise ValueError(f"Unsupported file type: {file.content_type}")
                
                # Generate temp path
                temp_path = os.path.join(
                    settings.BATCH_TEMP_DIR, 
                    f"upload_{request_id}_{file.filename}"
                )
                
                # Save file
                async with aiofiles.open(temp_path, 'wb') as f:
                    while chunk := await file.read(8192):
                        await f.write(chunk)
                
                temp_paths.append(temp_path)
                logger.info(f"[{request_id}] Saved {file.filename} to {temp_path}")
                
            except Exception as e:
                logger.error(f"[{request_id}] Error saving {file.filename}: {str(e)}")
                # Clean up any saved files
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing {file.filename}: {str(e)}"
                )
        
        # Create batch processing request
        batch_settings = BatchProcessingSettings(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            max_tokens_per_chunk=2000,
            language='english',
            ocr_enabled=True,
            image_processing_enabled=True
        )
        
        # Create metadata for batch request
        metadata = {
            'request_id': request_id,
            'original_filenames': [f.filename for f in files],
            'content_types': [f.content_type for f in files],
            'temp_paths': temp_paths,
            'timestamp': datetime.now().isoformat()
        }
        
        batch_request = BatchProcessingRequest(
            files=temp_paths,
            settings=batch_settings,
            priority=1,
            metadata=metadata
        )
        
        # Submit batch job
        batch_id = await batch_processor.submit_batch(batch_request)
        
        # Add processing to background tasks
        background_tasks.add_task(
            batch_processor._process_batch,
            batch_id,
            batch_request
        )
        
        # Add cleanup task for each temp file
        for temp_path in temp_paths:
            background_tasks.add_task(
                batch_processor.cleanup_temp_file,
                temp_path,
                batch_id
            )
        
        logger.info(f"[{request_id}] Successfully submitted batch {batch_id} with {len(files)} files")
        
        return {
            "message": "Files submitted for batch processing",
            "batch_id": batch_id,
            "request_id": request_id,
            "files_uploaded": len(temp_paths),
            "files": [
                {
                    "filename": f.filename,
                    "content_type": f.content_type,
                    "temp_path": tp
                }
                for f, tp in zip(files, temp_paths)
            ],
            "status": BatchStatus.QUEUED,
            "batch_status_endpoint": f"/api/v1/batch/{batch_id}/status"
        }
        
    except Exception as e:
        # Clean up temp files on error
        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass
        
        logger.error(f"[{request_id}] Batch upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "request_id": request_id
            }
        )

@router.post("/submit", response_model=Dict[str, str])
async def submit_batch_process(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Submit a batch processing request for existing files.
    Primarily used for system integration and retry scenarios.
    """
    request_id = str(uuid4())
    logger.info(f"[{request_id}] Received batch processing request")
    
    try:
        # Validate files exist
        for file_path in request.files:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
        # Submit batch job
        batch_id = await batch_processor.submit_batch(request)
        
        # Start processing in background
        background_tasks.add_task(
            batch_processor._process_batch,
            batch_id,
            request
        )
        
        logger.info(f"[{request_id}] Submitted batch job {batch_id}")
        return {"batch_id": batch_id}
        
    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[{request_id}] Error submitting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/{batch_id}", response_model=BatchProcessingStatus)
async def get_batch_status(
    batch_id: str = Path(..., description="The ID of the batch job"),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Get the current status of a batch processing job.
    """
    try:
        status = await batch_processor.get_batch_status(batch_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting batch status for {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{batch_id}")
async def cancel_batch(
    batch_id: str = Path(..., description="The ID of the batch job to cancel"),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Cancel a running or queued batch process.
    """
    try:
        # Attempt to cancel the batch
        await batch_processor.cancel_batch(batch_id)
        return {"message": f"Batch {batch_id} cancelled successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error cancelling batch {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=List[BatchProcessingStatus])
async def list_batches(
    status: Optional[BatchStatus] = Query(
        None, 
        description="Filter by batch status"
    ),
    limit: int = Query(
        10, 
        ge=1, 
        le=100, 
        description="Number of batches to return"
    ),
    offset: int = Query(
        0, 
        ge=0, 
        description="Number of batches to skip"
    ),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    List batch processing jobs with optional filtering.
    """
    try:
        # Get all batches
        batches = list(batch_processor.active_batches.values())
        
        # Apply status filter if provided
        if status:
            batches = [b for b in batches if b.status == status]
            
        # Sort by start time (newest first)
        batches.sort(key=lambda x: x.started_at, reverse=True)
        
        # Apply pagination
        batches = batches[offset:offset + limit]
        
        return batches
        
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{batch_id}/errors", response_model=List[Dict[str, str]])
async def get_batch_errors(
    batch_id: str = Path(..., description="The ID of the batch job"),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Get detailed error information for a batch job.
    """
    try:
        status = await batch_processor.get_batch_status(batch_id)
        return status.errors
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting batch errors for {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{batch_id}/retry")
async def retry_batch(
    batch_id: str = Path(..., description="The ID of the batch job to retry"),
    failed_only: bool = Query(
        True, 
        description="Only retry failed files"
    ),
    batch_processor: BatchProcessor = Depends(get_batch_processor)
):
    """
    Retry a failed batch job.
    """
    try:
        # Get original batch status
        status = await batch_processor.get_batch_status(batch_id)
        
        # Can only retry failed batches
        if status.status != BatchStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail="Can only retry failed batches"
            )
        
        # Create new batch request
        if failed_only:
            # Get list of failed files from error records
            failed_files = [
                error['file'] for error in status.errors
                if 'file' in error
            ]
            if not failed_files:
                raise HTTPException(
                    status_code=400,
                    detail="No failed files to retry"
                )
            files = failed_files
        else:
            # Retry all files from original request
            files = status.metadata.get('original_files', [])
            if not files:
                raise HTTPException(
                    status_code=400,
                    detail="Original file list not available"
                )
        
        # Create new batch request
        new_request = BatchProcessingRequest(
            files=files,
            callback_url=status.metadata.get('callback_url'),
            settings=status.metadata.get('settings'),
            priority=status.metadata.get('priority', 1),
            notification_email=status.metadata.get('notification_email')
        )
        
        # Submit new batch
        new_batch_id = await batch_processor.submit_batch(new_request)
        
        return {
            "message": f"Retry initiated for batch {batch_id}",
            "new_batch_id": new_batch_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrying batch {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))