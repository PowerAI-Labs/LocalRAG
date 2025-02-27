from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from uuid import uuid4
import os
from PIL import Image
import io
from ..services.rag_engine import EnhancedRAGEngine

from ..core import logger
from ..api.models import (
    BatchProcessingRequest, 
    BatchProcessingStatus,
    BatchStatus,
    DocumentChunk,
    BatchFileMetadata,
    BatchProcessingSettings,
    BatchProcessingError,
    
)

class BatchProcessor:
    """
    Handles asynchronous batch processing of documents.
    
    Features:
    - Priority-based queue processing
    - Progress tracking
    - Webhook callbacks
    - Error handling and recovery
    - File type detection and routing
    """
    
    """Handles asynchronous batch processing of documents."""
    
    # Class-level storage for batch processor instances
    _instances = {}
    
    def __new__(cls, rag_engine):
        """Implement singleton pattern based on RAG engine instance."""
        rag_engine_id = id(rag_engine)
        if rag_engine_id not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[rag_engine_id] = instance
        return cls._instances[rag_engine_id]
    
    def __init__(self, rag_engine):
        """Initialize the batch processor with RAG engine."""
        self.rag_engine = rag_engine  # Store the RAG engine instance
        self.active_batches = {}
        self.processing_queue = asyncio.PriorityQueue()
        self.batch_locks = {}
        self.is_running = False
        logger.info("BatchProcessor initialized with RAG engine")
        
    async def start(self):
        """Start the batch processor worker."""
        if self.is_running:
            return
            
        self.is_running = True
        asyncio.create_task(self._process_queue())
        logger.info("Batch processor started")
        
    async def stop(self):
        """Stop the batch processor worker."""
        self.is_running = False
        logger.info("Batch processor stopped")
        
    async def submit_batch(self, request: BatchProcessingRequest) -> str:
        """
        Submit a new batch processing request.
        
        Args:
            request: BatchProcessingRequest with files and settings
            
        Returns:
            str: Unique batch ID
        """
        batch_id = str(uuid4())
        
        # Validate files exist with more detailed logging
        for file_path in request.files:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise ValueError(f"File not found: {file_path}")
            else:
                logger.info(f"Validated file for batch processing: {file_path}")
        
        # Create batch status
        status = BatchProcessingStatus(
            batch_id=batch_id,
            status=BatchStatus.QUEUED,
            total_files=len(request.files),
            processed_files=0,
            failed_files=0,
            started_at=datetime.now(),
            metadata={
                'priority': request.priority,
                'callback_url': request.callback_url,
                'settings': request.settings,
                'notification_email': request.notification_email
            }
        )
        
        # Store batch status and create lock
        self.active_batches[batch_id] = status
        self.batch_locks[batch_id] = asyncio.Lock()
        
        # Add to priority queue (lower number = higher priority)
        await self.processing_queue.put((
            request.priority,    # Priority number for sorting
            datetime.now().timestamp(),  # Secondary sorting by submission time
            batch_id,           # Use batch_id directly instead of dict
            request            # Request object
        ))
        
        # Enhanced logging
        logger.info(f"Submitted batch {batch_id}")
        logger.info(f"Batch details:")
        logger.info(f"  - Total files: {len(request.files)}")
        logger.info(f"  - Priority: {request.priority}")
        logger.info(f"  - Callback URL: {request.callback_url or 'Not configured'}")
        logger.info(f"  - Notification Email: {request.notification_email or 'Not configured'}")
        
        return batch_id
        
    async def _process_queue(self):
        """Process items from the priority queue."""
        while self.is_running:
            try:
                # Log that we're about to retrieve a batch from the queue
                logger.info("Waiting for next batch in processing queue")
                
                # Get next batch from priority queue - now unpacking 4 items
                priority, timestamp, batch_id, request = await self.processing_queue.get()
                
                # Log detailed batch information
                logger.info(f"Retrieved batch {batch_id} from queue")
                logger.debug(f"Batch priority: {priority}")
                logger.debug(f"Number of files in batch: {len(request.files)}")
                logger.debug(f"Batch processing settings: {request.settings}")
                
                # Log the start of batch processing
                logger.info(f"Starting to process batch {batch_id}")
                
                # Create a task for batch processing
                batch_task = asyncio.create_task(
                    self._process_batch(batch_id, request),
                    name=f"batch-processor-{batch_id}"
                )
                
                # Log task creation
                logger.info(f"Created processing task for batch {batch_id}")
                
                # Optional: Add error handling for the batch processing task
                def batch_task_callback(fut):
                    try:
                        fut.result()  # This will raise any unhandled exceptions
                        logger.info(f"Batch {batch_id} processing completed successfully")
                    except Exception as e:
                        logger.error(f"Batch {batch_id} processing failed: {str(e)}", exc_info=True)
                
                batch_task.add_done_callback(batch_task_callback)
                
            except asyncio.CancelledError:
                logger.warning("Batch processing queue was cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in batch processing queue: {str(e)}", exc_info=True)
            finally:
                # Always mark the task as done, even if there was an error
                try:
                    self.processing_queue.task_done()
                    logger.debug("Marked current queue task as done")
                except Exception as final_error:
                    logger.error(f"Error in queue task_done: {str(final_error)}")
                
    async def _process_batch(self, batch_id: str, request: BatchProcessingRequest):
        """
        Process a batch of files.
        
        Features:
        - Concurrent file processing
        - Progress tracking
        - Error handling per file
        - Callback notifications
        """
        try:
            async with self.batch_locks[batch_id]:
                status = self.active_batches[batch_id]
                status.status = BatchStatus.PROCESSING
                logger.info(f"Processing batch {batch_id}")
                
                # Process files concurrently
                tasks = []
                for file_path in request.files:
                    task = asyncio.create_task(
                        self._process_file(batch_id, file_path, request.settings)
                    )
                    tasks.append(task)
                
                # Wait for all files to be processed
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        status.failed_files += 1
                        status.errors.append({
                            'error': str(result)
                        })
                    else:
                        status.processed_files += 1
                
                # Update final status
                status.status = (
                    BatchStatus.COMPLETED 
                    if status.failed_files == 0 
                    else BatchStatus.FAILED
                )
                status.completed_at = datetime.now()
                
                # Send notifications
                await self._send_notifications(batch_id)
                
                logger.info(
                    f"Completed batch {batch_id}: "
                    f"{status.processed_files} processed, "
                    f"{status.failed_files} failed"
                )
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {str(e)}")
            status = self.active_batches[batch_id]
            status.status = BatchStatus.FAILED
            status.errors.append({'error': str(e)})
            raise
            
    async def _process_file(
        self,
        batch_id: str,
        file_path: str,
        settings: Optional[BatchProcessingSettings] = None
    ) -> None:
        """Process a single file within a batch."""
        # Retrieve the batch status
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        status = self.active_batches[batch_id]
        filename = os.path.basename(file_path)
        
        # Log detailed file information at the start of processing
        logger.info(f"Processing file {filename} in batch {batch_id}")
        logger.debug(f"File path: {file_path}")
        logger.debug(f"File exists: {os.path.exists(file_path)}")
        
        # Initialize file metadata
        file_metadata = BatchFileMetadata(
            filename=filename,
            file_type=self._get_file_type(file_path),
            content_type=self._get_content_type(file_path),
            size=os.path.getsize(file_path),
            status=BatchStatus.PROCESSING,
            processing_started=datetime.now()
        )
        
        # Log file metadata details
        logger.info(f"File metadata - Type: {file_metadata.file_type}, "
                    f"Content Type: {file_metadata.content_type}, "
                    f"Size: {file_metadata.size} bytes")
        
        # Ensure files dictionary exists in status
        if not hasattr(status, 'files'):
            status.files = {}
        status.files[filename] = file_metadata
        
        try:
            # Detailed logging for file reading
            logger.debug("Attempting to read file content")
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            logger.info(f"Successfully read {len(content)} bytes from {filename}")
            
            # Detailed logging for metadata creation
            metadata = {
                'filename': filename,
                'file_type': file_metadata.file_type,
                'content_type': file_metadata.content_type,
                'batch_id': batch_id,
                'processing_settings': settings.dict() if settings else {},
                'processed_at': datetime.now().isoformat()
            }
            logger.debug(f"Metadata created: {metadata}")
            
            # Apply settings logging
            if settings:
                logger.info(f"Processing settings - Chunk Size: {settings.chunk_size}, "
                            f"Chunk Overlap: {settings.chunk_overlap}")
            
            # Log routing details
            logger.debug(f"Routing processing for file type: {file_metadata.file_type}")
            chunks = await self._route_processing(file_metadata.file_type, content, metadata)
            
            # Chunks generation logging
            logger.info(f"Generated {len(chunks)} chunks for {filename}")
            
            # Update metadata
            file_metadata.chunks_generated = len(chunks)
            file_metadata.status = BatchStatus.COMPLETED
            file_metadata.processing_completed = datetime.now()
            
            # Add chunks to RAG engine
            logger.debug("Adding chunks to RAG engine")
            await self.rag_engine.add_chunks(chunks)

            # Save the updated chunks and indices
            await self.rag_engine._save_data()
            
            # Just update metadata status, don't increment counter
            file_metadata.status = BatchStatus.COMPLETED
            file_metadata.processing_completed = datetime.now()
            logger.info(f"Successfully processed {filename} in batch {batch_id}")
            
        except Exception as e:
            error_msg = f"Error processing file {filename}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
            
    async def _route_processing(
        self, 
        file_type: str, 
        content: bytes,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Route file processing based on type."""
        # Check if rag_engine is properly initialized
        if not hasattr(self, 'rag_engine'):
            logger.error("RAG engine not initialized")
            raise RuntimeError("BatchProcessor not properly initialized with RAG engine")

        try:
            if file_type == 'pdf':
                return await self.rag_engine.process_pdf(content, metadata)
            elif file_type == 'docx':
                return await self.rag_engine.process_docx(content, metadata)
            elif file_type == 'csv':
                return await self.rag_engine.process_csv(content, metadata)
            elif file_type in ['xlsx', 'xls']:
                return await self.rag_engine.process_excel(content, metadata)
            elif file_type == 'xml':
                return await self.rag_engine.process_xml(content, metadata)
            elif file_type in ['jpg', 'jpeg', 'png', 'gif']:
                image = Image.open(io.BytesIO(content))
                return [await self.rag_engine.process_image(image, metadata)]
            else:
                text = content.decode('utf-8')
                return await self.rag_engine.process_text(text, metadata)
        except Exception as e:
            logger.error(f"Error in _route_processing for {file_type}: {str(e)}")
            raise
            
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from path."""
        extension = os.path.splitext(file_path)[1].lower()
        return extension[1:] if extension else 'unknown'
        
    async def _send_notifications(self, batch_id: str):
        """Send notifications about batch completion."""
        status = self.active_batches[batch_id]
        
        # Send webhook callback if configured
        if status.metadata.get('callback_url'):
            await self._send_callback(
                status.metadata['callback_url'],
                status
            )
            
        # Send email notification if configured
        if status.metadata.get('notification_email'):
            await self._send_email_notification(
                status.metadata['notification_email'],
                status
            )
            
    async def _send_callback(self, callback_url: str, status: BatchProcessingStatus):
        """Send webhook callback with batch status."""
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    callback_url,
                    json=status.dict(),
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                logger.info(f"Sent callback for batch {status.batch_id}")
        except Exception as e:
            logger.error(f"Callback error for batch {status.batch_id}: {str(e)}")
            
    async def _send_email_notification(self, email: str, status: BatchProcessingStatus):
        """Send email notification about batch completion."""
        try:
            # TODO: Implement email notification
            # This would integrate with your email service
            pass
        except Exception as e:
            logger.error(f"Email notification error for batch {status.batch_id}: {str(e)}")


            
    async def get_batch_status(self, batch_id: str) -> BatchProcessingStatus:
        """Get current status of a batch job."""
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        return self.active_batches[batch_id]
        
    async def cleanup_old_batches(self, max_age_hours: int = 24):
        """Clean up completed batches older than specified age."""
        now = datetime.now()
        to_remove = []
        
        for batch_id, status in self.active_batches.items():
            if status.completed_at:
                age = now - status.completed_at
                if age.total_seconds() > max_age_hours * 3600:
                    to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.active_batches[batch_id]
            if batch_id in self.batch_locks:
                del self.batch_locks[batch_id]

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type from file extension."""
        extension = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.xml': 'application/xml',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }
        return content_types.get(extension, 'application/octet-stream')

    async def cancel_batch(self, batch_id: str) -> None:
        """Cancel a running batch job."""
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
            
        status = self.active_batches[batch_id]
        if status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
            raise ValueError(f"Cannot cancel batch in {status.status} status")
        
        status.status = BatchStatus.CANCELLED
        status.completed_at = datetime.now()
        status.errors.append(
            BatchProcessingError(
                error_code="BATCH_CANCELLED",
                message="Batch processing cancelled by user"
            )
        )
    
   

    async def cleanup_temp_file(self, temp_path: str, batch_id: str):
        """
        Cleanup temporary file after batch processing is complete.
        
        Args:
            temp_path: Path to temporary file
            batch_id: ID of the batch job to monitor
        """
        try:
            # Wait for batch to complete
            while True:
                status = await self.get_batch_status(batch_id)
                if status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
                    break
                await asyncio.sleep(5)  # Check every 5 seconds
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file for batch {batch_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up temporary file for batch {batch_id}: {str(e)}")

