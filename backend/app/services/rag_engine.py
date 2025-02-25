from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import numpy as np
import faiss
import pickle
import os
import io
import time
import pytesseract
import fitz  # PyMuPDF
import docx
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import uuid4
import warnings
import pandas as pd
import xmltodict
import json
from datetime import datetime
from collections import Counter
import shutil
import glob
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core import settings, logger
from ..api.models import DocumentChunk
from ..utils.helpers import chunk_text

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from .search_enhancer import SearchEnhancer
from ..api.models import DocumentChunk, Query, SearchFilters, SortOptions, EnhancedQuery, SearchResult,SearchFacets,EnhancedSearchResponse


class EnhancedRAGEngine:
    """
    Enhanced RAG (Retrieval-Augmented Generation) Engine with support for multiple document types
    and improved search capabilities.
    
    Features:
    - Multi-format document processing (PDF, DOCX, CSV, Excel, XML, Images)
    - Enhanced search with content-type awareness
    - Automatic backup and recovery
    - Chunk validation and data integrity checks
    - Error resilience with retry mechanisms
    """

    def __init__(self):
        """Initialize the RAG engine with text and image models."""
        logger.info("Initializing RAG engine")
        
        # Set Tesseract path for Windows
        if os.name == 'nt':
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize text embedding model
        self.text_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)
        
        # Initialize image embedding model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.image_processor = ViTImageProcessor.from_pretrained(
                settings.IMAGE_EMBEDDING_MODEL,
                ignore_mismatched_sizes=True
            )
            self.image_model = ViTModel.from_pretrained(
                settings.IMAGE_EMBEDDING_MODEL,
                add_pooling_layer=True,
                ignore_mismatched_sizes=True
            )
        self.image_model.eval()
        
        # Initialize FAISS index
        self.text_dim = settings.TEXT_EMBEDDING_DIM
        self.text_index = faiss.IndexFlatL2(self.text_dim)
        
        # Storage for chunks
        self.chunks: List[DocumentChunk] = []
        
        logger.info(f"Initialized with embedding dimension: {self.text_dim}")
        
        # Load existing data if available
        self._load_data()
        
        logger.info("RAG engine initialized successfully")

    def _load_data(self):
        """
        Load existing indices and chunks from disk.
        
        This method:
        1. Loads the FAISS index if it exists
        2. Loads and validates chunk data
        3. Converts loaded data to DocumentChunk objects
        4. Handles loading errors gracefully
        """
        try:
            if os.path.exists(settings.INDEX_FILE):
                # Load FAISS index
                self.text_index = faiss.read_index(settings.INDEX_FILE)
                
                # Load chunks with error handling
                if os.path.exists(settings.CHUNKS_FILE):
                    try:
                        with open(settings.CHUNKS_FILE, 'rb') as f:
                            loaded_data = pickle.load(f)
                            # Convert loaded data to DocumentChunk objects
                            self.chunks = [
                                DocumentChunk(
                                    id=chunk.get('id', str(uuid4())),
                                    text=chunk.get('text', ''),
                                    embedding=np.array(chunk.get('embedding', [])),
                                    metadata=chunk.get('metadata', {}),
                                    chunk_type=chunk.get('chunk_type', 'text'),
                                    page_num=chunk.get('page_num', None)
                                ) if not isinstance(chunk, DocumentChunk) else chunk
                                for chunk in loaded_data
                            ]
                    except (pickle.UnpicklingError, AttributeError) as e:
                        logger.warning(f"Error loading chunks, starting fresh: {e}")
                        self.chunks = []
                        
                logger.info(f"Loaded {len(self.chunks)} existing chunks")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Initialize empty if loading fails
            self.text_index = faiss.IndexFlatL2(settings.TEXT_EMBEDDING_DIM)
            self.chunks = []

    async def _save_data(self):
        """
        Enhanced save functionality with better chunk handling and validation.
        
        Features:
        - Backup before saving
        - Chunk validation
        - Safe embedding conversion
        - Metadata enhancement
        - Data integrity verification
        """
        try:
            # Ensure data directory exists
            settings.setup_directories()
            
            # Backup existing files if they exist
            await self._backup_existing_data()
            
            try:
                # Save FAISS index
                faiss.write_index(self.text_index, settings.INDEX_FILE)
                logger.info(f"Saved FAISS index to: {settings.INDEX_FILE}")
                
                # Process chunks with enhanced metadata
                chunks_data = []
                for chunk in self.chunks:
                    # Validate chunk data
                    if not self._validate_chunk(chunk):
                        logger.warning(f"Skipping invalid chunk with ID: {chunk.id}")
                        continue
                        
                    # Process embeddings safely
                    try:
                        embedding_list = chunk.embedding.tolist()
                    except Exception as e:
                        logger.error(f"Error converting embedding for chunk {chunk.id}: {e}")
                        embedding_list = []
                    
                    # Create enhanced chunk data
                    chunk_data = {
                        'id': chunk.id,
                        'text': chunk.text,
                        'embedding': embedding_list,
                        'metadata': self._enhance_metadata(chunk.metadata),
                        'chunk_type': chunk.chunk_type,
                        'page_num': chunk.page_num,
                        'content_type': self._get_content_type(chunk),
                        'timestamp': datetime.now().isoformat(),
                        'version': settings.VERSION
                    }
                    
                    # Add type-specific data
                    type_specific_data = self._get_type_specific_data(chunk)
                    if type_specific_data:
                        chunk_data.update(type_specific_data)
                    
                    chunks_data.append(chunk_data)
                
                # Save chunks with compression
                with open(settings.CHUNKS_FILE, 'wb') as f:
                    pickle.dump(chunks_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Save metadata summary
                await self._save_metadata_summary(chunks_data)
                
                logger.info(f"Saved {len(chunks_data)} chunks to: {settings.CHUNKS_FILE}")
                
                # Verify saved data
                await self._verify_saved_data(chunks_data)
                
            except Exception as e:
                logger.error(f"Error during save operation: {e}")
                # Restore from backup if save failed
                await self._restore_from_backup()
                raise
                
        except Exception as e:
            logger.error(f"Critical error in save operation: {e}")
            raise

    async def _backup_existing_data(self):
        """
        Create backup of existing data files.
        
        Creates timestamped backups of:
        - FAISS index file
        - Chunks file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if os.path.exists(settings.INDEX_FILE):
                backup_index = f"{settings.INDEX_FILE}.{timestamp}.bak"
                shutil.copy2(settings.INDEX_FILE, backup_index)
                
            if os.path.exists(settings.CHUNKS_FILE):
                backup_chunks = f"{settings.CHUNKS_FILE}.{timestamp}.bak"
                shutil.copy2(settings.CHUNKS_FILE, backup_chunks)
                
            logger.info("Created backup of existing data files")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    async def _save_metadata_summary(self, chunks_data: List[Dict]):
        """
        Save summary of metadata for easy access.
        
        Creates a JSON summary containing:
        - Total chunk count
        - Chunk type distribution
        - Content type distribution
        - Timestamp
        - Version information
        """
        try:
            summary = {
                'total_chunks': len(chunks_data),
                'chunk_types': Counter(chunk['chunk_type'] for chunk in chunks_data),
                'content_types': Counter(chunk['content_type'] for chunk in chunks_data),
                'timestamp': datetime.now().isoformat(),
                'version': settings.VERSION
            }
            
            summary_file = os.path.join(os.path.dirname(settings.CHUNKS_FILE), 'chunks_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"Saved metadata summary to: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Error saving metadata summary: {e}")

    async def _verify_saved_data(self, chunks_data: List[Dict]):
        """
        Verify integrity of saved data.
        
        Checks:
        - Chunk count consistency
        - FAISS index size match
        - Data loading verification
        """
        try:
            # Verify chunks file
            with open(settings.CHUNKS_FILE, 'rb') as f:
                loaded_chunks = pickle.load(f)
                
            if len(loaded_chunks) != len(chunks_data):
                raise ValueError("Saved chunks count mismatch")
                
            # Verify FAISS index
            loaded_index = faiss.read_index(settings.INDEX_FILE)
            if loaded_index.ntotal != self.text_index.ntotal:
                raise ValueError("Saved index size mismatch")
                
            logger.info("Verified saved data integrity")
            
        except Exception as e:
            logger.error(f"Data verification failed: {e}")
            raise

    async def _restore_from_backup(self):
        """
        Restore data from backup if available.
        
        Features:
        - Finds latest backup files
        - Restores both index and chunks
        - Handles restoration errors
        """
        try:
            # Find latest backup files
            index_backups = glob.glob(f"{settings.INDEX_FILE}.*.bak")
            chunks_backups = glob.glob(f"{settings.CHUNKS_FILE}.*.bak")
            
            if index_backups and chunks_backups:
                latest_index = max(index_backups, key=os.path.getctime)
                latest_chunks = max(chunks_backups, key=os.path.getctime)
                
                # Restore files
                shutil.copy2(latest_index, settings.INDEX_FILE)
                shutil.copy2(latest_chunks, settings.CHUNKS_FILE)
                
                logger.info("Successfully restored from backup")
                
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise

    def _validate_chunk(self, chunk) -> bool:
        """
        Validate chunk data before saving.
        
        Checks:
        - Required fields presence
        - Embedding type and shape
        - Metadata structure
        """
        try:
            if not chunk.id or not chunk.text:
                return False
                
            if not isinstance(chunk.embedding, (np.ndarray, list)):
                return False
                
            if not isinstance(chunk.metadata, dict):
                return False
                
            return True
            
        except Exception:
            return False

    def _enhance_metadata(self, metadata: Dict) -> Dict:
        """
        Enhance metadata with additional information.
        
        Adds:
        - Timestamp if missing
        - Version information
        - Cleans up None values
        """
        enhanced = metadata.copy()
        
        # Add standard fields if missing
        if 'timestamp' not in enhanced:
            enhanced['timestamp'] = datetime.now().isoformat()
        if 'version' not in enhanced:
            enhanced['version'] = settings.VERSION
            
        # Clean up any None values
        enhanced = {k: v for k, v in enhanced.items() if v is not None}
        
        return enhanced

    def _get_content_type(self, chunk) -> str:
        """
        Determine content type from chunk data.
        
        Supports:
        - Tabular data (CSV, Excel)
        - Documents (PDF, DOCX)
        - Images
        - XML
        - Plain text
        """
        chunk_type = chunk.chunk_type.lower()
        metadata = chunk.metadata
        
        if 'csv' in chunk_type or 'excel' in chunk_type:
            return 'tabular'
        elif 'pdf' in chunk_type or 'docx' in chunk_type:
            return 'document'
        elif 'image' in chunk_type:
            return 'image'
        elif 'xml' in chunk_type:
            return 'xml'
        else:
            return 'text'

    def _get_type_specific_data(self, chunk) -> Dict:
        """
        Get additional data based on chunk type.
        
        Extracts specific metadata for:
        - CSV/Excel data (row info)
        - PDF data (page info)
        - Image data (dimensions, OCR status)
        """
        chunk_type = chunk.chunk_type.lower()
        metadata = chunk.metadata
        specific_data = {}
        
        if 'csv' in chunk_type or 'excel' in chunk_type:
            specific_data.update({
                'row_index': metadata.get('row_index'),
                'total_rows': metadata.get('total_rows'),
                'columns': metadata.get('columns')
            })
        elif 'pdf' in chunk_type:
            specific_data.update({
                'page_number': metadata.get('page'),
                'total_pages': metadata.get('total_pages')
            })
        elif 'image' in chunk_type:
            specific_data.update({
                'image_type': metadata.get('image_type'),
                'dimensions': metadata.get('dimensions'),
                'ocr_status': metadata.get('ocr_status')
            })
        
        # Remove None values
        return {k: v for k, v in specific_data.items() if v is not None}
    
    async def process_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process plain text content into chunks.
        
        Features:
        - Text chunking with overlap
        - Embedding generation
        - Metadata preservation
        - Error handling
        
        Args:
            text (str): The text content to process
            metadata (Dict[str, Any]): Metadata associated with the text
            
        Returns:
            List[DocumentChunk]: List of processed chunks
        """
        chunks = []
        try:
            # Split text into chunks
            text_chunks = chunk_text(text)
            
            # Process each chunk with retry for embedding generation
            for chunk_idx, chunk_tex in enumerate(text_chunks):
                try:
                    # Generate embedding with retry
                    text_embedding = await self._generate_embedding_with_retry(chunk_tex)
                    
                    # Create chunk with metadata
                    chunk = DocumentChunk(
                        id=str(uuid4()),
                        text=chunk_tex,
                        embedding=text_embedding,
                        metadata={
                            **metadata,
                            'chunk_type': 'text',
                            'chunk_index': chunk_idx,
                            'total_chunks': len(text_chunks)
                        },
                        chunk_type='text'
                    )
                    chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"Error processing text chunk {chunk_idx}: {str(e)}")
                    continue
            
            logger.info(f"Processed text: {len(chunks)} chunks extracted")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Operations:
        - Convert to RGB if needed
        - Resize within bounds
        - Maintain aspect ratio
        - Apply LANCZOS resampling
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large or small
            max_dimension = 3000
            min_dimension = 300
            width, height = image.size
            
            if max(width, height) > max_dimension:
                ratio = max_dimension / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            elif min(width, height) < min_dimension:
                ratio = min_dimension / min(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image

    async def process_image(self, image: Image.Image, metadata: Dict[str, Any]) -> DocumentChunk:
        """
        Process a single image with OCR and embedding.
        
        Steps:
        1. Preprocess image
        2. Perform OCR with fallback options
        3. Generate embedding
        4. Create chunk with metadata
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image)
            
            # Configure OCR with custom settings
            custom_config = r'--oem 3 --psm 3'  # Automatic page segmentation
            ocr_text = pytesseract.image_to_string(
                preprocessed_image,
                config=custom_config,
                lang='eng'
            )
            
            if not ocr_text.strip():
                # Try again with different page segmentation mode
                custom_config = r'--oem 3 --psm 6'  # Assume uniform block of text
                ocr_text = pytesseract.image_to_string(
                    preprocessed_image,
                    config=custom_config,
                    lang='eng'
                )

            if not ocr_text.strip():
                raise ValueError("OCR returned empty text")
                
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}. Using image metadata.")
            # Use basic image information as fallback
            width, height = image.size
            ocr_text = (
                f"Image {metadata.get('image_index', '')} "
                f"on page {metadata.get('page', '')}. "
                f"Dimensions: {width}x{height}. "
                f"File: {metadata.get('filename', 'unknown')}"
            )

        try:
            # Generate text embedding with retry
            text_embedding = await self._generate_embedding_with_retry(ocr_text)
            
            # Create chunk
            chunk = DocumentChunk(
                id=str(uuid4()),
                text=ocr_text,
                embedding=text_embedding,
                metadata={
                    **metadata,
                    'width': image.size[0],
                    'height': image.size[1],
                    'mode': image.mode,
                    'ocr_status': 'success' if ocr_text.strip() else 'fallback',
                    'preprocessed_dimensions': preprocessed_image.size
                },
                chunk_type='image'
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    async def process_pdf(self, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process PDF with enhanced text and image extraction.
        
        Features:
        - Text extraction per page
        - Image extraction with OCR
        - Metadata preservation
        - Error handling per page/image
        """
        chunks = []
        
        try:
            # Open PDF directly from memory stream
            memory_stream = io.BytesIO(content)
            doc = fitz.open(stream=memory_stream, filetype="pdf")
            
            logger.info(f"Processing PDF with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}")
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_embedding = await self._generate_embedding_with_retry(text)
                    chunk = DocumentChunk(
                        id=str(uuid4()),
                        text=text,
                        embedding=text_embedding,
                        metadata={**metadata, 'page': page_num + 1},
                        chunk_type='text',
                        page_num=page_num + 1
                    )
                    chunks.append(chunk)
                
                # Extract images
                images = page.get_images()
                logger.info(f"Found {len(images)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        logger.info(f"Processing image {img_index + 1} on page {page_num + 1}")
                        logger.debug(f"Image info: {base_image.get('colorspace', 'N/A')}, "
                                   f"BPC: {base_image.get('bpc', 'N/A')}, "
                                   f"Size: {len(image_bytes)} bytes")
                        
                        # Process image
                        image = Image.open(io.BytesIO(image_bytes))
                        image_chunk = await self.process_image(
                            image,
                            {
                                **metadata,
                                'page': page_num + 1,
                                'image_index': img_index,
                                'image_info': {
                                    'colorspace': base_image.get('colorspace'),
                                    'bpc': base_image.get('bpc'),
                                    'size_bytes': len(image_bytes)
                                }
                            }
                        )
                        chunks.append(image_chunk)
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_index} on page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            memory_stream.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        logger.info(f"Processed PDF: {len(chunks)} chunks extracted ({metadata.get('filename', 'unknown')})")
        return chunks
    
    async def process_docx(self, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process DOCX with text and image extraction.
        
        Features:
        - Text extraction from paragraphs
        - Embedded image extraction
        - Metadata preservation
        - Structure preservation
        """
        chunks = []
        doc = docx.Document(io.BytesIO(content))
        
        # Process text
        for para_index, para in enumerate(doc.paragraphs):
            if para.text.strip():
                text_embedding = await self._generate_embedding_with_retry(para.text)
                chunk = DocumentChunk(
                    id=str(uuid4()),
                    text=para.text,
                    embedding=text_embedding,
                    metadata={**metadata, 'paragraph': para_index},
                    chunk_type='text'
                )
                chunks.append(chunk)
        
        # Process images
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                image = Image.open(io.BytesIO(image_data))
                image_chunk = await self.process_image(
                    image,
                    {**metadata, 'image_rel': rel.target_ref}
                )
                chunks.append(image_chunk)
        
        return chunks

    async def process_csv(self, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process CSV file into chunks with accurate progress tracking.
        
        Features:
        - Batch processing
        - Accurate progress tracking
        - Memory efficient
        - Better row counting
        """
        chunks = []
        BATCH_SIZE = 1000  # Adjust based on your needs
        
        try:
            # Read CSV in chunks
            csv_file = io.BytesIO(content)
            
            # First, get total rows without loading entire file
            logger.info("Counting total rows...")
            total_rows = 0
            for chunk in pd.read_csv(csv_file, chunksize=BATCH_SIZE):
                total_rows += len(chunk)
            csv_file.seek(0)  # Reset file pointer
            
            logger.info(f"Total rows to process: {total_rows}")
            
            # Create DataFrame reader with chunks
            df_iterator = pd.read_csv(
                csv_file, 
                chunksize=BATCH_SIZE,
                low_memory=False,
                on_bad_lines='skip'  # Skip problematic lines
            )
            
            # Get columns from first chunk
            first_chunk = next(df_iterator)
            columns = first_chunk.columns
            
            # Store column names and metadata
            columns_text = f"CSV Columns: {', '.join(columns)}"
            columns_embedding = await self._generate_embedding_with_retry(columns_text)
            columns_chunk = DocumentChunk(
                id=str(uuid4()),
                text=columns_text,
                embedding=columns_embedding,
                metadata={
                    **metadata,
                    'chunk_type': 'csv_columns',
                    'num_columns': len(columns),
                },
                chunk_type='text'
            )
            chunks.append(columns_chunk)
            
            # Create summary chunk
            summary_text = (
                f"CSV Summary:\n"
                f"Total Rows: {total_rows}\n"
                f"Total Columns: {len(columns)}\n"
                f"Columns: {', '.join(columns)}"
            )
            summary_embedding = await self._generate_embedding_with_retry(summary_text)
            summary_chunk = DocumentChunk(
                id=str(uuid4()),
                text=summary_text,
                embedding=summary_embedding,
                metadata={
                    **metadata,
                    'chunk_type': 'csv_summary',
                    'num_rows': total_rows,
                    'num_columns': len(columns)
                },
                chunk_type='text'
            )
            chunks.append(summary_chunk)
            
            # Process first chunk
            processed_rows = 0
            batch_chunks = await self._process_csv_batch(first_chunk, processed_rows, metadata)
            chunks.extend(batch_chunks)
            processed_rows += len(first_chunk)
            
            # Log initial progress
            progress = (processed_rows / total_rows) * 100
            logger.info(f"CSV Processing Progress: {progress:.2f}% ({processed_rows:,}/{total_rows:,} rows)")
            
            # Process remaining chunks
            for df_chunk in df_iterator:
                batch_chunks = await self._process_csv_batch(df_chunk, processed_rows, metadata)
                chunks.extend(batch_chunks)
                processed_rows += len(df_chunk)
                
                # Log progress every batch
                progress = (processed_rows / total_rows) * 100
                logger.info(f"CSV Processing Progress: {progress:.2f}% ({processed_rows:,}/{total_rows:,} rows)")
                
                # Optional: Add progress milestone logging
                if processed_rows % 10000 == 0:
                    logger.info(f"Milestone: Processed {processed_rows:,} rows out of {total_rows:,}")
            
            logger.info(f"Completed CSV Processing: {len(chunks)} chunks extracted from {total_rows:,} rows")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    async def _process_csv_batch(self, df_chunk: pd.DataFrame, start_row: int, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Helper method to process a batch of CSV rows."""
        batch_chunks = []
        
        try:
            for idx, row in df_chunk.iterrows():
                # Format row as a clear string
                row_text = f"Row {start_row + idx + 1}:\n" + "\n".join(
                    f"{col}: {val}" for col, val in row.items()
                )
                
                # Generate embedding with retry
                try:
                    row_embedding = await self._generate_embedding_with_retry(row_text)
                    
                    # Create chunk
                    row_chunk = DocumentChunk(
                        id=str(uuid4()),
                        text=row_text,
                        embedding=row_embedding,
                        metadata={
                            **metadata,
                            'chunk_type': 'csv_row',
                            'row_index': start_row + idx + 1,
                            'num_columns': len(df_chunk.columns)
                        },
                        chunk_type='text'
                    )
                    batch_chunks.append(row_chunk)
                    
                except Exception as e:
                    logger.error(f"Error processing row {start_row + idx + 1}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing batch starting at row {start_row}: {str(e)}")
        
        return batch_chunks

    async def process_excel(self, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process Excel (XLSX/XLS) file into chunks.
        
        Features:
        - Multi-sheet support
        - Column information
        - Summary statistics
        - Sheet-level metadata
        """
        chunks = []
        
        try:
            # Read Excel from bytes
            excel_file = io.BytesIO(content)
            xls_file = pd.ExcelFile(excel_file)
            
            # Process each sheet
            for sheet_name in xls_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Create sheet metadata
                sheet_metadata = {**metadata, 'sheet_name': sheet_name}
                
                # Get column names as a separate chunk
                columns_text = f"Sheet: {sheet_name}, Columns: {', '.join(df.columns)}"
                columns_embedding = await self._generate_embedding_with_retry(columns_text)
                columns_chunk = DocumentChunk(
                    id=str(uuid4()),
                    text=columns_text,
                    embedding=columns_embedding,
                    metadata=sheet_metadata,
                    chunk_type='text'
                )
                chunks.append(columns_chunk)
                
                # Generate summary statistics for the sheet
                summary_text = (
                    f"Excel Sheet Summary:\n"
                    f"Sheet Name: {sheet_name}\n"
                    f"Total Rows: {len(df)}\n"
                    f"Total Columns: {len(df.columns)}\n"
                    f"Columns: {', '.join(df.columns)}"
                )
                
                summary_embedding = await self._generate_embedding_with_retry(summary_text)
                summary_chunk = DocumentChunk(
                    id=str(uuid4()),
                    text=summary_text,
                    embedding=summary_embedding,
                    metadata=sheet_metadata,
                    chunk_type='text'
                )
                chunks.append(summary_chunk)
                
                # Process each row
                for idx, row in df.iterrows():
                    row_text = f"Sheet: {sheet_name}, Row {idx + 1}:\n" + "\n".join(
                        f"{col}: {val}" for col, val in row.items()
                    )
                    row_embedding = await self._generate_embedding_with_retry(row_text)
                    chunk = DocumentChunk(
                        id=str(uuid4()),
                        text=row_text,
                        embedding=row_embedding,
                        metadata={
                            **sheet_metadata,
                            'row_index': idx + 1,
                            'num_columns': len(df.columns)
                        },
                        chunk_type='text'
                    )
                    chunks.append(chunk)
            
            logger.info(f"Processed Excel: {len(chunks)} chunks extracted")
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing Excel: {str(e)}")
            raise

    async def process_xml(self, content: bytes, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Process XML file into chunks.
        
        Features:
        - Structure preservation
        - Element hierarchy
        - Metadata enrichment
        - JSON conversion for processing
        """
        chunks = []
        
        try:
            # Decode bytes to string
            xml_str = content.decode('utf-8')
            
            # Parse XML to dictionary
            xml_dict = xmltodict.parse(xml_str)
            
            # Convert to JSON for easier processing
            json_str = json.dumps(xml_dict, indent=2)
            
            # Create initial XML structure chunk
            structure_text = f"XML File Structure:\n{json.dumps(list(xml_dict.keys()), indent=2)}"
            structure_embedding = await self._generate_embedding_with_retry(structure_text)
            structure_chunk = DocumentChunk(
                id=str(uuid4()),
                text=structure_text,
                embedding=structure_embedding,
                metadata={**metadata, 'chunk_type': 'xml_structure'},
                chunk_type='text'
            )
            chunks.append(structure_chunk)
            
            # Create full XML content chunks
            text_chunks = chunk_text(json_str)
            
            for chunk_text_xml in text_chunks:
                text_embedding = await self._generate_embedding_with_retry(chunk_text_xml)
                chunk = DocumentChunk(
                    id=str(uuid4()),
                    text=chunk_text_xml,
                    embedding=text_embedding,
                    metadata={**metadata, 'chunk_type': 'xml_content'},
                    chunk_type='text'
                )
                chunks.append(chunk)
            
            logger.info(f"Processed XML: {len(chunks)} chunks extracted")
            return chunks
        
        except Exception as e:
            logger.error(f"Error processing XML: {str(e)}")
            raise

    async def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Enhanced search implementation supporting multiple chunk types and embeddings.
        
        Features:
        - Query intent detection
        - Content type awareness
        - Smart chunk filtering
        - Enhanced relevance scoring
        - Redundancy removal
        """
        try:
            # Generate query embedding
            query_embedding = self.text_model.encode([query])[0]
            query_embedding = query_embedding.astype('float32')  # Ensure float32 type
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            logger.info(f"Searching through {len(self.chunks)} total chunks")
            
            # Classify query intent and type
            query_intent = self._classify_query_intent(query)
            query_type = self._classify_query_type(query)
            logger.info(f"Query intent: {query_intent}, Query type: {query_type}")
            
            # Extract any specific numbers or identifiers from query
            identifiers = self._extract_identifiers(query)
            logger.info(f"Extracted identifiers: {identifiers}")
            
            # Filter chunks based on query characteristics
            filtered_chunks = self._filter_chunks(
                self.chunks, 
                query_intent=query_intent,
                query_type=query_type,
                identifiers=identifiers
            )
            
            if not filtered_chunks:
                logger.warning("No chunks matched the filtering criteria, using all chunks")
                filtered_chunks = self.chunks
            
            # Create temporary index for filtered chunks
            temp_index = faiss.IndexFlatL2(self.text_dim)
            
            # Prepare chunk embeddings
            chunk_embeddings = np.vstack([chunk.embedding for chunk in filtered_chunks])
            chunk_embeddings = chunk_embeddings.astype('float32')  # Ensure float32 type
            faiss.normalize_L2(chunk_embeddings)
            temp_index.add(chunk_embeddings)
            
            # Perform semantic search
            D, I = temp_index.search(query_embedding, min(k, len(filtered_chunks)))
            
            # Process results with enhanced scoring
            results = []
            chunk_types_found = set()
            
            for idx, (distance, chunk_idx) in enumerate(zip(D[0], I[0])):
                if chunk_idx < len(filtered_chunks):
                    chunk = filtered_chunks[chunk_idx]
                    chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                    chunk_types_found.add(chunk_type)
                    
                    # Calculate relevance score with type-specific adjustments
                    base_score = 1.0 / (1.0 + distance)
                    relevance_score = self._score_chunk_relevance(
                        chunk=chunk,
                        query=query,
                        query_intent=query_intent,
                        query_type=query_type,
                        base_score=base_score,
                        identifiers=identifiers
                    )
                    
                    # Extract key information based on chunk type
                    chunk_info = self._extract_chunk_info(chunk, query_type)
                    
                    results.append({
                        'text': chunk.text,
                        'score': float(relevance_score),
                        'metadata': {
                            **chunk.metadata,
                            'chunk_info': chunk_info
                        },
                        'type': chunk.chunk_type
                    })
            
            # Remove redundant information
            unique_results = self._remove_redundant_chunks(results)
            
            # Sort by final relevance score
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Format context based on query type and results
            formatted_context = self._format_context(
                results=unique_results,
                query_type=query_type,
                query_intent=query_intent
            )
            
            logger.info(f"Found {len(unique_results)} relevant chunks")
            logger.info(f"Chunk types found: {chunk_types_found}")
            logger.debug(f"Result scores: {[r['score'] for r in unique_results]}")
            
            return {
                'chunks': unique_results,
                'formatted_context': formatted_context,
                'query_intent': query_intent,
                'query_type': query_type,
                'chunk_types': list(chunk_types_found),
                'identifiers': identifiers
            }
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def _classify_query_intent(self, query: str) -> str:
        """
        Classify the type of query to optimize search strategy.
        
        Intents:
        - row_lookup: Row-specific queries
        - column_lookup: Column-specific queries
        - summary: Overview requests
        - general: Other queries
        """
        query_lower = query.lower()
        
        # Row-based queries
        if any(term in query_lower for term in ['row', 'record', 'line', 'entry']):
            return 'row_lookup'
            
        # Column-based queries
        if any(term in query_lower for term in ['column', 'field']):
            return 'column_lookup'
            
        # Summary queries
        if any(term in query_lower for term in ['summary', 'overview', 'total', 'describe']):
            return 'summary'
            
        return 'general'

    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query for better search handling.
        
        Types:
        - image: Image-related queries
        - document: Document-related queries
        - tabular: Table-related queries
        - xml: XML-related queries
        - general: Other queries
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['image', 'picture', 'photo', 'ocr']):
            return 'image'
        elif any(word in query_lower for word in ['page', 'pdf', 'document']):
            return 'document'
        elif any(word in query_lower for word in ['row', 'column', 'cell', 'excel', 'csv']):
            return 'tabular'
        elif any(word in query_lower for word in ['xml', 'tag', 'element']):
            return 'xml'
        
        return 'general'

    def _extract_identifiers(self, query: str) -> Dict[str, Any]:
        """
        Extract various identifiers from the query.
        
        Extracts:
        - Page numbers
        - Row numbers
        - Ordinal references
        """
        identifiers = {}
        query_lower = query.lower().split()
        
        # Extract page numbers
        for i, word in enumerate(query_lower):
            if word == 'page' and i + 1 < len(query_lower) and query_lower[i + 1].isdigit():
                identifiers['page'] = int(query_lower[i + 1])
        
        # Extract row numbers
        for i, word in enumerate(query_lower):
            if word == 'row' and i + 1 < len(query_lower) and query_lower[i + 1].isdigit():
                identifiers['row'] = int(query_lower[i + 1])
        
        # Extract ordinal references
        ordinal_map = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'last': -1, 'previous': -2, 'next': 1
        }
        for word in query_lower:
            if word in ordinal_map:
                identifiers['ordinal'] = ordinal_map[word]
        
        return identifiers
        
    def _filter_chunks(self, chunks: List[DocumentChunk], **kwargs) -> List[DocumentChunk]:
        """
        Filter chunks based on multiple criteria.
        
        Parameters:
        - chunks: List of document chunks
        - query_type: Type of query (image, document, tabular, xml)
        - query_intent: Intent of query (row_lookup, column_lookup, summary)
        - identifiers: Dictionary of identified values (page, row, ordinal)
        
        Returns filtered list of chunks based on criteria.
        """
        filtered_chunks = []
        query_type = kwargs.get('query_type', 'general')
        query_intent = kwargs.get('query_intent', 'general')
        identifiers = kwargs.get('identifiers', {})
        
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', '')
            
            # Filter based on query type
            if query_type != 'general':
                if query_type == 'image' and 'image' not in chunk_type:
                    continue
                elif query_type == 'document' and 'pdf' not in chunk_type and 'docx' not in chunk_type:
                    continue
                elif query_type == 'tabular' and 'csv' not in chunk_type and 'excel' not in chunk_type:
                    continue
                elif query_type == 'xml' and 'xml' not in chunk_type:
                    continue
            
            # Filter based on identifiers
            if 'page' in identifiers and chunk.metadata.get('page') != identifiers['page']:
                continue
            if 'row' in identifiers and chunk.metadata.get('row_index') != identifiers['row']:
                continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks

    def _score_chunk_relevance(self, 
                             chunk: DocumentChunk, 
                             query: str, 
                             query_intent: str, 
                             query_type: str, 
                             base_score: float, 
                             identifiers: Dict[str, Any]) -> float:
        """
        Calculate relevance score for a chunk with type-specific adjustments.
        
        Features:
        - Base score from embedding similarity
        - Type-specific scoring adjustments
        - Intent-based scoring
        - Identifier matching bonuses
        """
        chunk_type = chunk.metadata.get('chunk_type', '')
        
        # Base multiplier
        type_multiplier = 1.0
        
        # Adjust score based on chunk type and query intent
        if query_intent == 'row_lookup':
            if chunk_type == 'csv_row':
                type_multiplier = 0.8
                # Check if query contains row number or ordinal
                if 'row' in identifiers:
                    chunk_row = chunk.metadata.get('row_index', 0)
                    if identifiers['row'] == chunk_row:
                        type_multiplier = 0.5  # Lower score is better
                elif 'ordinal' in identifiers:
                    chunk_row = chunk.metadata.get('row_index', 0)
                    if identifiers['ordinal'] == chunk_row:
                        type_multiplier = 0.5
        
        elif query_intent == 'summary':
            if chunk_type in ['csv_summary', 'csv_columns', 'excel_summary']:
                type_multiplier = 0.8
        
        # Handle document-specific scoring
        if query_type == 'document':
            if 'page' in identifiers:
                chunk_page = chunk.metadata.get('page')
                if chunk_page == identifiers['page']:
                    type_multiplier = 0.5
        
        # Handle image-specific scoring
        elif query_type == 'image':
            if chunk_type == 'image':
                type_multiplier = 0.8
        
        # Apply adjustments
        final_score = base_score * type_multiplier
        return final_score

    def _remove_redundant_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove redundant or duplicate information from chunks.
        
        Features:
        - Content-based deduplication
        - Maintains highest scoring versions
        - Preserves chunk ordering
        """
        seen_content = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a signature for the chunk's content
            content = chunk['text'].strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _format_context(self, results: List[Dict], query_type: str, query_intent: str) -> str:
        """
        Format search results into appropriate context based on type and intent.
        
        Features:
        - Type-specific formatting
        - Intent-based organization
        - Clear structure
        - Metadata incorporation
        """
        if not results:
            return ""
        
        formatted_parts = []
        
        if query_type == 'tabular':
            # Format tabular data
            if query_intent == 'row_lookup':
                # Format specific row data
                row_data = next((r for r in results if r['metadata'].get('chunk_type') == 'csv_row'), None)
                if row_data:
                    formatted_parts.append(row_data['text'])
            else:
                # Format general tabular data
                formatted_parts.extend(r['text'] for r in results)
        elif query_type == 'document':
            # Format document content with page information
            for result in results:
                page = result['metadata'].get('page', '')
                if page:
                    formatted_parts.append(f"Page {page}:\n{result['text']}")
                else:
                    formatted_parts.append(result['text'])
        else:
            # Default formatting
            formatted_parts.extend(r['text'] for r in results)
        
        return "\n\n".join(formatted_parts)

    def _extract_chunk_info(self, chunk: DocumentChunk, query_type: str) -> Dict[str, Any]:
        """
        Extract relevant information from chunk based on query type.
        
        Features:
        - Type-specific information extraction
        - Metadata organization
        - Structure preservation
        """
        info = {}
        
        if query_type == 'tabular':
            info['row_index'] = chunk.metadata.get('row_index')
            info['total_rows'] = chunk.metadata.get('total_rows')
            info['columns'] = chunk.metadata.get('columns', [])
        elif query_type == 'document':
            info['page'] = chunk.metadata.get('page')
            info['total_pages'] = chunk.metadata.get('total_pages')
        elif query_type == 'image':
            info['image_type'] = chunk.metadata.get('image_type')
            info['dimensions'] = chunk.metadata.get('dimensions')
        elif query_type == 'xml':
            info['element_type'] = chunk.metadata.get('element_type')
            info['path'] = chunk.metadata.get('path')
        
        return info
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_embedding_with_retry(self, text: str) -> np.ndarray:
        """
        Generate embeddings with retry logic.
        
        Features:
        - Automatic retries on failure
        - Exponential backoff
        - Validation of output
        - Type checking
        """
        try:
            embedding = self.text_model.encode([text])[0]
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Invalid embedding type")
            if embedding.shape[0] != self.text_dim:
                raise ValueError(f"Invalid embedding dimension: {embedding.shape[0]}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def add_chunks(self, new_chunks: List[DocumentChunk]):
        """
        Add chunks to indices.
        
        Features:
        - Handles both text and image chunks
        - Validates embeddings
        - Updates FAISS index
        - Maintains chunk list
        - Error handling per chunk type
        
        Args:
            new_chunks (List[DocumentChunk]): List of new chunks to add
        """
        try:
            text_chunks = [chunk for chunk in new_chunks if chunk.chunk_type == 'text']
            image_chunks = [chunk for chunk in new_chunks if chunk.chunk_type == 'image']
            
            logger.info(f"Processing {len(text_chunks)} text chunks and {len(image_chunks)} image chunks")
            
            # Add text embeddings
            if text_chunks:
                try:
                    text_embeddings = np.vstack([chunk.embedding for chunk in text_chunks])
                    if text_embeddings.shape[1] != self.text_dim:
                        raise ValueError(f"Text embedding dimension mismatch: {text_embeddings.shape[1]} != {self.text_dim}")
                    
                    # Convert to float32 and normalize
                    text_embeddings = text_embeddings.astype('float32')
                    faiss.normalize_L2(text_embeddings)
                    self.text_index.add(text_embeddings)
                    logger.info(f"Added {len(text_chunks)} text embeddings to index")
                except Exception as e:
                    logger.error(f"Error adding text embeddings: {str(e)}")
                    raise
            
            # Add image embeddings (using text embeddings since we're using OCR text)
            if image_chunks:
                try:
                    image_embeddings = np.vstack([chunk.embedding for chunk in image_chunks])
                    if image_embeddings.shape[1] != self.text_dim:
                        raise ValueError(f"Image embedding dimension mismatch: {image_embeddings.shape[1]} != {self.text_dim}")
                    
                    # Convert to float32 and normalize
                    image_embeddings = image_embeddings.astype('float32')
                    faiss.normalize_L2(image_embeddings)
                    self.text_index.add(image_embeddings)  # Use text_index for all embeddings
                    logger.info(f"Added {len(image_chunks)} image embeddings to index")
                except Exception as e:
                    logger.error(f"Error adding image embeddings: {str(e)}")
                    raise
            
            # Update chunks list
            self.chunks.extend(new_chunks)
            
            logger.info(f"Successfully added {len(new_chunks)} total chunks")
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG engine.
        
        Returns:
        - Document counts
        - Memory usage
        - Index statistics
        """
        text_chunks = sum(1 for chunk in self.chunks if chunk.chunk_type == 'text')
        image_chunks = sum(1 for chunk in self.chunks if chunk.chunk_type == 'image')
        
        text_size = text_chunks * settings.TEXT_EMBEDDING_DIM * 4  # 4 bytes per float
        image_size = image_chunks * settings.IMAGE_EMBEDDING_DIM * 4
        
        return {
            "has_context": len(self.chunks) > 0,
            "total_chunks": len(self.chunks),
            "text_chunks": text_chunks,
            "image_chunks": image_chunks,
            "memory_usage": {
                "text_embeddings_mb": text_size / (1024 * 1024),
                "image_embeddings_mb": image_size / (1024 * 1024),
                "total_mb": (text_size + image_size) / (1024 * 1024)
            },
            "indices_info": {
                "text_index_size": self.text_index.ntotal
            }
        }

    async def clear(self):
        """
        Clear all stored documents and reset indices.
        
        Features:
        - Complete cleanup
        - Index reset
        - File removal
        - State reset
        """
        try:
            # Reset indices
            self.text_index = faiss.IndexFlatL2(settings.TEXT_EMBEDDING_DIM)
            
            # Clear chunks
            self.chunks = []
            
            # Remove stored files
            if os.path.exists(settings.INDEX_FILE):
                os.remove(settings.INDEX_FILE)
            if os.path.exists(settings.CHUNKS_FILE):
                os.remove(settings.CHUNKS_FILE)
            
            # Save the cleared state
            await self._save_data()
            
            logger.info("Cleared all documents and reset indices")
        except Exception as e:
            logger.error(f"Error in clear(): {str(e)}")
            raise

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    async def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a specific chunk by its ID.
        
        Features:
        - Single chunk removal
        - Index rebuilding
        - State consistency
        """
        for i, chunk in enumerate(self.chunks):
            if chunk.id == chunk_id:
                # Remove from chunks list
                self.chunks.pop(i)
                
                # Rebuild index
                self.text_index = faiss.IndexFlatL2(self.text_dim)
                
                # Re-add all remaining chunks
                await self.add_chunks(self.chunks)
                return True
        
        return False

    async def _rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from chunks.
        
        Features:
        - Complete index rebuild
        - Validation of chunks
        - Error recovery
        """
        try:
            logger.info("Rebuilding FAISS index")
            
            # Create new index
            new_index = faiss.IndexFlatL2(self.text_dim)
            valid_chunks = []
            
            # Re-add all valid chunks
            for chunk in self.chunks:
                if self._validate_chunk_data(chunk):
                    valid_chunks.append(chunk)
                    embedding = chunk.embedding.astype('float32').reshape(1, -1)
                    faiss.normalize_L2(embedding)
                    new_index.add(embedding)
            
            # Update instance variables
            self.text_index = new_index
            self.chunks = valid_chunks
            
            # Save recovered state
            await self._save_data()
            
            logger.info(f"Successfully rebuilt index with {len(valid_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            return False

    def _validate_chunk_data(self, chunk: DocumentChunk) -> bool:
        """
        Enhanced chunk validation with detailed checks.
        
        Validates:
        - Required fields
        - Embedding format
        - Metadata structure
        - Content-specific requirements
        """
        try:
            # Basic validation
            if not chunk.id or not chunk.text:
                logger.error(f"Chunk {chunk.id} missing required fields")
                return False

            # Embedding validation
            if not isinstance(chunk.embedding, np.ndarray):
                logger.error(f"Chunk {chunk.id} has invalid embedding type")
                return False
            
            if chunk.embedding.shape[0] != self.text_dim:
                logger.error(f"Chunk {chunk.id} has wrong embedding dimension")
                return False

            # Metadata validation
            if not isinstance(chunk.metadata, dict):
                logger.error(f"Chunk {chunk.id} has invalid metadata type")
                return False

            # Content-specific validation
            chunk_type = chunk.metadata.get('chunk_type', '')
            if chunk_type == 'csv_row':
                if 'row_index' not in chunk.metadata:
                    logger.error(f"CSV row chunk {chunk.id} missing row_index")
                    return False
            elif chunk_type == 'pdf_text':
                if 'page' not in chunk.metadata:
                    logger.error(f"PDF chunk {chunk.id} missing page number")
                    return False

            return True
        except Exception as e:
            logger.error(f"Error validating chunk {chunk.id}: {str(e)}")
            return False

    async def process_chunk_safely(self, chunk_data: Dict[str, Any]) -> Optional[DocumentChunk]:
        """
        Process a single chunk with error handling and validation.
        
        Features:
        - Safe embedding generation
        - Validation checks
        - Error handling
        - Metadata enhancement
        """
        try:
            # Generate embedding with retry
            embedding = await self._generate_embedding_with_retry(chunk_data['text'])
            
            # Create chunk
            chunk = DocumentChunk(
                id=str(uuid4()),
                text=chunk_data['text'],
                embedding=embedding,
                metadata=chunk_data.get('metadata', {}),
                chunk_type=chunk_data.get('chunk_type', 'text'),
                page_num=chunk_data.get('page_num')
            )
            
            # Validate chunk
            if not self._validate_chunk_data(chunk):
                logger.error(f"Chunk validation failed for {chunk.id}")
                return None
                
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return None
        
    async def enhanced_search(self, query: EnhancedQuery) -> EnhancedSearchResponse:
        """
        Perform enhanced search with semantic understanding and filtering
        """
        try:
            search_start_time = datetime.now()
            
            # Initialize search enhancer if not exists
            if not hasattr(self, 'search_enhancer'):
                self.search_enhancer = SearchEnhancer()
            
            # 1. Query Expansion
            expanded_query, expanded_terms = query.question, []
            if query.semantic_search:
                expanded_query, expanded_terms = await self.search_enhancer.expand_query(
                    query.question
                )
            
            # 2. Initial Search
            search_results = await self._perform_initial_search(
                expanded_query,
                query.filters
            )
            
            # 3. Apply Fuzzy Matching if no results
            if query.fuzzy_matching and not search_results:
                text_contents = [chunk.text for chunk in self.chunks]
                fuzzy_indices = await self.search_enhancer.apply_fuzzy_matching(
                    query.question,
                    text_contents
                )
                for idx in fuzzy_indices:
                    chunk = self.chunks[idx]
                    search_results.append({
                        'id': chunk.id,
                        'text': chunk.text,
                        'score': 0.5,  # Base score for fuzzy matches
                        'metadata': chunk.metadata,
                        'chunk_type': chunk.chunk_type,
                        'embedding': chunk.embedding,
                        'page_num': chunk.page_num
                    })
            
            # 4. Rerank Results if semantic search is enabled
            if query.semantic_search and search_results:
                search_results = await self.search_enhancer.rerank_results(
                    query.question,
                    search_results
                )
            
            # 5. Generate Highlights
            for result in search_results:
                highlights = await self.search_enhancer.generate_highlights(
                    query.question,
                    result['text']
                )
                # Convert positions to strings
                if highlights:
                    for highlight in highlights:
                        highlight['position'] = str(highlight['position'])
                result['highlights'] = highlights
            
            # 6. Apply Sorting
            if query.sort:
                search_results = self._apply_sorting(search_results, query.sort)
            
            # 7. Generate Facets
            facets = await self._generate_facets(search_results) if query.include_facets else None
            
            # 8. Pagination
            total_results = len(search_results)
            paginated_results = self._paginate_results(
                search_results,
                query.page,
                query.page_size
            )
            
            # Prepare response
            response = EnhancedSearchResponse(
                results=[
                    SearchResult(
                        id=result['id'],
                        text=result['text'],
                        score=result['score'],
                        metadata=result['metadata'],
                        chunk_type=result['chunk_type'],
                        highlights=result.get('highlights'),
                        page_num=result.get('page_num')
                    ) for result in paginated_results
                ],
                facets=facets,
                total_results=total_results,
                page=query.page,
                total_pages=(total_results + query.page_size - 1) // query.page_size,
                query_expansion_used=bool(expanded_terms),
                expanded_terms=expanded_terms,
                metadata={
                    'processing_time': (datetime.now() - search_start_time).total_seconds(),
                    'filters_applied': bool(query.filters),
                    'semantic_search_used': query.semantic_search,
                    'fuzzy_matching_used': query.fuzzy_matching
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced search error: {str(e)}")
            raise

   

    async def _perform_initial_search(self, query: str, filters: Optional[SearchFilters]) -> List[Dict]:
        """Perform initial search with filters"""
        try:
            # Generate query embedding
            query_embedding = self.text_model.encode([query])[0]
            query_embedding = query_embedding.astype('float32')
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Apply pre-filtering based on metadata
            filtered_chunks = self._apply_metadata_filters(self.chunks, filters) if filters else self.chunks
            
            if not filtered_chunks:
                return []
            
            # Create temporary index for filtered chunks
            temp_index = faiss.IndexFlatL2(self.text_dim)
            chunk_embeddings = np.vstack([chunk.embedding for chunk in filtered_chunks])
            chunk_embeddings = chunk_embeddings.astype('float32')
            faiss.normalize_L2(chunk_embeddings)
            temp_index.add(chunk_embeddings)
            
            # Perform search
            D, I = temp_index.search(query_embedding, len(filtered_chunks))
            
            # Process results
            results = []
            for idx, (distance, chunk_idx) in enumerate(zip(D[0], I[0])):
                if chunk_idx < len(filtered_chunks):
                    chunk = filtered_chunks[chunk_idx]
                    
                    # Calculate relevance score
                    score = 1.0 / (1.0 + distance)
                    
                    # Apply minimum relevance filter if specified
                    if filters and filters.min_relevance and score < filters.min_relevance:
                        continue
                    
                    results.append({
                        'id': chunk.id,
                        'text': chunk.text,
                        'score': score,
                        'metadata': chunk.metadata,
                        'chunk_type': chunk.chunk_type,
                        'embedding': chunk.embedding,
                        'page_num': chunk.page_num
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Initial search error: {str(e)}")
            return []

    def _apply_metadata_filters(self, chunks: List[DocumentChunk], filters: SearchFilters) -> List[DocumentChunk]:
        """Apply metadata-based filters to chunks"""
        filtered_chunks = []
        
        for chunk in chunks:
            # Check document type filter
            if filters.document_types and chunk.metadata.get('document_type') not in filters.document_types:
                continue
                
            # Check content type filter
            if filters.content_type and chunk.metadata.get('content_type') != filters.content_type:
                continue
                
            # Check date range filter
            if filters.date_range:
                chunk_date = chunk.metadata.get('timestamp')
                if chunk_date:
                    chunk_date = datetime.fromisoformat(chunk_date)
                    if (filters.date_range.get('start') and chunk_date < filters.date_range['start']) or \
                    (filters.date_range.get('end') and chunk_date > filters.date_range['end']):
                        continue
                        
            # Check custom metadata filters
            if filters.metadata_filters:
                skip_chunk = False
                for key, value in filters.metadata_filters.items():
                    if chunk.metadata.get(key) != value:
                        skip_chunk = True
                        break
                if skip_chunk:
                    continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks

    def _apply_sorting(self, results: List[Dict], sort_options: SortOptions) -> List[Dict]:
        """Apply sorting to search results"""
        try:
            def get_sort_key(result):
                if sort_options.field == 'score':
                    return result['score']
                elif sort_options.field == 'date':
                    return datetime.fromisoformat(result['metadata'].get('timestamp', '1970-01-01'))
                else:
                    return result['metadata'].get(sort_options.field, '')
            
            reverse = sort_options.order.lower() == 'desc'
            return sorted(results, key=get_sort_key, reverse=reverse)
            
        except Exception as e:
            logger.error(f"Sorting error: {str(e)}")
            return results

    async def _generate_facets(self, results: List[Dict]) -> SearchFacets:
        """Generate facets from search results"""
        try:
            document_types = {}
            content_types = {}
            date_ranges = {
                'last_24h': 0,
                'last_week': 0,
                'last_month': 0,
                'older': 0
            }
            metadata_facets = {}
            
            now = datetime.now()
            
            for result in results:
                # Document types
                doc_type = result['metadata'].get('document_type')
                if doc_type:
                    document_types[doc_type] = document_types.get(doc_type, 0) + 1
                
                # Content types
                content_type = result['metadata'].get('content_type')
                if content_type:
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Date ranges
                if 'timestamp' in result['metadata']:
                    try:
                        date = datetime.fromisoformat(result['metadata']['timestamp'])
                        delta = now - date
                        
                        if delta.days < 1:
                            date_ranges['last_24h'] += 1
                        elif delta.days < 7:
                            date_ranges['last_week'] += 1
                        elif delta.days < 30:
                            date_ranges['last_month'] += 1
                        else:
                            date_ranges['older'] += 1
                    except:
                        pass
                
                # Custom metadata facets
                for key, value in result['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        if key not in metadata_facets:
                            metadata_facets[key] = {}
                        metadata_facets[key][str(value)] = metadata_facets[key].get(str(value), 0) + 1
            
            return SearchFacets(
                document_types=document_types,
                content_types=content_types,
                date_ranges=date_ranges,
                metadata_facets=metadata_facets
            )
            
        except Exception as e:
            logger.error(f"Facet generation error: {str(e)}")
            return SearchFacets(
                document_types={},
                content_types={},
                date_ranges={},
                metadata_facets={}
            )

    def _paginate_results(self, results: List[Dict], page: int, page_size: int) -> List[Dict]:
        """Paginate search results"""
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return results[start_idx:end_idx]
    
   
