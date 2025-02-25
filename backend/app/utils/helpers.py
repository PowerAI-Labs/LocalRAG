from typing import List
from fastapi import UploadFile
import io
from ..core import settings, logger


def chunk_text(text: str, chunk_size: int = None) -> List[str]:
        """Split text into chunks with overlap."""
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE
            
        words = text.split()
        chunks = []
        overlap = settings.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
async def read_file_content(file: UploadFile) -> bytes:
        """Safely read file content with proper cleanup."""
        try:
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            return content
        except Exception as e:
            logger.error(f"Error reading file {file.filename}: {str(e)}")
            raise
def get_file_type(filename: str) -> str:
        """Determine file type from filename."""
        lower_filename = filename.lower()
        if lower_filename.endswith('.pdf'):
            return 'pdf'
        elif lower_filename.endswith('.docx'):
            return 'docx'
        elif lower_filename.endswith('.txt'):
            return 'text'
        elif any(lower_filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            return 'image'
        else:
            return 'unknown'
def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"


def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
        """Validate if file size is within limits."""
        max_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_bytes

