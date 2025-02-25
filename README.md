# RAGLab Enhanced RAG Application
A fully local AI-powered document processing and search system.
A robust Retrieval-Augmented Generation (RAG) application that uses Ollama local models and FAISS for efficient document processing and semantic search capabilities.

## Overview

This application provides an enhanced RAG implementation with the following key features:

- Multi-format document processing (PDF, DOCX, CSV, Excel, XML, Images)
- Semantic search with content-type awareness
- Batch processing for large files
- Rate limiting and error handling
- Automatic backup and recovery
- Enhanced query expansion and results reranking
- Faceted search capabilities
- OCR support for images
- Progress tracking and webhook notifications

## Project Structure

```
rag-application/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints and models
│   │   ├── core/             # Core settings and logging
│   │   ├── prompts/          # Prompt handling
│   │   ├── services/         # Core services (RAG, Ollama, etc.)
│   │   └── utils/            # Helper utilities
│   ├── data/                 # Data storage
│   └── tests/                # Test files
└── frontend/                 # Frontend React application
```

## Requirements

- Python 3.8+
- FastAPI
- Ollama
- FAISS
- Sentence Transformers
- PyTesseract (for OCR)
- PyMuPDF
- python-docx
- pandas
- numpy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-application.git
cd rag-application
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (if not already installed):
```bash
# For Linux/macOS
curl https://ollama.ai/install.sh | sh

# For Windows
# Download from https://ollama.ai/download
```

4. Install Tesseract for OCR support:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The application can be configured through `config.py` in the core module. Key settings include:

- `OLLAMA_API_URL`: URL for Ollama API (default: "http://localhost:11434")
- `TEXT_EMBEDDING_MODEL`: Model for text embeddings
- `IMAGE_EMBEDDING_MODEL`: Model for image embeddings
- `CHUNK_SIZE`: Size of text chunks for processing
- `BATCH_PROCESSING_SIZE_THRESHOLD`: Threshold for batch processing
- `RATE_LIMIT_MAX_REQUESTS`: Rate limiting settings

## Usage

1. Start the Ollama service:
```bash
ollama run deepseek-r1:8b
```

2. Start the backend server:
```bash
cd backend
python main.py
```

3. The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /upload`: Upload and process documents
- `POST /query`: Query processed documents
- `POST /enhanced-search`: Advanced search with filters and facets
- `POST /api/v1/batch`: Batch processing endpoints
- `GET /health`: Health check endpoint

### Batch Processing

For large files or bulk processing:

```python
batch_request = {
    "files": ["file1.pdf", "file2.docx"],
    "settings": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "ocr_enabled": true
    }
}
response = requests.post("http://localhost:8000/api/v1/batch", json=batch_request)
```

## Features

### Document Processing
- PDF processing with text and image extraction
- DOCX processing with formatting preservation
- CSV/Excel processing with structure preservation
- Image processing with OCR
- XML processing with structure awareness

### Search Capabilities
- Semantic search using embeddings
- Fuzzy matching for approximate searches
- Query expansion for better results
- Faceted search with filtering
- Result highlighting
- Sorting and pagination

### Enhanced Features
- Automatic backup and recovery
- Progress tracking for batch operations
- Webhook notifications
- Rate limiting
- Error handling and logging
- Data validation

## Testing

Run the test suite:
```bash
pytest tests/
```

For enhanced search testing:
```bash
python tests/test_enhanced_search.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Ollama](https://ollama.ai/) for the local LLM support
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework