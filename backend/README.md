# Local Enhanced RAG Application - Backend

This repository contains the backend for an Enhanced Retrieval-Augmented Generation (RAG) application that supports multiple document types and advanced search capabilities.

## Features

- **Multi-format document processing**: Support for PDF, DOCX, CSV, Excel, XML, and images
- **Enhanced search capabilities**: Semantic search, fuzzy matching, faceted search
- **Batch processing**: Handle large files and multiple files efficiently
- **Automatic backup and recovery**: Ensure data integrity
- **Rate limiting**: Protect the API from abuse
- **Background task processing**: Non-blocking operations
- **OCR support**: Extract text from images
- **Enhanced prompting**: Context-aware prompt generation

## System Requirements

- Python 3.9+
- FastAPI
- Sentence Transformers
- PyTorch
- FAISS
- Ollama (running locally or on a remote server)
- Required Python libraries (see requirements.txt)

## Project Structure

```
app/
├── __init__.py        # Package initialization
├── main.py            # Application entry point
├── core/              # Core configuration
│   ├── __init__.py
│   ├── config.py      # Application settings
│   └── logging.py     # Logging configuration
├── api/               # API endpoints
│   ├── __init__.py
│   ├── endpoints.py   # Main API endpoints
│   ├── batch_endpoints.py # Batch processing API
│   ├── models.py      # Data models
│   └── rate_limiter.py # Rate limiting functionality
├── services/          # Business logic services
│   ├── __init__.py
│   ├── rag_engine.py  # Enhanced RAG engine
│   ├── ollama_service.py # Ollama integration
│   ├── search_enhancer.py # Search enhancement
│   └── batch_processor.py # Batch processing service
├── prompts/           # Prompt templates
│   ├── __init__.py
│   └── handler.py     # Prompt generation logic
└── utils/             # Utility functions
    ├── __init__.py
    └── helpers.py     # Helper utilities
```

## Setup and Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/localrag.git
   cd localrag
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Ollama
   - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Pull required models:
     ```bash
     ollama pull deepseek-r1:8b
     ```

5. Create required directories
   ```bash
   mkdir -p data/batch_temp
   ```

## Running the Application

1. Start the backend server
   ```bash
   python main.py
   ```

2. The server will start at `http://127.0.0.1:8000`

## API Endpoints

### Document Processing

- `POST /upload`: Upload and process documents
- `POST /query`: Query the system with a question
- `POST /enhanced-search`: Perform enhanced search with filtering options
- `POST /clear-context`: Clear all stored documents
- `GET /context-status`: Get information about stored documents
- `GET /models`: List available models
- `GET /health`: Check API health

### Batch Processing

- `POST /api/v1/batch`: Submit a batch processing request
- `GET /api/v1/batch/{batch_id}`: Get batch status
- `DELETE /api/v1/batch/{batch_id}`: Cancel a batch job
- `GET /api/v1/batch`: List all batch jobs
- `GET /api/v1/batch/{batch_id}/errors`: Get batch job errors
- `POST /api/v1/batch/{batch_id}/retry`: Retry a failed batch job

## Configuration

The application can be configured through the `config.py` file. Key settings include:

- `OLLAMA_API_URL`: URL for Ollama API (default: `http://localhost:11434`)
- `TEXT_EMBEDDING_MODEL`: Model for text embeddings (default: `all-MiniLM-L6-v2`)
- `CHUNK_SIZE`: Size of text chunks (default: `10000`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `100`)
- `BATCH_PROCESSING_SIZE_THRESHOLD`: Threshold for batch processing (default: `30MB`)
- `RATE_LIMIT_MAX_REQUESTS`: Maximum requests per minute (default: `100`)

## Usage Examples

### 1. Upload a Document

```python
import requests

url = "http://127.0.0.1:8000/upload"
files = {"file": ("document.pdf", open("document.pdf", "rb"), "application/pdf")}
response = requests.post(url, files=files)
print(response.json())
```

### 2. Query the System

```python
import requests
import json

url = "http://127.0.0.1:8000/query"
data = {
    "question": "What are the key points in the document?",
    "model": "deepseek-r1:8b",
    "temperature": 0.7
}
response = requests.post(url, json=data)
print(response.json())
```

### 3. Enhanced Search

```python
import requests
import json

url = "http://127.0.0.1:8000/enhanced-search"
data = {
    "question": "Find information about machine learning",
    "filters": {
        "document_types": ["pdf", "docx"],
        "min_relevance": 0.5
    },
    "semantic_search": True,
    "fuzzy_matching": True,
    "include_facets": True
}
response = requests.post(url, json=data)
print(response.json())
```

## License

[MIT License](/LICENSE)