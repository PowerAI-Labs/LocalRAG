from fastapi import HTTPException
import httpx
from datetime import datetime
from typing import Dict, Any

from ..core import settings, logger

async def query_ollama(prompt: str, model_params: Dict[str, Any]) -> str:
    """Query Ollama API with detailed logging"""
    start_time = datetime.now()
    request_id = start_time.strftime("%Y%m%d_%H%M%S")
    
    try:
        model_name = model_params.get('model', 'deepseek-r1:8b')
        logger.info(f"[{request_id}] Starting query with model: {model_name}")
        
        # Prepare request data based on model type
        if "deepseek" in model_name.lower():
            request_data = {
                "model": model_name,
                "prompt": f"<｜User｜>{prompt}\n<｜Assistant｜>",
                "stream": False,
                "options": {
                    "temperature": model_params.get('temperature', 0.7),
                    "top_p": model_params.get('top_p', 0.95),
                    "num_ctx": model_params.get('context_window', 10000),
                    "stop": [
                        "<｜begin▁of▁sentence｜>",
                        "<｜end▁of▁sentence｜>",
                        "<｜User｜>",
                        "<｜Assistant｜>"
                    ]
                }
            }
            logger.info(f"[{request_id}] Sending request to Ollama")
            logger.info(f"[{request_id}] Request data: {request_data}")
           
        else:
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": model_params.get('temperature', 0.7),
                    "top_p": model_params.get('top_p', 0.95),
                    "num_ctx": model_params.get('context_window', 10000)
                }
            }
            logger.info(f"[{request_id}] Sending request to Ollama")
            logger.info(f"[{request_id}] Request data: {request_data}")
        
        
        async with httpx.AsyncClient(timeout=model_params.get('timeout', 300.0)) as client:
            response = await client.post(
                f"{settings.OLLAMA_API_URL}/api/generate",
                json=request_data,
                timeout=model_params.get('timeout', 300.0)
            )
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] Ollama error response: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ollama error: {response.text}"
                )
            
            response_data = response.json()
            completion_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{request_id}] Query completed in {completion_time:.2f} seconds")
            
            return response_data["response"]
            
    except Exception as e:
        logger.error(f"[{request_id}] Error in query_ollama: {str(e)}", exc_info=True)
        if not isinstance(e, HTTPException):
            raise HTTPException(status_code=500, detail=str(e))
        raise e

async def list_available_models():
    """List available models from Ollama"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_API_URL}/api/tags")
            response.raise_for_status()
            
            models = response.json()
            
            # Add additional model information
            for model in models.get('models', []):
                model['supports_images'] = any(
                    name in model['name'].lower() 
                    for name in ['llava', 'bakllava']
                )
            
            return models
            
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))