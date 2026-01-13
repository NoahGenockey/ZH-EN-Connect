"""
LinguaBridge Local - FastAPI Web Service
Local web API for translation (alternative to GUI).

CONTEXT FOR AI ASSISTANTS:
==========================
This is the REST API deployment option (alternative to GUI).

WHY FASTAPI?
- Modern async Python framework (vs Flask's older sync approach)
- Auto-generated OpenAPI docs at /docs (Swagger UI)
- Pydantic validation (type-safe requests/responses)
- High performance (uvicorn ASGI server)
- Production-ready with minimal code

API ENDPOINTS:

1. GET /
   - Root endpoint with API information
   - Returns: Project name, version, available endpoints
   - Use: Quick check that API is running

2. GET /health
   - Health check endpoint
   - Returns: Status, model_loaded flag, cache_size
   - Use: Monitoring, load balancer health checks

3. POST /translate
   - Single text translation
   - Request: {"text": "English text", "use_cache": true}
   - Response: {"source": "...", "translation": "...", "cached": false}
   - Use: Main translation endpoint

4. POST /translate/batch
   - Multiple text translations in one request
   - Request: {"texts": ["Text 1", "Text 2"], "use_cache": true}
   - Response: {"translations": [{...}, {...}]}
   - Use: Efficient bulk translation

5. POST /cache/clear
   - Clear translation cache
   - Returns: {"status": "success", "message": "Cache cleared"}
   - Use: Free memory, reset after model update

6. GET /cache/stats
   - Cache statistics
   - Returns: Size, max_size, sample keys
   - Use: Monitoring, debugging

REQUEST/RESPONSE MODELS (Pydantic):
- TranslationRequest: Single translation input
- BatchTranslationRequest: Multiple translations input
- TranslationResponse: Translation output with metadata
- BatchTranslationResponse: Multiple translations output
- HealthResponse: Health check output

These provide:
- Automatic validation (FastAPI rejects invalid requests)
- Type safety (caught at development time)
- Auto-generated OpenAPI schema

CACHING:
- Same LRU cache as GUI (InferenceCache)
- Shared across all API requests
- Key: input text
- Value: cached translation
- Cache hit significantly reduces latency

ERROR HANDLING:
- 503 Service Unavailable: Model not loaded
- 500 Internal Server Error: Translation failed
- 422 Unprocessable Entity: Invalid request format (FastAPI automatic)
- All errors include descriptive messages

ASYNC OPERATIONS:
- Endpoints are async def (non-blocking)
- Model inference is CPU-bound (not truly async)
- But server can handle multiple requests concurrently
- Good for: Serving multiple users, batch requests

CONFIGURATION (config.yaml under 'deployment.api'):
- host: "127.0.0.1" (localhost only, change to "0.0.0.0" for network access)
- port: 8000
- workers: 1 (increase for production, but uses more RAM)

STARTUP SEQUENCE:
1. FastAPI app initializes
2. @app.on_event("startup") triggers
3. Load config.yaml
4. Initialize inference engine (model loading)
5. Create cache
6. Start uvicorn server
7. API ready for requests

USAGE:

# Start server:
python -m src.app_api
# Or:
python run.py api

# Test with curl (Windows PowerShell):
curl -X POST http://127.0.0.1:8000/translate `
  -H "Content-Type: application/json" `
  -d '{"text": "Hello, world!"}''

# Test with Python requests:
import requests
response = requests.post(
    "http://127.0.0.1:8000/translate",
    json={"text": "Hello, world!"}
)
print(response.json())

# Interactive docs:
# Open browser: http://127.0.0.1:8000/docs
# Try out endpoints directly in Swagger UI

COMPARISON WITH GUI:
API (this file):
+ Multi-user capable
+ RESTful interface (integrate with other apps)
+ Better for production/server deployment
+ Can scale horizontally (multiple workers)
- Requires HTTP client
- More complex for non-technical users

GUI (src/app_gui.py):
+ Easier for end users
+ No network needed
+ Better for personal desktop use
- Single user only
- No programmatic access

PRODUCTION DEPLOYMENT:
For real production (beyond local testing):

1. Change host to "0.0.0.0" (accept network connections)
2. Add authentication (JWT tokens, API keys)
3. Rate limiting (prevent abuse)
4. HTTPS/TLS (nginx reverse proxy)
5. Multiple workers (gunicorn + uvicorn)
6. Monitoring (Prometheus, Grafana)
7. Logging (structured logs, log aggregation)

Example production command:
uvicorn src.app_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info

DOCKER DEPLOYMENT:
Could containerize with:
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "src.app_api"]

CLOUD DEPLOYMENT OPTIONS:
- Railway.app (free tier, easy deployment)
- Render.com (free tier, auto-deploy from GitHub)
- Fly.io (edge deployment)
- AWS Lambda (serverless, cold start penalty)
- Note: Free tiers may have limited CPU/RAM for model

PERFORMANCE:
- Single request latency: ~100ms (uncached)
- Throughput: ~10 req/sec (1 worker, CPU-bound)
- With 4 workers: ~30 req/sec
- Bottleneck: Model inference on CPU
- Improvement: GPU instance, quantization, model caching

MONITORING:
- Check /health regularly
- Monitor cache hit rate (should be 40-60%)
- Watch response times (should be <200ms for short text)
- Alert on 503 errors (model load failure)
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import uvicorn
import os

try:
    from .inference import TranslationInference, InferenceCache
    from .utils import load_config, setup_logging
except ImportError:
    from inference import TranslationInference, InferenceCache
    from utils import load_config, setup_logging


# Request/Response models
class TranslationRequest(BaseModel):
    """Single translation request."""
    text: str = Field(..., description="English text to translate", min_length=1)
    use_cache: bool = Field(True, description="Whether to use cached results")


class BatchTranslationRequest(BaseModel):
    """Batch translation request."""
    texts: List[str] = Field(..., description="List of English texts to translate", min_items=1)
    use_cache: bool = Field(True, description="Whether to use cached results")


class TranslationResponse(BaseModel):
    """Translation response."""
    source: str = Field(..., description="Original English text")
    translation: str = Field(..., description="Chinese translation")
    cached: bool = Field(False, description="Whether result was from cache")


class BatchTranslationResponse(BaseModel):
    """Batch translation response."""
    translations: List[TranslationResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    cache_size: int


# Initialize FastAPI app
app = FastAPI(
    title="LinguaBridge Local API",
    description="Offline English-to-Chinese Neural Machine Translation API",
    version="1.0.0"
)

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web frontend
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

# Global state
inference_engine: Optional[TranslationInference] = None
cache: Optional[InferenceCache] = None
logger: Optional[logging.Logger] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global inference_engine, cache, logger
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting LinguaBridge API...")
    
    # Initialize inference engine
    try:
        inference_engine = TranslationInference(config)
        cache = InferenceCache()
        logger.info("Inference engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LinguaBridge API...")


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the web frontend."""
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    index_path = os.path.join(web_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {
            "name": "LinguaBridge Local API",
            "version": "1.0.0",
            "description": "Offline English-to-Chinese Neural Machine Translation",
            "endpoints": {
                "POST /translate": "Translate a single text",
                "POST /translate/batch": "Translate multiple texts",
                "GET /health": "Health check",
                "POST /cache/clear": "Clear translation cache",
                "GET /docs": "Interactive API documentation"
            }
        }


@app.get("/api", response_model=dict)
async def api_info():
    """API information endpoint."""
    return {
        "name": "LinguaBridge Local API",
        "version": "1.0.0",
        "description": "Offline English-to-Chinese Neural Machine Translation",
        "endpoints": {
            "POST /translate": "Translate a single text",
            "POST /translate/batch": "Translate multiple texts",
            "GET /health": "Health check",
            "POST /cache/clear": "Clear translation cache",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        cache_size=len(cache.cache) if cache else 0
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate English text to Chinese.
    
    Args:
        request: Translation request
        
    Returns:
        Translation response
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Translation model not loaded")
    
    try:
        # Check cache
        cached_result = None
        if request.use_cache and cache:
            cached_result = cache.get(request.text)
        
        if cached_result:
            logger.info(f"Cache hit for text: {request.text[:50]}...")
            return TranslationResponse(
                source=request.text,
                translation=cached_result,
                cached=True
            )
        
        # Perform translation
        logger.info(f"Translating: {request.text[:50]}...")
        translation = inference_engine.translate(request.text)
        
        # Cache result
        if cache:
            cache.put(request.text, translation)
        
        return TranslationResponse(
            source=request.text,
            translation=translation,
            cached=False
        )
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """
    Translate multiple English texts to Chinese.
    
    Args:
        request: Batch translation request
        
    Returns:
        Batch translation response
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Translation model not loaded")
    
    try:
        translations = []
        
        for text in request.texts:
            # Check cache
            cached_result = None
            if request.use_cache and cache:
                cached_result = cache.get(text)
            
            if cached_result:
                translation = cached_result
                is_cached = True
            else:
                # Perform translation
                translation = inference_engine.translate(text)
                is_cached = False
                
                # Cache result
                if cache:
                    cache.put(text, translation)
            
            translations.append(TranslationResponse(
                source=text,
                translation=translation,
                cached=is_cached
            ))
        
        logger.info(f"Batch translation completed: {len(translations)} texts")
        
        return BatchTranslationResponse(translations=translations)
    
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")


@app.post("/cache/clear")
async def clear_cache():
    """Clear the translation cache."""
    if cache:
        cache.clear()
        logger.info("Translation cache cleared")
        return {"status": "success", "message": "Cache cleared"}
    return {"status": "error", "message": "Cache not available"}


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    if cache:
        return {
            "size": len(cache.cache),
            "max_size": cache.max_size,
            "keys": list(cache.cache.keys())[:10]  # First 10 keys
        }
    return {"error": "Cache not available"}


def main():
    """Main entry point for API server."""
    config = load_config()
    api_config = config['deployment']['api']
    
    # Run server
    uvicorn.run(
        "src.app_api:app",
        host=api_config.get('host', '127.0.0.1'),
        port=api_config.get('port', 8000),
        workers=api_config.get('workers', 1),
        log_level="info"
    )


if __name__ == "__main__":
    main()
