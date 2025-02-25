import uvicorn
from app.core import settings, logger, setup_logging

def main():
    # Initialize logging
    print("Setting up logging...")
    setup_logging()
    
    print("Starting server...")
    logger.info(f"Starting {settings.PROJECT_NAME}")
    
    # Configure and run uvicorn
    uvicorn.run(
        "app.api.endpoints:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()