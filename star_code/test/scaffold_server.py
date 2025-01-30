from fastapi import FastAPI, HTTPException, Request
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI scaffold_server
scaffold_server = FastAPI()


@scaffold_server.post("/api/generate")
async def generate(request: Request):
    """Handle generation requests"""
    try:
        body = await request.json()
        logger.info(f"Received request with prompt:\n {body.get('prompt')}")
        return {"response": "Hi, I am alive"}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@scaffold_server.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "scaffold_server:scaffold_server",
        host="localhost",
        port=11434,
        reload=True
    )
