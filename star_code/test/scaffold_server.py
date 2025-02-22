import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI scaffold_server
scaffold_server = FastAPI()

@scaffold_server.post("/api/generate")
async def generate(request: Request):
    """Handle generation requests with streaming by default"""
    try:
        # Parse the request body as JSON
        body = await request.json()
        prompt = body.get('prompt')
        logger.info(f"Received request with prompt:\n {prompt}")

        # Streaming response by default
        async def generate_words():
            words = "Hi, I am alive".split()
            for word in words:
                yield f"{word} "
                await asyncio.sleep(0.5)  # Simulate delay between words

        return StreamingResponse(generate_words(), media_type="text/plain")

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
