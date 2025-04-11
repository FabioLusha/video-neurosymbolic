import asyncio
import json
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

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
        prompt = body.get("prompt")
        logger.info(f"Received request with prompt:\n {prompt}")

        # Streaming response by default
        async def generate_words():
            words = "Hi, I am alive".split()
            for word in words:
                yield json.dumps({"response": f"{word} "}) + "\n"
                await asyncio.sleep(0.1)  # Simulate delay between words

        return StreamingResponse(generate_words(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@scaffold_server.post("/api/chat")
async def chat(request: Request):
    try:
        # Parse the request body as JSON
        body = await request.json()
        messages = body.get("messages")
        logger.info(f"Received request with {len(messages)} messages")

        # Streaming the response
        async def generate_words():
            words = "Hi, I am stupid chat bot. I repeat the last message: "
            words = words + "\n" + messages[-1]["content"]
            words = words.split()
            for word in words:
                payload = {"message": {"role": "assistant", "content": f"{word} "}}
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0.1)  # Simulate delay

        return StreamingResponse(generate_words(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@scaffold_server.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "scaffold_server:scaffold_server", host="localhost", port=5555, reload=True
    )
