import os
from google import genai
import asyncio


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

async def delete_all_files():
        client = genai.Client(api_key=GEMINI_API_KEY)
        files = await client.aio.files.list()
        jobs = [client.aio.files.delete(name=i.name) for i in files]
        await asyncio.gather(*jobs)

        return jobs


jobs = asyncio.run(delete_all_files())
print(f"Deleted {len(jobs)} files")
