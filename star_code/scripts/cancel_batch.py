import os
from google import genai
from google.genai import types
import asyncio


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)

async def delete_all_files():
        client = genai.Client(api_key=GEMINI_API_KEY)
        jobs = [client.aio.batches.delete(name=i.name)
                for i in client.batches.list(config=types.ListBatchJobsConfig(page_size=100))]
        await asyncio.gather(*jobs)

        return jobs


jobs = asyncio.run(delete_all_files())
print(f"Deleted {len(jobs)} files")
