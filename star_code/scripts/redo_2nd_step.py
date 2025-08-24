import sys
from pathlib import Path
import json

import asyncio

from gemini_batch_processing import (
    batch_processor,
)

import logging
logger = logging.getLogger("experiment")

async def main(client):

    new_batch_paths = [
        "data/vqa_gemini_flash_chunk_03_2nd_lean.jsonl",
        "data/vqa_gemini_flash_chunk_06_2nd_lean.jsonl",
    ]
    model_name = "gemini-2.5-flash"


    final_results = await batch_processor(client, model_name, *new_batch_paths)

    # Create a file with all teh parts to infer the chat_history
    final_paths = []
    for i, result in enumerate(final_results):
        if isinstance(result, Exception):
            logger.warning(
                f"The batch job for {new_batch_paths[i]} failed. Skipping this chunk"
            )
            continue

        else:
            input_fpath = Path(new_batch_paths[i])
            response_fpath = Path(result)
            with input_fpath.open("r") as f:
                input_batch = [json.loads(line) for line in f.readlines()]
            with response_fpath.open("r") as f:
                out_batch = [json.loads(line) for line in f.readlines()]

            new_batch = await append_response_to_query(input_batch, out_batch)

            out_fpath = input_fpath.with_stem(f"{input_fpath.stem}_chat_history")
            with out_fpath.open("w") as f:
                for entry in new_batch:
                    f.write(json.dumps(entry) + "\n")

            final_paths.append(str(out_fpath))

    return final_paths



if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(SRC_DIR))

    from src.utils import gemini_utils, logg

    logg.logging_setup("gemini-pipeline")
    client = gemini_utils.get_client()

    asyncio.run(main(client))
