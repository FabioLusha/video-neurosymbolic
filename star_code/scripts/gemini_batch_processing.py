import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from google.genai import types as gtypes

logger = logging.getLogger("experiment")


async def upload_file(client, input_file: str):
    fpath = Path(input_file).resolve()

    # TODO: add check if file is already uploaded using the display name
    # if the file is already uploaded notify it with a log and skip the uploading
    # TODO: Add a ttl (Time To Leave) for files, Google charges for upload files,
    # but it might be useful having file uploaded for a certain interval of time
    logger.info(f"Uploading {fpath}.")

    uploaded_file = await client.aio.files.upload(
        file=str(fpath),
        config=gtypes.UploadFileConfig(
            display_name=f"batch-{fpath.stem}", mime_type="jsonl"
        ),
    )

    logger.info(f"{fpath.stem} uploaded: {uploaded_file.name}")

    return uploaded_file


async def create_batch_job(client, model, remote_file):
    batch_job = await client.aio.batches.create(
        model=model,
        src=remote_file.name,
        config={
            "display_name": f"batch-job-{remote_file.display_name}",
        },
    )
    logger.info(f"Batch job created: {batch_job.name}")

    return batch_job


async def monitor_batch_job(
    client,
    batch_job,
    initial_wait=30,
    max_wait=1200,  # 20min
):
    completed_states = {
        gtypes.JobState.JOB_STATE_SUCCEEDED,
        gtypes.JobState.JOB_STATE_FAILED,
        gtypes.JobState.JOB_STATE_CANCELLED,
        gtypes.JobState.JOB_STATE_PAUSED,
    }

    job_name = batch_job.name
    batch_job = await client.aio.batches.get(name=job_name)  # Initial get

    current_wait = initial_wait
    while batch_job.state not in completed_states:
        logger.info(
            f"Job: {batch_job.name} - Current state: {batch_job.state.name}. Next check in: {current_wait}s"
        )
        prev_state = batch_job.state

        await asyncio.sleep(current_wait)  # Wait before polling again

        batch_job = await client.aio.batches.get(name=job_name)  # Initial get
        if batch_job.state == prev_state:
            current_wait = min(current_wait * 2, max_wait)
        else:
            # Resetting the waiting time when changing status state
            current_wait = initial_wait

    logger.info(f"Job {job_name} finished with state: {batch_job.state.name}")
    if batch_job.state == gtypes.JobState.JOB_STATE_FAILED:
        logger.error(f"Error: {batch_job.error}")

    return batch_job.state


async def download_result_file(client, batch_job, dest_folder=None):
    dest_folder = dest_folder or "outputs"

    logger.info(f"Downloading results for: {batch_job.name}.")
    try:
        result_file_name = batch_job.dest.file_name
        result_file = await client.aio.files.download(file=result_file_name)

        destination_path = Path(dest_folder, f"{batch_job.display_name}.jsonl")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if the dest_file exist, if it exist create new name
        copy_counter = 0
        base = destination_path.stem
        while destination_path.exists():
            copy_counter += 1
            new_path = destination_path.with_stem(f"{base}_copy{copy_counter}")
            logger.warning(
                f"{destination_path.name} already exists! Saving in {new_path.name}."
            )
            destination_path = new_path

        with open(destination_path, "wb") as f:
            f.write(result_file)

        logger.info(f"Result saved to: {destination_path}")
        return str(destination_path)
    except Exception as e:
        logger.error(f"Failed to download result for {batch_job.name}. Error: {e}")
        raise e




async def run_batch_job(client, input_file: str, model, dest_folder=None):
    remote_batch_file = await upload_file(client, input_file)
    batch_job = await create_batch_job(client, model, remote_batch_file)
    result_state = await monitor_batch_job(client, batch_job)

    if result_state == gtypes.JobState.JOB_STATE_SUCCEEDED:
        # refreshing batch_obj reference
        final_batch_job = await client.aio.batches.get(name=batch_job.name)
        try:
            result_filepath = await download_result_file(
                client, final_batch_job, dest_folder
            )

            await client.aio.batches.delete(name=batch_job.name)
            await client.aio.files.delete(name=remote_batch_file.name)
        except Exception as e:
            raise e

    else:
        logger.error(
            f"Batch job {batch_job.name} failed with state: {result_state.name}"
        )
        raise Exception(f"The Batch process fails with state: {result_state.name}")

    return result_filepath


async def append_response_to_query(input_batch, response_batch):
    query_data_df = pd.DataFrame(input_batch)
    response_data_df = pd.DataFrame(response_batch)

    # Gemini can return multiple candidates for a question
    # the content of a candidate contains the 'Content' for the response
    # with the role and parts of the response
    response_data_df["response_content"] = response_data_df["response"].apply(
        # We need to make some existence cheks because the anser the result response
        # may fail for different reasons (i.e. Blocked becuse of "safety settings"
        lambda x: x["candidates"][0]["content"]
        if x  # first check that response is not NaN
        and isinstance(x, dict)
        and "candidates" in x.keys()
        and x["candidates"]  # check the existence of the list
        and "content" in x["candidates"][0].keys()
        else None
    )

    # Filter out responses with empty text
    response_data_df = response_data_df[
        response_data_df["response_content"].apply(
            lambda x: (x is not None) and (x["parts"][0]["text"] != "")
        )
    ]

    new_batch = query_data_df.merge(response_data_df, on="key", how="inner")
    for _, row in new_batch.iterrows():
        # the assigment has effect because `request` is a dict (a mutable object)
        # and the change is reflected on the DataFrame
        row["request"]["contents"].append(row.pop("response_content"))

    return new_batch[["key", "request"]].to_dict(orient="records")


async def append_default_reply(input_batch: List[Dict], reply_filepath):
    with open(reply_filepath, "r") as f:
        reply = f.read()

    for request in input_batch:
        request["request"]["contents"].append(
            {"role": "user", "parts": [{"text": reply}]}
        )

    return input_batch


async def batch_processor(client, model_name, *input_files):
    """
    Run batch processing for multiple input files in parallel.

    Args:
        model_name (str): The model to use for batch processing
        *input_files (str): Variable number of input file paths
    """
    if not input_files:
        logger.error("No input files provided")
        return []

    # Validate input files exist
    valid_files = []
    for file_path in input_files:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")

    if not valid_files:
        logger.error("No valid input files found")
        return []

    logger.info(
        f"Starting parallel batch processing for {len(valid_files)} files with model: {model_name}"
    )

    # Run all batch uploads in parallel
    tasks = [
        run_batch_job(client, input_file, model_name) for input_file in valid_files
    ]

    # Execute all tasks concurrently, allowing individual failures
    out_files = await asyncio.gather(*tasks, return_exceptions=True)

    # Log results
    success_count = 0
    for i, result in enumerate(out_files):
        if isinstance(result, Exception):
            logger.error(f"Batch processing failed for {valid_files[i]}: {result}!")
        else:
            success_count += 1
            logger.info(f"Batch processing completed for {valid_files[i]}")

    logger.info(
        f"Batch processing summary: {success_count}/{len(valid_files)} successful"
    )

    return out_files


async def zero_shot_cot_batch_pipeline(client, model_name, reply_file, *input_files):
    # first question
    out_files = await batch_processor(client, model_name, *input_files)

    new_batch_paths = []
    for i, result in enumerate(out_files):
        if isinstance(result, Exception):
            logger.warning(
                f"The batch job for {input_files[i]} failed. Skipping this chunk"
            )
            continue

        else:
            input_fpath = Path(input_files[i])
            response_fpath = Path(result)
            with input_fpath.open("r") as f:
                input_batch = [json.loads(line) for line in f.readlines()]
            with response_fpath.open("r") as f:
                out_batch = [json.loads(line) for line in f.readlines()]

            logger.info(
                f"Appending response from {response_fpath.name} to {input_fpath.name}"
            )
            new_batch = await append_response_to_query(input_batch, out_batch)
            logger.info("Appending default_reply")
            new_batch = await append_default_reply(new_batch, reply_file)

            out_fpath = input_fpath.with_stem(f"{input_fpath.stem}_2nd")
            with out_fpath.open("w") as f:
                for entry in new_batch:
                    f.write(json.dumps(entry) + "\n")

            new_batch_paths.append(str(out_fpath))

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

            logger.info(
                f"Appending response from {response_fpath.name} to {input_fpath.name}"
            )
            new_batch = await append_response_to_query(input_batch, out_batch)

            out_fpath = input_fpath.with_stem(f"{input_fpath.stem}_chat_history")
            with out_fpath.open("w") as f:
                for entry in new_batch:
                    f.write(json.dumps(entry) + "\n")

            final_paths.append(str(out_fpath))

    return final_paths


if __name__ == "__main__":
    SRC_DIR = str(Path(__file__).resolve().parent.parent)
    sys.path.append(str(SRC_DIR))
    from src.utils import logg

    logg.logging_setup(f"batch-processing-{datetime.now().strftime('%Y%m%d_%H:%M:%S')}")
