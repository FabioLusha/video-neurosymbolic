import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from google.genai import types as gtypes

SRC_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(SRC_DIR))

from src.utils import gemini_utils, logg

logg.logging_setup(f"batch-processing-{datetime.now().strftime('%Y%m%d_%H:%M:%S')}")
logger = logging.getLogger("experiment")


async def upload_file(client, input_file):
    fpath = Path(input_file).resolve()

    # TODO: add check if file is already uploaded using the dispaly name
    # if the file is already uploaded notify it with a log and skip the uploading
    # TODO: Separate logi of file uploading, run batch and monitorin in different
    # functions
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
            "display_name": f"batch-job-{remote_file.stem}",
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
        Path(dest_folder).mkdir(exist_ok=True)

        result_file_name = batch_job.dest.file_name
        result_file = await client.aio.files.download(file=result_file_name)

        # Construct the local save path
        destination_path = Path(dest_folder) / f"{batch_job.display_name}.jsonl"

        # Write the file content to the local path
        with open(destination_path, "wb") as f:
            f.write(result_file)

        logger.info(f"Result saved to: {destination_path}")
        return destination_path
    except Exception as e:
        logger.error(f"Failed to download result for {batch_job.name}. Error: {e}")
        return None


async def run_batch_job(client, input_file, model, dest_folder=None):
    remote_batch_file = await upload_file(client, input_file)
    batch_job = await create_batch_job(client, remote_batch_file, model)
    result_state = await monitor_batch_job(client, batch_job)

    if result_state == gtypes.JobState.JOB_STATE_SUCCEEDED:
        # refreshing batch_obj reference
        final_batch_job = await client.aio.batches.get(name=batch_job.name)
        await download_result_file(client, final_batch_job)

    return


async def main(model_name, *input_files):
    """
    Run batch processing for multiple input files in parallel.

    Args:
        model_name (str): The model to use for batch processing
        *input_files (str): Variable number of input file paths
    """
    if not input_files:
        logger.error("No input files provided")
        return

    # Validate input files exist
    valid_files = []
    for file_path in input_files:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")

    if not valid_files:
        logger.error("No valid input files found")
        return
    # Initialize client (assuming this is defined elsewhere in your code)
    client = gemini_utils.get_client()  # Replace with your client setup
    logger.info(
        f"Starting parallel batch processing for {len(valid_files)} files with model: {model_name}"
    )

    # Run all batch uploads in parallel
    tasks = [
        run_batch_job(client, input_file, model_name) for input_file in valid_files
    ]

    # Execute all tasks concurrently, allowing individual failures
    results = await asyncio.gather(*tasks)  # , return_exceptions=True)

    # Log results
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch processing failed for {valid_files[i]}: {result}")
        else:
            success_count += 1
            logger.info(f"Batch processing completed for {valid_files[i]}")

    logger.info(
        f"Batch processing summary: {success_count}/{len(valid_files)} successful"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <model_name> <input_file1> [input_file2] [...]")
        sys.exit(1)

    model_name = sys.argv[1]
    input_files = sys.argv[2:]

    asyncio.run(main(model_name, *input_files))
