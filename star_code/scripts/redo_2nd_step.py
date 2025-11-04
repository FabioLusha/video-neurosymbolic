import asyncio
import json
import logging
import sys
from pathlib import Path

from gemini_batch_processing import append_response_to_query, batch_processor

logger = logging.getLogger("experiment")
SRC_DIR = Path(__file__).resolve().parent.parent.resolve()


async def main(client, output_file=None):
    # new_batch_paths = [
    #     "data/vqa_gemini_flash_chunk_02_2nd_lean.jsonl",
    #     "data/vqa_gemini_flash_chunk_03_2nd_lean.jsonl",
    #     "data/vqa_gemini_flash_chunk_06_2nd_lean.jsonl",
    # ]
    if output_file is None:
        output_file = str(
            "/megaverse/storage/lusha/graph_und/gu_gemini_val_part2bis_20250917_08:18:00.jsonl"
        )
    task = "graph-understanding"
    out_path = Path(output_file)

    second_batches = list(out_path.parent.glob(f"{out_path.stem}*_2nd.jsonl"))
    finished_batches = list(
        out_path.parent.glob(f"{out_path.stem}*_2nd_chat_history.jsonl")
    )

    finsished_stems = {x.stem for x in finished_batches}

    new_batch_paths = [
        i for i in second_batches if f"{i.stem}_chat_history" not in finsished_stems
    ]

    finished_filenames = [str(x) for x in finished_batches]

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

    results = finished_filenames + final_paths

    logger.info("Aggregating chunked results.")
    concat = []
    for result in results:
        if isinstance(result, Exception):
            continue

        with open(result, "r") as f:
            concat += [json.loads(line) for line in f.readlines()]

    if task == "sgg":
        filtered_entries = []
        for entry in concat:
            try:
                stsg = entry["request"]["contents"][-1]["parts"][0]["text"]
            except Exception:
                logger.error(
                    f"Error while extracting stsg for key {entry['key']}. Skipping..."
                )
                continue

            entry["stsg"] = stsg
            entry["question_id"] = entry["key"]
            filtered_entries.append(entry)
        concat = filtered_entries

    out_path = Path(output_file)
    agg_filepath = out_path.with_stem(f"aggregated_final_{out_path.stem}")
    with agg_filepath.open("w") as f:
        for entry in concat:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Aggregated file saved in {str(agg_filepath)}")

    return


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(SRC_DIR))

    from src.utils import gemini_utils, logg

    logg.logging_setup("gemini-pipeline")
    client = gemini_utils.get_client()

    out_file = None
    if len(sys.argv) > 1:
        out_file = Path(sys.argv[1]).resolve()

    asyncio.run(main(client, out_file))
