import asyncio
import sys
import json
from pathlib import Path
import logging

import gemini_batch_creation
import gemini_batch_processing

logger = logging.getLogger("experiment")

def main(client):

    # ===========================================================================
    #
    task = "vqa"

    input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    stsg_file = None

    limit_n = None
    n_chunks = 10
    user_prompt = "../data/prompts/vqa/user_prompt.txt"
    videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    fps = 1
    max_frames = 64
    output_file = "../scripts/data/vqa_gemini_flash.jsonl"

    reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"

    # ===========================================================================

    # task = "sgg"
    #
    # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    # stsg_file = None
    #
    # limit_n = None
    # n_chunks = 10
    # user_prompt = "../data/prompts/graph-gen/user_prompt_v2_gemini.txt"
    # videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    # fps = 1
    # max_frames = 64
    # output_file = "../scripts/data/sgg_gemini2.5flash_1000.jsonl"
    #
    # reply_file = "../data/prompts/graph-gen/format_instructions_v2corrected.txt"

    # ===========================================================================

    # task = "graph-understanding"

    # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    # stsg_file = "data/aggregated_final_sgg_gemini25flash_1000.jsonl"
    # limit_n = None
    # n_chunks = 10
    # user_prompt = "../data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v3.txt"
    # videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    # fps = 1
    # max_frames = 64
    # output_file = "../scripts/data/gu_u3_1000_on_geminiSGG_gemini25flash_20250805_15:48:00.jsonl"

    # # vqa or graph-understanding
    # reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"

    # ===========================================================================

    chunks_filenames = gemini_batch_creation.preprocess_dataset_to_request(
        input_dataset_path=input_dataset,
        stsg_file_path=stsg_file,
        task=task,
        user_prompt_path=user_prompt,
        videos_dir=videos_dir,
        fps=fps,
        max_frames=max_frames,
        output_file_path=output_file,
        limit_n=limit_n,
        n_chunks=n_chunks,
    )

    # sgg
    model_name = "gemini-2.5-flash"
    results = asyncio.run(
        gemini_batch_processing.zero_shot_cot_batch_pipeline(
            client,
            model_name,
            reply_file,
            *chunks_filenames,
        )
    )
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
            entry["question_id"] = entry["key"]
            try:
                entry["stsg"] = entry["request"]["contents"][-1]["parts"][0]["text"]
                filtered_entries.append(entry)
            except Exception:
                continue
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

    main(client)
