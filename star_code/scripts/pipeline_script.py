import asyncio
import json
import logging
import sys
from pathlib import Path

import gemini_batch_creation
import gemini_batch_processing

logger = logging.getLogger("experiment")

SRC_DIR = Path(__file__).resolve().parent.parent.resolve()


def main(client):
    chunks_filenames = []
    finished_filenames = []

    # ===========================================================================
    # To be used when some of the chunk fails
    # ===========================================================================
    #
    # chunks_filenames = [
    #   Path("/megaverse/storage/lusha/graph_und/gu_gemini_val_part2bis_20250917_08:18:00_chunk_04.jsonl")
    # ]
    #
    # finished_filenames = [
    #     Path("/megaverse/storage/lusha/graph_und/gu_gemini_val_part7_20250917_08:18:00_chunk_02_2nd_chat_history.jsonl"),
    #     Path("/megaverse/storage/lusha/graph_und/gu_gemini_val_part7_20250917_08:18:00_chunk_03_2nd_chat_history.jsonl"),
    # ]
    # 
    # chunks_filenames = [str(i) for i in chunks_filenames if i.with_stem(f"{i.stem}_2nd_chat_history") not in finished_filenames]
    # finished_filenames = [str(i) for i in finished_filenames]



    # ===========================================================================
    # task = "vqa"

    # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    # stsg_file = None

    # limit_n = None
    # n_chunks = 10
    # user_prompt = "../data/prompts/vqa/user_prompt.txt"
    # videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    # fps = 1
    # max_frames = 64
    # output_file = "../scripts/data/vqa_gemini2.5flash_ongerOutTokens_20250829_18:43:00.jsonl"

    # reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"

    # model_name = "gemini-2.5-flash"
    # ===========================================================================

    # task = "sgg"

    # # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    # # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"
    # input_dataset = str(
    #     SRC_DIR / "data/datasets/STAR/STAR_annotations/STAR_val_part7.json"
    # )
    # stsg_file = None

    # limit_n = None
    # n_chunks = 10
    # user_prompt = str(SRC_DIR / "data/prompts/graph-gen/user_prompt_v2_gemini.txt")
    # videos_dir = str(SRC_DIR / "data/datasets/action-genome/Charades_v1_480")
    # fps = 1
    # max_frames = 64
    # output_file = "/megaverse/storage/lusha/valset_sgg/sgg_gemini2.5flash_val_part7_OutTokens8192_20250913_09:25:00.jsonl"

    # reply_file = str(
    #     SRC_DIR / "data/prompts/graph-gen/format_instructions_v2corrected.txt"
    # )

    # model_name = "gemini-2.5-flash"
    # ===========================================================================

    task = "graph-understanding"

    input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"
    # input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"

    # The file containing the STSG associated to each video or question
    stsg_file = "/megaverse/storage/lusha/valset_sgg/aggregated_final_sgg_gemini2.5flash_val_part2_OutTokens8192_20250913_09:25:00.jsonl"

    # GT STSG
    stsg_file = "../data/datasets/STAR_verbalized_stsg_val.json"

    # set limit_n != None if you want limit the processing to the first `limit_n` instances of the dataset
    limit_n = None
    # in how many batch-files to divide the dataset
    n_chunks = 5
    # path to the user prompt template
    user_prompt = "../data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v3.txt"
    videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    # sampling rate
    fps = 1
    # maximum number of frames to extract
    max_frames = 64
    output_file = "tmp/test_output.jsonl"
    reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"

    # model name from the one provided by Gemini
    model_name = "gemini-2.5-flash"
    # ===========================================================================

    if not chunks_filenames:
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

    results = asyncio.run(
        gemini_batch_processing.zero_shot_cot_batch_pipeline(
            client,
            model_name,
            reply_file,
            *chunks_filenames,
        )
    )

    results += finished_filenames

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

    main(client)
