import asyncio
import sys
from pathlib import Path

import gemini_batch_creation
import gemini_batch_processing


def main(client):
    input_dataset = "../data/datasets/STAR/STAR_annotations/STAR_val_small_200.json"
    limit_n = 200
    n_chunks = 4
    user_prompt = "../data/prompts/vqa/user_prompt.txt"
    videos_dir = "../data/datasets/action-genome/Charades_v1_480"
    fps = 1
    max_frames = 64
    output_file = "../scripts/data/vqa_gemini_flash.jsonl"

    chunks_filenames = gemini_batch_creation.preprocess_dataset_to_request(
        input_dataset_path=input_dataset,
        user_prompt_path=user_prompt,
        videos_dir=videos_dir,
        fps=fps,
        max_frames=max_frames,
        output_file_path=output_file,
        limit_n=limit_n,
        n_chunks=n_chunks,
    )

    reply_file = "../data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt"
    model_name = "gemini-2.5-flash"
    results = asyncio.run(
        gemini_batch_processing.zero_shot_cot_batch_pipeline(
            client,
            model_name,
            reply_file,
            *chunks_filenames,
        )
    )

    return


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(SRC_DIR))

    from src.utils import gemini_utils, logg

    logg.logging_setup("gemini-pipeline")
    client = gemini_utils.get_client()

    main(client)
