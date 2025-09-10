import argparse
import base64
import json
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(SRC_DIR))

from src import (
    prompt_formatters,
    video_tools,
)
from src.datasets import STARDataset
from src.utils import logg

logger = logging.getLogger("data_preprocessing")

TASKS = ["vqa", "sgg", "graph-understanding"]
# If you need to disable safety settings
# SAFETY_SETTINGS = {
#     "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
#     "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
#     "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
#     "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
# }

# Config schema:
# https://ai.google.dev/api/generate-content#generationconfig
# {
#   "stopSequences": [
#     string
#   ],
#   "responseMimeType": string,
#   "responseSchema": {
#     object (Schema)
#   },
#   "responseJsonSchema": value,
#   "responseModalities": [
#     enum (Modality)
#   ],
#   "candidateCount": integer,
#   "maxOutputTokens": integer,
#   "temperature": number,
#   "topP": number,
#   "topK": integer,
#   "seed": integer,
#   "presencePenalty": number,
#   "frequencyPenalty": number,
#   "responseLogprobs": boolean,
#   "logprobs": integer,
#   "enableEnhancedCivicAnswers": boolean,
#   "speechConfig": {
#     object (SpeechConfig)
#   },
#   "thinkingConfig": {
#     object (ThinkingConfig)
#   },
#   "mediaResolution": enum (MediaResolution)
# }
DEFAULT_GEN_CONFIG_ENTRY = {
    "thinkingConfig": {
        "thinkingBudget": 0,  # Disable thinking
        "includeThoughts": True,  # For troublehshooting
    },
    "maxOutputTokens": 8_192,
    "seed": 6,
    "temperature": 0.7,
}


def vqa_format(key, text, b64images, gen_config=None, safety_settings=None):
    gen_config = gen_config or DEFAULT_GEN_CONFIG_ENTRY
    format = {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}]
                    + [
                        {"inline_data": {"mime_type": "image/png", "data": b64_enc}}
                        for b64_enc in b64images
                    ],
                }
            ],
            "generationConfig": gen_config,
        },
    }

    # TODO: Add safety settings if needed
    return format


def sgg_format(key, text, b64images, gen_config=None, safety_settings=None):
    gen_config = gen_config or DEFAULT_GEN_CONFIG_ENTRY

    text_prompt = text.split("[img]")
    images = b64images

    parts = []
    for j, text in enumerate(text_prompt):
        parts += (
            [
                {"text": text},
                {"inline_data": {"mime_type": "image/png", "data": images[j]}},
            ]
            if j < len(images)
            else [{"text": text}]
        )
    format = {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": gen_config,
        },
    }

    # TODO: Add safety settings if needed
    return format


def gu_format(key, text, gen_config=None, system_instruction=None, safety_settings=None):
    gen_config = gen_config or DEFAULT_GEN_CONFIG_ENTRY
    format = {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": text}]
                }
            ],
            "generationConfig": gen_config,
        },
    }

    if system_instruction:
        format["request"]["system_instruction"] = {"parts": [{"text": system_instruction}]}
    # TODO: Add safety settings if needed
    return format
 


def batch(
    star_dataset,
    videos_dir,
    fps,
    max_frames,
    out_filepath,
    task,
    gen_config=DEFAULT_GEN_CONFIG_ENTRY,
    limit_n=None,
):
    out_filepath = Path(out_filepath)
    if out_filepath.exists():
        logger.info(f"File {out_filepath} already exists! Not writing")
        return

    out_filepath.parent.mkdir(parents=True, exist_ok=True)

    limit_n = limit_n or len(star_dataset)
    for i in range(limit_n):
        sample = star_dataset[i]

        if task == "vqa" or task == "sgg":
            _, frame_paths = video_tools.extract_frames(
                video_path=f"{videos_dir}/{sample['video_id']}.mp4",
                start_time=float(sample["start"]),
                end_time=float(sample["end"]),
                fps=fps,
                max_frames=max_frames,
            )

            b64encodings = []
            for fpath in frame_paths:
                with open(fpath, "rb") as f:
                    enc = base64.b64encode(f.read()).decode("utf-8")
                    b64encodings.append(enc)

            if task == "vqa":
                generate_content_request = vqa_format(
                    key=sample["question_id"],
                    text=sample["prompt"],
                    b64images=b64encodings,
                    gen_config=gen_config,
                )
            else:
                img_pformatter = prompt_formatters.ImgPromptDecorator(
                    prompt_formatters.PromptFormatter(
                        sample["prompt"].replace("<images>", "{images}")
                    ),
                    img_field="images",  # expecting a format string with {images}
                    tag="[img]",  # using ollama images tag
                )
                text = img_pformatter.format({"images": b64encodings})

                generate_content_request = sgg_format(
                    key=sample["question_id"],
                    text=text,
                    b64images=b64encodings,
                    gen_config=gen_config,
                )
        elif task == "graph-understanding":
            # FIXME: remove the hardcoded system instructions
            # make it extensible
            with open(SRC_DIR / "data/prompts/zero-shot-cot/MCQ_system_prompt_ZS_CoT.txt") as f:
                sys_prompt = f.read()

            generate_content_request = gu_format(
                key=sample["question_id"],
                text=sample["prompt"],
                gen_config=gen_config,
                system_instruction=sys_prompt
            )
        else:
            raise ValueError(f"mode {task} not implemented yet")

        with open(out_filepath, "a") as out:
            line = json.dumps(generate_content_request) + "\n"
            out.write(line)

    return


def preprocess_dataset_to_request(
    input_dataset_path,
    user_prompt_path,
    videos_dir,
    fps,
    max_frames,
    output_file_path,
    task,
    gen_config=None,
    limit_n=None,
    stsg_file_path=None,
    n_chunks=1,
):
    with open(user_prompt_path, "r") as f:
        user_prompt_text = f.read()

    if task == "vqa":
        prompt_formatter = prompt_formatters.MCQPromptWoutSTSG(user_prompt_text)
        dataset = STARDataset(input_dataset_path, prompt_formatter)
    elif task == "sgg":
        prompt_formatter = prompt_formatters.PromptFormatter(user_prompt_text)
        dataset = STARDataset(input_dataset_path, prompt_formatter)
    elif task == "graph-understanding":
        #FIXME: Add system prompt to configurations for graph-understanding task
        # The system instuction don't go in the genConfig
        prompt_formatter = prompt_formatters.MCQPrompt(user_prompt_text)
        dataset = STARDataset(
            input_dataset_path,
            prompt_formatter,
            stsg_file_path,
        )
    else:
        raise ValueError(f"Unexpected task: {task}! Admissibile tasks: {TASKS}")


    if limit_n and limit_n < len(dataset):
        dataset = [dataset[i] for i in range(limit_n)]

    size = len(dataset)
    if size == 0:
        logger.warning("Dataset is empty. No files will be written.")
        raise IndexError("The Dataset is empty")

    if n_chunks > 1:
        chunk_size = int(size / n_chunks)
        chunks = [
            [dataset[j] for j in range(i * chunk_size, (i + 1) * chunk_size)]
            for i in range(n_chunks)
        ]

        if (rem := size % n_chunks) > 0:
            start = chunk_size * n_chunks
            end = start + rem  # == len(dataset)
            chunks[-1] += [dataset[i] for i in range(start, end)]

        orig_file = Path(output_file_path)
        out_files = []
        for i, chunk in enumerate(chunks):
            out_file = str(orig_file.with_stem(f"{orig_file.stem}_chunk_{i + 1:02d}"))

            logger.info(f"Chunk {i + 1}")
            batch(
                task=task,
                star_dataset=chunk,
                videos_dir=videos_dir,
                fps=fps,
                max_frames=max_frames,
                out_filepath=out_file,
                gen_config=gen_config,
            )

            out_files.append(out_file)

        return out_files
    else:
        batch(
            task=task,
            star_dataset=dataset,
            videos_dir=videos_dir,
            fps=fps,
            max_frames=max_frames,
            out_filepath=output_file_path,
            gen_config=gen_config,
        )

        return [output_file_path]


def define_cli():
    """
    Defines and parses command-line arguments for the batch function.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process a star dataset to generate VQA requests from video frames."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS,
        help="Choose the task to be performed",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Path to the STAR dataset."
    )
    parser.add_argument(
        "--stsg-file",
        default=None,
        help="File with the spatio-temporal scene graphs if these are not included in the main dataset",
    )
    parser.add_argument(
        "--user-prompt",
        help="User prompt (pass default to use 'defualt' prompt)",
        required=True,
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Directory containing the video files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="Frames per second to extract from videos.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        required=True,
        help="Maximum number of frames to extract.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file where VQA requests will be written.",
    )
    parser.add_argument(
        "--gen-config",
        type=str,
        default=None,
        help=f"JSON string of generation configuration (e.g., '{{\"temperature\": 0.7}}'). "
        f"Defaults to {json.dumps(DEFAULT_GEN_CONFIG_ENTRY)}.",
    )
    parser.add_argument(
        "--limit-n",
        type=int,
        default=None,
        help="Limit the number of samples to process from the dataset (first n).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=1,
        help="Split the output into this many separate files.",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Convert gen_config string back to dict
    gen_config = None
    if args.gen_config:
        try:
            with open(args.gen_config, "r") as f:
                gen_config = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                "Warning: --gen-config could not be parsed as JSON. Using default."
            )

    return preprocess_dataset_to_request(
        task=args.task,
        input_dataset_path=args.input_dataset,
        stsg_file_path=args.stsg_file,
        user_prompt_path=args.user_prompt,
        videos_dir=args.videos_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        output_file_path=args.output_file,
        limit_n=args.limit_n,
        n_chunks=args.chunks,
        gen_config=gen_config,
    )


if __name__ == "__main__":
    print("Parsing CLI arguments...")
    args = define_cli()

    run_name = Path(args.output_file).stem
    logg.logging_setup(run_name)

    main(args)
