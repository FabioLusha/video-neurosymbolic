import argparse
import json
import logging
from pathlib import Path

# relative imports work only with the 'from' form of the import
from . import batch_processor, frames_tools, graph_gen
from . import prompt_formatters as pf
from . import video_tools
from ._const import (BASE_DIR, DEFAULT_INPUT_FILE, DEFAULT_MODEL_OPTIONS,
                     DEFAULT_PROMPTS, OLLAMA_URL, PROMPT_TYPES, TASK_TYPES)
from .datasets import CVRRDataset, JudgeDataset, STARDataset
from .ollama_manager import OllamaRequestManager
from .utils import logg


def main(args):

    # Load system and user prompts based on arguments.
    system_prompt_path, user_prompt_path = DEFAULT_PROMPTS[args.prompt_type]

    system_prompt = None
    if args.sys_prompt:
        if args.sys_prompt != "default":
            system_prompt_path = args.sys_prompt
        elif args.sys_prompt == 'default' and system_prompt_path is None:
            raise ValueError(f"There no default system prompt for the {args.prompt_type}")
        system_prompt = _load_prompt_fromfile(system_prompt_path)

    # --user-prompt is a required argument
    if args.user_prompt != "default":
        user_prompt_path = args.user_prompt
    user_prompt = _load_prompt_fromfile(user_prompt_path)

    # Step 3: Create prompt formatter
    prompt_formatter = create_prompt_formatter(args, user_prompt)

    # Step 4: Initialize dataset and load prompts
    dataset = initialize_dataset(
        args, args.input_file or DEFAULT_INPUT_FILE, prompt_formatter, args.ids_file
    )

    # iterating over the dataset formats the prompts
    dataset = [dataset[i] for i in range(len(dataset))]

    # Step 5: Load model options and initialize Ollama manager
    model_options = _load_model_options(args.model_options)
    ollama_client = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": args.model,
            "system": system_prompt,
            "stream": True,
            "options": model_options,
        },
    )

    # Step 6: Load model and process prompts
    ollama_client.load_model()
    if args.task == "graph-understanding" or args.task == "vqa":
        # TODO:
        # For Now VQA is set only for chat mode and is inside the condition of what
        # to apply is inside process_promtps
        process_prompts(ollama_client, dataset, args.mode, args, args.output_file)


def create_prompt_formatter(args, user_prompt):
    """Create the appropriate prompt formatter based on arguments."""
    return PROMPT_TYPES[args.prompt_type](user_prompt)


def _load_prompt_fromfile(filename):
    """Load prompt content from a file."""
    try:
        with open(filename) as in_file:
            return in_file.read().strip()
    except IOError as e:
        raise IOError(f"Error reading prompt file: {e}")


def _load_model_options(options_file=None):
    """Load model options from a JSON file."""
    options_file = options_file or DEFAULT_MODEL_OPTIONS
    try:
        with open(options_file) as in_file:
            return json.load(in_file)
    except IOError as e:
        raise IOError(
            f"Error reading the model's options file {options_file}: {e}"
        ) from e


def initialize_dataset(args, input_filepath, prompt_formatter, ids_filepath):
    """Initialize the appropriate dataset based on type."""
    if not input_filepath:
        input_filepath = BASE_DIR / "data/datasets/STAR_QA_and_stsg_val.json"

    ids = None
    if ids_filepath:
        logger.info(f"=== Loading file with ids: {ids_filepath}")
        with open(ids_filepath, "r") as f:
            ids = [line.strip() for line in f.readlines()]
    else:
        logger.warning("=== No ids file chosen")

    logger.info(f"=== Generating prompts from: {input_filepath}")

    # When using a LLMasJudge Dataset even if prompt_formatter is not suitable
    # for a STAR or CVRR dataset it is safe to intilize it with the llm-as-judge
    # prompt format because it will be overriden after, before being called.
    # The format function is called only when iterating through the elements of
    # the dataset.

    dataset = None
    if args.dataset_type == "star":
        dataset = STARDataset(
            input_filepath, prompt_formatter, stsg_file_path=args.stsg_file, ids=ids
        )
    elif args.dataset_type == "cvrr":
        dataset = CVRRDataset(
            input_filepath, prompt_formatter, stsg_file_path=args.stsg_file, ids=ids
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    if args.task == "llm-judge":
        logger.info("=== Loading judge type dataset")
        dataset = JudgeDataset(dataset, args.responses_file, prompt_formatter)

    return dataset


def process_prompts(ollama_client, dataset, mode, args, output_filepath):
    """Process prompts based on the selected mode."""
    if mode == "generate":
        logger.info("=== Mode: generate")
        batch_processor.batch_generate(
            ollama_client, dataset, output_file_path=output_filepath
        )
    elif mode == "chat":
        logger.info("=== Mode: chat")
        if not args.reply_file:
            raise ValueError(
                "Chat mode requires a reply prompt file. Please provide one using the --reply-file parameter."
            )

        reply = _load_prompt_fromfile(args.reply_file)
        if args.task == "vqa":
            if args.dataset_type == "star":

                if args.frames_dir:
                    if not args.keyframes_info:
                        raise ValueError(
                            "When using VQA task frames mode you need to provide file with keyframes data"
                        )

                    logger.info(f"=== Loading file with videos metadata: {args.keyframes_info}")
                    # remove duplicates and keeps only relevant metadata
                    video_keyframes_info = frames_tools.extract_video_keyframes_info(
                        args.keyframes_info
                    )
                    videos_info = frames_tools.preprocess_videos_metadata(
                        dataset, video_keyframes_info, filter=False
                    )

                    stream_vqa(
                        ollama_client,
                        dataset,
                        reply,
                        args.frames_dir,
                        videos_info,
                        args.max_samples,
                        output_filepath,
                    )
                elif args.videos_dir:
                    stream_vqa_video(
                        ollama_client   =  ollama_client,
                        dataset         = dataset,
                        reply           = reply,
                        videos_dir      = args.videos_dir,
                        fps             = args.fps,
                        output_filepath = output_filepath,
                    )
            else:
                raise NotImplementedError("The VQA works only for the STAR dataset")
        elif args.task == "sgg":
            print(
                "If you want the SGG task look at the graph_gen.py module.\n"
                "Otherwise if you were looking for the sgg with question prompt bias "
                "look at the script notebooks/0707_sgg_qbias_script.py"
            )
        else:
            reply = _load_prompt_fromfile(args.reply_file)
            batch_processor.batch_automatic_chat_reply(
                ollama_client, dataset, reply, output_file_path=output_filepath
            )
    else:
        logger.info("Error: You must select one of the available modes: 'generate' or 'chat'")
        return


def stream_vqa(
    ollama_client,
    dataset,
    reply,
    frames_dir,
    videos_info,
    max_sample,
    output_filepath
):

    def _payload_gen(dataset, videos_info):
        videos_info = {v["question_id"]: v for v in videos_info}

        for datum in dataset:
            question_id = datum["question_id"]

            keyframes = frames_tools.generate_frames(
                frames_dir, videos_info[question_id], max_sample
            )

            if not keyframes:
                logger.warning(
                    f"Warning! Couldn't extract frames for question <{question_id}>. Skipping..."
                )
            encodings = [frame["encoding"] for frame in keyframes]

            req_obj = {
                "qid": question_id,  # situation is list of [{question_id, frame_id, encoding}]
                "payload": {
                    **ollama_client.ollama_params,
                    "messages": [
                        {
                            "role": "user",
                            "content": datum["prompt"],
                            "images": encodings,
                        }
                    ],
                },
            }

            yield req_obj

    bp = batch_processor
    pipeline = bp.Pipeline(
        # the first generator converts the prompt to the right format
        lambda dataset_gen: _payload_gen(dataset_gen, videos_info),
        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, "chat"),
        lambda resp_stream: (o for o in resp_stream if o["status"] == "ok"),
        lambda resp_stream: bp.auto_reply_gen(resp_stream, reply),
        lambda resp_stream: bp.stream_save(
            resp_stream, bp.ChatResponseFormatter(), output_filepath
        ),
    )

    pipeline.consume(dataset)
    return



def stream_vqa_video(
    ollama_client,
    dataset,
    reply,
    videos_dir,
    fps,
    output_filepath
):

    def _payload_gen(dataset):

        for datum in dataset:
            question_id = datum["question_id"]
            video_id = datum['video_id']
            start = datum.get('start', None)
            end = datum.get('end', None)

            keyframes = video_tools.generate_video_frames(
                f"{videos_dir}/{video_id}.mp4",
                fps,
                start,
                end,
            )

            if not keyframes:
                logger.warning(
                    f"Warning! Couldn't extract frames for question <{question_id}>. Skipping..."
                )
                continue

            encodings = [frame["encoding"] for frame in keyframes]

            req_obj = {
                "qid": question_id,  # situation is list of [{question_id, frame_id, encoding}]
                "payload": {
                    **ollama_client.ollama_params,
                    "messages": [
                        {
                            "role": "user",
                            "content": datum["prompt"],
                            "images": encodings,
                        }
                    ],
                },
            }

            yield req_obj

    bp = batch_processor
    pipeline = bp.Pipeline(
        # the first generator converts the prompt to the right format
        lambda dataset_gen: _payload_gen(dataset_gen),
        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, "chat"),
        lambda resp_stream: (o for o in resp_stream \
            if o["status"] == "ok" \
                # A trick to log the error and skip the object.
                # if the status is not ok, we will log the error,
                # logging returns None, therefore the condition will always be False
                or logger.error(
                    f"VLM failed to generate a proper output for {o.get('qid', 'unknown')}:"
                    f"{o.get('error', 'No error info')}"
                )
        ),
        lambda resp_stream: bp.auto_reply_gen(resp_stream, reply),
        lambda resp_stream: (o for o in resp_stream \
            if o["status"] == "ok" \
                # A trick to log the error and skip the object.
                # if the status is not ok, we will log the error,
                # logging returns None, therefore the condition will always be False
                or logger.error(
                    f"LLM failed to generate a proper response to the auto_reply {o.get('qid', 'unknown')}:"
                    f"{o.get('error', 'No error info')}"
                )
        ),
        lambda resp_stream: bp.stream_save(
            resp_stream, bp.ChatResponseFormatter(), output_filepath
        ),
    )

    pipeline.consume(dataset)
    return


def img_payload(
    ollama_params,
    sample,
    raw_videos_dir,
    fps,
    max_frames=None,
    batch_images=False
):
    qid = sample["question_id"]
    video_id = sample["video_id"]
    start = sample.get("start", None)
    end = sample.get("end", None)

    frames = video_tools.generate_video_frames(
        Path(raw_videos_dir, f"{video_id}.mp4"),
        fps=fps,
        start=start,
        end=end,
        max_frames=max_frames
    )

    if not frames:
        logger.warning(f"Warning: Couldn't extract frames from video {video_id}. Skipping")
        return None

    logger.info(f"\nVideo: {video_id}")
    logger.info(f" - interval: {start}-{end}")
    logger.info(f" - {len(frames)} frames.")

    payloads = []
    if batch_images:
        # add img tags delimited by text to help the VLM separate frames
        img_pformatter = pf.ImgPromptDecorator(
            # manually adding the images tag
            pf.PromptFormatter(f"{sample['prompt']}\n{{images}}"),
            img_field="images", # expecting a format string with {images}
            tag="[img]" # using ollama images tag
        )

        usr_prompt_wtags = img_pformatter.format({"images": frames})
        logger.debug(f"The prompt is:\n{usr_prompt_wtags}")
        req_obj = {
            # qid for backward compatibility
            "qid": qid,
            "question_id": qid,
            "payload": {
                **ollama_params,
                "messages": [{
                    "role": "user",
                    "content": usr_prompt_wtags,
                    "images": [f["encoding"] for f in frames],
                }],
            },
        }
        payloads = [req_obj]
    else:
        for frame in frames:
            req_obj = {
                # qid for backward compatibility
                "qid": qid,
                "question_id": qid,
                "frame_id": frame["frame_id"],
                "payload": {
                    **ollama_params,
                    "messages": [{
                        "role": "user",
                        "content": sample["prompt"],
                        "images": [frame["encoding"]],
                    }],
                },
            }

            payloads.append(req_obj)

    return payloads

def stream_sgg(
    ollama_client,
    dataset,
    reply,
    videos_dir,
    fps,
    output_filepath,
    max_frames=None,
    batch_images=False
):

    def _payload_gen(dataset):
        for i in dataset:
            payloads = img_payload(
                ollama_params=ollama_client.ollama_params,
                sample=i,
                raw_videos_dir=videos_dir,
                fps=fps,
                max_frames=max_frames,
                batch_images=batch_images
            )
            if payloads:
                yield from payloads

    # Helper function for the pipeline to distiniguish what do if batch_images is set
    def _stream_batch_condition(stream):
        for obj in stream:
            content = obj["response"]["content"]
            if batch_images:
                # Introduce spurious modifier to satisfy GeneratedGraphFormatter interface
                yield {**obj, "stsg": content}
            else:
                # Map each object to include 'sg', then aggregate the mapped stream
                mapped = (
                    {**obj, "sg": graph_gen.extract_frame_description(obj["response"]["content"])}
                    for obj in stream
                )
                # now let the aggregator generate the stream
                yield from graph_gen.frame_aggregator(mapped)

    bp = batch_processor
    pipeline = bp.Pipeline(
        # the first generator converts the prompt to the right format
        lambda dataset_gen: _payload_gen(dataset_gen),
        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, "chat"),
        lambda stream: (o for o in stream \
            if o["status"] == "ok" \
                # A trick to log the error and skip the object.
                # if the status is not ok, we will log the error,
                # logging returns None, therefore the condition will always be False
                or logger.error(
                    f"VLM failed to generate a proper output for {o.get('qid', 'unknown')}:"
                    f"{o.get('error', 'No error info')}"
                )
        ),
        lambda resp_stream: bp.auto_reply_gen(resp_stream, reply),
        lambda resp_stream: (o for o in resp_stream \
            if o["status"] == "ok" \
                # A trick to log the error and skip the object.
                # if the status is not ok, we will log the error,
                # logging returns None, therefore the condition will always be False
                or logger.error(
                    f"LLM failed to generate a proper response to the auto_reply {o.get('qid', 'unknown')}:"
                    f"{o.get('error', 'No error info')}"
                )
        ),
        _stream_batch_condition,
        lambda resp_stream: bp.stream_save(
            resp_stream, bp.GeneratedGraphFormatter(), output_filepath
        ),
    )

    pipeline.consume(dataset)


def define_cli():
    """Main entry point for the application."""
    # Step 1: Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM with different prompt types")

    parser.add_argument(
        "--task",
        choices=TASK_TYPES.keys(),
        default="graph-understanding",
        help="Choose the task to be performed",
    )
    parser.add_argument(
        "--prompt-type",
        choices=PROMPT_TYPES.keys(),
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--sys-prompt",
        help="Optional system prompt (pass 'default' to use default system prompt).",
    )
    parser.add_argument(
        "--user-prompt",
        help="User prompt (pass default to use 'defualt' prompt)",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Which model to use from those available in Ollama",
        required=True,
    )
    parser.add_argument(
        "--model-options", help="Path to a JSON file containing model options"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["star", "cvrr"],
        required=True,
        help="Type of dataset to use (STAR or CVRR)",
    )
    parser.add_argument("--input-file", help="Input dataset file path")
    parser.add_argument(
        "--ids-file",
        help="Path to a file containing question IDs to process (one ID per line)",
    )
    parser.add_argument(
        "--stsg-file",
        help="File with the spatio-temporal scene graphs if these are not included in the main dataset",
    )
    parser.add_argument(
        "--responses-file",
        help="File with the responses to be evaluated by the judge"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "chat"],
        help="How to run the model, 'chat' or 'generate' mode",
        default="generate",
    )
    parser.add_argument(
        "--reply-file",
        help="File with the text for the automatic reply when run in chat mode",
    )
    parser.add_argument(
        "--output-file",
        help="file path where to save the response"
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        help="Directory with subfolders containing the extracted frames for each video",
    )

    parser.add_argument(
        "--keyframes-info",
        type=str,
        help="A CSV file with the mapping question_id - video_id - keyframes",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to sample per video (default: 5)",
    )

    parser.add_argument(
        "--videos-dir",
        type=str,
        help="Directory with the videos associated to the questions",
    )

    parser.add_argument(
        "--fps",
        type=float,
        help="frame-rate at which to sample images from the videos"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = define_cli()
    experiment_name = Path(args.output_file).stem

    logg.logging_setup(experiment_name)

    logger = logging.getLogger("experiment")
    main(args)
