import argparse
import ast
import base64
import json
import os
import re
from pathlib import Path

import pandas as pd

from . import batch_processor
from .ollama_manager import OllamaRequestManager

SEED = 13471225022025

# Base directory is parent of current file's directory
BASE_DIR = Path(__file__).parent.parent


def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")


def _load_model_options(options_file=None):
    options_file = options_file or BASE_DIR / "ollama_model_options.json"
    try:
        with open(options_file) as in_file:
            return json.load(in_file)
    except IOError as e:
        raise IOError(
            f"Error reading the model's options file {options_file}: {e}"
        ) from e


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate scene graph descriptions from video frames using Ollama models."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ollama model to use for image captioning",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the generated scene graph descriptions",
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory with subfolders containing the extracted frames for each video",
    )

    parser.add_argument(
        "--question-dataset",
        type=str,
        required=True,
        help="A JSON file with the questions for which to generate the Scene Graph",
    )

    parser.add_argument(
        "--keyframes-info",
        type=str,
        required=True,
        help="A CSV file with the mapping question_id - video_id - keyframes",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of frames to sample per video (default: 10)",
    )

    parser.add_argument(
        "--sys-prompt",
        type=str,
        default=None,
        help="Path to text file containing system prompt (default: empty)",
    )

    parser.add_argument(
        "--usr-prompt",
        type=str,
        required=True,
        help="Path to text file containing user prompt",
    )

    parser.add_argument(
        "--auto-reply",
        type=str,
        required=True,
        help="Path to text file containing auto-reply prompt",
    )

    parser.add_argument(
        "--model-options", type=str, help="Path to a JSON file containing model options"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Process IDs if provided
    video_info = None
    if not args.question_dataset:
        raise NotImplementedError("Handling video frames without the corresponding question_id dataset has not been implemented yet!")

    question_dataset_path = Path(args.question_dataset)

    if not question_dataset_path.exists():
        raise FileNotFoundError(f"File not found: {question_dataset_path}")

    ext = question_dataset_path.suffix.lower()

    print(f"=== Loading file with videos metadata: {args.question_dataset}")
    with open(question_dataset_path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
            if isinstance(data, list):
                video_info = data
            else:
                video_info = [data]  # Wrap for consistency
        elif ext == ".jsonl":
            video_info = [json.loads(line.strip()) for line in f.readlines()]
        else:
            raise ValueError(
                f"Unsupported file extension {ext}. " "Expected .json or .jsonl"
            )

        # remove duplicates and keeps only relevant metadata
        video_keyframes_info = extract_video_keyframes_info(args.keyframes_info)
        video_info = preprocess_videos_metadata(video_info, video_keyframes_info, filter=False)
        print(f"=== Generating graphs for {len(video_info)} videos")

    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Load system prompt
    sys_prompt = None
    if args.sys_prompt:
        sys_prompt = _load_prompt_fromfile(args.sys_prompt)

    # Load model options
    model_options = _load_model_options(args.model_options)

    # Set up Ollama parameters
    ollama_params = {
        "model": args.model,
        "system": sys_prompt,
        "stream": True,
        "options": model_options,
    }

    # Create Ollama client
    client = OllamaRequestManager(url, ollama_params)

    usr_prompt = _load_prompt_fromfile(args.usr_prompt)
    reply = _load_prompt_fromfile(args.auto_reply)
    # Run frame generation
    streaming_frame_generation(
        client,
        args.frames_dir,
        args.output_file,
        videos_info=video_info,
        usr_prompt=usr_prompt,
        reply=reply,
        max_samples=args.max_samples,
    )


def extract_video_keyframes_info(csv_file_path):
    df = pd.read_csv(csv_file_path)

    video_keyframes_info = {}

    for _, row in df.iterrows():
        question_id = row["question_id"]
        video_id = row["video_id"]

        # Convert string representation of list to actual list (e.g., "['000205', ...]" → ['000205', ...])
        # Safely convert the string-represented list into a Python list
        keyframes = ast.literal_eval(row["Keyframe_IDs"])

        video_keyframes_info[question_id] = {
            "video_id": video_id,
            "keyframes": keyframes,
        }

    return video_keyframes_info


def preprocess_videos_metadata(dataset, video_keyframes_info, filter=False):
    # We do not aggreagate the same keyframes for each video
    # indepenent of the quesiton because ideally each video should
    # be independent, i.e. each video_id with its keyframes determines
    # the video to which the question is associated
    video_info = []
    seen = set()

    for data_point in dataset:
        video_id = data_point["video_id"]
        qid = data_point["question_id"]
        keyframes = sorted(video_keyframes_info[qid]["keyframes"])

        if filter and (video_id, "_".join(keyframes)) in seen:
            continue

        seen.add((video_id, "_".join(keyframes)))
        video_info.append({"question_id": qid, "video_id": video_id, "keyframes": keyframes})

    return video_info


def sample_frames(key_frames, max_sample):

    key_frames = sorted(key_frames)
    max_sample = min(len(key_frames), max_sample)

    if max_sample == 0:
        return []
    if max_sample == 1:
        return [key_frames[len(key_frames) // 2]]

    step = (len(key_frames) - 1) / (max_sample - 1)
    indices = [round(i * step) for i in range(max_sample)]

    frame_ids = [key_frames[i] for i in indices]

    return frame_ids


def generate_frames(frames_dir, video_info, num_frames=5):
    frames_dir = Path(frames_dir)

    video_id = video_info["video_id"]
    keyframes = sample_frames(video_info["keyframes"], num_frames)

    b64_encodings = []
    for keyframe in keyframes:
        frame_path = frames_dir / f"{video_id}.mp4/{keyframe}.png"

        if not frame_path.exists():
            print(
                f"Warning: Frame {frame_path} was not extracted successfully, skipping"
            )
            continue

        try:
            with open(frame_path, "rb") as f:
                img_bytes = f.read()
                b64_encodings.append(
                    {
                        "frame_id": keyframe,
                        "encoding": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                )
        except Exception as e:
            print(f"Error reading frame {keyframe}: {str(e)}")
            continue

    if not b64_encodings:
        print(f"Warning: No valid frames were extracted for video {video_id}")

    return b64_encodings


def extract_frame_description(text):
    """
    Extract scene graph description from the model's response.

    Args:
        text: Response text from the model

    Returns
        Extracted scene graph description or empty string
    """
    if not text:
        return ""

    # the ?s: in the middle capturing group sets the flag re.DOTALL
    pattern = "(?<=<scene_graph>)(?s:.+)(?=</scene_graph>)"
    match = re.search(pattern, text)

    return match.group(0) if match else ""


def frame_aggregator(stream):
    """
    Aggregate frame descriptions for each question.

    Args:
        stream: Stream of frame descriptions

    Yields:
        Aggregated scene graph descriptions
    """
    try:
        o1 = next(stream)
        agg = []
        while o1 is not None:
            try:
                frame_id = o1.pop("frame_id", "unknown")
                sg = o1.pop("sg", "")
                agg.append(f"\nFrame {frame_id}:\n{sg}")
                try:
                    o2 = next(stream)
                    if o2["qid"] != o1["qid"]:
                        yield {**o1, "stsg": "".join(agg)}
                        agg = []

                    o1 = o2
                except (StopIteration, TypeError):
                    yield {**o1, "stsg": "".join(agg)}
                    return  # Generator stops here
            except KeyError as e:
                print(f"Warning: Missing key in frame data: {e}")
                continue
    except StopIteration:
        return  # Handle case when stream is empty


def img_payload_gen(
    ollama_params,
    usr_prompt,
    frames_dir,
    videos_info,
    max_sample,
    batch_images=False
):

    for video in videos_info:
        video_id = video["video_id"]
        start = video.get("start", None)
        end = video.get("end", None)

        frames = generate_frames(frames_dir, video, max_sample)
        print(f"\nVideo: {video_id}")
        print(f" - interval: {start}-{end}")
        print(f" - {len(frames)} frames.")


        payloads = []
        if not frames:
            print(f"Warning: Couldn't extract frames from video {video_id}. Skipping")

        if batch_images:
            req_obj = {
                # qid for backward compatibility
                "qid": video_id,
                **video,
                "payload": {
                    **ollama_params,
                    "messages": [
                        {
                            "role": "user",
                            "content": usr_prompt,
                            "images": [f["encoding"] for f in frames],
                        }
                    ],
                },
            }
            payloads = [req_obj]
        else:
            for frame in frames:
                req_obj = {
                    # qid for backward compatibility
                    "qid": video_id,
                    **video,
                    "frame_id": frame["frame_id"],
                    "payload": {
                        **ollama_params,
                        "messages": [
                            {
                            "role": "user",
                                "content": usr_prompt,
                                "images": [frame["encoding"]],
                            }
                        ],
                    },
                }

                payloads.append(req_obj)

        for payload in payloads:
            yield payload

def streaming_frame_generation(
    ollama_client,
    frames_dir,
    output_file_path,
    usr_prompt,
    reply,
    videos_info,
    max_samples=5,
):
    """
    Generate scene graph descriptions for video frames.

    Args:
        ollama_client: OllamaRequestManager instance
        output_file_path: Path to save the generated descriptions
        video_info: Dictionary[ video_id -> metadata] metadata of the videos
        iters: Number of iterations to run (-1 for all data)
        max_samples: Maximum number of frames to sample per video
    """

    bp = batch_processor
    graph_gen_pipeline = bp.Pipeline(
        lambda videos_info: img_payload_gen(
            ollama_client.ollama_params,
            usr_prompt,
            frames_dir,
            videos_info,
            max_samples,
            batch_images=False
            ),
        lambda payload_gen: bp.stream_request(
            payload_gen, ollama_client, endpoint="chat"
        ),
        lambda stream: bp.auto_reply_gen(stream, reply),
        # check the response is ok before passing to frame_extraction,
        lambda stream: (o for o in stream if o["status"] == "ok"),
        lambda stream: (
            {
                **stream_obj,
                "sg": extract_frame_description(stream_obj["response"]["content"]),
            }
            for stream_obj in stream
        ),
        lambda stream: frame_aggregator(stream),
        lambda stream: bp.stream_save(
            stream, bp.GeneratedGraphFormatter(), output_file_path
        ),
    )

    videos = (video for video in videos_info)
    graph_gen_pipeline.consume(videos)
    return


if __name__ == "__main__":
    main()
