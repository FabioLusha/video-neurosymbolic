import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import batch_processor
from .ollama_manager import OllamaRequestManager
from .video_tools import generate_frames

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
        raise IOError(f"Error reading the model's options file {options_file}: {e}") from e


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate scene graph descriptions from video frames using Ollama models.'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ollama model to use for image captioning'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Path to save the generated scene graph descriptions'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        required=True,
        help='Directory containing the videos to process'
    )
    
    parser.add_argument(
        '--videos-metadata',
        type=str,
        help='A JSON file containing the video-ids of the videos to be processed and metadata such as \'star\' and \'end\' specifying which part of the video to process'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second to sample from each video (default: 1.0)'
    )
    
    parser.add_argument(
        '--sys-prompt',
        type=str,
        default=None,
        help='Path to text file containing system prompt (default: empty)'
    )

    parser.add_argument(
        '--usr-prompt',
        type=str,
        required=True,
        help='Path to text file containing user prompt'
    )

    parser.add_argument(
        '--auto-reply',
        type=str,
        required=True,
        help='Path to text file containing auto-reply prompt'
    )

    parser.add_argument(
        '--model-options',
        type=str,
        help='Path to a JSON file containing model options'
    )
    
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Process IDs if provided
    video_info = None
    if args.videos_metadata:
        videos_metadata_path = Path(args.videos_metadata)

        if not videos_metadata_path.exists():
            raise FileNotFoundError(f"File not found: {videos_metadata_path}")

        ext = videos_metadata_path.suffix.lower()

        print(f"=== Loading file with videos metadata: {args.videos_metadata}")
        with open(videos_metadata_path, "r", encoding="utf-8") as f:
            if ext == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    video_info = data
                else:
                    video_info = [data] # Wrap for consistency
            elif ext == '.jsonl':
                video_info = [json.loads(line.strip()) for line in f.readlines()]
            else:
                raise ValueError(
                    f"Unsupported file extension {ext}. "
                    "Expected .json or .jsonl"
                )
            # remove duplicates and keeps only relevant metadata
            video_info = preprocess_videos_metadata(video_info)
            print(f"=== Generating graphs for {len(video_info)} videos")
    else:
        print("=== No video metadata file chosen")
    
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
        args.video_dir, 
        args.output_file, 
        video_info=video_info,
        usr_prompt=usr_prompt,
        reply=reply,
        fps=args.fps,
    )

def preprocess_videos_metadata(dataset):
    video_info = []
    seen = set()

    for data_point in dataset:
        video_id = data_point['video_id']
        start = data_point.get('start', None)
        end = data_point.get('end', None)
        
        if (video_id, start, end) in seen:
            continue


        seen.add((video_id, start, end))
        video_info.append({
            'video_id': video_id,
            'start': start,
            'end': end
        })

    return video_info

def extract_frame_description(text):
    """
    Extract scene graph description from the model's response.
    
    Args:
        text: Response text from the model
        
    Returns:
        Extracted scene graph description or empty string
    """
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
            frame_id = o1.pop("frame_id")
            sg = o1.pop("sg")
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
    except StopIteration:
        return  # Handle case when stream is empty

def streaming_frame_generation(ollama_client, video_dir, output_file_path, usr_prompt, reply, video_info=None, iters=-1, fps=1.0):
    """
    Generate scene graph descriptions for video frames.
    Args:
        ollama_client: OllamaRequestManager instance
        output_file_path: Path to save the generated descriptions
        video_info: Dictionary[ video_id -> metadata] metadata of the videos
        iters: Number of iterations to run (-1 for all data)
        fps: Frames per second to sample per video
    """
    def payload_gen(situations):
        for video in situations:
            video_id = video['video_id']
            start = video.get('start', None)
            end = video.get('end', None)

            print(f"\nVideo: {video_id}")
            print(f" - interval: {start}-{end}")
            print(f" - {len(video['frames'])} frames.")
            
            for frame in video['frames']:
                req_obj = {
                    # qid for backward compatibility
                    "qid": video_id,
                    "video_id": video_id,
                    "start": start,
                    "end": end,
                    "frame_id": frame["frame_id"],
                    "payload": {
                        **ollama_client.ollama_params,
                        "messages": [
                            {
                                "role": "user",
                                "content": usr_prompt,
                                "images": [frame["encoding"]],
                            }
                        ],
                    },
                }

                yield req_obj

    bp = batch_processor
    graph_gen_pipeline = bp.Pipeline(
        payload_gen,
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

    situations = (
        situation_frames
        for situation_frames in generate_frames(video_dir, fps, video_info=video_info)
    )
    graph_gen_pipeline.consume(situations)
    return

if __name__ == "__main__":
    main()