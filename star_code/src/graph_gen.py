import argparse
import json
import logging
import os
import re
from pathlib import Path

from . import batch_processor
from .ollama_manager import OllamaRequestManager
from .prompt_formatters import ImgPromptDecorator, PromptFormatter
from .video_tools import generate_frames

SEED = 13471225022025

# Base directary is parent of current file's directory - star_code
BASE_DIR = Path(__file__).parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.NOTSET) # delegate filtering to logger
ch_fmt = logging.Formatter(
    "=[%(levelname)s] :- %(message)s"
)
ch.setFormatter(ch_fmt)

fh = logging.FileHandler(str(LOG_DIR / "star_code.log"))
fh.setLevel(logging.WARNING)
fh_fmt = logging.Formatter(
    "=[%(asctime)s][%(levelname)s] - %(name)s :- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(fh_fmt)

logger.addHandler(ch)
logger.addHandler(fh)


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
    parser.add_argument(
        '--batch-images',
        action='store_true',
        help='If set, send all extracted frames for a video in a single batch request'
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

        logger.info(f"=== Loading file with videos metadata: {args.videos_metadata}")
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
            logger.warning(f"=== Generating graphs for {len(video_info)} videos")
    else:
        logger.warning("=== No video metadata file chosen")
    
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
        batch_images=args.batch_images,
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


def img_payload_gen(
    ollama_params,
    situations,
    usr_prompt,
    batch_images=False
):

    for video in situations:
        video_id = video["video_id"]
        start = video.get("start", None)
        end = video.get("end", None)

        frames = video["frames"]
        logger.info(f"\nVideo: {video_id}")
        logger.info(f" - interval: {start}-{end}")
        logger.info(f" - {len(frames)} frames.")


        payloads = []
        if not frames:
            logger.warning(f"Warning: Couldn't extract frames from video {video_id}. Skipping")
            continue

        if batch_images:
            # add img tags delimited by text to help the VLM separate frames
            img_pformatter = ImgPromptDecorator(
                    PromptFormatter(usr_prompt), 
                    img_field="images", # expecting a format string with {images}
                    tag="[img]" # using ollama images tag
            )
            usr_promtp_wtags = img_pformatter.format({"images": frames})
            req_obj = {
                # qid for backward compatibility
                "qid": video_id,
                "start": start,
                "end": end,
                "payload": {
                    **ollama_params,
                    "messages": [{
                        "role": "user",
                        "content": usr_promtp_wtags,
                        "images": [f["encoding"] for f in frames],
                    }],
                },
            }
            payloads = [req_obj]
        else:
            for frame in frames:
                req_obj = {
                    # qid for backward compatibility
                    "qid": video_id,
                    "start": start,
                    "end": end,
                    "frame_id": frame["frame_id"],
                    "payload": {
                        **ollama_params,
                        "messages": [{
                            "role": "user",
                            "content": usr_prompt,
                            "images": [frame["encoding"]],
                        }],
                    },
                }

                payloads.append(req_obj)

        for payload in payloads:
            yield payload


def streaming_frame_generation(
    ollama_client,
    video_dir,
    output_file_path,
    usr_prompt,
    reply,
    video_info=None,
    fps=1.0,
    batch_images=False
):
    """
    Generate scene graph descriptions for video frames.
    Args:
        ollama_client: OllamaRequestManager instance
        output_file_path: Path to save the generated descriptions
        video_info: Dictionary[ video_id -> metadata] metadata of the videos
        iters: Number of iterations to run (-1 for all data)
        fps: Frames per second to sample per video
    """

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
                    {**obj, "sg": extract_frame_description(obj["response"]["content"])}
                    for obj in stream
                )
                # now let the aggregator generate the stream
                yield from frame_aggregator(mapped)


    bp = batch_processor
    graph_gen_pipeline = bp.Pipeline(
        lambda situations: img_payload_gen(
            ollama_client.ollama_params,
            situations,
            usr_prompt,
            batch_images
        ),
        lambda payload_gen: bp.stream_request(
            payload_gen,
            ollama_client,
            endpoint="chat"
        ),
        lambda stream: bp.auto_reply_gen(stream, reply),
        # check the response is ok before passing to frame_extraction,
        lambda stream: (o for o in stream if o["status"] == "ok"),
        _stream_batch_condition,
        lambda stream: bp.stream_save(
            stream,
            bp.GeneratedGraphFormatter(),
            output_file_path
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
