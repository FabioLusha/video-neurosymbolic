import os, sys
import subprocess
import re
import json
import base64
import argparse
from pathlib import Path
import shutil
import tempfile



import batch_processor
from ollama_manager import OllamaRequestManager

SEED = 13471225022025

# Base directory is parent of current file's directory
BASE_DIR = Path(__file__).parent.parent


def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")


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
        '--ids-file',
        type=str,
        help='Path to a file containing question IDs to process (one ID per line)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10,
        help='Maximum number of frames to sample per video (default: 10)'
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
    
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Process IDs if provided
    ids = None
    if args.ids:
        # Handle comma-separated IDs or multiple space-separated IDs
        ids = []
        for id_item in args.ids:
            if ',' in id_item:
                ids.extend(id_item.split(','))
            else:
                ids.append(id_item)
        ids = [id_str.strip() for id_str in ids if id_str.strip()]
        
    if args.ids_file:
        print(f"=== Loading file with ids: {args.ids_file}")
        with open(args.ids_file, "r") as f:
            ids = [line.strip() for line in f.readlines()]
    else:
        print("=== No ids file chosen")
    
    url = args.ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    
    # Load system prompt
    sys_prompt = None
    if args.sys_prompt:
        sys_prompt = _load_prompt_fromfile(args.system_prompt)
    
    # Set up Ollama parameters
    ollama_params = {
        "model": args.model,
        "system": sys_prompt,
        "stream": True,
        "options": {
            "num_ctx": args.context_length,
            "temperature": args.temperature,
            "num_predict": args.max_tokens,
            "seed": SEED,
        },
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
        ids=ids,
        usr_prompt=usr_prompt,
        reply=reply, 
        iters=args.iterations,
        max_samples=args.max_samples
    )
    
def get_video_duration(video_path):
    """Get the duration of the video in seconds."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Error getting video duration: {result.stderr.decode()}")
        sys.exit(1)
    
    data = json.loads(result.stdout)
    return float(data['format']['duration'])


def extract_frames(video_path, num_frames, output_dir=None):
    """Extract num_frames uniformly sampled frames from the video."""
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    if output_dir:
        temp_dir = output_dir
    
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    # Calculate time intervals
    if num_frames <= 1:
        intervals = [duration / 2]  # Just the middle frame
    else:
        # Calculate time points ensuring we include frames from the beginning to end
        # but not exactly at 0s (to avoid black frames) and not at the very end
        intervals = [i * duration / (num_frames - 1) for i in range(num_frames)]
        
        # Adjust the first and last intervals slightly to avoid black frames
        if num_frames > 1:
            intervals[0] = max(0.1, intervals[0])  # Avoid the very beginning
            intervals[-1] = min(duration - 0.1, intervals[-1])  # Avoid the very end
    
    frame_paths = []
    # Extract frames at calculated intervals
    for i, time_point in enumerate(intervals):
        output_file = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
        frame_paths.append(output_file)
        
        cmd = [
            'ffmpeg',
            '-ss', str(time_point),  # Seek to time position
            '-i', video_path,        # Input file
            '-vframes', '1',         # Extract just one frame
            '-q:v', '2',             # Quality (2 is high quality, lower values are better)
            output_file
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error extracting frame at {time_point}s: {result.stderr.decode()}")
        else:
            print(f"Extracted frame at {time_point:.2f}s: {output_file}")
    
    return temp_dir, frame_paths

def generate_frames(video_dir, ids, num_frames=10):
    """
    Generate frames from specific videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        ids: List of video IDs to process (without .mp4 extension)
        num_frames: Number of frames to sample per video
        
    Yields:
        List of frames with their base64 encodings
    """
    video_dir = Path(video_dir)
    
    for video_id in ids:
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"Warning: Video {video_id}.mp4 not found in {video_dir}")
            continue

        # Extract frames from video
        temp_dir, frame_paths = extract_frames(video_path, num_frames, None)
        
        # Convert frames to base64
        b64_encodings = []
        for i, frame_path in enumerate(frame_paths):
            with open(frame_path, "rb") as f:
                img_bytes = f.read()
                b64_encodings.append(
                    {
                        "frame_id": i,
                        "video_id": video_id,
                        "encoding": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                )

        # Clean up temporary files
        shutil.rmtree(temp_dir)

        yield b64_encodings

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

def streaming_frame_generation(ollama_client, video_dir, output_file_path, usr_prompt, reply, ids=None, iters=-1, max_samples=10):
    """
    Generate scene graph descriptions for video frames.
    
    Args:
        ollama_client: OllamaRequestManager instance
        output_file_path: Path to save the generated descriptions
        ids: Specific question IDs to process
        iters: Number of iterations to run (-1 for all data)
        max_samples: Maximum number of frames to sample per video
    """
    
    def payload_gen(situations):
        for frames in situations:
            for frame in frames:
                req_obj = {
                    "qid": frame["video_id"],
                    "video_id": frame["video_id"],
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
        for situation_frames in generate_frames(video_dir, ids, num_frames=max_samples)
    )
    graph_gen_pipeline.consume(situations)
    return

if __name__ == "__main__":
    main()