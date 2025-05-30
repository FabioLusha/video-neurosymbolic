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
            elif ext == '.josnl':
                video_info = [json.loads(line.strip()) for line in f.readlines()]
            else:
                raise ValueError(
                    f"Unsupported file extension {ext}. "
                    "Expected .json or .jsonl"
                )
            # remove duplicates and keeps only relevant metadata
            video_info = preprocess_videos_metadata(video_info)
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
        max_samples=args.max_samples,
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


        video_info.append({
            'video_id': video_id,
            'start': start,
            'end': end
        })

    return video_info



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


def extract_frames(video_path, num_frames, max_fps=1, output_dir=None, start_time=None, end_time=None):
    """Extract num_frames uniformly sampled frames from the video within specified time range.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        output_dir: Directory to save frames (optional)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None, meaning end of video)
    """
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    if output_dir:
        temp_dir = output_dir
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    if start_time is None:
        start_time = 0
    # Set end_time if not provided
    if end_time is None or end_time > duration:
        end_time = duration
    
    # Ensure start_time is within valid range
    start_time = max(0, min(start_time, end_time - 0.1))
    
    # Limit the FPS sampling
    if num_frames / (end_time - start_time) > max_fps:
        num_frames = int((end_time - start_time) / max_fps)
    # Calculate time intervals within the specified range
    if num_frames <= 1:
        intervals = [(start_time + end_time) / 2]  # Just the middle of the range
    else:
        # Calculate time points within the specified range
        intervals = [start_time + i * (end_time - start_time) / (num_frames - 1) 
                    for i in range(num_frames)]
        
        # Adjust the first and last intervals slightly to avoid black frames
        if num_frames > 1:
            intervals[0] = max(start_time + 0.1, intervals[0])  # Avoid the very beginning
            intervals[-1] = min(end_time - 0.1, intervals[-1])  # Avoid the very end
    
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

def generate_frames(video_dir, video_info=None, num_frames=10):
    """
    Generate frames from specific videos in a directory with custom time ranges.
    
    Args:
        video_dir: Directory containing video files
        video_info: Dictionary in format {video_id: {'start': s, 'end': e}}
        num_frames: Number of frames to sample per video
        
    Yields:
        List of frames with their base64 encodings
    """
    video_dir = Path(video_dir)
    
    if video_info is None:
        video_files = list(video_dir.glob("*.mp4"))
        video_info = [{'video_id': v.stem} for v in video_files]
        
    for video_metadata in video_info:
        video_id = video_metadata['video_id']
        start_time = video_metadata.get('start', None)
        end_time = video_metadata.get('end', None)

        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"Warning: Video {video_id}.mp4 not found in {video_dir}")
            continue

        try:
            # Extract frames from video within specified time range
            temp_dir, frame_paths = extract_frames(
                video_path, 
                num_frames,
                start_time=start_time,
                end_time=end_time
            )
            
            # Convert frames to base64
            b64_encodings = []
            for i, frame_path in enumerate(frame_paths):
                if not os.path.exists(frame_path):
                    print(f"Warning: Frame {i} was not extracted successfully, skipping")
                    continue
                    
                try:
                    with open(frame_path, "rb") as f:
                        img_bytes = f.read()
                        b64_encodings.append(
                            {
                                "frame_id": i,
                                "encoding": base64.b64encode(img_bytes).decode("utf-8"),
                            }
                        )
                except Exception as e:
                    print(f"Error reading frame {i}: {str(e)}")
                    continue

            # Only yield if we have at least one valid frame
            if b64_encodings:
                yield {**video_metadata, 'frames': b64_encodings}
            else:
                print(f"Warning: No valid frames were extracted for video {video_id}")

        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            continue
        finally:
            # Clean up temporary files
            if 'temp_dir' in locals():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Error cleaning up temporary directory: {str(e)}")

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

def streaming_frame_generation(ollama_client, video_dir, output_file_path, usr_prompt, reply, video_info=None, iters=-1, max_samples=10):
    """
    Generate scene graph descriptions for video frames.
    
    Args:
        ollama_client: OllamaRequestManager instance
        output_file_path: Path to save the generated descriptions
        video_info: Dictionary[ video_id -> metadata] metadata of the videos
        iters: Number of iterations to run (-1 for all data)
        max_samples: Maximum number of frames to sample per video
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
        for situation_frames in generate_frames(video_dir, video_info, num_frames=max_samples)
    )
    graph_gen_pipeline.consume(situations)
    return

if __name__ == "__main__":
    main()