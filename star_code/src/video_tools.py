import base64
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# star_code
BASE_DIR = Path(__file__).parent.parent / "logs"
BASE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.NOTSET) # delegate filtering to logger
ch_fmt = logging.Formatter(
    "=[%(levelname)s] :- %(message)s"
)
ch.setFormatter(ch_fmt)

fh = logging.FileHandler(str(BASE_DIR / "star_code.log"))
fh.setLevel(logging.WARNING)
fh_fmt = logging.Formatter(
    "=[%(asctime)s][%(levelname)s] - %(name)s :- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(fh_fmt)

logger.addHandler(ch)
logger.addHandler(fh)

def get_video_stream_info(video_path):
    """Get the details of the video streams."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-select_streams",
        "v",  # select only video stream
        "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        logger.error(f"Error getting video duration: {result.stderr.decode()}")
        sys.exit(1)

    data = json.loads(result.stdout)["streams"][0]
    return data


def extract_frames(
    video_path,
    fps=1,
    max_frames=None,
    start_time=None,
    end_time=None,
    output_dir=None,
):
    """Extract num_frames uniformly sampled frames from the video within specified time range.

    Args:
        video_path: Path to the video file
        max_frames: The maximum number of frames to extract from the video
        output_dir: Directory to save frames (optional)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None, meaning end of video)
    """

    temp_dir = tempfile.mkdtemp()
    if output_dir:
        temp_dir = output_dir
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Get video duration
    duration = float(get_video_stream_info(video_path)["duration"])

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
    ]

    # set start time
    start_time = start_time or 0.0
    if start_time and start_time > 0:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", video_path]
    # Limit duration if end_time is set
    if end_time:
        if end_time > duration:
            end_time = duration
        cmd += ["-t", str(end_time - start_time)]

    if start_time > duration or (end_time and start_time > end_time): 
        raise ValueError(
            "The provided 'start_time' exceeds the end_time or duration of the video-clip"
            f"---- start_time:     {start_time}"
            f"---- end_time:       {end_time}"
            f"---- video duration: {duration}"
        )

    # video filter for fps
    cmd += ["-vf", f"fps={fps}"]

    # Limit the number of frames
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]

    # Quality (2 is high qyality, lower values are better)
    cmd += ["-q:v", "2"]

    out_pattern = str(Path(temp_dir, "frame_%06d.png"))
    cmd.append(out_pattern)
    subprocess.run(cmd, check=True)

    frame_paths = list(Path(temp_dir).glob("frame_*.png"))

    return temp_dir, frame_paths

def generate_video_frames(
        video_path,
        fps=1,
        start=None,
        end=None,
        max_frames=None
):
    temp_dir = None
    try:
        # Extract frames from video within specified time range
        temp_dir, frame_paths = extract_frames(
            video_path,
            fps,
            start_time=start,
            end_time=end,
            max_frames=max_frames,
        )

        # Convert frames to base64
        b64_encodings = []
        for i, frame_path in enumerate(frame_paths):
            if not Path(frame_path).exists():
                logger.warning(
                    f"Warning: Frame {i} was not extracted successfully, skipping"
                )
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
                logger.error(f"Error reading frame {i}: {str(e)}")
                continue
        # Only yield if we have at least one valid frame
        if b64_encodings:
            return b64_encodings
        else:
            logger.warning("Warning: No valid frames were extracted for the video")
            return None
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        if "temp_dir" in locals():
            try:
                if temp_dir:
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Warning: Error cleaning up temporary directory: {str(e)}")



def generate_frames(video_dir, fps, video_info=None, output_dir=None):
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
        video_info = [{"video_id": v.stem} for v in video_files]

    for video_metadata in video_info:
        video_id   = video_metadata["video_id"]
        start_time = video_metadata.get("start", None)
        end_time   = video_metadata.get("end", None)

        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.warning(f"Warning: Video {video_id}.mp4 not found in {video_dir}")
            continue

        try:
            video_frames_dir = Path(output_dir, f"{video_id}.mp4") if output_dir else None
            # Extract frames from video within specified time range
            temp_dir, frame_paths = extract_frames(
                video_path,
                fps,
                start_time=start_time,
                end_time=end_time,
                output_dir=video_frames_dir,
            )

            # Convert frames to base64
            b64_encodings = []
            for i, frame_path in enumerate(frame_paths):
                if not Path(frame_path).exists():
                    logger.warning(
                        f"Warning: Frame {i} was not extracted successfully, skipping"
                    )
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
                    logger.error(f"Error reading frame {i}: {str(e)}")
                    continue

            # Only yield if we have at least one valid frame
            if b64_encodings:
                yield {**video_metadata, "frames": b64_encodings}
            else:
                logger.warning(f"Warning: No valid frames were extracted for video {video_id}")

        except Exception as e:
            logger.error(f"Error processing video {video_id}: {str(e)}")
            continue
        finally:
            # Clean up temporary files
            if not output_dir and "temp_dir" in locals():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Warning: Error cleaning up temporary directory: {str(e)}")
