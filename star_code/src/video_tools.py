import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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
        print(f"Error getting video duration: {result.stderr.decode()}")
        sys.exit(1)

    data = json.loads(result.stdout)["streams"][0]
    return data


def extract_frames(
    video_path,
    fps=1,
    max_frames=None,
    output_dir=None,
    start_time=0.0,
    end_time=0.0,
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

    # Get video duration
    duration = float(get_video_stream_info(video_path)["duration"])

    cmd = [
        "ffmpeg",
        "-hide-banner",
        "-loglevel",
        "error",
    ]

    # set start time
    if start_time and start_time > 0:
        cmd += ["-ss", str(start_time)]

    # Set end_time if not provided
    # not 0.0 := True
    if not end_time or end_time > duration:
        end_time = duration

    if start_time > end_time or (end_time - start_time) < 0.1:
        raise ValueError(
            "The provided 'start_time' is exceeds the end_time or duration of the video-clip"
            f"---- start_time:     {start_time}"
            f"---- end_time:       {end_time}"
            f"---- video duration: {duration}"
        )

    # duration
    duration = end_time - start_time
    cmd += ["-t", str(duration)]

    # video filter for fps
    cmd += ["-fv", f"fps={fps}"]

    # Limit the number of frames
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]

    # Quality (2 is high qyality, lower values are better)
    cmd += ["-q:v", 2]

    out_pattern = f"{temp_dir}/frame_%06d.png"

    cmd.append(out_pattern)
    subprocess.run(cmd, check=True)

    frame_paths = Path(temp_dir).glob("frame_*.png")

    return temp_dir, frame_paths


def generate_frames(video_dir, fps, video_info=None, **kwargs):
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
        video_info  = [{"video_id": v.stem} for v in video_files]

    for video_metadata in video_info:
        video_id   = video_metadata["video_id"]
        start_time = float(video_metadata.get("start", 0))
        end_time   = float(video_metadata.get("end", 0))

        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"Warning: Video {video_id}.mp4 not found in {video_dir}")
            continue

        try:
            # Extract frames from video within specified time range
            temp_dir, frame_paths = extract_frames(
                video_path,
                fps,
                start_time=start_time,
                end_time=end_time,
                **kwargs,
            )

            # Convert frames to base64
            b64_encodings = []
            for i, frame_path in enumerate(frame_paths):
                if not Path(frame_path).exists():
                    print(
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
                    print(f"Error reading frame {i}: {str(e)}")
                    continue

            # Only yield if we have at least one valid frame
            if b64_encodings:
                yield {**video_metadata, "frames": b64_encodings}
            else:
                print(f"Warning: No valid frames were extracted for video {video_id}")

        except Exception as e:
            print(f"Error processing video {video_id}: {str(e)}")
            continue
        finally:
            # Clean up temporary files
            if "temp_dir" in locals():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Error cleaning up temporary directory: {str(e)}")
