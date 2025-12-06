# Graph Generation Module Documentation

This document provides a detailed guide on using the Graph Generation module (`star_code/src/graph_gen.py`) to extract Spatio-Temporal Scene Graphs (STSG) from videos.

## Overview

The Graph Generation module is responsible for:
1.  **Frame Extraction**: Sampling frames from videos based on FPS or a maximum count.
2.  **Prompt Assembly**: Combining user prompts, system prompts, and images.
3.  **Model Inference**: Sending requests to an Ollama-hosted VLM (Vision-Language Model).
4.  **Response Aggregation**: Compiling frame-level or batch-level descriptions into a structured output.

## Prerequisites

Before running the module, ensure you have:
-   **Ollama** running and accessible (default: `http://localhost:11434`).
-   **Video Dataset**: A directory containing video files (e.g., `.mp4`).
-   **Metadata File**: A JSON or JSONL file specifying which videos (and optionally which segments) to process.

## Step-by-Step Guide

### 1. Prepare Data

You need a metadata file to tell the script which videos to process. This can be a JSON or JSONL file.

**Format (JSON):**
```json
[
  {
    "video_id": "001",
    "start": 0.0,
    "end": 10.0
  },
  {
    "video_id": "002"
  }
]
```

**Format (JSONL):**
```json
{"video_id": "001", "start": 0.0, "end": 10.0}
{"video_id": "002"}
```

*Note: `start` and `end` are optional. If omitted, the entire video is processed.*

### 2. Craft Prompts

The module uses three types of prompts:

1.  **System Prompt** (`--sys-prompt`): Optional. Sets the behavior or persona of the model.
2.  **User Prompt** (`--usr-prompt`): Required. The main instruction for the model.
3.  **Auto-Reply** (`--auto-reply`): Required. Used to simulate a conversation turn or enforce formatting (often empty or simple acknowledgement like "Here is the scene graph:").

#### Prompt Formatting & Batch Images

-   **Standard Mode** (Frame-by-frame): The user prompt is sent with each frame individually.
-   **Batch Mode** (`--batch-images`): Multiple frames are sent in a single request.

**Crucial for Batch Mode:**
If you use `--batch-images`, your user prompt file **MUST** contain the `{images}` placeholder. The code dynamically replaces this placeholder with a sequence of image tags (e.g., `Image 0: [img] ...`).

**Example User Prompt (Batch Mode):**
```text
Analyze the following frames from a video:
{images}

Generate a spatio-temporal scene graph describing the actions and relationships.
```

### 3. Run Generation

Use the following command to run the generation process:

```bash
python -m star_code.src.graph_gen \
  --model gemma3:4b-it-qat \
  --video-dir /path/to/videos \
  --videos-metadata /path/to/metadata.json \
  --output-file outputs/graph_generation.jsonl \
  --usr-prompt star_code/data/prompts/graph-gen/usr_prompt.txt \
  --auto-reply star_code/data/prompts/graph-gen/auto_reply.txt \
  --fps 1 \
  --batch-images
```

## Deep Dive

### Dataset Loading
The `videos-metadata` argument accepts both `.json` and `.jsonl` files. The script uses `preprocess_videos_metadata` to deduplicate entries based on `video_id`, `start`, and `end`.

### Prompt Formatting Logic
The `ImgPromptDecorator` class in `prompt_formatters.py` handles image insertion for batch processing.
-   It calculates the number of frames.
-   It generates a string like `\n\nImage 0:\n[img]\n\nImage 1:\n[img]...`.
-   It injects this string into the `{images}` field of the user prompt.

### Output Format
The output is saved as a JSONL file. Each line corresponds to a processed video/segment.

**Structure:**
```json
{
  "video_id": "001",
  "start": 0.0,
  "end": 10.0,
  "chat_history": [
    {
      "role": "user",
      "content": "Analyze... (prompt with images)"
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "stsg": "Frame 1:\nperson -> holding -> cup..."
}
```

-   **`stsg`**: Contains the extracted scene graph.
    -   In **Standard Mode**, this is an aggregation of individual frame responses, formatted by `frame_aggregator`.
    -   In **Batch Mode**, this is the direct output from the model (which should ideally generate the full graph for all frames).
