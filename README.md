# STAR Ollama Toolkit

A comprehensive toolkit for generating and understanding spatio-temporal scene graphs (STSG) using Ollama-powered language models.

## TODO
[ ] Add ollama options config file to be passed as argument
## Overview

This toolkit consists of two complementary modules:

1. **Graph Generation Module**: Extracts frames from videos and generates spatio-temporal scene graphs using vision-language models.
2. **Graph Understanding Module**: Processes existing scene graphs to answer questions and perform reasoning tasks.

Both modules use pre-configured prompts tailored for their specific tasks to ensure consistent model behavior.

## Requirements

- Python 3.6+
- Ollama running locally or on a server
- ffmpeg (for video processing in the Graph Generation module)
- Required Python packages (install via `pip install -r requirements.txt`)

## Setup

### Ollama
Download the ollama container:
```bash
docker pull ollama/ollama:latest
```
Run the ollama container:
```
docker compose -f ollama/ollama-compose.yaml up
```

### Dev env
Build the python devel contain in the dev_container directory (may require several minutes):
```bash
cd dev_container
docker build -t lusha/pydev
```

Run the container:
```bash
docker compose -f dev_container/compose.yaml up
```
Attach to the container:
```bash
docker exec -it <username>_pydev_env bash
```

Run the script (here the graph generation pipeline):
```bash
python star_code/src/graph_gen.py \
      --model gemma3:4b \
      --video-dir /multiverse/datasets/shared/action-genome/Charades_v1_480 \
      --videos-metadata /multiverse/datasets/shared/STAR/STAR_annotations/STAR_val.json \
      --output-file outputs/generated_stsg.jsonl \
      --usr-prompt star_code/data/prompts/graph_gen/usr_prompt.txt \
      --auto-reply star_code/data/prompts/graph_gen/format_instructions.txt \
      --max-samples 5
```

### Set-up details
Info about the compose files:  

1. Attach the `ollama_models` directory in `multiverse` to the `/.ollama`. The shared directory contain
the weights of a bunch of models used in the previous experiments. In this manner you don't need to 
download again the model
```bash
  -v /multiverse/datasets/shared/ollama_models:/.ollama
```

2.Use lusha/pydev as the container to run the script;
Set the `OLLAMA_URL` as the name of the running of the ollama container (specifying also the port):
```bash
  -e OLLAMA_URL=http://lusha_ollama:11435
```



## Graph Generation Module

A Vision-Language Model (VLM) powered pipeline that constructs Spatio-Temporal Scene Graphs (STSG) from video input through frame-by-frame analysis.

### Process Flow
1. **Frame Sampling**: Systematically extracts `--max-samples` evenly spaced frames from the input video
2. **Graph Construction**: For each sampled frame:
   - Analyzes visual elements using the specified VLM
   - Identifies objects, relationships, and temporal connections
   - Generates hierarchical scene graph descriptions with explicit frame markers
3. **Output Generation**: Stores results in JSON Lines format with:
   - `video_id`: Original video identifier
   - `chat_history`: Complete prompt chain used for graph generation
   - `stsg`: Textual representation containing:
     - Sequential frame entries marked with `Frame <frame_n>` headers
     - Per-frame object relationships in standardized notation

### STSG Textual Format
The scene graph representation follows this structure:
```text
Frame 1:
  person → holding → phone
  phone → on → table
  ...
Frame 2:
person → typing-on → laptop
laptop → placed-on → table
...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | **(Required)** Ollama model to use for image captioning |
| `--output-file` | **(Required)** Path to save the generated scene graph descriptions |
| `--video-dir` | **(Required)** Directory containing the videos to process |
| `[--ids-file]` | **(Optional)** Path to a file containing video IDs to process (one ID per line) |
| `[--max-samples]` | Maximum number of frames to sample per video (default: 10) |
| `[--sys-prompt]` | **(Optional)** Path to text file containing system prompt |
| `--usr-prompt` | **(Required)** Path to text file containing user prompt |
| `--auto-reply` | **(Required)** Path to text file containing auto-reply prompt |

### Usage Example

```bash
# Generate scene graphs from video frames
python star_code/src/generate_graphs.py \
  --model gemma3:4b \
  --video-dir data/videos \
  --output-file outputs/out_file.json
  --usr-prompt data/prompts/graph_gen/user_prompt.txt \
  --auto-reply data/prompts/graph_gen/auto_reply.txt \
  --max-samples 10
```

## Graph Understanding Module

### Arguments

| Argument | Description |
|----------|-------------|
| `--task` | Task type: `graph-understanding` (default) |
| `--prompt-type` | **(Required)** Type of prompt template: `open_qa`, `mcq`, `mcq_html`, `mcq_zs_cot`, `bias_check`, `judge` |
| `--model` | Ollama model to use |
| `--input-file` | Dataset file containing questions (JSON format) |
| `--stsg-file` | File with spatio-temporal scene graphs if not included in main dataset (JSONL format)|
| `--output-file` | File path to save model responses |
| `--ids-file` | Path to file with question IDs to process (one ID per line) |
| `--responses-file` | File with responses for evaluation by the judge (used with `judge` prompt type) |
| `--mode` | Run mode: `generate` (one-shot responses) or `chat` (conversation with reply) |
| `--reply-file` | File with text for automatic follow-up in chat mode |

### Prompt Types

- `open_qa`: Open-ended question answering with scene graphs
- `mcq`: Multiple-choice questions with standard formatting
- `mcq_html`: Multiple-choice questions using HTML-tag formatting
- `mcq_zs_cot`: Multiple-choice with zero-shot chain-of-thought reasoning
- `bias_check`: Questions without scene graphs to check model bias
- `judge`: LLM-as-judge evaluation of previous responses

### Usage Example

```bash
# Run open-ended QA with chain-of-thought reasoning in chat mode
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b \
  --prompt-type open_qa \
  --mode chat \
  --input-file star_code/notebooks/outputs/qa.json \
  --stsg-file star_code/notebooks/outputs/out_file.jsonl \
  --reply-file star_code/data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt \
  --output-file star_code/test_output_new_version.jsonl
```

### Additional Examples

```bash
# Run multiple-choice questions on the validation dataset
python main.py \
  --task graph-understanding \
  --prompt-type mcq \
  --model gemma3:4b \
  --output-file results/mcq_responses.json

# Evaluate model responses using LLM-as-judge
python main.py \
  --prompt-type judge \
  --model gemma3:27b \
  --responses-file results/mcq_responses.json \
  --output-file results/evaluation.json
```

## Complete Workflow Example

This example shows how to use both modules together in a complete workflow:

1. First, generate scene graphs from videos:
```bash
python star_code/src/generate_graphs.py \
  --model gemma3:4b \
  --video-dir data/videos \
  --output-file star_code/notebooks/outputs/out_file.jsonl \
  --usr-prompt data/prompts/graph_gen/user_prompt.txt \
  --auto-reply data/prompts/graph_gen/auto_reply.txt
```

2. Then, use the generated graphs to answer questions:
```bash
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b \
  --prompt-type open_qa \
  --mode chat \
  --input-file star_code/notebooks/outputs/qa.json \
  --stsg-file star_code/notebooks/outputs/out_file.jsonl \
  --reply-file star_code/data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt \
  --output-file star_code/test_output_new_version.jsonl
```

## Notes

- Both modules use pre-defined prompts stored in the `data/prompts/` directory
- Default datasets are located in `data/datasets/`
- Responses are streamed for real-time monitoring
- A fixed seed (13471225022025) is used for reproducibility
- The Graph Generation module requires ffmpeg to be installed for video frame extraction
- The generated scene graphs from the first module can be directly used as input to the second module

## Environment Variables

- `OLLAMA_URL`: URL to Ollama server (default: "http://localhost:11434")
