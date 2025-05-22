# STAR Ollama Toolkit

A comprehensive toolkit for generating and understanding spatio-temporal scene graphs (STSG) using Ollama-powered language models.

## Features

- Generate spatio-temporal scene graphs from video content
- Process existing scene graphs for question answering and reasoning
- Compatible with various LLMs through Ollama
- Configurable prompting strategies for different tasks

## TODO

- [x] Add Ollama options config file to be passed as argument
- [x] Add functionality to pass system/user/reply prompts from arguments
- [ ] Support for batch processing of multiple videos
- [ ] Add documentation with examples of prompt specifications for each prompt type

## Overview

This toolkit consists of two complementary modules:

1. **Graph Generation Module**: Extracts frames from videos and generates spatio-temporal scene graphs using vision-language models.
2. **Graph Understanding Module**: Processes existing scene graphs to answer questions and perform reasoning tasks.

Both modules use pre-configured prompts tailored for their specific tasks to ensure consistent model behavior.

## Requirements

- Python 3.6+
- Ollama running locally or on a server
- ffmpeg (for video processing in the Graph Generation module)
- Docker and Docker Compose (for containerized setup)
- Required Python packages (install via `pip install -r requirements.txt`)

## Setup

### Repository Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/FabioLusha/video-neurosymbolic.git
   cd video-neurosymbolic
   ```

### Ollama Setup

1. Download the Ollama container:
   ```bash
   docker pull ollama/ollama:latest
   ```

2. Run the Ollama container:
   ```bash
   docker compose -f ollama/ollama-compose.yaml up -d
   ```

### Development Environment

1. Build the Python development container (contains all dependencies):
   ```bash
   cd dev_container
   docker build -t lusha/pydev .
   ```

2. Run the container:
   Look carefully at the mounting points:
   - Option 1: If you have access to the multiverse server, use the shared datasets:
     - The compose file is configured to mount datasets from `/multiverse/datasets/shared/`
     - I have given read permission to the my private partition `/multiverse/datasets/lusha`
   - Option 2: If you don't have access to multiverse, download the datasets locally:
     - Download ActionGenome and STAR dataset following the instruction from: [STAR Dataset](https://bobbywu.com/STAR/)
     - Update the volume mounts in `dev_container/compose.yaml` to point to your local dataset paths

   The mounting points are relative to the container environment:
 
   ```yaml
   networks:
     default:
       name: ${USER}_ollama_net
 
   services:
     env_container:
       image: lusha/pydev
       container_name: ${USER}_pydev_env
       env_file:
         - .env
       user: "${UID}:${UID}"
       stdin_open: true  # same as -i
       tty: true         # same as -t
       command: /bin/bash
       ports: # HOST:CONTAINER
         - "10882:8888"
       volumes:
         - ../:/home/${USER}
         - /multiverse/datasets/lusha:/home/${USER}/star_code/data/datasets
         - /multiverse/datasets/shared/action-genome:/home/${USER}/star_code/data/datasets/action-genome
         - /multiverse/datasets/shared/action-genome/Charades_v1_480:/home/${USER}/star_code/data/datasets/action-genome/Charades_v1_480
         - /multiverse/datasets/shared/STAR:/home/${USER}/star_code/data/datasets/STAR
       deploy:
         resources:
           reservations:
             devices:
               - capabilities: [gpu]
                 device_ids: ["3"]
       cpuset: "0-7"
   ```

   ```bash
   docker compose -f dev_container/compose.yaml up -d
   ```

3. Attach to the container:
   ```bash
   docker exec -it ${USER}_pydev_env bash
   ```

### Configuration Details

#### Critical Container Configuration

When running Ollama in a container, you **must** set the `OLLAMA_URL` environment variable to point to the container name where Ollama is running:

```yaml
environment:
  - OLLAMA_URL=http://${USER}_ollama:11435
```

> **Important**: Replace `lusha_ollama` with the actual name of your running Ollama container. The toolkit will fail to connect to Ollama if this configuration is incorrect.

#### Additional Docker Compose Settings

The Docker Compose setup also includes:

1. Pre-loaded models directory to avoid re-downloading models:
   ```yaml
   volumes:
     - /multiverse/datasets/shared/ollama_models:/.ollama
   ```

This volume mapping allows you to reuse existing model files across container restarts.

## Quick Usage

### Graph Generation

Run the graph generation pipeline:

```bash
python star_code/src/graph_gen.py \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --video-dir star_code/data/datasets/action-genome/Charades_v1_480 \
  --videos-metadata star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --output-file outputs/generated_stsg.jsonl \
  --usr-prompt star_code/data/prompts/graph_gen/usr_prompt.txt \
  --auto-reply star_code/data/prompts/graph_gen/format_instructions.txt \
  --max-samples 5

```

### Graph Understanding

Process generated STSGs for question answering:

```bash
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --prompt-type mcq_zs_cot \
  --mode chat \
  --input-file star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --stsg-file star_code/data/datasets/STAR_QA_and_stsg_val.json \
  --reply-file star_code/data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt \
  --output-file gemma3_4b_qa.jsonl
```

> **Note**: You can customize the prompts by using `--usr-prompt` and `--sys-prompt` arguments. If not specified, default prompts will be used. To disable the system prompt, pass an empty string `""` to `--sys-prompt`. Model options can be customized through the `--model-options` argument, which should point to a JSON file containing the desired parameters. If not specified, it will look for `ollama_model_options.json` in the base directory.

## Some useful commands for ollama

Here are some useful commands for managing Ollama in the containerized environment:

```bash
# List all available models
docker exec ${USER}_ollama ollama list

# Pull a new model (e.g., gemma3:4b-it-qat)
docker exec ${USER}_ollama ollama pull gemma3:4b-it-qat

# Remove a model
docker exec ${USER}_ollama ollama rm gemma3:4b-it-qat

# Show model information
docker exec ${USER}_ollama ollama show gemma3:4b-it-qat

# Run a model in interactive mode
docker exec -it ${USER}_ollama ollama run gemma3:4b-it-qat
```

> **Note**: `${USER}` should replace your username or the actual container name if different. The container name follows the pattern `username_ollama` as defined in the Docker Compose configuration.


## Project Structure

```
├── star_code/
│   ├── src/
│   │   ├── graph_gen.py        # Graph generation module
│   │   ├── main.py             # Main entry point
│   │   └── ...
│   ├── data/
│       ├── prompts/            # Prompt templates
│       │   ├── graph_gen/
│       │   └── zero-shot-cot/
│       └── datasets/           # Sample datasets
├── ollama/                     # Ollama configuration
├── dev_container/              # Development environment
└── requirements.txt           # Python dependencies
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
   - `start` and `end`: Start and end time metadat to reference a sub-sequence in video
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
| `--videos-metadata` | **(Optional)** A JSON file containing video-ids and metadata (start/end times) specifying which parts of videos to process |
| `[--max-samples]` | Maximum number of frames to sample per video (default: 10) |
| `[--sys-prompt]` | **(Optional)** Path to text file containing system prompt (default: empty) |
| `--usr-prompt` | **(Required)** Path to text file containing user prompt |
| `--auto-reply` | **(Required)** Path to text file containing auto-reply prompt |
| `--model-options` | **(Optional)** Path to JSON file containing model options |

> **Note**: For the graph generation module, the `--usr-prompt` and `--auto-reply` files are handled as plain text without any specific formatting requirements. The content is passed directly to the model as-is.

### Usage Example

```bash
# Generate scene graphs from video frames
python star_code/src/graph_gen.py \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --video-dir data/datasets/action-genome/Charades_v1_480 \
  --output-file outputs/out_file.json \
  --usr-prompt data/prompts/graph_gen/usr_prompt.txt \
  --auto-reply data/prompts/graph_gen/format_instructions.txt \
  --max-samples 10
```

## Graph Understanding Module

### Arguments

| Argument | Description |
|----------|-------------|
| `--task` | Task type: `graph-understanding` (default) |
| `--prompt-type` | Type of prompt template: `open_qa`, `mcq`, `mcq_html`, `mcq_zs_cot`, `bias_check`, `judge` |
| `--model` | Ollama model to use |
| `--input-file` | Dataset file containing questions (JSON format) |
| `--stsg-file` | File with spatio-temporal scene graphs if not included in main dataset (JSONL format)|
| `--output-file` | File path to save model responses |
| `--ids-file` | Path to file with question IDs to process (one ID per line) |
| `--responses-file` | File with responses for evaluation by the judge (used with `judge` prompt type) |
| `--mode` | Run mode: `generate` (one-shot responses) or `chat` (conversation with reply) |
| `--reply-file` | File with text for automatic follow-up in chat mode |
| `--usr-prompt` | Path to file containing custom user prompt. If not specified, default prompt will be used |
| `--sys-prompt` | Path to file containing custom system prompt. Pass empty string `""` to disable system prompt. If not specified, default prompt will be used |
| `--model-options` | **(Optional)** Path to JSON file containing model options |

### Prompt Types

- `open_qa`: Open-ended question answering with scene graphs
- `mcq`: Multiple-choice questions with standard formatting
- `mcq_html`: Multiple-choice questions using HTML-tag formatting
- `mcq_zs_cot`: Multiple-choice with zero-shot chain-of-thought reasoning
- `bias_check`: Questions without scene graphs to check model bias
- `judge`: LLM-as-judge evaluation of previous responses

> **Important**: When providing custom user prompts via `--usr-prompt`, ensure they follow the format and requirements of the selected `--prompt-type`. Each prompt type has specific formatting needs and expected input/output structures that must be respected for proper functioning.

### Usage Example

```bash
# Run open-ended QA with chain-of-thought reasoning in chat mode
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --prompt-type mcq_zs_cot \
  --mode chat \
  --input-file star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --stsg-file star_code/data/datasets/STAR_QA_and_stsg_val.json \
  --reply-file star_code/data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt \
  --output-file gemma3_4b_qa.jsonl 
```

### Additional Examples

```bash
# Run multiple-choice questions on the validation dataset
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b \
  --prompt-type mcq \
  --input-file star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --stsg-file star_code/data/datasets/STAR_QA_and_stsg_val.json \
  --output-file results/mcq_responses.json
```

## Complete Workflow Example

This example shows how to use both modules together in a complete workflow:

1. First, generate scene graphs from videos:
```bash
python star_code/src/graph_gen.py \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --video-dir star_code/data/datasets/action-genome/Charades_v1_480 \
  --videos-metadata star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --output-file outputs/generated_stsg.jsonl \
  --usr-prompt star_code/data/prompts/graph_gen/usr_prompt.txt \
  --auto-reply star_code/data/prompts/graph_gen/format_instructions.txt \
  --max-samples 5 
```

2. Then, use the generated graphs to answer questions:
```bash
python star_code/src/main.py \
  --task graph-understanding \
  --model gemma3:4b-it-qat \
  --model-options star_code/ollama_model_options.json \
  --prompt-type mcq_zs_cot \
  --mode chat \
  --input-file star_code/data/datasets/STAR/STAR_annotations/STAR_val.json \
  --stsg-file outputs/generated_stsg.jsonl \
  --reply-file star_code/data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt \
  --output-file gemma3_4b_qa.jsonl 
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
