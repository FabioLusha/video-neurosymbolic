# Graph Understanding Module Documentation

This document explains the Graph Understanding pipeline (`star_code/src/main.py`), detailing how it loads data, formats prompts, and interacts with the model.

## Overview

The Graph Understanding module is designed to:
1.  **Load Datasets**: Ingest Question-Answering (QA) data and Spatio-Temporal Scene Graphs (STSG).
2.  **Format Prompts**: Dynamically insert data (questions, choices, scene graphs) into prompt templates.
3.  **Execute Pipeline**: Manage the flow of data to the LLM/VLM using `main.py`.

## Pipeline Execution (`main.py`)

The `main.py` script orchestrates the entire process. Here is how it works and the key parameters involved:

### 1. Initialization
The script starts by parsing command-line arguments to determine the task, model, and data sources.
-   `--task`: Defines the high-level goal (e.g., `graph-understanding`, `vqa`).
-   `--model`: Specifies the Ollama model to use (e.g., `gemma3:4b-it-qat`).
-   `--model-options`: Points to a JSON file with model hyperparameters (temperature, context window, etc.).

### 2. Dataset Loading & Merging
It initializes the dataset (specifically `STARDataset` for STAR data).
-   **QA Data** (`--input-file`): Loads questions and answers.
-   **Scene Graphs** (`--stsg-file`): Loads the generated scene graphs.
-   **Merging Logic**: The script merges these two sources. It matches questions to their corresponding scene graphs using `video_id` (and start/end times) or `question_id`. If a question cannot be linked to a scene graph, it may be filtered out depending on the logic.

### 3. Prompt Formatting
For each data point, a `PromptFormatter` is used to create the final string sent to the model.
-   `--prompt-type`: Selects the specific formatter (e.g., `mcq`, `open_qa`) and the default template structure.
-   `--user-prompt`: The actual template file containing placeholders like `{question}` or `{stsg}`.

### 4. Model Interaction
The pipeline processes prompts in one of two modes:
-   **Generate Mode** (`--mode generate`): Sends a single prompt and saves the response. Useful for simple QA.
-   **Chat Mode** (`--mode chat`): Maintains a conversation history.
    -   It sends the user prompt.
    -   It receives the model's response.
    -   It sends an **Auto-Reply** (`--reply-file`) to simulate a follow-up or enforce formatting.
    -   It saves the full chat history.

## Detailed Components

### Dataset Loading (STAR)
The `STARDataset` class handles the specific structure of the STAR benchmark.
-   **Input**: JSON or JSONL files.
-   **Structure**: It expects QA items to have keys like `question`, `choices`, and `video_id`.
-   **STSG Integration**: When `--stsg-file` is provided, it builds a lookup dictionary mapping `video_id` (or `question_id`) to the scene graph text. During iteration, it injects the `stsg` field into the sample data, making it available for the prompt formatter.

### Prompt Crafting

Prompts are text files with Python-style string placeholders.

**Common Placeholders:**
-   `{question}`: The text of the question.
-   `{stsg}`: The textual scene graph.
-   `{c1}`, `{c2}`, `{c3}`, `{c4}`: The multiple-choice options (for MCQ tasks).

**Example Template (`mcq` type):**
```text
Here is a scene graph describing a video:
{stsg}

Question: {question}
Options:
1. {c1}
2. {c2}
3. {c3}
4. {c4}

Answer with the option number.
```

### Implementing Custom Prompting

To implement your own prompting strategy, you need to:

1.  **Create a Prompt Template**: Write a text file with your desired structure and placeholders.
2.  **Define a Formatter (Optional)**: If your template uses standard fields (`question`, `stsg`, `choices`), you can reuse existing formatters like `MCQPrompt` or `OpenEndedPrompt`.
3.  **Implement Custom Logic (Advanced)**:
    -   Go to `star_code/src/prompt_formatters.py`.
    -   Create a new class inheriting from `PromptFormatter`.
    -   Implement `init_fields(self, sample)` to map dataset fields to your template placeholders.
    -   **Register the Formatter**: Add your new class to the `PROMPT_TYPES` dictionary in `star_code/src/_const.py`.

**Example Custom Formatter:**
```python
# In prompt_formatters.py
class MyCustomPrompt(PromptFormatter):
    def init_fields(self, sample):
        self.field_values = {
            "my_custom_field": sample["some_data_key"],
            "question": sample["question"]
        }

# In _const.py
PROMPT_TYPES = {
    ...,
    "my_custom_type": pf.MyCustomPrompt
}
```
You can then use `--prompt-type my_custom_type` in your command.

## Execution Examples

**Basic MCQ with Scene Graphs:**
```bash
python -m star_code.src.main \
  --task graph-understanding \
  --model gemma3:4b-it-qat \
  --prompt-type mcq \
  --mode chat \
  --dataset-type star \
  --input-file data/star_val.json \
  --stsg-file data/generated_graphs.jsonl \
  --user-prompt data/prompts/my_mcq_prompt.txt \
  --reply-file data/prompts/auto_reply.txt \
  --output-file outputs/results.jsonl
```
