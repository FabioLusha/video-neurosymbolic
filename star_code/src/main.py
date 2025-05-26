import argparse
import json
from os import system
from pathlib import Path

# relative imports work only with the 'from' form of the import
from . import batch_processor
from . import prompt_formatters as pf
from ._const import (BASE_DIR, DEFAULT_INPUT_FILE, DEFAULT_MODEL_OPTIONS,
                     DEFAULT_PROMPTS, OLLAMA_URL, PROMPT_TYPES, TASK_TYPES)
from .datasets import CVRRDataset, JudgeDataset, STARDataset
from .ollama_manager import OllamaRequestManager
from .STAR_utils.visualization_tools import vis_utils


def main():
    """Main entry point for the application."""
    # Step 1: Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM with different prompt types")

    parser.add_argument(
        "--task",
        choices=TASK_TYPES.keys(),
        default="graph-understanding",
        help="Choose the task to be performed",
    )
    parser.add_argument(
        "--prompt-type",
        choices=PROMPT_TYPES.keys(),
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--sys-prompt",
        help="Optional system prompt (pass 'default' to use default system prompt).",
    )
    parser.add_argument(
        "--user-prompt",
        help="User prompt (pass default to use 'defualt' prompt)",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Which model to use from those available in Ollama",
        required=True,
    )
    parser.add_argument(
        "--model-options", help="Path to a JSON file containing model options"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["star", "cvrr"],
        required=True,
        help="Type of dataset to use (STAR or CVRR)",
    )
    parser.add_argument("--input-file", help="Input dataset file path")
    parser.add_argument(
        "--ids-file",
        help="Path to a file containing question IDs to process (one ID per line)",
    )
    parser.add_argument(
        "--stsg-file",
        help="File with the spatio-temporal scene graphs if these are not included in the main dataset",
    )
    parser.add_argument(
        "--responses-file", help="File with the responses to be evaluated by the judge"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "chat"],
        help="How to run the model, 'chat' or 'generate' mode",
        default="generate",
    )
    parser.add_argument(
        "--reply-file",
        help="File with the text for the automatic reply when run in chat mode",
    )
    parser.add_argument("--output-file", help="file path where to save the response")

    args = parser.parse_args()

    # Step 2: Load prompts
    # Load system and user prompts based on arguments.
    system_prompt_path, user_prompt_path = DEFAULT_PROMPTS[args.prompt_type]

    system_prompt = None
    if args.sys_prompt:
        if args.sys_prompt != "default":
            system_prompt_path = args.sys_prompt
        system_prompt = _load_prompt_fromfile(system_prompt_path)

    # --user-prompt is a required argument
    if args.user_prompt != "default":
        user_prompt_path = args.user_prompt
    user_prompt = _load_prompt_fromfile(user_prompt_path)


    # Step 3: Create prompt formatter
    prompt_formatter = create_prompt_formatter(args, user_prompt)

    # Step 4: Initialize dataset and load prompts
    dataset = initialize_dataset(
        args, args.input_file or DEFAULT_INPUT_FILE, prompt_formatter, args.ids_file
    )
    prompts = [dataset[i] for i in range(len(dataset))]

    # Step 5: Load model options and initialize Ollama manager
    model_options = _load_model_options(args.model_options)
    ollama_client = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": args.model,
            "system": system_prompt,
            "stream": True,
            "options": model_options,
        },
    )

    # Step 6: Load model and process prompts
    ollama_client.load_model()
    process_prompts(ollama_client, prompts, args.mode, args, args.output_file)


def create_prompt_formatter(args, user_prompt):
    """Create the appropriate prompt formatter based on arguments."""
    return PROMPT_TYPES[args.prompt_type](user_prompt)


def _load_prompt_fromfile(filename):
    """Load prompt content from a file."""
    try:
        with open(filename) as in_file:
            return in_file.read().strip()
    except IOError as e:
        raise IOError(f"Error reading prompt file: {e}")


def _load_model_options(options_file=None):
    """Load model options from a JSON file."""
    options_file = options_file or DEFAULT_MODEL_OPTIONS
    try:
        with open(options_file) as in_file:
            return json.load(in_file)
    except IOError as e:
        raise IOError(
            f"Error reading the model's options file {options_file}: {e}"
        ) from e


def initialize_dataset(args, input_filepath, prompt_formatter, ids_filepath):
    """Initialize the appropriate dataset based on type."""
    if not input_filepath:
        input_filepath = BASE_DIR / "data/datasets/STAR_QA_and_stsg_val.json"

    ids = None
    if ids_filepath:
        print(f"=== Loading file with ids: {ids_filepath}")
        with open(ids_filepath, "r") as f:
            ids = [line.strip() for line in f.readlines()]
    else:
        print("=== No ids file chosen")

    print(f"=== Generating prompts from: {input_filepath}")

    # When using a LLMasJudge Dataset even if prompt_formatter is not suitable
    # for a STAR or CVRR dataset it is safe to intilize it with the llm-as-judge
    # prompt format because it will be overriden after, before being called.
    # The format function is called only when iterating through the elements of
    # the dataset.

    dataset = None
    if args.dataset_type == "star":
        dataset = STARDataset(
            input_filepath, prompt_formatter, stsg_file_path=args.stsg_file, ids=ids
        )
    elif args.dataset_type == "cvrr":
        dataset = CVRRDataset(
            input_filepath, prompt_formatter, stsg_file_path=args.stsg_file, ids=ids
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    if args.task == "llm-judge":
        print("=== Loading judge type dataset")
        dataset = JudgeDataset(dataset, args.responses_file , prompt_formatter)

    return dataset


def process_prompts(ollama_client, prompts, mode, args, output_filepath):
    """Process prompts based on the selected mode."""
    if mode == "generate":
        print("=== Mode: generate")
        batch_processor.batch_generate(
            ollama_client, prompts, output_file_path=output_filepath
        )
    elif mode == "chat":
        print("=== Mode: chat")
        if not args.reply_file:
            raise ValueError(
                "Chat mode requires a reply prompt file. Please provide one using the --reply-file parameter."
            )

        if args.task == "vqa":
            print("This feature is not implemented yet!!!")
            return
        else:
            reply = _load_prompt_fromfile(args.reply_file)
            batch_processor.batch_automatic_chat_reply(
                ollama_client, prompts, reply, output_file_path=output_filepath
            )
    else:
        print("Error: You must select one of the available modes: 'generate' or 'chat'")
        return


if __name__ == "__main__":
    main()
