import argparse
import base64
import functools
import json
import os
import re
from pathlib import Path

import batch_processor
import prompt_formatters as pf
from ollama_manager import OllamaRequestManager, STARDataset
from STAR_utils.visualization_tools import vis_utils

SEED = 13471225022025

BASE_DIR = Path(__file__).parent.parent

def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()
    except IOError as e:
        raise IOError(f"Error reading prompt file: {e}")

def _load_model_options(options_file=None):
    options_file = options_file or BASE_DIR / "ollama_model_options.json"
    try:
        with open(options_file) as in_file:
            return json.load(in_file)
    except IOError as e:
        raise IOError(f"Error reading the model's options file {options_file}: {e}") from e

def run_with_prompts(
    args,
    system_prompt,
    prompt_formatter,
    model_name,
    mode="generate",
    input_filepath=None,
    ids_filepath=None,
    output_filepath=None,
):
    # Load model options
    model_options = _load_model_options(args.model_options)
    
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_client = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": model_name,
            "system": system_prompt,
            "stream": True,
            "options": model_options,
        },
    )

    # Initialize the prompt generator with the appropriate file
    if not input_filepath:
        input_filepath = (
            BASE_DIR / "data/datasets/STAR_QA_and_stsg_val.json"
        )  # Default to MCQ

    # Handle ids file
    ids = None
    if ids_filepath:
        print(f"=== Loading file with ids: {ids_filepath}")
        with open(ids_filepath, "r") as f:
            ids = [line.strip() for line in f.readlines()]
    else:
        print("=== No ids file chosen")

    print(f"=== Generating prompts from: {input_filepath}")
    dataset = STARDataset(input_filepath, prompt_formatter, stsg_file_path=args.stsg_file, ids=ids)
    prompts = [dataset[i] for i in range(len(dataset))]

    # Generate responses
    ollama_client.load_model()

    if mode == "generate":
        print("=== Mode: generate")
        batch_processor.batch_generate(
            ollama_client, prompts, output_file_path=output_filepath
        )
    elif mode == "chat":
        print("=== Mode: chat")
        if args.reply_file:
            reply = _load_prompt_fromfile(args.reply_file)
        else:
            raise ValueError(
                "Chat mode requires a reply prompt file. Please provide one using the --reply-file parameter.")
        if args.task == "vqa":
            # TODO: Implement the feature and remove the print
            print("This feature is not implemented yet!!!")
            # stream_vqa(
            #     ollama_client, prompts, ids, reply, output_filepath=output_filepath
            # )
            return
        else:
            batch_processor.batch_automatic_chat_reply(
                ollama_client, prompts, reply, output_file_path=output_filepath
            )
    else:
        print("Error: You must select one of the available modes: 'generate' or 'chat'")
        return

def main():
    # Define available prompt types and their corresponding formatter classes
    default_prompts = {
        "open_qa": (
            BASE_DIR / "data/prompts/system_prompt.txt",
            BASE_DIR / "data/prompts/open-qa/OPEN_QA_stsg_usr_prompt.txt"
        ),
        "mcq": (
            BASE_DIR / "data/prompts/mcq/MCQ_system_prompt_v2_oneshot.txt",
            BASE_DIR / "data/prompts/mcq/MCQ_usr_prompt.txt"
        ),
        "mcq_html": (
            BASE_DIR / "data/prompts/mcq/MCQ_system_prompt_v3.txt",
            BASE_DIR / "data/prompts/mcq/MCQ_usr_prompt_html_tags.txt"
        ),
        "mcq_zs_cot": (
            BASE_DIR / "data/prompts/zero-shot-cot/MCQ_system_prompt_ZS_CoT.txt",
            BASE_DIR / "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v2.txt"
        ),
        "bias_check": (
            BASE_DIR / "data/prompts/mcq/system_prompt_bias_check.txt",
            BASE_DIR / "data/prompts/mcq/MCQ_user_prompt_bias_check.txt"
        ),
        "judge": (
            BASE_DIR / "data/prompts/llm-as-jdudge/LLM_judge_system_v2.txt",
            BASE_DIR / "data/prompts/llm-as-judge/LLM_judge_user_v2.txt"
        ),
        "vqa": (
            BASE_DIR / "data/prompts/img_answer/system_prompt.txt",
            BASE_DIR / "data/prompts/img_answer/user_prompt.txt"
        ),
    }

    prompt_types = {
        "open_qa": pf.OpenEndedPrompt,
        "mcq": pf.MCQPrompt,
        "mcq_html": pf.MCQPrompt,
        "mcq_zs_cot": pf.MCQPrompt,
        "bias_check": pf.MCQPromptWoutSTSG,
        "judge": pf.LlmAsJudgePrompt,
        "vqa": pf.MCQPromptWoutSTSG,
    }

    task_types = {"graph-gen": 0, "vqa": 0, "graph-understanding": 0}
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with different prompt types")

    parser.add_argument(
        "--task",
        choices=task_types.keys(),
        default="graph-understanding",
        help="Choose the task to be performed",
    )
    parser.add_argument(
        "--prompt-type",
        choices=prompt_types.keys(),
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--sys-prompt",
        help="Optional system prompt (overrides default). Use empty string for no system prompt.")
    parser.add_argument(
        "--user-prompt",
        help="Optional user prompt (overrides default)"
    )
    parser.add_argument(
        "--model",
        help="Which model to use from those available in Ollama",
        required=True
    )
    parser.add_argument(
        "--model-options",
        help="Path to a JSON file containing model options"
    )
    parser.add_argument(
        "--input-file",
        help="Input dataset file path (STAR dataset specification)"
    )
    parser.add_argument(
        "--stsg-file",
        help="File with the spatio-temporal scene graphs if these are not included in the main dataset"
    )
    parser.add_argument(
        "--output-file",
        help="file path where to save the response"
    )
    parser.add_argument(
        '--ids-file',
        help='Path to a file containing question IDs to process (one ID per line)'
    )
    parser.add_argument(
        "--responses-file",
        help="File with the responses to be evaluated by the judge"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "chat"],
        help="How to run the model, 'chat' or 'generate' mode",
        default="generate"
    )
    parser.add_argument(
        "--reply-file",
        help="File with the text for the automatic reply when run in chat mode",
    )

    args = parser.parse_args()

    # Set default input file based on prompt type
    input_file = args.input_file
    if not input_file:
        input_file = BASE_DIR / "data/datasets/STAR/STAR_annotations/STAR_val.json"
        print(f"Using default: {input_file}")

  
    # Handle ids file
    # TODO: Add support for ids file
    ids_file_path = args.ids_file
    ids = None
    if ids_file_path:
        with open(ids_file_path, "r") as f:
            ids = [line.strip() for line in f.readlines()]
            
    # Load the selected prompts
    system_prompt_path, user_prompt_path = default_prompts[args.prompt_type]
    
    # Override default prompts if provided
    if args.sys_prompt:
        system_prompt_path = args.sys_prompt
    if args.user_prompt:
        user_prompt_path = args.user_prompt

    system_prompt = _load_prompt_fromfile(system_prompt_path)
    user_prompt = _load_prompt_fromfile(user_prompt_path)

    # Special handling for judge prompt type
    responses_file = None
    if args.prompt_type == "judge":
        responses_file = args.responses_file
        prompt_formatter = prompt_types[args.prompt_type](user_prompt, responses_filepath=responses_file)
    else:
        prompt_formatter = prompt_types[args.prompt_type](user_prompt)

    # Run with the selected configuration
    run_with_prompts(
        args=args,
        system_prompt=system_prompt,
        prompt_formatter=prompt_formatter,
        model_name=args.model,
        mode=args.mode,
        input_filepath=input_file,
        output_filepath=args.output_file,
        ids_filepath=ids_file_path,
    )

#def stream_vqa(ollama_client, prompts, ids, reply, output_filepath, iters=-1):
#
#    def payload_gen(situations):
#        prompts_dict = {p["qid"]: p["prompt"] for p in prompts}
#        for situation in situations:
#            frame_encodings = [frame["encoding"] for frame in situation]
#
#            req_obj = {
#                "qid": situation[0][
#                    "question_id"
#                ],  # situation is list of [{question_id, frame_id, encoding}]
#                "payload": {
#                    **ollama_client.ollama_params,
#                    "messages": [
#                        {
#                            "role": "user",
#                            "content": prompts_dict[situation[0]["question_id"]],
#                            "images": frame_encodings,
#                        }
#                    ],
#                },
#            }
#
#            yield req_obj
#
#    bp = batch_processor
#    pipe = bp.Pipeline(
#        # the first generator converts the prompt to the right format
#        payload_gen,
#        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, "chat"),
#        lambda stream: (o for o in stream if o["status"] == "ok"),
#        lambda resp_gen: bp.auto_reply_gen(resp_gen, reply),
#        lambda resp_gen: bp.stream_save(
#            resp_gen, bp.ChatResponseFormatter(), output_filepath
#        ),
#    )
#
#    situations = (
#        situation_frames
#        for situation_frames in generate_frames(ids=ids, iters=iters, max_sample=10)
#    )
#    pipe.consume(situations)
#    return


if __name__ == "__main__":
    main()
