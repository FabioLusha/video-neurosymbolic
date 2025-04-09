import argparse
import functools
import json
import os
import pathlib

import cv2

import prompt_formatters as pf
from ollama_manager import OllamaRequestManager, STARPromptGenerator
import batch_processor

SEED = 13471225022025

MODELS = ["llama3.2", 
          "llama3.1:8b", 
          "deepseek-r1:1.5b", 
          "deepseek-r1:7b",
          "phi3:3.8b",
          "gemma3:4b",
          "gemma3:12b"]


def load_open_qa_prompts():
    system_prompt = _load_prompt_fromfile("data/system_prompt.txt")
    prompt_format = "QUESTION: {question}\n" "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

    pformatter = pf.OpenEndedPrompt(prompt_format)
    return system_prompt, pformatter


def load_mcq_prompts():
    # PROMPT FOR MULTI-CHOICE QA
    mcq_system_prompt = _load_prompt_fromfile(
        "data/prompts/MCQ_system_prompt_v2_oneshot.txt"
    )
    mcq_pformat = """\
        Q: {question}
        Alternatives:
        A. {c1}
        B. {c2}
        C. {c3}
        D. {c4}

        STSG: {stsg}
        A:
        """
    mcq_pformatter = pf.MCQPrompt(mcq_pformat)

    return mcq_system_prompt, mcq_pformatter


def load_mcq_html_prompts():
    # PROMPT FOR MULTI-CHOICE QA WITH HTML TAGS
    mcq_system_prompt = _load_prompt_fromfile("data/MCQ_system_prompt_v3.txt")
    mcq_pformat = """\
        <STSG>\n{stsg}\n<\STSG>
        <Question>
        {question}
        Alternatives:
        A. {c1}
        B. {c2}
        C. {c3}
        D. {c4}
        <\Question>
        """

    mcq_pformatter = pf.MCQPrompt(mcq_pformat)

    return mcq_system_prompt, mcq_pformatter


def load_mcq_zs_cot_prompts():
    sys_prompt = _load_prompt_fromfile(
        "data/prompts/zero-shot-cot/MCQ_system_prompt_ZS_CoT.txt"
    )
    user_prompt = _load_prompt_fromfile(
        # "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT.txt"
        "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v2.txt"
    )

    mcq_pformatter = pf.MCQPrompt(user_prompt)

    return sys_prompt, mcq_pformatter


def load_bias_check_prompts():
    # PROMPTS FOR BIAS CHECK - I.E. QUESTION WITHOUT STSG
    mcq_system_prompt_bias = _load_prompt_fromfile(
        "data/prompts/system_prompt_bias_check.txt"
    )
    mcq_bias_pformat = "Q: {question}\n" "{c1}\n{c2}\n{c3}\n{c4}\n" "A:"
    mcq_bias_pfromatter = pf.MCQPromptWoutSTSG(mcq_bias_pformat)

    return mcq_system_prompt_bias, mcq_bias_pfromatter


def load_llm_as_judge_prompts(responses_filepath):
    llm_judge_sys_prompt = _load_prompt_fromfile("data/prompts/LLM_judge_system_v2.txt")
    llm_judge_usr_prompt = _load_prompt_fromfile("data/prompts/LLM_judge_user_v2.txt")

    judge_pformatter = pf.LlmAsJudgePrompt(llm_judge_usr_prompt, responses_filepath)
    return llm_judge_sys_prompt, judge_pformatter


def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")


def run_with_prompts(
    system_prompt,
    prompt_formatter,
    model_name,
    mode='generate',
    input_filepath=None,
    ids_filepath=None,
    output_filepath=None
):
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_client = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": model_name,
            "system": system_prompt,
            "stream": True,
            "options": {
                "num_ctx": 10240,
                "temperature": 0.1,
                "num_predict": 2048,
                "seed": SEED,
            },
        },
    )

    # Initialize the prompt generator with the appropriate file
    if not input_filepath:
        input_filepath = "data/datasets/STAR_QA_and_stsg_val.json"  # Default to MCQ

    # Handle ids file
    ids = None
    if ids_filepath:
        with open(ids_filepath, "r") as f:
            ids = [json.loads(line)["qid"] for line in f.readlines()]

    prompt_generator = STARPromptGenerator(input_filename=input_filepath)

    # Generate prompts
    prompts = list(
        prompt_generator.generate(prompt_formatter=prompt_formatter, ids=ids)
    )

    # Generate responses
    ollama_client.load_model()
    if mode == 'generate':
        print('=== Mode: generate')
        batch_processor.batch_generate(
            ollama_client,
            prompts,
            output_file_path=output_filepath
        )

    elif mode == 'chat':
        print('=== Mode: chat')
        reply = \
        """\
        Therefore the final answer is?
        
        Your response must be provided in valid JSON format as follows:
        {"answer": "your complete answer here"}
        
        IMPORTANT: Always include both the letter (A, B, C, D, etc.) AND the full text of the answer in your response.
        Do not abbreviate or shorten the answer. For example, if the correct answer is "A. the laptop", your response 
        should be {"answer": "A. the laptop"}, not {"answer": "laptop"} or {"answer": "A"}.\
        """

        batch_processor.batch_automatic_chat_reply(
            ollama_client,
            prompts,
            reply,
            output_file_path=output_filepath
        )



def main():
    # Define available prompt types
    prompt_types = {
        "open_qa": load_open_qa_prompts,
        "mcq": load_mcq_prompts,
        "mcq_html": load_mcq_html_prompts,
        "mcq_zs_cot": load_mcq_zs_cot_prompts,
        "bias_check": load_bias_check_prompts,
        "judge": load_llm_as_judge_prompts,
    }

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with different prompt types")
    parser.add_argument(
        "--prompt-type",
        choices=prompt_types.keys(),
        required=True,
        help="Type of prompt to use",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2:latests - 3b)",
    )
    parser.add_argument(
        "--input-file", help="Input dataset file path (defaults based on prompt type)"
    )
    parser.add_argument(
        "--output-file", help="file path where to save the response)"
    )
    parser.add_argument(
        "--ids-file",
        help="File were to extract ids for which to run the model",
    )
    parser.add_argument(
        "--responses-file", help="File with the responses to be evaluated by the judge"
    )
    parser.add_argument(
        "--mode",
        choices=['generate', 'chat'],
        help="How to run the model, 'chat' or 'generate' mode"
    )

    args = parser.parse_args()

    # Set default input file based on prompt type
    input_file = args.input_file
    if not input_file:
        if args.prompt_type in ["open_qa"]:
            input_file = "data/datasets/STAR_question_and_stsg.json"
        else:
            input_file = "data/datasets/STAR_QA_and_stsg_val.json"

    # Special handling for judge prompt type
    responses_file = None
    if args.prompt_type == "judge":
        responses_file = args.responses_file
        prompt_types["judge"] = functools.partial(prompt_types["judge"], responses_file)

    # Load the selected prompts
    load_func = prompt_types[args.prompt_type]
    system_prompt, prompt_formatter = load_func()

    ids_file_path = args.ids_file
    # Run with the selected configuration
    run_with_prompts(
        system_prompt=system_prompt,
        prompt_formatter=prompt_formatter,
        model_name=args.model,
        mode=args.mode,
        input_filepath=input_file,
        output_filepath=args.output_file,
        ids_filepath=ids_file_path,
    )

def generate_frames(max_sample=10):
    from STAR_utils.visualization_tools import vis_utils
    
    def load_frames(frame_list, frame_dir):
        select = []
        for i in range(len(frame_list)):
            frame = cv2.imread(frame_dir / f"{frame_list[i]}.png")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            select.append(frame)

        return select

    star_data = []
    with open("../data/datasets/STAR/STAR_annotations/STAR_val.json") as in_file:
        star_data = json.load(in_file)
        
    raw_frame_dir = pathlib.Path('../data/datasets/action-genome/frames/')
    
    for sample in star_data:
        frame_ids = vis_utils.sample_frames(list(sample['situations'].keys()), max_sample)
        frame_ids
        
        frame_dir = raw_frame_dir / f"{sample['video_id']}.mp4"
        frames = load_frames(frame_ids, frame_dir)
        
        yield sample['question_id'], frames

def streaming_frame_generation(ollama_client, frames, output_file):
    def
    
if __name__ == "__main__":
    main()

prompt1 = """\
    Look carefully at this image and identify all objects and relationships present.

    First, list all distinct objects you can detect in the image. Be thorough and specific with your object labels (e.g., "young woman" rather than just "person", "wooden chair" rather than just "chair").

    Then, describe the key relationships between these objects in free-form text. Consider:
    - Spatial relationships (above, below, behind, inside, etc.)
    - Action-based relationships (holding, looking at, sitting on, etc.)
    - Physical connections (attached to, part of, touching, etc.)
    - Relative positions (next to, between, surrounding, etc.)

    Think step by step.\
"""

prompt2 = """\
    Thank you. Now organize the objects and relationships you identified into a formal scene graph using this format:
    object1 ---- relationship ---- object2
    
    The list of relationship predicates should be introduced by the tag <scene_graph> and terminated by the tag </scene_graph>
    For example:
    woman ---- sitting_on ---- chair
    dog ---- lying_under ---- table
    book ---- on_top_of ---- shelf

    Please follow these guidelines:
    1. Create at least 10 relationship triplets (more if the image is complex)
    2. Use specific and consistent object labels
    3. Use concise but descriptive relationship terms (connect words with underscores)
    4. Include all meaningful relationships between objects
    5. Verify that all objects you identified in step 1 appear in at least one relationship

    Your scene graph:\
    """