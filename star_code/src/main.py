import argparse
import functools
import json
import os

import prompt_formatters as pf
from ollama_manager import OllamaRequestManager, STARPromptGenerator

SEED = 13471225022025

MODELS = [
    "llama3.2",
    "llama3.1:8b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b"
    "phi3:3.8b"
]

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
    mcq_pformat = '''\
        Q: {question}
        Alternatives:
        A. {c1}
        B. {c2}
        C. {c3}
        D. {c4}

        STSG: {stsg}
        A:
        '''
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
        "data/prompts/zero-shot-CoT/MCQ_system_prompt_ZS_CoT.txt"
        )
    user_prompt = _load_prompt_fromfile(
        "data/prompts/zero-shot-CoT/MCQ_user_prompt_ZS_CoT.txt"
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
    return llm_judge_sys_prompt, llm_judge_usr_prompt


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
    input_filepath=None,
    ids_filepath=None,
):
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": model_name,
            "system": system_prompt,
            "stream": True,
            "options": {
                "num_ctx": 10240,
                "temperature": 0.1,
                "num_predict": 8192,
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
    ollama.load_model()
    ollama.batch_requests(prompts=prompts)


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
        "--ids-file",
        help="File were to extract ids for which to run the model",
    )
    parser.add_argument(
        "--responses-file", help="File with the responses to be evaluated by the judge"
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
        input_filepath=input_file,
        ids_filepath=ids_file_path,
    )


if __name__ == "__main__":
    main()
