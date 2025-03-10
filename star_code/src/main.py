from ollama_manager import STARPromptGenerator, OllamaRequestManager
import prompt_formatters as pf

import os
import json

SEED = 13471225022025


def main():

    # system_prompt = _load_prompt_fromfile('data/system_prompt.txt')
    # prompt_format = "QUESTION: {question}\n"\
    #                 "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

    # PROMPT FOR MULTI-CHOICE QA
    # mcq_system_prompt = \
    #   _load_prompt_fromfile('data/prompts/MCQ_system_prompt_v2_oneshot.txt')
    # mcq_pformat = "Q: {question}\n"\
    #               "{c1}\n{c2}\n{c3}\n{c4}\n"\
    #               "STSG: {stsg}\n"\
    #               "A:"
    #
    # mcq_pfromatter = pf.MCQPrompt(mcq_pformat)

    # PROMPT FOR MULTI-CHOICE QA WITH HTML TAGS
    # mcq_system_prompt = \
    #     _load_prompt_fromfile('data/MCQ_system_prompt_v3.txt')
    # mcq_pformat = "<Question>\n"\
    #               "{question}\n"\
    #               "Alternatives:\n"\
    #               "{c1}\n{c2}\n{c3}\n{c4}\n"\
    #               "<\Question>\n"\
    #               "<STSG>\n{stsg}\n<\STSG>"

    # PROMPTS FOR BIAS CHECK - I.E. QUESTION WITHOUT STSG
    # mcq_system_prompt_bias = \
    #   _load_prompt_fromfile('data/prompts/system_prompt_bias_check.txt')
    # mcq_bias_pformat = "Q: {question}\n"\
    #                    "{c1}\n{c2}\n{c3}\n{c4}\n"\
    #                    "A:"
    #
    # mcq_bias_pfromatter = pf.MCQPromptWoutSTSG(mcq_bias_pformat)

    llm_judge_sys_prompt = _load_prompt_fromfile(
        'data/prompts/LLM_judge_system_v2.txt')
    llm_judge_usr_prompt = _load_prompt_fromfile(
        'data/prompts/LLM_judge_user_v2.txt')

    mispredictions_filepath = 'data/llama3b_extracted_ans.jsonl'
    judge_pformatter = pf.LlmAsJudgePrompt(
        llm_judge_usr_prompt, mispredictions_filepath)

    with open(mispredictions_filepath, 'r') as f:
        ids = [json.loads(line)['qid'] for line in f.readlines()]

    # Initialize Ollama manager
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    ollama = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            # 'model': 'llama3.2',
            'model': 'llama3.1:8b',
            # 'model': 'phi3:3.8b',
            # 'model': 'deepseek-r1:1.5b',

            # 'system': llm_judge_sys_prompt,
            'system': llm_judge_sys_prompt,

            'stream': True,
            'options': {
                'num_ctx': 10240,       # increasing the context window
                # less createive and more focuesed generation (default: 0.8)
                'temperature': 0.1,
                'num_predict': 8192,    # fixing the number of max output tokens
                'seed': SEED            # For reproducible results
            }
        }
    )

    # Initialize the prompt generator
    prompt_generator = STARPromptGenerator(
        # input_filename='data/datasets/STAR_question_and_stsg.json',    # Generative
        input_filename='data/datasets/STAR_QA_and_stsg_val.json',        # MCQ
    )

    # start from where the server crashed (repeat the last generation to
    # test start parm actually works)
    prompts = list(
        prompt_generator.generate(
            prompt_formatter=judge_pformatter,
            ids=ids
        )
    )

    # generate responses
    ollama.batch_requests(
        prompts=prompts
    )


def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")


if __name__ == "__main__":
    main()
