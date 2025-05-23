import os
from pathlib import Path

import prompt_formatters as pf

# Base directory and environment variables
BASE_DIR = Path(__file__).parent.parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SEED = 13471225022025

# Default prompt configurations
DEFAULT_PROMPTS = {
    "open_qa": (
        BASE_DIR / "data/prompts/system_prompt.txt",
        BASE_DIR / "data/prompts/open-qa/OPEN_QA_stsg_usr_prompt.txt",
    ),
    "mcq": (
        BASE_DIR / "data/prompts/mcq/MCQ_system_prompt_v2_oneshot.txt",
        BASE_DIR / "data/prompts/mcq/MCQ_usr_prompt.txt",
    ),
    "mcq_html": (
        BASE_DIR / "data/prompts/mcq/MCQ_system_prompt_v3.txt",
        BASE_DIR / "data/prompts/mcq/MCQ_usr_prompt_html_tags.txt",
    ),
    "mcq_zs_cot": (
        BASE_DIR / "data/prompts/zero-shot-cot/MCQ_system_prompt_ZS_CoT.txt",
        BASE_DIR / "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v2.txt",
    ),
    "bias_check": (
        BASE_DIR / "data/prompts/mcq/system_prompt_bias_check.txt",
        BASE_DIR / "data/prompts/mcq/MCQ_user_prompt_bias_check.txt",
    ),
    "judge": (
        BASE_DIR / "data/prompts/llm-as-jdudge/LLM_judge_system_v2.txt",
        BASE_DIR / "data/prompts/llm-as-judge/LLM_judge_user_v2.txt",
    ),
    "vqa": (
        BASE_DIR / "data/prompts/img_answer/system_prompt.txt",
        BASE_DIR / "data/prompts/img_answer/user_prompt.txt",
    ),
}

# Prompt type mappings
PROMPT_TYPES = {
    "open_qa": pf.OpenEndedPrompt,
    "mcq": pf.MCQPrompt,
    "mcq_html": pf.MCQPrompt,
    "mcq_zs_cot": pf.MCQPrompt,
    "bias_check": pf.MCQPromptWoutSTSG,
    "judge": pf.LlmAsJudgePrompt,
    "vqa": pf.MCQPromptWoutSTSG,
}

# Task types
TASK_TYPES = {"graph-gen": 0, "llm-judge": 0, "vqa": 0, "graph-understanding": 0}

# Default paths
DEFAULT_INPUT_FILE = BASE_DIR / "data/datasets/STAR/STAR_annotations/STAR_val.json"
DEFAULT_MODEL_OPTIONS = BASE_DIR / "ollama_model_options.json"
