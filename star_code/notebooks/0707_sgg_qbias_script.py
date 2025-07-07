import os, sys
import json
from pathlib import Path
from importlib import reload

import matplotlib.pyplot as plt
import base64
import io

import numpy as np
import pandas as pd
import PIL

WORK_DIR = Path.cwd().parent

sys.path.append(str(WORK_DIR))

from src import (
    main,
    datasets,
    video_tools,
    ollama_manager,
    prompt_formatters as pf,
    _const,
)
from src.STAR_utils.visualization_tools import qa_visualization as qaviz


STAR_SMALL = WORK_DIR / "data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
RAW_VIDEO_DIR = Path(WORK_DIR / 'data/datasets/action-genome/Charades_v1_480/')


# intialize model
model_name = "gemma3:4b-it-qat"
model_options_path = WORK_DIR / "ollama_model_options.json"
model_options = main._load_model_options(str(model_options_path))

system_prompt = None
ollama_client = ollama_manager.OllamaRequestManager(
    base_url=_const.OLLAMA_URL,
    ollama_params={
        "model": model_name,
        "system": system_prompt,
        "stream": True,
        "options": model_options,
    },
)

# Prompt with only questions
user_prompt = main._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/q-bias/user_prompt.txt")
user_pformatter = pf.OpenEndedPrompt(user_prompt)
output_filepath = WORK_DIR / "outputs/sgg/sgg_qbias_onlyq_gemma3:4b-it-qat_20250707_21:59:00.jsonl"

# Prompt with alternatives
# user_prompt = main._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/q-bias/user_prompt_alt.txt")
# output_filepath = WORK_DIR / "outputs/sgg/sgg_qbias_alt1_gemma3:4b-it-qat_20250707_21:59:00.jsonl"
# # + None ot the others alt.
# # user_prompt = main._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/q-bias/user_prompt_alt2.txt")
# # output_filepath = WORK_DIR / "outputs/sgg/sgg_qbias_alt2_gemma3:4b-it-qat_20250707_21:59:00.jsonl"
# user_pformatter = pf.MCQPromptWoutSTSG(user_prompt)

star_dataset = datasets.STARDataset(
    STAR_SMALL,
    user_pformatter,
)

follow_up = main._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/q-bias/format_instructions.txt")


fps = 1
main.stream_sgg(
    ollama_client=ollama_client,
    dataset=star_dataset,
    videos_dir=RAW_VIDEO_DIR,
    reply=follow_up,
    fps=fps,
    output_filepath=output_filepath,
    batch_images=True,
    max_frames=None
)
