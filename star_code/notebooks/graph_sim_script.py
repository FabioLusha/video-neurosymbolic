import os, sys
from pathlib import Path
import json

import pandas as pd

WORK_DIR = Path.cwd().parent

sys.path.append(str(WORK_DIR))

from src import (
    main,
    graph_gen,
    ollama_manager,
    prompt_formatters,
    batch_processor
)
from src.datasets import STARDataset


STAR_VAL_PATH = WORK_DIR / "data/datasets/STAR/STAR_annotations/STAR_val.json"
GEN_GRAPH_PATH = WORK_DIR / "outputs/gen_stsg_gemma3:12b-it-qat_20250522_14:30_99.jsonl"
GT_GRAPH_PATH = WORK_DIR / "data/datasets/STAR_verbalized_stsg_val.json"

user_promp_format = main._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-sim/usr_prompt_v2.txt")
pformatter = prompt_formatters.PromptFormatter(user_promp_format, fields=['gen-stsg', 'gt-stsg'])

star = []
with open(STAR_VAL_PATH, 'r') as f:
    star = json.load(f)
    
star_df = pd.DataFrame(star)


gen_stsg = []
with open(GEN_GRAPH_PATH, 'r') as f:
    gen_stsg = [json.loads(line) for line in f.readlines()]
    
gen_stsg_df = pd.DataFrame(gen_stsg)
gen_stsg_df.iloc[0]


gt_stsg = []
with open(GT_GRAPH_PATH, 'r') as f:
    gt_stsg = json.load(f)
    

gt_stsg_df = pd.DataFrame(gt_stsg)
gt_stsg_df.iloc[0]



unique_by_vid_gen = gen_stsg_df.groupby(['video_id', 'start', 'end']).nth(0)
unique_by_vid_gen = unique_by_vid_gen.rename(columns={'stsg': 'gen-stsg'})
unique_by_vid_gen = unique_by_vid_gen.drop(columns=['chat_history'])



unique_by_vid_gt = gt_stsg_df.groupby(['video_id', 'start', 'end']).agg({
    'stsg': 'first',
    'question_id': list       # Collect all 'question_id's into a list
}).rename(columns={'question_id': 'q_ids', 'stsg': 'gt-stsg'}).reset_index()


r_df = unique_by_vid_gen.merge(
    unique_by_vid_gt,
    on=['video_id', 'start', 'end'],
    how='left'
)
# add qid for prompt compatibiliy 
r_df['qid'] = r_df.apply(lambda x: x['q_ids'][0], axis=1)
r_df['prompt'] = r_df.apply(lambda x: pformatter.format(x), axis=1)



prompts = r_df.to_dict('records')



model_options = main._load_model_options(WORK_DIR / "ollama_model_options.json")

sys_prompt = None
ollama_params = {
    "model": "gemma3:27b",
    "system": sys_prompt,
    "stream": True,
    "options": model_options
}

url = os.environ.get("OLLAMA_URL", "http://lusha_ollama:11435")
client = ollama_manager.OllamaRequestManager(url, ollama_params)

# to filter prompts
#prompts = prompts[]

OUT_FILE = WORK_DIR / 'outputs/sim_between_frames_v2_gemma3:27b-it-qat_20250606_13:50:00.jsonl'
batch_processor.batch_generate(
    client,
    prompts,
    output_file_path=OUT_FILE
)
