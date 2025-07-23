from importlib import reload

import os, sys
from pathlib import Path
import json
import time
from tqdm import tqdm

import pandas as pd 

# Can't use __file__ or __filename__ inside a Jupyter notebook
WORK_DIR = Path.cwd().parent

sys.path.append(str(WORK_DIR))

from src import (
    graph_gen,
    ollama_manager,
    video_tools,
    prompt_formatters,
    datasets,
)



from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

GEMINI_API_KEY = "AIzaSyBl01O8N7j2jiAyAqmfdRKrBB4yE7WI4Jk"
VIDEO_DIR = WORK_DIR / "data/datasets/action-genome/Charades_v1_480"

# low safety settings
safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
    safety_settings=safety_settings,
    
)
client = genai.Client(
    api_key=GEMINI_API_KEY,
)
star_vsmall_df = pd.read_json(WORK_DIR / "data/datasets/STAR/STAR_annotations/STAR_val_small_200.json")
star_samples = star_vsmall_df.to_dict(orient='records')
# GEMINI requests

user_prompt = graph_gen._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/user_prompt_v2.txt")
pformatter = prompt_formatters.ImgPromptDecorator(
    prompt_formatters.PromptFormatter(user_prompt),
    img_field="images",
    tag="[img]"
)

follow_up = graph_gen._load_prompt_fromfile(WORK_DIR / "data/prompts/graph-gen/format_instructions_v2corrected.txt")

start = time.time()

for i in tqdm(range(95, len(star_samples))):
    sample = star_samples[i]
    # limit the request per minute
    # 2 request per samples
    if ((i+1)*2) % 10 == 0 and ((now := time.time()) - start) < 60:
        wait = start + 60 - now
        print(f"Waiting {wait:2.2f}s")
        time.sleep(wait)
        start = time.time()
    print(f"Request: {sample['question_id']}")
    
    _, frame_paths = video_tools.extract_frames(
        video_path= f"{VIDEO_DIR}/{sample['video_id']}.mp4",
        start_time=  float(sample['start']),
        end_time=   float(sample['end']),
        fps=        1.0,
        max_frames= 64,
    )
    
    if not frame_paths:
        print(f"Error in extracting frames for {sample['question_id']}")
        print("Skipping...")
        continue
    
    sample['images'] = [Image.open(frame_path) for frame_path in frame_paths]
    sample['prompt'] = pformatter.format(sample)
    
    text_prompt = sample['prompt']
    
    chat_history = [{
        "role": "user",
        "content": text_prompt
    }]
    
    text_prompt = text_prompt.split("[img]")
    images = sample['images']
    
    to_send = []
    for j, text in enumerate(text_prompt):
        to_send += [text, images[j]] if j < len(images) else [text]
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=to_send, # Pillow images can be directly passed as inputs (which will be converted by the SDK)
        config=config
    )

    if not response:
        print(f"Error in generating the 1st response for {sample['question_id']}")
        print("Skipping")
        continue
    
    chat_history.append({
        "role": "assistant",
        "content": response.text
    })
    chat_history.append({
        "role": "user",
        "content": follow_up
    })
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=to_send + [response.text, follow_up],
        config=config
    )
    
    if not response:
        print(f"Error in generating the 2nd response for {sample['question_id']}")
        print("Skipping")
        continue
    
    chat_history.append({
        "role": "assistant",
        "content": response.text
    })

        
    with open(WORK_DIR / "notebooks/gemini_responses_small200_20250721_18:26:00.jsonl", 'a') as f:
        out_format = {
            "question_id": sample["question_id"],
            "chat_history": chat_history,
            "stsg": chat_history[-1]['content']
        }
        
        line = json.dumps(out_format) + "\n"
        f.write(line)
        f.flush()
        

