import os
import sys
from pathlib import Path
import json
import time
from tqdm import tqdm
import pandas as pd

# Can't use __file__ or __filename__ inside a Jupyter notebook
WORK_DIR = Path(__file__).parent.parent

sys.path.append(str(WORK_DIR))

from src import (
    graph_gen,
    video_tools,
    prompt_formatters,
)
from src.datasets import STARDataset 


from google import genai
from google.genai import types
from PIL import Image

class RateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.request_times = []

    def record_request(self):
        self.request_times.append(time.time())

    def wait_if_needed(self):
        now = time.time()
        # Remove request older than 1minute
        self.request_times = [t for t in self.request_times if now - t < 60.5]

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]) + 1 # +1 for margin
            print(f"Rate limit rached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY2", None)
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

# GEMINI requests

user_prompt = graph_gen._load_prompt_fromfile(WORK_DIR / "data/prompts/vqa/user_prompt.txt")
pformatter = prompt_formatters.MCQPromptWoutSTSG(user_prompt)


star_samples = STARDataset(
    WORK_DIR / "data/datasets/STAR/STAR_annotations/STAR_val_small_200.json",
    pformatter,
)
follow_up = graph_gen._load_prompt_fromfile(WORK_DIR / "data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt")

start = time.time()

rate_limiter = RateLimiter(requests_per_minute=10)
for i in tqdm(range(95, len(star_samples))):
    sample = star_samples[i]

    rate_limiter.wait_if_needed()
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
    first_prompt = pformatter.format(sample)
    
    sample['prompt'] = prompt_formatters.ImgPromptDecorator(
        prompt_formatters.PromptFormatter(first_prompt + "\n{images}"),
        img_field="images",
        tag="[img]"
    ).format(sample)

    text_prompt = sample['prompt']
    
    if i == 0:
        print(text_prompt)

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

    rate_limiter.record_request()

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
    rate_limiter.record_request()

    if not response:
        print(f"Error in generating the 2nd response for {sample['question_id']}")
        print("Skipping")
        continue
    
    chat_history.append({
        "role": "assistant",
        "content": response.text
    })

        
    with open(WORK_DIR / "notebooks/gemini_vqa_small200_20250722_15:44:00.jsonl", 'a') as f:
        out_format = {
            "qid": sample["question_id"],
            "question_id": sample["question_id"],
            "chat_history": chat_history,
        }
        
        line = json.dumps(out_format) + "\n"
        f.write(line)
        f.flush()
        

