import sys
from pathlib import Path
import time
from tqdm import tqdm

# Can't use __file__ or __filename__ inside a Jupyter notebook
WORK_DIR = Path(__file__).parent.parent

sys.path.append(str(WORK_DIR))

from src import (
    video_tools,
    prompt_formatters,
)
from src.datasets import STARDataset 

from google import genai
from google.genai import types as gtypes


class RateLimiter:
    def __init__(self, rpm):
        """ rpm: requests per minute """
        self.rpm = rpm
        self.request_times = []

    def record_request(self):
        self.request_times.append(time.time())

    def wait_if_needed(self):
        now = time.time()
        # Remove request older than 1minute
        self.request_times = [t for t in self.request_times if now - t < 60.5]

        if len(self.request_times) >= self.rpm:
            sleep_time = 60 - (now - self.request_times[0]) + 1 # +1 for margin
            print(f"Rate limit rached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)


class GeminiClientWrapper:
    def __init__(self, client=None, api_key=None):
        self.client = client or genai.Client(api_key=api_key)

    def send_request(self, model, contents):
        response = self.client.models.generate_content(
                model=model,
                contents=contents
        )

        return response

def vqa(
    client_wrapper,
    input_dataset,
    output_file,
    video_dir,
    fps,
    max_frames,
    user_prompt: str,
    follow_up,
    system_prompt=None,
    rpm=10,
):

    pformatter = prompt_formatters.MCQPromptWoutSTSG(user_prompt)

    star_dataset = STARDataset(
            input_dataset,
            pformatter,
    )


    rate_limiter = RateLimiter(rpm=rpm)
    for i in tqdm(range(len(star_dataset))):
        sample = star_dataset[i]
        rate_limiter.wait_if_needed()

        _, frame_paths = video_tools.extract_frames(
                video_path = f"{video_dir}/{sample['video_id']}.mp4",
                start_time = float(sample['start']),
                end_time   = float(sample['end']),
                fps        = fps,
                max_frames = max_frames,
        )


        contents = [sample['prompt'], 
        
    return

async def send_request(client, content):
    
