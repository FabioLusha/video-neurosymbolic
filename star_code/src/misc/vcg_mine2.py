import sys
import json
import os
from pathlib import Path


STAR_CODE_PATH = str(Path.cwd().parent.parent)
print(STAR_CODE_PATH)
sys.path.append(STAR_CODE_PATH)

from src import (
    main,
    datasets,
    video_tools,
    ollama_manager,
    prompt_formatters as pf,
    _const,
)

from os.path import exists
import random
import torch
import numpy as np
import gc,csv
import warnings
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm
from decord import VideoReader, cpu


sys.path.append(str(Path.cwd()))

#sys.path.append('/home/jupyter/democ_egodatasets/')
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.eval.model_utils import initialize_model
from huggingface_hub import hf_hub_download
from transformers.utils import logging
logging.set_verbosity_error()





# ======= CONFIGURAZIONE RIPRODUCIBILITÀ E MEMORIA =======

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.simplefilter(action='ignore', category=Warning)
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"

def cleanup_memory():
    """Pulisce memoria GPU e CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def check_gpu_memory(threshold_mb=2000):
    """Ritorna True se memoria libera GPU > threshold_mb."""
    if not torch.cuda.is_available():
        return False
    mem_reserved = torch.cuda.memory_reserved() / 1e6
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6
    mem_free = total_mem - mem_reserved
    return mem_free > threshold_mb



# ======= FUNZIONI UTILI =======

#def load_frames_from_folder(folder_path):
 #   frames = []
  #  for i in range(100):
   #     frame_path = os.path.join(folder_path, f"{i:03d}.png")
    #    if not os.path.exists(frame_path):
     #       print(f"Frame mancante: {frame_path}")
      #      continue
       # img = Image.open(frame_path).convert("RGB")
        #frames.append(img)
    #return frames

def load_frames_from_folder(folder_path, num_frm=100, target_size=(224, 224)):
    """
    Load video frames from a folder of pre-extracted images.

    Parameters:
    folder_path (str): Path to the folder containing 100 images named '000.png' to '099.png'.
    num_frm (int): Number of frames to load. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Predefine target image size
    target_h, target_w = 224, 224

    # Load and sort image filenames
    # frame_files = [f"{i:03d}.png" for i in range(num_frm)]
    frame_files = sorted(list(Path(folder_path).glob("frame_*.png")))
    img_array = []

    for fname in frame_files:
        img_path = os.path.join(folder_path, fname)
        img = Image.open(img_path).convert("RGB")
        img_array.append(np.array(img))

    # Convert list to numpy array
    img_array = np.stack(img_array, axis=0)  # Shape: (num_frm, H, W, C)

    # Resize if needed
    if img_array.shape[1] != target_h or img_array.shape[2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()  # (N, H, W, C)

    # Convert to list of PIL Images
    num_frm = min(len(frame_files), num_frm)
    clip_imgs = [Image.fromarray(img_array[j]) for j in range(num_frm)]

    return clip_imgs
def get_spatio_temporal_features_torch(features):
    t, s, c = features.shape
    temporal_tokens = torch.mean(features, dim=1)
    padding_size = 100 - t
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
    spatial_tokens = torch.mean(features, dim=0)
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()
    return concat_tokens

def video_chatgpt_infer(
    frames,
    question,
    model,
    vision_tower,
    tokenizer,
    image_processor,
    video_token_len,
    conv_mode="video-chatgpt_v1"
):
    DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
    DEFAULT_VID_START_TOKEN = "<vid_start>"
    DEFAULT_VID_END_TOKEN = "<vid_end>"

    if model.get_model().vision_config.use_vid_start_end:
        qs = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN + '\n' + question
    else:
        qs =  DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    image_tensor = image_processor.preprocess(frames, return_tensors='pt')['pixel_values'].half().cuda()

    with torch.no_grad():
        vision_out = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = vision_out.hidden_states[-2][:, 1:]

    video_features = get_spatio_temporal_features_torch(frame_features)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_features.unsqueeze(0),
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_new_tokens=2048,
            eos_token_id=tokenizer.eos_token_id,
            #stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return outputs.strip().rstrip(stop_str).strip()

if __name__ == "__main__":
    # Pulizia iniziale
    cleanup_memory()
    WORK_DIR = _const.BASE_DIR
    STAR_SMALL = WORK_DIR / "data/datasets/STAR/STAR_annotations/STAR_val_small_1000.json"
    RAW_VIDEO_DIR = WORK_DIR / 'data/datasets/action-genome/Charades_v1_480/'
    
    
    system_prompt = None
    
    # Prompt with only questions
    user_prompt = main._load_prompt_fromfile(WORK_DIR / "data/prompts/vqa/user_prompt_v2.txt")
    user_pformatter = pf.MCQPromptWoutSTSG(user_prompt)
    output_filepath = WORK_DIR / "outputs/vqa_videochatgpt2_complex_2stage_prompting.jsonl"

    star_dataset = datasets.STARDataset(
        STAR_SMALL,
        user_pformatter,
    )

    follow_up = main._load_prompt_fromfile(WORK_DIR / "data/prompts/zero-shot-cot/auto_reply_ZS_CoT.txt")
    fps = 1

    # ======= CARICAMENTO MODELLO =======

    model_name = "mmaaz60/LLaVA-7B-Lightening-v1-1"
    projection_path = hf_hub_download(
        repo_id="MBZUAI/Video-ChatGPT-7B",
        filename="video_chatgpt-7B.bin"
    )

    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
        model_name, projection_path
    )

    # ======= CARICAMENTO JSON E INFERENZA =======
    # path = "/home/jupyter/democ_egodatasets/"
    # json_path = os.path.join(path, "part_test2.json")
    # frame_folder = os.path.join(path, "Frame_extraction_100")   
    # output_csv = path +"vcg_100_uniform3.csv"

    cleanup_memory()



    samples = [star_dataset[i] for i in range(len(star_dataset))]
    for idx, sample in enumerate(tqdm(samples, desc="Esecuzione inference")):
        sample_id = sample["question_id"]
        ground_truth = sample["answer"]

        prompt = sample["prompt"] #+ "\n" + follow_up

        video_id = sample["video_id"]
        start_time = sample["start"]
        end_time = sample["end"]
        frame_folder, frames_path = video_tools.extract_frames(
            video_path=str(RAW_VIDEO_DIR / f"{video_id}.mp4"),
            fps=fps,
            start_time=start_time,
            end_time=end_time,
        )
        
        print(prompt)
        chat_history = [{"role": "user", "content": prompt}]
        if not os.path.exists(frame_folder):
            print(f"⚠️ Video non trovato: {frame_folder}")
            generated_answer = "[VIDEO NON TROVATO]"
        else:
            try:
                frames = load_frames_from_folder(frame_folder)
                generated_answer = video_chatgpt_infer(frames, prompt, model, vision_tower, tokenizer, image_processor, video_token_len)
                if generated_answer:
                    prompt = (
                        '<start_header_id|>user<|end_header_id|>' +
                        prompt +
                        '<|eot_id|>' +
                        '<start_header_id|>assistant<|end_header_id|>' +
                        generated_answer +
                        '<|eot_id|>' +
                        '<start_header_id|>user<|end_header_id|>' +
                        follow_up
                    )

                    print(prompt)
                    chat_history.append({"role": "assitant", "content": generated_answer})
                    chat_history.append({"role": "user", "content": follow_up})

                    generated_answer = video_chatgpt_infer(frames, prompt, model, vision_tower, tokenizer, image_processor, video_token_len)
                    print(generated_answer)

                    chat_history.append({"role": "assitant", "content": generated_answer})
            except Exception as e:
                print(f"❌ Errore su {sample_id}: {e}")
                generated_answer = "[ERRORE]"

        print(generated_answer)
        row_data = {
            "qid": sample_id,
            "question_id": sample_id,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "chat_history": chat_history,
        }

        # Scrive ad ogni iterazione
        with open(output_filepath, "a", encoding="utf-8") as f:
            line = json.dumps(row_data) + "\n"
            f.write(line)
            f.flush()

        # Cleanup memoria anche dopo scrittura
        cleanup_memory()
    print(f"\n✅ Risultati salvati in: {output_filepath}")
