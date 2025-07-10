# ===== Importazioni generali e setup ambiente =====
import os, sys, gc, json, csv, torch, warnings, random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

# Imposta il seed per garantire la riproducibilità
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Warnings off
warnings.simplefilter(action='ignore', category=Warning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Impostazioni per CUDA e memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"
torch.cuda.empty_cache()
gc.collect()

# ===== Setup modello VideoLLaMA3 =====
model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def cleanup_memory():
    """Pulisce la memoria GPU/CPU per evitare morte del kernel."""
    torch.cuda.empty_cache()
    gc.collect()


def check_gpu_memory(threshold_mb=2000):
    """Controlla se c'è memoria sufficiente, ritorna True/False."""
    mem_reserved = torch.cuda.memory_reserved() / 1e6
    mem_allocated = torch.cuda.memory_allocated() / 1e6
    mem_free = torch.cuda.get_device_properties(0).total_memory / 1e6 - mem_reserved
    return mem_free > threshold_mb


def generate_text(video_path, question, max_tokens=128):#1024 potrebbe dover essere modificato per questioni allineamento con VCG. Indica il max numero di nuovi token per generare l'output
    cleanup_memory()
    
    conversation = [
        {"role": "system", "content": "You are a helpfull assistant. Your answers must be short, direct, and without explanations. Do not say 'I don't know', 'I cannot answer', or mention lacking context. Avoid any justifications, apologies, or references to yourself."},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {
                        "video_path": str(video_path),
                         "fps": 1, "max_frames": 100
                    }
                },
                {"type": "text", "text": question},
            ]
        },
    ]

    try:
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
  

        # ======= INFERENCE =======
        if not check_gpu_memory():
            raise RuntimeError("❌ GPU memory troppo bassa, skip sample.")

        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

    except Exception as e:
        cleanup_memory()
        raise e


def run_inference(json_path, frame_folder, output_csv="videollama3_results.csv"):
    from os.path import exists

    with open(json_path, "r") as f:
        samples = json.load(f)

    file_exists = exists(output_csv)
    fieldnames = ["sample_id", "question", "ground_truth", "generated_answer"]
    for idx, sample in enumerate(tqdm(samples, desc="Esecuzione inference")):
        question = sample["question"]
        ground_truth = sample["answer"]
        sample_id = sample["sample_id"]
        frames_path = Path(frame_folder) / sample_id

        if not frames_path.exists():
            print(f"⚠️ Video non trovato: {frames_path}")
            generated_answer = "[VIDEO NON TROVATO]"
        else:
            try:
                generated_answer = generate_text(frames_path, question)
            except Exception as e:
                print(f"❌ Errore su {sample_id}: {e}")
                generated_answer = "[ERRORE]"

        row_data = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer
        }

        # Scrive ad ogni iterazione
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists and idx == 0:
                writer.writeheader()
                file_exists = True

            writer.writerow(row_data)

        # Cleanup memoria anche dopo scrittura
        cleanup_memory()

    print(f"\n✅ Risultati salvati in: {output_csv}")


json_path =  "part_test3.json"        
frame_folder = "output_directory"
run_inference(
    json_path,
    frame_folder,
    output_csv= "vl3_100_v2_4.csv"
)