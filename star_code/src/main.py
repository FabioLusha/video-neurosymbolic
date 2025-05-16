import argparse
import base64
import functools
import json
import os

import re
from pathlib import Path

import batch_processor
import prompt_formatters as pf
from ollama_manager import OllamaRequestManager, PromptDataset
from STAR_utils.visualization_tools import vis_utils

SEED = 13471225022025

MODELS = [
    "llama3.2",
    "llama3.1:8b",
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "phi3:3.8b",
    "gemma3:4b",
    "gemma3:4b-it-q4_K_M", #instruction tuned
    "gemma3:4b-it-qat", #it quantization aware training
    "gemma3:12b",
    "gemma3:27b",
]

BASE_DIR = Path(__file__).parent.parent 

def load_open_qa_prompts():
    system_prompt = _load_prompt_fromfile(BASE_DIR / "data/system_prompt.txt")
    prompt_format = "QUESTION: {question}\n" "SPATIO-TEMPORAL SCENE-GRAPH: {stsg}"

    pformatter = pf.OpenEndedPrompt(prompt_format)
    return system_prompt, pformatter


def load_mcq_prompts():
    # PROMPT FOR MULTI-CHOICE QA
    mcq_system_prompt = _load_prompt_fromfile(
        BASE_DIR / "data/prompts/MCQ_system_prompt_v2_oneshot.txt"
    )
    mcq_pformat = """\
        Q: {question}
        Alternatives:
        A. {c1}
        B. {c2}
        C. {c3}
        D. {c4}

        STSG: {stsg}
        A:
        """
    mcq_pformatter = pf.MCQPrompt(mcq_pformat)

    return mcq_system_prompt, mcq_pformatter


def load_mcq_html_prompts():
    # PROMPT FOR MULTI-CHOICE QA WITH HTML TAGS
    mcq_system_prompt = _load_prompt_fromfile(BASE_DIR / "data/MCQ_system_prompt_v3.txt")
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
        BASE_DIR / "data/prompts/zero-shot-cot/MCQ_system_prompt_ZS_CoT.txt"
    )
    user_prompt = _load_prompt_fromfile(
        # "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT.txt"
        BASE_DIR / "data/prompts/zero-shot-cot/MCQ_user_prompt_ZS_CoT_v2.txt"
    )

    mcq_pformatter = pf.MCQPrompt(user_prompt)

    return sys_prompt, mcq_pformatter


def load_bias_check_prompts():
    # PROMPTS FOR BIAS CHECK - I.E. QUESTION WITHOUT STSG
    mcq_system_prompt_bias = _load_prompt_fromfile(
        BASE_DIR / "data/prompts/system_prompt_bias_check.txt"
    )
    mcq_bias_pformat = "Q: {question}\n" "{c1}\n{c2}\n{c3}\n{c4}\n" "A:"
    mcq_bias_pfromatter = pf.MCQPromptWoutSTSG(mcq_bias_pformat)

    return mcq_system_prompt_bias, mcq_bias_pfromatter


def load_llm_as_judge_prompts(responses_filepath):
    llm_judge_sys_prompt = _load_prompt_fromfile(BASE_DIR / "data/prompts/LLM_judge_system_v2.txt")
    llm_judge_usr_prompt = _load_prompt_fromfile(BASE_DIR / "data/prompts/LLM_judge_user_v2.txt")

    judge_pformatter = pf.LlmAsJudgePrompt(llm_judge_usr_prompt, responses_filepath)
    return llm_judge_sys_prompt, judge_pformatter

def load_vqa_prompts():
    user_prompt = _load_prompt_fromfile(BASE_DIR / "data/prompts/img_answer/user_prompt.txt")
    mcq_bias_pfromatter = pf.MCQPromptWoutSTSG(user_prompt)

    return None, mcq_bias_pfromatter

def _load_prompt_fromfile(filename):
    try:
        with open(filename) as in_file:
            return in_file.read().strip()

    except IOError as e:
        raise IOError(f"Error reading system prompt file: {e}")


def run_with_prompts(
    args,
    system_prompt,
    prompt_formatter,
    model_name,
    mode="generate",
    input_filepath=None,
    ids_filepath=None,
    output_filepath=None,
):
    # Initialize Ollama manager
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_client = OllamaRequestManager(
        base_url=OLLAMA_URL,
        ollama_params={
            "model": model_name,
            "system": system_prompt,
            "stream": True,
            "options": {
                "num_ctx": 10240,
                "temperature": 0.1,
                "num_predict": 2048,
                "seed": SEED,
            },
        },
    )

    # Initialize the prompt generator with the appropriate file
    if not input_filepath:
        input_filepath = BASE_DIR / "data/datasets/STAR_QA_and_stsg_val.json"  # Default to MCQ

    # Handle ids file
    ids = None
    if ids_filepath:
        print(f"=== Loading file with ids: {ids_filepath}")
        with open(ids_filepath, "r") as f:
            ids = [line.strip() for line in f.readlines()]
    else:
        print(f"=== No ids file chosen")

    print(f"=== Generating prompts from: {input_filepath}")
    dataset = PromptDataset(input_filepath, prompt_formatter, ids=ids)
    prompts = [dataset[i] for i in range(len(dataset))]

    # Generate responses
    ollama_client.load_model()
    
    if mode == "generate":
        print("=== Mode: generate")
        batch_processor.batch_generate(
            ollama_client, prompts, output_file_path=output_filepath
        )
    elif mode == "chat":
        print("=== Mode: chat")
        reply = """\
        Therefore the final answer is?
        
        Your response must be provided in valid JSON format as follows:
        {"answer": "your complete answer here"}
        
        IMPORTANT: Always include both the letter (A, B, C, D, etc.) AND the full text of the answer in your response.
        Do not abbreviate or shorten the answer. For example, if the correct answer is "A. the laptop", your response 
        should be {"answer": "A. the laptop"}, not {"answer": "laptop"} or {"answer": "A"}.\
        """
        if args.task == 'vqa':
            stream_vqa(
                ollama_client, prompts, ids, reply, output_filepath=output_filepath
            )
        else:    
            batch_processor.batch_automatic_chat_reply(
                ollama_client, prompts, reply, output_file_path=output_filepath
            )


def main():
    # Define available prompt types
    prompt_types = {
        "open_qa": load_open_qa_prompts,
        "mcq": load_mcq_prompts,
        "mcq_html": load_mcq_html_prompts,
        "mcq_zs_cot": load_mcq_zs_cot_prompts,
        "bias_check": load_bias_check_prompts,
        "judge": load_llm_as_judge_prompts,
        "vqa": load_vqa_prompts
    }

    task_types = {
        "graph-gen": 0,
        "vqa": 0,
        "graph-understanding": 0
    }
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run LLM with different prompt types")
    parser.add_argument(
        "--task",
        choices=task_types.keys(),
        default="graph-understanding",
        help="Choose the task to be performed",
    )
    
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
    parser.add_argument("--output-file", help="file path where to save the response)")
    parser.add_argument(
        "--ids-file",
        help="File were to extract ids for which to run the model",
    )
    parser.add_argument(
        "--responses-file", help="File with the responses to be evaluated by the judge"
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "chat"],
        help="How to run the model, 'chat' or 'generate' mode",
    )

    args = parser.parse_args()

    # Set default input file based on prompt type
    input_file = args.input_file
    if not input_file:
        if args.prompt_type in ["open_qa"]:
            input_file = BASE_DIR / "data/datasets/STAR_question_and_stsg.json"
        else:
            input_file = BASE_DIR / "data/datasets/STAR_QA_and_stsg_val.json"

    # Special handling for judge prompt type
    responses_file = None
    if args.prompt_type == "judge":
        responses_file = args.responses_file
        prompt_types["judge"] = functools.partial(prompt_types["judge"], responses_file)

    # Handle ids file
    ids_file_path = args.ids_file
    ids = None
    if ids_file_path:
        with open(ids_file_path, "r") as f:
            ids = [line.strip() for line in f.readlines()]
            
    if args.task == 'graph-gen':
        from pathlib import Path
        
        url = os.environ.get("OLLAMA_URL", "http://lusha_ollama:11435")

        sys_file_path = BASE_DIR / 'data/prompts/img_captioning/system_prompt.txt'
        
        sys_prompt = _load_prompt_fromfile(sys_file_path)
        ollama_params={
                    "model": args.model,
                    "system": sys_prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": 10240,
                        "temperature": 0.1,
                        "num_predict": 1024,
                        "seed": SEED,
                    },
                }

        client = OllamaRequestManager(url, ollama_params)
        streaming_frame_generation(client, args.output_file, ids=ids, iters=-1)
        return

    # Load the selected prompts
    load_func = prompt_types[args.prompt_type]
    system_prompt, prompt_formatter = load_func()

    
    # Run with the selected configuration
    run_with_prompts(
        args=args,
        system_prompt=system_prompt,
        prompt_formatter=prompt_formatter,
        model_name=args.model,
        mode=args.mode,
        input_filepath=input_file,
        output_filepath=args.output_file,
        ids_filepath=ids_file_path,
    )


def generate_frames(iters=-1, max_sample=10, ids=None):

    star_data = []
    val_path = BASE_DIR / "data/datasets/STAR/STAR_annotations/STAR_val.json"
    with open(val_path) as in_file:
        star_data = json.load(in_file)

    raw_frame_dir = BASE_DIR / "data/datasets/action-genome/frames"
    
    if ids:
        id_to_dict = {d['question_id']: d for d in star_data}
        star_data = [id_to_dict[id] for id in ids]
        
    for sample in star_data:       
        frame_ids = sorted(list(sample['situations'].keys()))
        frame_ids = vis_utils.sample_frames(frame_ids, max_sample)

        frame_dir = raw_frame_dir / f"{sample['video_id']}.mp4"
        b64_encodings = []
        for f_id in frame_ids:
            img_path = frame_dir / f"{f_id}.png"
            if not img_path.exists():
                continue
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                b64_encodings.append(
                    {
                        "frame_id": f_id,
                        "encoding": base64.b64encode(img_bytes).decode("utf-8"),
                    }
                )

        frames = []
        for encoding in b64_encodings:
            frames.append({"question_id": sample["question_id"], **encoding})
            
        yield frames

        iters -= 1
        if iters == 0:
            return # stop generation

def extract_frame_description(text):
    # the ?s: in the middle capturing group sets the flag
    # re.DOTALL
    pattern = "(?<=<scene_graph>)(?s:.+)(?=</scene_graph)"
    match = re.search(pattern, text)

    return match.group(0) if match else ""

def frame_aggregator(stream):

    o1 = next(stream)
    agg = []
    while o1 is not None:
        frame_id = o1.pop("frame_id")
        sg = o1.pop("sg")
        agg.append(f"\nFrame {frame_id}:\n{sg}")
        try:
            o2 = next(stream)
            if o2["qid"] != o1["qid"]:
                yield {**o1, "stsg": ''.join(agg)}
                agg = []

            o1 = o2
        except StopIteration | TypeError:
            yield {**o1, "stsg": ''.join(agg)}
            return # Generator stops here

def streaming_frame_generation(ollama_client, output_file_path, ids=None, iters=-1):
    
    prompt1 = """\
    Look carefully at this image and identify all objects and relationships present.

    First, list all distinct objects you can detect in the image. Be thorough and specific with your object labels (e.g., "young woman" rather than just "person", "wooden chair" rather than just "chair").

    Then, describe the key relationships between these objects in free-form text. Consider:
    - Spatial relationships (above, below, behind, inside, etc.)
    - Action-based relationships (holding, looking at, sitting on, etc.)
    - Physical connections (attached to, part of, touching, etc.)
    - Relative positions (next to, between, surrounding, etc.)

    Think step by step.\
    """

    prompt2 = """\
    Now organize the objects and relationships you identified into a formal scene graph using this format:
    object1 ---- relationship ---- object2
    
    The list of relationship predicates should be introduced by the tag <scene_graph> and terminated by the tag </scene_graph>
    For example:
    woman ---- sitting_on ---- chair
    dog ---- lying_under ---- table
    book ---- on_top_of ---- shelf

    Please follow these guidelines:
    1. Create at least 10 relationship triplets (more if the image is complex)
    2. Use specific and consistent object labels
    3. Use concise but descriptive relationship terms (connect words with underscores)
    4. Include all meaningful relationships between objects
    5. Verify that all objects you identified in step 1 appear in at least one relationship

    Your scene graph:\
    """

    def payload_gen(situations):
        for frames in situations:
            for frame in frames:
                req_obj = {
                    "qid": frame["question_id"],
                    "frame_id": frame["frame_id"],
                    "payload": {
                        **ollama_client.ollama_params,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt1,
                                "images": [frame["encoding"]],
                            }
                        ],
                    },
                }

                yield req_obj


    bp = batch_processor
    graph_gen_pipeline = bp.Pipeline(
        payload_gen,
        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, endpoint='chat'),
        lambda stream: bp.auto_reply_gen(stream, prompt2),
        # check the response is ok before passing to frame_extraction,
        lambda stream: (o for o in stream if o['status'] == 'ok'),
        lambda stream: ({**stream_obj, 'sg': extract_frame_description(stream_obj['response']['content'])} for stream_obj in stream),
        lambda stream: frame_aggregator(stream),
        lambda stream: bp.stream_save(
            stream, bp.GeneratedGraphFormatter(), output_file_path
        ),

    )

    situations = (situation_frames for situation_frames in generate_frames(ids=ids, iters=iters, max_sample=10))
    graph_gen_pipeline.consume(situations)
    return

def stream_vqa(ollama_client, prompts, ids, reply, output_filepath, iters=-1):
    
    def payload_gen(situations):
        prompts_dict = {p['qid']: p['prompt'] for p in prompts}
        for situation in situations:
            frame_encodings = [frame['encoding'] for frame in situation]
                
            req_obj = {
                "qid": situation[0]["question_id"], # situation is list of [{question_id, frame_id, encoding}]
                "payload": {
                    **ollama_client.ollama_params,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompts_dict[situation[0]["question_id"]], 
                            "images": frame_encodings,
                        }
                    ],
                },
            }
            
            yield req_obj
    
    bp = batch_processor
    pipe = bp.Pipeline(
        # the first generator converts the prompt to the right format
        payload_gen,
        lambda payload_gen: bp.stream_request(payload_gen, ollama_client, "chat"),
        lambda stream: (o for o in stream if o['status'] == 'ok'),
        lambda resp_gen: bp.auto_reply_gen(resp_gen, reply),
        lambda resp_gen: bp.stream_save(
            resp_gen, bp.ChatResponseFormatter(), output_filepath
        ),
    )
    
    situations = (situation_frames for situation_frames in generate_frames(ids=ids, iters=iters, max_sample=10))
    pipe.consume(situations)
    return
                    
if __name__ == "__main__":
    main()


