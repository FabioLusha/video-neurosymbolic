import json
import logging
import os
from collections import OrderedDict, namedtuple

import requests
from torch.utils.data import Dataset

import prompt_formatters as pf

Result = namedtuple("Result", ["status", "client", "id", "payload", "response"])


class OllamaRequestManager:

    def __init__(self, base_url, ollama_params, verbosity=None):
        self.base_url = base_url
        self.model = ollama_params["model"]

        # Setting some of the params to pass to ollama
        self.ollama_params = ollama_params

        self.default_handlers = dict()
        self.default_handlers["generate"] = OllamaGenerateHandler()
        self.default_handlers["chat"] = OllamaChatHandler()

        # TODO: implement logging functionality
        self.verbosity = verbosity or int(os.getenv("DEBUG_LEVEL", 0))
        logger = self._setup_logger()

    def _setup_logger(self):
        # TODO:
        # map verbosity to logging levels?

        class ScaffoldLogger:
            def log(self, message):
                print(message)
                logger = logging.getLogger(self.__class__.__name__)

            def info(self, msg):
                self.log(msg)

            def debug(self, msg):
                self.log(msg)

            def warning(self, msg):
                self.log(msg)

        return ScaffoldLogger()

    def load_model(self):
        print(f" Loading model {self.ollama_params['model']} ".center(80, "="))
        # An empty prompt will cause ollama to load the specified model
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.ollama_params["model"]},
            )
        except requests.ConnectionError as e:
            raise requests.ConnectionError("Error while connecting to ollama") from e
        except requests.RequestException as e:
            raise requests.RequestExcetpion("Error while connecting to ollama") from e

    def unload_model(self):
        requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.ollama_params["model"], "keep_alive": 0},
        )

    def generate_completion(self, prompt, req_timeout=60):
        return self.ollama_completion_request(
            endpoint="generate",
            payload={**self.ollama_params, "prompt": prompt},
            req_timeout=req_timeout,
        )

    def chat_completion(self, messages, req_timeout=60):
        return self.ollama_completion_request(
            endpoint="chat",
            payload={**self.ollama_params, "messages": messages},
            req_timeout=req_timeout,
        )

    def ollama_completion_request(
        self, payload, endpoint="generate", handler=None, req_timeout=60
    ):

        if handler is None:
            handler = self.default_handlers[endpoint]

        result = handler.init_result()
        try:
            server_response = requests.post(
                f"{self.base_url}/api/{endpoint}",
                json=payload,
                timeout=req_timeout,
                stream=True,
            )

            # Raise an exception for HTTP errors
            server_response.raise_for_status()

            for chunk in server_response.iter_lines():
                # Filter out keep-alive chunks
                if chunk:
                    data = json.loads(chunk)

                    if data.get("done", ""):
                        # the last message in the stream does not contain
                        # any tokens in response, it contains metadata about
                        # the generated response
                        elapsed = data.get("eval_duration", "")
                        ntokens = data.get("eval_count", "")

                        print(f"\n\nResponse at: {ntokens/elapsed * 10**9:.1f} tk/s")
                        break

                    result = handler.process_chunk(data, result)
                    print(result["token_stream"][-1], end="", flush=True)

        except requests.RequestException as e:
            e.response = handler.finalize_result(result)
            raise

        return handler.finalize_result(result)


# Handler classes for different API endpoints
class OllamaStreamHandler:
    def init_result(self):
        return {"token_stream": []}

    def process_chunk(self, data, result):
        pass  # Implemented by subclasses

    def finalize_result(self, result):
        return "".join(result["token_stream"])


class OllamaGenerateHandler(OllamaStreamHandler):
    def process_chunk(self, data, result):
        token = data.get("response", "")
        result["token_stream"].append(token)
        return result


class OllamaChatHandler(OllamaStreamHandler):
    def init_result(self):
        return {"token_stream": [], "role": None}

    def process_chunk(self, data, result):
        message = data.get("message", {})
        token = message.get("content", "")

        # Set role if not already set
        if result["role"] is None:
            result["role"] = message.get("role", "assistant")

        result["token_stream"].append(token)
        return result

    def finalize_result(self, result):
        return {"content": "".join(result["token_stream"]), "role": result["role"]}


class PromptDataset(Dataset):
    def __init__(
        self,
        qa_file_path,
        prompt_formatter,
        stsg_file_path=None,
        ids=None,
        limit=None,
    ):
        if not os.path.exists(qa_file_path):
            raise OSError(f"No such file or directory: '{qa_file_path}'")
        self.qa_file_path = qa_file_path
        self.stsg_file_path = stsg_file_path
        self.prompt_formatter = prompt_formatter

        # Load QA data
        self.qa = self._load_qa_file()
        
        # Get question ID key (auto-detect between 'qid' and 'question_id')
        if len(self.qa) > 0:
            self.q_id_key = "qid" if "qid" in self.qa[0] else "question_id"
        # Set in _load_stsg_data
        self.stsg_id_key = None
        # Filter by IDs if provided
        if ids:
            self.qa = [q for q in self.qa if q[self.q_id_key] in ids]
        
        # Apply limit if provided
        if limit:
            self.qa = self.qa[:limit]
        
        # Load STSG data (video_id -> stsg mapping)
        self.stsgs = {}
        if self.stsg_file_path:
            self._load_stsg_data()
        
        self.preprocess()
        self.print_stats()

    def print_stats(self):
        """Print statistics about the dataset."""
        print("\nDataset Statistics:")
        print("=" * 40)

        # QA stats
        print(f"QA File: {os.path.basename(self.qa_file_path)}")
        print(f"Number of QA samples: {len(self.qa)}")

        if len(self.qa) > 0:
            # Print example keys in QA data
            sample_keys = list(self.qa[0].keys())
            print(f"QA sample keys: {', '.join(sample_keys)}")

        # STSG stats
        if self.stsg_file_path:
            print(f"\nSTSG File: {os.path.basename(self.stsg_file_path)}")
            print(f"Number of unique video IDs with STSG: {len(self.stsgs)}")
            
            if len(self.stsgs) > 0:
                # Print example of first video_id and STSG keys if available
                first_vid = next(iter(self.stsgs))
                if isinstance(self.stsgs[first_vid], dict):
                    stsg_keys = list(self.stsgs[first_vid].keys())
                    print(f"STSG keys: {', '.join(stsg_keys)}")

        print("=" * 40 + "\n")

    def load_jsons(self, filepath):
        """Load a JSON or JSONL file."""
        ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in (".jsonl", ".ndjson"):
                return [json.loads(line) for line in f]
            else:
                raise IOError(f"{self.qa_file_path} must be either JSON or JSONL")

    def _load_qa_file(self):
        """Load QA data from JSON or JSONL file."""
        ext = os.path.splitext(self.qa_file_path)[1].lower()
        with open(self.qa_file_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in (".jsonl", ".ndjson"):
                return [json.loads(line) for line in f]
            else:
                raise IOError(f"{self.qa_file_path} must be either JSON or JSONL")

    def _load_stsg_data(self):
        """Load all STSG data into memory (video_id -> stsg dict)."""
        if not os.path.exists(self.stsg_file_path):
            raise OSError(f"STSG file not found: {self.stsg_file_path}")

        data = self.load_jsons(self.stsg_file_path)
        for item in data:
            try:
                if 'question_id' in item.keys():
                    self.stsg_id_key = 'question_id'
                elif 'video_id' in item.keys():
                    self.stsg_id_key = 'video_id'
                else:
                    raise ValueError("Expected 'question_id' or 'video_id' as id in the STSG file")
                
                id = item.get(self.stsg_id_key)
                stsg = item.get("stsg")
                # equivalent:
                # if video_id is not None and stsg is not None
                if id and stsg:
                    self.stsgs[id] = stsg
            except json.JSONDecodeError:
                continue

    def preprocess(self):
        pass

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample = self.qa[idx]
        question_id = sample.get(self.q_id_key)
        video_id = sample.get("video_id")

        # Add STSG to sample if available
        if video_id and video_id in self.stsgs:
            sample["stsg"] = self.stsgs[video_id]

        sample['qid'] = sample['question_id']
        sample['prompt'] = self.prompt_formatter.format(sample)

        return sample 

class STARDataset(PromptDataset):

    def _load_stsg_data(self):
        """Load all STSG data into memory (video_id -> stsg dict)."""
        if not os.path.exists(self.stsg_file_path):
            raise OSError(f"STSG file not found: {self.stsg_file_path}")
        
        data = self.load_jsons(self.stsg_file_path)
        for item in data:
            try:
                if 'question_id' in item.keys():
                    self.stsg_id_key = 'question_id'
                    
                    id = item.get('question_id')
                    stsg = item.get("stsg")
                    
                    self.stsgs[id] = stsg
                elif 'video_id' in item.keys():
                    self.stsg_id_key = 'video_id'
                    
                    video_id = item.get("video_id")
                    stsg = item.get("stsg")
                    if video_id is not None and stsg is not None:
                        # if the key is not present in the dict is initialized with the empty
                        # list and then append the new value.
                        # If the key exists setdefault returns the value (the list), to which we
                        # append the new element
                        self.stsgs.setdefault(video_id, []).append({
                            'stsg': stsg,
                            'start': item.get("start", None),
                            'end': item.get("end", None),
                        })
                else:
                    raise ValueError("Expected 'question_id' or 'video_id' as id in the STSG file")
                    
            except json.JSONDecodeError:
                continue
                
    def preprocess(self):
        """If a stsg file is specified, than filter out the qa without an stsg
        """
        filtered_qas = []
        if self.stsg_file_path:
            for sample in self.qa:
                if self.stsg_id_key == 'question_id':
                    if isinstance(sample['choices'][0], dict):
                        sample['stsg'] = self.stsgs[sample.get('question_id')]
                        choices = {str(choice['choice_id']): choice['choice'] for choice in sample['choices']}
                        sample['choices'] = choices
                elif self.stsg_id_key == 'video_id':
                    video_id = sample.get('video_id')
                    start = sample.get('start')
                    end = sample.get('end')

                    if video_id in self.stsgs:
                        # Look inside all the sub-clip of the video for the one referenced
                        # by the question (i.e. matching video_id, start, and end)
                        for situation in self.stsgs[video_id]:
                            if situation['start'] == start and situation['end'] == end:
                                sample['stsg'] = situation['stsg']

                                # check if the choices attribute is a string or the dict
                                # used in STAR
                                if isinstance(sample['choices'][0], dict):
                                    sample['choices'] = {str(choice['choice_id']): choice['choice'] for choice in sample['choices']}

                                filtered_qas.append(sample)
                    else:
                        #TODO: warn qa is not associated to a stsg
                        continue
        # self.qa updated with filtered_qas if filtered is not null
        # i.e. 'question_id' => filtered_qas == [] --> (A => B) == (not A) or B
        assert not (self.stsg_id_key == 'question_id') or (filtered_qas == [])
        self.qa = filtered_qas or self.qa
        return


    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample = self.qa[idx]

        sample['qid'] = sample[self.q_id_key] # question_id
        sample['prompt'] = self.prompt_formatter.format(sample)
        return sample

class CVRRDataset(PromptDataset):
    
    def preprocess(self):
        for item in self.qa:
            item['question_id'] = item.pop('unique_id')
            item['question'] = item.pop('Q')
            item['video_id'] = item.pop('video_path').split(".")[0]
            item['answer'] = item.pop('A')

class JudgeDataset(PromptDataset):

    def __init__(
        self,
        prompt_dataset,
        predictions_filepath,
        prompt_formatter
    ):
        self.base_dataset = prompt_dataset
        self.predictions = {}
        data = self.load_jsons(predictions_filepath)
        for pred in data:
            key = pred.get('qid', None)
            if key is None:
                key = pred.get('question_id')

            self.predictions[key] = pred

        # Ovveride the original prompt_formatter with that of the LLM as a Judge
        prompt_dataset.prompt_formatter = prompt_formatter
        
