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
        
        with open(self.stsg_file_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    video_id = item.get("video_id")
                    stsg = item.get("stsg")
                    if video_id is not None and stsg is not None:
                        self.stsgs[video_id] = stsg
                except json.JSONDecodeError:
                    continue

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
        
        prompt = self.prompt_formatter.format(sample)
        return {
            "qid": question_id,
            "prompt": prompt,
        }
