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


class STARPromptGenerator:

    def __init__(self, questions_file_path, stsg_file_path=None):
        if not os.path.exists(questions_file_path):
            raise OSError(f"No such file or directory: '{questions_file_path}'")
        self.question_file_path = questions_file_path

        if stsg_file_path is None:
            stsg_file_path = questions_file_path
        else:
            if not os.path.exists(stsg_file_path):
                raise OSError(f"No such file or directory: '{stsg_file_path}'")

        self.stsg_file_path = stsg_file_path

    def load_stsg_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"STSG file not found: {file_path}")

        with open(file_path, "r") as f:
            try:
                stsg_items = [json.loads(line) for line in f]
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in STSG file: {e}")

        if not stsg_items:
            return {}

        # Determine the ID key dynamically
        first_item = stsg_items[0]
        id_key = "qid" if "qid" in first_item else "question_id"

        return {
            item.pop(id_key): item["stsg"]
            for item in stsg_items
            if id_key in item and "stsg" in item
        }

    def generate(self, prompt_formatter, ids=None, start=0, limit=None):
        """
        Args:
            prompt_formatter (PromptFormatter):
                An object that given a data point form the STAR dataset
                formats the prompt according to what is specified in the object

            start (int): from which sample to start generation

            limit (int): how many prompt to generate

            mcq (boolean): specifies if we need to use the MCQ prompt


        Returns:
            prompt
        """
        try:
            with open(self.question_file_path, "r") as q_file:
                question_data = json.load(q_file)

                stsg_data = None
                if self.stsg_file_path != self.question_file_path:
                    with open(self.stsg_file_path, "r") as stsg_file:
                        list_data = [json.loads(line) for line in stsg_file.readlines()]

                        s_id_key = "qid" if "qid" in stsg_data[0] else "question_id"
                        for item in list_data:
                            item_id = item.pop(s_id_key)
                            stsg_data[item_id] = item["stsg"]

                q_id_key = "qid" if "qid" in question_data[0] else "question_id"

                for i, sample in enumerate(question_data, 1):
                    if i < start:
                        continue
                    if limit and i > (limit + start):
                        break

                    if ids and sample[q_id_key] in ids:
                        continue

                    # Below I merge the question key-value pairs with
                    # the stsg (note the key-values pairs on the right take priority
                    # when there are keys in common, i.e. the stsg data on the right overwrites
                    # the stsg data on sample if there is any)
                    sample = sample | stsg
                    prompt = prompt_formatter.format(sample)

                    yield {"qid": sample[q_id_key], "prompt": prompt}

        except IOError as e:
            raise IOError(f"Error reading question and stsg file: {e}") from e

    def generate_and_save_prompts(
        self, output_file_path, prompt_formatter, start=0, limit=None
    ):
        try:
            # Open file with line buffering
            with open(output_file_path, "w", buffering=1, encoding="utf-8") as f:
                for prompt_data in self.generate(prompt_formatter, start, limit):
                    f.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
                    f.flush()

                print(f"Prompts saved to {output_file_path}")

                return True

        except IOError as e:
            raise IOError("Error saving prompts") from e


class PromptDataset(Dataset):
    def __init__(
        self,
        qa_file_path,
        prompt_formatter,
        stsg_file_path=None,
        ids=None,
        limit=None,
        stsg_buffer_size=1000,
    ):
        if not os.path.exists(qa_file_path):
            raise OSError(f"No such file or directory: '{qa_file_path}'")
        self.qa_file_path = qa_file_path

        if stsg_file_path and not os.path.exists(stsg_file_path):
            raise OSError(f"No such file or directory: '{stsg_file_path}'")

        self.prompt_formatter = prompt_formatter
        self.stsg_file_path = stsg_file_path

        self._stsg_index = None
        self._stsg_buffer = OrderedDict()
        self.stsg_buffer_size = 1000
        self._stsg_file_handle = None
        self.q_id_key = None

        # Load qa and build STSG index
        self.qa = []
        with open(self.qa_file_path, "r") as f:
            self.qa = json.load(f)

        if len(self.qa) > 0:
            self.q_id_key = "qid" if "qid" in self.qa[0] else "question_id"

        if ids:
            self.qa = [
                q
                for i, q in enumerate(self.qa)
                if q[self.q_id_key] in ids and (limit is None or i < limit)
            ]
        if self.stsg_file_path:
            self._build_stsg_index()

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError

        sample = self.qa[idx]
        question_id = sample.get(self.q_id_key)

        # Get STSG data if available
        stsg_data = {}
        if self.stsg_file_path and question_id:
            # TODO: revert to simple access, the buffer needs attention
            with open(self.stsg_file_path) as f:
                f.seek(self._stsg_index[question_id])
                line = f.readline()
                stsg_data = json.loads(line)

            # stsg_data = self._get_stsg_data(question_id)

        # Merge the question key-value pairs with
        # the stsg (note the key-values pairs on the right take priority
        # when there are keys in common, i.e. the stsg data on the right overwrites
        # the stsg data on sample if there is any)
        sample = {**sample, **stsg_data}
        prompt = self.prompt_formatter.format(sample)

        return {"qid": question_id, "prompt": prompt}

    def _build_stsg_index(self):
        self._stsg_index = {}
        id_key = None
        first_line = True
        try:
            with open(self.stsg_file_path, "rb") as f:
                while True:
                    byte_offset = f.tell()
                    line = f.readline()

                    if not line:
                        break

                    try:
                        item = json.loads(line)

                        if first_line:
                            if "qid" in item:
                                id_key = "qid"
                            elif "question_id" in item:
                                id_key = "question_id"
                            else:
                                raise ValueError(
                                    "STSG file lines must contain either 'qid' or 'question_id'"
                                )
                            first_line = False

                        if id_key not in item:
                            print(
                                f"Warning: Line at offset {byte_offset} missing ID key '{id_key}'. Skipping."
                            )
                            continue

                        self._stsg_index[item[id_key]] = byte_offset

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Unable to decode line at offset {byte_offset}. Skipping."
                        )
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(
                f"STSG file not found during indexing: {self.stsg_file_path}"
            )
        except IOError as e:
            raise IOError(f"Error reading STSG file during indexing: {e}") from e

    def _load_stsg_chunk(self, question_ids):
        """Load a chunk of STSG data for the given question IDs."""
        if not self._stsg_file_handle:
            self._stsg_file_handle = open(self.stsg_file_path, "r")

        # Clear buffer if it's too large
        if len(self._stsg_buffer) > self.stsg_buffer_size * 2:
            self._stsg_buffer.clear()

        # Find positions we need to load
        load_positions = {}
        for qid in question_ids:
            if qid in self._stsg_index and qid not in self._stsg_buffer:
                load_positions[qid] = self._stsg_index[qid]

        # Sort positions for sequential reads
        sorted_positions = sorted(load_positions.items(), key=lambda x: x[1])

        # Load in sorted order
        for qid, pos in sorted_positions:
            self._stsg_file_handle.seek(pos)
            line = self._stsg_file_handle.readline()
            try:
                data = json.loads(line)
                self._stsg_buffer[qid] = {"stsg": data.get("stsg", "No STSG data!")}
            except json.JSONDecodeError:
                continue

    def _get_stsg_data(self, question_id):
        """Get STSG data for a question with buffered loading."""
        if question_id in self._stsg_buffer:
            return self._stsg_buffer[question_id]

        # Pre-load a chunk around this question
        if question_id in self._stsg_index:
            idx = list(self._stsg_index.keys()).index(question_id)
            start = max(0, idx - self.stsg_buffer_size // 2)
            end = min(len(self._stsg_index), idx + self.stsg_buffer_size // 2)
            chunk_ids = list(self._stsg_index.keys())[start:end]
            self._load_stsg_chunk(chunk_ids)

        # TODO: handle default value
        if question_id not in self._stsg_buffer:
            print(
                f"Warning: {question_id} has no STSG data, filling the default value 'No STSG data'"
            )
        return self._stsg_buffer.get(question_id, "No STSG data")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_stsg_file_handle") and self._stsg_file_handle:
            self._stsg_file_handle.close()


class QuestionProcessor:

    def __init__(self, stsg_buffer_size=1000):
        self.stsg_buffer_size = stsg_buffer_size
        self._stsg_index = None
        self._stsg_buffer = OrderedDict()
        self._stsg_file_handle = None

    def _build_stsg_index(self, file_path):
        self._stsg_index = {}
        id_key = None
        first_line = True
        try:
            with open(file_path, "rb") as f:
                while True:
                    byte_offset = f.tell()
                    line = f.readline()

                    if not line:
                        break

                    try:
                        item = json.loads(line)

                        if first_line:
                            if "qid" in item:
                                id_key = "qid"
                            elif "question_id" in item:
                                id_key = "question_id"
                            else:
                                raise ValueError(
                                    "STSG file lines must contain either 'qid' or 'question_id'"
                                )
                            first_line = False

                        if id_key not in item:
                            print(
                                f"Warning: Line at offset {byte_offset} missing ID key '{id_key}'. Skipping."
                            )
                            continue

                        self._stsg_index[item[id_key]] = byte_offset

                    except json.JSONDecodeError:
                        print(
                            f"Warning: Unable to decode line at offset {byte_offset}. Skipping."
                        )
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"STSG file not found during indexing: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading STSG file during indexing: {e}") from e

    def _load_stsg_buffer(self, target_ids):
        """Load a buffer of STSG items of the given target IDa."""
        if not self._stsg_file_handle:
            self._stsg_file_hanlde = open(self.stsg_file_path, "r")

        # Clear buffer if it's getting to large
        if len(self._stsg_buffer) > self.stsg_buffer_size * 2:
            self._stsg_buffer.clear()

        # Find all positions we need to load
        load_index = {}
        for id in target_ids:
            if id in self._stsg_index and id not in self._stsg_buffer:
                load_index[id] = self._stsg_index[id]

        # Reorder based on the seek position in the file
        sorted_index = sorted(load_index.items(), key=lambda x: x[1])

        for id, pos in sorted_index:
            self._stsg_file_handle.seek(pos)
            line = self._stsg_file_handle.readline()

            data = json.loads(line)
            # TODO: Warn of NO STSG data
            if "stsg" not in self._stsg_buffer[id]:
                print(
                    f"Warning: item {id} has no STSG Data. Using default value 'No STSG data'."
                )
            self._stsg_buffer[id] = data.get("stsg", "No STSG data")

    def _get_stsg_data(self, question_id):
        if question_id in self._stsg_buffer:
            return self._stsg_buffer.pop(question_id)
