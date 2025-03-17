import json
import logging
import os
import traceback
from datetime import datetime

import prompt_formatters as pf
import requests


class Chat:
    def __init__(ollama_client):
        self.client = ollama_client
        self.chat_history = []

    def send_msg(self, content):
        self.chat_history.append({"role": "user", "content": content})

        response = ollama_client.chat_completion(chat_history)

        self.chat_history.append(
            {"role": response.get("role", ""), "content": response.get("content", "")}
        )


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
            payload={**self.ollama_parmas, "messages": messages},
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
            e.result = result
            raise

        return handler.finalize_result(result)

    def batch_requests(self, prompts, output_file_path=None):
        # Create logs directory if it doesn't exist
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        start_timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")

        if output_file_path:
            # Create directory if it doesn't exist
            if os.path.exists(output_file_path):
                raise FileExistsError(f"The file {output_file_path} already exists!")

            # Create directory if it doesn't exists
            dir = os.path.dirname(output_file_path)
            if dir != "":
                output_dir = dir
                os.makedirs(dir)

        else:
            # Saving the response as JSON in jsonl format,
            # where each json object is saved in one line
            output_file_path = os.path.join(
                output_dir, f"responses_{self.model}_{start_timestamp}.jsonl"
            )

        error_file = os.path.join(
            output_dir, f"errors_{self.model}_{start_timestamp}.txt"
        )

        print(f"Responses will be saved to: {output_file_path}")
        print(f"Errors will be logged to: {error_file}")
        print(" Starting Response Generation ".center(80, "="))

        # Using line buffering
        with open(output_file_path, "w", buffering=1, encoding="utf-8") as res_f:

            consecutive_errors = 0
            error = False
            for i, sample in enumerate(prompts, 1):
                id = sample["qid"]
                prompt = sample["prompt"]

                try:
                    print(f"\nGenerating response for prompt {i}")

                    response = self.generate_completion(prompt)
                    if response:
                        # Create response object
                        response_obj = {
                            "qid": id,
                            "response": response,
                        }

                        res_f.write(json.dumps(response_obj) + "\n")
                        res_f.flush()

                        error = False
                        consecutive_errors = 0

                except requests.RequestException as e:
                    if error:
                        consecutive_errors += 1

                    error = True
                    with open(
                        error_file, "a", buffering=1, encoding="utf-8"
                    ) as error_f:
                        result = getattr(e, "result")
                        response = handler.finalize_result(result)

                        error_msg = {
                            "qid": id,
                            "prompt": prompt,
                            "response": response,
                            "error": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat(),
                        }

                        error_f.write(
                            json.dumps(error_msg, indent=2, ensure_ascii=False) + "\n"
                        )
                        error_f.flush()

                        response_file_msg = {
                            "qid": id,
                            "response": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n"
                            + response,
                        }

                        res_f.write(json.dumps(response_file_msg) + "\n")
                        res_f.flush()

                        print(
                            f"Error at iteration {i}\n"
                            f"Prompt id:{id}\n"
                            "Look at the log file for specifics on the error"
                        )

                finally:
                    if consecutive_errors > 5:
                        raise Exception(
                            "There have been {5} consecutive errors!\n"
                            "The process has stopped!"
                        )


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

    def __init__(self, input_filename):
        if not os.path.exists(input_filename):
            raise OSError(f"No such file or directory: '{input_filename}'")

        self.input_filename = input_filename

    def generate(self, prompt_formatter, ids=None, start=0, limit=None):
        """
        Args:
            prompt_formatter (PromptFormatter):
                An object that given a data point form the STAR dataset
                formats the prompt according to what is sepcified in the object

            start (int): from which sample to start generation

            limit (int): how many prompt to generate

            mcq (boolean): specifies if we need to use the MCQ prompt


        Returns:
            prompt
        """
        try:
            with open(self.input_filename, "r") as in_file:
                q_stsg_data = json.load(in_file)

                if ids:
                    q_stsg_data = [
                        sample for sample in q_stsg_data if sample["question_id"] in ids
                    ]

                for i, sample in enumerate(q_stsg_data, 1):
                    if i < start:
                        continue
                    if limit and i > (limit + start):
                        break

                    prompt = prompt_formatter.format(sample)

                    yield {"qid": sample["question_id"], "prompt": prompt}

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
