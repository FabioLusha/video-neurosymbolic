import json
import logging
import os
from collections import namedtuple
from pathlib import Path

import requests

Result = namedtuple("Result", ["status", "client", "id", "payload", "response"])

# Base directary is parent of current file's directory - star_code
BASE_DIR = Path(__file__).parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.NOTSET) # delegate filtering to logger
ch_fmt = logging.Formatter(
    "=[%(levelname)s] :- %(message)s"
)
ch.setFormatter(ch_fmt)

fh = logging.FileHandler(str(LOG_DIR / "star_code.log"))
fh.setLevel(logging.WARNING)
fh_fmt = logging.Formatter(
    "=[%(asctime)s][%(levelname)s] - %(name)s :- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
fh.setFormatter(fh_fmt)

logger.addHandler(ch)
logger.addHandler(fh)



class OllamaRequestManager:

    def __init__(self, base_url, ollama_params):
        self.base_url = base_url
        self.model = ollama_params["model"]

        # Setting some of the params to pass to ollama
        self.ollama_params = ollama_params

        self.default_handlers = dict()
        self.default_handlers["generate"] = OllamaGenerateHandler()
        self.default_handlers["chat"] = OllamaChatHandler()


    def load_model(self):
        logger.info(f" Loading model {self.ollama_params['model']} ".center(40, "="))
        # An empty prompt will cause ollama to load the specified model
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.ollama_params["model"]},
            )
        except requests.ConnectionError as e:
            raise requests.ConnectionError("Error while connecting to ollama") from e
        except requests.RequestException as e:
            raise requests.RequestException("Error while connecting to ollama") from e

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
        token = data.get("response", "")
        result["token_stream"].append(token)
        return result

    def finalize_result(self, result):
        return {"content": "".join(result["token_stream"]), "role": result["role"]}


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
