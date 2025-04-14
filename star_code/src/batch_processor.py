import json
import os
import traceback
from datetime import datetime

import requests
from ollama_manager import Result


class Pipeline:

    def __init__(self, *generators_funs):
        self.generator_transforms = generators_funs

        # Compose the generators
        self.pipeline = self.generator_transforms[0]

    def pipe(self, generator):
        self.generator_transforms.append(generator)

    def consume(self, data):
        # consume the generator to have the effect
        generator = (d for d in data)
        for apply_new_gen in self.generator_transforms:
            generator = apply_new_gen(generator)

        for _ in generator:
            continue

        return

    def to_list(self, data):
        # consume the generator to have the effect
        generator = (d for d in data)
        for apply_new_gen in self.generator_transforms:
            generator = apply_new_gen(generator)

        result = []
        for i in generator:
            result.append(i)

        # some pipeline may produce only side-effects
        return result if result != [] else None


def stream_request(payload_gen, ollama_client, endpoint, **kwargs):
    consecutive_errors = 0
    threshold = 10

    print(" Starting Response Generation ".center(80, "="))
    for i, sample in enumerate(payload_gen, 1):
        id = sample["qid"]
        payload = sample["payload"]

        try:
            print(f"\nGenerating response for iteration {i} - id: {id}")
            response = ollama_client.ollama_completion_request(
                payload, endpoint, **kwargs
            )

            # When a response is successful reset the error count
            consecutive_errors = 0
            yield {
                **sample,
                "status": "ok",
                "id": id,
                "client": ollama_client,
                "response": response,
                # don't need the payload anymore because it is included in **sample
            }

        except requests.RequestException as e:
            # If an execption occurs return the exception object
            # which also contains an attribute ['response'] with the
            # partial response
            consecutive_errors += 1
            if consecutive_errors > threshold:
                print(
                    f"There have been more than {threshold} consecutive errors! Stopping the program!"
                )
                raise ValueError(
                    f"There have been {consecutive_errors} conscutive errors!"
                )
            yield {
                **sample,
                "status": "error",
                "id": id,
                "client": ollama_client,
                "response": e,
                # don't need the payload anymore because it is included in **sample
            }


def stream_save(response_generator, response_formatter, output_file_path=None):
    start_timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")

    if output_file_path:
        # Create directory if it doesn't exist
        if os.path.exists(output_file_path):
            raise FileExistsError(f"The file {output_file_path} already exists!")

        # Create directory if it doesn't exists
        dir = os.path.dirname(output_file_path)
        output_dir = dir if dir != "" else "outputs"
        os.makedirs(dir, exist_ok=True)

    else:
        # Create logs directory if it doesn't exist
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Saving the response as JSON in jsonl format,
        # where each json object is saved in one line
        output_file_path = os.path.join(
            output_dir, f"responses_{start_timestamp}.jsonl"
        )

    error_file = os.path.join(output_dir, f"errors_{start_timestamp}.txt")

    print(f"Responses will be saved to: {output_file_path}")
    print(f"Errors will be logged to: {error_file}")

    # Using line buffering
    with open(output_file_path, "w", buffering=1, encoding="utf-8") as res_f:

        for i, result in enumerate(response_generator, 1):
            status = result["status"]
            id = result["id"]

            if status == "ok":
                # Create response object
                response_obj = response_formatter.format_success_response(result)

                res_f.write(json.dumps(response_obj) + "\n")
                res_f.flush()
            elif status == "error":
                with open(error_file, "a", buffering=1, encoding="utf-8") as error_f:
                    log_msg, response_file_msg = (
                        response_formatter.format_error_response(result)
                    )

                    error_f.write(
                        json.dumps(log_msg, indent=2, ensure_ascii=False) + "\n"
                    )
                    error_f.flush()

                    res_f.write(json.dumps(response_file_msg) + "\n")
                    res_f.flush()

                    print(
                        f"Error at iteration {i}\n"
                        f"Prompt id:{id}\n"
                        "Look at the log file for specifics on the error"
                    )

            yield result


def batch_generate(ollama_client, prompts, output_file_path=None):

    def payload_gen(prompts):
        for p in prompts:
            yield {
                "qid": p["qid"],
                "payload": {**ollama_client.ollama_params, "prompt": p["prompt"]},
            }

    pipe = Pipeline(
        payload_gen,
        lambda gen: stream_request(gen, ollama_client, endpoint="generate"),
        lambda gen: stream_save(gen, GenerateResponseFormatter(), output_file_path),
    )
    return pipe.consume(prompts)


def auto_reply_gen(result_gen, reply):
    for result in result_gen:
        if result["status"] == "ok":
            messages = result["payload"]["messages"]
            messages.append(result["response"])
            messages.append({"role": "user", "content": reply})

            new_response = result["client"].chat_completion(messages)

            result["payload"]["messages"] = messages
            result["response"] = new_response

        yield result


def batch_automatic_chat_reply(ollama_client, prompts, reply, output_file_path=None):
    def payload_gen(prompts):
        for p in prompts:
            yield {
                "qid": p["qid"],
                "payload": {
                    **ollama_client.ollama_params,
                    "messages": [{"role": "user", "content": p["prompt"]}],
                },
            }

    pipe = Pipeline(
        # the first generator converts the prompt to the right format
        payload_gen,
        lambda payload_gen: stream_request(payload_gen, ollama_client, "chat"),
        lambda resp_gen: auto_reply_gen(resp_gen, reply),
        lambda resp_gen: stream_save(
            resp_gen, ChatResponseFormatter(), output_file_path
        ),
    )

    return pipe.consume(prompts)


class ResponseFormatter:
    def format_response(self, response_data):
        pass

    def format_error_response(self, response_data):
        pass


class GenerateResponseFormatter(ResponseFormatter):
    def format_success_response(self, response_data):
        success_response = {
            "qid": response_data["id"],
            "response": response_data["response"],
        }
        return success_response

    def format_error_response(self, response_data):
        error = response_data["response"]

        traceback_text = ""
        if hasattr(error, "__traceback__"):
            tb = error.__traceback__
            # Format the traceback
            tb_lines = traceback.format_exception(type(error), error, tb)
            traceback_text = "".join(tb_lines)

        partial_error_response = getattr(error, "response", "")
        error_log = {
            "qid": response_data["id"],
            "payload": response_data["payload"],
            "response": partial_error_response,
            "error": traceback_text,
            "timestamp": datetime.now().isoformat(),
        }
        error_response_message = {
            "qid": response_data["id"],
            "response": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n"
            + partial_error_response,
        }

        return error_log, error_response_message


class ChatResponseFormatter(ResponseFormatter):
    def format_success_response(self, response_data):
        chat_history = response_data["payload"]["messages"]

        chat_history.append(
            {
                "role": response_data["response"].get("role"),
                "content": response_data["response"].get("content", ""),
            }
        )

        success_response = {"qid": response_data["id"], "chat_history": chat_history}

        return success_response

    def format_error_response(self, response_data):
        error = response_data["response"]

        traceback_text = ""
        if hasattr(error, "__traceback__"):
            tb = error.__traceback__
            # Format the traceback
            tb_lines = traceback.format_exception(type(error), error, tb)
            traceback_text = "".join(tb_lines)

        partial_error_response = getattr(error, "response", "")

        print(response_data.keys())
        chat_history = response_data["message"]
        chat_history.append(
            {
                "role": partial_error_response.get("role", "unk"),
                "content": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n"
                + partial_error_response.get("content", ""),
            }
        )

        error_log = {
            "qid": response_data["id"],
            "payload": response_data["payload"],
            "chat_history": chat_history,
            "error": traceback_text,
            "timestamp": datetime.now().isoformat(),
        }
        error_response_message = {
            "qid": response_data["id"],
            "chat_history": chat_history,
        }

        return error_log, error_response_message


class GeneratedGraphFormatter(ResponseFormatter):
    def format_success_response(self, response_data):
        chat_history = response_data["payload"]["messages"]

        chat_history.append(
            {
                "role": response_data["response"].get("role"),
                "content": response_data["response"].get("content", ""),
            }
        )
        # remove img encoding from the chat_history
        if "images" in chat_history[0]:
            print("Removing image entry")
            del chat_history[0]["images"]

        stsg = response_data["stsg"]
        success_response = {
            "qid": response_data["id"],
            "chat_history": chat_history,
            "stsg": stsg,
        }

        return success_response

    def format_error_response(self, response_data):
        error = response_data["response"]

        traceback_text = ""
        if hasattr(error, "__traceback__"):
            tb = error.__traceback__
            # Format the traceback
            tb_lines = traceback.format_exception(type(error), error, tb)
            traceback_text = "".join(tb_lines)

        partial_error_response = getattr(error, "response", "")

        chat_history = response_data["payload"]["message"]
        chat_history.append(
            {
                "role": partial_error_response.get("role", "unk"),
                "content": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n"
                + partial_error_response.get("content", ""),
            }
        )

        print(f"Chat history keys: {chat_history[0].keys()}")
        if "images" in chat_history[0]:
            print("Error branch: Removing image entry")
            del chat_history[0]["images"]

        stsg = response_data["stsg"]
        error_log = {
            "qid": response_data["id"],
            "payload": response_data["payload"],
            "chat_history": chat_history,
            "stsg": stsg,
            "error": traceback_text,
            "timestamp": datetime.now().isoformat(),
        }
        error_response_message = {
            "qid": response_data["id"],
            "chat_history": chat_history,
        }

        return error_log, error_response_message
