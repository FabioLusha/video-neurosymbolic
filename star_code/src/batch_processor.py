import traceback
from datetime import datetime
import json
import os
import requests
from ollama_manager import Result

class Pipeline:

    def __init__(self, *generators_funs):
        self.generator_transforms = generators_funs

    def consume(self, data):
        # I create a generator for data
        generator = (d for d in data)
        for apply_new_gen in self.generator_transforms:
            generator = apply_new_gen(generator)

        # consume the generator to have the effect
        result = []
        for i in generator:
            result.append(i)

        # some pipeline may produce only side-effects
        return result if result != [] else None
    

class BatchProcessor:

    def pipeline(self, data, *generators_f):
        data_gen = (d for d in data)
        
        new_gen = data_gen
        for apply_gen in generators_f:
            new_gen = apply_gen(new_gen)

        for _ in new_gen:
            pass

        return True

    def batch_request(self, payload_gen, ollama_client, endpoint, **kwargs):
        consectuive_errors = 0
        threshold = 10
        
        print(" Starting Response Generation ".center(80, "="))
        for i, sample in enumerate(payload_gen, 1):
            id = sample["qid"]
            payload = sample["payload"]

            try:
                print(f"\nGenerating respone for iteration {i} - id: {id}")
                response = ollama_client.ollama_completion_request(
                    payload, endpoint, **kwargs
                )

                consectuive_errors = 0
                yield Result("ok", ollama_client, id, payload, response)

            except requests.RequestException as e:
                # If an execption occurs return the exception object
                # which also contains an attribute .response with the
                # partial response
                consectuive_errors += 1
                if consectuive_errors > threshold:
                    print(f"There have been more than {threshold} consecutive errors! Stopping the program!")
                    raise ValueError(f"There have been {consectuive_errors} conscutive errors!")
                
                yield Result("error", ollama_client, id, payload, e)

    def stream_save(self, response_generator, response_formatter, output_file_path=None):
        start_timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")

        if output_file_path:
            # Create directory if it doesn't exist
            if os.path.exists(output_file_path):
                raise FileExistsError(f"The file {output_file_path} already exists!")

            # Create directory if it doesn't exists
            dir = os.path.dirname(output_file_path)
            output_dir = dir if dir != "" else "outputs"
            os.makedirs(dir)

        else:
            # Create logs directory if it doesn't exist
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)

            # Saving the response as JSON in jsonl format,
            # where each json object is saved in one line
            output_file_path = os.path.join(
                output_dir, f"responses_{start_timestamp}.jsonl"
            )

        error_file = os.path.join(
            output_dir, f"errors_{start_timestamp}.txt"
        )

        print(f"Responses will be saved to: {output_file_path}")
        print(f"Errors will be logged to: {error_file}")

        # Using line buffering
        with open(output_file_path, "w", buffering=1, encoding="utf-8") as res_f:

            for i, result in enumerate(response_generator, 1):
                status = result.status
                id = result.id

                if status == "ok":
                    # Create response object
                    response_obj = response_formatter.format_success_response(result)

                    res_f.write(json.dumps(response_obj) + "\n")
                    res_f.flush()
                elif status == "error":
                    with open(error_file, "a", buffering=1, encoding="utf-8") as error_f:
                        log_msg, response_file_msg = response_formatter.format_error_response(result)

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

    def batch_generate(self, ollama_client, prompts, output_file_path=None):
        pipe = Pipeline(
            lambda gen: self.batch_request(gen, ollama_client, endpoint="generate"),
            lambda gen: self.stream_save(gen, GenerateResponseFormatter(), output_file_path)
        )
        return pipe.consume(prompts)

class ResponseFormatter:
    def format_response(self, response_data):
        pass

    def format_error_response(self, response_data):
        pass

class GenerateResponseFormatter(ResponseFormatter):
    def format_success_response(self, response_data):
        success_response = {"qid": response_data.id, "response": response_data.response}
        return success_response

    def format_error_response(self, response_data):
        error = response_data.response

        traceback_text = ""
        if hasattr(error, "__traceback__"):
            tb = error.__traceback__
            # Format the traceback
            tb_lines = traceback.format_exception(type(error), error, tb)
            traceback_text = "".join(tb_lines)

        partial_error_response = getattr(error, "response", "")
        error_log = {
            "qid": response_data.id,
            "prompt": response_data.payload,
            "response": partial_error_response,
            "error": traceback_text,
            "timestamp": datetime.now().isoformat(),
        }
        error_response_message = {
            "qid": response_data.id,
            "response": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n"
            + partial_error_response,
        }

        return error_log, error_response_message

class ChatResponseFormatter(ResponseFormatter):
    def format_success_response(self, response_data):
        chat_history = response_data.payload

        chat_history.append({
            "role": response_data.response.get("role"),
            "content": response_data.response.get("content", "")
        })

        success_response = {
            "qid": response_data.id,
            "chat_history": chat_history
        }

        return success_response

    def format_error_response(self, response_data):
        error = response_data.response

        traceback_text = ""
        if hasattr(error, "__traceback__"):
            tb = error.__traceback__
            # Format the traceback
            tb_lines = traceback.format_exception(type(error), error, tb)
            traceback_text = "".join(tb_lines)

        partial_error_response = getattr(error, "response", "")

        chat_history = response_data.payload
        chat_history.append({
            "role": partial_error_response.get("role", "unk"),
            "content": \
                "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n" +
                partial_error_response.get("content", "")
        })

        error_log = {
            "qid": response_data.id,
            "prompt": response_data.payload,
            "chat_history": chat_history,
            "error": traceback_text,
            "timestamp": datetime.now().isoformat(),
        }
        error_response_message = {
            "qid": response_data.id,
            "chat_history": chat_history
        }

        return error_log, error_response_message

