import json
import batch_processor
from ollama_manager import Result

class ChatServer:
    def __init__(self, ollama_client):
        self.client = ollama_client
        self.chat_history = []

    def send_msg(self, content):
        self.chat_history.append({"role": "user", "content": content})

        response = self.client.chat_completion(self.chat_history)

        self.chat_history.append(
            {"role": response.get("role", ""), "content": response.get("content", "")}
        )
        return response

    def flush_chat(self):
        self.chat_history = []

def auto_reply_gen(result_gen, reply, ollama_client):
    for result in result_gen:
        messages = result.pyload
        messages.append(result.response)
        chat_completion

    yield Result()
class AutoChat:

    def __init__(self, chat_server=None, ollama_client=None):
        self.chat = chat_server
        if chat_server is None:
            if ollama_client is None:
                raise ValueError(
                    "Your have to provide argument between chat_server or ollama_clinet"
                )
            self.chat = ChatServer(ollama_client)

    def batch_chat(self, prompts, auto_reply_f, output_file_path=None):
        

        auto_reply_gen = lambda gen: (auto_reply_f(i) for i in gen)

        reply, next_reply_f = auto_reply_f(response)
        while reply is not None:
            self.chat.send_msg(reply)
            reply, next_reply_f = next_reply_f(response)


        b_processor = batch_processor.BatchProcessor()
        pipe = batch_processor.Pipeline(
            lambda gen: b_processor.batch_request(gen, self.ollama_client, endpoint='chat'),
            auto_reply_gen,
            lambda gen: b_processor.stream_save(gen, batch_processor.ChatResponseFormatter())
        )

        pipe.consume(prompts)

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
                output_dir, f"responses_{self.model}_{start_timestamp}.jsonl"
            )

        error_file = os.path.join(
            output_dir, f"errors_{self.chat.client.model}_{start_timestamp}.txt"
        )

        print(f"Responses will be saved to: {output_file_path}")
        print(f"Errors will be logged to: {error_file}")
        print(" Starting Response Generation ".center(80, "="))

        # Using line buffering
        with open(output_file_path, "w", buffering=1, encoding="utf-8") as res_f:

            consecutive_errors = 0
            error = False
            for i, sample in enumerate(prompts, 1):
                self.chat.flush_chat()

                id = sample["qid"]
                prompt = sample["prompt"]

                try:
                    print(f"\nGenerating response for prompt {i}")

                    # The call is synchronous, therfore we wait for the response
                    response = self.chat.send_msg(prompt)
                    # Generate reply

                    

                    if self.chat.chat_history:
                        # Create response object
                        response_obj = {
                            "qid": id,
                            "chat_history": self.chat.chat_history,
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
                        response = getattr(e, "response")

                        self.chat.chat_history.append(
                            {
                                "role": response["role"],
                                "content": "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\n\n"
                                + response["content"],
                            }
                        )

                        error_msg = {
                            "qid": id,
                            "prompt": prompt,
                            "chat_history": self.chat.chat_history,
                            "error": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat(),
                        }

                        error_f.write(
                            json.dumps(error_msg, indent=2, ensure_ascii=False) + "\n"
                        )
                        error_f.flush()

                        response_file_msg = {
                            "qid": id,
                            "chat_history": self.chat.chat_history,
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
