from collections import namedtuple


class BatchProcessor:

    Result = namedtuple("Result", ["status", "client", "id", "response"])

    def pipeline(self, data, *transforms):
        for e in data:
            res = e
            for f in transforms:
                res = f(res)

    def batch_requests(self, payload_gen, ollama_client, endpoint, kwargs=None):
        consectuive_errors = 0
        for i, sample in enumerate(payload_gen, 1):
            id = sample["qid"]
            payload = sample["payload"]

            try:
                print(f"\nGenerating respone for iteration {i} - id: {id}")
                response = ollama_client.ollama_completion_request(
                    payload, endpoint, **kwargs
                )

                yield Result("ok", ollama_client, id, payload, response)

            except requests.RequestException as e:
                yield Result(
                    "error", ollama_client, id, payload, getattr(e, "response")
                )

    def save_task(self, generator, output_file_path=None):
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
            output_dir, f"errors_{self.model}_{start_timestamp}.txt"
        )

        print(f"Responses will be saved to: {output_file_path}")
        print(f"Errors will be logged to: {error_file}")
        print(" Starting Response Generation ".center(80, "="))

        # Using line buffering
        with open(output_file_path, "w", buffering=1, encoding="utf-8") as res_f, open(
            error_file, "w", buffering=1, encoding="utf-8"
        ) as error_f:

            for result in generator:
                status = result.status
                response = result.response
                id = result.id

                if status == "ok":
                    # Create response object
                    response_obj = {
                        "qid": id,
                        "response": response,
                    }

                    res_f.write(json.dumps(response_obj) + "\n")
                    res_f.flush()
                elif status == "error":
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

                yield result

    def batch_generate(self, ollama_client, prompts, output_file_path=None):
        self.pipeline(
            prompts,
            lambda gen: self.batch_requests(gen, ollama_client, endpoint="generate"),
            lambda gen: self.save_task(gen, output_file_path),
        )
