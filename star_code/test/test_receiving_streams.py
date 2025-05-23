import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

import requests

sys.path.append("../src")

import batch_processor
import batch_processor as bp
# noqa: E402 - disables the warning for this line
import ollama_manager as om
import prompt_formatters as pf
from ollama_manager import OllamaRequestManager  # noqa: E402
from prompt_datasets import PromptDataset


class StreamingReceiverTestUnit(unittest.TestCase):
    def setUp(self):
        # Test data - creating a small JSON file with sample data
        test_data = [
            {
                "question_id": f"q{i}",
                "question": f"Test question {i}",
                "stsg": f"Test graph {i}",
            }
            for i in range(10)
        ]

        # Save test data
        self.temp_data_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

        with open(self.temp_data_file.name, "w") as f:
            json.dump(test_data, f)

        self.temp_sys_prompt_file = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        )

        # Create a simple system prompt
        with open(self.temp_sys_prompt_file.name, "w") as f:
            f.write("You are a helpful assistant.")

        # Start the server
        self.server = subprocess.Popen(["python", "scaffold_server.py"])
        print("Started the Scaffold Server")

        # Wait for the server to start
        time.sleep(2)

    def tearDown(self):
        # Close and Remove temporary files
        for temp_file in [self.temp_data_file, self.temp_sys_prompt_file]:
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

        if self.server:
            self.server.terminate()  # Gracefully terminate the server
            self.server.wait()  # Wait for the processer to exit

            print("Stopped Scaffold server")

    def test_streaming_receiver(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        response = manager.generate_completion("hi")
        server_response = "Hi, I am alive "
        self.assertEqual(response, server_response)

    def test_timeout(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        payload = {"model": "llama2", "prompt": "hi"}

        try:
            response = manager.ollama_completion_request(
                payload,
                "timeout",
                om.OllamaGenerateHandler(),
                req_timeout=2,
            )
            print(response)
            server_response = "Hi, I am alive "
        except requests.RequestException as e:
            self.assertTrue(hasattr(e, "response"))
            self.assertIsInstance(e.response, str)
            self.assertEqual(e.response.strip(), "Hi, I")
        else:
            self.fail("Exception not raiesed")

    def test_stream_request(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:5555",
            ollama_params={"model": "llama2"},
        )

        prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
        prompt_formatter = pf.OpenEndedPrompt(prompt_format)

        # Initialize the generator
        dataset = PromptDataset(self.temp_data_file.name, prompt_formatter)

        prompts = []
        for i in range(len(dataset)):
            if i >= 10:
                break
            prompts.append(dataset[i])

        payload_gen = (
            {
                "qid": p["qid"],
                "payload": {**manager.ollama_params, "prompt": p["prompt"]},
            }
            for p in prompts
        )

        for i in batch_processor.stream_request(payload_gen, manager, "generate"):
            self.assertIsInstance(i, dict)
            self.assertTrue(
                set(["id", "status", "client", "payload", "response"]).issubset(
                    i.keys()
                )
            )
            self.assertEqual(i["status"], "ok")

    def test_timeout_stream_request(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:5555",
            ollama_params={"model": "llama2"},
        )

        prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
        prompt_formatter = pf.OpenEndedPrompt(prompt_format)

        # Initialize the generator
        dataset = PromptDataset(self.temp_data_file.name, prompt_formatter)

        prompts = []
        for i in range(len(dataset)):
            if i >= 10:
                break
            prompts.append(dataset[i])

        payload_gen = (
            {
                "qid": p["qid"],
                "payload": {**manager.ollama_params, "prompt": p["prompt"]},
            }
            for p in prompts
        )

        gen = batch_processor.stream_request(
            payload_gen,
            manager,
            "timeout",
            handler=om.OllamaGenerateHandler(),
            req_timeout=5,
        )

        response = next(gen)
        self.assertEqual(response["status"], "error")
        self.assertTrue(hasattr(response["error"], "response"))

    def test_request_save_pipeline(self):
        # Initialize the Ollama manager
        ollama_client = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        test_data = [
            {
                "question_id": f"q{i}",
                "question": f"Test question {i}",
                "stsg": f"Test graph {i}",
            }
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_f:
            json.dump(test_data, in_f)
            in_f.seek(0)

            prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
            prompt_formatter = pf.OpenEndedPrompt(prompt_format)

            # Initialize the generator
            dataset = PromptDataset(self.temp_data_file.name, prompt_formatter)

            # Get first 10 prompts
            prompts = []
            for i in range(len(dataset)):
                if i >= 10:
                    break
                prompts.append(dataset[i])

            output_filename = "test_output/out_resp.jsonl"
            batch_processor.batch_generate(ollama_client, prompts, output_filename)

            self.assertTrue(os.path.exists(output_filename))
            with open(output_filename, "r") as out_f:
                responses = [json.loads(line) for line in out_f.readlines()]

                for resp in responses:
                    content = resp["response"]
                    server_response = "Hi, I am alive "
                    self.assertEqual(content, server_response)

    def test_request_save_errors_pipeline(self):
        # Initialize the Ollama manager
        ollama_client = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        test_data = [
            {
                "question_id": f"q{i}",
                "question": f"Test question {i}",
                "stsg": f"Test graph {i}",
            }
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_f:
            json.dump(test_data, in_f)
            in_f.seek(0)

            prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
            prompt_formatter = pf.OpenEndedPrompt(prompt_format)

            # Initialize the generator
            dataset = PromptDataset(in_f.name, prompt_formatter)

            # Get first 10 prompts
            prompts = []
            for i in range(len(dataset)):
                if i >= 10:
                    break
                prompts.append(dataset[i])

            output_filename = "test_output/out_resp_2.jsonl"
            error_filename = "test_output/errors.jsonl"

            def payload_gen(prompts):
                for p in prompts:
                    yield {
                        "qid": p["qid"],
                        "payload": {
                            **ollama_client.ollama_params,
                            "prompt": p["prompt"],
                        },
                    }

            pipe = bp.Pipeline(
                payload_gen,
                lambda gen: bp.stream_request(
                    gen,
                    ollama_client,
                    endpoint="timeout",
                    consecutive_errors_thresh=10,
                    handler=om.OllamaGenerateHandler(),
                    req_timeout=2,
                ),
                lambda gen: bp.stream_save(
                    gen, bp.GenerateResponseFormatter(), output_filename, error_filename
                ),
            )

            pipe.consume(prompts)

            # This test is deprecated, we don't write anymore faulty errors in the output_file
            # self.assertTrue(os.path.exists(output_filename))
            # with open(output_filename, "r") as out_f:
            #     responses = [json.loads(line) for line in out_f.readlines()]

            #     # if this assertion fails check if the consecutive_errors_threshold, it must be set >= 10
            #     self.assertEqual(
            #         len(responses),
            #         len(prompts),
            #         "The pipeline was broken! The pipeline should have continued to process subsequent requests",
            #     )
            #     for resp in responses:
            #         content = resp["response"]
            #         expected_response = (
            #             "CAREFUL! THE FOLLOWING RESPONSE GENERATED AN ERROR.\nHi, I"
            #         )
            #         self.assertEqual(content.strip(), expected_response)

            self.assertTrue(os.path.exists(error_filename))
            with open(error_filename, "r") as out_f:
                errors = [json.loads(line) for line in out_f.readlines()]

                # if this assertion fails check if the consecutive_errors_threshold, it must be set >= 10
                self.assertEqual(
                    len(errors),
                    len(prompts),
                    "The pipeline was broken! The pipeline should have continued to process subsequent requests",
                )
                self.assertGreater(
                    len(errors[0]["error"]), 0, "The error stacktrace was not saved"
                )

                for err in errors:
                    content = err["response"]
                    expected_response = "Hi, I"
                    self.assertEqual(content.strip(), expected_response)

    def test_consecutive_errors_stop(self):
        # Initialize the Ollama manager
        ollama_client = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        test_data = [
            {
                "question_id": f"q{i}",
                "question": f"Test question {i}",
                "stsg": f"Test graph {i}",
            }
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_f:
            json.dump(test_data, in_f)
            in_f.seek(0)

            prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
            prompt_formatter = pf.OpenEndedPrompt(prompt_format)

            # Initialize the generator
            dataset = PromptDataset(self.temp_data_file.name, prompt_formatter)

            # Get first 10 prompts
            prompts = []
            for i in range(len(dataset)):
                if i >= 10:
                    break
                prompts.append(dataset[i])

            def payload_gen(prompts):
                for p in prompts:
                    yield {
                        "qid": p["qid"],
                        "payload": {
                            **ollama_client.ollama_params,
                            "prompt": p["prompt"],
                        },
                    }

            pipe = bp.Pipeline(
                payload_gen,
                lambda gen: bp.stream_request(
                    gen,
                    ollama_client,
                    endpoint="timeout",
                    consecutive_errors_thresh=5,
                    handler=om.OllamaGenerateHandler(),
                    req_timeout=2,
                ),
            )

            try:
                pipe.consume(prompts)
            except ValueError as e:
                self.assertEqual(type(e), ValueError)
            else:
                self.fail("Should have thrown conecutive errors excpetion")


if __name__ == "__main__":
    unittest.main()
