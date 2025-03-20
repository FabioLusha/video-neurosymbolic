import json
import os
import sys
import subprocess
import tempfile
import time
import unittest

sys.path.append("../src")

# noqa: E402 - disables the warning for this line
from ollama_manager import OllamaRequestManager  # noqa: E402
from ollama_manager import STARPromptGenerator
import batch_processor
import prompt_formatters as pf

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
            suffix=".json", delete=False
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
            base_url="http://localhost:8000", ollama_params={"model": "llama2"}
        )

        response = manager.generate_completion("hi")
        print("response")
        server_response = "Hi, I am alive "
        self.assertEqual(response, server_response)
        
    def test_batch_request(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:8000", ollama_params={"model": "llama2"}
        )

        # Initialize the generator
        generator = STARPromptGenerator(input_filename=self.temp_data_file.name)

        prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
        pformatter = pf.OpenEndedPrompt(prompt_format)

        prompts = []
        for i, prompt in enumerate(generator.generate(pformatter)):
            if i >= 10:
                break
            prompts.append(prompt)

        payload_gen = (
            {'qid': p['qid'], 'payload': {**manager.ollama_params, 'prompt': p['prompt']}}
            for p in prompts)
        
         
        for i in batch_processor.batch_request(payload_gen, manager, 'generate'):
            pass

    def test_request_save_pipeline(self):
        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:8000", ollama_params={"model": "llama2"}
        )

        test_data = [
            {
                "question_id": f"q{i}",
                "question": f"Test question {i}",
                "stsg": f"Test graph {i}",
            }
            for i in range(10)
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w+") as in_f:
            json.dump(test_data, in_f)

            in_f.seek(0)
            generator = STARPromptGenerator(input_filename=in_f.name)

            prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
            pformatter = pf.OpenEndedPrompt(prompt_format)
            
            # Get first 10 prompts
            prompts = []
            for i, prompt in enumerate(generator.generate(pformatter)):
                if i >= 10:
                    break
                prompts.append(prompt)

            payload_gen = (
                {'qid': p['qid'], 'payload': {**manager.ollama_params, 'prompt': p['prompt']}}
                for p in prompts)


            payloads = list(payload_gen)
            output_filename = "test_output/out_resp.jsonl"

            batch_processor.batch_generate(manager, payloads, output_filename)

            self.assertTrue(os.path.exists(output_filename))

            with open(output_filename, 'r') as out_f:
                responses = [json.loads(line) for line in out_f.readlines()]

                for resp in responses:
                    content = resp['response']
                    server_response = "Hi, I am alive "
                    self.assertEqual(content, server_response)




if __name__ == "__main__":
    unittest.main()
