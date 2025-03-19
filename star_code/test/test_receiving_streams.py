import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.append("../src")
import prompt_formatters as pf
from batch_processor import BatchProcessor

# noqa: E402 - disables the warning for this line
from ollama_manager import OllamaRequestManager  # noqa: E402
from ollama_manager import STARPromptGenerator


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
        # Initialize the generator
        generator = STARPromptGenerator(input_filename=self.temp_data_file.name)

        prompt_format = "QUESTION: {question}\n" "STSG: {stsg}"
        pformatter = pf.OpenEndedPrompt(prompt_format)

        # Initialize the Ollama manager
        manager = OllamaRequestManager(
            base_url="http://localhost:8000", ollama_params={"model": "llama2"}
        )

        # Create output directory
        output_filename = "test_output/out_resp.jsonl"

        # Get first 10 prompts
        prompts = []
        for i, prompt in enumerate(generator.generate(pformatter)):
            if i >= 10:
                break
            prompts.append(prompt)

        batch_processor = BatchProcessor()
        # Process the prompts
        batch_processor.batch_generate(manager, prompts, output_filename)


if __name__ == "__main__":
    unittest.main()
