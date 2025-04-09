import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.append("../src")

import batch_processor as bp
import prompt_formatters as pf

# noqa: E402 - disables the warning for this line
from ollama_manager import OllamaRequestManager  # noqa: E402
from ollama_manager import STARPromptGenerator


class GeneratorTestUnit(unittest.TestCase):

    def test_save(self):
        output_filename = "test_output_generators/output.jsonl"

        def resp_gen(prompts):
            for i in prompts:
                yield {
                    "status": "ok",
                    "client": None,
                    "id": None,
                    "payload": 0,
                    "response": i,
                }

        inp = ["hello", "I", "am", "testing"]
        gen = resp_gen(["hello", "I", "am", "testing"])
        for _ in bp.stream_save(
            response_generator=gen,
            response_formatter=bp.GenerateResponseFormatter(),
            output_file_path=output_filename,
        ):
            pass

        self.assertTrue(os.path.exists(output_filename))

        with open(output_filename, "r") as out_f:
            responses = [json.loads(line) for line in out_f.readlines()]

            for i, resp in zip(inp, responses):
                content = resp["response"]
                self.assertEqual(i, content)

