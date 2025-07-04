import json
import os, sys
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


# Add src directory to path FIRST (before other imports)
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)  # Insert at start to prioritize local imports

import batch_processor
import prompt_formatters as pf  # noqa: E402
from chat_utils import ChatServer

# noqa: E402 - disables warning for this line
from datasets import PromptDataset
from ollama_manager import OllamaRequestManager


class TestChatService(unittest.TestCase):

    def setUp(self):
        # start the server
        self.server = subprocess.Popen(["python", "scaffold_server.py"])
        print("Started the Scaffold Server")

        # Wait for the server to start
        time.sleep(2)

    def tearDown(self):
        if self.server:
            self.server.terminate()  # Gracefully terminate the server
            self.server.wait()  # Wait for the process to exit

            print("Stopped scaffold server")

    def test_chat_service(self):
        manager = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        chat_server = ChatServer(manager)

        msg = "This is a message"
        chat_server.send_msg(msg)

        self.assertEqual(len(chat_server.chat_history), 2)

    def test_auto_reply(self):
        reply = "Automatic reply"

        manager = OllamaRequestManager(
            base_url="http://localhost:5555", ollama_params={"model": "llama2"}
        )

        messages = [{"role": "user", "content": "First message"}]
        payload = {**manager.ollama_params, "messages": messages}
        resp = manager.chat_completion(messages)
        result_form = {
            "status": "ok",
            "client": manager,
            "id": None,
            "payload": payload,
            "response": resp,
        }

        new_result = None
        auto_rep_gen = batch_processor.auto_reply_gen((r for r in [result_form]), reply)

        for res in auto_rep_gen:
            new_result = res

        self.assertEqual(len(new_result["payload"]["messages"]), 3)
        # The scaffold server repeats the message
        self.assertTrue(new_result["response"].get("content").strip().endswith(reply))

    def test_batch_auto_chat(self):
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

            dataset = PromptDataset(in_f.name, prompt_formatter)

            manager = OllamaRequestManager(
                base_url="http://localhost:5555", ollama_params={"model": "llama2"}
            )

            # Create output directory
            output_filename = "test_output_chat/out_resp.jsonl"

            # Get first 10 prompts
            prompts = []
            for i in range(len(dataset)):
                if i >= 10:
                    break
                prompts.append(dataset[i])

            reply = "This is an automatic reply"
            batch_processor.batch_automatic_chat_reply(
                manager, prompts, reply, output_filename
            )

            with open(output_filename, "r") as out_f:
                responses = [json.loads(line) for line in out_f.readlines()]

                for resp in responses:
                    chat_history = resp["chat_history"]

                    self.assertEqual(len(chat_history), 4)
                    # The scaffold server repeats the message
                    self.assertTrue(chat_history[3]["content"].strip().endswith(reply))


if __name__ == "__main__":
    unittest.main()
