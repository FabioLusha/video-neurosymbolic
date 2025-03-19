import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.append("../src")
import prompt_formatters as pf  # noqa: E402

# noqa: E402 - disables warning for this line
from ollama_manager import AutoChat  # noqa: E402
from ollama_manager import ChatServer, OllamaRequestManager, STARPromptGenerator


class TestChatService(unittest.TestCase):

    def setUp(self):
        # start the server
        self.server = subprocess.Popen(["python", "scaffold_server.py"])
        print("Started the Scaffold Server")

        # Wait for the server to start
        time.sleep(2)

    def TearDown(self):
        if self.server:
            self.server.terminate()  # Gracefully terminate the server
            self.server.wait()  # Wait for the process to exit

            print("Stopped scaffold server")

    def test_chat_service(self):
        manager = OllamaRequestManager(
            base_url="http://localhost:8000", ollama_params={"model": "llama2"}
        )

        chat_server = ChatServer(manager)

        msg = "This is a message"
        chat_server.send_msg(msg)

        self.assertEqual(len(chat_server.chat_history), 2)

    def test_auto_chat(self):
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

            manager = OllamaRequestManager(
                base_url="http://localhost:8000", ollama_params={"model": "llama2"}
            )

            automatic_chat = AutoChat(ollama_client=manager)

            # Create output directory
            output_filename = "test_output2/out_resp.jsonl"

            # Get first 10 prompts
            prompts = []
            for i, prompt in enumerate(generator.generate(pformatter)):
                if i >= 10:
                    break
                prompts.append(prompt)

            def r_fun(response):
                n_reply = 0

                def next_f(response):
                    nonlocal n_reply
                    n_reply += 1

                    if n_reply > 1:
                        return None, None
                    else:
                        return "This is an automatic reply", next_f

                return next_f(response)

            # Process the prompts
            automatic_chat.batch_chat(prompts, r_fun, output_filename)


if __name__ == "__main__":
    unittest.main()
