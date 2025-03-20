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
