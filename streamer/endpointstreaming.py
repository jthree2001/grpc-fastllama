"""Callback Handler streams to stdout on new llm token."""
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
from typing import Any
import json
import sys
from langchain.schema import LLMResult

class EndpointStreaming(StreamingStdOutCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print(token, end='', flush=True)

        def reuse_send(flush):
            self.reply = self.reply + flush
            self.send_to_callback(flush)
            return ""

        self.flush = self.flush + token
        if '\n' in self.flush:
            if self.flush.count("```") != 1:
                # Note(Michael): just to double check we're not in a code block
                self.flush = reuse_send(self.flush)
        elif "===================" in self.flush:
            self.flush = reuse_send(self.flush)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.send_to_callback(self.flush)
                

    def send_to_callback(self, message):
        data = {
        "message": message
        }

        json_data = json.dumps(data)

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, data=json_data, headers=headers)
        if response.status_code == 200:
            print("Message sent successfully!")
        else:
            print("Error occurred: ", response.text)

    def __init__(self, url, config:dict) -> None:
        # workaround for non interactive mode
        self.config = config
        self.reply = ""
        self.flush = ""
        self.url = url 
