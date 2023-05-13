from typing import Any, List, Mapping, Optional, Generator, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests


class TxtGenLlm(LLM):
    streaming: bool = False
    """Whether to stream the results, token by token."""

    api_endpoint: str = "192.168.254.11:5000"
    """Whether to stream the results, token by token."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    async def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ):

        request =  self._default_params
        request['promp'] = prompt
        request['stopping_strings'] = stop
        
        websocket = websockets.connect(f'http://{self.api_endpoint}/api/v1/stream', ping_interval=None)
        websocket.send(json.dumps(request))

        while True:
            incoming_data = websocket.recv()
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    token = incoming_data['text']
                    if run_manager:
                        run_manager.on_llm_new_token(
                            token=token, verbose=self.verbose, log_probs=log_probs
                        )
                    return token
                case 'stream_end':
                    return
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:

        request =  self._default_params
        request['prompt'] = prompt
        request['stopping_strings'] = stop

        # if self.streaming:
        #     # If streaming is enabled, we use the stream
        #     # method that yields as they are generated
        #     # and return the combined strings from the first choices's text:
        #     combined_text_output = ""
        #     for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
        #         combined_text_output += token
        #     return combined_text_output
        # else:
        response = requests.post(f'http://{self.api_endpoint}/api/v1/generate', json=request)
        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            return result
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling llama_cpp."""
        return {
            'prompt': '',
            'max_new_tokens': 250,
            'do_sample': True,
            'temperature': 1.3,
            'top_p': 0.1,
            'typical_p': 1,
            'repetition_penalty': 1.18,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }