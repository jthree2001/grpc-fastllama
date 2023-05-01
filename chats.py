from fastLLaMa import Model
import gc
import json
import requests
import os.path

class ChatbotInstance():
    def create_chatbot(self):
        return Model(
            path=f"./models/{self.config['model']}", #path to model
            num_threads=self.config['threads'], #number of threads to use
            n_ctx=self.config['ctx_size'], #context size of model
        )
    def prepare_a_new_chatbot(self):
        self.chatbot_bindings = self.create_chatbot()
        self.condition_chatbot()

    def condition_chatbot(self, conditionning_message = """
Instruction: Act as Deka. A kind and helpful AI bot built to help users solve problems.
Deka: Welcome! I'm here to assist you with anything you need. What can I do for you today?"""
                          ):

        response = self.chatbot_bindings.ingest(conditionning_message, is_system_prompt=True)
        file_name = "session-"+ str(self.id)
    
    def generate_message(self):
        gc.collect()

        reply = self.chatbot_bindings.generate(
            self.prompt_message,#self.full_message,#self.current_message,
            n_predict=len(self.current_message)+self.config['n_predict'],
            temp=self.config['temp'],
            top_k=self.config['top_k'],
            top_p=self.config['top_p'],
            repeat_penalty=self.config['repeat_penalty'],
            repeat_last_n = self.config['repeat_last_n'],
            #seed=self.config['seed'],
            n_threads=self.config['threads']
        )
        # reply = self.chatbot_bindings.generate(self.full_message, n_predict=55)
        self.generating=False
        return reply
    def save(self):
        file_name = "models/session-"+ str(self.id)+".bin"
        self.chatbot_bindings.save_state(file_name)
    
    def load(self):
        file_name = "models/session-"+ str(self.id)+".bin"
        if os.path.isfile(file_name):
            self.chatbot_bindings.load_state(file_name)

    def __init__(self, id, config:dict) -> None:
        # workaround for non interactive mode
        self.config = config
        self.id = id

    @staticmethod
    def condition_in_worker(id, config):
        gc.collect()
        chat = ChatbotInstance(id, config)
        chat.chatbot_bindings = chat.create_chatbot()

        chat.condition_chatbot()

        chat.save()

    @staticmethod
    def generate_in_worker(id, config, call_back_url, message):
        gc.collect()
        chat = ChatbotInstance(id, config)
        chat.chatbot_bindings = chat.create_chatbot()
        chat.load()

        res = chat.chatbot_bindings.ingest(message)
        print("--------------------")
        print(res)
        reply = ""
        flush = ""

        def send_to_callback(message):
            data = {
            "message": message
            }

            json_data = json.dumps(data)

            headers = {"Content-Type": "application/json"}
            nonlocal call_back_url
            response = requests.post(call_back_url, data=json_data, headers=headers)
            if response.status_code == 200:
                print("Message sent successfully!")
            else:
                print("Error occurred: ", response.text)
        def stream_token(x: str) -> None:
            """
            This function is called by the llama library to stream tokens
            """
            nonlocal reply
            nonlocal flush
            flush = flush + x
            print(x, end='', flush=True)
            if '\n' in flush:
                if flush.count("```") != 1:
                    # Note(Michael): just to double check we're not in a code block
                    send_to_callback(flush)
                    reply = reply + flush
                    flush = ""
        
        generation_done = chat.chatbot_bindings.generate(
            num_tokens=500, 
            top_p=0.95, #top p sampling (Optional)
            temp=0.8, #temperature (Optional)
            repeat_penalty=1.0, #repetition penalty (Optional)
            streaming_fn=stream_token, #streaming function
            stop_words=["###"] #stop generation when this word is encountered (Optional)
            )
        
        if not flush in reply:
            send_to_callback(flush)
            reply = reply + flush
            flush = ""

        print("=========================1")
        print(reply)
        print(generation_done)

        chat.save()
        gc.collect()