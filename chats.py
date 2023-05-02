import gc
import json
import requests
import os.path
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from streamer.endpointstreaming import EndpointStreaming
from langchain.memory import RedisChatMessageHistory, ConversationBufferWindowMemory

class ChatbotInstance():
    def create_chatbot(self, call_back_url):
        return LlamaCpp(
            model_path=f"{self.config['model']}", verbose=True, callbacks=[EndpointStreaming(call_back_url, self.config['model'])], streaming=True,
            stop=["###"],
            n_threads=self.config['threads'],
            n_ctx=2048,
            use_mlock=True
        )
    def prepare_a_new_chatbot(self):
        self.chatbot_bindings = self.create_chatbot()
        self.condition_chatbot()

    def deka_prompt(self):
        template = """ The following is a friendly conversation between a human and an AI named Deka. A kind and helpful AI bot built to help users solve problems.
        Deka is talkative and provides lots of specific details from its context. If Deka does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        ### Human: {input}
        ### Assistant:"""

        return PromptTemplate(
            input_variables=["history", "input"], template=template
        )

    def condition_chatbot(self, conditionning_message = """
Instruction: Act as Deka. A kind and helpful AI bot built to help users solve problems.
Deka: Welcome! I'm here to assist you with anything you need. What can I do for you today?"""
                          ):

        response = self.chatbot_bindings.ingest(conditionning_message, is_system_prompt=True)
        file_name = "session-"+ str(self.id)
        
    def save(self, human, response):
        self.memory.add_user_message(human)
        self.memory.add_ai_message(response)
    
    def load(self):
        # NOTE(Michael): redis url strut redis://localhost:6379/0
        self.memory = RedisChatMessageHistory(key_prefix="ChatMemory:", session_id=str(self.id), ttl=21600, url="redis://"+self.config["redis_host"]+":"+str(self.config["redis_port"])+"/"+str(self.config["redis_db"]))

    def __init__(self, id, config:dict) -> None:
        # workaround for non interactive mode
        self.config = config
        self.id = id

    @staticmethod
    def condition_in_worker(id, config):
        chat = ChatbotInstance(id, config)
        chat.chatbot_bindings = chat.create_chatbot()

        chat.condition_chatbot()

        chat.save()

    @staticmethod
    def generate_in_worker(id, config, call_back_url, message):
        chat = ChatbotInstance(id, config)

        chat.chatbot_bindings = chat.create_chatbot(call_back_url)
        chat.load()

        conversation = ConversationChain(
                        prompt=chat.deka_prompt(),
                        llm=chat.chatbot_bindings, 
                        memory=ConversationBufferWindowMemory(memory_key="history", chat_memory=self.memory, human_prefix="### Human:", ai_prefix="### Assistant:", k=2)
                    )
        
        response = conversation.predict(input=message)

        print("=========================")
        print(response)

        chat.save(message, response)