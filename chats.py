import gc
import json
import requests
import os.path
import langchain
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain, LLMMathChain
from langchain.chains import ConversationChain
from streamer.endpointstreaming import EndpointStreaming
from langchain.memory import RedisChatMessageHistory, ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import WikipediaAPIWrapper, SearxSearchWrapper
from custom.txtgenllm import TxtGenLlm
from custom.dekaexecutor import DekaExecutor
from typing import Tuple
import re

class RetryTool():
    log: str
    tool: str
    def __init__(self, log, tool):
        super().__init__()
        self.log = log
        self.tool = tool
class CustomOutputParser(AgentOutputParser):
    tool_names=[]

    def __init__(self, tool_names):
        super().__init__()
        self.tool_names = tool_names

    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        print("Entering Parsing method")
        print(llm_output)
        print("=========================")
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action = [line.removeprefix('Action:') for line in llm_output.split('\n') if "Action:" in line]
        if not action:
            print("No valid command found, returning blank string")
            return ""
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = action[0]
        action_input = [line.removeprefix('Action Input:') for line in llm_output.split('\n') if "Action Input:" in line]
        if not action_input:
            action_input = ""
        else:
            action_input = action_input[0]
        action_input = action_input.split("Observation:")[0]
        
        for tool in self.tool_names:
            if tool.lower() in action.lower():
                action = tool
        # Return the action and action input
        print("Executing: "+action + " do: "+action_input)
        print("=========================")
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class ChatbotInstance():
    def create_chatbot(self, call_back_url):
        if self.config['model'] == "api":
            return TxtGenLlm(
                api_endpoint=self.config['model_api_endpoint'],
                verbose=True,
                stop=["###","</s>"],
                callbacks=[EndpointStreaming(call_back_url, self.config)]
            )
        else:
            return LlamaCpp(
                model_path=f"{self.config['model']}", verbose=True, callbacks=[EndpointStreaming(call_back_url, self.config)], streaming=True,
                stop=["###","</s>"],
                n_threads=self.config['threads'],
                n_ctx=2048,
                use_mlock=True
            )
        
    def create_chain_catbot(self, call_back_url):
        if self.config['model'] == "api":
            return TxtGenLlm(
                api_endpoint=self.config['model_api_endpoint'],
                verbose=True,
                stop=["Observation:", "</s>", "###", "\nObservation:"],
                callbacks=[EndpointStreaming(call_back_url, self.config)]
            )
        else:
            return LlamaCpp(
                model_path=f"{self.config['model']}", verbose=True, callbacks=[EndpointStreaming(call_back_url, self.config)], streaming=True,
                n_threads=self.config['threads'],
                n_ctx=2048,
                use_mlock=True
            )

    def prepare_a_new_chatbot(self):
        self.chatbot_bindings = self.create_chatbot()
        self.condition_chatbot()

    def deka_prompt_summary(self):
        template = """Please Summarize this conversation in relation to {input}:
        {chat_history}"""

        return PromptTemplate(
            input_variables=["chat_history", "input"], template=template
        )

    def deka_prompt(self, tools_object):
        template = """Answer the following Input as best you can. You have access to the following tools: \n"""
        for value in tools_object:
            template = template + f"{value.name}, {value.description}. \n"
        template = template + """\nTry to stay on topic of the Input, after you know the answer, reply with "Final Answer:."
`Action Input:` must follow by the action.
Use the following format:

Input: the input question or task you must answer
ASSISTANT:  think about what to do
Action: should be one of [{tool_names}]
Action Input: the input to the action above
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 2 times)
ASSISTANT: I now know the final answer
Final Answer: the final answer to the original input question

Input: Who is the Atlanta Football team?
ASSISTANT:  I should use Search.
Action: Search
Action Input: Who is the Atlanta Football team?
Observation: The Atlanta Falcons are a professional American football team based in Atlanta.
ASSISTANT: I now know the final answer
Final Answer: The Atlanta Falcons are a professional American football team based in Atlanta.

Input: What is Jus Soli?
ASSISTANT:  I should use Wikipedia.
Action: Wikipedia
Action Input: History of Savanah Georgia
Observation: Page: Jus soli
Summary: Jus soli (English:  juss SOH-ly,  yooss SOH-lee), commonly referred to as birthright citizenship, is the right of anyone born in the territory of a state to nationality or citizenship.Jus soli was part of the English common law, in contrast to jus sanguinis, which derives from the Roman law that influenced the civil-law systems of mainland Europe.Jus soli is the predominant rule in the Americas; explanations for this geographical phenomenon include: the establishment of lenient laws by past European colonial powers to entice immigrants from the Old World and displace native populations in the New World, along with the emergence of successful Latin American independence movements that widened the definition and granting of citizenship, as a prerequisite to the abolishment of slavery since the 19th century. Outside the Americas, however, jus soli is rare. Since the Twenty-seventh Amendment of the Constitution of Ireland was enacted in 2004, no European country grants citizenship based on unconditional or near-unconditional jus soli.Almost all states in Europe, Asia, Africa and Oceania grant citizenship at birth based upon the principle of jus sanguinis ("right of blood"), in which citizenship is inherited through parents rather than birthplace, or a restricted version of jus soli in which citizenship by birthplace is automatic only for the children of certain immigrants.
ASSISTANT: I now know the final answer
Final Answer: us soli refers to the legal principle whereby individuals are considered citizens of the country in which they were born, regardless of their ancestry or the status of their parents.

Input: Who founded new york?
ASSISTANT: I already know this one.
Final Answer: The Italian Giovanni da Verrazzano founded New York in 1524

Input: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            input_variables=["tool_names", "input", "agent_scratchpad"], template=template
        )

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
    def generate_in_worker(id, config, call_back_url, message):
        chat = ChatbotInstance(id, config)
        endpoint = EndpointStreaming(call_back_url, config)

        chat.chatbot_bindings = chat.create_chatbot(call_back_url)
        chat.load()

        template = """ The following is a friendly conversation between a human and an AI named Deka. A kind and helpful AI bot built to help users solve problems.
Deka is talkative and provides lots of specific details from its context. If Deka does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
USER: {input}
ASSISTANT:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )


        conversation = ConversationChain(
                        prompt=prompt,
                        llm=chat.chatbot_bindings, 
                        memory=ConversationBufferWindowMemory(memory_key="history", chat_memory=chat.memory, human_prefix="USER:", ai_prefix="ASSISTANT::", k=2)
                    )
        
        response = conversation.predict(input=message)
        print("==================")
        print(response)
        endpoint.send_to_callback(response)

        chat.save(message, response)

    @staticmethod
    def chain_in_worker(id, config, call_back_url, message):
        chat = ChatbotInstance(id, config)
        endpoint = EndpointStreaming(call_back_url, config)

        chat.chatbot_bindings = chat.create_chain_catbot(call_back_url)
        chat.load()

        readonlymemory = ReadOnlySharedMemory(memory=ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=chat.memory, human_prefix="USER:", ai_prefix="ASSISTANT::"))
        summry_chain = LLMChain(
            llm=chat.chatbot_bindings, 
            prompt=chat.deka_prompt_summary(), 
            verbose=True, 
            memory=readonlymemory, # use the read-only memory to prevent the tool from modifying the memory
        )

        search = SearxSearchWrapper(searx_host="http://192.168.254.85:8089", unsecure=True)
        wikipedia = WikipediaAPIWrapper()
        llm_math = LLMMathChain.from_llm(chat.chatbot_bindings, verbose=True)

        tools = [
            Tool(
            name = "Search",
            func=search.run,
            description="Useful for when you need to answer questions about current events."
            ),
            Tool(
                name = "Wikipedia",
                func=wikipedia.run,
                description="Useful for when you need to answer questions about any historical topic."
            ),
            Tool(
                name = "Math",
                func=llm_math.run,
                description="Userful when you need to do compilcated math problems."
            )
        ]
            
        # if len(chat.memory.messages) != 0:
        #     tools.append(Tool(
        #         name = "Summary",
        #         func=summry_chain.run,
        #         description="useful for when you summarize previous conversations."
        #         )
        #     )

        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser(tool_names)
        print("===========Tools Available==============")
        print(tool_names)
        # full_inputs = self.get_full_inputs(intermediate_steps, **kwargs)
        agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=chat.chatbot_bindings, prompt=chat.deka_prompt(tools)) , 
            output_parser=output_parser,
            stop=["Observation:", "</s>", "###", "\nObservation:"], 
            allowed_tools=tool_names,
            max_iterations=3, early_stopping_method="generate",
            memory=ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=chat.memory, human_prefix="USER:", ai_prefix="ASSISTANT::", k=2)
        )
        # agent = LLMSingleActionAgent(tools, chat.chatbot_bindings, prompt=chat.deka_prompt(), allowed_tools=tool_names, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=chat.memory, human_prefix="USER:", ai_prefix="ASSISTANT::", k=2))

        agent_executor = DekaExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        response = agent_executor.run(tools=tools, input=message, tool_names=tool_names, agent_scratchpad="")

        print("=========================")
        print(response)
        endpoint.send_to_callback(response)

        chat.save(message, response)