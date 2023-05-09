import gc
import json
import requests
import os.path
import langchain
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from streamer.endpointstreaming import EndpointStreaming
from langchain.memory import RedisChatMessageHistory, ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
import re

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
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            print("No valid command found, returning blank string")
            return ""
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        for tool in self.tool_names:
            if tool.lower() in action.lower():
                action = tool
        # Return the action and action input
        print("Executing: "+action + " do: "+action_input)
        print("=========================")
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
class ChatbotInstance():
    def create_chatbot(self, call_back_url):
        return LlamaCpp(
            model_path=f"{self.config['model']}", verbose=True, callbacks=[EndpointStreaming(call_back_url, self.config)], streaming=True,
            stop=["###","</s>"],
            n_threads=self.config['threads'],
            n_ctx=2048,
            use_mlock=True
        )
        
    def create_chain_catbot(self, call_back_url):
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

    def deka_prompt(self):
        template = template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Try to stay on topic of the question, after you know the answer, reply with "Final Answer:."
Remember that Action should always be one of these tools [{tool_names}].
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"], template=template
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

        chat.chatbot_bindings = chat.create_chatbot(call_back_url)
        chat.load()

        template = """ The following is a friendly conversation between a human and an AI named Deka. A kind and helpful AI bot built to help users solve problems.
        Deka is talkative and provides lots of specific details from its context. If Deka does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        ### Human: {input}
        ### Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )


        conversation = ConversationChain(
                        prompt=prompt,
                        llm=chat.chatbot_bindings, 
                        memory=ConversationBufferWindowMemory(memory_key="history", chat_memory=chat.memory, human_prefix="### Human:", ai_prefix="### Assistant:", k=2)
                    )
        
        response = conversation.predict(input=message)
        print(response)

        chat.save(message, response)

    @staticmethod
    def chain_in_worker(id, config, call_back_url, message):
        chat = ChatbotInstance(id, config)

        chat.chatbot_bindings = chat.create_chain_catbot(call_back_url)
        chat.load()

        readonlymemory = ReadOnlySharedMemory(memory=ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=chat.memory, human_prefix="### Human:", ai_prefix="### Assistant:"))
        summry_chain = LLMChain(
            llm=chat.chatbot_bindings, 
            prompt=chat.deka_prompt_summary(), 
            verbose=True, 
            memory=readonlymemory, # use the read-only memory to prevent the tool from modifying the memory
        )

        search = DuckDuckGoSearchRun()
        wikipedia = WikipediaAPIWrapper()
        tools = [Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
            ),
            Tool(
                name = "Wikipedia",
                func=wikipedia.run,
                description="useful for when you need to answer questions about any topic"
            )
        ]
        if len(chat.memory.messages) != 0:
            tools.append(Tool(
                name = "Summary",
                func=summry_chain.run,
                description="useful for when you summarize this conversation. The input to this tool should be a string, representing who will read this summary."
                )
            )

        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser(tool_names)
        print("===========Tools Available==============")
        print(tool_names)

        agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=chat.chatbot_bindings, prompt=chat.deka_prompt()) , 
            output_parser=output_parser,
            stop=["\nObservation:", "</s>"], 
            allowed_tools=tool_names,
            max_iterations=3, early_stopping_method="generate"
        )
        # agent = LLMSingleActionAgent(tools, chat.chatbot_bindings, prompt=chat.deka_prompt(), allowed_tools=tool_names, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=ConversationBufferWindowMemory(memory_key="chat_history", chat_memory=chat.memory, human_prefix="### Human:", ai_prefix="### Assistant:", k=2))

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

        response = agent_executor.run(tools=tools, input=message, tool_names=tool_names, agent_scratchpad="")

        print("=========================")
        print(response)

        chat.save(message, response)