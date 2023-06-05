from langchain.agents import AgentExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain.input import get_color_mapping
from langchain.callbacks.manager import (
    CallbackManagerForChainRun
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException
)
import time
from langchain.tools.base import BaseTool

class DekaExecutor(AgentExecutor):
    handle_parsing_errors = True

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"Observation: {observation}\n"
        return thoughts

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            # Call the LLM to see what to do.
            tries = 5
            force_change = ""
            while tries > 0:
                print("entering retry block")
                tries = tries - 1
                inputs["agent_scratchpad"] = inputs["agent_scratchpad"] + force_change
                output = self.agent.plan(
                    intermediate_steps,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **inputs,
                )
                print("--------------------------------------")
                if isinstance(output, str):
                    print("------------Updating adding action---------------")
                    output = output.split("Obs", 1)[0]
                    print("------------Stripping obs=============="+output)
                    combined_scratchpad = inputs["agent_scratchpad"] + output
                    output_last_line = combined_scratchpad.split("\n")[-1]
                    if "Thought" in output_last_line:
                        if "Action" in output_last_line:
                            force_change = f"""{output} \nActionInput:"""
                        else:
                            force_change = f"""\n{output} \nAction: """
                    elif "ActionInput" in output_last_line:
                        print("[[[[[[[[[[[[[[ENDGAME]]]]]]]]]]]]]]")
                        force_change = f"""{output}"""
                        output = self._agent_parser(inputs["agent_scratchpad"] + force_change)
                        tries = 0
                    elif "Action" in output_last_line:
                        force_change = f"""{output} \nActionInput:"""
                    else:
                        force_change = f"""\Thought: {output} \nAction: """
                elif not output:
                    print("------------Updating input---------------")
                    force_change = """\Thought: """
                else: 
                    tries = 0
                    force_change = ""
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result


    def _agent_parser(self, llm_output: str):
        #  [line.removeprefix('Action:') for line in llm_output.split('\n') if "Action:" in line]
        for line in llm_output.split('\n'):
            if "action" in line.lower():
                action = line.lower().split("action", 1)[-1].replace(':','')
                if action:
                    break
        if not action:
            print("No valid command found, returning llm string")
            return llm_output
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        for line in llm_output.split('\n'):
            if "action input" in line.lower():
                action_input = line.lower().split("action input", 1)[-1].replace(':','')
                if action_input:
                    break
            if "actioninput" in line.lower():
                action_input = line.lower().split("actioninput", 1)[-1].replace(':','')
                if action_input:
                    break
        if not action_input:
            action_input = ""
        action_input = action_input.split("Observation:")[0]
        
        tool_names = [tool.name for tool in self.tools]
        for tool in tool_names:
            if tool.lower() in action.lower():
                action = tool
                break
        # Return the action and action input
        print("Executing: "+action + " do: "+action_input)
        print("=========================")
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], old_inputs: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        
        thoughts = self._construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts}
        full_inputs = {**old_inputs, **new_inputs}
        return full_inputs

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        force_change =""
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            print("taking next step")
            full_inputs = self.get_full_inputs(intermediate_steps, inputs)
            next_step_output = ""
            
            i = 0
            
            
            print(inputs)
            
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                full_inputs,
                intermediate_steps,
                run_manager=run_manager,
            )

            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            print("ddddddddddddddddddddd")
            print(intermediate_steps)

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)