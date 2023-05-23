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
)
import time

class DekaExecutor(AgentExecutor):

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"{observation}\n"
        return thoughts

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
            tries = 3
            i = 0
            
            while not next_step_output:
                print("entering retry block")
                full_inputs["agent_scratchpad"] = full_inputs["agent_scratchpad"] + force_change
                print(full_inputs)
                tries = tries - 1
                next_step_output = self._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    full_inputs,
                    intermediate_steps,
                    run_manager=run_manager,
                )
                if not next_step_output:
                    print("------------Updating input---------------")
                    force_change = """\nASSISTANT: """
                else: 
                    force_change = ""
                    
                if 0 < tries - 1: # i is zero indexed
                    break

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