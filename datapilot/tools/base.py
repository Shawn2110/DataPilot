"""
base.py — Base class for all pipeline tools.

LangChain tools work by:
  1. The agent reads each tool's name + description
  2. Based on the current task, the agent decides which tool to call
  3. LangChain calls the tool's _run() method with the agent's input
  4. The tool returns a string result
  5. The agent reads the result and decides what to do next (ReAct loop)

Our tools also need access to the shared PipelineState.
We solve this by storing a reference to the state in each tool instance.
This way, tools can both read from and write to the shared state.

The `args_schema` tells LangChain what input the tool expects.
LangChain validates the input before calling _run().
"""

from langchain_core.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field

from datapilot.pipeline.state import PipelineState


class EmptyInput(BaseModel):
    """
    Schema for tools that don't need any input.

    Most of our tools read directly from PipelineState,
    so they don't need the agent to provide input arguments.
    The agent just calls them with no args.
    """
    pass


class PipelineTool(LangChainBaseTool):
    """
    Base class for all DataPilot pipeline tools.

    Extends LangChain's BaseTool with:
      - Access to shared PipelineState
      - Default empty input schema

    All 8 pipeline tools inherit from this.

    Example tool:
        class MyTool(PipelineTool):
            name = "my_tool"
            description = "Does something useful"

            def _run(self, **kwargs) -> str:
                df = self.state.raw_df
                # ... do work ...
                self.state.profile = results
                return "Done: found 10 columns"
    """

    # Pydantic v2 model config — allows arbitrary types like DataFrame
    model_config = {"arbitrary_types_allowed": True}

    # The shared state object — every tool reads/writes to this
    state: PipelineState

    # Default: no input args needed (tools read from state)
    args_schema: type[BaseModel] = EmptyInput
