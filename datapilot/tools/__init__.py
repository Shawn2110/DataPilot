"""
tools package — Contains all 8 pipeline tools.

Each tool is a LangChain-compatible tool that:
  1. Receives the shared PipelineState
  2. Performs one step of the ML pipeline
  3. Writes results back to PipelineState
  4. Returns a text summary for the agent to reason about

The agent decides which tool to call and in what order.
"""
