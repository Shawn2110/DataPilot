"""
Agent runner.

Thin wrapper around LangGraph's prebuilt ReAct agent. Exposes:

  - create_agent(cfg, tools, mode): compile a ReAct graph for the picked mode
  - propose_step(agent, history): return (narration, suggested_code) for the
    next step; the session controller is responsible for displaying the
    proposal, asking the user to approve, and calling execute() separately.

The controller — not this module — decides when to execute code. That keeps
the user-approval gate in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from datapilot.agent.prompts import system_prompt_for_mode
from datapilot.agent.providers.factory import create_llm
from datapilot.config import AgentConfig


@dataclass
class StepProposal:
    narration: str
    code: str
    raw_message: str


def create_agent(cfg: AgentConfig, tools: list, mode: str):
    """Build a compiled LangGraph ReAct agent."""
    from langgraph.prebuilt import create_react_agent

    llm = create_llm(cfg)
    return create_react_agent(model=llm, tools=tools), system_prompt_for_mode(mode)


async def propose_step(
    agent,
    system_prompt: str,
    user_goal: str,
    history: list[tuple[str, str]] | None = None,
) -> StepProposal:
    """
    Ask the agent for its next step. Returns a StepProposal.

    `history` is an optional list of (narration, code) pairs for prior cells —
    the controller passes them in so the agent has continuity without needing
    to re-read the raw notebook.
    """
    messages = [SystemMessage(content=system_prompt)]
    for narr, code in history or []:
        messages.append(HumanMessage(content=f"[prior step]\n{narr}\n```python\n{code}\n```"))
    messages.append(HumanMessage(content=user_goal))

    response = await agent.ainvoke({"messages": messages})
    last = response["messages"][-1]
    text = getattr(last, "content", "") or ""
    narration, code = _split_narration_and_code(text)
    return StepProposal(narration=narration, code=code, raw_message=text)


def _split_narration_and_code(text: str) -> tuple[str, str]:
    """
    Split a response into (narration, first python code block).
    Tolerates responses with no code fence.
    """
    fence = "```python"
    if fence not in text:
        return text.strip(), ""
    head, _, tail = text.partition(fence)
    code, _, _ = tail.partition("```")
    return head.strip(), code.strip()
