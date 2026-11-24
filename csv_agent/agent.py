"""Build the LlamaIndex agent (step 5 of the build)."""

from llama_index.core.agent.workflow import FunctionAgent

from csv_agent.config import get_llm
from csv_agent.prompts import SYSTEM_PROMPT
from csv_agent.tools import DataStore, build_tools


def build_agent(llm=None, store=None, output_dir="outputs"):
    """Create the agent and the DataStore it works on.

    Returns (agent, store). The store is returned so you can inspect the
    cleaned DataFrame yourself with `store.df`.
    """
    store = store or DataStore(output_dir=output_dir)
    llm = llm or get_llm()
    agent = FunctionAgent(
        name="csv_analyst",
        description="Explores, cleans and analyses CSV and Excel files.",
        tools=build_tools(store),
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, store
