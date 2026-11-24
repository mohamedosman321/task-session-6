"""Helper to ask the agent one question and show which tools it uses."""

from llama_index.core.agent.workflow import ToolCall


async def ask(agent, ctx, question, verbose=True, max_iterations=20):
    """Send one question. The Context keeps the chat memory between questions."""
    handler = agent.run(user_msg=question, ctx=ctx, max_iterations=max_iterations)
    async for event in handler.stream_events():
        if verbose and isinstance(event, ToolCall):
            print(f"  [tool] {event.tool_name} {event.tool_kwargs}")
    return str(await handler)
