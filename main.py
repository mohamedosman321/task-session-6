"""Chat with the CSV/Excel analysis agent in the terminal.

Usage:
    python main.py                      # then type: load data/sample_sales.csv
    python main.py --file data/sample_sales.csv
"""

import argparse
import asyncio

from llama_index.core.workflow import Context

from csv_agent import build_agent
from csv_agent.runner import ask


async def chat(file_path):
    agent, _ = build_agent()
    ctx = Context(agent)

    print("CSV/Excel analysis agent. Type 'exit' to quit.\n")

    if file_path:
        print(f"You: load {file_path}")
        answer = await ask(agent, ctx, f"Load the file {file_path} and give me a short overview of it.")
        print(f"\nAgent: {answer}\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            answer = await ask(agent, ctx, question)
        except Exception as exc:  # network errors, rate limits, bad tool calls
            print(f"\nSomething went wrong: {exc}\nTry asking again or rephrase.\n")
            continue
        print(f"\nAgent: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with a CSV/Excel analysis agent.")
    parser.add_argument("--file", help="CSV or Excel file to load at the start")
    args = parser.parse_args()
    asyncio.run(chat(args.file))


if __name__ == "__main__":
    main()
