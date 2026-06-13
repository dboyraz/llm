"""Agentic RAG over the course materials.

Gives the LLM a `search` tool over a chunked index and lets it decide when (and
what) to search. Runs through the Claude Code subscription via the Claude Agent
SDK — no API key required.

Run it with:

    uv run agent.py
"""

import sys

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index

QUESTION = "How does the agentic loop work, and how is it different from plain RAG?"

INSTRUCTIONS = (
    "You're a course teaching assistant. Answer the student's question using "
    "the search tool. Make multiple searches with different keywords before "
    "answering."
)


def build_index():
    """Read the lesson pages, chunk them, and build a minsearch index."""
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [f.parse() for f in reader.read()]
    chunks = chunk_documents(documents, size=2000, step=1000)

    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    return index


index = build_index()


@tool("search", "Search the course materials for relevant chunks", {"query": str})
async def search(args):
    """Search the course chunk index and return the top matches."""
    results = index.search(args["query"], num_results=5)
    text = "\n\n".join(r["filename"] + "\n" + r["content"] for r in results)
    return {"content": [{"type": "text", "text": text}]}


server = create_sdk_mcp_server(name="rag", version="1.0.0", tools=[search])

options = ClaudeAgentOptions(
    mcp_servers={"rag": server},
    allowed_tools=["mcp__rag__search"],
    system_prompt=INSTRUCTIONS,
    model="sonnet",
)


async def main():
    search_calls = 0
    answer_parts = []

    async for msg in query(prompt=QUESTION, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and block.name == "mcp__rag__search":
                    search_calls += 1
                    print(f"[search #{search_calls}] {block.input.get('query')!r}")
                elif isinstance(block, TextBlock):
                    answer_parts.append(block.text)

    print("\n=== ANSWER ===")
    print("".join(answer_parts))
    print(f"\n=== SEARCH CALLS: {search_calls} ===")
    return search_calls


if __name__ == "__main__":
    # Make Unicode in the answer safe to print on the Windows console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    anyio.run(main)
