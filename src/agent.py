"""
Phase 6: the combined agent. Don't start here — get test_llm.py, ingest.py,
rag_chain.py, and both mcp_servers/*.py --test runs working first.

This wires three tools into one LangChain agent:
  - retrieve_travel_knowledge  (wraps your Chroma retriever, phase 3-4)
  - get_weather_forecast       (MCP tool, phase 5)
  - convert_currency           (MCP tool, phase 5)

It uses langchain-mcp-adapters (MultiServerMCPClient) to load the MCP tools
and launches your two servers as subprocesses over stdio automatically.

This uses LangChain 1.0's `create_agent` (langchain.agents), the current
standard way to build tool-using agents — the older AgentExecutor /
create_tool_calling_agent pattern was moved to the separate
`langchain-classic` package in 1.0.

NOTE: langchain-mcp-adapters' exact API can still shift between versions —
check `pip show langchain-mcp-adapters` and its README if the MCP import or
MultiServerMCPClient call fails, and adjust accordingly.

Usage:
    python src/agent.py "Plan a 3-day Singapore itinerary and adjust for the weather forecast"
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
SERVERS_DIR = Path(__file__).parent / "mcp_servers"

SYSTEM_PROMPT = """You are an AI Travel Planning Assistant for Singapore.

You have three tools:
- retrieve_travel_knowledge: use this for ANY question about attractions,
  neighbourhoods, transport, culture, food, or itinerary ideas. This is your
  ONLY source for destination facts — never answer these from general
  knowledge.
- get_weather_forecast: use this ONLY for current/forecast weather questions.
- convert_currency: use this ONLY for currency conversion questions.

Rules:
- Never invent destination facts, weather data, or exchange rates.
- If retrieve_travel_knowledge doesn't return enough information, say so
  clearly rather than guessing.
- If an MCP tool fails or is unavailable, tell the user plainly instead of
  making up a number.
- In your final answer, label information by source: "From knowledge base:",
  "From weather tool:", "From currency tool:", and "Suggestion:" for anything
  you are recommending yourself rather than quoting a source.
- Preserve user preferences mentioned earlier in the conversation (e.g.
  traveling with kids, budget, interests) when making suggestions.
"""


def make_retrieval_tool():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)

    @tool
    def retrieve_travel_knowledge(query: str) -> str:
        """Retrieve Singapore destination knowledge (attractions, transport,
        culture, food, itineraries) relevant to the query, with sources."""
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "NO_RESULTS: nothing relevant found in the knowledge base."
        parts = []
        for d in docs:
            title = d.metadata.get("source_title", "Unknown source")
            parts.append(f"[Source: {title}]\n{d.page_content}")
        return "\n\n".join(parts)

    return retrieve_travel_knowledge


async def build_agent():
    mcp_client = MultiServerMCPClient({
        "weather": {
            "command": "python",
            "args": [str(SERVERS_DIR / "weather_server.py")],
            "transport": "stdio",
        },
        "currency": {
            "command": "python",
            "args": [str(SERVERS_DIR / "currency_server.py")],
            "transport": "stdio",
        },
    })
    mcp_tools = await mcp_client.get_tools()

    tools = [make_retrieval_tool()] + mcp_tools

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent


async def main():
    question = " ".join(sys.argv[1:]) or (
        "Create a three-day Singapore itinerary and adjust it based on the weather forecast."
    )
    agent = await build_agent()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
