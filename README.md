# Singapore AI Travel Planning Assistant

RAG over a Singapore travel knowledge base + two MCP tools (weather, currency),
combined by a LangChain agent, served through a Streamlit chat UI.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your OpenAI key into .env
```

## Build order (do these in sequence — each depends on the last working)

1. **`python src/test_llm.py`** — confirms your API key and environment work.
2. **Fill in `data/knowledge_base/*.txt`** — replace the placeholder content
   with real Singapore travel info (see file for format). Add more .txt
   files as needed, one per source, each with a `SOURCE_TITLE` /
   `SOURCE_URL` header.
3. **`python src/ingest.py`** — chunks, embeds, and stores everything in
   `./chroma_db`. Re-run this any time you change the knowledge base.
4. **`python src/rag_chain.py "your question"`** — test pure destination
   Q&A with citations, no MCP involved yet.
5. **Test the MCP tools standalone:**
   ```bash
   python src/mcp_servers/weather_server.py --test
   python src/mcp_servers/currency_server.py --test
   ```
6. **`python src/agent.py "your question"`** — the full combined agent
   (RAG + both MCP tools) from the command line.
7. **`streamlit run src/app.py`** — the chat UI with multi-turn memory.

## Architecture

- **RAG**: `data/knowledge_base/*.txt` → `ingest.py` (chunk + embed) →
  Chroma vector store (`./chroma_db`) → retrieved at query time and passed
  to the LLM as context, with source titles/URLs preserved as metadata for
  citations.
- **MCP**: two standalone MCP servers (`src/mcp_servers/`) each expose one
  tool over stdio — `get_weather_forecast` (Open-Meteo, no key needed) and
  `convert_currency` (open.er-api.com, no key needed). `agent.py` connects
  to both via `MultiServerMCPClient`.
- **Agent**: a LangChain tool-calling agent (`agent.py`) has all three tools
  (retrieval + 2 MCP) and a system prompt instructing it which tool to use
  for which question type, to never fabricate data, and to label sources in
  its final answer.
- **UI**: Streamlit (`app.py`), keeps chat history in `st.session_state` and
  passes it to the agent each turn for multi-turn context.

## Prompt strategy

(Expand this section for your submission — see the assignment's Section 5.)
The system prompt in `agent.py` explicitly separates the three tools by
purpose, forbids answering destination questions without the retrieval tool,
forbids fabricating MCP data, and requires the final answer to label each
piece of information by source (knowledge base / weather tool / currency
tool / LLM suggestion).

## Known gaps to fill in before submission

- Real knowledge base content (currently placeholder text).
- Sample questions + responses document (deliverable #19).
- Demo recording (deliverable #20).
- `langchain-mcp-adapters` API may need small adjustments depending on the
  version you install — check its docs if `agent.py` errors on import/setup.
