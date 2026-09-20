"""
Phase 4: ask destination questions against the knowledge base only (no MCP yet).

Run ingest.py first so ./chroma_db exists.

Usage:
    python src/rag_chain.py "What are the must-visit attractions in Singapore?"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"

SYSTEM_PROMPT = """You are a travel assistant for Singapore. Answer ONLY using the
CONTEXT below, which was retrieved from a travel knowledge base.

Rules:
- Do not invent facts that are not in the context.
- If the context does not contain enough information to answer, say so clearly
  instead of guessing.
- After your answer, list the sources you used under "Sources:", using the
  source titles given in the context.
- Distinguish between facts stated in the context and any suggestions you are
  adding yourself; label suggestions as "Suggestion:".

CONTEXT:
{context}
"""


def format_context(docs) -> str:
    parts = []
    for d in docs:
        title = d.metadata.get("source_title", "Unknown source")
        parts.append(f"[Source: {title}]\n{d.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, k: int = 4) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)

    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "I don't have information about that in the knowledge base."

    context = format_context(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What are the must-visit attractions in Singapore?"
    print(f"\nQ: {question}\n")
    print(answer_question(question))
