"""
Phase 2-3: turn data/knowledge_base/*.txt into a persisted Chroma vector store.

Each source .txt file must start with two header lines:
    SOURCE_TITLE: <title>
    SOURCE_URL: <url>
    ---
followed by the actual content. This script strips the header, uses it as
metadata on every chunk from that file, and saves everything to
./chroma_db (created automatically).

Run this once whenever you add or change knowledge base files:
    python src/ingest.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"


def load_source_file(path: Path) -> Document:
    """Parse the SOURCE_TITLE / SOURCE_URL header and return one Document
    per file (splitting happens afterward)."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    title, url, body_start = path.stem, "", 0
    for i, line in enumerate(lines):
        if line.startswith("SOURCE_TITLE:"):
            title = line.replace("SOURCE_TITLE:", "").strip()
        elif line.startswith("SOURCE_URL:"):
            url = line.replace("SOURCE_URL:", "").strip()
        elif line.strip() == "---":
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:]).strip()
    return Document(page_content=body, metadata={"source_title": title, "source_url": url, "file": path.name})


def main():
    files = sorted(KB_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {KB_DIR}. Add your knowledge base files first.")
        return

    print(f"Found {len(files)} source file(s): {[f.name for f in files]}")

    docs = [load_source_file(f) for f in files]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    print(f"Vector store persisted to {PERSIST_DIR}")

    # Quick sanity check: run one retrieval and print what comes back
    print("\n--- SANITY CHECK: retrieving for 'family friendly activities' ---")
    results = vectorstore.similarity_search("family friendly activities", k=3)
    for r in results:
        print(f"\n[{r.metadata.get('source_title')}] {r.metadata.get('source_url')}")
        print(r.page_content[:200] + "...")


if __name__ == "__main__":
    main()
