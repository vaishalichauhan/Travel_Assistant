"""
Phase 1 sanity check.

Run this first, before touching RAG or MCP. If this doesn't work, nothing
downstream will either — so get this green before moving on.

Usage:
    python src/test_llm.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads OPENAI_API_KEY from .env

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("In one sentence, what makes Singapore a good place for a first-time visitor?")
    print("\n--- LLM RESPONSE ---")
    print(response.content)
    response = llm.invoke("Is Singapore a country?")
    print(response.content)
    print("--------------------\n")
    print("If you see a real sentence above, your environment and API key are working.")

if __name__ == "__main__":
    main()
