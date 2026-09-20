"""
Phase 7: Streamlit UI. Build this LAST, after agent.py works from the
command line.

Run:
    streamlit run src/app.py
"""

import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent  # reuses everything from agent.py

st.set_page_config(page_title="Singapore Travel Assistant", page_icon="\U0001F3E0")
st.title("Singapore Travel Planning Assistant")
st.caption("Destination knowledge via RAG · live weather & currency via MCP tools")


@st.cache_resource
def get_loop():
    """One event loop, reused for the whole app session. Streamlit reruns
    the script on every interaction, but @st.cache_resource keeps this same
    loop object alive across reruns instead of asyncio.run() creating and
    closing a new one each time (which breaks the MCP subprocess connections
    tied to the first loop)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


@st.cache_resource
def get_agent():
    loop = get_loop()
    return loop.run_until_complete(build_agent())


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage / AIMessage

# render past turns
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask about attractions, weather, currency, or a full itinerary...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    agent = get_agent()
    loop = get_loop()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = st.session_state.chat_history + [HumanMessage(content=user_input)]
            result = loop.run_until_complete(agent.ainvoke({"messages": messages}))
            answer = result["messages"][-1].content
            st.markdown(answer)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))
