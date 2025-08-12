# ==============================
# ECHO CHATBOT WITH API KEY MANAGEMENT
# ==============================
import streamlit as st
import os
import uuid
import sqlite3
from dotenv import load_dotenv, set_key
from typing import TypedDict, Annotated

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# ==============================
# API KEY SETUP
# ==============================
file_name = ".env"
load_dotenv(file_name, override=True)

def save_api(api_key):
    """Save API key to .env file without removing other variables."""
    api_key = api_key.strip()
    set_key(file_name, "GOOGLE_API_KEY", api_key)
    os.environ["GOOGLE_API_KEY"] = api_key
    st.success("API Key Saved Successfully")
    st.rerun()

def del_api():
    """Delete the GOOGLE_API_KEY from .env file and current session."""
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            lines = f.readlines()
        with open(file_name, "w") as f:
            for line in lines:
                if not line.startswith("GOOGLE_API_KEY="):
                    f.write(line)
    os.environ.pop("GOOGLE_API_KEY", None)
    st.success("API Key Deleted Successfully")
    st.rerun()


# ==============================
# SIDEBAR: API KEY MANAGEMENT
# ==============================
st.sidebar.title("🔑 API Key Management")
current_api = os.environ.get("GOOGLE_API_KEY")
if current_api:
    st.sidebar.success("✅ API Key is set")
    st.sidebar.code(current_api)
    if st.sidebar.button("Delete API Key", type="primary"):
        del_api()
else:
    st.sidebar.warning("⚠️ No API Key found")
    with st.sidebar.form("api_form"):
        api_key = st.text_input("Enter Your GOOGLE API Key", type="password")
        submitted = st.form_submit_button("Save API Key")
        if submitted and api_key:
            save_api(api_key)
        elif submitted and not api_key:
            st.error("Please enter an API key")

# Stop execution if no API key is set
if not current_api:
    st.stop()


# ==============================
# LANGGRAPH BACKEND
# ==============================
class ChatBot_State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
cnn = sqlite3.connect(database="Chat_DataBase.db", check_same_thread=False)
chekpointer = SqliteSaver(conn=cnn)

def chat_response(state: ChatBot_State):
    messages = f"""
    You are a helpful assistant, your name is Echo. Solve the user's query.
    \nQuery:{state["messages"]}
    """
    response = chat_llm.invoke(messages)
    return {
        "messages": [AIMessage(content=response.content)]
    }

graph = StateGraph(ChatBot_State)
graph.add_node("chat_response", chat_response)
graph.add_edge(START, "chat_response")
graph.add_edge("chat_response", END)
chat_bot = graph.compile(checkpointer=chekpointer)


# ==============================
# CHAT HISTORY FUNCTIONS
# ==============================
def get_all_threads():
    all_threads = set()
    for check in chekpointer.list(None):
        all_threads.add(check.config["configurable"]["thread_id"])
    return list(all_threads)

def create_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = create_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chat_bot.get_state(config={"configurable": {"thread_id": thread_id}}).values
    return state.get("messages", [])

def get_thread_title(thread_id):
    messages = load_conversation(thread_id)
    if messages:
        return messages[0].content[:30] + ("..." if len(messages[0].content) > 30 else "")
    return f"Chat {thread_id[:12]}"


# ==============================
# SESSION STATE INIT
# ==============================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = create_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_thread(st.session_state["thread_id"])
CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# ==============================
# SIDEBAR: CHAT HISTORY
# ==============================
st.sidebar.title("🤖 ECHO")
if st.sidebar.button("💬 New Chat"):
    reset_chat()

st.sidebar.header("Recent Chats")
for thread_id in st.session_state["chat_threads"][::-1]:
    thread_title = get_thread_title(thread_id)
    if st.sidebar.button(thread_title, key=f"btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]


# ==============================
# MAIN SCREEN
# ==============================
st.title("🤖 Hi there, I'm Echo!")

# Display chat history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Store user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        ai_response = st.write_stream(
            msg_chunk.content
            for msg_chunk, _ in chat_bot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    # Store assistant response
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
