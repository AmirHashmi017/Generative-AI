from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
import os

os.environ['LANGCHAIN_PROJECT']='Chatbot-Project'

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages= state['messages']
    response= llm.invoke(messages)
    return{"messages":AIMessage(content=response.content)}

conn= sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer= SqliteSaver(conn)

cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

graph= StateGraph(ChatState)

graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot= graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads= set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return all_threads

def save_chat_title(thread_id: str, title: str):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (str(thread_id), title)
    )
    conn.commit()

def get_chat_title(thread_id: str) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (str(thread_id),))
    result = cursor.fetchone()
    return result[0] if result else None

def get_all_chat_titles() -> dict:
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, title FROM chat_titles ORDER BY created_at DESC")
    results = cursor.fetchall()
    return {thread_id: title for thread_id, title in results}
