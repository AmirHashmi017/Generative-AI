from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool,InjectedToolArg
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import os

os.environ['LANGCHAIN_PROJECT']='Chatbot-Project'

load_dotenv()
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")
EXCHANGERATE_API_KEY= os.getenv("EXCHANGERATE_API_KEY")

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

client= MultiServerMCPClient(
    {
        "calculator":{
            "transport": "stdio",
            "command": "python",
            "args": ["./mcp_server.py"]
        },
        "expense":
        {
            "transport":"streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

search_tool= DuckDuckGoSearchRun(region="us-en")


@tool
def get_weather_data(city: str) -> str:
  """"This tool takes location and gives it's temprature in output"""
  url= f"https://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"

  response = requests.get(url)

  return response.json()

@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """
    This function fecthes the currency conversion factor between the given 
    base currency and target currency.
    """
    url=f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base_currency}/{target_currency}"
    response= requests.get(url)
    return response.json()

@tool
def convert_currency(base_value: float, conversion_rate:Annotated[float,InjectedToolArg])->float:
    """Given a currency conversion rate this function calculates the target currency value
    from given base currecy value"""
    return base_value*conversion_rate


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

async def init_db():
    conn = await aiosqlite.connect('chatbot.db')
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.commit()
    return conn

conn = None
checkpointer = None

async def setup_checkpointer():
    global conn, checkpointer
    conn = await init_db()
    checkpointer = AsyncSqliteSaver(conn)

async def build_graph():
    base_tools= [search_tool, get_weather_data, get_conversion_factor, convert_currency]
    mcp_tools= await client.get_tools()
    tools= base_tools+mcp_tools
    llm_with_tools= llm.bind_tools(tools)
    tool_node= ToolNode(tools)
    async def chat_node(state: ChatState):
        messages= state['messages']
        response= await llm_with_tools.ainvoke(messages)
        return{"messages": response}


    graph= StateGraph(ChatState)

    graph.add_node("chat_node",chat_node)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,"chat_node")
    graph.add_conditional_edges("chat_node",tools_condition)
    graph.add_edge("tools","chat_node")
    graph.add_edge("chat_node",END)

    chatbot= graph.compile(checkpointer=checkpointer)
    return chatbot

chatbot = None

async def initialize_chatbot():
    global chatbot
    if chatbot is None:
        await setup_checkpointer()
        chatbot = await build_graph()
    return chatbot

async def retrieve_all_threads():
    all_threads= set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return all_threads

async def save_chat_title(thread_id: str, title: str):
    await conn.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (str(thread_id), title)
    )
    await conn.commit()

async def get_chat_title(thread_id: str) -> str:
    cursor = await conn.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (str(thread_id),))
    result = await cursor.fetchone()
    return result[0] if result else None

async def get_all_chat_titles() -> dict:
    cursor = await conn.execute("SELECT thread_id, title FROM chat_titles ORDER BY created_at DESC")
    results = await cursor.fetchall()
    return {thread_id: title for thread_id, title in results}

async def get_messages_for_thread(thread_id: str):
    state = await chatbot.aget_state(
        config={"configurable": {'thread_id': thread_id}}
    )
    messages = state.values.get("messages", [])
    return messages
    