from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool,InjectedToolArg
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
import requests
import os

os.environ['LANGCHAIN_PROJECT']='Chatbot-Project'

load_dotenv()
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")
ALPHAVANTAGE_API_KEY= os.getenv("ALPHAVANTAGE_API_KEY")
EXCHANGERATE_API_KEY= os.getenv("EXCHANGERATE_API_KEY")

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool= DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Performs a basic arithmetic operation on two numbers.
    Supported Operations: add, sub, mul, div
    """
    try:
        if operation=="add":
            result= first_num + second_num
        elif operation=="sub":
            result= first_num - second_num
        elif operation=="mul":
            result= first_num * second_num
        elif operation=="div":
            if second_num==0:
                return {"error":"Division by Zero is not Allowed"}
            result= first_num / second_num
        else:
            return {"error":f"Unsupported Operation {operation}"}
        return {"first_num":first_num,"second_num":second_num,
                "operation":operation,"result":result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol:str)->dict:
    """
    Fetch Latest Stock Price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API Key in the URL
    """
    url= f"https://www.alphavantage.com/query?function=GLOBAL_QUOTE&symbol={symbol}&api_key={ALPHAVANTAGE_API_KEY}"
    r= requests.get(url)
    return r.json()

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

tools= [search_tool, calculator, get_stock_price, get_weather_data, get_conversion_factor, convert_currency]

llm_with_tools= llm.bind_tools(tools)

tool_node= ToolNode(tools)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages= state['messages']
    response= llm_with_tools.invoke(messages)
    return{"messages": response}

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
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
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
