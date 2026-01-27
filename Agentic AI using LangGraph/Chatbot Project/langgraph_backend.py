from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import interrupt, Command

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool,InjectedToolArg
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from typing import List,TypedDict,Annotated,Optional,Dict, Any
from dotenv import load_dotenv
import aiosqlite
import requests
import tempfile
import os
import contextvars

os.environ['LANGCHAIN_PROJECT']='Chatbot-Project'

load_dotenv()
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")
EXCHANGERATE_API_KEY= os.getenv("EXCHANGERATE_API_KEY")

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding= GoogleGenerativeAIEmbeddings(model="text-embedding-004")

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

_current_thread_id = contextvars.ContextVar('thread_id', default=None)

def set_thread_context(thread_id: str):
    return _current_thread_id.set(thread_id)

def get_thread_context() -> Optional[str]:
    return _current_thread_id.get()


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


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
def rag_tool(query: str) -> dict:
    """
    Retrieves relevant information from document.
    Use this tool when the user asks factual/conceptual questions 
    that might be answered from stored documents.
    """
    thread_id = get_thread_context()
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return{
        'query': query,
        'context': context,
        'metadata': metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

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
    decision= interrupt(f"HITL: Allow Currency Tool to fetch conversion rate from {base_currency} to {target_currency}? (Yes/No)")
    if decision and decision.get("approved") == "no":
        return {"response": "Tool Call Denied by User"}
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
    base_tools= [search_tool, rag_tool, get_weather_data, get_conversion_factor, convert_currency]
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
    