from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import asyncio
import os

os.environ['LANGCHAIN_PROJECT']='Chatbot-Project'

load_dotenv()
WEATHERSTACK_API_KEY= os.getenv("WEATHERSTACK_API_KEY")
ALPHAVANTAGE_API_KEY= os.getenv("ALPHAVANTAGE_API_KEY")
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


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]



async def build_graph():
    tools= await client.get_tools()
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

    chatbot= graph.compile()
    return chatbot

async def main():
    chatbot= await build_graph()
    response= await chatbot.ainvoke({"messages":[HumanMessage(content="Tell memy total expense from 20/01/2026 to 30/01/2026 with details")]})
    print(response["messages"][-1].content)
    
if __name__=="__main__":
    asyncio.run(main())
    