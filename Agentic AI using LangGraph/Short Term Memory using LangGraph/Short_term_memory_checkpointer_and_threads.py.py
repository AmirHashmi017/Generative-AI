from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN_PROJECT']='Short-Term-Memory'

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]


def chat_node(state: ChatState):
    messages= state['messages']
    response= llm.invoke(messages)
    return{"messages": response}


graph= StateGraph(ChatState)

checkpointer= InMemorySaver()

graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
chatbot= graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"123"}}

response= chatbot.invoke({"messages":[HumanMessage(content="Hi My name is Amir")]},config=config)
print(response["messages"][-1].content)
    
response= chatbot.invoke({"messages":[HumanMessage(content="What is my name")]},config=config)
print(response["messages"][-1].content) 

message_history= chatbot.get_state(config=config).values.get("messages")
print("Message History")
print(message_history)
