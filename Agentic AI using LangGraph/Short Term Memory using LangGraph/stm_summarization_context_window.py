from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN_PROJECT']='Short-Term-Memory'

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

MAX_TOKENS= 150

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]
    summary: str


def chat_node(state: ChatState):
    messages= []
    if state['summary']:
        messages.append(SystemMessage(content=state['summary']))
    messages.extend(state['messages'])
    response= llm.invoke(messages)
    return{"messages": response}

def summary_node(state: ChatState):
    existing_summary= state['summary']
    if existing_summary:
        prompt= f""""
        Existing Summary: \n{existing_summary}\n\n
        Extend the summary using new conversation above
        """
    else:
        prompt= f""""Summarize THE conversation above"""
    messages_for_summary= state["messages"] + [HumanMessage(content=prompt)]
    response= llm.invoke(messages_for_summary)
    to_remove= state['messages'][:-2]
    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in to_remove]
        }

def should_summarize(state: ChatState):
    return len(state['messages'])>6


graph= StateGraph(ChatState)

checkpointer= InMemorySaver()

graph.add_node("chat_node",chat_node)
graph.add_node("summarize_node",summary_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",
                            should_summarize,
                            {
                                True: "summarize_node",
                                False: "__end__"
                            })
graph.add_edge("summarize_node",END)
chatbot= graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"123"}}

response= chatbot.invoke({"messages":[HumanMessage(content="Hi My name is Amir")],"summary":""},config=config)
print(response)
    
response= chatbot.invoke({"messages":[HumanMessage(content="What is my name")]},config=config)
print(response)

response= chatbot.invoke({"messages":[HumanMessage(content="I am learning LangGraph Give me roadmap of 300 characters")]},config=config)
print(response)

response= chatbot.invoke({"messages":[HumanMessage(content="Summarize the roadmap you provided")]},config=config)
print(response) 

