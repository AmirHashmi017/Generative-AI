from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage,SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
import uuid
import os

os.environ['LANGCHAIN_PROJECT']='Long-Term-Memory'

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class MemoryItem(BaseModel):
    text: str= Field(description="Atomic User Memory as a short sentence")
    is_new: bool= Field(description="True if this memory is new and should be stored. False if it duplicates/already known")

class remember_ltm(BaseModel):
    should_remember: bool= Field(description="Is anything to remember in LTM from the message")
    memories: List[MemoryItem]= Field(description="Atomic User Memories in store")

remember_llm= llm.with_structured_output(remember_ltm)

user_id="u1"
user_details= ("user",user_id,"details")


SYSTEM_PROMPT_TEMPLATE= """"You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize
your responses based on what you know about the user.

Your goal is to provide relevant, friendly and tailored
assistance that reflects the user's preferences, context and past interactions

If the user's name or relevant person context is available. always personalize your responses by:
 - Always Adress the user by name (e.g Sure Amir ...) when appropriate
 - Reflecting known projects, tools and preferences
 - Adjusting the tone to be friendly, natural and directly aimed at user
 
 Avoid generic phrasing when personalization is possible. For example instead of "In Typescript apps
 say "Since your project is in typescript..."

 Use personalization especially in:
  - Greetings and transitions
  - Help pr guidance tailored to tools or framework that user uses
  - Follow up messages that continue from past context

  Always ensure that personalization is based only on known user details and not assumed

  In the end suggest 3 relevant further questions on the current response and user profile.

  The user's memory which may be empty is provided as {user_details_content}
 """

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

def remember_ltm_node(state: ChatState, config:RunnableConfig, store: BaseStore):
    user_id= config["configurable"]["user_id"]
    user_details= ("user",user_id,"details")
    items= store.search(user_details)

    user_msg= state["messages"][-1].content

    if items:
        user_details_content= "\n".join(f" {it.value.get('data','')}" for it in items)
    else:
        user_details_content=""
    
    prompt = f"""
    You are a memory extraction system for a long-term memory store.

    Your task is to analyze the user's latest message and decide whether it contains
    information that should be saved as long-term memory.

    You are also given the user's EXISTING long-term memories.
    Use them to determine whether extracted memories are NEW or DUPLICATES.

    EXISTING USER MEMORIES:
    {user_details_content}

    Only extract information that is:
    - Stable over time (preferences, profile details, habits, projects, skills, goals)
    - Useful for personalizing future responses
    - Explicitly stated by the user

    DO NOT extract:
    - One-off questions or temporary context
    - General opinions without personal relevance
    - Information already implied or assumed

    DUPLICATE RULES:
    - If the meaning of a memory already exists in the stored memories, mark it as duplicate
    - Minor wording changes still count as duplicates
    - If unsure, prefer marking as duplicate

    If there is nothing worth remembering:
    - Set should_remember to false
    - Return an empty memories list

    If there IS something worth remembering:
    - Set should_remember to true
    - Extract each memory as a short, factual sentence
    - Write memories in third-person (e.g. "User prefers concise explanations")
    - For each memory:
      - Set is_new = true ONLY if it is NOT already present
      - Set is_new = false if it duplicates existing memory

    User message: {user_msg}
    """

    response: remember_ltm= remember_llm.invoke(prompt)
    if response.should_remember:
        user_details= ("user",user_id,"details")
        for mem in response.memories:
            if mem.is_new:
                store.put(user_details,str(uuid.uuid4()),{"data": mem.text})
    

def chat_node(state: ChatState, config:RunnableConfig, store: BaseStore):
    user_id= config["configurable"]["user_id"]
    user_details= ("user",user_id,"details")
    items= store.search(user_details)

    if items:
        user_details_content= "\n".join(f" {it.value.get('data','')}" for it in items)
    else:
        user_details_content=""
    
    system_prompt= SYSTEM_PROMPT_TEMPLATE.format(user_details_content=user_details_content)
    system_msg= SystemMessage(content= system_prompt)
    messages= [system_msg] + state['messages']
    response= llm.invoke(messages)
    return{"messages": response}


graph= StateGraph(ChatState)

checkpointer= InMemorySaver()

graph.add_node("chat_node",chat_node)
graph.add_node("remember_ltm_node",remember_ltm_node)
graph.add_edge(START,"remember_ltm_node")
graph.add_edge("remember_ltm_node","chat_node")
graph.add_edge("chat_node",END)

DB_URI= "postgresql://postgres:postgres@localhost:5442/postgres" 

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    chatbot= graph.compile(checkpointer=checkpointer,store=store)

    config={"configurable":{"thread_id":"123","user_id":"u1"}}

    response= chatbot.invoke({"messages":[HumanMessage(content="Hi My name is Amir")]}, config=config)
    response= chatbot.invoke({"messages":[HumanMessage(content="I am AI/ML Developer ")]},config=config)
    response= chatbot.invoke({"messages":[HumanMessage(content="I Prefer concise answers")]}, config=config)
    response= chatbot.invoke({"messages":[HumanMessage(content="I Like examples in python")]},config=config)
    response= chatbot.invoke({"messages":[HumanMessage(content="I am Building MCP Servers Python Based Project")]},config=config)
    response= chatbot.invoke({"messages":[HumanMessage(content="I am AI/ML Developer")]},config=config)

    response= chatbot.invoke({"messages":[HumanMessage(content="Explain me LangChain")]},
                             config=config)
    print(response["messages"][-1].content) 

    items= store.search(user_details)
    for item in items:
        print(item.value.get("data"))

