from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing import TypedDict,Annotated,List
from dotenv import load_dotenv

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]

def chat_node(state:ChatState):
    decision= interrupt({
        "type":"approval",
        "reason":"Model is about to answer a user question",
        "question": state["messages"][-1].content,
        "instruction": "Approve this question? Yes/No"
    })

    if decision["approved"]=='no':
        return{"messages":[AIMessage(content="Not Approved")]}
    else:
        messages= state['messages']
        response= llm.invoke(messages)
        return {"messages":[AIMessage(content=response.content)]}

graph= StateGraph(ChatState)

checkpointer= MemorySaver()
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot= graph.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"1234"}}

initial_input= {"messages":[HumanMessage(content="What is LangChain?")]}

result= chatbot.invoke(initial_input,config=config)

message= result["__interrupt__"][0].value

user_input= input(f"Backend Message - {message} \n Approve this question? (yes/no): ")

final_result= chatbot.invoke(
    Command(resume= {"approved":user_input}),
    config=config
)

print(final_result)