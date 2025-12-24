from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    question: str
    answer: str

def ask_llm(state:ChatState):
    question= state["question"]
    prompt= f"Answer this question {question}"
    answer= model.invoke(prompt).content
    state['answer']=answer
    return state

graph= StateGraph(ChatState)

graph.add_node("ask_llm",ask_llm)

graph.add_edge(START,"ask_llm")
graph.add_edge("ask_llm",END)

workflow= graph.compile()

initial_state= {"question":"What is the capital of Pakistan?"}

final_state= workflow.invoke(initial_state)

print(final_state)