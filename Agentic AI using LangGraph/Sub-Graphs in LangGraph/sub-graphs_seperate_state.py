from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class SubState(TypedDict):
    input_text: str
    translated_text: str


def translate_node(state:SubState):
    prompt=f"""
Translate the Following Text from English to Urdu. Keep it natural and clear. Do not add 
extra content 
Text: {state['input_text']}
"""
    response= llm.invoke(prompt)
    return {"translated_text":response.content}

sub_graph_builder= StateGraph(SubState)
sub_graph_builder.add_node("translate_node",translate_node)
sub_graph_builder.add_edge(START,"translate_node")
sub_graph_builder.add_edge("translate_node",END)

subgraph = sub_graph_builder.compile()

class ParentState(TypedDict):
    question: str
    answer_english: str
    answer_urdu: str

def generate_answer(state:ParentState):
    prompt=f"""
You are a helpful assistant generate answer of the following question
Question: {state['question']}
"""
    response= llm.invoke(prompt)
    return{"answer_english":response.content}

def translate_answer(state:ParentState):
    result= subgraph.invoke({'input_text':state["answer_english"]})
    return {"answer_urdu": result['translated_text']}

parent_graph_builder= StateGraph(ParentState)
parent_graph_builder.add_node("generate_answer",generate_answer)
parent_graph_builder.add_node("translate_answer",translate_answer)
parent_graph_builder.add_edge(START,"generate_answer")
parent_graph_builder.add_edge("generate_answer","translate_answer")
parent_graph_builder.add_edge("translate_answer",END)

parentgraph= parent_graph_builder.compile()

result= parentgraph.invoke({"question":"What is the capital of Pakistan"})
print(result)