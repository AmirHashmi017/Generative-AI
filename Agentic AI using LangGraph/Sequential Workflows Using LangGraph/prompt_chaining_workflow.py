from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class PromptChainState(TypedDict):
    topic: str
    outline: str
    blog_content: str
    feedback: str

def generate_outline(state:PromptChainState):
    topic= state["topic"]
    prompt= f"Genearte a Detailed outline on topic {topic}"
    outline= model.invoke(prompt).content
    state['outline']=outline
    return state

def generate_blog(state:PromptChainState):
    topic= state["topic"]
    outline= state["topic"]
    prompt= f"Genearte a blog on topic {topic} using the outline \n {outline}"
    blog_content= model.invoke(prompt).content
    state['blog_content']=blog_content
    return state

def evaluate_blog(state:PromptChainState):
    topic= state["topic"]
    outline= state["topic"]
    blog_content= state["blog_content"]
    prompt= f"Evaluate the blog and give feedback \n {blog_content} using the topic {topic} and the outline \n {outline}"
    feedback= model.invoke(prompt).content
    state['feedback']=feedback
    return state

graph= StateGraph(PromptChainState)

graph.add_node("generate_outline",generate_outline)
graph.add_node("generate_blog",generate_blog)
graph.add_node("evaluate_blog",evaluate_blog)

graph.add_edge(START,"generate_outline")
graph.add_edge("generate_outline","generate_blog")
graph.add_edge("generate_blog","evaluate_blog")
graph.add_edge("evaluate_blog",END)

workflow= graph.compile()

initial_state= {"topic":"Rise of AI in Pakistan"}

final_state= workflow.invoke(initial_state)

topic= final_state['topic']
outline= final_state['outline']
blog_content= final_state['blog_content']
feedback= final_state['feedback']

print(f"Topic: {topic}")
print(f"Outline: {outline}")
print(f"Blog: {blog_content}")
print(f"Feedback: {feedback}")