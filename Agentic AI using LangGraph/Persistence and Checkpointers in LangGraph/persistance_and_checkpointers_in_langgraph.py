from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def generate_joke(state:JokeState):
    prompt=f"Generate a joke on topic {state['topic']}"
    joke= llm.invoke(prompt).content
    return {"joke":joke}

def generate_joke_explanation(state:JokeState):
    prompt=f"Generate explanation of this joke {state['joke']}"
    explanation= llm.invoke(prompt).content
    return {"explanation":explanation}

graph= StateGraph(JokeState)
graph.add_node("generate_joke",generate_joke)
graph.add_node("generate_joke_explanation",generate_joke_explanation)

graph.add_edge(START,"generate_joke")
graph.add_edge("generate_joke","generate_joke_explanation")
graph.add_edge("generate_joke_explanation",END)

checkpointer= InMemorySaver()

workflow= graph.compile(checkpointer=checkpointer)

config1= {"configurable":{"thread_id":"1"}}

final_state= workflow.invoke({"topic":"PIA"},config=config1)
print(f"Thread 1 Output: {final_state}")

# Final Value of thread
final_state_history= workflow.get_state(config1)
print(f"Thread 1 Final State History: {final_state_history}")

# Intermediate values of checkpoints
intermediate_state_history= list(workflow.get_state_history(config1))
print(f"Thread 1 Intermediate State History: {intermediate_state_history}")

config2= {"configurable":{"thread_id":"2"}}

final_state_2= workflow.invoke({"topic":"Pak Railway"},config=config2)
print(f"Thread 2 Output: {final_state_2}")

final_state_history_2= workflow.get_state(config2)
print(f"Thread 2 Final State History: {final_state_history_2}")

intermediate_state_history_2= list(workflow.get_state_history(config2))
print(f"Thread 2 Intermediate State History: {intermediate_state_history_2}")

# Time Travel
workflow.get_state({
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1f0e496c-df48-691f-8000-f64674a776a"
    }
})
# Resume from that checkpoint
print(workflow.invoke(None,{"configurable":{"thread_id":"1","checkpoint_id":"1f0e496c-df48-691f-8000-f64674a776a"}}))

# Update State Value in Time Travel
workflow.update_state({"configurable":{"thread_id":"1","checkpoint_id":"1f0e496c-df48-691f-8000-f64674a776a","checkpoint_ns":""}},{"topic":"samosa"})