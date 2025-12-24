from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BMIState(TypedDict):
    weight: float
    height: float
    bmi:float

def calculate_bmi(state:BMIState):
    weight= state["weight"]
    height= state["height"]
    bmi= weight/(height**2)
    state["bmi"]=bmi
    return state

graph= StateGraph(BMIState)

graph.add_node("calculate_bmi",calculate_bmi)

graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", END)

workflow= graph.compile()

start_state={"weight":60,"height":150}

final_state= workflow.invoke(start_state)

print(final_state)

