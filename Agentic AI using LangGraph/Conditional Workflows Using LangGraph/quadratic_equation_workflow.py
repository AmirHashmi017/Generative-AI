from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class QuadState(TypedDict):
    a: int
    b: int
    c: int

    equation: str
    descriminant: float
    result: str

def show_equation(state: QuadState):
    equation= f'{state["a"]}x2+({state["b"]})x+({state["c"]})'
    return {"equation":equation}

def calculate_descriminant(state: QuadState):
    descriminant= ((state["b"]**2)-(4*state["a"]*state["c"]))
    return {"descriminant":descriminant}

def real_roots(state:QuadState):
    root1= (-state["b"] + state["descriminant"]**0.5)/(2*state["a"])
    root2= (-state["b"] - state["descriminant"]**0.5)/(2*state["a"])
    result= f"The Roots are {root1} and {root2}"
    return {"result":result}

def repeated_root(state:QuadState):
    root= (-state["b"])/(2*state["a"])
    result= f"Only Reapeating Root is {root}"
    return {"result":result}

def no_real_roots(state:QuadState):
    result= "No Real Roots"
    return {"result":result}

def check_condition(state:QuadState)->Literal["real_roots","repeated_root","no_real_roots"]:
    if state["descriminant"]>0:
        return "real_roots"
    elif state["descriminant"]==0:
        return "repeated_root"
    else:
        return "no_real_roots"

graph= StateGraph(QuadState)
graph.add_node("show_equation",show_equation)
graph.add_node("calculate_descriminant",calculate_descriminant)
graph.add_node("real_roots",real_roots)
graph.add_node("repeated_root",repeated_root)
graph.add_node("no_real_roots",no_real_roots)

graph.add_edge(START,"show_equation")
graph.add_edge("show_equation","calculate_descriminant")
graph.add_conditional_edges("calculate_descriminant",check_condition)

graph.add_edge("real_roots",END)
graph.add_edge("repeated_root",END)
graph.add_edge("no_real_roots",END)

workflow= graph.compile()

initial_state= {"a":4,"b":-5,"c":-4}
final_state= workflow.invoke(initial_state)
print(final_state)