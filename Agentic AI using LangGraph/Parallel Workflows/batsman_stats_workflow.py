from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    balls_per_boundary: float
    boundary_percentage: float
    stats_report: str

def calculate_strike_rate(state: BatsmanState):
    runs= state['runs']
    balls= state['balls']
    strike_rate= (runs/balls)*100
    return {"strike_rate":strike_rate}

def calculate_balls_per_boundary(state: BatsmanState):
    balls= state['balls']
    fours= state['fours']
    sixes= state['sixes']
    balls_per_boundary= balls/(fours+sixes)
    return {"balls_per_boundary":balls_per_boundary}

def calculate__boundary_percentage(state: BatsmanState):
    runs= state['runs']
    four_runs= state['fours']*4
    six_runs= state['sixes']*6
    boundary_percentage= (runs/(four_runs+six_runs))*100
    return {"boundary_percentage":boundary_percentage}

def generate_stats(state: BatsmanState):
    stats_report= f"""
    Strike Rate: {state['strike_rate']}
    Balls Per Boundary: {state['balls_per_boundary']}
    Boundary Percentage: {state['boundary_percentage']}
"""
    return {"stats_report":stats_report}

graph= StateGraph(BatsmanState)

graph.add_node("calculate_strike_rate",calculate_strike_rate)
graph.add_node("calculate_balls_per_boundary",calculate_balls_per_boundary)
graph.add_node("calculate__boundary_percentage",calculate__boundary_percentage)
graph.add_node("generate_stats",generate_stats)

graph.add_edge(START,"calculate_strike_rate")
graph.add_edge(START,"calculate_balls_per_boundary")
graph.add_edge(START,"calculate__boundary_percentage")

graph.add_edge("calculate_strike_rate","generate_stats")
graph.add_edge("calculate_balls_per_boundary","generate_stats")
graph.add_edge("calculate__boundary_percentage","generate_stats")

graph.add_edge("generate_stats", END)

workflow= graph.compile()

initial_state= {"runs":100,"balls":50, "fours":8, "sixes":4}
final_state= workflow.invoke(initial_state)
print(final_state)