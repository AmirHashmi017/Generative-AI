from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import operator

load_dotenv()

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class EssayEvaluate(BaseModel):
    feedback: str= Field(description="Feedback on essay")
    score: int= Field(description="Score out of 10",ge=0,le=10)

structured_model= model.with_structured_output(EssayEvaluate)

class EssayEvaluationState(TypedDict):
    essay_content: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int],operator.add]
    average_score: float

def evaluate_language(state:EssayEvaluationState):
    essay= state["essay_content"]
    prompt= f"""Evaluate the essay on the basis of language. Provide feedback of 3 to 
    4 lines and score out of 10. The essay content is \n {essay}"""
    result= structured_model.invoke(prompt)
    language_feedback= result.feedback
    individual_score= result.score
    return {"language_feedback": language_feedback, "individual_score": [individual_score]}

def evaluate_analysis(state:EssayEvaluationState):
    essay= state["essay_content"]
    prompt= f"""Evaluate the essay on the basis of Depth of Analysis. Provide feedback of 3 to 
    4 lines and score out of 10. The essay content is \n {essay}"""
    result= structured_model.invoke(prompt)
    analysis_feedback= result.feedback
    individual_score= result.score
    return {"analysis_feedback": analysis_feedback, "individual_score": [individual_score]}

def evaluate_clarity(state:EssayEvaluationState):
    essay= state["essay_content"]
    prompt= f"""Evaluate the essay on the basis of Clarity of Thought. Provide feedback of 3 to 
    4 lines and score out of 10. The essay content is \n {essay}"""
    result= structured_model.invoke(prompt)
    clarity_feedback= result.feedback
    individual_score= result.score
    return {"clarity_feedback": clarity_feedback, "individual_score": [individual_score]}

def evaluate_overall(state:EssayEvaluationState):
    language_feedback= state["language_feedback"]
    analysis_feedback= state["analysis_feedback"]
    clarity_feedback= state["clarity_feedback"]

    prompt= f"""I have evaluated essay on the basis of Language \n{language_feedback}
        Depth of Analysis \n {analysis_feedback} Clarity of Thought \n {clarity_feedback}
        Write an overall feedback of 3 to 4 lines"""
    overall_feedback= model.invoke(prompt).content
    
    average_score= sum(state["individual_score"])/len(state["individual_score"])
    return {"overall_feedback": overall_feedback, "average_score": average_score}



graph= StateGraph(EssayEvaluationState)

graph.add_node("evaluate_language",evaluate_language)
graph.add_node("evaluate_analysis",evaluate_analysis)
graph.add_node("evaluate_clarity",evaluate_clarity)
graph.add_node("evaluate_overall",evaluate_overall)

graph.add_edge(START,"evaluate_language")
graph.add_edge(START,"evaluate_analysis")
graph.add_edge(START,"evaluate_clarity")

graph.add_edge("evaluate_language","evaluate_overall")
graph.add_edge("evaluate_analysis","evaluate_overall")
graph.add_edge("evaluate_clarity","evaluate_overall")

graph.add_edge("evaluate_overall",END)

workflow= graph.compile()

essay= """
Artificial Intelligence (AI) is rapidly transforming the technological landscape of Pakistan. 
From education to healthcare and business, AI-based solutions are helping improve efficiency 
and decision-making processes. Universities are now introducing AI-related programs, allowing 
students to gain skills that are highly demanded in the global market.

In the healthcare sector, AI is being used to assist doctors in diagnosing diseases more accurately 
and at an early stage. This is particularly beneficial for Pakistan, where access to medical 
experts is limited in rural areas. AI-powered systems can analyze medical data quickly and help 
bridge the gap between patients and healthcare professionals.

However, the rise of AI also brings challenges. Job displacement due to automation is a major 
concern, especially for low-skilled workers. Additionally, Pakistan faces issues such as lack 
of proper infrastructure, data security concerns, and limited funding for research and 
development.

Despite these challenges, AI has immense potential to drive economic growth in Pakistan. With 
proper government policies, investment in education, and ethical use of technology, AI can 
play a vital role in shaping a smarter and more prosperous future for the country.
"""

initial_state= {"essay_content":essay}

final_state= workflow.invoke(initial_state)

print(final_state)