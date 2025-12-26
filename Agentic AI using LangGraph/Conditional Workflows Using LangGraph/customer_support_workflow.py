from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class FeedbackSentiment(BaseModel):
    sentiment: Literal["Positive","Negative"]= Field(description="Sentiment of Customer Review")

class Diagnosis(BaseModel):
    issue_type: str= Field(description="Issue Type in Customer Negative Review")
    tone: str= Field(description="Tone of Customer Negative Review")
    urgency: str= Field(description="Urgency in Customer Negative Review")

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
sentiment_model= model.with_structured_output(FeedbackSentiment)
diagnosis_model= model.with_structured_output(Diagnosis)

class CustomerSupportState(TypedDict):
    customer_review: str
    sentiment: Literal["Positive","Negative"]
    diagnosis: dict
    response: str

def find_sentiment(state:CustomerSupportState):
    prompt=f"""Give Sentiment either Positive or Negative of the Cusotmer Review 
    {state['customer_review']}"""
    sentiment= sentiment_model.invoke(prompt).sentiment
    return {"sentiment":sentiment}

def run_diagnosis(state:CustomerSupportState):
    prompt= f"""Generate Review Diagnosis for this Customer Review {state['customer_review']}
                Return issue_type, tone and urgency"""
    resp= diagnosis_model.invoke(prompt)
    return {"diagnosis": resp.model_dump()}

def positive_response(state: CustomerSupportState):
    prompt= f"""You are a Support Assistant. Generate response for this 
    Positive Customer Review {state['customer_review']} This reponse will be direclty provided
    to customer remember this so generate in that way"""
    response= model.invoke(prompt).content
    return {"response":response}

def negative_response(state: CustomerSupportState):
    diagnosis= state["diagnosis"]
    prompt= f"""You are a supposrt assistant. The user had a {diagnosis["issue_type"] } Issue,
     with {diagnosis["tone"]} tone and marked urgency {diagnosis["urgency"]} Write a helpful, 
     empathetic and helpful resolution message. The Customer Original review is 
     {state["customer_review"]}This reponse will be direclty provided
    to customer remember this so generate in that way"""
    response= model.invoke(prompt).content
    return {"response":response}

def check_sentiment(state:CustomerSupportState)->Literal['run_diagnosis','positive_response']:
    if(state["sentiment"]=="Positive"):
        return 'positive_response'
    else:
        return 'run_diagnosis'

graph= StateGraph(CustomerSupportState)
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("positive_response",positive_response)
graph.add_node("negative_response",negative_response)


graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("positive_response",END)

graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)

workflow= graph.compile()

positive_review= """I am really happy with the service! The support team responded quickly 
and resolved my issue in no time. The app is easy to use and works perfectly.
Keep up the great work!"""


negative_review= """I’m extremely frustrated with this service. My payment failed twice, 
and no one has responded to my support request yet. This is unacceptable and needs to be fixed 
immediately."""

initial_state={"customer_review":negative_review}
final_state= workflow.invoke(initial_state)
print(final_state)