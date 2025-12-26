from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict,Literal,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

load_dotenv()

class EvaluateTweet(BaseModel):
    evaluation: Literal['approved','needs_improvement']= Field(description="Evaluation on Tweet")
    feedback: str= Field(description="Feedback on Tweet")

model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluation_model= model.with_structured_output(EvaluateTweet)

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal['approved','needs_improvement']
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[List[str], operator.add]
    feedback_history: Annotated[List[str], operator.add]

def generate_tweet(state: TweetState):
    messages= [
        SystemMessage(content="You are a funny and clever Twitter/X influencer"),
        HumanMessage(content=f"""
                    Write a short, original and hilarious tweet on the topic: "{state['topic']}".
                    Rules:
                    - Do not use question-answer format.
                    - Max 200 characters.
                    - Use Observation Humour, irony, sacrasm or cultural references.
                    - Think in meme logic, punch lines or relatable takes.
                    - Use simple day to day english.
                    """)
    ]
    tweet= model.invoke(messages).content
    return {"tweet":tweet,"tweet_history":[tweet]}

def evaluate_tweet(state: TweetState):
    messages= [
        SystemMessage(content="You are a rutheless, no laugh-given Twitter/X Critic who evaluate tweets."),
        HumanMessage(
            content=f"""
        Evaluate the following tweet:

        Tweet:
        "{state['tweet']}"

        Criteria:
        - Is it funny or clever?
        - Is it original?
        - Does it feel like real Twitter humor?
        - Is the punchline strong?
        - Is it under 200 characters?
        - It should not be in Question-Answer Format

        Respond STRICTLY in this JSON format:
        {{
          "evaluation": "approved" OR "needs_improvement",
          "feedback": "Clear, blunt feedback with specific improvement suggestions"
        }}
        """
        )
    ]
    resp= evaluation_model.invoke(messages)
    return {"evaluation":resp.evaluation,"feedback":resp.feedback, "feedback_history":[resp.feedback]}

def optimize_tweet(state: TweetState):
    messages= [
        SystemMessage(content="You punch up tweets for virality and humour based on given feedback"),
        HumanMessage(content=f"""
                    Improve the tweet based on this feedback "{state['feedback']}".
                    Topic: {state['topic']}
                    Original Tweet: {state['tweet']}
                    Rewrite it.
                    """)
    ]
    tweet= model.invoke(messages).content
    iteration= state['iteration']+1
    return {"tweet":tweet,"iteration":iteration,"tweet_history":[tweet]}

def route_evaluation(state:TweetState)->Literal['approved','needs_improvement']:
    if state["evaluation"]=="approved" or state["iteration"]>=state["max_iteration"]:
        return "approved"
    else:
        return "needs_improvement"

graph= StateGraph(TweetState)
graph.add_node("generate_tweet",generate_tweet)
graph.add_node("evaluate_tweet",evaluate_tweet)
graph.add_node("optimize_tweet",optimize_tweet)

graph.add_edge(START,"generate_tweet")
graph.add_edge("generate_tweet","evaluate_tweet")
graph.add_conditional_edges("evaluate_tweet",route_evaluation,{"approved": END, "needs_improvement":"optimize_tweet"})
graph.add_edge("optimize_tweet","evaluate_tweet")

workflow= graph.compile()

initial_state={
    "topic": "PIA",
    "iteration":1,
    "max_iteration":5
}

final_state= workflow.invoke(initial_state)
print(final_state)