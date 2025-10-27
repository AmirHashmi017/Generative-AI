from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

model= ChatOpenAI()
parser= StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative']= Field(description='Give the sentiment of the feedback')

pydantic_parser= PydanticOutputParser(pydantic_object=Feedback)

prompt1= PromptTemplate(
    template='''Classify the sentiment of the following feedback text into positive or
    negative \n {feedback} \n {format_instruction}''',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_parser.get_format_instructions()}
)

classifier_chain= prompt1 | model | pydantic_parser

prompt2= PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3= PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain= RunnableBranch(
    (lambda x:x.sentiment== 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment== 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment") # This is default if no condition goes true
)

final_chain= classifier_chain | branch_chain
result= final_chain.invoke({'feedback':'This is a wonderful Phone'})
print(result)
final_chain.get_graph().print_ascii()