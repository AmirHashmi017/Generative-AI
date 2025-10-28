# Using this we can convert a ptyhon or lambda function to a Runnable and then 
# we can merge it with other runnables

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1= PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model= ChatOpenAI()

parser= StrOutputParser()

def word_counter(text):
    return len(text.split())

report_generator_chain= RunnableSequence(prompt1,model,parser)

branch_chain= RunnableBranch(
    (lambda x: len(x.split())>100, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain= RunnableSequence(report_generator_chain,branch_chain)

result= final_chain.invoke({"topic":"Russia vs Ukraine"})
print(result)