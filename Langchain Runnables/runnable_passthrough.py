from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1= PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Explain the following joke {joke}',
    input_variables=['joke']
)

model= ChatOpenAI()

parser= StrOutputParser()

joke_generator_chain= RunnableSequence(prompt1,model,parser)

parallel_chain= RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2,model,parser)
    }
)

final_chain= RunnableSequence(joke_generator_chain,parallel_chain)

result= final_chain.invoke({"topic":"cricket"})
print(result)