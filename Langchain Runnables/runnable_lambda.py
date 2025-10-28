# Using this we can convert a ptyhon or lambda function to a Runnable and then 
# we can merge it with other runnables

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

prompt= PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model= ChatOpenAI()

parser= StrOutputParser()

def word_counter(text):
    return len(text.split())

joke_generator_chain= RunnableSequence(prompt,model,parser)

parallel_chain= RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'no_of_words_in_joke': RunnableLambda(word_counter)
    }
)

final_chain= RunnableSequence(joke_generator_chain,parallel_chain)

result= final_chain.invoke({"topic":"cricket"})
print(result)