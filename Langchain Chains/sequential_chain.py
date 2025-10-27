from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1= PromptTemplate(
    template="Generate a detailed report on topic {topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template="Generate a five point summary from the following text {text}",
    input_variables=['text']
)

model= ChatOpenAI(model='gpt-5')
parser= StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

result= chain.invoke({"topic": "Unemployment in Pakistan"})

print(result)
chain.get_graph().print_ascii()